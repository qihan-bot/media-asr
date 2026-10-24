from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from .config import (
    DATA_DIR,
    DEFAULT_COMPUTE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_THREADS,
    ITEMS_DIR,
    ensure_dirs,
)
from .db import Store
from .fetch import download, normalize_audio, resolve_source
from .remote import RemoteGPU
from .render import render_all
from .summarize import available_engines, summarize
from .transcribe import Transcriber, TranscriptResult
from .util import atomic_write, ffprobe_duration, hms, now_iso, sanitize_proxy_env, slugify


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model (default: %(default)s)")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE or "", help="language code, empty = auto")
    parser.add_argument("--prompt", default=None, help="initial prompt / hotwords for names and punctuation")
    parser.add_argument("--traditional", action="store_true", help="keep Traditional Chinese instead of converting to Simplified")
    parser.add_argument("--beam", type=int, default=5, help="beam size (default: %(default)s)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="CPU threads")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE, help="int8, float16, ...")
    parser.add_argument("--keep-audio", action="store_true", help="keep the normalized 16 kHz WAV")
    parser.add_argument("--offline", action="store_true", help="never download a model")
    parser.add_argument("--limit", type=int, default=100, help="max items to process")
    parser.add_argument("--feed-contains", default=None, help="only process items whose feed URL contains this text")
    parser.add_argument("--retry-failed", action="store_true", help="retry failed items")
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failure")
    add_remote_options(parser)


def add_remote_options(parser: argparse.ArgumentParser, *, backend: bool = True, mode: bool = True) -> None:
    if backend:
        parser.add_argument(
            "--backend",
            choices=["local", "remote"],
            default=os.environ.get("MEDIA_ASR_BACKEND", "local"),
            help="run transcription locally or on the remote GPU host",
        )
    parser.add_argument("--remote-host", default=os.environ.get("MEDIA_ASR_REMOTE_HOST", ""))
    parser.add_argument(
        "--remote-user",
        default=os.environ.get("MEDIA_ASR_REMOTE_USER", ""),
        help="SSH user (default: current local user)",
    )
    parser.add_argument("--remote-port", type=int, default=int(os.environ.get("MEDIA_ASR_REMOTE_PORT", 22)))
    parser.add_argument(
        "--remote-workdir",
        default=os.environ.get("MEDIA_ASR_REMOTE_WORKDIR", "media-asr-work"),
        help="remote directory relative to the SSH home",
    )
    parser.add_argument(
        "--remote-python",
        default=os.environ.get("MEDIA_ASR_REMOTE_PYTHON", r".venv\Scripts\python.exe"),
    )
    parser.add_argument(
        "--remote-shell",
        choices=["cmd", "sh"],
        default=os.environ.get("MEDIA_ASR_REMOTE_SHELL", "cmd"),
        help="remote shell used for cd/python commands",
    )
    parser.add_argument(
        "--remote-model",
        default=os.environ.get("MEDIA_ASR_REMOTE_MODEL", "large-v3-turbo"),
    )
    parser.add_argument(
        "--remote-model-dir",
        default=os.environ.get("MEDIA_ASR_REMOTE_MODEL_DIR", "models"),
    )
    if mode:
        parser.add_argument(
            "--remote-mode",
            choices=["asr", "full"],
            default=os.environ.get("MEDIA_ASR_REMOTE_MODE", "asr"),
            help="remote: ASR only, or download + ffmpeg + ASR + render",
        )
    parser.add_argument("--remote-key", default=os.environ.get("MEDIA_ASR_REMOTE_KEY"))
    parser.add_argument(
        "--remote-timeout",
        type=int,
        default=int(os.environ.get("MEDIA_ASR_REMOTE_TIMEOUT", 7200)),
    )
    parser.add_argument("--remote-condition-on-previous-text", action="store_true")
    parser.add_argument("--remote-auto-add-host-key", action="store_true")
    parser.add_argument("--remote-keep", action="store_true", help="keep remote job files")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="run speaker diarization in the remote full pipeline",
    )
    parser.add_argument("--num-speakers", type=int, default=-1, help="-1 = auto-detect")
    parser.add_argument(
        "--diarization-segmentation",
        default="models/diarization/segmentation.onnx",
    )
    parser.add_argument(
        "--diarization-embedding",
        default="models/diarization/speaker_embedding.onnx",
    )
    parser.add_argument("--diarization-threads", type=int, default=8)
    parser.add_argument("--diarization-threshold", type=float, default=0.5)
    parser.add_argument(
        "--diarization-backend",
        choices=["sherpa", "pyannote"],
        default=os.environ.get("MEDIA_ASR_DIARIZATION_BACKEND", "sherpa"),
    )
    parser.add_argument(
        "--pyannote-pipeline",
        default=os.environ.get("MEDIA_ASR_PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1"),
    )


def build_parser() -> argparse.ArgumentParser:
    run_parent = argparse.ArgumentParser(add_help=False)
    add_run_options(run_parent)

    parser = argparse.ArgumentParser(
        prog="media-asr",
        description="Local speech-to-text pipeline for videos and podcasts.",
    )
    parser.add_argument("--version", action="version", version="media-asr 0.01")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", parents=[run_parent], help="add a URL, feed, or local file")
    add.add_argument("url")
    add.add_argument("--latest", type=int, default=1, help="for feeds/playlists: how many recent items")
    add.add_argument("--title", default=None, help="override title for a single item")
    add.add_argument("--run", action="store_true", help="process immediately")

    run = sub.add_parser("run", parents=[run_parent], help="process pending items")
    run.add_argument("--ids", nargs="+", default=None, help="process exactly these items, whatever their status")
    run.set_defaults(command="run")

    refresh = sub.add_parser("refresh", parents=[run_parent], help="check subscribed feeds for new episodes")
    refresh.add_argument("--latest", type=int, default=None, help="override each feed's item count")
    refresh.add_argument("--run", action="store_true", help="process new items immediately")

    listing = sub.add_parser("list", help="list items")
    listing.add_argument("--status", default=None)
    listing.add_argument("--limit", type=int, default=50)

    show = sub.add_parser("show", help="show one item")
    show.add_argument("id")
    show.add_argument("--full", action="store_true")

    search = sub.add_parser("search", help="search transcripts")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)

    retranscribe = sub.add_parser("retranscribe", help="re-run an existing item with another model")
    retranscribe.add_argument("id")
    retranscribe.add_argument("--model", default=DEFAULT_MODEL)
    retranscribe.add_argument("--language", default=DEFAULT_LANGUAGE or "")
    retranscribe.add_argument("--prompt", default=None)
    retranscribe.add_argument("--traditional", action="store_true")
    retranscribe.add_argument("--beam", type=int, default=5)
    retranscribe.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    retranscribe.add_argument("--device", default="cpu")
    retranscribe.add_argument("--compute-type", default=DEFAULT_COMPUTE)
    retranscribe.add_argument("--offline", action="store_true")
    retranscribe.add_argument("--label", default=None, help="output subdirectory name")
    add_remote_options(retranscribe, mode=False)

    summary = sub.add_parser("summarize", help="summarize a transcript")
    summary.add_argument("id")
    summary.add_argument("--engine", choices=["auto", "agy", "deepseek", "none"], default="auto")
    summary.add_argument("--timeout", type=int, default=900)

    remove = sub.add_parser("remove", help="delete an item and its files")
    remove.add_argument("id")
    remove.add_argument("--yes", action="store_true")

    remote_check = sub.add_parser("remote-check", help="check the remote GPU host")
    add_remote_options(remote_check, backend=False)

    enrich = sub.add_parser("enrich", help="podcast library: punctuation and speaker names")
    enrich.add_argument("--ids", nargs="*", default=None, help="only these item ids")
    enrich.add_argument("--force", action="store_true", help="redo items that have enrich.json")
    enrich.add_argument("--limit", type=int, default=0, help="0 = all pending items")
    enrich.add_argument("--workers", type=int, default=6, help="items in parallel")
    enrich.add_argument("--chunk-workers", type=int, default=4, help="requests per item")

    terms = sub.add_parser("correct-terms", help="podcast library: audited term correction")
    terms.add_argument("--ids", nargs="*", default=None, help="only these item ids")
    terms.add_argument("--force", action="store_true", help="redo items that have terms.json")
    terms.add_argument("--workers", type=int, default=6, help="items in parallel")

    rediarize = sub.add_parser(
        "rediarize", help="podcast library: re-diarize episodes with merged speakers"
    )
    rediarize.add_argument(
        "--share", type=float, default=0.75, help="largest-label share that triggers a check"
    )
    rediarize.add_argument(
        "--dry-run", action="store_true", help="only report decisions; change nothing"
    )
    rediarize.add_argument("--shard", default="0/1", help="i/n: every n-th item, for parallel GPUs")

    library = sub.add_parser(
        "library-update",
        help="podcast library: refresh, transcribe, rediarize, enrich, correct-terms",
    )
    library.add_argument(
        "--gpu-env",
        default=os.environ.get("MEDIA_ASR_GPU_ENV"),
        help="env file of the preferred GPU host, used if remote-check passes",
    )
    library.add_argument(
        "--fallback-gpu-env",
        default=os.environ.get("MEDIA_ASR_FALLBACK_GPU_ENV"),
        help="env file of the GPU host to use otherwise (default: media-asr-remote's)",
    )
    library.add_argument("--latest", type=int, default=5, help="episodes to check per feed")
    library.add_argument(
        "--workers", type=int, default=4, help="items in parallel for enrich and correct-terms"
    )

    sub.add_parser("feeds", help="list subscribed feeds")
    sub.add_parser("stats", help="show counts")
    sub.add_parser("init", help="create data directories")

    return parser


def main(argv: list[str] | None = None) -> int:
    sanitize_proxy_env()
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    store = Store()
    try:
        if args.command == "add":
            return cmd_add(store, args)
        if args.command == "run":
            return cmd_run(store, args, item_ids=args.ids)
        if args.command == "refresh":
            return cmd_refresh(store, args)
        if args.command == "list":
            return cmd_list(store, args)
        if args.command == "show":
            return cmd_show(store, args)
        if args.command == "search":
            return cmd_search(store, args)
        if args.command == "retranscribe":
            return cmd_retranscribe(store, args)
        if args.command == "remote-check":
            return cmd_remote_check(args)
        if args.command == "summarize":
            return cmd_summarize(store, args)
        if args.command == "remove":
            return cmd_remove(store, args)
        if args.command == "enrich":
            return cmd_enrich(store, args)
        if args.command == "correct-terms":
            return cmd_correct_terms(store, args)
        if args.command == "rediarize":
            return cmd_rediarize(store, args)
        if args.command == "library-update":
            return cmd_library_update(store, args)
        if args.command == "feeds":
            return cmd_feeds(store)
        if args.command == "stats":
            return cmd_stats(store)
        if args.command == "init":
            print(f"data: {DATA_DIR}")
            print(f"database: {DATA_DIR / 'media.db'}")
            print(f"models: {Path.home() / '.cache/whisper'}")
            return 0
        parser.error(f"unknown command: {args.command}")
    finally:
        store.close()
    return 0


def _log(message: str) -> None:
    print(message, flush=True)


def cmd_add(store: Store, args: argparse.Namespace) -> int:
    try:
        result = resolve_source(args.url, latest=args.latest)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.feed_url:
        store.add_feed(result.feed_url, result.feed_title, args.latest)
        _log(f"feed: {result.feed_title or result.feed_url}")

    if not result.specs:
        _log("no items found")
        return 1

    added = 0
    new_ids: list[str] = []
    for spec in result.specs:
        if args.title and len(result.specs) == 1:
            spec.title = args.title
        item_id, created = store.add_item(
            source=spec.source,
            source_type=spec.source_type,
            title=spec.title,
            duration=spec.duration,
            published_at=spec.published_at,
            feed_url=spec.feed_url or result.feed_url,
            meta_json=spec.meta_json(),
        )
        state = "added" if created else "exists"
        _log(f"{state:6} {item_id}  {spec.title}")
        added += int(created)
        if created:
            new_ids.append(item_id)

    _log(f"{added} new item(s), {len(result.specs) - added} existing")
    if args.run and new_ids:
        return cmd_run(store, args, item_ids=new_ids)
    return 0


def cmd_refresh(store: Store, args: argparse.Namespace) -> int:
    feeds = store.list_feeds()
    if not feeds:
        _log("no subscribed feeds; add one with: media-asr add <rss-url> --latest 3")
        return 0

    total_new = 0
    new_ids: list[str] = []
    checked = 0
    failed = 0
    for feed in feeds:
        checked += 1
        latest = args.latest if args.latest is not None else feed["latest"]
        try:
            result = resolve_source(feed["url"], latest=latest)
        except Exception as exc:
            failed += 1
            _log(f"feed failed: {feed['url']}: {exc}")
            continue
        new = 0
        for spec in result.specs:
            item_id, created = store.add_item(
                source=spec.source,
                source_type=spec.source_type,
                title=spec.title,
                duration=spec.duration,
                published_at=spec.published_at,
                feed_url=feed["url"],
                meta_json=spec.meta_json(),
            )
            new += int(created)
            if created:
                new_ids.append(item_id)
        store.touch_feed(feed["url"])
        total_new += new
        _log(f"{feed['title'] or feed['url']}: {new} new / {len(result.specs)} checked")
    _log(f"{total_new} new item(s)")
    if checked and failed == checked:
        _log("all feeds failed")
        return 1
    if args.run and new_ids:
        return cmd_run(store, args, item_ids=new_ids)
    return 0


def cmd_run(
    store: Store,
    args: argparse.Namespace,
    item_ids: list[str] | None = None,
) -> int:
    if item_ids is not None:
        items = [
            row for item_id in item_ids if (row := store.get_item(item_id)) is not None
        ][: args.limit]
    else:
        feed_filter = getattr(args, "feed_contains", None)
        fetch = 1_000_000 if feed_filter else args.limit
        if args.retry_failed:
            items = store.list_items(retry_failed=True, limit=fetch)
        else:
            items = store.list_items(status="pending", limit=fetch)
        if feed_filter:
            items = [row for row in items if feed_filter in (row["feed_url"] or "")][: args.limit]
    if not items:
        _log("nothing to do")
        return 0

    if args.diarize and not (args.backend == "remote" and args.remote_mode == "full"):
        print(
            "error: --diarize requires --backend remote --remote-mode full",
            file=sys.stderr,
        )
        return 1

    remote = None
    transcriber = None
    if args.backend == "remote":
        remote = RemoteGPU.from_args(args)
        _log(f"remote backend: {remote.user}@{remote.host} model={remote.model}")
        try:
            remote.connect()
        except Exception as exc:
            print(f"error connecting to remote GPU: {exc}", file=sys.stderr)
            return 1
    else:
        _log(f"loading model: {args.model} ({args.compute_type}, {args.threads} threads)")
        try:
            transcriber = Transcriber(
                args.model,
                device=args.device,
                compute_type=args.compute_type,
                threads=args.threads,
                offline=args.offline,
            )
        except Exception as exc:
            print(f"error loading model: {exc}", file=sys.stderr)
            return 1

    failures = 0
    try:
        for row in items:
            item = dict(row)
            _log(f"\n== {item['id']}  {item['title']}")
            ok = _process_item(store, item, transcriber, args, remote=remote)
            if not ok:
                failures += 1
                if args.fail_fast:
                    break
    finally:
        if remote is not None:
            remote.close()
    if failures:
        _log(f"\n{failures} item(s) failed")
        return 1
    return 0


def _process_item(
    store: Store,
    item: dict,
    transcriber: Transcriber | None,
    args: argparse.Namespace,
    *,
    remote: RemoteGPU | None = None,
) -> bool:
    item_id = item["id"]
    item_dir = ITEMS_DIR / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    try:
        if remote is not None and remote.mode == "full":
            store.set_status(item_id, "downloading")
            _log("remote full pipeline: download + ffmpeg + ASR + render")
            full = remote.run_full(
                item,
                item_dir,
                language=args.language or None,
                prompt=args.prompt,
                model=args.remote_model,
                keep_audio=args.keep_audio,
                diarize=args.diarize,
                num_speakers=args.num_speakers,
                beam=args.beam,
                traditional=args.traditional,
                progress=_log,
            )
            result = TranscriptResult(
                segments=full["segments"],
                language=full.get("language"),
                language_probability=None,
                duration=full.get("duration"),
                model_label=str(full.get("model_label")),
            )
            duration = full.get("duration") or item.get("duration")
            if not result.segments:
                _log("warning: no speech detected")
            store.save_result(
                item_id,
                segments=result.segments,
                duration=duration,
                language=result.language,
                model=result.model_label,
                transcript_path=str(item_dir / "transcript.md"),
            )
            atomic_write(
                item_dir / "meta.json",
                json.dumps(
                    {
                        "id": item_id,
                        "title": item["title"],
                        "source": item["source"],
                        "source_type": item["source_type"],
                        "duration": duration,
                        "published_at": item.get("published_at"),
                        "feed_url": item.get("feed_url"),
                        "processed_at": now_iso(),
                        "worker": "remote",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            audio_path = full.get("audio_path")
            store.update_fields(item_id, audio_path=str(audio_path) if audio_path else None)
            _log(
                f"done: {len(result.segments)} segments, language={result.language}, "
                f"transcript={item_dir / 'transcript.md'}"
            )
            return True

        store.set_status(item_id, "downloading")
        source_path, is_local = download(item, item_dir, progress=_log)
        duration = ffprobe_duration(source_path) or item.get("duration")
        if duration is None or duration <= 0:
            raise RuntimeError(f"could not determine audio duration: {source_path}")
        _log(f"audio: {source_path} ({hms(duration)})")
        atomic_write(
            item_dir / "meta.json",
            json.dumps(
                {
                    "id": item_id,
                    "title": item["title"],
                    "source": item["source"],
                    "source_type": item["source_type"],
                    "duration": duration,
                    "published_at": item.get("published_at"),
                    "feed_url": item.get("feed_url"),
                    "downloaded_from": str(source_path),
                    "processed_at": now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

        wav_path = item_dir / "audio.wav"
        normalize_audio(source_path, wav_path)
        if not is_local:
            source_path.unlink(missing_ok=True)

        store.set_status(item_id, "transcribing")
        _log("transcribing...")
        if remote is not None:
            result = remote.transcribe(
                wav_path,
                item_id=item_id,
                language=args.language or None,
                prompt=args.prompt,
                model=args.remote_model,
                beam=args.beam,
                traditional=args.traditional,
                progress=_log,
            )
        else:
            assert transcriber is not None
            result = transcriber.transcribe(
                wav_path,
                language=args.language or None,
                prompt=args.prompt,
                beam_size=args.beam,
                simplify_zh=not args.traditional,
                progress=_log,
            )

        if not result.segments:
            _log("warning: no speech detected")
        paths = render_all(
            out_dir=item_dir,
            item_id=item_id,
            title=item["title"],
            source=item["source"],
            duration=duration,
            language=result.language,
            model=result.model_label,
            transcribed_at=now_iso(),
            segments=result.segments,
        )
        store.save_result(
            item_id,
            segments=result.segments,
            duration=duration,
            language=result.language,
            model=result.model_label,
            transcript_path=str(paths["markdown"]),
        )
        if args.keep_audio:
            store.update_fields(item_id, audio_path=str(wav_path))
        else:
            wav_path.unlink(missing_ok=True)
            store.update_fields(item_id, audio_path=None)

        _log(
            f"done: {len(result.segments)} segments, language={result.language}, "
            f"transcript={paths['markdown']}"
        )
        return True
    except KeyboardInterrupt:
        store.set_status(item_id, "pending")
        raise
    except Exception:
        error = traceback.format_exc(limit=6)
        store.set_status(item_id, "failed", error=error)
        _log(f"failed: {item_id}")
        print(error, file=sys.stderr)
        return False


def cmd_list(store: Store, args: argparse.Namespace) -> int:
    rows = store.list_items(status=args.status, limit=args.limit)
    if not rows:
        _log("no items")
        return 0
    _log(f"{'ID':42} {'STATUS':13} {'DURATION':>9}  TITLE")
    for row in rows:
        title = (row["title"] or "")[:78]
        _log(f"{row['id']:42} {row['status']:13} {hms(row['duration']):>9}  {title}")
    return 0


def cmd_show(store: Store, args: argparse.Namespace) -> int:
    row = store.get_item(args.id)
    if not row:
        print(f"not found: {args.id}", file=sys.stderr)
        return 1
    for key in (
        "id",
        "title",
        "status",
        "source",
        "duration",
        "language",
        "model",
        "transcript_path",
        "summary_path",
        "error",
        "created_at",
        "updated_at",
    ):
        value = row[key]
        if key == "duration":
            value = hms(value)
        if value:
            _log(f"{key:16} {value}")
    transcript_path = row["transcript_path"]
    if transcript_path and Path(transcript_path).exists():
        text = Path(transcript_path).read_text(encoding="utf-8")
        if not args.full and len(text) > 5000:
            text = text[:5000] + "\n\n... (use --full to see everything)"
        _log("\n" + text)
    return 0


def cmd_search(store: Store, args: argparse.Namespace) -> int:
    results = store.search(args.query, limit=args.limit)
    if not results:
        _log("no matches")
        return 0
    for result in results:
        start = result["start"]
        stamp = f"[{hms(start)}] " if start is not None else ""
        speaker = result.get("speaker")
        speaker_prefix = f"{speaker} " if speaker else ""
        _log(f"{result['title']}  {stamp}{speaker_prefix}{result['snippet']}")
        _log(f"  id: {result['id']}")
    return 0


def cmd_retranscribe(store: Store, args: argparse.Namespace) -> int:
    row = store.get_item(args.id)
    if not row:
        print(f"not found: {args.id}", file=sys.stderr)
        return 1
    item = dict(row)
    item_dir = ITEMS_DIR / item["id"]
    item_dir.mkdir(parents=True, exist_ok=True)

    audio_path = item_dir / "audio.wav"
    if not audio_path.exists():
        source = Path(item["source"]).expanduser()
        if not source.exists():
            print(f"no audio.wav and source is not local: {item['source']}", file=sys.stderr)
            return 1
        normalize_audio(source, audio_path)

    remote = None
    transcriber = None
    if args.backend == "remote":
        label = args.label or slugify(f"remote-{args.remote_model}")
        out_dir = item_dir / "runs" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        remote = RemoteGPU.from_args(args)
        _log(f"remote backend: {remote.user}@{remote.host} model={remote.model}")
        try:
            remote.connect()
            _log(f"transcribing {audio_path} on remote GPU ...")
            started = time.monotonic()
            result = remote.transcribe(
                audio_path,
                item_id=item["id"],
                language=args.language or None,
                prompt=args.prompt,
                model=args.remote_model,
                beam=args.beam,
                traditional=args.traditional,
                progress=_log,
            )
            elapsed = time.monotonic() - started
        except Exception as exc:
            print(f"remote transcription failed: {exc}", file=sys.stderr)
            return 1
        finally:
            remote.close()
        compute_type = "fp16 (remote)"
    else:
        label = args.label or slugify(f"{args.model}-{args.compute_type}")
        out_dir = item_dir / "runs" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"loading model: {args.model} ({args.compute_type}, {args.threads} threads)")
        transcriber = Transcriber(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            threads=args.threads,
            offline=args.offline,
        )
        _log(f"transcribing {audio_path} ...")
        started = time.monotonic()
        result = transcriber.transcribe(
            audio_path,
            language=args.language or None,
            prompt=args.prompt,
            beam_size=args.beam,
            simplify_zh=not args.traditional,
            progress=_log,
        )
        elapsed = time.monotonic() - started
        compute_type = args.compute_type

    paths = render_all(
        out_dir=out_dir,
        item_id=item["id"],
        title=item["title"],
        source=item["source"],
        duration=item["duration"],
        language=result.language,
        model=result.model_label,
        transcribed_at=now_iso(),
        segments=result.segments,
    )
    chars = sum(len(segment.text) for segment in result.segments)
    atomic_write(
        out_dir / "run.json",
        json.dumps(
            {
                "model": result.model_label,
                "compute_type": compute_type,
                "beam": args.beam,
                "language": result.language,
                "duration": item["duration"],
                "segments": len(result.segments),
                "chars": chars,
                "elapsed_seconds": round(elapsed, 1),
                "realtime_factor": round((item["duration"] or 0) / elapsed, 2) if elapsed else None,
                "created_at": now_iso(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _log(
        f"done: {len(result.segments)} segments, {chars} chars, "
        f"{elapsed:.0f}s ({elapsed / 60:.1f} min), transcript={paths['markdown']}"
    )
    return 0


def cmd_remote_check(args: argparse.Namespace) -> int:
    remote = RemoteGPU.from_args(args)
    try:
        info = remote.check()
    except Exception as exc:
        print(f"remote check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        remote.close()
    _log(f"host:   {info.get('host')}")
    _log(f"base:   {info.get('base')}")
    _log(f"gpu:    {info.get('gpu')}")
    _log(f"python: {info.get('python')}")
    _log(f"deps:   {info.get('deps')}")
    return 0


def cmd_summarize(store: Store, args: argparse.Namespace) -> int:
    row = store.get_item(args.id)
    if not row:
        print(f"not found: {args.id}", file=sys.stderr)
        return 1
    transcript_path = row["transcript_path"]
    if not transcript_path or not Path(transcript_path).exists():
        print(f"no transcript for {args.id}; run it first", file=sys.stderr)
        return 1
    out_path = Path(transcript_path).parent / "summary.md"
    try:
        engine, text = summarize(
            Path(transcript_path),
            title=row["title"],
            out_path=out_path,
            engine=args.engine,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"summary failed: {exc}", file=sys.stderr)
        print(f"available engines: {', '.join(available_engines())}", file=sys.stderr)
        return 1
    store.update_fields(args.id, summary_path=str(out_path))
    _log(f"summary ({engine}): {out_path}")
    _log("")
    _log(text)
    return 0


def cmd_remove(store: Store, args: argparse.Namespace) -> int:
    row = store.get_item(args.id)
    if not row:
        print(f"not found: {args.id}", file=sys.stderr)
        return 1
    if not args.yes:
        answer = input(f"delete {row['id']} ({row['title']})? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            _log("aborted")
            return 0
    shutil.rmtree(ITEMS_DIR / row["id"], ignore_errors=True)
    store.delete_item(row["id"])
    _log(f"deleted {row['id']}")
    return 0


def _deepseek_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY; run through `vibe vault run`", file=sys.stderr)
    return key


def cmd_enrich(store: Store, args: argparse.Namespace) -> int:
    from . import enrich
    from .library import LibraryError, load_shows
    from .llm import LLM

    key = _deepseek_key()
    if not key:
        return 1
    try:
        shows = load_shows(missing_ok=True)  # only for the recurring-host hints
    except LibraryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with LLM(key) as llm:
        return enrich.run(
            store.conn,
            llm,
            shows,
            ids=args.ids,
            force=args.force,
            limit=args.limit,
            workers=args.workers,
            chunk_workers=args.chunk_workers,
        )


def cmd_correct_terms(store: Store, args: argparse.Namespace) -> int:
    try:
        from . import terms
    except ImportError as exc:
        print(f"error: {exc}; install the extras: pip install -e '.[library]'", file=sys.stderr)
        return 1
    from .llm import LLM

    key = _deepseek_key()
    if not key:
        return 1
    with LLM(key) as llm:
        return terms.run(store.conn, llm, ids=args.ids, force=args.force, workers=args.workers)


def cmd_rediarize(store: Store, args: argparse.Namespace) -> int:
    from . import rediarize
    from .llm import LLM

    key = _deepseek_key()
    if not key:
        return 1
    with LLM(key) as llm:
        return rediarize.run(
            store.conn, llm, share=args.share, dry_run=args.dry_run, shard=args.shard
        )


def cmd_library_update(store: Store, args: argparse.Namespace) -> int:
    from .library import LibraryError, load_shows, update

    try:
        return update(
            store,
            load_shows(),
            gpu_env=args.gpu_env,
            fallback_gpu_env=args.fallback_gpu_env,
            latest=args.latest,
            workers=args.workers,
        )
    except LibraryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_feeds(store: Store) -> int:
    feeds = store.list_feeds()
    if not feeds:
        _log("no feeds")
        return 0
    for feed in feeds:
        _log(
            f"{feed['url']}  latest={feed['latest']}  "
            f"checked={feed['last_checked'] or '-'}  {feed['title'] or ''}"
        )
    return 0


def cmd_stats(store: Store) -> int:
    stats = store.stats()
    counts = stats["counts"]
    _log(
        "items: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        if counts
        else "items: none"
    )
    _log(f"hours transcribed: {stats['total_duration'] / 3600:.2f}")
    _log(f"segments: {stats['segments']}")
    _log(f"feeds: {stats['feeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())