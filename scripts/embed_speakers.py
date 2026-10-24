#!/usr/bin/env python3
"""Fetch speaker embeddings for every diarized All-In run from the remote GPU host.

Writes ``data/speaker_embeddings/<item_id>.npz``; see
``scripts/embed_speakers_worker.py`` for the format. Each file records the
fingerprint of the diarized segmentation it was computed from. Items whose
local file is missing, unreadable, or stale (fingerprint mismatch, e.g.
after re-diarizing) are (re)embedded; valid ones are skipped. Run one driver
at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_speakers_worker import valid_output  # noqa: E402
from speaker_common import is_allin, segments_fingerprint  # noqa: E402

from media_asr.config import DATA_DIR, ITEMS_DIR  # noqa: E402
from media_asr.remote import RemoteGPU  # noqa: E402

EMB_DIR = DATA_DIR / "speaker_embeddings"


def valid_npz(path: Path, fingerprint: str) -> bool:
    """Same rule as the remote worker, so both sides agree on what to rebuild."""
    return valid_output(path, fingerprint)


def episodes(label: str) -> list[dict]:
    out = []
    for item_dir in sorted(ITEMS_DIR.iterdir()):
        diarized = item_dir / "runs" / label / "transcript.json"
        if not diarized.exists() or not is_allin(item_dir):
            continue
        meta = json.loads((item_dir / "meta.json").read_text(encoding="utf-8"))
        segs = json.loads(diarized.read_text(encoding="utf-8"))["segments"]
        out.append(
            {
                "id": item_dir.name,
                "source": meta["source"],
                "source_type": meta.get("source_type", "audio"),
                "audio_url": meta.get("audio_url"),
                "segments": [[s["start"], s["end"]] for s in segs],
                "fp": segments_fingerprint(segs),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="diarized")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    todo = [e for e in episodes(args.label) if not valid_npz(EMB_DIR / f"{e['id']}.npz", e["fp"])]
    if args.only:
        todo = [e for e in todo if e["id"] in args.only]
    if args.limit:
        todo = todo[: args.limit]
    print(f"episodes to embed: {len(todo)}", flush=True)
    if not todo:
        return 0

    remote = RemoteGPU(
        os.environ["MEDIA_ASR_REMOTE_HOST"],
        user=os.environ.get("MEDIA_ASR_REMOTE_USER", ""),
        port=int(os.environ.get("MEDIA_ASR_REMOTE_PORT", 22)),
        workdir=os.environ.get("MEDIA_ASR_REMOTE_WORKDIR", "media-asr-work"),
        python=os.environ.get("MEDIA_ASR_REMOTE_PYTHON", r".venv\Scripts\python.exe"),
        shell=os.environ.get("MEDIA_ASR_REMOTE_SHELL", "cmd"),
        key_filename=os.environ.get("MEDIA_ASR_REMOTE_KEY"),
        timeout=4 * 3600,
    )
    remote.connect()
    failures = 0
    try:
        remote._upload_script("embed_speakers_worker.py")
        remote._upload_package()
        out_rel = "jobs/speaker_embeddings"
        remote._mkdirs(remote._path(out_rel))
        for start in range(0, len(todo), args.batch):
            batch = todo[start : start + args.batch]
            job_rel = posixpath.join("jobs", f"embed_job_{os.getpid()}_{start}.json")
            with remote._sftp.open(remote._path(job_rel), "w") as handle:
                handle.write(json.dumps({"out_dir": out_rel, "episodes": batch}))
            command = remote._cd_prefix() + f'{remote.python} embed_speakers_worker.py --job "{job_rel}"'
            try:
                remote._exec(command, timeout=remote.timeout, progress=lambda m: print(m, flush=True))
            except Exception as exc:  # per-episode failures are reported by the worker
                print(f"batch {start} reported failures: {str(exc)[-600:]}", flush=True)
            for episode in batch:
                remote_file = remote._path(out_rel, f"{episode['id']}.npz")
                target = EMB_DIR / f"{episode['id']}.npz"
                part = target.with_name(target.name + ".part")
                try:
                    remote._sftp.stat(remote_file)
                    remote._sftp.get(remote_file, str(part))
                    if not valid_npz(part, episode["fp"]):
                        raise ValueError("downloaded file failed validation")
                    part.replace(target)
                    remote._sftp.remove(remote_file)
                except (IOError, ValueError) as exc:
                    failures += 1
                    print(f"MISSING {episode['id']}: {exc}", flush=True)
                finally:
                    part.unlink(missing_ok=True)
            try:
                remote._sftp.remove(remote._path(job_rel))
            except IOError:
                pass
            print(f"progress: batch {start // args.batch + 1} done", flush=True)
    finally:
        remote.close()
    print(f"finished failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
