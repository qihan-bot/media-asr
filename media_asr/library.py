"""Podcast library: the show list, RSS show notes and the daily update.

A podcast library is a data directory (``MEDIA_ASR_DATA``) whose
``library.toml`` lists the shows, one ``[[show]]`` table each:

    [[show]]
    name = "Latent Space"                     # optional, for logs
    feed = "https://api.substack.com/feed/podcast/1084089.rss"
    language = "en"                           # zh or en
    hosts = "swyx and Alessio Fanelli"        # optional speaker-naming hint

An episode belongs to a show when its feed URL contains ``feed``; use the URL
the feed was subscribed with (``media-asr add <feed> --latest N``).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import feedparser
import httpx

from .config import DATA_DIR
from .db import Store
from .render import render_json, render_markdown, render_srt, render_text
from .transcribe import TranscriptSegment
from .util import atomic_write, file_lock, now_iso

ROOT = Path(__file__).resolve().parent.parent
# The source checkout's wrapper: loads MEDIA_ASR_REMOTE_ENV, remote full mode.
REMOTE = ROOT / "media-asr-remote"
MEDIA_ASR = [sys.executable, "-m", "media_asr"]
LANGUAGES = ("zh", "en")
SHOW_KEYS = {"name", "feed", "language", "hosts"}
# Completion markers of the steps built on transcript.json; `media-asr run`
# drops them before it writes a new transcript, so these steps run again.
STEP_MARKERS = ("rediarize.json", "enrich.json", "terms.json")
# Remote ASR progress lines; dropped from the update log.
PROGRESS = re.compile(r"frames/s|still running")


class LibraryError(ValueError):
    """library.toml is missing or invalid."""


@dataclass(frozen=True)
class Show:
    feed: str
    language: str
    name: str = ""
    hosts: str = ""

    def matches(self, feed_url: str | None) -> bool:
        return self.feed in (feed_url or "")


def config_path() -> Path:
    return DATA_DIR / "library.toml"


def load_shows(path: Path | None = None, *, missing_ok: bool = False) -> list[Show]:
    path = path or config_path()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if missing_ok:
            return []
        raise LibraryError(
            f"{path} not found; start from docs/examples/podcast-library.toml"
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise LibraryError(f"{path}: {exc}") from None
    entries = data.get("show", [])
    if set(data) - {"show"} or not all(isinstance(entry, dict) for entry in entries):
        raise LibraryError(f"{path}: expected only [[show]] tables")
    shows = []
    for number, entry in enumerate(entries, start=1):
        where = f"{path}: show {number}"
        if set(entry) - SHOW_KEYS:
            raise LibraryError(f"{where}: unknown keys {sorted(set(entry) - SHOW_KEYS)}")
        if not isinstance(entry.get("feed"), str) or not entry["feed"].strip():
            raise LibraryError(f"{where}: feed (the RSS URL) is required")
        if entry.get("language") not in LANGUAGES:
            raise LibraryError(f"{where}: language must be one of {', '.join(LANGUAGES)}")
        if any(not isinstance(entry.get(key, ""), str) for key in ("name", "hosts")):
            raise LibraryError(f"{where}: name and hosts must be strings")
        shows.append(
            Show(
                feed=entry["feed"].strip(),
                language=entry["language"],
                name=entry.get("name", "").strip(),
                hosts=entry.get("hosts", "").strip(),
            )
        )
    if not shows:
        raise LibraryError(f"{path}: no [[show]] entries")
    for show in shows:
        for other in shows:
            if other is not show and show.feed in other.feed:
                raise LibraryError(f"{path}: feed {show.feed} also matches {other.feed}")
    return shows


def host_hints(shows: list[Show], feed_url: str | None) -> str:
    """Recurring hosts of the show an episode belongs to, or ''."""
    return next((show.hosts for show in shows if show.hosts and show.matches(feed_url)), "")


def load_show_notes(db: sqlite3.Connection) -> tuple[dict[str, str], dict[str, str]]:
    """Map audio URL and "feed URL<TAB>episode title" -> plain-text show notes,
    and feed URL -> show title. The title key survives a feed changing hosts."""
    notes: dict[str, str] = {}
    titles: dict[str, str] = {}
    for url, title in db.execute("SELECT url, title FROM feeds"):
        titles[url] = title or url
        try:
            parsed = feedparser.parse(httpx.get(url, timeout=60, follow_redirects=True).text)
        except Exception as exc:  # noqa: BLE001 - show notes are optional evidence
            print(f"feed fetch failed {url}: {exc}", flush=True)
            continue
        for entry in parsed.entries:
            body = entry.get("summary") or entry.get("description") or ""
            text = html.unescape(re.sub(r"<[^>]+>", " ", body))
            text = re.sub(r"\s+", " ", text).strip()[:3000]
            notes[f"{url}\t{entry.get('title', '')}"] = text
            for link in entry.get("enclosures", []):
                if link.get("href"):
                    notes[link["href"]] = text
    return notes, titles


def episode_notes(notes: dict, row: sqlite3.Row) -> str:
    meta = json.loads(row["meta_json"] or "{}")
    return (
        notes.get(meta.get("audio_url") or row["source"])
        or notes.get(f"{row['feed_url']}\t{row['title']}")
        or "(not available)"
    )


def read_json(path: Path):
    """Parse a JSON file and return it with the SHA-256 of the bytes read, so a
    step can check before committing that its input has not changed."""
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def write_named(
    item_dir: Path,
    *,
    item_id: str,
    title: str,
    source: str,
    duration: float | None,
    language: str | None,
    model: str,
    segments: list[dict],
) -> None:
    """Write transcript_named.{md,srt,txt,json} for these segments."""
    named = [
        TranscriptSegment(
            start=float(s["start"]), end=float(s["end"]), text=s["text"], speaker=s.get("speaker")
        )
        for s in segments
    ]
    meta = {
        "title": title,
        "source": source,
        "duration": duration,
        "language": language,
        "model": model,
    }
    texts = {
        "md": render_markdown(**meta, transcribed_at=now_iso(), segments=named),
        "srt": render_srt(named),
        "txt": render_text(named),
        "json": render_json(item_id=item_id, **meta, segments=named),
    }
    for suffix, text in texts.items():
        atomic_write(item_dir / f"transcript_named.{suffix}", text)


def log(message: str) -> None:
    print(message, flush=True)


class Shell:
    """Runs the update's commands from the source root; tests replace it."""

    def check(self, cmd: list[str], env: dict[str, str], timeout: float) -> bool:
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0

    def stream(
        self, cmd: list[str], env: dict[str, str], drop: re.Pattern | None = None
    ) -> tuple[int, str]:
        """Print the command's output except lines matching ``drop``; return
        the exit code and the printed output."""
        kept = []
        with subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as proc:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace")
                if drop is None or not drop.search(line):
                    print(line, end="", flush=True)
                    kept.append(line)
        return proc.returncode, "".join(kept)


def update(
    store: Store,
    shows: list[Show],
    *,
    gpu_env: str | None = None,
    fallback_gpu_env: str | None = None,
    latest: int = 5,
    workers: int = 4,
    shell: Shell | None = None,
    sleep=time.sleep,
) -> int:
    """Daily refresh: feeds -> per-show remote GPU transcription with pyannote
    diarization -> rediarize -> enrich -> correct-terms.

    The GPU host is ``gpu_env`` if it passes ``remote-check``, else
    ``fallback_gpu_env`` (else the wrapper's default env file). Every step is
    idempotent, so a failed run can be re-run as is. Returns 1 if any feed,
    episode or step failed, after doing all the work it could. One update per
    library at a time: a second one exits 1 at once.
    """
    lock = store.path.parent / ".library-update.lock"
    with file_lock(lock, wait=False) as acquired:
        if not acquired:
            log(f"error: another library-update holds {lock}")
            return 1
        return _update(store, shows, gpu_env, fallback_gpu_env, latest, workers, shell, sleep)


def _update(store, shows, gpu_env, fallback_gpu_env, latest, workers, shell, sleep) -> int:
    if not REMOTE.exists():
        raise LibraryError(f"library-update needs the source checkout wrapper {REMOTE}")
    shell = shell or Shell()
    status = 0
    log(f"== {now_iso()} start")
    env = dict(os.environ, MEDIA_ASR_DATA=str(DATA_DIR.resolve()), PYTHONUNBUFFERED="1")
    if gpu_env and shell.check(
        [str(REMOTE), "remote-check"], {**env, "MEDIA_ASR_REMOTE_ENV": gpu_env}, timeout=90
    ):
        env["MEDIA_ASR_REMOTE_ENV"] = gpu_env
    else:
        if gpu_env:
            log(f"gpu: {gpu_env} failed remote-check")
        if fallback_gpu_env:
            env["MEDIA_ASR_REMOTE_ENV"] = fallback_gpu_env
    log(f"gpu: {env.get('MEDIA_ASR_REMOTE_ENV', '~/.config/media-asr/remote.env (default)')}")

    log("== refresh feeds")
    for attempt in (1, 2, 3):
        code, out = shell.stream([*MEDIA_ASR, "refresh", "--latest", str(latest)], env)
        if code == 0 and not re.search(r"^feed failed", out, re.M):
            break
        if attempt == 3:
            log("WARN: some feeds failed 3 times")
            status = 1
        else:
            sleep(attempt * 30)

    log("== transcribe")
    subscribed = [row["url"] for row in store.list_feeds()]
    for show in shows:
        if not any(show.matches(url) for url in subscribed):
            log(f"WARN: {show.name or show.feed} is not subscribed: media-asr add {show.feed}")
            status = 1
            continue
        code, _ = shell.stream(
            [
                str(REMOTE),
                "run",
                "--retry-failed",
                "--feed-contains",
                show.feed,
                "--language",
                show.language,
                "--diarize",
                "--diarization-backend",
                "pyannote",
            ],
            env,
            drop=PROGRESS,
        )
        if code:
            status = 1
    failed = store.list_items(status="failed", limit=1000)
    if failed:
        log("FAILED episodes:")
        for row in failed:
            log(f"{row['id']}  {row['title']}")
        status = 1

    for title, cmd, drop in (
        # Through the wrapper: rediarize re-runs episodes on the GPU in-process.
        (
            "re-diarize episodes where one label merged several people",
            [str(REMOTE), "rediarize"],
            PROGRESS,
        ),
        ("punctuation + speaker names", [*MEDIA_ASR, "enrich", "--workers", str(workers)], None),
        (
            "term correction (audited replacement pairs, see terms.json)",
            [*MEDIA_ASR, "correct-terms", "--workers", str(workers)],
            None,
        ),
    ):
        log(f"== {title}")
        code, _ = shell.stream(cmd, env, drop=drop)
        if code:
            status = 1

    log(f"== {now_iso()} end status={status}")
    shell.stream([*MEDIA_ASR, "stats"], env)
    return status
