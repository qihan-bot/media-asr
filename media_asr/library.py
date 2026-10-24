"""Podcast library: the show list and RSS show notes.

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

import html
import json
import re
import sqlite3
import tomllib
from dataclasses import dataclass
from pathlib import Path

import feedparser
import httpx

from .config import DATA_DIR

ROOT = Path(__file__).resolve().parent.parent
# The source checkout's wrapper: loads MEDIA_ASR_REMOTE_ENV, remote full mode.
REMOTE = ROOT / "media-asr-remote"
LANGUAGES = ("zh", "en")
SHOW_KEYS = {"name", "feed", "language", "hosts"}


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
