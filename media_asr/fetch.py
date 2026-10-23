from __future__ import annotations

import calendar
import html
import json
import mimetypes
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

from .util import run_cmd, sanitize_proxy_env

sanitize_proxy_env()

import feedparser  # noqa: E402
import httpx  # noqa: E402
import yt_dlp  # noqa: E402

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 media-asr/0.1"
)

MEDIA_EXTS = {
    ".mp3",
    ".m4a",
    ".m4b",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".m4v",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
}

VIDEO_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "bilibili.com",
    "www.bilibili.com",
    "m.bilibili.com",
    "vimeo.com",
    "www.vimeo.com",
    "ted.com",
    "www.ted.com",
}


@dataclass
class ItemSpec:
    source: str
    source_type: str
    title: str
    duration: float | None = None
    published_at: str | None = None
    audio_url: str | None = None
    page_url: str | None = None
    feed_url: str | None = None
    extra: dict = field(default_factory=dict)

    def meta_json(self) -> str:
        return json.dumps(
            {
                "audio_url": self.audio_url,
                "page_url": self.page_url,
                "feed_url": self.feed_url,
                **self.extra,
            },
            ensure_ascii=False,
        )


@dataclass
class ResolveResult:
    specs: list[ItemSpec]
    feed_url: str | None = None
    feed_title: str | None = None


def resolve_source(url: str, *, latest: int = 1) -> ResolveResult:
    """Turn a URL, feed, or local file into one or more downloadable items."""
    path = Path(url).expanduser()
    if path.exists() and path.is_file():
        return ResolveResult(
            [
                ItemSpec(
                    source=str(path.resolve()),
                    source_type="file",
                    title=path.stem,
                    audio_url=None,
                    page_url=None,
                    extra={"local": True},
                )
            ]
        )

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"not a local file or http(s) URL: {url}")

    suffix = Path(parsed.path).suffix.lower()
    if suffix in MEDIA_EXTS:
        return ResolveResult(
            [
                ItemSpec(
                    source=url,
                    source_type="audio",
                    title=Path(parsed.path).stem or "untitled",
                    audio_url=url,
                    page_url=url,
                )
            ]
        )

    if parsed.netloc.lower() in VIDEO_HOSTS:
        return ResolveResult(_ytdlp_items(url, latest=latest))

    if _looks_like_feed_url(url):
        text = _http_text(url)
        specs, title = _feed_items(text, url, latest)
        return ResolveResult(specs, feed_url=url, feed_title=title)

    kind, payload, page_title = _probe(url)
    if kind == "feed":
        specs, title = _feed_items(_http_text(url), url, latest)
        return ResolveResult(specs, feed_url=url, feed_title=title)
    if kind == "feed_link":
        text = _http_text(payload)
        specs, title = _feed_items(text, payload, latest)
        return ResolveResult(specs, feed_url=payload, feed_title=title)
    if kind == "media":
        return ResolveResult(
            [
                ItemSpec(
                    source=url,
                    source_type="podcast",
                    title=page_title or Path(parsed.path).stem or "untitled",
                    audio_url=payload or url,
                    page_url=url,
                )
            ]
        )

    return ResolveResult(_ytdlp_items(url, latest=latest))


def _looks_like_feed_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith((".xml", ".rss", ".atom")):
        return True
    return bool(re.search(r"(^|/)(feed|rss|atom)(/|$)", path))


def _http_text(url: str, timeout: float = 45) -> str:
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _probe(url: str) -> tuple[str, str, str | None]:
    """Fetch just enough of an unknown URL to classify it.

    Returns ``(kind, payload, title)``.  For ``media``, payload is the direct
    audio URL found in the page (``og:audio`` or ``<audio>``).
    """
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=30,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            with client.stream("GET", url, headers={"Range": "bytes=0-65535"}) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes(8192):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= 65536:
                        break
                head = b"".join(chunks)
    except Exception:
        return "page", "", None

    page_title = _html_title(head.decode("utf-8", errors="ignore"))

    if content_type.startswith("audio/") or content_type.startswith("video/"):
        return "media", url, page_title
    if "xml" in content_type or head.lstrip().startswith(b"<?xml"):
        text = head.decode("utf-8", errors="ignore")
        if "<rss" in text[:2000] or "<feed" in text[:2000]:
            return "feed", "", page_title
    text = head.decode("utf-8", errors="ignore")
    if "<rss" in text[:2000] or "<feed" in text[:2000]:
        return "feed", "", page_title

    match = re.search(
        r"<link[^>]+type=[\"']application/(?:rss|atom)\+xml[\"'][^>]*>", text, re.I
    )
    if match:
        href = re.search(r"href=[\"']([^\"']+)", match.group(0), re.I)
        if href:
            return "feed_link", urljoin(url, html.unescape(href.group(1))), page_title

    audio = re.search(
        r"<(?:audio|source)[^>]+src=[\"']([^\"']+\.(?:mp3|m4a|aac|ogg|opus|wav|flac))",
        text,
        re.I,
    )
    if audio:
        return "media", urljoin(url, html.unescape(audio.group(1))), page_title
    meta_audio = re.search(
        r"<meta[^>]+property=[\"']og:audio[\"'][^>]+content=[\"']([^\"']+)", text, re.I
    )
    if meta_audio:
        return "media", urljoin(url, html.unescape(meta_audio.group(1))), page_title
    meta_audio = re.search(
        r"<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+property=[\"']og:audio[\"']", text, re.I
    )
    if meta_audio:
        return "media", urljoin(url, html.unescape(meta_audio.group(1))), page_title
    return "page", text, page_title


def _html_title(text: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    if not match:
        return None
    title = html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()
    # 小宇宙 appends "| 小宇宙 - 听播客，上小宇宙" to every page title.
    title = re.sub(r"\s*\|\s*小宇宙.*$", "", title).strip()
    return title or None


def _parse_duration(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" not in text:
            return float(text)
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _entry_timestamp(entry: dict) -> float:
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            try:
                return float(calendar.timegm(value))
            except Exception:
                pass
    return 0.0


def _entry_audio(entry: dict) -> str | None:
    for enclosure in entry.get("enclosures", []) or []:
        href = enclosure.get("href") or enclosure.get("url")
        if href:
            return href
    for link in entry.get("links", []) or []:
        if link.get("rel") == "enclosure" and link.get("href"):
            return link["href"]
        if link.get("type", "").startswith("audio/") and link.get("href"):
            return link["href"]
    for media in entry.get("media_content", []) or []:
        if media.get("url"):
            return media["url"]
    return None


def _feed_items(text: str, feed_url: str, latest: int) -> tuple[list[ItemSpec], str | None]:
    parsed = feedparser.parse(text)
    if not parsed.entries and parsed.bozo:
        raise RuntimeError(f"cannot parse feed: {feed_url}")
    feed_title = parsed.feed.get("title") if parsed.feed else None
    entries = list(parsed.entries)
    if any(_entry_timestamp(entry) for entry in entries):
        entries.sort(key=_entry_timestamp, reverse=True)
    if latest > 0:
        entries = entries[:latest]

    specs: list[ItemSpec] = []
    for entry in entries:
        audio_url = _entry_audio(entry)
        page_url = entry.get("link")
        source = audio_url or page_url
        if not source:
            continue
        timestamp = _entry_timestamp(entry)
        published = (
            datetime.fromtimestamp(timestamp, timezone.utc).replace(microsecond=0).isoformat()
            if timestamp
            else entry.get("published") or entry.get("updated")
        )
        specs.append(
            ItemSpec(
                source=source,
                source_type="podcast",
                title=entry.get("title") or page_url or "untitled",
                duration=_parse_duration(
                    entry.get("itunes_duration") or entry.get("duration")
                ),
                published_at=published,
                audio_url=audio_url,
                page_url=page_url,
                feed_url=feed_url,
                extra={"feed_title": feed_title, "author": entry.get("author")},
            )
        )
    return specs, feed_title


def _ytdlp_items(url: str, *, latest: int) -> list[ItemSpec]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(f"yt-dlp could not resolve {url}: {exc}") from exc
    if not info:
        raise RuntimeError(f"yt-dlp returned no metadata for {url}")

    if info.get("_type") == "playlist":
        entries = [entry for entry in (info.get("entries") or []) if entry]
        if latest > 0:
            entries = entries[:latest]
        return [_spec_from_info(entry, fallback_url=url) for entry in entries]
    return [_spec_from_info(info, fallback_url=url)]


def _spec_from_info(info: dict, *, fallback_url: str) -> ItemSpec:
    page_url = info.get("webpage_url") or info.get("original_url") or info.get("url")
    if not page_url or page_url == "NA":
        page_url = fallback_url
    upload_date = info.get("upload_date")
    published = None
    if upload_date and re.fullmatch(r"\d{8}", str(upload_date)):
        published = (
            f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}T00:00:00+00:00"
        )
    return ItemSpec(
        source=page_url,
        source_type="video",
        title=info.get("title") or "untitled",
        duration=info.get("duration"),
        published_at=published,
        audio_url=page_url,
        page_url=page_url,
        extra={
            "uploader": info.get("uploader"),
            "extractor": info.get("extractor"),
            "webpage_url": page_url,
        },
    )


def download(item: dict, dest_dir: Path, progress: Callable[[str], None] | None = None) -> tuple[Path, bool]:
    """Download an item into ``dest_dir``.  Returns (path, is_local)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    source = item["source"]
    local = Path(source).expanduser()
    if local.exists() and local.is_file():
        return local, True

    meta = {}
    if item.get("meta_json"):
        try:
            meta = json.loads(item["meta_json"])
        except json.JSONDecodeError:
            meta = {}
    download_url = meta.get("audio_url") or source

    if progress:
        progress(f"downloading {download_url}")

    referer = meta.get("page_url")
    direct_first = item.get("source_type") in {"podcast", "audio"}
    if direct_first:
        try:
            path = _download_direct_retry(download_url, dest_dir, referer=referer, progress=progress)
        except Exception as direct_error:
            if progress:
                progress(f"direct download failed, trying yt-dlp: {direct_error}")
            try:
                path = _download_ytdlp(download_url, dest_dir, referer=referer)
            except Exception as ytdlp_error:
                raise RuntimeError(
                    f"download failed for {download_url}\n"
                    f"direct: {direct_error}\nytdlp: {ytdlp_error}"
                ) from ytdlp_error
    else:
        try:
            path = _download_ytdlp(download_url, dest_dir, referer=referer)
        except Exception as ytdlp_error:
            if progress:
                progress(f"yt-dlp failed, trying direct download: {ytdlp_error}")
            try:
                path = _download_direct(download_url, dest_dir, referer=referer)
            except Exception as direct_error:
                raise RuntimeError(
                    f"download failed for {download_url}\n"
                    f"yt-dlp: {ytdlp_error}\ndirect: {direct_error}"
                ) from direct_error

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"downloaded file is empty: {path}")
    return path, False


def _download_ytdlp(url: str, dest_dir: Path, referer: str | None = None) -> Path:
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
    }
    if referer:
        options["http_headers"] = {"Referer": referer}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    candidates = [
        path
        for path in dest_dir.glob("source.*")
        if path.suffix not in {".part", ".ytdl"} and path.is_file()
    ]
    if not candidates:
        raise RuntimeError("yt-dlp reported success but produced no file")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _download_direct_retry(
    url: str, dest_dir: Path, *, referer: str | None, progress=None, attempts: int = 3
) -> Path:
    """Direct download with retries: podcast CDNs drop connections now and then
    (SSL EOF, server disconnect) and a single failure would fail the episode."""
    for attempt in range(1, attempts + 1):
        try:
            return _download_direct(url, dest_dir, referer=referer)
        except Exception as exc:
            if attempt == attempts:
                raise
            if progress:
                progress(f"direct download attempt {attempt} failed, retrying: {exc}")
            time.sleep(15 * attempt)
    raise AssertionError("unreachable")


def _download_direct(url: str, dest_dir: Path, referer: str | None = None) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=300),
        headers=headers,
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            if suffix not in MEDIA_EXTS:
                guessed = mimetypes.guess_extension(
                    response.headers.get("content-type", "").split(";")[0].strip()
                )
                suffix = guessed or ".bin"
            target = dest_dir / f"source{suffix}"
            with target.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
    return target


def normalize_audio(source: Path, target: Path) -> None:
    """Decode any audio/video container to 16 kHz mono PCM WAV."""
    run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )