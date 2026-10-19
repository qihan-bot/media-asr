from __future__ import annotations

import hashlib
import os
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


def sanitize_proxy_env() -> None:
    """Drop IPv6 CIDR entries that httpx mis-parses as a URL port.

    The host environment can contain e.g. ``fd7a:115c:a1e0::/48`` in
    ``NO_PROXY``.  httpx splits on ``:`` and raises ``InvalidURL``.  IPv6
    suffixes are not needed for the hosts this tool talks to.
    """
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key)
        if not raw:
            continue
        kept = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "::/" in part or re.fullmatch(r"[0-9a-fA-F:]+/\d+", part):
                continue
            kept.append(part)
        os.environ[key] = ",".join(kept)


def hms(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_ts(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def slugify(text: str, maxlen: int = 48) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text, flags=re.UNICODE)
    text = text.strip("-")
    return text[:maxlen].strip("-") or "item"


def item_id(source: str) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(source, 30)}-{digest}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1200:]
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def ffprobe_duration(path: Path) -> float | None:
    proc = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)