"""Helpers shared by the All-In speaker scripts (scope, transcript fingerprint)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FEED_MARKER = "allinchamathjason"
# All-In items that did not come from the RSS feed (the YouTube copy of an episode).
EXTRA_ITEM_IDS = {"https-www-youtube-com-watch-v-f9423524"}


def is_allin(item_dir: Path) -> bool:
    meta = item_dir / "meta.json"
    if not meta.exists():
        return False
    data = json.loads(meta.read_text(encoding="utf-8"))
    return FEED_MARKER in (data.get("feed_url") or "") or item_dir.name in EXTRA_ITEM_IDS


def segments_fingerprint(segments: list[dict]) -> str:
    """Hash of segment boundaries; embeddings are only valid for this exact segmentation."""
    bounds = [[round(float(s["start"]), 3), round(float(s["end"]), 3)] for s in segments]
    return hashlib.sha256(json.dumps(bounds).encode()).hexdigest()[:16]


def labels_fingerprint(segments: list[dict]) -> str:
    """Hash of boundaries *and* speaker labels; a speaker map is only valid for this labelling."""
    rows = [[round(float(s["start"]), 3), round(float(s["end"]), 3), s.get("speaker")] for s in segments]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]
