#!/usr/bin/env python3
"""Fix Whisper misspellings of All-In host and company names in place.

Touches only transcript text: ``transcript*.{json,txt,md,srt}`` in each
All-In item directory (primary, ``runs/*``) and the ``segments`` rows in
``media.db`` (the FTS index follows via triggers). Dry run by default.
Back up ``data/items`` and ``media.db`` before running with ``--apply``.

The rules are tuned for the All-In corpus: "Sachs"/"Saks"/"Sax" are almost
always David Sacks there. Known non-host uses (Goldman Sachs, other people
named Sachs, Saks the retailer, the saxophone) are skipped by context, but
review the dry-run counts before applying to new material.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

from media_asr.config import DB_PATH, ITEMS_DIR
from media_asr.util import atomic_write

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_common import is_allin  # noqa: E402

# (pattern, canonical, skip_before, skip_after). Patterns are case-insensitive;
# the replacement keeps the casing style of the match (lowercase stays
# lowercase). A match is left alone when the text just before it matches
# ``skip_before`` or the text just after it matches ``skip_after`` (other
# people named Sachs/Saks, Goldman Sachs, the saxophone, the SaaS joke).
OTHER_FIRST_NAMES = r"(?:jeffrey|jeff|lee|jonathan|andrew|oliver|robert|dr\.?|professor|prof\.?|mr\.?|ms\.?|mrs\.?)"
RULES = [
    (r"\bSachs\b", "Sacks", rf"(?:goldman|{OTHER_FIRST_NAMES})\s+$", r"^\s+group\b"),
    (r"\bSaks\b", "Sacks", rf"{OTHER_FIRST_NAMES}\s+$", r"^\s+(?:fifth|global|off|inc|filed|bankrupt\w*|stores?|department)\b"),
    (r"\bSax\b", "Sacks", r"\b(?:the|a|tenor|alto|baritone|soprano|plays?|playing|played)\s+$",
     r"^\s+(?:apocalypse|player|solo|section)\b"),
    (r"\b(?:Freeberg|Freiberg)\b", "Friedberg", None, None),
    (r"\b(?:Jamath|Shamath|Chamas)\b", "Chamath", None, None),
    (r"\bPali(?:ha|a)p[a-z]*\b|\bPaliha Petitiv\b", "Palihapitiya", None, None),
    (r"\bCalicanus\b|\bCalcanus\b", "Calacanis", None, None),
    (r"\bCerebrus\b", "Cerebras", None, None),
    (r"\bGerstener\b", "Gerstner", None, None),
]
COMPILED = [
    (
        re.compile(p, re.IGNORECASE),
        c,
        re.compile(b, re.IGNORECASE) if b else None,
        re.compile(a, re.IGNORECASE) if a else None,
    )
    for p, c, b, a in RULES
]


def fix(text: str, counter: collections.Counter) -> str:
    for pattern, canonical, skip_before, skip_after in COMPILED:
        def repl(match: re.Match, canonical: str = canonical, skip_before=skip_before, skip_after=skip_after) -> str:
            before = match.string[max(0, match.start() - 40) : match.start()]
            after = match.string[match.end() : match.end() + 40]
            if match.group(0).lower() == canonical.lower():
                return match.group(0)
            if (skip_before and skip_before.search(before)) or (skip_after and skip_after.search(after)):
                return match.group(0)
            counter[f"{match.group(0)} -> {canonical}"] += 1
            return canonical.lower() if match.group(0).islower() else canonical
        text = pattern.sub(repl, text)
    return text


def allin_items() -> list:
    return [d for d in sorted(ITEMS_DIR.iterdir()) if is_allin(d)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    counter: collections.Counter = collections.Counter()
    files_changed = 0
    items = allin_items()
    for item_dir in items:
        for path in sorted(item_dir.glob("**/transcript*.*")):
            if path.suffix not in {".json", ".txt", ".md", ".srt"}:
                continue
            raw = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                data = json.loads(raw)
                for seg in data.get("segments", []):
                    seg["text"] = fix(seg.get("text", ""), counter)
                new = json.dumps(data, ensure_ascii=False, indent=2) + ("\n" if raw.endswith("\n") else "")
                if json.loads(new) == json.loads(raw):
                    continue
            else:
                new = fix(raw, collections.Counter())
                if new == raw:
                    continue
            files_changed += 1
            if args.apply:
                atomic_write(path, new)

    conn = sqlite3.connect(DB_PATH)
    ids = [d.name for d in items]
    rows = conn.execute(
        f"SELECT rowid, text FROM segments WHERE item_id IN ({','.join('?' * len(ids))})", ids
    ).fetchall()
    db_counter: collections.Counter = collections.Counter()
    updates = [(new, rowid) for rowid, text in rows if (new := fix(text, db_counter)) != text]
    if args.apply:
        with conn:
            conn.executemany("UPDATE segments SET text = ? WHERE rowid = ?", updates)
    conn.close()

    print(f"items={len(items)} files_changed={files_changed} db_rows_changed={len(updates)}")
    print("replacements in transcript JSON segments (all runs):")
    for key, n in counter.most_common():
        print(f"  {n:>5}  {key}")
    print("applied" if args.apply else "dry run (use --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
