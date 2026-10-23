#!/usr/bin/env python3
"""Find episodes where diarization merged several people into one label and redo them.

pyannote sometimes puts host and guest into a single label (e.g. one label with
98% of the speaking time that both asks and answers the questions). For every
done item without ``rediarize.json`` whose largest label holds more than
``--share`` of the speaking time, DeepSeek reads samples of that label and
decides whether it mixes several people and how many people talk in the
episode. Merged items whose largest label holds over 90% of the time, or with
3+ people, are re-run on the remote GPU with that count plus one
(an extra cluster absorbs intro voices and clips; with the bare count pyannote
can still merge host and guest, e.g. 硅谷101 E247),
their ``enrich.json`` is removed so ``enrich_podcasts.py`` names them again,
and ``rediarize.json`` records the decision so no item is checked twice.

Run through Vault with ``DEEPSEEK_API_KEY`` and ``HF_TOKEN``; the GPU host comes
from ``MEDIA_ASR_REMOTE_ENV`` as for ``media-asr-remote``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import httpx

from media_asr.config import DB_PATH, ITEMS_DIR
from media_asr.util import atomic_write, now_iso

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_podcasts import chat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """A podcast episode was split into speaker labels by automatic diarization.
Episode title: {title}
Speaking-time share per label: {shares}

Below are samples spread across the episode from the largest label {label},
followed by the opening of the transcript.

Samples of {label}:
{samples}

Opening of the transcript:
{opening}

Does {label} clearly contain more than one person (for example an interviewer
asking questions and a guest answering, or two people introducing themselves)?
How many distinct people take part in the conversation (hosts and guests; do
not count ads, music or short clips of other recordings)?

Return only JSON: {{"merged": true, "people": 2, "reason": "short reason"}}
"""


def label_shares(segments: list[dict]) -> Counter:
    shares: Counter = Counter()
    for seg in segments:
        shares[seg.get("speaker") or "?"] += float(seg["end"]) - float(seg["start"])
    total = sum(shares.values()) or 1.0
    return Counter({k: v / total for k, v in shares.items()})


def build_prompt(title: str, segments: list[dict], shares: Counter) -> tuple[str, str]:
    label = shares.most_common(1)[0][0]
    own = [s for s in segments if (s.get("speaker") or "?") == label]
    step = max(1, len(own) // 12)
    samples = "\n".join(f"- {s['text'][:300]}" for s in own[::step][:12])
    opening, size = [], 0
    for seg in segments:
        line = f"{seg.get('speaker') or '?'}: {seg['text']}"
        if size + len(line) > 4000:
            break
        opening.append(line)
        size += len(line)
    text = PROMPT.format(
        title=title,
        shares=", ".join(f"{k} {v:.0%}" for k, v in shares.most_common()),
        label=label,
        samples=samples,
        opening="\n".join(opening),
    )
    return label, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share", type=float, default=0.75, help="largest-label share that triggers a check"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only report decisions; change nothing"
    )
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th item, for parallel GPUs")
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY; run through `vibe vault run`", file=sys.stderr)
        return 1

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    shard, shards = (int(x) for x in args.shard.split("/"))
    rows = db.execute("SELECT * FROM items WHERE status = 'done' ORDER BY published_at").fetchall()
    failed = 0
    with httpx.Client(trust_env=False) as client:
        for row in rows[shard::shards]:
            item_dir = ITEMS_DIR / row["id"]
            marker = item_dir / "rediarize.json"
            transcript = item_dir / "transcript.json"
            if marker.exists() or not transcript.exists():
                continue
            segments = [
                s
                for s in json.loads(transcript.read_text(encoding="utf-8"))["segments"]
                if str(s.get("text", "")).strip()
            ]
            shares = label_shares(segments)
            if not segments or shares.most_common(1)[0][1] <= args.share:
                continue
            label, prompt = build_prompt(row["title"], segments, shares)
            try:
                verdict = chat(
                    client,
                    key,
                    "You audit speaker diarization. Output valid JSON only.",
                    prompt,
                    400,
                )
            except Exception as exc:  # noqa: BLE001 - leave unmarked, retried next run
                failed += 1
                print(f"FAIL check {row['id']}: {exc}", flush=True)
                continue
            merged = bool(verdict.get("merged"))
            people = int(verdict.get("people") or 0)
            before = round(shares[label], 3)
            print(
                f"{'MERGED' if merged else 'ok    '} {before:.0%} people={people} {row['title'][:50]} "
                f"| {verdict.get('reason', '')[:100]}",
                flush=True,
            )
            if args.dry_run:
                continue
            record = {"checked_at": now_iso(), "label": label, "share_before": before, **verdict}
            # Library run 2026-10-23: re-diarizing every flagged episode moved the largest
            # label by <3 points in 81 of 86 cases (a guest holding ~80% of a long
            # interview is normal); it only helped hard merges and 3+ people.
            if merged and people >= 2 and (before > 0.9 or people >= 3):
                lang = "zh" if (row["language"] or "").startswith("zh") else "en"
                cmd = [
                    str(ROOT / "media-asr-remote"),
                    "run",
                    "--ids",
                    row["id"],
                    "--language",
                    lang,
                    "--diarize",
                    "--diarization-backend",
                    "pyannote",
                    "--num-speakers",
                    str(people + 1),
                ]
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if result.returncode != 0:
                    failed += 1
                    print(
                        f"FAIL rerun {row['id']}: {result.stdout[-300:]} {result.stderr[-300:]}",
                        flush=True,
                    )
                    continue
                after = label_shares(
                    json.loads(transcript.read_text(encoding="utf-8"))["segments"]
                ).most_common(1)[0][1]
                record["share_after"] = round(after, 3)
                (item_dir / "enrich.json").unlink(missing_ok=True)
                print(f"       rerun: largest label {before:.0%} -> {after:.0%}", flush=True)
            atomic_write(marker, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
