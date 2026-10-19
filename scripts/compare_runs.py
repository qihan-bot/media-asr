#!/usr/bin/env python3
"""Compare two transcript runs of the same media item.

Usage:
    python3 scripts/compare_runs.py <item-dir> \
        --a . --b runs/medium-int8-beam5 --out comparison.md

The script aligns runs by 30-second windows, reports key-term hits, and shows
the windows where the two models disagree the most.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path


TERMS = [
    "厚雪长波",
    "王浩岚",
    "美轮美换",
    "中期选举",
    "通胀",
    "MAGA",
    "数据中心",
    "美联储",
    "债券卫士",
    "门罗主义",
    "万斯",
    "鲁比奥",
    "格陵兰",
    "爱荷华",
    "俄亥俄",
    "堪萨斯",
    "德州",
    "参议院",
    "众议院",
    "AI",
    "AGI",
    "共和党",
    "民主党",
]

PUNCT = re.compile(r"[\s，。！？；：、,.!?;:\"'“”‘’（）()《》\[\]【】—\-]+")


def load_run(path: Path) -> dict:
    transcript = path / "transcript.json"
    if not transcript.exists():
        raise SystemExit(f"missing {transcript}")
    data = json.loads(transcript.read_text(encoding="utf-8"))
    run = {}
    run_path = path / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    text = "".join(segment.get("text", "") for segment in segments)
    return {"data": data, "run": run, "segments": segments, "text": text}


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def buckets(segments: list[dict], size: int = 30) -> dict[int, str]:
    result: dict[int, list[str]] = defaultdict(list)
    for segment in segments:
        start = int(float(segment.get("start", 0)) // size)
        result[start].append(segment.get("text", ""))
    return {key: "".join(value) for key, value in result.items()}


def max_gap(segments: list[dict]) -> float:
    starts = sorted(float(segment.get("start", 0)) for segment in segments)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    return max(gaps) if gaps else 0.0


def hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def term_counts(text: str) -> dict[str, int]:
    return {term: text.count(term) for term in TERMS}


def first_hit(segments: list[dict], term: str) -> tuple[float, str] | None:
    for segment in segments:
        if term in segment.get("text", ""):
            return float(segment.get("start", 0)), segment.get("text", "")
    return None


def truncate(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("item_dir", type=Path)
    parser.add_argument("--a", default=".", help="primary run directory, relative to item dir")
    parser.add_argument("--b", required=True, help="comparison run directory, relative to item dir")
    parser.add_argument("--out", default="comparison.md")
    parser.add_argument("--a-elapsed", type=float, default=None)
    args = parser.parse_args()

    item_dir = args.item_dir.expanduser().resolve()
    a = load_run((item_dir / args.a).resolve())
    b = load_run((item_dir / args.b).resolve())

    buckets_a = buckets(a["segments"])
    buckets_b = buckets(b["segments"])
    fine_a = buckets(a["segments"], size=10)
    fine_b = buckets(b["segments"], size=10)
    common = sorted(set(buckets_a) & set(buckets_b))
    scored = []
    for key in common:
        left = normalize(buckets_a[key])
        right = normalize(buckets_b[key])
        if not left or not right:
            continue
        ratio = difflib.SequenceMatcher(None, left, right).ratio()
        scored.append((ratio, key))

    avg_ratio = sum(ratio for ratio, _ in scored) / len(scored) if scored else 0.0
    worst = sorted(scored)[:8]
    best = sorted(scored, reverse=True)[:5]

    counts_a = term_counts(a["text"])
    counts_b = term_counts(b["text"])

    lines: list[str] = []
    lines.append("# Transcript model comparison\n")
    lines.append(f"Item: `{a['data'].get('title', item_dir.name)}`\n")
    lines.append(f"Source: {a['data'].get('source', '')}\n")

    lines.append("## Run metrics\n")
    lines.append("| Run | Model | Segments | Chars | Elapsed | Realtime | Max gap |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for label, run in (("A", a), ("B", b)):
        meta = run["run"]
        elapsed = meta.get("elapsed_seconds") if meta else None
        if label == "A" and elapsed is None:
            elapsed = args.a_elapsed
        duration = run["data"].get("duration") or 0
        realtime = f"{duration / elapsed:.2f}×" if elapsed and duration else "—"
        elapsed_text = f"{elapsed / 60:.1f} min" if elapsed else "—"
        model = run["data"].get("model") or meta.get("model") or "—"
        lines.append(
            f"| {label} | {model} | {len(run['segments'])} | {len(run['text'])} | "
            f"{elapsed_text} | {realtime} | {max_gap(run['segments']):.0f}s |"
        )
    lines.append("")
    lines.append(f"Average 30-second window similarity: **{avg_ratio:.1%}**\n")

    lines.append("## Key terms\n")
    lines.append("| Term | A hits | B hits | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for term in TERMS:
        left, right = counts_a[term], counts_b[term]
        if left == 0 and right == 0:
            continue
        lines.append(f"| {term} | {left} | {right} | {right - left:+d} |")
    lines.append("")

    lines.append("## Likely corrections (A missing, B present)\n")
    corrections = [term for term in TERMS if counts_a[term] == 0 and counts_b[term] > 0]
    if not corrections:
        lines.append("_None in the configured term list._\n")
    for term in corrections:
        hit = first_hit(b["segments"], term)
        if not hit:
            continue
        start, text = hit
        lines.append(f"- **{term}** at `{hms(start)}` — {truncate(text)}")
    lines.append("")

    all_fine = sorted(set(fine_a) | set(fine_b))
    missing_b = [
        key
        for key in all_fine
        if len(normalize(fine_a.get(key, ""))) > 8
        and len(normalize(fine_b.get(key, ""))) < 3
    ]
    missing_a = [
        key
        for key in all_fine
        if len(normalize(fine_b.get(key, ""))) > 8
        and len(normalize(fine_a.get(key, ""))) < 3
    ]

    lines.append("## Possible omissions in B\n")
    if not missing_b:
        lines.append("_None._\n")
    for key in missing_b:
        lines.append(
            f"- `{hms(key * 10)}–{hms(key * 10 + 10)}` A: {truncate(fine_a.get(key, ''), 220)}"
        )
    lines.append("")

    lines.append("## Possible omissions in A\n")
    if not missing_a:
        lines.append("_None._\n")
    for key in missing_a:
        lines.append(
            f"- `{hms(key * 10)}–{hms(key * 10 + 10)}` B: {truncate(fine_b.get(key, ''), 220)}"
        )
    lines.append("")

    lines.append("## Windows with the largest disagreement\n")
    for ratio, key in worst:
        start = key * 30
        end = start + 30
        lines.append(f"### {hms(start)}–{hms(end)} (similarity {ratio:.0%})\n")
        lines.append("**A**\n")
        lines.append(f"> {truncate(buckets_a[key])}\n")
        lines.append("**B**\n")
        lines.append(f"> {truncate(buckets_b[key])}\n")

    lines.append("## Windows with the closest agreement\n")
    for ratio, key in best:
        start = key * 30
        lines.append(f"- {hms(start)}–{hms(start + 30)}: {ratio:.0%}")

    out_path = item_dir / args.out
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"average similarity: {avg_ratio:.1%}")
    print(f"segments: A={len(a['segments'])}, B={len(b['segments'])}")
    print(f"chars: A={len(a['text'])}, B={len(b['text'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())