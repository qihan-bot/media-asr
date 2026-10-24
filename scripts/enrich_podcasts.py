#!/usr/bin/env python3
"""Punctuate and name speakers in a multi-show podcast library.

For every done item under ``MEDIA_ASR_DATA`` that has no ``enrich.json`` yet:

1. Punctuation: DeepSeek restores punctuation (and English casing) segment by
   segment. A segment is accepted only if its letters and digits are unchanged
   (NFKC, case-folded, punctuation and whitespace removed); otherwise the
   original text is kept, so this pass can never change what was said.
2. Speakers: DeepSeek maps each ``SPEAKER_XX`` to a person using the show name,
   episode title, the RSS show notes and the transcript. Only high/medium
   confidence names are used; the rest become ``说话人N`` / ``Speaker N``.

Writes ``transcript_named.{md,srt,txt,json}`` and ``enrich.json`` (the
completion marker) next to the original transcript, which is left untouched,
then points the SQLite index at the enriched text and speaker names.

Run through Vault: ``vibe vault run --env DEEPSEEK_API_KEY=DEEPSEEK_API_KEY.value
-- python3 scripts/enrich_podcasts.py``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import httpx

from media_asr.config import DB_PATH, ITEMS_DIR
from media_asr.render import render_all
from media_asr.transcribe import TranscriptSegment
from media_asr.util import atomic_write, now_iso, sanitize_proxy_env

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("MEDIA_ASR_ENRICH_MODEL", "deepseek-flash")
CHUNK_CHARS = {"zh": 2500, "en": 6000}

PUNCT_PROMPT = {
    "zh": (
        "下面是播客语音识别文稿，每行一段，没有标点。请只为每行补上中文标点符号"
        "（，。？！、：；“”《》），行内的英文按英文习惯加标点。不得增删、替换或调换任何"
        "文字、数字和字母，不得合并或拆分行，不得纠正错别字。"
        '只返回 JSON：{"lines": ["第1行", "第2行", ...]}，行数必须与输入相同。'
    ),
    "en": (
        "Below are podcast speech-recognition lines, one segment per line. Restore "
        "punctuation and capitalization only. Do not add, remove, replace or reorder "
        "any word, do not fix grammar or misheard words, do not merge or split lines. "
        'Return only JSON: {"lines": ["line 1", "line 2", ...]} with exactly as many '
        "lines as the input."
    ),
}

# Recurring hosts who rarely say their own names on air (feed URL substring ->
# hint). Only shows where the transcript alone left hosts unnamed.
SHOW_HOSTS = {
    "feed/podcast/1084089": (
        "swyx (Shawn Wang; editor of Latent Space, organizer of the AI Engineer "
        "conference, AIE) and Alessio Fanelli (partner at Kernel Labs)"
    ),
    "rss.buzzsprout.com/2630998": "Brian Heater",
    "destinations/2292818": "Claire Asher",
}
UNNAMED = {
    "zh": {"host": "主持人", "guest": "嘉宾", "other": "说话人"},
    "en": {"host": "Host", "guest": "Guest", "other": "Speaker"},
}

SPEAKER_PROMPT = """You label speakers in a podcast transcript produced by speech recognition and diarization.

Show: {show}
Known recurring hosts: {hosts}
Episode title: {title}
Show notes (from the RSS feed; may list host and guests):
{notes}

Speaker statistics (label: share of speaking time, first lines):
{stats}

Transcript excerpt:
{transcript}

For each SPEAKER_XX label, give the person's name as it is used in the show
(Chinese names in Chinese, English names in English), their role (host, guest,
or other), and your confidence (high, medium, low). Use only evidence from the
show notes and transcript; never invent a name. A known host may be named from role and context
(for example who welcomes the guest or talks about their own projects), but
only with medium or high confidence, and only when something in the transcript or
show notes points to that particular host. Being the host is not enough when
the show has several recurring hosts: then use null for the name with role
"host". If one person was split across
several labels, give each of those labels the same name. If a label is music,
an ad or an intro voice, use role "other". If there is not enough evidence,
use null for the name.

Return only JSON:
{{"speakers": {{"SPEAKER_00": {{"name": "...", "role": "host", "confidence": "high", "evidence": "short reason"}}}}}}
"""


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()


def chat(client: httpx.Client, key: str, system: str, user: str, max_tokens: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    last: Exception | None = None
    for attempt in range(3):
        # Greedy decoding repeats a malformed answer; retries sample a little.
        payload["temperature"] = 0 if attempt == 0 else 0.3
        try:
            response = client.post(
                API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=300
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M))
        except Exception as exc:  # noqa: BLE001 - retried, then surfaced
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"DeepSeek request failed: {last}")


def punctuate(client, key, texts: list[str], lang: str, workers: int) -> tuple[list[str], int]:
    """Return punctuated texts and how many segments were accepted.

    Pass 1 sends large chunks; segments it did not return verbatim (count
    drift, truncation, a "corrected" word) are retried in small chunks once.
    """
    out = list(texts)
    done: set[int] = set()

    def chunked(indices: list[int], budget: int, max_lines: int) -> list[list[int]]:
        chunks, current, size = [], [], 0
        for index in indices:
            if current and (size + len(texts[index]) > budget or len(current) >= max_lines):
                chunks.append(current)
                current, size = [], 0
            current.append(index)
            size += len(texts[index])
        return chunks + ([current] if current else [])

    def run(chunk: list[int]) -> list[tuple[int, str]]:
        lines = [texts[i].replace("\n", " ") for i in chunk]
        tokens = min(
            8000, int(sum(len(line) for line in lines) * (1.6 if lang == "zh" else 0.6)) + 500
        )
        try:
            result = chat(client, key, PUNCT_PROMPT[lang], "\n".join(lines), tokens).get(
                "lines", []
            )
        except Exception:  # noqa: BLE001 - keep the original text for this chunk
            return []
        return [(i, str(new)) for i, new in zip(chunk, result) if isinstance(new, str)]

    todo = [i for i, text in enumerate(texts) if norm(text)]
    budget = CHUNK_CHARS[lang]
    for chunks in (chunked(todo, budget, 400), None):
        if chunks is None:
            chunks = chunked([i for i in todo if i not in done], budget // 3, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for pairs in pool.map(run, chunks):
                for index, new in pairs:
                    if index not in done and new.strip() and norm(new) == norm(texts[index]):
                        out[index] = new.strip()
                        done.add(index)
    return out, len(done)


def speaker_context(segments: list[dict], limit: int = 16000) -> tuple[str, str]:
    totals: dict[str, float] = {}
    samples: dict[str, list[str]] = {}
    for seg in segments:
        label = seg.get("speaker")
        if not label:
            continue
        totals[label] = totals.get(label, 0.0) + float(seg["end"]) - float(seg["start"])
        if len(samples.setdefault(label, [])) < 3:
            samples[label].append(seg["text"][:200])
    whole = sum(totals.values()) or 1.0
    stats = "\n".join(
        f"{label}: {totals[label] / whole:.0%}; " + " / ".join(samples[label])
        for label in sorted(totals)
    )
    lines, size = [], 0
    for seg in segments:
        start = float(seg["start"])
        line = f"[{int(start // 60):02d}:{int(start % 60):02d}] {seg.get('speaker') or '?'}: {seg['text']}"
        if size + len(line) > limit:
            break
        lines.append(line)
        size += len(line) + 1
    return stats, "\n".join(lines)


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


def enrich_item(row: sqlite3.Row, notes: dict, titles: dict, client, key, workers: int) -> dict:
    item_dir = ITEMS_DIR / row["id"]
    data = json.loads((item_dir / "transcript.json").read_text(encoding="utf-8"))
    segments = [s for s in data.get("segments", []) if str(s.get("text", "")).strip()]
    lang = "zh" if (row["language"] or data.get("language") or "").startswith("zh") else "en"
    texts, accepted = punctuate(client, key, [s["text"] for s in segments], lang, workers)
    for seg, text in zip(segments, texts):
        seg["text"] = text

    labels = sorted({s["speaker"] for s in segments if s.get("speaker")})
    names: dict[str, dict] = {}
    if labels:
        stats, excerpt = speaker_context(segments)
        prompt = SPEAKER_PROMPT.format(
            show=titles.get(row["feed_url"], row["feed_url"] or ""),
            hosts=next(
                (v for k, v in SHOW_HOSTS.items() if k in (row["feed_url"] or "")), "(not given)"
            ),
            title=row["title"],
            notes=episode_notes(notes, row),
            stats=stats,
            transcript=excerpt,
        )
        raw = chat(client, key, "You label podcast speakers. Output valid JSON only.", prompt, 2000)
        raw = raw.get("speakers", raw)
        unknown: dict[str, int] = {}
        for label in labels:
            info = raw.get(label) or {}
            name = (info.get("name") or "").strip() if isinstance(info, dict) else ""
            confidence = info.get("confidence", "low") if isinstance(info, dict) else "low"
            if not name or confidence not in ("high", "medium"):
                role = info.get("role") if isinstance(info, dict) else None
                role = role if role in ("host", "guest") else "other"
                unknown[role] = unknown.get(role, 0) + 1
                sep = "" if lang == "zh" else " "
                name = f"{UNNAMED[lang][role]}{sep}{unknown[role]}"
            names[label] = {
                "name": name,
                "role": info.get("role") if isinstance(info, dict) else None,
                "confidence": confidence,
                "evidence": info.get("evidence", "") if isinstance(info, dict) else "",
            }
        for seg in segments:
            if seg.get("speaker") in names:
                seg["speaker"] = names[seg["speaker"]]["name"]

    named = [
        TranscriptSegment(
            start=float(s["start"]), end=float(s["end"]), text=s["text"], speaker=s.get("speaker")
        )
        for s in segments
    ]
    temp_dir = item_dir / "_named_tmp"
    render_all(
        out_dir=temp_dir,
        item_id=row["id"],
        title=row["title"],
        source=data.get("source") or row["source"],
        duration=data.get("duration") or row["duration"],
        language=data.get("language") or row["language"],
        model=(data.get("model") or "") + f" + {MODEL} punctuation/speaker names",
        transcribed_at=now_iso(),
        segments=named,
    )
    for suffix in ("md", "srt", "txt", "json"):
        source = temp_dir / f"transcript.{suffix}"
        atomic_write(item_dir / f"transcript_named.{suffix}", source.read_text(encoding="utf-8"))
        source.unlink(missing_ok=True)
    temp_dir.rmdir()
    report = {
        "enriched_at": now_iso(),
        "model": MODEL,
        "segments": len(segments),
        "punctuated_segments": accepted,
        "speakers": names,
    }
    return {"row": row, "segments": segments, "report": report}


def write_index(db: sqlite3.Connection, result: dict) -> None:
    row = result["row"]
    item_dir = ITEMS_DIR / row["id"]
    with db:
        db.execute("DELETE FROM segments WHERE item_id = ?", (row["id"],))
        db.executemany(
            "INSERT INTO segments (item_id, idx, start, end, text, speaker) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (row["id"], i, float(s["start"]), float(s["end"]), s["text"], s.get("speaker"))
                for i, s in enumerate(result["segments"])
            ],
        )
        db.execute(
            "UPDATE items SET transcript_path = ?, updated_at = ? WHERE id = ?",
            (str(item_dir / "transcript_named.md"), now_iso(), row["id"]),
        )
    # Punctuated, named text before term correction; correct_terms.py always
    # starts from this file, and must run again after every enrichment.
    atomic_write(
        item_dir / "named_base.json",
        json.dumps(result["segments"], ensure_ascii=False) + "\n",
    )
    (item_dir / "terms.json").unlink(missing_ok=True)
    # enrich.json is the completion marker, written after the index is updated.
    atomic_write(
        item_dir / "enrich.json", json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all pending items")
    parser.add_argument("--ids", nargs="*", default=None, help="only these item ids")
    parser.add_argument("--force", action="store_true", help="redo items that have enrich.json")
    parser.add_argument("--workers", type=int, default=6, help="items in parallel")
    parser.add_argument("--chunk-workers", type=int, default=4, help="requests per item")
    args = parser.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY; run through `vibe vault run`", file=sys.stderr)
        return 1
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    rows = [
        row
        for row in db.execute("SELECT * FROM items WHERE status = 'done' ORDER BY published_at")
        if (args.ids is None or row["id"] in args.ids)
        and (args.force or not (ITEMS_DIR / row["id"] / "enrich.json").exists())
    ]
    if args.limit:
        rows = rows[: args.limit]
    print(f"items: {len(rows)}", flush=True)
    if not rows:
        return 0
    sanitize_proxy_env()  # httpx rejects the IPv6 CIDR entries in NO_PROXY
    notes, titles = load_show_notes(db)
    failed = 0
    # DeepSeek is reached directly; the office proxy is not needed for it.
    with httpx.Client(trust_env=False) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(enrich_item, row, notes, titles, client, key, args.chunk_workers): row
                for row in rows
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result = future.result()
                    write_index(db, result)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    failed += 1
                    print(
                        f"[{index}/{len(rows)}] FAIL {row['id']} {row['title'][:40]}: {exc}",
                        flush=True,
                    )
                    continue
                report = result["report"]
                names = ", ".join(sorted({v["name"] for v in report["speakers"].values()}))
                print(
                    f"[{index}/{len(rows)}] ok {row['title'][:40]} | punct "
                    f"{report['punctuated_segments']}/{report['segments']} | {names}",
                    flush=True,
                )
    print(f"done={len(rows) - failed} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
