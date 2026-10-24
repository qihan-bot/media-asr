"""Punctuate and name speakers in a podcast library (``media-asr enrich``).

For every done item that has no ``enrich.json`` yet:

1. Punctuation: DeepSeek restores punctuation (and English casing) segment by
   segment. A segment is accepted only if its letters and digits are unchanged
   (NFKC, case-folded, punctuation and whitespace removed); otherwise the
   original text is kept, so this pass can never change what was said.
2. Speakers: DeepSeek maps each ``SPEAKER_XX`` to a person using the show name,
   the recurring hosts from ``library.toml``, the episode title, the RSS show
   notes and the transcript. Only high/medium confidence names are used; the
   rest become ``主持人N`` / ``Host N``, ``嘉宾N`` / ``Guest N`` or
   ``说话人N`` / ``Speaker N``. In an episode with ``voiceprint.json`` (shows
   with reference voices), labels whose voice matched a reference keep that
   name (``"method": "voiceprint"``; role host if the show's hosts list the
   name, else guest) and the model is asked only about the other labels; a
   model name that is one of the reference voices but not what the label
   sounds like is rejected.

Writes ``transcript_named.{md,srt,txt,json}``, ``named_base.json`` (the input
of ``correct-terms``) and ``enrich.json`` (the completion marker) next to the
original transcript, which is left untouched, removes ``terms.json`` so the
item is corrected again, and points the SQLite index at the named transcript.
The writes happen under the item's lock, after checking that the transcript
did not change meanwhile, and start by removing both markers, so a failure
halfway leaves the item to be redone rather than marked done.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import ITEMS_DIR
from .library import (
    Show,
    episode_notes,
    file_digest,
    host_hints,
    load_show_notes,
    read_json,
    write_named,
)
from .llm import LLM
from .util import atomic_write, item_lock, now_iso
from .voiceprint import voices

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

# Appended to SPEAKER_PROMPT when voiceprints named some labels.
VOICE_NOTE = """
Already identified by voice; keep these and do not label them again:
{known}
Label only: {ask}
"""


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()


def punctuate(llm: LLM, texts: list[str], lang: str, workers: int) -> tuple[list[str], int]:
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
            result = llm.chat(PUNCT_PROMPT[lang], "\n".join(lines), tokens).get("lines", [])
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


def enrich_item(
    row: sqlite3.Row,
    notes: dict,
    titles: dict,
    shows: list[Show],
    llm: LLM,
    workers: int,
    items_dir: Path,
) -> dict:
    data, digest = read_json(items_dir / row["id"] / "transcript.json")
    segments = [s for s in data.get("segments", []) if str(s.get("text", "")).strip()]
    lang = "zh" if (row["language"] or data.get("language") or "").startswith("zh") else "en"
    texts, accepted = punctuate(llm, [s["text"] for s in segments], lang, workers)
    for seg, text in zip(segments, texts):
        seg["text"] = text

    labels = sorted({s["speaker"] for s in segments if s.get("speaker")})
    voice = voices(items_dir / row["id"], digest) or {}
    sounds = voice.get("speakers", {})
    heard = {
        label: sounds[label] for label in labels if sounds.get(label, {}).get("voice") == "match"
    }
    roster = set(voice.get("references", []))
    hosts = host_hints(shows, row["feed_url"])
    names: dict[str, dict] = {}
    if labels:
        ask = [label for label in labels if label not in heard]
        raw: dict = {}
        if ask:
            stats, excerpt = speaker_context(segments)
            prompt = SPEAKER_PROMPT.format(
                show=titles.get(row["feed_url"], row["feed_url"] or ""),
                hosts=hosts or "(not given)",
                title=row["title"],
                notes=episode_notes(notes, row),
                stats=stats,
                transcript=excerpt,
            )
            if heard:
                known = "\n".join(f"{label}: {heard[label]['name']}" for label in sorted(heard))
                prompt += VOICE_NOTE.format(known=known, ask=", ".join(ask))
            raw = llm.chat("You label podcast speakers. Output valid JSON only.", prompt, 2000)
            raw = raw.get("speakers", raw)
        unknown: dict[str, int] = {}
        for label in labels:
            if label in heard:
                match = heard[label]
                names[label] = {
                    "name": match["name"],
                    "role": "host" if match["name"] in hosts else "guest",
                    "confidence": match["confidence"],
                    "evidence": f"voiceprint {match['score']:.2f} (margin {match['margin']:.2f})",
                    "method": "voiceprint",
                }
                continue
            info = raw.get(label) or {}
            name = (info.get("name") or "").strip() if isinstance(info, dict) else ""
            confidence = info.get("confidence", "low") if isinstance(info, dict) else "low"
            role = info.get("role") if isinstance(info, dict) else None
            evidence = info.get("evidence", "") if isinstance(info, dict) else ""
            sound = sounds.get(label, {})
            if name in roster and sound.get("voice") == "no-match" and sound.get("best") != name:
                evidence = (
                    f"voice does not match {name} (closest {sound['best']} {sound['score']:.2f}); "
                    "model name rejected"
                )
                name, confidence, role = "", "low", "other"
            if not name or confidence not in ("high", "medium"):
                kind = role if role in ("host", "guest") else "other"
                unknown[kind] = unknown.get(kind, 0) + 1
                sep = "" if lang == "zh" else " "
                name = f"{UNNAMED[lang][kind]}{sep}{unknown[kind]}"
            names[label] = {
                "name": name,
                "role": role,
                "confidence": confidence,
                "evidence": evidence,
            }
        for seg in segments:
            if seg.get("speaker") in names:
                seg["speaker"] = names[seg["speaker"]]["name"]

    report = {
        "enriched_at": now_iso(),
        "model": llm.model,
        "segments": len(segments),
        "punctuated_segments": accepted,
        "speakers": names,
    }
    render = {
        "title": row["title"],
        "source": data.get("source") or row["source"],
        "duration": data.get("duration") or row["duration"],
        "language": data.get("language") or row["language"],
        "model": (data.get("model") or "") + f" + {llm.model} punctuation/speaker names",
    }
    return {"row": row, "segments": segments, "report": report, "render": render, "input": digest}


def commit(db: sqlite3.Connection, result: dict, items_dir: Path, *, force: bool = False) -> bool:
    """Write the item's named transcript, index, named_base.json and enrich.json.

    Holds the item's lock throughout. Returns False without writing if another
    run enriched the item meanwhile (unless ``force``); raises if the
    transcript changed after it was read.
    """
    row = result["row"]
    item_dir = items_dir / row["id"]
    with item_lock(item_dir):
        if not force and (item_dir / "enrich.json").exists():
            return False
        if file_digest(item_dir / "transcript.json") != result["input"]:
            raise RuntimeError("transcript.json changed while enriching; the next run redoes it")
        for marker in ("enrich.json", "terms.json"):
            (item_dir / marker).unlink(missing_ok=True)
        write_named(item_dir, item_id=row["id"], segments=result["segments"], **result["render"])
        _write_index(db, result, item_dir)
    return True


def _write_index(db: sqlite3.Connection, result: dict, item_dir: Path) -> None:
    row = result["row"]
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
    # Punctuated, named text before term correction; correct-terms always
    # starts from this file, and must run again after every enrichment.
    atomic_write(
        item_dir / "named_base.json",
        json.dumps(result["segments"], ensure_ascii=False) + "\n",
    )
    # enrich.json is the completion marker, written after the index is updated.
    atomic_write(
        item_dir / "enrich.json", json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n"
    )


def run(
    db: sqlite3.Connection,
    llm: LLM,
    shows: list[Show],
    *,
    ids: list[str] | None = None,
    force: bool = False,
    limit: int = 0,
    workers: int = 6,
    chunk_workers: int = 4,
    items_dir: Path = ITEMS_DIR,
) -> int:
    """Enrich pending items (or ``ids``); return 1 if any item failed."""
    rows = [
        row
        for row in db.execute("SELECT * FROM items WHERE status = 'done' ORDER BY published_at")
        if (ids is None or row["id"] in ids)
        and (force or not (items_dir / row["id"] / "enrich.json").exists())
    ]
    if limit:
        rows = rows[:limit]
    print(f"items: {len(rows)}", flush=True)
    if not rows:
        return 0
    notes, titles = load_show_notes(db)
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(enrich_item, row, notes, titles, shows, llm, chunk_workers, items_dir): row
            for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
                if not commit(db, result, items_dir, force=force):
                    print(
                        f"[{index}/{len(rows)}] skip {row['id']}: enriched by another run",
                        flush=True,
                    )
                    continue
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
