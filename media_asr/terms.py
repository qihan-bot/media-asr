"""Correct misrecognized names and terms in enriched podcast transcripts.

``media-asr correct-terms`` runs after ``enrich`` and keeps its guarantee
auditable: the model never rewrites text. It only proposes ``wrong -> right``
pairs; this step filters them and applies the survivors as literal
replacements, recording every pair and its count in ``terms.json``.

A pair is applied only if ``wrong`` occurs verbatim in the transcript, sounds
like a mishearing of ``right``, and has a basis:

* sound: a Chinese pair keeps its character (syllable) count; an English pair
  keeps a spelling similarity of at least 0.5 (0.65 for tokens of four
  characters or fewer); and ``wrong`` is not a common
  word (wordfreq Zipf >= 3.5 zh / 3.0 en, so "福利" or "skew" is never
  replaced), unless ``right`` is a speaker's name of the same length (a
  misheard self-introduction such as "红军" -> "泓君"). Mixed Chinese/English
  pairs need high confidence.
* tier ``reference``: ``right`` appears in the episode's show notes, the show's
  episode titles, or the speaker names (high or medium confidence); or
* tier ``model``: a widely known name or term the model rates high confidence.

Pairs that map one ``wrong`` to two different forms, or whose ``wrong`` is
under two characters, are dropped. ASCII ``wrong`` forms match whole words only.
Needs the ``library`` extras (wordfreq, jieba).

Input is ``named_base.json`` (punctuated, named, uncorrected; written by
``enrich``); output is ``transcript_named.*`` plus the search index.
``terms.json`` is the completion marker, and ``enrich`` removes it, so every
re-enrichment is corrected again.
"""

from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

from wordfreq import zipf_frequency

from .config import ITEMS_DIR
from .library import episode_notes, load_show_notes
from .llm import LLM, TruncatedResponse
from .render import render_all
from .transcribe import TranscriptSegment
from .util import atomic_write, now_iso

CHUNK_CHARS = {"zh": 6000, "en": 12000}
# 3000 cut off the answer for chunks with many fixes; a chunk whose answer is
# still cut off is split in two and retried.
MAX_TOKENS = 8000
CJK = re.compile(r"[一-鿿]")

PROMPT = """You check a podcast transcript produced by speech recognition for misrecognized
proper nouns and technical terms: people, companies, products, models, papers,
places and jargon (e.g. "Skading Law" for "Scaling Law", "Calcode" for
"Claude Code", "强化学系" for "强化学习").

Reference material (spellings here are authoritative):
Show: {show}
Episode title: {title}
Speakers: {speakers}
Show notes: {notes}
Other episode titles of this show: {titles}

Transcript chunk:
{chunk}

List only clear misrecognitions. For each, give "wrong" exactly as it appears
in the chunk (copy the characters verbatim, keep it as short as possible but
unambiguous), "right" (the correct form), "basis" ("reference" if the correct
form is in the reference material, otherwise "well_known"), and "confidence"
(high, medium, low). Do not fix grammar, filler words, style, punctuation, or
ordinary words; do not translate; skip anything you are not sure about. If the
"wrong" form is itself a real name or product that could be what the speaker
meant (e.g. OpenCode vs OpenClaw), skip it.

Return only JSON: {{"fixes": [{{"wrong": "...", "right": "...", "basis": "reference", "confidence": "high"}}]}}
"""


def chunks_of(texts: list[str], budget: int) -> list[list[str]]:
    out, current, size = [], [], 0
    for text in texts:
        if current and size + len(text) > budget:
            out.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    return out + ([current] if current else [])


def pattern(wrong: str) -> re.Pattern:
    if wrong.isascii():
        return re.compile(rf"(?<![A-Za-z0-9]){re.escape(wrong)}(?![A-Za-z0-9])")
    return re.compile(re.escape(wrong))


def script_of(text: str) -> str:
    if CJK.search(text):
        return "mixed" if re.search(r"[A-Za-z]", text) else "zh"
    return "en"


def sound_problem(wrong: str, right: str, speakers: set[str], confidence) -> str | None:
    """Reason a pair does not look like a mishearing, or None."""
    kinds = {script_of(wrong), script_of(right)}
    if kinds == {"zh"}:
        if len(wrong) != len(right):
            return "Chinese pair changes the syllable count"
        if zipf_frequency(wrong, "zh") >= 3.5 and right not in speakers:
            return "wrong form is a common word"
    elif kinds == {"en"}:
        a, b = (re.sub(r"[^a-z0-9]", "", x.lower()) for x in (wrong, right))
        # Short tokens ("OE", "AME") match almost anything at 0.5.
        if SequenceMatcher(None, a, b).ratio() < (0.65 if min(len(a), len(b)) <= 4 else 0.5):
            return "spelling too different"
        if zipf_frequency(wrong, "en") >= 3.0 and not (right in speakers and len(a) == len(b)):
            return "wrong form is a common word"
    elif confidence != "high":
        return "mixed-script pair without high confidence"
    return None


def select(
    fixes: list[dict], text: str, reference: str, speakers: set[str]
) -> tuple[list[dict], list[dict]]:
    """Return (accepted, rejected) pairs with their tier or rejection reason."""
    ref = reference.casefold()
    by_wrong: dict[str, set[str]] = {}
    for fix in fixes:
        by_wrong.setdefault(str(fix.get("wrong", "")).strip(), set()).add(
            str(fix.get("right", "")).strip()
        )
    accepted, rejected, seen = [], [], set()
    for fix in fixes:
        wrong = str(fix.get("wrong", "")).strip()
        right = str(fix.get("right", "")).strip()
        if (wrong, right) in seen:
            continue
        seen.add((wrong, right))
        confidence = fix.get("confidence")
        entry = {
            "wrong": wrong,
            "right": right,
            "basis": fix.get("basis"),
            "confidence": confidence,
        }
        count = len(pattern(wrong).findall(text)) if len(wrong) >= 2 else 0
        if len(wrong) < 2 or not right or wrong == right:
            reason = "empty or too short"
        elif len(by_wrong[wrong]) > 1:
            reason = "ambiguous: several corrections"
        elif count == 0:
            reason = "wrong form not in transcript"
        elif problem := sound_problem(wrong, right, speakers, confidence):
            reason = problem
        elif right.casefold() in ref and confidence in ("high", "medium"):
            reason, entry["tier"] = None, "reference"
        elif confidence == "high" and right.casefold() not in ref:
            reason, entry["tier"] = None, "model"
        else:
            reason = "not high confidence"
        if reason:
            entry["rejected"] = reason
            rejected.append(entry)
        else:
            entry["count"] = count
            accepted.append(entry)
    return accepted, rejected


def find_fixes(llm: LLM, lines: list[str], prompt_for, label: str = "") -> list[dict]:
    """Ask for the fixes in one chunk of lines. An answer cut off at the token
    limit (too many fixes) splits the chunk in two and asks for each half."""
    try:
        result = llm.chat(
            "You find speech-recognition errors. Output valid JSON only.",
            prompt_for("\n".join(lines)),
            MAX_TOKENS,
        )
    except TruncatedResponse:
        if len(lines) < 2:
            raise
        print(f"{label}: answer cut off, splitting a chunk of {len(lines)} lines", flush=True)
        half = len(lines) // 2
        return find_fixes(llm, lines[:half], prompt_for, label) + find_fixes(
            llm, lines[half:], prompt_for, label
        )
    return [f for f in result.get("fixes", []) if isinstance(f, dict)]


def correct_item(row, notes, titles_by_feed, show_titles, llm: LLM, items_dir: Path) -> dict:
    item_dir = items_dir / row["id"]
    named = json.loads((item_dir / "transcript_named.json").read_text(encoding="utf-8"))
    segments = json.loads((item_dir / "named_base.json").read_text(encoding="utf-8"))
    lang = "zh" if (row["language"] or "").startswith("zh") else "en"
    speakers = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    note_text = episode_notes(notes, row)
    other_titles = " | ".join(t for t in show_titles.get(row["feed_url"], []) if t != row["title"])[
        :4000
    ]
    reference = "\n".join([row["title"], note_text, other_titles, " ".join(speakers)])

    def prompt_for(chunk: str) -> str:
        return PROMPT.format(
            show=titles_by_feed.get(row["feed_url"], ""),
            title=row["title"],
            speakers=", ".join(speakers) or "(unknown)",
            notes=note_text,
            titles=other_titles or "(none)",
            chunk=chunk,
        )

    fixes: list[dict] = []
    for lines in chunks_of([s["text"] for s in segments], CHUNK_CHARS[lang]):
        fixes.extend(find_fixes(llm, lines, prompt_for, row["id"]))

    full_text = "\n".join(s["text"] for s in segments)
    accepted, rejected = select(fixes, full_text, reference, set(speakers))
    # Longer wrong forms first, so "Open Cloud Moment" wins over "Open Cloud".
    for fix in sorted(accepted, key=lambda f: -len(f["wrong"])):
        rx = pattern(fix["wrong"])
        for seg in segments:
            seg["text"] = rx.sub(fix["right"], seg["text"])

    temp_dir = item_dir / "_terms_tmp"
    render_all(
        out_dir=temp_dir,
        item_id=row["id"],
        title=named.get("title") or row["title"],
        source=named.get("source") or row["source"],
        duration=named.get("duration") or row["duration"],
        language=named.get("language") or row["language"],
        model=(named.get("model") or "").split(" + term")[0] + f" + term correction ({llm.model})",
        transcribed_at=now_iso(),
        segments=[
            TranscriptSegment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s["text"],
                speaker=s.get("speaker"),
            )
            for s in segments
        ],
    )
    for suffix in ("md", "srt", "txt", "json"):
        source = temp_dir / f"transcript.{suffix}"
        atomic_write(item_dir / f"transcript_named.{suffix}", source.read_text(encoding="utf-8"))
        source.unlink(missing_ok=True)
    temp_dir.rmdir()
    report = {
        "corrected_at": now_iso(),
        "model": llm.model,
        "applied": accepted,
        "rejected": rejected,
        "replacements": sum(f["count"] for f in accepted),
    }
    return {"row": row, "segments": segments, "report": report}


def write_index(db: sqlite3.Connection, result: dict, items_dir: Path) -> None:
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
    atomic_write(
        items_dir / row["id"] / "terms.json",
        json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n",
    )


def run(
    db: sqlite3.Connection,
    llm: LLM,
    *,
    ids: list[str] | None = None,
    force: bool = False,
    workers: int = 6,
    items_dir: Path = ITEMS_DIR,
) -> int:
    """Correct enriched items without ``terms.json`` (or ``ids``); return 1 if any failed."""
    every = db.execute("SELECT * FROM items WHERE status = 'done' ORDER BY published_at").fetchall()
    show_titles: dict[str, list[str]] = {}
    for row in every:
        show_titles.setdefault(row["feed_url"], []).append(row["title"])
    rows = [
        row
        for row in every
        if (ids is None or row["id"] in ids)
        and (items_dir / row["id"] / "enrich.json").exists()
        and (force or not (items_dir / row["id"] / "terms.json").exists())
    ]
    print(f"items: {len(rows)}", flush=True)
    if not rows:
        return 0
    notes, titles_by_feed = load_show_notes(db)
    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {
            pool.submit(correct_item, row, notes, titles_by_feed, show_titles, llm, items_dir): row
            for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
                write_index(db, result, items_dir)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                print(
                    f"[{index}/{len(rows)}] FAIL {row['id']} {row['title'][:40]}: {exc}", flush=True
                )
                continue
            report = result["report"]
            pairs = "; ".join(f"{f['wrong']}->{f['right']}" for f in report["applied"][:6])
            print(
                f"[{index}/{len(rows)}] ok {row['title'][:36]} | {len(report['applied'])} pairs, "
                f"{report['replacements']} replacements, {len(report['rejected'])} rejected | {pairs}",
                flush=True,
            )
    print(f"done={len(rows) - failed} failed={failed}", flush=True)
    return 1 if failed else 0
