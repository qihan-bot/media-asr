"""Find episodes where diarization merged several people into one label and redo them.

pyannote sometimes puts host and guest into a single label (e.g. one label with
98% of the speaking time that both asks and answers the questions). For every
done item without ``rediarize.json`` whose largest label holds more than
``share`` of the speaking time, DeepSeek reads samples of that label and
decides whether it mixes several people and how many people talk in the
episode. Merged items whose largest label holds over 90% of the time, or with
3+ people, are re-run on the remote GPU with that count plus one
(an extra cluster absorbs intro voices and clips; with the bare count pyannote
can still merge host and guest, e.g. 硅谷101 E247),
their ``enrich.json`` is removed so ``enrich`` names them again,
and ``rediarize.json`` records the decision so no item is checked twice.

Check, re-run and marker happen under one hold of the item's lock: the verdict
is acted on only if the transcript the model judged is still the episode's
transcript and no other run has decided meanwhile; the re-run happens in this
process, and the marker records the transcript that re-run wrote. No other run
can write the episode until the marker is down.

Run it as ``media-asr-remote rediarize`` (``library-update`` does) so the
re-runs reach the GPU host of ``MEDIA_ASR_REMOTE_ENV``; it needs
``DEEPSEEK_API_KEY`` and ``HF_TOKEN``.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from .config import ITEMS_DIR
from .library import file_digest, read_json
from .llm import LLM
from .util import atomic_write, item_lock, now_iso

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


def parse_verdict(verdict) -> tuple[bool, int]:
    """(merged, people) from the model's answer; ValueError if it is malformed:
    merged must be true or false, people a whole number of at least 1."""
    if not isinstance(verdict, dict) or not isinstance(verdict.get("merged"), bool):
        raise ValueError(f"malformed verdict: {json.dumps(verdict, ensure_ascii=False)[:200]}")
    people = verdict.get("people")
    if isinstance(people, bool) or not isinstance(people, int) or people < 1:
        raise ValueError(f"people is not a count of at least 1: {people!r}")
    return verdict["merged"], people


def check(row, llm: LLM, *, share: float, dry_run: bool, items_dir: Path, rerun) -> bool:
    """Check one episode and redo it if needed; False if the check or re-run failed."""
    item_dir = items_dir / row["id"]
    marker = item_dir / "rediarize.json"
    transcript = item_dir / "transcript.json"
    if marker.exists() or not transcript.exists():
        return True
    data, digest = read_json(transcript)
    segments = [s for s in data["segments"] if str(s.get("text", "")).strip()]
    shares = label_shares(segments)
    if not segments or shares.most_common(1)[0][1] <= share:
        return True
    label, prompt = build_prompt(row["title"], segments, shares)
    try:
        verdict = llm.chat(
            "You audit speaker diarization. Output valid JSON only.",
            prompt,
            400,
        )
        merged, people = parse_verdict(verdict)
    except Exception as exc:  # noqa: BLE001 - leave unmarked, retried next run
        print(f"FAIL check {row['id']}: {exc}", flush=True)
        return False
    before = round(shares[label], 3)
    print(
        f"{'MERGED' if merged else 'ok    '} {before:.0%} people={people} {row['title'][:50]} "
        f"| {str(verdict.get('reason', ''))[:100]}",
        flush=True,
    )
    if dry_run:
        return True
    record = {"checked_at": now_iso(), "label": label, "share_before": before, **verdict}
    # Library run 2026-10-23: re-diarizing every flagged episode moved the largest
    # label by <3 points in 81 of 86 cases (a guest holding ~80% of a long
    # interview is normal); it only helped hard merges and 3+ people.
    rediarize = merged and people >= 2 and (before > 0.9 or people >= 3)
    with item_lock(item_dir):
        # One hold of the lock from the check to the marker: the verdict is acted
        # on only while its transcript is current, and no other run can write the
        # episode before the marker records what this run did.
        if marker.exists():
            return True  # another run decided meanwhile
        if file_digest(transcript) != digest:
            print(f"FAIL check {row['id']}: transcript.json changed meanwhile", flush=True)
            return False
        if rediarize:
            if rerun is None:
                raise RuntimeError("no re-run configured; use media-asr-remote rediarize")
            lang = "zh" if (row["language"] or "").startswith("zh") else "en"
            try:
                produced = rerun(row["id"], lang, people + 1)
            except Exception as exc:  # noqa: BLE001 - left unmarked, retried next run
                print(f"FAIL rerun {row['id']}: {exc}", flush=True)
                return False
            new, current = read_json(transcript)
            if current != produced:
                raise RuntimeError("transcript.json is not the output of the re-run")
            after = label_shares(new["segments"]).most_common(1)[0][1]
            record["share_after"] = round(after, 3)
            (item_dir / "enrich.json").unlink(missing_ok=True)
        atomic_write(marker, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if rediarize:
        print(f"       rerun: largest label {before:.0%} -> {after:.0%}", flush=True)
    return True


def run(
    db: sqlite3.Connection,
    llm: LLM,
    *,
    share: float = 0.75,
    dry_run: bool = False,
    shard: str = "0/1",
    items_dir: Path = ITEMS_DIR,
    rerun=None,
) -> int:
    """Check and redo merged episodes; return 1 if any episode failed.

    ``rerun(item_id, language, num_speakers)`` transcribes an episode again
    while this step holds its lock and returns the SHA-256 of the
    transcript.json it wrote (``media-asr rediarize`` passes the remote GPU
    run). A failure (bad transcript, malformed verdict, failed re-run) affects
    only its episode, which stays unmarked and is checked again on the next run.
    """
    index, shards = (int(x) for x in shard.split("/"))
    rows = db.execute("SELECT * FROM items WHERE status = 'done' ORDER BY published_at").fetchall()
    failed = 0
    for row in rows[index::shards]:
        try:
            ok = check(row, llm, share=share, dry_run=dry_run, items_dir=items_dir, rerun=rerun)
        except Exception as exc:  # noqa: BLE001 - this episode only; retried next run
            ok = False
            print(f"FAIL {row['id']}: {exc}", flush=True)
        failed += not ok
    return 1 if failed else 0
