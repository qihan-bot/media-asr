#!/usr/bin/env python3
"""Map pyannote speaker labels to names using the DeepSeek chat API.

For every ``runs/diarized/transcript.json`` under ``data/items`` this script
asks DeepSeek to identify each ``SPEAKER_XX`` from the episode context, then
writes ``speaker_map.json`` and ``transcript_named.*`` next to the diarized
transcript. The original transcript is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from media_asr.config import ITEMS_DIR
from media_asr.render import render_all
from media_asr.transcribe import TranscriptSegment
from media_asr.util import atomic_write, now_iso, sanitize_proxy_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_common import labels_fingerprint  # noqa: E402

sanitize_proxy_env()

import httpx  # noqa: E402

API_URL = "https://api.deepseek.com/chat/completions"
HOSTS = [
    "Jason Calacanis",
    "Chamath Palihapitiya",
    "David Sacks",
    "David Friedberg",
]
PROMPT = """You are labeling speakers in an All-In Podcast episode transcript.

Known recurring hosts: Jason Calacanis, Chamath Palihapitiya, David Sacks,
David Friedberg. A guest may also appear. Use only evidence from the supplied
transcript. For each SPEAKER_XX label, return the person's canonical name and
confidence (high/medium/low). If there is not enough evidence, use null.
If diarization split one person across two labels, map both labels to that
person. Never invent a name that is not mentioned or clearly implied.

Episode title: {title}

Return only a JSON object with this shape (no markdown, no explanation):
{{"speakers": {{"SPEAKER_00": {{"name": "Jason Calacanis", "confidence": "high"}}, ...}}}}

Transcript excerpt:
{transcript}
"""


def compact_transcript(segments: list[dict], limit: int = 22000) -> str:
    lines = []
    total = 0
    for segment in segments:
        start = float(segment.get("start", 0))
        speaker = segment.get("speaker") or "UNKNOWN"
        text = str(segment.get("text", "")).strip()
        line = f"[{int(start // 60):02d}:{int(start % 60):02d}] {speaker}: {text}"
        if total + len(line) > limit:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def normalize_name(name: str | None) -> str | None:
    if not name:
        return None
    value = name.strip()
    aliases = {
        "jason": "Jason Calacanis",
        "jason calacanis": "Jason Calacanis",
        "chamath": "Chamath Palihapitiya",
        "chamath palihapitiya": "Chamath Palihapitiya",
        "sacks": "David Sacks",
        "david sacks": "David Sacks",
        "friedberg": "David Friedberg",
        "david friedberg": "David Friedberg",
    }
    return aliases.get(value.lower(), value)


def request_mapping(client: httpx.Client, key: str, title: str, transcript: str) -> dict:
    payload = {
        "model": os.environ.get("MEDIA_ASR_SPEAKER_MODEL", "deepseek-flash"),
        "messages": [
            {
                "role": "system",
                "content": "You label podcast speakers. Output valid JSON only.",
            },
            {
                "role": "user",
                "content": PROMPT.format(title=title, transcript=transcript),
            },
        ],
        "temperature": 0,
        "max_tokens": 1500,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    last_error = None
    for attempt in range(3):
        try:
            response = client.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                speakers = {}
                for label, block in re.findall(
                    r'"(SPEAKER_\d+)"\s*:\s*\{(.*?)\}', content, re.S
                ):
                    name_match = re.search(
                        r'"name"\s*:\s*(?:"([^"]*)"|null)', block
                    )
                    conf_match = re.search(
                        r'"confidence"\s*:\s*"([^"]*)"', block
                    )
                    if name_match or conf_match:
                        speakers[label] = {
                            "name": name_match.group(1) if name_match else None,
                            "confidence": conf_match.group(1) if conf_match else "low",
                        }
                if speakers:
                    return {"speakers": speakers}
                raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek mapping failed: {last_error}")


def process_run(run_dir: Path, client: httpx.Client, key: str, force: bool) -> str:
    transcript_path = run_dir / "transcript.json"
    map_path = run_dir / "speaker_map.json"
    if map_path.exists() and not force:
        return "skip"
    if map_path.exists():
        # Move the old completion marker aside so a failed refresh is retried.
        map_path.replace(run_dir / "speaker_map.prev.json")
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    labels = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    if not labels:
        return "no-labels"
    mapping = request_mapping(
        client,
        key,
        data.get("title") or run_dir.parent.parent.name,
        compact_transcript(segments),
    )
    speakers = mapping.get("speakers", mapping)
    cleaned = {}
    for label in labels:
        item = speakers.get(label, {})
        if isinstance(item, str):
            item = {"name": item, "confidence": "low", "evidence": ""}
        name = normalize_name(item.get("name"))
        cleaned[label] = {
            "name": name or label,
            "confidence": item.get("confidence", "low"),
            "evidence": item.get("evidence", ""),
        }
    payload = {
        "title": data.get("title"),
        "source": data.get("source"),
        "mapped_at": now_iso(),
        "labels_fp": labels_fingerprint(segments),
        "speakers": cleaned,
    }

    named_segments = [
        TranscriptSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            text=str(s.get("text", "")),
            speaker=cleaned.get(s.get("speaker"), {}).get("name", s.get("speaker")),
        )
        for s in segments
        if str(s.get("text", "")).strip()
    ]
    temp_dir = run_dir / "_named_tmp"
    render_all(
        out_dir=temp_dir,
        item_id=data.get("id", run_dir.parent.parent.name),
        title=data.get("title") or "untitled",
        source=data.get("source") or "",
        duration=data.get("duration"),
        language=data.get("language"),
        model=(data.get("model") or "") + " + named speakers",
        transcribed_at=now_iso(),
        segments=named_segments,
    )
    for suffix in ("md", "srt", "txt", "json"):
        source = temp_dir / f"transcript.{suffix}"
        target = run_dir / f"transcript_named.{suffix}"
        atomic_write(target, source.read_text(encoding="utf-8"))
        source.unlink(missing_ok=True)
    temp_dir.rmdir()
    # The map is the completion marker (runs with a map are skipped), so it is
    # written only after every named transcript is in place.
    atomic_write(map_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return "done"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY; run through `vibe vault run`", flush=True)
        return 1

    runs = sorted(ITEMS_DIR.glob("*/runs/diarized/transcript.json"))
    if args.limit:
        runs = runs[: args.limit]
    print(f"runs: {len(runs)}", flush=True)
    results: list[str] = []
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_run, run.parent, client, key, args.force): run
                for run in runs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                run = futures[future]
                try:
                    status = future.result()
                except Exception as exc:  # noqa: BLE001
                    status = f"failed: {exc}"
                    print(f"[{index}/{len(runs)}] FAIL {run.parent.parent.parent.name}: {exc}", flush=True)
                else:
                    print(f"[{index}/{len(runs)}] {status} {run.parent.parent.parent.name}", flush=True)
                results.append(status)
    done = sum(1 for item in results if item == "done")
    skipped = sum(1 for item in results if item == "skip")
    failed = len(results) - done - skipped
    print(f"done={done} skipped={skipped} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())