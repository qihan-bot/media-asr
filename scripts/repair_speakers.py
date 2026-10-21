#!/usr/bin/env python3
"""Second-pass repair for unresolved speaker labels.

The first pass names the obvious speakers. This pass sends the full transcript
back to DeepSeek for labels that are still ``SPEAKER_XX`` or ``null`` and asks
it to use host/guest context. If no name can be established, the label becomes
``Guest 1``, ``Guest 2``, ... so no raw labels remain.
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
PROMPT = """You are repairing speaker labels in an All-In Podcast transcript.

Known hosts: Jason Calacanis, Chamath Palihapitiya, David Sacks, David Friedberg.
A guest may also appear. Existing labels that are already named must not be
changed. Identify only the unresolved labels below using evidence from the full
transcript: introductions, forms of address, who asks/answers questions, and
topic ownership. If a label is a guest whose name is not stated, use
"Guest 1", "Guest 2", etc. Never invent a name that is not in the transcript.

Already mapped:
{mapped}

Unresolved labels to identify:
{unresolved}

Return only JSON:
{{"speakers": {{"SPEAKER_00": {{"name": "Name", "confidence": "high"}}, ...}}}}

Transcript:
{transcript}
"""


def full_transcript(segments: list[dict]) -> str:
    lines = []
    for segment in segments:
        start = float(segment.get("start", 0))
        lines.append(
            f"[{int(start // 60):02d}:{int(start % 60):02d}] "
            f"{segment.get('speaker')}: {segment.get('text', '')}"
        )
    return "\n".join(lines)


def call(client: httpx.Client, key: str, prompt: str) -> dict:
    payload = {
        "model": os.environ.get("MEDIA_ASR_SPEAKER_MODEL", "deepseek-flash"),
        "messages": [
            {"role": "system", "content": "Label podcast speakers. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 900,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    last = None
    for attempt in range(3):
        try:
            response = client.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=240,
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
                    name = re.search(r'"name"\s*:\s*(?:"([^"]*)"|null)', block)
                    conf = re.search(r'"confidence"\s*:\s*"([^"]*)"', block)
                    if name or conf:
                        speakers[label] = {
                            "name": name.group(1) if name else None,
                            "confidence": conf.group(1) if conf else "low",
                        }
                if speakers:
                    return {"speakers": speakers}
                raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


def render_named(run_dir: Path, data: dict, mapping: dict) -> None:
    segments = [
        TranscriptSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            text=str(s.get("text", "")),
            speaker=mapping.get(s.get("speaker"), {}).get("name", s.get("speaker")),
        )
        for s in data.get("segments", [])
        if str(s.get("text", "")).strip()
    ]
    temp = run_dir / "_named_tmp"
    render_all(
        out_dir=temp,
        item_id=data.get("id", run_dir.parent.parent.name),
        title=data.get("title") or "untitled",
        source=data.get("source") or "",
        duration=data.get("duration"),
        language=data.get("language"),
        model=(data.get("model") or "") + " + named speakers",
        transcribed_at=now_iso(),
        segments=segments,
    )
    for suffix in ("md", "srt", "txt", "json"):
        source = temp / f"transcript.{suffix}"
        target = run_dir / f"transcript_named.{suffix}"
        atomic_write(target, source.read_text(encoding="utf-8"))
        source.unlink(missing_ok=True)
    temp.rmdir()


def process(run_dir: Path, client: httpx.Client, key: str) -> str:
    map_path = run_dir / "speaker_map.json"
    data = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    if mapping.get("labels_fp") != labels_fingerprint(data.get("segments", [])):
        # Patching would stamp a map from an older diarization as current.
        raise RuntimeError("speaker map does not match current diarization; re-run map_speakers.py --force")
    unresolved = [
        label
        for label, item in mapping["speakers"].items()
        if not item.get("name") or item["name"] == label
    ]
    if not unresolved:
        return "skip"
    prompt = PROMPT.format(
        mapped=json.dumps(
            {label: item.get("name") for label, item in mapping["speakers"].items()},
            ensure_ascii=False,
        ),
        unresolved=", ".join(unresolved),
        transcript=full_transcript(data.get("segments", [])),
    )
    result = call(client, key, prompt).get("speakers", {})
    for label in unresolved:
        item = result.get(label, {})
        if isinstance(item, str):
            item = {"name": item, "confidence": "low"}
        name = (item.get("name") or "").strip()
        if name and name.upper() != label:
            mapping["speakers"][label] = {
                "name": name,
                "confidence": item.get("confidence", "low"),
                "evidence": "second pass",
            }
    # Give remaining unknowns stable guest labels instead of raw SPEAKER_XX.
    used = {item.get("name") for item in mapping["speakers"].values()}
    guest_index = 0
    for label in unresolved:
        item = mapping["speakers"][label]
        if not item.get("name") or item["name"] == label:
            guest_index += 1
            while f"Guest {guest_index}" in used:
                guest_index += 1
            item["name"] = f"Guest {guest_index}"
            used.add(item["name"])
            item["confidence"] = "low"
            item["evidence"] = "unnamed guest"
    mapping["repaired_at"] = now_iso()
    mapping["labels_fp"] = labels_fingerprint(data.get("segments", []))
    # Named transcripts first, map last: an interrupted run leaves the map
    # unresolved, so the next run retries instead of skipping.
    render_named(run_dir, data, mapping["speakers"])
    atomic_write(map_path, json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    return "done"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY; run through `vibe vault run`", flush=True)
        return 1

    runs = sorted(ITEMS_DIR.glob("*/runs/diarized/speaker_map.json"))
    if args.limit:
        runs = runs[: args.limit]
    print(f"maps: {len(runs)}", flush=True)
    results = []
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, path.parent, client, key): path
                for path in runs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                path = futures[future]
                try:
                    status = future.result()
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    print(f"[{index}/{len(runs)}] FAIL {path.parent.parent.parent.name}: {exc}", flush=True)
                else:
                    print(f"[{index}/{len(runs)}] {status} {path.parent.parent.parent.name}", flush=True)
                results.append(status)
    done = sum(1 for item in results if item == "done")
    skipped = sum(1 for item in results if item == "skip")
    failed = len(results) - done - skipped
    print(f"repaired={done} skipped={skipped} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())