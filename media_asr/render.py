from __future__ import annotations

import json
from pathlib import Path

from .transcribe import TranscriptSegment
from .util import atomic_write, hms, srt_ts


def render_srt(segments: list[TranscriptSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        prefix = f"[{segment.speaker}] " if segment.speaker else ""
        blocks.append(
            f"{index}\n{srt_ts(segment.start)} --> {srt_ts(segment.end)}\n{prefix}{segment.text}\n"
        )
    return "\n".join(blocks)


def render_text(segments: list[TranscriptSegment]) -> str:
    lines = []
    for segment in segments:
        prefix = f"{segment.speaker}: " if segment.speaker else ""
        lines.append(prefix + segment.text)
    return "\n".join(lines) + ("\n" if lines else "")


def render_markdown(
    *,
    title: str,
    source: str,
    duration: float | None,
    language: str | None,
    model: str,
    transcribed_at: str,
    segments: list[TranscriptSegment],
) -> str:
    frontmatter = "\n".join(
        [
            "---",
            f"title: {json.dumps(title, ensure_ascii=False)}",
            f"source: {json.dumps(source, ensure_ascii=False)}",
            f"duration: {json.dumps(hms(duration))}",
            f"language: {json.dumps(language or 'unknown')}",
            f"model: {json.dumps(model)}",
            f"transcribed_at: {json.dumps(transcribed_at)}",
            "---",
            "",
            f"# {title}",
            "",
            f"> 来源：{source}",
            f"> 时长：{hms(duration)} · 语言：{language or 'auto'} · 模型：{model}",
            "",
            "## 文稿",
            "",
        ]
    )
    lines = []
    for index, segment in enumerate(segments):
        prefix = f"**{segment.speaker}:** " if segment.speaker else ""
        lines.append(f"**[{hms(segment.start)}]** {prefix}{segment.text}  ")
        if index and index % 12 == 0:
            lines.append("")
    return frontmatter + "\n".join(lines) + "\n"


def render_json(
    *,
    item_id: str,
    title: str,
    source: str,
    duration: float | None,
    language: str | None,
    model: str,
    segments: list[TranscriptSegment],
) -> str:
    payload = {
        "id": item_id,
        "title": title,
        "source": source,
        "duration": duration,
        "language": language,
        "model": model,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": segment.speaker,
            }
            for segment in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render_all(
    *,
    out_dir: Path,
    item_id: str,
    title: str,
    source: str,
    duration: float | None,
    language: str | None,
    model: str,
    transcribed_at: str,
    segments: list[TranscriptSegment],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out_dir / "transcript.md",
        "text": out_dir / "transcript.txt",
        "srt": out_dir / "transcript.srt",
        "json": out_dir / "transcript.json",
    }
    atomic_write(
        paths["markdown"],
        render_markdown(
            title=title,
            source=source,
            duration=duration,
            language=language,
            model=model,
            transcribed_at=transcribed_at,
            segments=segments,
        ),
    )
    atomic_write(paths["text"], render_text(segments))
    atomic_write(paths["srt"], render_srt(segments))
    atomic_write(
        paths["json"],
        render_json(
            item_id=item_id,
            title=title,
            source=source,
            duration=duration,
            language=language,
            model=model,
            segments=segments,
        ),
    )
    return paths