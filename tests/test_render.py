from __future__ import annotations

from media_asr.render import render_markdown, render_srt, render_text
from media_asr.transcribe import TranscriptSegment

SEGMENTS = [
    TranscriptSegment(0.0, 2.5, "第一段。", speaker="SPEAKER_00"),
    TranscriptSegment(3.0, 6.25, "第二段。", speaker="SPEAKER_01"),
]


def test_srt() -> None:
    text = render_srt(SEGMENTS)
    assert "00:00:00,000 --> 00:00:02,500" in text
    assert "00:00:03,000 --> 00:00:06,250" in text
    assert "[SPEAKER_00] 第一段。" in text
    assert text.startswith("1\n")


def test_text() -> None:
    assert render_text(SEGMENTS) == "SPEAKER_00: 第一段。\nSPEAKER_01: 第二段。\n"


def test_markdown() -> None:
    text = render_markdown(
        title="测试",
        source="https://example.com",
        duration=6.25,
        language="zh",
        model="small/int8",
        transcribed_at="2026-10-19T00:00:00+00:00",
        segments=SEGMENTS,
    )
    assert text.startswith("---\n")
    assert 'title: "测试"' in text
    assert "**SPEAKER_00:** 第一段。" in text
    assert "## 文稿" in text