from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DEFAULT_COMPUTE, DEFAULT_MODEL, DEFAULT_THREADS, MODEL_DIR
from .util import sanitize_proxy_env


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptResult:
    segments: list[TranscriptSegment]
    language: str | None
    language_probability: float | None
    duration: float | None
    model_label: str


class Transcriber:
    """Thin wrapper around faster-whisper with CPU-friendly defaults."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        compute_type: str = DEFAULT_COMPUTE,
        threads: int = DEFAULT_THREADS,
        model_dir: Path = MODEL_DIR,
        offline: bool = False,
    ):
        sanitize_proxy_env()
        from faster_whisper import WhisperModel

        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.model_label = f"faster-whisper {model_size}/{compute_type}"
        cached = Path(model_size).expanduser().exists() or any(
            model_dir.glob(f"models--*faster-whisper-{model_size}")
        )
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=threads,
            download_root=str(model_dir),
            local_files_only=offline or cached,
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
        beam_size: int = 5,
        simplify_zh: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        raw_segments, info = self.model.transcribe(
            str(audio_path),
            language=language or None,
            beam_size=beam_size,
            best_of=beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
            word_timestamps=False,
        )

        simplify_zh = simplify_zh and str(getattr(info, "language", "")).startswith("zh")
        segments: list[TranscriptSegment] = []
        for segment in raw_segments:
            text = _clean_text(segment.text)
            if simplify_zh:
                text = _to_simplified(text)
            if not text:
                continue
            segments.append(
                TranscriptSegment(start=float(segment.start), end=float(segment.end), text=text)
            )
            if progress:
                progress(f"[{_clock(segment.start)}] {text}")

        return TranscriptResult(
            segments=segments,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=getattr(info, "duration", None),
            model_label=self.model_label,
        )


def _to_simplified(text: str) -> str:
    try:
        import zhconv
    except ImportError:
        return text
    return zhconv.convert(text, "zh-cn")


def _clean_text(text: str) -> str:
    """Remove spaces that Whisper inserts between Chinese characters."""
    text = text.strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、])", r"\1", text)
    return text.strip()


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"