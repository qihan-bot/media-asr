#!/usr/bin/env python3
"""Transcribe a local audio file with OpenAI Whisper on CUDA or CPU.

This is the GPU companion to the main faster-whisper pipeline.  It is useful on
machines where CTranslate2 does not yet support the GPU (for example Blackwell
/sm_120), because PyTorch's CUDA wheels do.

Example on the Windows RTX 5090 host:

    .venv\\Scripts\\python.exe scripts\\transcribe_openai_whisper.py ^
        --audio audio.wav --model large-v3-turbo --output turbo_gpu.json ^
        --language zh --prompt "厚雪长波，王浩岚，中期选举，美联储，通胀"

The output JSON can be rendered with ``media_asr.render.render_all`` or compared
with ``scripts/compare_runs.py`` after converting it to a transcript run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

HALLUCINATION_PATTERNS = (
    r"字幕志愿者",
    r"字幕由[^。]{0,40}提供",
    r"校对[:：]",
    r"請不吝點贊訂閱轉發打賞",
    r"请不吝点赞订阅转发打赏",
    r"点赞订阅打赏",
    r"明鏡與點點欄目",
    r"明镜与点点栏目",
    r"\b(?:subs?|subtitles?) by\b",
    r"\bamara\.org\b",
    r"thanks for watching",
)


def is_hallucination(text: str) -> bool:
    if len(text) > 80:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in HALLUCINATION_PATTERNS)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、])", r"\1", text)
    return text.strip()


def simplify(text: str) -> str:
    try:
        import zhconv
    except ImportError:
        return text
    return zhconv.convert(text, "zh-cn")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="fixed temperature; omit to use Whisper's fallback schedule",
    )
    parser.add_argument("--download-root", default=None)
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="enable Whisper's cross-window conditioning (default: off)",
    )
    parser.add_argument(
        "--keep-traditional",
        action="store_true",
        help="do not convert Chinese output to Simplified",
    )
    args = parser.parse_args()
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    import torch
    import whisper

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    print(
        f"torch {torch.__version__} device={device} "
        f"gpu={torch.cuda.get_device_name(0) if device == 'cuda' else '-'}",
        flush=True,
    )

    started = time.time()
    model = whisper.load_model(
        args.model,
        device=device,
        download_root=args.download_root,
    )
    print(f"model_loaded_seconds={time.time() - started:.1f}", flush=True)

    started = time.time()
    transcribe_kwargs = dict(
        language=args.language or None,
        initial_prompt=args.prompt,
        fp16=device == "cuda",
        verbose=False,
        beam_size=args.beam,
        condition_on_previous_text=args.condition_on_previous_text,
    )
    if args.temperature is not None:
        transcribe_kwargs["temperature"] = args.temperature
    result = model.transcribe(
        str(Path(args.audio).expanduser().resolve()),
        **transcribe_kwargs,
    )
    elapsed = time.time() - started

    segments = []
    dropped = 0
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if is_hallucination(text):
            dropped += 1
            continue
        text = clean_text(text)
        if not args.keep_traditional:
            text = simplify(text)
        if text:
            segments.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": text,
                }
            )

    payload = {
        "model": f"openai-whisper {args.model} ({device})",
        "language": result.get("language"),
        "duration": segments[-1]["end"] if segments else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "dropped_hallucinations": dropped,
        "segments": segments,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"DONE segments={len(segments)} dropped={dropped} "
        f"elapsed_seconds={elapsed:.1f} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())