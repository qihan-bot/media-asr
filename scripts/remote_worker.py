#!/usr/bin/env python3
"""Remote worker for the full download → normalize → ASR → render pipeline.

Runs inside the remote GPU work directory.  The local CLI uploads:

    jobs/<id>/job.json
    jobs/<id>/source.<ext>      (only for local files)
    jobs/<id>/media_asr.zip
    jobs/<id>/remote_worker.py

runs ``jobs/<id>/remote_worker.py`` from the work directory and downloads
``jobs/<id>/bundle.zip``.  Each job has its own copy of the code, so jobs can
run side by side; they take turns on the GPU through ``gpu.lock`` in the work
directory, held from loading Whisper to the end of diarization.

A job with ``"kind": "embed"`` (``media-asr voiceprint``) instead embeds the
given segments of the episode with the speaker-embedding model, under the same
lock, and leaves ``embeddings.npz`` in its job directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import wave
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# In the work directory: one job at a time loads models and runs on the GPU.
GPU_LOCK = "gpu.lock"
# Voiceprint jobs ("kind": "embed"): the speaker-embedding model of pyannote 3.1.
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
PAD = 0.2  # trim segment edges to avoid boundary bleed from neighbours
MIN_LEN = 1.0  # seconds of audio after trimming
MAX_LEN = 20.0  # embed at most the middle MAX_LEN seconds

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


@contextmanager
def gpu_lock(path: Path) -> Iterator[None]:
    """Hold the host's GPU lock for the block, waiting while another job has it.

    An flock, so the kernel releases it if the worker dies. A no-op where
    fcntl is unavailable (Windows).
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"worker: waiting for the GPU ({path.name} is held by another job)", flush=True)
            started = time.time()
            fcntl.flock(handle, fcntl.LOCK_EX)
            print(f"worker: gpu_wait_seconds={time.time() - started:.1f}", flush=True)
        yield


def is_hallucination(text: str) -> bool:
    if len(text) > 80:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in HALLUCINATION_PATTERNS)


def clean_text(text: str, simplify: bool = True) -> str:
    text = text.strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、])", r"\1", text)
    if simplify:
        try:
            import zhconv
        except ImportError:
            pass
        else:
            text = zhconv.convert(text, "zh-cn")
    return text.strip()


def run_diarization(wav_path: Path, job: dict) -> list[tuple[float, float, int]]:
    import numpy as np
    import sherpa_onnx

    with wave.open(str(wav_path), "rb") as handle:
        if handle.getframerate() != 16000:
            raise RuntimeError(f"expected 16 kHz audio, got {handle.getframerate()}")
        samples = (
            np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=job.get("diarization_segmentation", "models/diarization/segmentation.onnx")
            ),
            num_threads=int(job.get("diarization_threads", 8)),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=job.get("diarization_embedding", "models/diarization/speaker_embedding.onnx"),
            num_threads=int(job.get("diarization_threads", 8)),
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=int(job.get("num_speakers", -1)),
            threshold=float(job.get("diarization_threshold", 0.5)),
        ),
        min_duration_on=float(job.get("min_duration_on", 0.3)),
        min_duration_off=float(job.get("min_duration_off", 0.5)),
    )
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = diarizer.process(samples).sort_by_start_time()
    return [(float(turn.start), float(turn.end), int(turn.speaker)) for turn in result]


def assign_speakers(segments: list, turns: list[tuple[float, float, int]]) -> None:
    for segment in segments:
        overlaps: dict[int, float] = {}
        for start, end, speaker in turns:
            overlap = min(segment.end, end) - max(segment.start, start)
            if overlap > 0:
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        if overlaps:
            speaker = max(overlaps, key=overlaps.get)
            segment.speaker = f"SPEAKER_{speaker:02d}"


def speaker_for_interval(
    start: float, end: float, turns: list[tuple[float, float, int]]
) -> int | None:
    overlaps: dict[int, float] = {}
    for turn_start, turn_end, speaker in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > 0:
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
    if overlaps:
        return max(overlaps, key=overlaps.get)
    nearest = None
    distance = 1.5
    for turn_start, turn_end, speaker in turns:
        current = min(abs(start - turn_end), abs(end - turn_start))
        if current < distance:
            distance = current
            nearest = speaker
    return nearest


def build_speaker_segments(
    raw_segments: list,
    turns: list[tuple[float, float, int]],
    simplify: bool = True,
) -> list:
    from media_asr.transcribe import TranscriptSegment

    words = []
    for raw in raw_segments:
        for word in raw.get("words") or []:
            words.append(
                (
                    float(word.get("start", raw["start"])),
                    float(word.get("end", raw["start"])),
                    str(word.get("word", "")),
                )
            )
    words.sort(key=lambda item: item[0])

    if not words:
        segments = []
        for raw in raw_segments:
            speaker = speaker_for_interval(float(raw["start"]), float(raw["end"]), turns)
            text = clean_text(raw.get("text", ""), simplify=simplify)
            if text:
                segments.append(
                    TranscriptSegment(
                        start=float(raw["start"]),
                        end=float(raw["end"]),
                        text=text,
                        speaker=f"SPEAKER_{speaker:02d}" if speaker is not None else None,
                    )
                )
        return segments

    # Merge adjacent turns from the same speaker, and absorb very short turns
    # into their neighbour so word assignment does not flicker on 0.2 s turns.
    ordered = sorted(turns)
    merged: list[list[float | int]] = []
    for start, end, speaker in ordered:
        if merged and speaker == merged[-1][2] and start - float(merged[-1][1]) < 1.0:
            merged[-1][1] = max(float(merged[-1][1]), end)
        else:
            merged.append([start, end, speaker])

    smoothed: list[list[float | int]] = []
    for turn in merged:
        if smoothed and float(turn[1]) - float(turn[0]) < 0.8 and turn[0] - float(smoothed[-1][1]) < 0.8:
            smoothed[-1][1] = max(float(smoothed[-1][1]), float(turn[1]))
        else:
            smoothed.append(turn)

    buckets: dict[int, list[tuple[float, float, str]]] = {index: [] for index in range(len(smoothed))}
    for word_start, word_end, text in words:
        midpoint = (word_start + word_end) / 2
        chosen = None
        for index, (turn_start, turn_end, _speaker) in enumerate(smoothed):
            if float(turn_start) <= midpoint <= float(turn_end):
                chosen = index
                break
        if chosen is None:
            chosen = min(
                range(len(smoothed)),
                key=lambda index: min(
                    abs(midpoint - float(smoothed[index][0])),
                    abs(midpoint - float(smoothed[index][1])),
                ),
            )
        buckets[chosen].append((word_start, word_end, text))

    segments = []
    for index, (turn_start, turn_end, speaker) in enumerate(smoothed):
        bucket = buckets.get(index) or []
        if not bucket:
            continue
        text = clean_text("".join(item[2] for item in bucket), simplify=simplify)
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=bucket[0][0],
                end=bucket[-1][1],
                text=text,
                speaker=f"SPEAKER_{int(speaker):02d}" if speaker is not None else None,
            )
        )
    return segments


def run_pyannote(wav_path: Path, job: dict) -> list[tuple[float, float, int]]:
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(
            job.get("pyannote_pipeline", "pyannote/speaker-diarization-3.1"),
            token=token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            job.get("pyannote_pipeline", "pyannote/speaker-diarization-3.1"),
            use_auth_token=token,
        )
    if pipeline is None:
        raise RuntimeError("pyannote pipeline could not be loaded; check HF access")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    kwargs = {}
    num_speakers = int(job.get("num_speakers", -1))
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    # Avoid torchcodec/FFmpeg DLL issues on Windows: feed the waveform directly.
    import wave

    import numpy as np

    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        samples = (
            np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
            .reshape(-1, channels)
            .T.astype(np.float32)
            / 32768.0
        )
    waveform = torch.from_numpy(samples)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)

    annotation = getattr(
        diarization, "exclusive_speaker_diarization", diarization
    )
    labels = sorted(annotation.labels())
    index = {label: position for position, label in enumerate(labels)}
    return [
        (float(segment.start), float(segment.end), index[label])
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def embed_segments(wav_path: Path, segments: list) -> tuple:
    """WeSpeaker embedding of each ``[start, end]`` segment (edges trimmed by
    PAD, at most the middle MAX_LEN seconds, skipped under MIN_LEN): the
    embeddings [n, 256], their segment indices and the samples each covers."""
    import numpy as np
    import torch
    from pyannote.audio import Model

    with wave.open(str(wav_path), "rb") as handle:
        if handle.getframerate() != 16000 or handle.getnchannels() != 1:
            raise RuntimeError(f"expected 16 kHz mono audio: {wav_path}")
        audio = (
            np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model.from_pretrained(EMBEDDING_MODEL)
    model.to(device).eval()
    embeddings, kept, samples = [], [], []
    with torch.inference_mode():
        for index, (start, end) in enumerate(segments):
            start, end = float(start) + PAD, float(end) - PAD
            if end - start < MIN_LEN:
                continue
            if end - start > MAX_LEN:
                middle = (start + end) / 2
                start, end = middle - MAX_LEN / 2, middle + MAX_LEN / 2
            chunk = audio[int(start * 16000) : int(end * 16000)]
            if len(chunk) < MIN_LEN * 16000:
                continue
            tensor = torch.from_numpy(chunk).to(device)[None, None, :]
            embeddings.append(model(tensor).squeeze(0).float().cpu().numpy())
            kept.append(index)
            samples.append(len(chunk))
    return (
        np.stack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 256), np.float32),
        np.array(kept, dtype=np.int32),
        np.array(samples, dtype=np.int64),
    )


def embed_job(job: dict, root: Path, job_dir: Path, download, normalize_audio) -> int:
    """A voiceprint job: download and decode the episode beside other jobs, embed
    its segments under the GPU lock, write ``embeddings.npz`` to the job directory."""
    import numpy as np

    print(f"worker: {job['id']} voiceprint, {len(job['segments'])} segments", flush=True)
    item = {
        "source": job["source"],
        "source_type": job.get("source_type", "audio"),
        "meta_json": json.dumps(
            {"audio_url": job.get("audio_url"), "page_url": job.get("page_url")},
            ensure_ascii=False,
        ),
    }
    source_path, is_local = download(
        item, job_dir, progress=lambda message: print(message, flush=True)
    )
    wav_path = job_dir / "audio.wav"
    normalize_audio(source_path, wav_path)
    if not is_local and source_path != wav_path:
        source_path.unlink(missing_ok=True)
    with gpu_lock(root / GPU_LOCK):
        started = time.time()
        emb, seg, samples = embed_segments(wav_path, job["segments"])
        print(
            f"worker: embedded {len(seg)}/{len(job['segments'])} segments "
            f"in {time.time() - started:.1f}s",
            flush=True,
        )
    np.savez(job_dir / "embeddings.npz", emb=emb, seg=seg, samples=samples, fp=np.array(job["fp"]))
    wav_path.unlink(missing_ok=True)
    print(f"DONE embeddings={len(seg)}", flush=True)
    return 0


def diarize(wav_path: Path, job: dict) -> tuple[list[tuple[float, float, int]], float]:
    """Speaker turns of the episode and the seconds it took."""
    print("worker: running speaker diarization", flush=True)
    started = time.time()
    if job.get("diarization_backend", "sherpa") == "pyannote":
        turns = run_pyannote(wav_path, job)
    else:
        turns = run_diarization(wav_path, job)
    return turns, round(time.time() - started, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)

    root = Path.cwd()
    here = Path(__file__).resolve().parent  # the job's own copy of the code
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    token_file = job.get("hf_token_file")
    if token_file:
        token_path = Path(token_file)
        if token_path.exists():
            os.environ["HF_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
            token_path.unlink()
    job_dir = Path(job["job_dir"])
    out_dir = job_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make ffmpeg/ffprobe and the uploaded package available.
    os.environ["PATH"] = str(root) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(here))
    package = here / "media_asr.zip"
    if package.exists():
        with zipfile.ZipFile(package) as archive:
            archive.extractall(here)

    from media_asr.fetch import download, normalize_audio
    from media_asr.render import render_all
    from media_asr.transcribe import TranscriptSegment
    from media_asr.util import ffprobe_duration, now_iso

    if job.get("kind") == "embed":
        return embed_job(job, root, job_dir, download, normalize_audio)
    print(f"worker: {job['id']} source_type={job.get('source_type')}", flush=True)
    item = {
        "source": job["source"],
        "source_type": job.get("source_type", "audio"),
        "meta_json": json.dumps(
            {
                "audio_url": job.get("audio_url"),
                "page_url": job.get("page_url"),
            },
            ensure_ascii=False,
        ),
    }
    source_path, is_local = download(item, job_dir, progress=lambda message: print(message, flush=True))
    duration = ffprobe_duration(source_path) or job.get("duration")
    if not duration or duration <= 0:
        raise RuntimeError(f"cannot determine duration: {source_path}")
    print(f"worker: audio={source_path} duration={duration:.1f}s", flush=True)

    wav_path = job_dir / "audio.wav"
    normalize_audio(source_path, wav_path)
    if not is_local and source_path != wav_path:
        source_path.unlink(missing_ok=True)

    import torch
    import whisper

    model_name = job.get("model", "large-v3-turbo")
    diarization_elapsed = None
    turns: list[tuple[float, float, int]] = []
    pyannote = bool(job.get("diarize")) and job.get("diarization_backend") == "pyannote"
    # Download and ffmpeg above, rendering and bundling below run beside other
    # jobs; loading and running the models waits for the GPU.
    with gpu_lock(root / GPU_LOCK):
        print(
            f"worker: torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
            flush=True,
        )
        started = time.time()
        model = whisper.load_model(
            model_name,
            device="cuda",
            download_root=job.get("download_root", "models"),
        )
        print(f"worker: model_loaded_seconds={time.time() - started:.1f}", flush=True)

        started = time.time()
        transcribe_kwargs = dict(
            language=job.get("language") or None,
            initial_prompt=job.get("prompt") or None,
            fp16=torch.cuda.is_available(),
            verbose=False,
            beam_size=int(job.get("beam", 5)),
            word_timestamps=bool(job.get("diarize")),
            condition_on_previous_text=bool(job.get("condition_on_previous_text", False)),
        )
        if job.get("temperature") is not None:
            transcribe_kwargs["temperature"] = float(job["temperature"])
        result = model.transcribe(str(wav_path), **transcribe_kwargs)
        elapsed = time.time() - started
        if pyannote:
            turns, diarization_elapsed = diarize(wav_path, job)
    if job.get("diarize") and not pyannote:  # sherpa-onnx runs on the CPU
        turns, diarization_elapsed = diarize(wav_path, job)

    raw_segments = result.get("segments", [])
    speakers: list[str] = []
    if job.get("diarize"):
        segments = build_speaker_segments(
            raw_segments, turns, simplify=not bool(job.get("traditional"))
        )
        speakers = sorted({segment.speaker for segment in segments if segment.speaker})
        print(
            f"worker: diarization {len(turns)} turns, {len(speakers)} speakers, "
            f"{diarization_elapsed}s",
            flush=True,
        )
    else:
        segments = []
        for segment in raw_segments:
            text = clean_text(
                segment.get("text", ""), simplify=not bool(job.get("traditional"))
            )
            if text:
                segments.append(
                    TranscriptSegment(
                        start=float(segment["start"]),
                        end=float(segment["end"]),
                        text=text,
                    )
                )

    dropped = 0
    filtered = []
    for segment in segments:
        if is_hallucination(segment.text):
            dropped += 1
            continue
        if segment.text:
            filtered.append(segment)
    segments = filtered

    model_label = f"openai-whisper {model_name} (CUDA fp16, remote)"
    render_all(
        out_dir=out_dir,
        item_id=job["id"],
        title=job.get("title") or "untitled",
        source=job.get("original_source") or job.get("source") or "",
        duration=duration,
        language=result.get("language") or job.get("language"),
        model=model_label,
        transcribed_at=now_iso(),
        segments=segments,
    )

    meta = {
        "id": job["id"],
        "title": job.get("title"),
        "source": job.get("original_source") or job.get("source"),
        "source_type": job.get("source_type"),
        "duration": duration,
        "published_at": job.get("published_at"),
        "feed_url": job.get("feed_url"),
        "processed_at": now_iso(),
        "worker": "remote",
    }
    (job_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run = {
        "model": model_label,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "compute_type": "fp16" if torch.cuda.is_available() else "float32",
        "beam": int(job.get("beam", 5)),
        "language": result.get("language") or job.get("language"),
        "duration": duration,
        "segments": len(segments),
        "chars": sum(len(segment.text) for segment in segments),
        "elapsed_seconds": round(elapsed, 1),
        "realtime_factor": round(duration / elapsed, 2) if elapsed else None,
        "dropped_hallucinations": dropped,
        "diarization": bool(job.get("diarize")),
        "diarization_backend": job.get("diarization_backend", "sherpa"),
        "diarization_elapsed_seconds": diarization_elapsed,
        "speakers": speakers,
        "created_at": now_iso(),
        "worker": "remote",
    }
    (job_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    keep_audio = bool(job.get("keep_audio"))
    bundle = job_dir / "bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.iterdir()):
            archive.write(path, path.name)
        archive.write(job_dir / "meta.json", "meta.json")
        archive.write(job_dir / "run.json", "run.json")
        if keep_audio and wav_path.exists():
            archive.write(wav_path, "audio.wav")

    print(
        f"DONE segments={len(segments)} dropped={dropped} "
        f"elapsed_seconds={elapsed:.1f} bundle={bundle}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())