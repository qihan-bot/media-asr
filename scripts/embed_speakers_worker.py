#!/usr/bin/env python3
"""Remote worker: speaker embeddings for already-diarized segments.

Runs on the GPU host. For every episode in the job it downloads the source,
decodes it to 16 kHz mono, and embeds each diarized segment with the same
WeSpeaker model used inside pyannote/speaker-diarization-3.1. Results are
written as ``<out_dir>/<id>.npz``: ``emb`` float32 [n, 256], ``seg`` int
indices into the episode's diarized segment list, ``samples`` the number of
16 kHz samples actually embedded per row, and ``fp`` the segmentation
fingerprint supplied by the driver. Files are written to a temp path and
renamed, so an existing output is always complete and is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
import zipfile
from pathlib import Path

PAD = 0.2  # trim segment edges to avoid boundary bleed from neighbours
MIN_LEN = 1.0  # seconds of audio after trimming
MAX_LEN = 20.0  # embed at most the middle MAX_LEN seconds


def load_wav(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def valid_output(path: Path, fingerprint: str) -> bool:
    """Complete output for this segmentation? Shared with the local driver (embed_speakers.py)."""
    import numpy as np

    try:
        with np.load(path) as z:
            return (
                str(z["fp"]) == fingerprint
                and z["emb"].ndim == 2
                and len(z["emb"]) == len(z["seg"]) == len(z["samples"])
            )
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    os.environ["PATH"] = str(root) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(root))
    package = root / "media_asr.zip"
    if package.exists():
        with zipfile.ZipFile(package) as archive:
            archive.extractall(root)

    import numpy as np
    import torch
    from pyannote.audio import Model

    from media_asr.fetch import download, normalize_audio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    model.to(device).eval()

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, episode in enumerate(job["episodes"], start=1):
        target = out_dir / f"{episode['id']}.npz"
        if target.exists() and valid_output(target, episode["fp"]):
            print(f"[{index}/{len(job['episodes'])}] skip {episode['id']}", flush=True)
            continue
        work = out_dir / f"tmp_{episode['id']}_{os.getpid()}"
        started = time.time()
        try:
            item = {
                "source": episode["source"],
                "source_type": episode.get("source_type", "audio"),
                "meta_json": json.dumps({"audio_url": episode.get("audio_url")}),
            }
            source_path = None
            for attempt in range(3):
                try:
                    source_path, _ = download(item, work)
                    break
                except Exception as exc:  # transient CDN/SSL errors
                    print(f"download attempt {attempt + 1} failed: {exc}", flush=True)
                    time.sleep(5 * (attempt + 1))
            if source_path is None:
                raise RuntimeError("download failed 3 times")
            wav_path = work / "audio.wav"
            normalize_audio(source_path, wav_path)
            audio = load_wav(wav_path)
            embs, kept, samples = [], [], []
            with torch.inference_mode():
                for seg_index, (start, end) in enumerate(episode["segments"]):
                    start, end = start + PAD, end - PAD
                    if end - start < MIN_LEN:
                        continue
                    if end - start > MAX_LEN:
                        middle = (start + end) / 2
                        start, end = middle - MAX_LEN / 2, middle + MAX_LEN / 2
                    chunk = audio[int(start * 16000) : int(end * 16000)]
                    if len(chunk) < MIN_LEN * 16000:
                        continue
                    tensor = torch.from_numpy(chunk).to(device)[None, None, :]
                    embs.append(model(tensor).squeeze(0).float().cpu().numpy())
                    kept.append(seg_index)
                    samples.append(len(chunk))
            partial = work / "result.npz"
            np.savez(
                partial,
                emb=np.stack(embs).astype(np.float32) if embs else np.zeros((0, 256), np.float32),
                seg=np.array(kept, dtype=np.int32),
                samples=np.array(samples, dtype=np.int64),
                fp=np.array(episode["fp"]),
            )
            partial.replace(target)
            print(
                f"[{index}/{len(job['episodes'])}] done {episode['id']} "
                f"{len(kept)}/{len(episode['segments'])} segs {time.time() - started:.0f}s",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED {episode['id']}: {exc}", flush=True)
        finally:
            for path in sorted(work.glob("**/*"), reverse=True) if work.exists() else []:
                path.unlink() if path.is_file() else path.rmdir()
            if work.exists():
                work.rmdir()
    print(f"embed done failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
