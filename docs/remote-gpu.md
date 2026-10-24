# Remote GPU backend

`media-asr` can offload work to a machine with a CUDA GPU over SSH. The local
side keeps the SQLite index and CLI state; the remote side runs download
(optional), ffmpeg, Whisper, and rendering (optional).

Two modes are available:

- `asr`: upload a normalized 16 kHz WAV, run Whisper remotely, download JSON,
  render locally.
- `full`: send a URL (or upload a local source); the remote host downloads,
  normalizes, transcribes, diarizes (optional), and renders; the local side
  receives a bundle and indexes it.

## Requirements

- SSH access with key or password authentication.
- Remote Python 3.10+ (3.12 recommended).
- FFmpeg / ffprobe on the remote host.
- PyTorch with CUDA for GPU inference.
- OpenAI Whisper for the remote worker.
- For full mode: `yt-dlp`, `feedparser`, `httpx`, `zhconv`.
- Optional: `sherpa-onnx` and/or `pyannote.audio` for diarization.

The examples below are generic; replace paths and Python executable names for
your OS.

## Windows host setup

A self-contained setup under `%USERPROFILE%\media-asr-work`:

```powershell
$root = Join-Path $env:USERPROFILE 'media-asr-work'
New-Item -ItemType Directory -Force $root | Out-Null
Set-Location $root

# uv as a single executable
curl.exe -L -o uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
Expand-Archive uv.zip .
.\uv.exe python install 3.12
.\uv.exe venv .venv --python 3.12

# CUDA PyTorch
.\uv.exe pip install --python .venv\Scripts\python.exe torch torchaudio `
  --index-url https://download.pytorch.org/whl/cu128

# Whisper and full-mode helpers
.\uv.exe pip install --python .venv\Scripts\python.exe openai-whisper
.\uv.exe pip install --python .venv\Scripts\python.exe yt-dlp feedparser httpx zhconv

# Portable FFmpeg
curl.exe -L -o ffmpeg.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
Expand-Archive ffmpeg.zip ffmpeg
Copy-Item (Get-ChildItem ffmpeg -Recurse -Filter ffmpeg.exe | Select-Object -First 1).FullName .
Copy-Item (Get-ChildItem ffmpeg -Recurse -Filter ffprobe.exe | Select-Object -First 1).FullName .
```

Use `MEDIA_ASR_REMOTE_WORKDIR=media-asr-work` and the default
`MEDIA_ASR_REMOTE_PYTHON=.venv\Scripts\python.exe`.

## Linux host setup

```bash
sudo apt-get install -y ffmpeg python3 python3-venv
python3 -m venv ~/media-asr-work/.venv
~/media-asr-work/.venv/bin/pip install --upgrade pip
~/media-asr-work/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/media-asr-work/.venv/bin/pip install openai-whisper yt-dlp feedparser httpx zhconv
```

Use `MEDIA_ASR_REMOTE_WORKDIR=media-asr-work` and
`MEDIA_ASR_REMOTE_PYTHON=.venv/bin/python` (the worker's `cd` command assumes a
POSIX shell when configured for Linux; see "Linux notes" below).

## SSH configuration

```bash
export MEDIA_ASR_REMOTE_HOST=gpu-host
export MEDIA_ASR_REMOTE_USER="$USER"
export MEDIA_ASR_REMOTE_KEY="$HOME/.ssh/id_ed25519"
export MEDIA_ASR_REMOTE_WORKDIR=media-asr-work
```

Test:

```bash
media-asr remote-check
```

Host keys are checked against `~/.ssh/known_hosts`. Only use
`--remote-auto-add-host-key` for a first trusted connection.

If a password is required, inject it through the environment rather than a
repository file:

```bash
MEDIA_ASR_REMOTE_PASSWORD=... media-asr run --backend remote
```

With Avibe Vault, the `media-asr-remote` wrapper can do this from a reference:

```bash
# ~/.config/media-asr/remote.env
MEDIA_ASR_REMOTE_HOST=gpu-host
MEDIA_ASR_REMOTE_USER=me
MEDIA_ASR_REMOTE_KEY=/home/me/.ssh/id_ed25519
MEDIA_ASR_REMOTE_WORKDIR=media-asr-work
MEDIA_ASR_REMOTE_SECRET=<vault-ref>
```

## Usage

```bash
# Remote ASR only
media-asr run --backend remote --remote-mode asr

# Full remote pipeline
media-asr run --backend remote --remote-mode full

# Convenience wrapper (defaults to full mode)
media-asr-remote run
media-asr-remote retranscribe <id> --label turbo-remote
```

Remote temporary job directories are removed unless `--remote-keep`.

## Several jobs on one host

Jobs can run side by side on one GPU host (`media-asr library-update --jobs N`
does). Each job uploads its own copy of the worker and the package to
`jobs/<id>/` and runs it from there, so jobs never rewrite each other's code.
They take turns on the GPU through an flock on `gpu.lock` in the work
directory, held from loading Whisper to the end of pyannote diarization (the
ASR-only mode holds it for loading and transcribing). Download, ffmpeg,
sherpa-onnx diarization (CPU), rendering and bundling run outside it, so one
job's downloads overlap another's GPU work. A job that has to wait logs
`worker: waiting for the GPU` and then `worker: gpu_wait_seconds=...`.

- The lock belongs to the work directory: give every client of a host the same
  `MEDIA_ASR_REMOTE_WORKDIR`.
- A job keeps its GPU memory until its worker exits, a few seconds after it
  releases the lock (rendering and bundling); one Whisper large-v3-turbo plus
  pyannote 3.1 job peaks at 7226 MiB on an RTX 3080 Ti, so a 12 GB card has
  room for that.
- Without `fcntl` (Windows hosts) the lock is a no-op; run one job at a time
  there.
- Voiceprint jobs (`media-asr voiceprint`, `"kind": "embed"`) take the same
  lock to load the speaker-embedding model and embed an episode's segments,
  in `jobs/<id>.voiceprint/`; their download and ffmpeg run outside it.

Measured on an RTX 3080 Ti host (2026-10-24; `library-update` on a scratch
library of 3 podcast episodes, 158 min of audio; nvidia-smi sampled every 0.5 s):

| | `--jobs 1` | `--jobs 3` |
| --- | ---: | ---: |
| Transcription wall time | 625 s | 580 s |
| GPU busy (utilization > 0) | 84% | 90% |
| Mean GPU utilization | 62% | 68% |
| Peak GPU memory | 7226 MiB | 7226 MiB |

Downloads ran at 7–17 MB/s on that host, so little was left to overlap; most
of the remaining idle time is inside the locked phase (model loading,
pyannote's CPU work). A slower download hides behind another job's GPU phase
the same way, so the gain grows with download time (not measured).

## Diarization

### sherpa-onnx (offline)

```powershell
.\uv.exe pip install --python .venv\Scripts\python.exe sherpa-onnx
New-Item -ItemType Directory -Force models\diarization | Out-Null
curl.exe -L -o models\diarization\segmentation.onnx `
  https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx
curl.exe -L -o models\diarization\speaker_embedding.onnx `
  https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/nemo_en_speakerverification_speakernet.onnx
```

```bash
media-asr run --backend remote --remote-mode full \
  --diarize --num-speakers 5 --diarization-backend sherpa
```

### pyannote (gated models)

Install:

```powershell
.\uv.exe pip install --python .venv\Scripts\python.exe torchaudio `
  --index-url https://download.pytorch.org/whl/cu128
.\uv.exe pip install --python .venv\Scripts\python.exe pyannote.audio
```

Accept the terms for these repositories with the HuggingFace account that owns
the token:

- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-3.1`
- `pyannote/speaker-diarization-community-1`

Run:

```bash
HF_TOKEN=... media-asr run --backend remote --remote-mode full \
  --diarize --num-speakers 5 \
  --diarization-backend pyannote \
  --pyannote-pipeline pyannote/speaker-diarization-3.1
```

The worker feeds the waveform directly to pyannote, avoiding torchcodec's
Windows FFmpeg DLL requirement. The HF token is written to the remote job
directory only for the run and removed after the pipeline loads.

## Benchmarks

Reference hardware: 6-core/12-thread CPU, and an RTX 5090 Laptop GPU. Podcast:
85 minutes Chinese; video: 94 minutes English.

| Host / mode | Model | Time | Realtime |
| --- | --- | ---: | ---: |
| CPU, local | faster-whisper small/int8 | 15.0 min | 5.66× |
| CPU, local | faster-whisper medium/int8 | 39.2 min | 2.17× |
| GPU, direct | large-v3-turbo/fp16 | 8.8 min | 9.61× |
| GPU, full remote | large-v3-turbo/fp16 | 7.9 min | 10.70× |
| GPU, direct | medium/fp16 | 23.0 min | 3.70× |

Diarization on the 94-minute, 5-speaker episode:

| Backend | Time | Turns | Notes |
| --- | ---: | ---: | --- |
| pyannote 3.1 | 99.7 s | 786 | clean five-way split |
| sherpa-onnx | 250.8 s | 576 | split/merged similar male voices |

## Linux notes

Set `MEDIA_ASR_REMOTE_SHELL=sh` and `MEDIA_ASR_REMOTE_PYTHON=.venv/bin/python`
for a POSIX remote host. The worker uses `cd "$HOME/<workdir>"` (or the
absolute workdir when one is configured). Absolute Windows directories are
also supported without collapsing them to a single path component.