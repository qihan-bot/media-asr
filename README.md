# media-asr

[English](README.md) | [简体中文](README.zh-CN.md)

**Version 0.01** — local-first speech-to-text for videos and podcasts.

`media-asr` turns a video URL, podcast RSS feed, or local media file into a
searchable transcript. Transcription runs locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) on CPU, or can be
offloaded to a remote GPU host running OpenAI Whisper. Outputs are written as
Markdown, SRT, plain text, and JSON, and every segment is indexed in SQLite FTS5
for full-text search.

The project is intentionally small and hackable: a CLI, a handful of Python
modules, and optional remote-worker scripts.

## Features

- YouTube, Bilibili, Vimeo, direct audio URLs, podcast RSS feeds, and local files.
- Local CPU transcription with `faster-whisper` (`small` by default).
- Optional remote GPU backend with OpenAI Whisper `large-v3-turbo` over SSH.
- Two remote modes:
  - `asr`: local download/ffmpeg/render, remote ASR only.
  - `full`: remote download, ffmpeg, ASR, and rendering; local indexing only.
- Speaker diarization with `sherpa-onnx` (offline) or `pyannote.audio` (gated
  HuggingFace models).
- Word-level timestamps and speaker turn alignment when diarization is enabled.
- Markdown / SRT / TXT / JSON transcripts and SQLite FTS5 search.
- Optional summaries with Antigravity (`agy`) or the DeepSeek API.
- Chinese output is normalized to Simplified Chinese by default
  (`--traditional` keeps the original script).
- Runs on Linux and Windows; no cloud account is required for local ASR.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe`
- For remote mode: SSH access to a GPU host with Python, `ffmpeg`, and the
  remote dependencies described in [docs/remote-gpu.md](docs/remote-gpu.md).

## Install

```bash
git clone https://github.com/qihan-bot/media-asr.git
cd media-asr
python3 -m pip install -e .
```

Or use the existing system environment and run `./media-asr` from the checkout.
The command wrapper resolves its own directory, so it works from any cwd.

Optional extras used by some features:

```bash
python3 -m pip install feedparser zhconv paramiko
```

## Quick start

```bash
# A video
./media-asr add 'https://www.youtube.com/watch?v=...' --run

# A Bilibili video in Chinese
./media-asr add 'https://www.bilibili.com/video/BV...' --language zh --run

# A local file
./media-asr add ~/Downloads/interview.mp4 --title 'Interview' --language zh --run

# A podcast RSS feed: enqueue the latest 3 episodes
./media-asr add 'https://example.com/feed.xml' --latest 3
./media-asr run

# Inspect and search
./media-asr list
./media-asr search 'information overload'
./media-asr show <id>
```

`--run` is optional. Without it, items are queued as `pending` and processed by
`./media-asr run`.

## Commands

| Command | Purpose |
| --- | --- |
| `add <url\|file>` | Resolve a URL/feed/file and enqueue items (`--latest N`, `--title`, `--run`) |
| `run` | Process pending items (`--model`, `--language`, `--prompt`, `--keep-audio`) |
| `retranscribe <id>` | Re-run an existing item with another model into `runs/<label>/` |
| `refresh` | Re-check subscribed RSS feeds for new episodes (`--latest N`, `--run`) |
| `list` | List items (`--status pending`, `--limit`) |
| `show <id>` | Show metadata and the beginning of the transcript (`--full`) |
| `search <query>` | Search all transcript segments with timestamps and speakers |
| `summarize <id>` | Write `summary.md` (`--engine auto\|agy\|deepseek\|none`) |
| `remote-check` | Check SSH, GPU, Python, and remote dependencies |
| `feeds` | List subscribed feeds |
| `stats` | Counts, transcribed hours, segments, feeds |
| `remove <id>` | Delete an item and its files (`--yes`) |

Useful run options:

```text
--model small|medium|large-v3-turbo|<path>   # default: small
--language zh|en|...                        # empty = auto-detect
--prompt 'Name, term, acronym'              # hotwords / punctuation hint
--beam 5                                    # accuracy vs. speed
--threads 8                                 # CPU threads
--keep-audio                                # keep the 16 kHz WAV
--offline                                   # never download a model
--traditional                               # keep Traditional Chinese
--backend local|remote                      # local CPU or remote GPU
--remote-mode asr|full                      # remote scope
--diarize                                   # speaker diarization
--num-speakers 4                            # -1 = auto-detect
--diarization-backend sherpa|pyannote       # default: sherpa
```

## Remote GPU backend

The local CLI can offload work to a remote GPU host over SSH. Audio is uploaded
to a temporary job directory, transcribed on the remote host, downloaded as a
bundle, and indexed locally. Remote files are removed unless `--remote-keep`.

Set the connection once:

```bash
export MEDIA_ASR_REMOTE_HOST=gpu-host
export MEDIA_ASR_REMOTE_USER="$USER"
export MEDIA_ASR_REMOTE_KEY="$HOME/.ssh/id_ed25519"
export MEDIA_ASR_REMOTE_WORKDIR=media-asr-work
```

Then:

```bash
# ASR only: local download/ffmpeg/render, remote Whisper
./media-asr run --backend remote --remote-mode asr

# Full remote pipeline: remote download/ffmpeg/ASR/render
./media-asr run --backend remote --remote-mode full

# Health check
./media-asr remote-check
```

The `media-asr-remote` wrapper defaults to full mode and can read a local
config file at `~/.config/media-asr/remote.env` (not committed):

```bash
# ~/.config/media-asr/remote.env
MEDIA_ASR_REMOTE_HOST=gpu-host
MEDIA_ASR_REMOTE_USER=me
MEDIA_ASR_REMOTE_KEY=/home/me/.ssh/id_ed25519
MEDIA_ASR_REMOTE_WORKDIR=media-asr-work
# Optional: inject the SSH password from Avibe Vault
# MEDIA_ASR_REMOTE_SECRET=<vault-ref>
```

```bash
./media-asr-remote run
./media-asr-remote run --remote-mode asr
./media-asr-remote retranscribe <id> --label turbo-remote
```

SSH host keys are checked against `~/.ssh/known_hosts`. Only use
`--remote-auto-add-host-key` for a first trusted connection. Passwords can be
provided through `MEDIA_ASR_REMOTE_PASSWORD`; do not put them in repository
files or shell history.

Remote setup (CUDA PyTorch, OpenAI Whisper, ffmpeg, yt-dlp, diarization models)
is documented in [docs/remote-gpu.md](docs/remote-gpu.md).

## Speaker diarization

Diarization is available in the remote **full** pipeline.

### sherpa-onnx (default, offline)

```bash
./media-asr run --backend remote --remote-mode full \
  --diarize --num-speakers 4
```

Models live in `models/diarization/` on the remote host. Setup is in
[docs/remote-gpu.md](docs/remote-gpu.md).

### pyannote (higher accuracy, gated models)

Accept the terms for these HuggingFace repositories with the account that owns
`HF_TOKEN`:

- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-3.1`
- `pyannote/speaker-diarization-community-1`

Then:

```bash
HF_TOKEN=... ./media-asr run --backend remote --remote-mode full \
  --diarize --num-speakers 5 \
  --diarization-backend pyannote \
  --pyannote-pipeline pyannote/speaker-diarization-3.1
```

Both backends use Whisper word timestamps, align words to diarization turns,
and write `SPEAKER_00` labels into Markdown, SRT, TXT, JSON, and the SQLite
segments/search index. Labels are approximate; map them to names after checking
the first turns. On a 94-minute, 5-speaker episode, pyannote 3.1 took about
100 s for diarization and produced a clean five-way split; sherpa took about
250 s and split/merged several voices.

## Output layout

```text
data/
  media.db
  items/<id>/
    meta.json
    audio.wav              # only with --keep-audio
    transcript.md
    transcript.txt
    transcript.srt
    transcript.json
    summary.md             # optional
    run.json               # remote run metrics
    runs/<label>/          # retranscribe / diarized variants
```

Override the data directory with `MEDIA_ASR_DATA`. Model cache defaults to
`~/.cache/whisper` (`MEDIA_ASR_MODEL_DIR`).

## Configuration

| Variable | Meaning |
| --- | --- |
| `MEDIA_ASR_HOME` | Project root for defaults (default: source checkout) |
| `MEDIA_ASR_DATA` | Data directory (default: `<project>/data`) |
| `MEDIA_ASR_MODEL` | Local faster-whisper model (default `small`) |
| `MEDIA_ASR_LANGUAGE` | Default language |
| `MEDIA_ASR_THREADS` | CPU threads |
| `MEDIA_ASR_COMPUTE` | CTranslate2 compute type (default `int8`) |
| `MEDIA_ASR_BACKEND` | `local` or `remote` |
| `MEDIA_ASR_REMOTE_HOST` | SSH host |
| `MEDIA_ASR_REMOTE_USER` | SSH user (default: current user) |
| `MEDIA_ASR_REMOTE_PORT` | SSH port |
| `MEDIA_ASR_REMOTE_KEY` | SSH private key |
| `MEDIA_ASR_REMOTE_WORKDIR` | Remote work directory (default `media-asr-work`) |
| `MEDIA_ASR_REMOTE_PYTHON` | Remote Python (default `.venv\Scripts\python.exe`) |
| `MEDIA_ASR_REMOTE_SHELL` | Remote shell for `cd`/Python commands (`cmd` or `sh`) |
| `MEDIA_ASR_REMOTE_MODEL` | Remote Whisper model (default `large-v3-turbo`) |
| `MEDIA_ASR_REMOTE_MODE` | `asr` or `full` |
| `MEDIA_ASR_REMOTE_PASSWORD` | SSH password (prefer a key or a secret manager) |
| `MEDIA_ASR_REMOTE_SECRET` | Optional Vault reference used by the wrapper |
| `HF_TOKEN` | HuggingFace token for pyannote models |

## Performance

Reference numbers, not guarantees. Podcast: 85 minutes of Chinese audio;
video: 94 minutes of English audio.

| Host / mode | Model | Time | Realtime |
| --- | --- | ---: | ---: |
| 6-core CPU, local | faster-whisper small/int8 | 15.0 min | 5.66× |
| 6-core CPU, local | faster-whisper medium/int8 | 39.2 min | 2.17× |
| RTX 5090 Laptop, GPU | large-v3-turbo/fp16 | 8.8 min | 9.61× |
| RTX 5090 Laptop, full remote | large-v3-turbo/fp16 | 7.9 min | 10.70× |
| RTX 5090 Laptop, GPU | medium/fp16 | 23.0 min | 3.70× |

Diarization adds about 100 s (pyannote 3.1) or 250 s (sherpa) on the
94-minute episode. Word timestamps make ASR somewhat slower.

## Limitations

- `small` is fast but weak on names, numbers, and mixed Chinese/English terms.
  Use `--prompt`, `medium`, or the remote GPU for better accuracy.
- No translation, chaptering, or speaker-name recognition.
- pyannote requires gated HuggingFace models and a token.
- OpenAI Whisper can skip short passages or hallucinate credits at the end;
  the remote worker filters common credit lines but does not guarantee
  transcript perfection.
- Diarization clusters can split or merge similar voices; treat labels as
  approximate.
- FTS5 uses the trigram tokenizer; queries shorter than 3 characters fall back
  to `LIKE`.
- Remote mode assumes a POSIX-like SSH shell or Windows OpenSSH with `cmd.exe`.
  The worker currently targets a Windows-style remote layout by default; adjust
  `MEDIA_ASR_REMOTE_PYTHON` and `MEDIA_ASR_REMOTE_WORKDIR` for Linux.

## Development

```bash
python3 -m pytest -q
python3 -m ruff check .
```

Project layout:

```text
media_asr/          CLI, fetch, ASR, DB, rendering, remote client
scripts/            remote worker, OpenAI Whisper helper, comparison tool
tests/              offline unit tests
docs/remote-gpu.md  remote GPU setup and benchmarks
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)