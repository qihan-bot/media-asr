# Changelog

## 0.0.2 — 2026-10-24

Supersedes 0.01 (PEP 440 reads "0.01" as 0.1, which would sort after 0.0.2).

Podcast library: the process scripts became package modules and subcommands.

- `media-asr library-update`: the daily pipeline in Python (refresh → per-show
  remote GPU transcription with pyannote diarization → `rediarize` → `enrich` →
  `correct-terms`). Exits non-zero if any feed, episode or step failed, after
  doing all the work it could. GPU host: `--gpu-env` if it passes
  `remote-check`, else `--fallback-gpu-env`. `scripts/podcasts_library_daily.sh`
  is now a short wrapper that injects the API keys and calls it.
- `media-asr enrich`, `correct-terms` and `rediarize` replace
  `scripts/enrich_podcasts.py`, `correct_terms.py` and `rediarize_merged.py`;
  on-disk formats are unchanged.
- Shows, languages and host hints live in `$MEDIA_ASR_DATA/library.toml`
  instead of hard-coded lists; see `docs/examples/podcast-library.toml`.
- One DeepSeek client (`media_asr/llm.py`) for all steps; the model comes from
  `MEDIA_ASR_LLM_MODEL` (was `MEDIA_ASR_ENRICH_MODEL`).
- `correct-terms`: `max_tokens` 3000 → 8000, and an answer cut off at the limit
  splits the chunk in two instead of failing the episode. Chinese pairs must
  sound alike: toneless pinyin similarity of at least 0.4 for every character
  pair, one exception allowed in three or more syllables (巨星智能 → 具身智能
  is kept, 余昊 → 宇树 rejected); English pairs need a spelling similarity of
  0.55 (was 0.5). `named_base.json` (written by `enrich`) is required.
- `run --feed-contains` limits a run to one feed, `run --ids` processes exactly
  the given items; the same title in the same feed is the same episode; direct
  downloads are retried 3 times.
- Optional extras `library` (wordfreq, jieba, pypinyin).
- `scripts/`: All-In speaker-naming tools (pyannote diarization of existing
  transcripts, LLM speaker maps, voiceprint verification) and the All-In daily
  script `podcast_daily.sh`.

## 0.01 — 2026-10-19

First public release.

- Local CPU transcription with faster-whisper.
- YouTube / Bilibili / direct audio / RSS / local file ingestion.
- Markdown, SRT, TXT, and JSON transcripts.
- SQLite FTS5 search with timestamps and optional speaker labels.
- Remote GPU backend over SSH:
  - `asr` mode: local download/ffmpeg/render, remote Whisper.
  - `full` mode: remote download/ffmpeg/ASR/render.
- Optional speaker diarization:
  - sherpa-onnx (offline).
  - pyannote.audio 3.1 (gated HuggingFace models).
- Word-level timestamp alignment for diarized transcripts.
- Optional summaries via Antigravity or DeepSeek.
- Simplified Chinese normalization.
