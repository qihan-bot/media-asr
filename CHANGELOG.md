# Changelog

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
