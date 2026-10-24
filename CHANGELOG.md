# Changelog

## 0.0.4 — 2026-10-24

All-In is the ninth show of the podcast library; its voiceprint naming became a
library step for any show with reference voices.

- `library.toml`: optional `voiceprints = "<file>"` per show (reference voices,
  relative to the data directory). `docs/examples/podcast-library.toml` adds
  All-In (`en`; hosts Jason Calacanis, Chamath Palihapitiya, David Sacks, David
  Friedberg; `voiceprints/all-in.npz`), transcribed with diarization in one pass
  like the other shows.
- `media-asr voiceprint` (run as `media-asr-remote voiceprint`; `library-update`
  runs it between rediarize and enrich when a show has voiceprints): the GPU
  host embeds each diarized segment with pyannote 3.1's speaker-embedding model,
  under the GPU lock, in a job of its own (`"kind": "embed"`, a fresh remote
  directory per run, so two runs on one episode keep apart); each speaker
  label's mean voice is scored against the references with the rules of the
  All-In `voiceprint_relabel.py`, and `voiceprint.json` records per label match /
  no-match / insufficient with the closest reference, score, margin, seconds and
  mean voice, plus the SHA-256 of the transcript it judged. It is a step marker:
  `run` drops it, the step redoes a changed transcript, writing it drops
  `enrich.json` and `terms.json`.
- `enrich`: names matched by voice are kept (`"method": "voiceprint"`; host if
  the show's hosts list the name, else guest) and the model is asked only about
  the other labels, with the known ones listed; a model name that is a reference
  voice the label does not sound like is rejected (`Speaker N`). Its commit
  checks the `voiceprint.json` it used along with the transcript, so names made
  before a voiceprint landed are not committed. Episodes without
  `voiceprint.json` are named exactly as before.
- `scripts/migrate_allin.py` moves the All-In archive (`data/`) into the
  library, keeping ids: diarized transcripts, database rows and segments, the
  feed, `voiceprint.json` computed from the archived embeddings, and the
  reference voices built from the archive. Dry run by default, `--apply`
  idempotent; the archive is only read, and overlapping archive and library
  directories are refused. Each episode is checked again under its lock and a
  database write lock before anything is written, so one the library got after
  the survey is reported as a conflict, not overwritten. On a copy of the live
  directories: 108 episodes, 19,029 segments, all 534 voice decisions as in the
  archive's final speaker maps, 60.6% of the speaking time named by voice; with
  the archive's own model names replayed, 99.7% of the speaking time gets the
  same speaker name (the rest is placeholder wording).
- Retired, as their work is now the library's: `scripts/podcast_daily.sh`,
  `diarize_existing.py`, `map_speakers.py`, `repair_speakers.py`,
  `voiceprint_relabel.py`, `embed_speakers.py`, `embed_speakers_worker.py`,
  `speaker_common.py`, `normalize_names.py`, `run_diarize_2026.sh`.
- Tests no longer leave an empty `items/` lock directory in the real data
  directory (`test_run_filter`).

## 0.0.3 — 2026-10-24

One GPU host, kept busy: several remote jobs share it and take turns on its GPU.

- Remote workers hold an flock on `gpu.lock` in the remote work directory from
  loading Whisper to the end of pyannote diarization (the ASR-only script, when
  given `--gpu-lock`: loading and transcribing). Download, ffmpeg, sherpa-onnx
  (CPU), rendering and bundling run outside it. A no-op where `fcntl` is
  unavailable (Windows). Output files are unchanged.
- Each remote job uploads and runs its own copy of the worker and the package
  in `jobs/<id>/`. They used to share the work directory root, so a starting
  job could rewrite code another job was importing. Jobs that start together
  on a fresh host may all create the shared `jobs/` directory; a `mkdir` that
  fails because the directory now exists counts as success.
- `library-update --jobs N` (default 1): the shows' pending and failed episodes
  are queued (oldest first, at most 100 per show, each once) and transcribed N
  at a time, each with its show's language in its own
  `media-asr-remote run --ids`. A failed episode fails only itself; each
  episode's output is logged as one block, and the step ends with a count.
  Exit codes are unchanged.
- GPU host: `--gpu-env` with `--jobs`; else `--fallback-gpu-env`, one episode
  at a time, now only if it passes `remote-check`. The fallback is optional:
  when no given host passes, transcription and `rediarize` are skipped and the
  run exits 1 (`enrich` and `correct-terms` still run).
- `rediarize --workers N` (default 1) runs the model checks N at a time; the
  verdicts are acted on one episode at a time, in order, so re-runs still go
  to the GPU one after another. `library-update --workers` applies to it too.
- `scripts/podcasts_library_daily.sh` runs `--jobs 3` on the 5900x, with the
  5090 laptop (`remote.env`) as fallback. `scripts/podcast_daily.sh` (All-In)
  now picks its host the same way (`MEDIA_ASR_GPU_ENV`,
  `MEDIA_ASR_FALLBACK_GPU_ENV`; `MEDIA_ASR_REMOTE_ENV` forces one) instead of
  always using `remote.env`.
- Measured on the RTX 3080 Ti host with 3 episodes (158 min of audio):
  transcription 625 s with `--jobs 1`, 580 s with `--jobs 3`; GPU busy 84% →
  90% of the time, peak memory 7226 MiB in both (see docs/remote-gpu.md).

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
- `correct-terms`: accepted pairs replace text literally in one pass over the
  uncorrected transcript (longest match first; text that already reads as a
  right form is left alone; no replacement is matched again, which wrote
  "Rhoda AI AI" before), and `terms.json` counts the replacements actually
  made. `max_tokens` 3000 → 8000, and an answer cut off at the limit
  splits the chunk in two instead of failing the episode. Chinese pairs must
  sound alike: toneless pinyin similarity of at least 0.4 for every character
  pair, one exception allowed in three or more syllables (巨星智能 → 具身智能
  is kept, 余昊 → 宇树 rejected); English pairs need a spelling similarity of
  0.55 (was 0.5). `named_base.json` (written by `enrich`) is required.
- Each library step writes an episode under a lock on `items/<id>/.lock`, after
  checking that its input has not changed and after removing its markers, so an
  interrupted write is redone; `run` removes the later steps' markers before it
  writes a new transcript; one `library-update` per library at a time;
  `rediarize` checks, re-runs (in-process; run it as `media-asr-remote rediarize`)
  and marks an episode under one hold of its lock, and keeps going when one
  episode fails.
- `run --feed-contains` limits a run to one feed, `run --ids` processes exactly
  the given items; the same title in the same feed is the same episode (an
  episode not yet transcribed takes the feed's current download URL); direct
  downloads are retried 3 times.
- Optional extras `library` (wordfreq, jieba, pypinyin).
- `scripts/`: All-In speaker-naming tools (pyannote diarization of existing
  transcripts, LLM speaker maps, voiceprint verification, using the configured
  remote shell and port) and the All-In daily script `podcast_daily.sh`.

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
