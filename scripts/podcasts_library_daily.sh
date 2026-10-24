#!/usr/bin/env bash
# Daily update of the podcast library in data-podcasts/ (Avibe task); see `media-asr library-update -h`.
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export MEDIA_ASR_DATA="$ROOT/data-podcasts" PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
exec vibe vault run --env DEEPSEEK_API_KEY=DEEPSEEK_API_KEY.value --env HF_TOKEN=HF_TOKEN.value -- "$ROOT/media-asr" \
  library-update --jobs 3 --gpu-env "$HOME/.config/media-asr/remote-5900x.env" --fallback-gpu-env "$HOME/.config/media-asr/remote.env"
