#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
ENV_FILE="${MEDIA_ASR_REMOTE_ENV:-$HOME/.config/media-asr/remote.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

echo "waiting for media-asr-allin-2026 to finish..."
while systemctl --user is-active --quiet media-asr-allin-2026; do
  sleep 60
done

echo "starting 2026 All-In diarization"
exec vibe vault run --env HF_TOKEN=HF_TOKEN.value -- \
  python3 "$ROOT/scripts/diarize_existing.py" \
    --feed-contains allinchamathjason \
    --label diarized \
    --language en \
    "$@"
