#!/usr/bin/env bash
# Daily All-In refresh: new episodes -> transcript -> diarization -> names.
#
# Every step is idempotent and skips finished items, so a failed run can be
# re-run as is. Prints the ids of All-In items that have a named transcript,
# one per line, after the marker line "READY:".
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export PYTHONPATH="$ROOT"
cd "$ROOT"

# GPU host, as library-update picks it: the 5900x if it passes remote-check,
# else remote.env (the 5090 laptop). Setting MEDIA_ASR_REMOTE_ENV forces a host.
if [[ -z "${MEDIA_ASR_REMOTE_ENV:-}" ]]; then
  GPU_ENV="${MEDIA_ASR_GPU_ENV:-$HOME/.config/media-asr/remote-5900x.env}"
  MEDIA_ASR_REMOTE_ENV="${MEDIA_ASR_FALLBACK_GPU_ENV:-$HOME/.config/media-asr/remote.env}"
  if MEDIA_ASR_REMOTE_ENV="$GPU_ENV" timeout 90 ./media-asr-remote remote-check >/dev/null 2>&1; then
    MEDIA_ASR_REMOTE_ENV="$GPU_ENV"
  else
    echo "gpu: $GPU_ENV failed remote-check"
  fi
fi
export MEDIA_ASR_REMOTE_ENV  # media-asr-remote below loads it too
echo "gpu: $MEDIA_ASR_REMOTE_ENV"
set -a
# shellcheck disable=SC1090
. "$MEDIA_ASR_REMOTE_ENV"
set +a

echo "== refresh feed"
# libsyn intermittently drops TLS connections (SSL EOF); retry before failing.
for attempt in 1 2 3; do
  if ./media-asr-remote refresh --latest 3; then break; fi
  if [[ $attempt == 3 ]]; then echo "ERROR: feed refresh failed 3 times" >&2; exit 1; fi
  sleep $((attempt * 30))
done
echo "== transcribe pending"
./media-asr-remote run --language en
echo "== diarize"
vibe vault run --env HF_TOKEN=HF_TOKEN.value -- \
  python3 scripts/diarize_existing.py --feed-contains allinchamathjason --label diarized --language en
echo "== LLM speaker names"
for step in map_speakers repair_speakers; do
  vibe vault run --env DEEPSEEK_API_KEY=DEEPSEEK_API_KEY.value -- \
    env -u NO_PROXY -u no_proxy python3 "scripts/$step.py" --workers 4
done
echo "== voiceprints"
python3 scripts/embed_speakers.py
python3 scripts/voiceprint_relabel.py --apply | tail -3
echo "PENDING:"
# Episodes ready for a 播客精读: named transcript done, published on/after the
# newest episode already in the site index, and not yet published (matched by
# episode title, so a second copy of an episode, e.g. RSS + YouTube, is skipped).
INDEX_JSON="$(mktemp)"
trap 'rm -f "$INDEX_JSON"' EXIT
code=$(curl -sS -o "$INDEX_JSON" -w '%{http_code}' -A "Mozilla/5.0 (podcast-daily)" \
  -H "x-qihan-internal: $(cat "$HOME/.config/qihan/internal-token")" \
  https://qihan.ai/docs/data/podcast-digests.json)
if [[ "$code" == 404 ]]; then echo '{"digests": []}' > "$INDEX_JSON"
elif [[ "$code" != 200 ]]; then echo "ERROR: podcast index fetch returned HTTP $code" >&2; exit 1; fi
python3 - "$INDEX_JSON" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from speaker_common import is_allin
from media_asr.config import ITEMS_DIR
digests = json.load(open(sys.argv[1]))["digests"]
published = {d["episode"].strip().casefold() for d in digests}
since = max((d["date"] for d in digests), default="0000-00-00")
for d in sorted(ITEMS_DIR.iterdir()):
    run = d / "runs" / "diarized"
    if not is_allin(d) or not (run / "transcript_named.md").exists():
        continue
    if "voiceprint" not in json.loads((run / "speaker_map.json").read_text()):
        continue
    meta = json.loads((d / "meta.json").read_text())
    if (meta.get("published_at") or "")[:10] >= since and meta["title"].strip().casefold() not in published:
        print(d.name, (meta.get("published_at") or "")[:10], meta["title"], sep="\t")
PY
