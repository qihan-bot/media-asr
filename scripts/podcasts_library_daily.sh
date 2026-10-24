#!/usr/bin/env bash
# Daily refresh of the multi-show podcast library (data-podcasts/):
# new episodes -> remote GPU transcript + pyannote diarization (per-show
# language) -> re-diarize episodes whose largest label merged several people
# -> DeepSeek punctuation + speaker names -> audited term correction.
#
# GPU: 5900x-128g (RTX 3080 Ti) when reachable, else x18-max-5090 (shared with
# the All-In daily). Every step is idempotent, so a failed run can be re-run as
# is. Exits non-zero if any feed, episode or enrichment step failed, after
# doing all the work it could.
set -uo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export MEDIA_ASR_DATA="$ROOT/data-podcasts"
export PYTHONPATH="$ROOT"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$ROOT"

# feed URL substring -> Whisper language
declare -A SHOWS=(
  [feeds.fireside.fm/sv101]=zh        # 硅谷101
  [feeds.fireside.fm/latetalk]=zh     # 晚点聊 LateTalk
  [dk4yh3pkpjp3]=zh                   # 张小珺Jùn｜商业访谈录
  [q4dnrcam497p]=zh                   # 邵亦波｜一波一会
  [q9a6lueucj6a]=zh                   # 开始连接 LinkStart
  [feed/podcast/1084089]=en           # Latent Space
  [rss.buzzsprout.com/2630998]=en     # Automated with Brian Heater
  [destinations/2292818]=en           # Robot Talk
)
status=0

echo "== $(date -Is) start"
GPU_ENV="$HOME/.config/media-asr/remote-5900x.env"
if MEDIA_ASR_REMOTE_ENV="$GPU_ENV" timeout 90 ./media-asr-remote remote-check >/dev/null 2>&1; then
  export MEDIA_ASR_REMOTE_ENV="$GPU_ENV"
  echo "gpu: 5900x-128g (3080 Ti)"
else
  unset MEDIA_ASR_REMOTE_ENV  # default ~/.config/media-asr/remote.env
  echo "gpu: x18-max-5090 (5900x-128g unreachable)"
fi

echo "== refresh feeds"
for attempt in 1 2 3; do
  out="$(./media-asr refresh --latest 5 2>&1)"
  echo "$out"
  grep -q '^feed failed' <<<"$out" || break
  if [[ $attempt == 3 ]]; then echo "WARN: some feeds failed 3 times" >&2; status=1; fi
  sleep $((attempt * 30))
done

echo "== transcribe"
for feed in "${!SHOWS[@]}"; do
  vibe vault run --env HF_TOKEN=HF_TOKEN.value -- \
    ./media-asr-remote run --retry-failed --feed-contains "$feed" --language "${SHOWS[$feed]}" \
    --diarize --diarization-backend pyannote 2>&1 | grep -avE 'frames/s|still running' || true
done
failed="$(./media-asr list --status failed --limit 1000 | tail -n +2)"
if [[ -n $failed ]]; then echo "FAILED episodes:"; echo "$failed"; status=1; fi

echo "== re-diarize episodes where one label merged several people"
vibe vault run --env DEEPSEEK_API_KEY=DEEPSEEK_API_KEY.value --env HF_TOKEN=HF_TOKEN.value -- \
  ./media-asr rediarize || status=1

echo "== punctuation + speaker names"
vibe vault run --env DEEPSEEK_API_KEY=DEEPSEEK_API_KEY.value -- \
  ./media-asr enrich --workers 4 || status=1

echo "== term correction (audited replacement pairs, see terms.json)"
vibe vault run --env DEEPSEEK_API_KEY=DEEPSEEK_API_KEY.value -- \
  ./media-asr correct-terms --workers 4 || status=1

echo "== $(date -Is) end status=$status"
./media-asr stats
exit $status
