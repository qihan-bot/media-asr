"""Stand-ins for the GPU host, DeepSeek and the RSS feed, installed into a real
``media-asr`` process by tests/test_allin_e2e.py (``python -c`` bootstrap)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FEED = "https://allinchamathjason.libsyn.com/rss"
EPISODE = "https://allin.example/e300.mp3"
# The voices of the episode's three labels: A Host, B Host and a guest nobody knows.
VOICES = {
    "SPEAKER_00": [1.0, 0.05, 0.0],
    "SPEAKER_01": [0.0, 1.0, 0.05],
    "SPEAKER_02": [0.05, 0.0, 1.0],
}
LINES = [
    ("SPEAKER_00", "welcome back to the show everybody"),
    ("SPEAKER_01", "great to be here this week"),
    ("SPEAKER_02", "thanks for having me on"),
]


def install() -> None:
    from media_asr import cli, enrich, fetch, llm, remote, terms
    from media_asr.render import render_all
    from media_asr.transcribe import TranscriptSegment

    def resolve_source(url, *, latest=1):
        assert url == FEED, url
        spec = fetch.ItemSpec(
            source=EPISODE,
            source_type="podcast",
            title="E300: All-In",
            audio_url=EPISODE,
            page_url="https://allin.example/e300",
            feed_url=FEED,
            published_at="2026-10-25T00:00:00+00:00",
        )
        return fetch.ResolveResult([spec], feed_url=FEED, feed_title="All-In")

    cli.resolve_source = resolve_source

    def run_full(self, item, item_dir, **kwargs):
        assert kwargs["diarize"] and self.diarization_backend == "pyannote"
        assert kwargs["language"] == "en"
        segments = [
            TranscriptSegment(i * 100.0, i * 100.0 + 100.0, text, label)
            for i, (label, text) in enumerate(LINES)
        ]
        render_all(
            out_dir=item_dir,
            item_id=item["id"],
            title=item["title"],
            source=item["source"],
            duration=300.0,
            language="en",
            model="fake-whisper",
            transcribed_at="t",
            segments=segments,
        )
        _record("run_full", item["id"])
        return {
            "segments": segments,
            "language": "en",
            "duration": 300.0,
            "model_label": "fake-whisper",
        }

    def run_embed(self, item, segments, fingerprint, *, progress=None):
        _record("run_embed", item["id"])
        emb = np.array([VOICES[s["speaker"]] for s in segments], np.float32)
        samples = np.array([int((s["end"] - s["start"] - 0.4) * 16000) for s in segments])
        return emb, np.arange(len(segments)), samples, fingerprint

    remote.RemoteGPU.connect = lambda self: None
    remote.RemoteGPU.close = lambda self: None
    remote.RemoteGPU.run_full = run_full
    remote.RemoteGPU.run_embed = run_embed

    class FakeLLM:
        model = "fake-llm"

        def __init__(self, key, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            pass

        def chat(self, system, user, max_tokens):
            if "label podcast speakers" in system:
                _record("speakers", user)
                return {
                    "speakers": {
                        "SPEAKER_02": {"name": "Carol Guest", "role": "guest", "confidence": "high"}
                    }
                }
            if "speech-recognition errors" in system:
                return {"fixes": []}
            if "audit speaker diarization" in system:
                return {"merged": False, "people": 3, "reason": "balanced"}
            return {"lines": [line.capitalize() + "." for line in user.split("\n")]}

    llm.LLM = FakeLLM
    enrich.load_show_notes = terms.load_show_notes = lambda db: ({}, {FEED: "All-In"})


def _record(kind: str, value: str) -> None:
    with open(Path(os.environ["ALLIN_FAKES_LOG"]), "a", encoding="utf-8") as handle:
        handle.write(json.dumps([kind, value]) + "\n")
