"""All-In through ``library-update``, every step a real ``media-asr`` process.

Only the edges are fakes (tests/allin_fakes.py): the feed, the GPU host
(transcription and embeddings) and DeepSeek.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from media_asr import library as L
from media_asr import voiceprint as V
from media_asr.db import Store

TESTS = Path(__file__).resolve().parent
BOOT = (
    "import sys, allin_fakes; allin_fakes.install(); "
    "from media_asr.cli import main; raise SystemExit(main(sys.argv[1:]))"
)


class Processes(L.Shell):
    """Runs each step as a media-asr process with the fakes installed, in place
    of media-asr-remote (whose env file is not needed) or python -m media_asr."""

    def __init__(self, extra_env: dict) -> None:
        self.extra_env = extra_env

    def check(self, cmd, env, timeout):
        return True

    def stream(self, cmd, env, drop=None, echo=True):
        args = cmd[1:] if cmd[0] == str(L.REMOTE) else cmd[3:]
        return super().stream(
            [sys.executable, "-c", BOOT, *args], {**env, **self.extra_env}, drop, echo
        )


def test_all_in_goes_through_the_library_named_by_voice(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data = tmp_path / "data"
    store = Store(data / "media.db")
    store.add_feed("https://allinchamathjason.libsyn.com/rss", "All-In", 5)
    show = L.Show(
        feed="https://allinchamathjason.libsyn.com/rss",
        language="en",
        name="All-In",
        hosts="A Host and B Host",
        voiceprints="voiceprints/all-in.npz",
    )
    V.save_references(data / show.voiceprints, {"A Host": np.eye(3)[0], "B Host": np.eye(3)[1]}, {})
    (data / "library.toml").write_text(
        '[[show]]\nname = "All-In"\nfeed = "https://allinchamathjason.libsyn.com/rss"\n'
        'language = "en"\nhosts = "A Host and B Host"\nvoiceprints = "voiceprints/all-in.npz"\n'
    )
    log = tmp_path / "fakes.log"
    monkeypatch.setattr(L, "DATA_DIR", data)
    shell = Processes(
        {
            "PYTHONPATH": f"{TESTS}:{L.ROOT}",
            "ALLIN_FAKES_LOG": str(log),
            "MEDIA_ASR_BACKEND": "remote",
            "MEDIA_ASR_REMOTE_MODE": "full",
            "MEDIA_ASR_REMOTE_HOST": "gpu.example",
            "LAN_MODEL_API_KEY": "fake",
            "MEDIA_ASR_LLM_MODEL": "fake-llm",
        }
    )
    status = L.update(store, [show], gpu_env="gpu.env", jobs=2, shell=shell, sleep=None)
    out = capsys.readouterr().out
    assert status == 0, out
    steps = [line for line in out.splitlines() if line.startswith("== ") and "+00:00" not in line]
    assert steps == [
        "== refresh feeds",
        "== transcribe",
        "== re-diarize episodes where one label merged several people",
        "== voiceprints (shows with reference voices)",
        "== punctuation + speaker names",
        "== term correction (audited replacement pairs, see terms.json)",
    ]
    (row,) = store.list_items()
    item = data / "items" / row["id"]
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [kind for kind, _ in calls] == ["run_full", "run_embed", "speakers"]
    # The model was asked only about the label the voices did not settle.
    assert calls[2][1].endswith("SPEAKER_00: A Host\nSPEAKER_01: B Host\nLabel only: SPEAKER_02\n")
    marker = json.loads((item / "voiceprint.json").read_text())
    assert {k: v["voice"] for k, v in marker["speakers"].items()} == {
        "SPEAKER_00": "match",
        "SPEAKER_01": "match",
        "SPEAKER_02": "no-match",
    }
    speakers = json.loads((item / "enrich.json").read_text())["speakers"]
    assert {k: (v["name"], v["role"], v.get("method")) for k, v in speakers.items()} == {
        "SPEAKER_00": ("A Host", "host", "voiceprint"),
        "SPEAKER_01": ("B Host", "host", "voiceprint"),
        "SPEAKER_02": ("Carol Guest", "guest", None),
    }
    named = json.loads((item / "transcript_named.json").read_text())["segments"]
    assert [(s["speaker"], s["text"]) for s in named] == [
        ("A Host", "Welcome back to the show everybody."),
        ("B Host", "Great to be here this week."),
        ("Carol Guest", "Thanks for having me on."),
    ]
    assert (item / "terms.json").exists()
    assert [r["speaker"] for r in store.segments_for(row["id"])] == [
        "A Host",
        "B Host",
        "Carol Guest",
    ]
    # A second run finds nothing to do.
    log.unlink()
    assert L.update(store, [show], gpu_env="gpu.env", jobs=2, shell=shell, sleep=None) == 0
    assert not log.exists()
    store.close()


def test_the_bootstrap_runs_the_real_cli(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", BOOT, "--version"],
        env={
            "PYTHONPATH": f"{TESTS}:{L.ROOT}",
            "PATH": "/usr/bin:/bin",
            "MEDIA_ASR_DATA": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and proc.stdout.startswith("media-asr ")
