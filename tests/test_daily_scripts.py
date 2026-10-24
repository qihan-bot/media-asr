"""The daily wrapper script, run for real against a stub vibe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Records its name, the env file it sees and its arguments, one line per call.
STUB = """#!/bin/sh
echo "$(basename "$0") env=${MEDIA_ASR_REMOTE_ENV:-} host=${MEDIA_ASR_REMOTE_HOST:-} $*" \
  >> "$STUB_LOG"
"""


def write(path: Path, text: str, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    write(home / ".local/bin" / "vibe", STUB, 0o755)
    return home


def run(script: Path, home: Path, **env: str) -> tuple[str, list[str]]:
    log = home / "calls.log"
    proc = subprocess.run(
        ["bash", str(script)],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin", "STUB_LOG": str(log), **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, log.read_text().splitlines() if log.exists() else []


def test_library_wrapper_runs_three_jobs_on_the_5900x_with_the_laptop_as_fallback(
    home: Path,
) -> None:
    _, calls = run(ROOT / "scripts/podcasts_library_daily.sh", home)
    config = home / ".config/media-asr"
    assert calls == [
        "vibe env= host= vault run --env LAN_MODEL_API_KEY=LAN_MODEL_API_KEY.value "
        f"--env HF_TOKEN=HF_TOKEN.value -- {ROOT}/media-asr library-update --jobs 3 --workers 2 "
        f"--gpu-env {config}/remote-5900x.env --fallback-gpu-env {config}/remote.env"
    ]
