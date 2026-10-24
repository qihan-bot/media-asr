"""The daily wrapper scripts, run for real against stub commands."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Records its name, the GPU env file it sees and its arguments, one line per call.
STUB = """#!/bin/sh
echo "$(basename "$0") env=${MEDIA_ASR_REMOTE_ENV:-} host=${MEDIA_ASR_REMOTE_HOST:-} $*" \
  >> "$STUB_LOG"
"""
# remote-check passes only for the env file named in $STUB_GPU_OK.
REMOTE_STUB = (
    STUB
    + """[ "$1" != remote-check ] || [ "${MEDIA_ASR_REMOTE_ENV:-}" = "${STUB_GPU_OK:-}" ]
"""
)


def write(path: Path, text: str, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    for name in ("vibe", "python3", "curl"):
        write(home / ".local/bin" / name, STUB + ("echo 404\n" if name == "curl" else ""), 0o755)
    write(home / ".config/qihan/internal-token", "token\n")
    write(home / ".config/media-asr/remote-5900x.env", "MEDIA_ASR_REMOTE_HOST=5900x\n")
    write(home / ".config/media-asr/remote.env", "MEDIA_ASR_REMOTE_HOST=x18\n")
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
        "vibe env= host= vault run --env DEEPSEEK_API_KEY=DEEPSEEK_API_KEY.value "
        f"--env HF_TOKEN=HF_TOKEN.value -- {ROOT}/media-asr library-update --jobs 3 "
        f"--gpu-env {config}/remote-5900x.env --fallback-gpu-env {config}/remote.env"
    ]


@pytest.fixture
def all_in(tmp_path: Path) -> Path:
    """A source root with the real podcast_daily.sh and a stub media-asr-remote."""
    root = tmp_path / "root"
    shutil.copy(ROOT / "scripts/podcast_daily.sh", write(root / "scripts/podcast_daily.sh", ""))
    write(root / "media-asr-remote", REMOTE_STUB, 0o755)
    return root / "scripts/podcast_daily.sh"


def test_all_in_daily_uses_the_5900x_when_it_passes_remote_check(all_in: Path, home) -> None:
    gpu = str(home / ".config/media-asr/remote-5900x.env")
    out, calls = run(all_in, home, STUB_GPU_OK=gpu)
    assert out.startswith(f"gpu: {gpu}\n== refresh feed\n")
    assert calls[0] == f"media-asr-remote env={gpu} host= remote-check"
    # Every later step sees the 5900x: the wrapper through the env file, the
    # scripts through the variables loaded from it.
    assert {line.split(" host=")[0].split(" env=")[1] for line in calls[1:]} == {gpu}
    assert all(" host=5900x " in line for line in calls[1:])
    assert [line.split()[0] for line in calls[1:]] == [
        "media-asr-remote",  # refresh
        "media-asr-remote",  # run
        "vibe",  # diarize_existing.py
        "vibe",  # map_speakers.py
        "vibe",  # repair_speakers.py
        "python3",  # embed_speakers.py
        "python3",  # voiceprint_relabel.py
        "curl",
        "python3",  # PENDING list
    ]


def test_all_in_daily_falls_back_to_remote_env(all_in: Path, home: Path) -> None:
    gpu, fallback = (
        str(home / ".config/media-asr" / name) for name in ("remote-5900x.env", "remote.env")
    )
    out, calls = run(all_in, home)  # the 5900x fails remote-check
    assert out.startswith(f"gpu: {gpu} failed remote-check\ngpu: {fallback}\n")
    assert all(f"env={fallback} host=x18 " in line for line in calls[1:])


def test_all_in_daily_honors_an_explicit_env_file(all_in: Path, home: Path) -> None:
    forced = str(home / ".config/media-asr/remote.env")
    out, calls = run(all_in, home, MEDIA_ASR_REMOTE_ENV=forced)
    assert out.startswith(f"gpu: {forced}\n")
    assert not [line for line in calls if "remote-check" in line]
    assert all(f"env={forced} host=x18 " in line for line in calls)
