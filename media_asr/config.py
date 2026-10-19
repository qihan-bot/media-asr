from __future__ import annotations

import os
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent.parent
if os.environ.get("MEDIA_ASR_HOME"):
    PROJECT_ROOT = Path(os.environ["MEDIA_ASR_HOME"]).expanduser()
elif (_SOURCE_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = _SOURCE_ROOT
else:
    PROJECT_ROOT = Path.home() / ".local/share/media-asr"
DATA_DIR = Path(os.environ.get("MEDIA_ASR_DATA", PROJECT_ROOT / "data")).expanduser()
ITEMS_DIR = DATA_DIR / "items"
DB_PATH = DATA_DIR / "media.db"
MODEL_DIR = Path(os.environ.get("MEDIA_ASR_MODEL_DIR", Path.home() / ".cache/whisper")).expanduser()

DEFAULT_MODEL = os.environ.get("MEDIA_ASR_MODEL", "small")
DEFAULT_LANGUAGE = os.environ.get("MEDIA_ASR_LANGUAGE", "") or None
DEFAULT_THREADS = int(os.environ.get("MEDIA_ASR_THREADS", min(8, os.cpu_count() or 4)))
DEFAULT_COMPUTE = os.environ.get("MEDIA_ASR_COMPUTE", "int8")


def ensure_dirs() -> None:
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)