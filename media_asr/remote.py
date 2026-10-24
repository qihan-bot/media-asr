from __future__ import annotations

import getpass
import json
import os
import posixpath
import re
import stat
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Callable

from .transcribe import TranscriptResult, TranscriptSegment

# The remote workers' GPU lock, in the work directory (scripts/remote_worker.py).
GPU_LOCK = "gpu.lock"


class RemoteError(RuntimeError):
    pass


def _clean_text(text: str, *, simplify: bool = True) -> str:
    text = text.strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、])", r"\1", text)
    if simplify:
        try:
            import zhconv
        except ImportError:
            pass
        else:
            text = zhconv.convert(text, "zh-cn")
    return text.strip()


class RemoteGPU:
    """Run OpenAI Whisper on a remote Windows/GPU host over SSH."""

    def __init__(
        self,
        host: str,
        *,
        user: str = "",
        port: int = 22,
        workdir: str = "media-asr-work",
        python: str = r".venv\Scripts\python.exe",
        shell: str = "cmd",
        model: str = "large-v3-turbo",
        model_dir: str = "models",
        language: str = "zh",
        mode: str = "asr",
        diarize: bool = False,
        num_speakers: int = -1,
        diarization_segmentation: str = "models/diarization/segmentation.onnx",
        diarization_embedding: str = "models/diarization/speaker_embedding.onnx",
        diarization_threads: int = 8,
        diarization_threshold: float = 0.5,
        diarization_backend: str = "sherpa",
        pyannote_pipeline: str = "pyannote/speaker-diarization-3.1",
        password: str | None = None,
        key_filename: str | None = None,
        timeout: int = 7200,
        condition_on_previous_text: bool = False,
        auto_add_host_key: bool = False,
        keep_remote: bool = False,
    ):
        self.host = host
        self.user = user or getpass.getuser()
        self.port = port
        self.workdir = workdir
        self.python = python
        self.shell = shell
        self.model = model
        self.model_dir = model_dir
        self.language = language
        self.mode = mode
        self.diarize = diarize
        self.num_speakers = num_speakers
        self.diarization_segmentation = diarization_segmentation
        self.diarization_embedding = diarization_embedding
        self.diarization_threads = diarization_threads
        self.diarization_threshold = diarization_threshold
        self.diarization_backend = diarization_backend
        self.pyannote_pipeline = pyannote_pipeline
        self.hf_token = os.environ.get("HF_TOKEN")
        self.password = password or os.environ.get("MEDIA_ASR_REMOTE_PASSWORD")
        self.key_filename = key_filename or os.environ.get("MEDIA_ASR_REMOTE_KEY")
        if not self.key_filename:
            candidate = Path.home() / ".ssh" / "id_ed25519"
            if candidate.exists():
                self.key_filename = str(candidate)
        self.timeout = timeout
        self.condition_on_previous_text = condition_on_previous_text
        self.auto_add_host_key = auto_add_host_key
        self.keep_remote = keep_remote
        self._client = None
        self._sftp = None
        self._home: str | None = None

    @classmethod
    def from_args(cls, args) -> "RemoteGPU":
        return cls(
            getattr(args, "remote_host", None) or os.environ.get("MEDIA_ASR_REMOTE_HOST", ""),
            user=getattr(args, "remote_user", None) or os.environ.get("MEDIA_ASR_REMOTE_USER", ""),
            port=int(getattr(args, "remote_port", None) or os.environ.get("MEDIA_ASR_REMOTE_PORT", 22)),
            workdir=getattr(args, "remote_workdir", None)
            or os.environ.get("MEDIA_ASR_REMOTE_WORKDIR", "media-asr-work"),
            python=getattr(args, "remote_python", None)
            or os.environ.get("MEDIA_ASR_REMOTE_PYTHON", r".venv\Scripts\python.exe"),
            shell=getattr(args, "remote_shell", None)
            or os.environ.get("MEDIA_ASR_REMOTE_SHELL", "cmd"),
            model=getattr(args, "remote_model", None)
            or os.environ.get("MEDIA_ASR_REMOTE_MODEL", "large-v3-turbo"),
            model_dir=getattr(args, "remote_model_dir", None)
            or os.environ.get("MEDIA_ASR_REMOTE_MODEL_DIR", "models"),
            language=getattr(args, "language", "") or os.environ.get("MEDIA_ASR_REMOTE_LANGUAGE", "zh"),
            mode=getattr(args, "remote_mode", None)
            or os.environ.get("MEDIA_ASR_REMOTE_MODE", "asr"),
            diarize=bool(getattr(args, "diarize", False)),
            num_speakers=int(getattr(args, "num_speakers", -1)),
            diarization_segmentation=getattr(
                args, "diarization_segmentation", "models/diarization/segmentation.onnx"
            ),
            diarization_embedding=getattr(
                args, "diarization_embedding", "models/diarization/speaker_embedding.onnx"
            ),
            diarization_threads=int(getattr(args, "diarization_threads", 8)),
            diarization_threshold=float(getattr(args, "diarization_threshold", 0.5)),
            diarization_backend=getattr(args, "diarization_backend", None)
            or os.environ.get("MEDIA_ASR_DIARIZATION_BACKEND", "sherpa"),
            pyannote_pipeline=getattr(args, "pyannote_pipeline", None)
            or os.environ.get("MEDIA_ASR_PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1"),
            key_filename=getattr(args, "remote_key", None),
            timeout=int(getattr(args, "remote_timeout", None) or os.environ.get("MEDIA_ASR_REMOTE_TIMEOUT", 7200)),
            condition_on_previous_text=bool(getattr(args, "remote_condition_on_previous_text", False)),
            auto_add_host_key=bool(getattr(args, "remote_auto_add_host_key", False)),
            keep_remote=bool(getattr(args, "remote_keep", False)),
        )

    # ---- connection ----------------------------------------------------
    def connect(self) -> None:
        if self._client is not None:
            return
        try:
            import paramiko
        except ImportError as exc:
            raise RemoteError("paramiko is required for the remote backend: pip install paramiko") from exc

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        known_hosts = Path.home() / ".ssh/known_hosts"
        if known_hosts.exists():
            client.load_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(
            paramiko.AutoAddPolicy() if self.auto_add_host_key else paramiko.RejectPolicy()
        )
        kwargs = {
            "hostname": self.host,
            "port": self.port,
            "username": self.user,
            "timeout": 20,
            "banner_timeout": 20,
            "auth_timeout": 20,
            "allow_agent": True,
            "look_for_keys": True,
        }
        if self.password:
            kwargs["password"] = self.password
        if self.key_filename:
            kwargs["key_filename"] = self.key_filename
        try:
            client.connect(**kwargs)
        except Exception as exc:
            raise RemoteError(
                "cannot connect to remote GPU host "
                f"{self.user}@{self.host}:{self.port}: {exc}\n"
                "Provide a password via MEDIA_ASR_REMOTE_PASSWORD (for example by "
                "wrapping the command with `vibe vault run --env ...`), or set up "
                "an SSH key / agent."
            ) from exc
        self._client = client
        self._sftp = client.open_sftp()
        self._home = self._sftp.normalize(".")

    def close(self) -> None:
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._home = None

    def __enter__(self) -> "RemoteGPU":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _base(self) -> str:
        if not self.host:
            raise RemoteError("set --remote-host or MEDIA_ASR_REMOTE_HOST")
        if not self._home:
            raise RemoteError("not connected")
        workdir = self.workdir.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", workdir):
            return workdir.rstrip("/")
        if workdir.startswith("/"):
            return workdir.rstrip("/")
        return posixpath.join(self._home, workdir).rstrip("/")

    def _path(self, *parts: str) -> str:
        return posixpath.join(self._base(), *parts)

    def _cd_prefix(self) -> str:
        workdir = self.workdir.replace("\\", "/")
        if self.shell == "sh":
            target = workdir if workdir.startswith("/") else f"$HOME/{workdir}"
            return f'cd "{target}" && '
        target = workdir if re.match(r"^[A-Za-z]:/", workdir) else f"%USERPROFILE%\\{workdir}"
        return f'cd /d "{target}" && '

    # ---- checks --------------------------------------------------------
    def check(self, progress: Callable[[str], None] | None = None) -> dict[str, object]:
        self.connect()
        commands = [
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
            self._cd_prefix()
            + f'{self.python} -c "import torch, whisper; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"',
            self._cd_prefix()
            + f'{self.python} -c "import yt_dlp, feedparser, httpx; print(yt_dlp.version.__version__, feedparser.__version__, httpx.__version__)"',
        ]
        gpu = self._exec(commands[0], timeout=60, progress=progress).strip()
        python = self._exec(commands[1], timeout=60, progress=progress).strip()
        deps = self._exec(commands[2], timeout=60, progress=progress).strip()
        return {
            "host": f"{self.user}@{self.host}:{self.port}",
            "base": self._base(),
            "gpu": gpu,
            "python": python,
            "deps": deps,
        }

    # ---- transcription -------------------------------------------------
    def transcribe(
        self,
        audio_path: Path,
        *,
        item_id: str,
        language: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
        beam: int = 5,
        traditional: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        self.connect()
        model = model or self.model
        language = language or self.language
        job_rel = posixpath.join("jobs", re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id))
        job_dir = self._path(job_rel)
        self._mkdirs(job_dir)

        remote_audio = posixpath.join(job_rel, "audio.wav")
        remote_prompt = posixpath.join(job_rel, "prompt.txt")
        remote_output = posixpath.join(job_rel, "result.json")

        log = progress or (lambda _: None)
        success = False
        local_result = audio_path.parent / "remote_result.json"
        try:
            log(f"uploading audio to {self.host}:{remote_audio}")
            self._upload(
                audio_path,
                posixpath.join(self._base(), remote_audio),
                progress=log,
            )
            with self._sftp.open(posixpath.join(self._base(), remote_prompt), "w") as handle:
                handle.write(prompt or "")
            self._upload_script(job_rel=job_rel)

            command = (
                self._cd_prefix()
                + f'{self.python} "{job_rel}/transcribe_openai_whisper.py" '
                f'--audio "{remote_audio}" --output "{remote_output}" '
                f'--model "{model}" --language "{language}" '
                f'--download-root "{self.model_dir}" '
                f'--beam {int(beam)} '
                f'--prompt-file "{remote_prompt}" '
                f'--gpu-lock "{GPU_LOCK}"'
                + (
                    " --condition-on-previous-text"
                    if self.condition_on_previous_text
                    else ""
                )
                + (" --keep-traditional" if traditional else "")
            )
            log(f"remote transcribing with {model} on {self.host} ...")
            self._exec(command, timeout=self.timeout, progress=log)
            log("downloading remote result")
            self._sftp.get(posixpath.join(self._base(), remote_output), str(local_result))
            payload = json.loads(local_result.read_text(encoding="utf-8"))
            success = True
        finally:
            if not success or not self.keep_remote:
                self._cleanup(
                    job_dir,
                    [
                        "audio.wav",
                        "prompt.txt",
                        "result.json",
                        "remote_result.json",
                        "transcribe_openai_whisper.py",
                    ],
                )
            else:
                log(f"remote job kept at {job_dir}")

        segments = [
            TranscriptSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=_clean_text(
                    str(segment.get("text", "")), simplify=not traditional
                ),
            )
            for segment in payload.get("segments", [])
            if str(segment.get("text", "")).strip()
        ]
        label = f"{payload.get('model', 'openai-whisper ' + model)} @ {self.host}"
        log(
            f"remote done: {len(segments)} segments, "
            f"{payload.get('elapsed_seconds', '?')}s"
        )
        return TranscriptResult(
            segments=segments,
            language=payload.get("language") or language,
            language_probability=None,
            duration=payload.get("duration"),
            model_label=label,
        )

    def run_full(
        self,
        item: dict,
        item_dir: Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
        keep_audio: bool = False,
        diarize: bool | None = None,
        num_speakers: int | None = None,
        beam: int = 5,
        traditional: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        """Run download, ffmpeg, Whisper, and rendering on the remote host."""
        self.connect()
        log = progress or (lambda _: None)
        model = model or self.model
        item_id = str(item["id"])
        job_rel = posixpath.join("jobs", re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id))
        job_dir = self._path(job_rel)

        source = str(item["source"])
        source_type = str(item.get("source_type") or "audio")
        job_source = source
        try:
            self._mkdirs(job_dir)
            if Path(source).expanduser().is_file():
                local_source = Path(source).expanduser()
                remote_name = f"source{local_source.suffix.lower() or '.bin'}"
                log(f"uploading source file {local_source.name} to {self.host}")
                self._upload(
                    local_source,
                    posixpath.join(self._base(), job_rel, remote_name),
                    progress=log,
                )
                job_source = posixpath.join(job_rel, remote_name)
                source_type = "file"
        except BaseException:
            self._rmtree(job_dir)
            raise

        meta = {}
        if item.get("meta_json"):
            try:
                meta = json.loads(item["meta_json"])
            except json.JSONDecodeError:
                meta = {}

        job = {
            "id": item_id,
            "title": item.get("title"),
            "source": job_source,
            "original_source": source,
            "source_type": source_type,
            "audio_url": meta.get("audio_url"),
            "page_url": meta.get("page_url"),
            "published_at": item.get("published_at"),
            "feed_url": item.get("feed_url"),
            "language": language or self.language,
            "prompt": prompt or "",
            "model": model,
            "download_root": self.model_dir,
            "beam": int(beam),
            "traditional": bool(traditional),
            "keep_audio": bool(keep_audio),
            "diarize": self.diarize if diarize is None else bool(diarize),
            "num_speakers": self.num_speakers if num_speakers is None else int(num_speakers),
            "diarization_segmentation": self.diarization_segmentation,
            "diarization_embedding": self.diarization_embedding,
            "diarization_threads": self.diarization_threads,
            "diarization_threshold": self.diarization_threshold,
            "diarization_backend": self.diarization_backend,
            "pyannote_pipeline": self.pyannote_pipeline,
            "condition_on_previous_text": self.condition_on_previous_text,
            "temperature": None,
            "job_dir": job_rel,
        }
        # Each job runs its own copy of the code, so jobs can run side by side.
        command = (
            self._cd_prefix()
            + f'{self.python} "{job_rel}/remote_worker.py" --job "{job_rel}/job.json"'
        )
        local_bundle = item_dir / "bundle.zip"
        token_rel = None
        success = False
        try:
            job_path = posixpath.join(self._base(), job_rel, "job.json")
            if self.diarization_backend == "pyannote" and self.hf_token:
                token_rel = posixpath.join(job_rel, "hf_token.txt")
                with self._sftp.open(posixpath.join(self._base(), token_rel), "w") as handle:
                    handle.write(self.hf_token + "\n")
                job["hf_token_file"] = token_rel
            with self._sftp.open(job_path, "w") as handle:
                handle.write(json.dumps(job, ensure_ascii=False, indent=2))

            self._upload_script("remote_worker.py", job_rel)
            self._upload_package(job_rel)

            log(f"remote full pipeline on {self.host} ...")
            self._exec(command, timeout=self.timeout, progress=log)
            log("downloading bundle")
            self._sftp.get(posixpath.join(self._base(), job_rel, "bundle.zip"), str(local_bundle))
            with zipfile.ZipFile(local_bundle) as archive:
                for member in archive.infolist():
                    name = Path(member.filename).name
                    if not name or name.startswith(".."):
                        continue
                    with archive.open(member) as src, (item_dir / name).open("wb") as dst:
                        dst.write(src.read())
            transcript = json.loads((item_dir / "transcript.json").read_text(encoding="utf-8"))
            run = json.loads((item_dir / "run.json").read_text(encoding="utf-8"))
            local_bundle.unlink(missing_ok=True)
            remote_meta = json.loads((item_dir / "meta.json").read_text(encoding="utf-8"))
            success = True
        finally:
            if token_rel:
                try:
                    self._sftp.remove(posixpath.join(self._base(), token_rel))
                except Exception:
                    pass
            if not success or not self.keep_remote:
                self._rmtree(job_dir)
            else:
                log(f"remote job kept at {job_dir}")

        segments = [
            TranscriptSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment.get("text", "")),
                speaker=segment.get("speaker"),
            )
            for segment in transcript.get("segments", [])
            if str(segment.get("text", "")).strip()
        ]
        audio = item_dir / "audio.wav"
        log(
            f"remote full done: {len(segments)} segments, "
            f"{run.get('elapsed_seconds', '?')}s"
        )
        return {
            "segments": segments,
            "language": transcript.get("language") or job.get("language"),
            "duration": transcript.get("duration") or run.get("duration"),
            "model_label": run.get("model") or f"openai-whisper {model} @ {self.host}",
            "audio_path": str(audio) if keep_audio and audio.exists() else None,
            "run": run,
            "meta": remote_meta,
        }

    def run_embed(
        self,
        item: dict,
        segments: list[dict],
        fingerprint: str,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> tuple:
        """Embed each segment of a diarized episode on the remote host (download
        and ffmpeg there; the speaker-embedding model under the GPU lock).

        Returns ``(emb, seg, samples, fingerprint)``, see
        ``voiceprint.label_voices``. Each call has its own job directory,
        ``jobs/<id>.voiceprint-<random>``: two runs may embed the same episode at
        once (the voiceprint step takes the item lock only to commit)."""
        import numpy as np

        if Path(str(item["source"])).expanduser().is_file():
            raise RemoteError("voiceprints need a downloadable source, not a local file")
        self.connect()
        log = progress or (lambda _: None)
        item_id = str(item["id"])
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id) + f".voiceprint-{uuid.uuid4().hex[:8]}"
        job_rel = posixpath.join("jobs", name)
        job_dir = self._path(job_rel)
        try:
            meta = json.loads(item.get("meta_json") or "{}")
        except json.JSONDecodeError:
            meta = {}
        job = {
            "kind": "embed",
            "id": item_id,
            "source": str(item["source"]),
            "source_type": str(item.get("source_type") or "audio"),
            "audio_url": meta.get("audio_url"),
            "page_url": meta.get("page_url"),
            "segments": [[float(s["start"]), float(s["end"])] for s in segments],
            "fp": fingerprint,
            "job_dir": job_rel,
        }
        local = Path(tempfile.mkdtemp(prefix="media-asr-voiceprint-")) / "embeddings.npz"
        success = False
        try:
            self._mkdirs(job_dir)
            with self._sftp.open(posixpath.join(job_dir, "job.json"), "w") as handle:
                handle.write(json.dumps(job, ensure_ascii=False))
            self._upload_script("remote_worker.py", job_rel)
            self._upload_package(job_rel)
            log(f"remote voiceprint on {self.host} ...")
            self._exec(
                self._cd_prefix()
                + f'{self.python} "{job_rel}/remote_worker.py" --job "{job_rel}/job.json"',
                timeout=self.timeout,
                progress=log,
            )
            self._sftp.get(posixpath.join(job_dir, "embeddings.npz"), str(local))
            with np.load(local) as z:
                result = (z["emb"], z["seg"], z["samples"], str(z["fp"]))
            success = True
            return result
        finally:
            if not success or not self.keep_remote:
                self._rmtree(job_dir)
            local.unlink(missing_ok=True)
            local.parent.rmdir()

    def _upload_script(
        self, name: str = "transcribe_openai_whisper.py", job_rel: str | None = None
    ) -> None:
        """Upload a helper script to the work directory, or to the job's directory."""
        package_root = Path(__file__).resolve().parent
        candidates = [
            package_root.parent / "scripts" / name,
            package_root / "remote_scripts" / name,
        ]
        script = next((path for path in candidates if path.exists()), None)
        if script is None:
            raise RemoteError(f"missing remote helper script: {name}")
        self._upload(script, self._path(*filter(None, [job_rel, name])), progress=None)

    def _upload_package(self, job_rel: str | None = None) -> None:
        """Upload media_asr.zip to the work directory, or to the job's directory."""
        package_root = Path(__file__).resolve().parent
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        handle.close()
        archive_path = Path(handle.name)
        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(package_root.glob("*.py")):
                    archive.write(path, f"media_asr/{path.name}")
            self._upload(
                archive_path, self._path(*filter(None, [job_rel, "media_asr.zip"])), progress=None
            )
        finally:
            archive_path.unlink(missing_ok=True)

    def _rmtree(self, path: str) -> None:
        try:
            for entry in self._sftp.listdir_attr(path):
                child = posixpath.join(path, entry.filename)
                if stat.S_ISDIR(entry.st_mode):
                    self._rmtree(child)
                else:
                    self._sftp.remove(child)
            self._sftp.rmdir(path)
        except IOError:
            pass

    def _upload(
        self,
        local: Path,
        remote: str,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        last = [0]

        def callback(done: int, total: int) -> None:
            if not progress:
                return
            if done - last[0] >= 20 * 1024 * 1024 or done == total:
                last[0] = done
                progress(f"upload {done / 1e6:.1f}/{total / 1e6:.1f} MB")

        self._sftp.put(str(local), remote, callback=callback)

    def _mkdirs(self, path: str) -> None:
        parts = [part for part in path.strip("/").split("/") if part]
        current = "/" if path.startswith("/") else ""
        # Do not try to mkdir the drive component (C:) on Windows.
        for index, part in enumerate(parts):
            current = posixpath.join(current, part) if current else part
            if index == 0 and re.fullmatch(r"[A-Za-z]:", part):
                continue
            try:
                self._sftp.stat(current)
            except IOError:
                try:
                    self._sftp.mkdir(current)
                except IOError:
                    # Jobs side by side create the shared jobs/ at once; SFTP
                    # reports an existing path as a plain failure.
                    if not self._is_dir(current):
                        raise

    def _is_dir(self, path: str) -> bool:
        try:
            mode = self._sftp.stat(path).st_mode
        except IOError:
            return False
        return mode is None or stat.S_ISDIR(mode)

    def _cleanup(self, job_dir: str, names: list[str]) -> None:
        for name in names:
            try:
                self._sftp.remove(posixpath.join(job_dir, name))
            except IOError:
                pass
        try:
            self._sftp.rmdir(job_dir)
        except Exception:
            pass

    def _exec(
        self,
        command: str,
        *,
        timeout: int,
        progress: Callable[[str], None] | None = None,
    ) -> str:
        if not self._client:
            raise RemoteError("not connected")
        transport = self._client.get_transport()
        if transport is None:
            raise RemoteError("SSH transport is not available")
        transport.set_keepalive(30)
        channel = transport.open_session()
        channel.settimeout(1.0)
        channel.exec_command(command)
        started = time.time()
        last_report = started
        chunks: list[bytes] = []
        buffer = b""
        while True:
            if channel.recv_ready():
                data = channel.recv(4096)
                if data:
                    chunks.append(data)
                    buffer += data
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        text = line.decode("utf-8", "ignore").rstrip("\r")
                        if text and progress:
                            progress(text)
            if channel.recv_stderr_ready():
                data = channel.recv_stderr(4096)
                if data:
                    chunks.append(data)
                    buffer += data
            if channel.exit_status_ready() and not channel.recv_ready():
                break
            if time.time() - started > timeout:
                channel.close()
                raise RemoteError(f"remote command timed out after {timeout}s: {command}")
            if progress and time.time() - last_report >= 30:
                progress(f"remote still running ({int(time.time() - started)}s)")
                last_report = time.time()
            time.sleep(0.4)
        if buffer and progress:
            progress(buffer.decode("utf-8", "ignore").rstrip("\r"))
        status = channel.recv_exit_status()
        text = b"".join(chunks).decode("utf-8", "ignore")
        if status != 0:
            raise RemoteError(f"remote command failed ({status}):\n{text[-3000:]}")
        return text