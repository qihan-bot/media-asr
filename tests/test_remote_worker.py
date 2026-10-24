"""The remote workers' GPU lock: one job at a time loads and runs models on the
host, while downloads, ffmpeg, rendering and bundling run beside other jobs."""

from __future__ import annotations

import fcntl
import io
import json
import stat
import sys
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest

from media_asr import fetch, render, util
from media_asr.remote import RemoteGPU
from media_asr.remote_scripts import remote_worker as W
from media_asr.remote_scripts import transcribe_openai_whisper as T

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_remote_scripts_match_the_source() -> None:
    # A source checkout uploads scripts/, an installed package remote_scripts/.
    for name in ("remote_worker.py", "transcribe_openai_whisper.py"):
        packaged = ROOT / "media_asr" / "remote_scripts" / name
        assert packaged.read_bytes() == (ROOT / "scripts" / name).read_bytes(), name


def lock_held(path: Path) -> bool:
    """Does another open of ``path`` hold its flock?"""
    if not path.exists():
        return False
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = "0+fake"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda index: "Fake GPU"
    )
    return torch


WHISPER_RESULT = {
    "language": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": " hello world",
            "words": [
                {"start": 0.0, "end": 1.0, "word": " hello"},
                {"start": 1.0, "end": 2.0, "word": " world"},
            ],
        }
    ],
}


class Phases:
    """Stand-ins for the worker's phases. ``record(phase)`` is called in each;
    every phase but the model calls sleeps ``delay`` seconds."""

    def __init__(self, monkeypatch, record, delay: float = 0.0) -> None:
        self.record, self.delay = record, delay
        monkeypatch.setitem(sys.modules, "torch", fake_torch())
        monkeypatch.setitem(sys.modules, "whisper", types.SimpleNamespace(load_model=self.load))
        monkeypatch.setattr(fetch, "download", self.download)
        monkeypatch.setattr(fetch, "normalize_audio", self.normalize)
        monkeypatch.setattr(util, "ffprobe_duration", lambda path: 2.0)
        real_render = render.render_all

        def render_all(**kwargs):
            self.record("render")
            return real_render(**kwargs)

        monkeypatch.setattr(render, "render_all", render_all)
        monkeypatch.setattr(W, "run_pyannote", lambda wav, job: self.diarize("pyannote"))
        monkeypatch.setattr(W, "run_diarization", lambda wav, job: self.diarize("sherpa"))

        class ZipFile(zipfile.ZipFile):
            def __init__(inner, file, mode="r", *args, **kwargs):  # noqa: N805
                if mode == "w":
                    self.record("bundle")
                super().__init__(file, mode, *args, **kwargs)

        monkeypatch.setattr(
            W, "zipfile", types.SimpleNamespace(ZipFile=ZipFile, ZIP_DEFLATED=zipfile.ZIP_DEFLATED)
        )

    def download(self, item, job_dir, progress=None):
        self.record("download")
        time.sleep(self.delay)
        path = Path(job_dir) / "source.mp3"
        path.write_bytes(b"mp3")
        return path, False

    def normalize(self, source, wav):
        self.record("ffmpeg")
        Path(wav).write_bytes(b"wav")

    def load(self, name, device, download_root):
        self.record("load model")
        phases = self

        class Model:
            def transcribe(self, audio, **kwargs):
                phases.record("transcribe")
                time.sleep(phases.delay)
                return json.loads(json.dumps(WHISPER_RESULT))

        return Model()

    def diarize(self, backend: str):
        self.record(f"diarize {backend}")
        time.sleep(self.delay)
        self.record(f"diarized {backend}")
        return [(0.0, 2.0, 0)]


def write_job(root: Path, job_id: str, backend: str = "pyannote") -> str:
    job_dir = root / "jobs" / job_id
    job_dir.mkdir(parents=True)
    job = {
        "id": job_id,
        "title": job_id,
        "source": f"https://a/{job_id}.mp3",
        "source_type": "podcast",
        "audio_url": f"https://a/{job_id}.mp3",
        "language": "en",
        "model": "large-v3-turbo",
        "beam": 5,
        "diarize": True,
        "diarization_backend": backend,
        "job_dir": f"jobs/{job_id}",
    }
    (job_dir / "job.json").write_text(json.dumps(job))
    return f"jobs/{job_id}/job.json"


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # the worker prepends the work directory
    monkeypatch.setattr(sys, "path", list(sys.path))
    return tmp_path


@pytest.mark.parametrize("backend", ["pyannote", "sherpa"])
def test_the_gpu_lock_covers_only_loading_and_running_the_models(
    workdir: Path, monkeypatch, backend: str
) -> None:
    seen: list[tuple[str, bool]] = []
    Phases(monkeypatch, lambda phase: seen.append((phase, lock_held(workdir / W.GPU_LOCK))))
    assert W.main(["--job", write_job(workdir, "ep1", backend)]) == 0
    assert seen == [
        ("download", False),
        ("ffmpeg", False),
        ("load model", True),
        ("transcribe", True),
        # pyannote runs on the GPU; sherpa-onnx on the CPU, beside other jobs
        (f"diarize {backend}", backend == "pyannote"),
        (f"diarized {backend}", backend == "pyannote"),
        ("render", False),
        ("bundle", False),
    ]
    assert not lock_held(workdir / W.GPU_LOCK)
    with zipfile.ZipFile(workdir / "jobs/ep1/bundle.zip") as bundle:
        assert {"transcript.json", "run.json", "meta.json"} <= set(bundle.namelist())
        run = json.loads(bundle.read("run.json"))
    assert run["speakers"] == ["SPEAKER_00"] and run["device"] == "Fake GPU"


def test_jobs_side_by_side_take_turns_on_the_gpu(workdir: Path, monkeypatch, capsys) -> None:
    jobs = [f"ep{n}" for n in range(3)]
    guard = threading.Lock()
    state = {"gpu": None, "downloading": 0, "most downloading": 0, "gpu phases": 0}
    overlaps: list[str] = []

    def record(phase: str) -> None:
        job = threading.current_thread().name
        with guard:
            if phase == "download":
                state["downloading"] += 1
                state["most downloading"] = max(state["most downloading"], state["downloading"])
            elif phase == "ffmpeg":
                state["downloading"] -= 1
            elif phase == "load model":  # a GPU phase starts
                if state["gpu"] is not None:
                    overlaps.append(f"{job} loads while {state['gpu']} is on the GPU")
                state["gpu"] = job
                state["gpu phases"] += 1
            elif phase in ("transcribe", "diarize pyannote", "diarized pyannote"):
                if state["gpu"] != job:
                    overlaps.append(f"{job} runs {phase} while {state['gpu']} is on the GPU")
                if phase == "diarized pyannote":  # the GPU phase ends with diarization
                    state["gpu"] = None

    Phases(monkeypatch, record, delay=0.2)
    paths = {job: write_job(workdir, job) for job in jobs}
    codes: dict[str, int] = {}
    threads = [
        threading.Thread(
            target=lambda job=job: codes.__setitem__(job, W.main(["--job", paths[job]])),
            name=job,
        )
        for job in jobs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert codes == {job: 0 for job in jobs}
    assert overlaps == []
    assert state["gpu phases"] == 3
    assert state["most downloading"] == 3  # the downloads ran side by side
    assert capsys.readouterr().out.count("worker: waiting for the GPU") == 2
    for job in jobs:
        assert (workdir / "jobs" / job / "bundle.zip").exists()


def test_without_fcntl_the_lock_is_a_no_op(workdir: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "fcntl", None)  # import fcntl -> ImportError (Windows)
    with W.gpu_lock(workdir / W.GPU_LOCK), T.gpu_lock(str(workdir / "asr.lock")):
        pass
    assert not (workdir / W.GPU_LOCK).exists() and not (workdir / "asr.lock").exists()


def test_asr_only_script_takes_the_lock_it_is_given(workdir: Path, monkeypatch) -> None:
    lock = workdir / "gpu.lock"
    seen: list[tuple[str, bool]] = []

    class Model:
        def transcribe(self, audio, **kwargs):
            seen.append(("transcribe", lock_held(lock)))
            return json.loads(json.dumps(WHISPER_RESULT))

    def load_model(name, device, download_root):
        seen.append(("load model", lock_held(lock)))
        return Model()

    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "whisper", types.SimpleNamespace(load_model=load_model))
    (workdir / "audio.wav").write_bytes(b"wav")
    argv = ["--audio", "audio.wav", "--output", "out.json", "--language", "en"]
    assert T.main([*argv, "--gpu-lock", "gpu.lock"]) == 0
    assert seen == [("load model", True), ("transcribe", True)]
    assert not lock_held(lock)
    seen.clear()
    assert T.main(argv) == 0  # run by hand: no lock file
    assert seen == [("load model", False), ("transcribe", False)]
    assert json.loads((workdir / "out.json").read_text())["segments"][0]["text"] == "hello world"


class FakeSFTP:
    """An in-memory remote file system."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = set()

    def stat(self, path):
        if path in self.dirs:
            return types.SimpleNamespace(st_mode=stat.S_IFDIR)
        if path in self.files:
            return types.SimpleNamespace(st_mode=stat.S_IFREG)
        raise IOError(path)

    def mkdir(self, path):
        self.dirs.add(path)

    def open(self, path, mode="r"):
        sftp = self

        class Handle(io.StringIO):
            def close(inner):  # noqa: N805
                sftp.files[path] = inner.getvalue().encode()
                super().close()

        return Handle()

    def put(self, local, remote, callback=None):
        self.files[remote] = Path(local).read_bytes()

    def get(self, remote, local):
        Path(local).write_bytes(self.files[remote])

    def remove(self, path):
        if path not in self.files:
            raise IOError(path)
        del self.files[path]

    def rmdir(self, path):
        self.dirs.discard(path)

    def listdir_attr(self, path):
        children = {
            name[len(path) + 1 :].split("/")[0]
            for name in [*self.files, *self.dirs]
            if name.startswith(path + "/")
        }
        return [
            types.SimpleNamespace(
                filename=child,
                st_mode=stat.S_IFDIR if f"{path}/{child}" in self.dirs else stat.S_IFREG,
            )
            for child in sorted(children)
        ]


def test_each_full_job_runs_its_own_copy_of_the_code(tmp_path: Path, monkeypatch) -> None:
    remote = RemoteGPU("gpu", shell="sh", workdir="work", python=".venv/bin/python")
    remote._client, remote._sftp, remote._home = object(), FakeSFTP(), "/home/u"
    job = "/home/u/work/jobs/ep_1"
    commands: list[str] = []

    def fake_exec(command, *, timeout, progress=None):
        commands.append(command)
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("transcript.json", json.dumps({"segments": []}))
            archive.writestr("run.json", "{}")
            archive.writestr("meta.json", "{}")
        remote._sftp.files[f"{job}/bundle.zip"] = bundle.getvalue()
        return ""

    monkeypatch.setattr(remote, "_exec", fake_exec)
    item = {"id": "ep:1", "source": "https://a/1.mp3", "source_type": "podcast", "title": "Ep"}
    uploads: list[str] = []
    real_put = remote._sftp.put
    monkeypatch.setattr(
        remote._sftp,
        "put",
        lambda local, path, callback=None: uploads.append(path) or real_put(local, path),
    )
    remote.run_full(item, tmp_path)
    assert uploads == [f"{job}/remote_worker.py", f"{job}/media_asr.zip"]
    assert commands == [
        'cd "$HOME/work" && .venv/bin/python "jobs/ep_1/remote_worker.py" '
        '--job "jobs/ep_1/job.json"'
    ]
    assert not [path for path in remote._sftp.files if path.startswith(job)]  # cleaned up


def test_each_asr_job_runs_its_own_copy_under_the_gpu_lock(tmp_path: Path, monkeypatch) -> None:
    remote = RemoteGPU("gpu", shell="sh", workdir="work", python=".venv/bin/python")
    remote._client, remote._sftp, remote._home = object(), FakeSFTP(), "/home/u"
    job = "/home/u/work/jobs/ep_1"
    commands: list[str] = []

    def fake_exec(command, *, timeout, progress=None):
        commands.append(command)
        result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}], "language": "en"}
        remote._sftp.files[f"{job}/result.json"] = json.dumps(result).encode()
        return ""

    monkeypatch.setattr(remote, "_exec", fake_exec)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    result = remote.transcribe(audio, item_id="ep:1", language="en")
    assert [s.text for s in result.segments] == ["hi"]
    (command,) = commands
    assert command.startswith(
        'cd "$HOME/work" && .venv/bin/python "jobs/ep_1/transcribe_openai_whisper.py" '
    )
    assert '--gpu-lock "gpu.lock"' in command
    assert not [path for path in remote._sftp.files if path.startswith(job)]  # cleaned up


class RacingSFTP(FakeSFTP):
    """One remote file system for several clients, as an SFTP server behaves:
    mkdir of an existing path fails ("Failure"), and every client finds jobs/
    missing before any of them creates it (a first run on a fresh host)."""

    def __init__(self, clients: int) -> None:
        super().__init__()
        self.dirs = {"/home", "/home/u", "/home/u/work"}
        self.guard = threading.Lock()
        self.first_look = threading.Barrier(clients)

    def stat(self, path):
        with self.guard:
            missing = path not in self.dirs and path not in self.files
        if missing and path == "/home/u/work/jobs":
            self.first_look.wait(5)
            raise IOError(path)  # what this client saw, though another may create it now
        return super().stat(path)

    def mkdir(self, path):
        with self.guard:
            if path in self.dirs or path in self.files:
                raise IOError("Failure")
            self.dirs.add(path)


def test_jobs_side_by_side_on_a_fresh_host_all_create_their_job_dirs(tmp_path: Path) -> None:
    # Review 0.0.3 P2: three first runs at once; two failed creating jobs/.
    sftp = RacingSFTP(clients=3)
    results: dict[str, object] = {}

    def run(n: int) -> None:
        remote = RemoteGPU("gpu", shell="sh", workdir="work", python=".venv/bin/python")
        remote._client, remote._sftp, remote._home = object(), sftp, "/home/u"

        def fake_exec(command, *, timeout, progress=None):
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("transcript.json", json.dumps({"segments": []}))
                archive.writestr("run.json", "{}")
                archive.writestr("meta.json", "{}")
            sftp.files[f"/home/u/work/jobs/ep{n}/bundle.zip"] = bundle.getvalue()
            return ""

        remote._exec = fake_exec
        item = {"id": f"ep{n}", "source": f"https://a/{n}.mp3", "source_type": "podcast"}
        out = tmp_path / f"ep{n}"
        out.mkdir()
        try:
            results[f"ep{n}"] = remote.run_full(item, out)["segments"]
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            results[f"ep{n}"] = exc

    threads = [threading.Thread(target=run, args=(n,)) for n in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == {"ep0": [], "ep1": [], "ep2": []}
    assert "/home/u/work/jobs" in sftp.dirs


def test_a_mkdir_that_really_failed_is_still_an_error() -> None:
    class Denied(RacingSFTP):
        def mkdir(self, path):
            raise IOError("Permission denied")

    remote = RemoteGPU("gpu", shell="sh", workdir="work")
    remote._client, remote._sftp, remote._home = object(), Denied(clients=1), "/home/u"
    with pytest.raises(IOError, match="Permission denied"):
        remote._mkdirs("/home/u/work/jobs/x")
