from __future__ import annotations

from types import SimpleNamespace

from media_asr.remote import RemoteGPU, _clean_text


def test_remote_paths_relative_workdir() -> None:
    remote = RemoteGPU("example", workdir="media-asr-test")
    remote._home = "/C:/Users/example"
    assert remote._base() == "/C:/Users/example/media-asr-test"
    assert remote._path("jobs", "abc", "audio.wav") == (
        "/C:/Users/example/media-asr-test/jobs/abc/audio.wav"
    )


def test_remote_paths_absolute_workdir() -> None:
    remote = RemoteGPU("example", workdir="D:/work/media-asr")
    remote._home = "/C:/Users/example"
    assert remote._base() == "D:/work/media-asr"


def test_remote_from_args() -> None:
    args = SimpleNamespace(
        remote_host="gpu.example",
        remote_user="user",
        remote_port=2222,
        remote_workdir="work",
        remote_python="python.exe",
        remote_model="medium",
        remote_model_dir="models",
        language="zh",
        remote_key=None,
        remote_timeout=60,
        remote_auto_add_host_key=False,
        remote_keep=False,
        remote_condition_on_previous_text=True,
    )
    remote = RemoteGPU.from_args(args)
    assert remote.host == "gpu.example"
    assert remote.port == 2222
    assert remote.model == "medium"
    assert remote.condition_on_previous_text is True


def test_clean_text_simplifies_and_removes_cjk_spaces() -> None:
    assert _clean_text("這是 一個 測試") == "这是一个测试"