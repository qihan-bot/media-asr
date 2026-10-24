from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

from media_asr import cli
from media_asr import library as L
from media_asr.db import Segment, Store
from media_asr.util import file_lock

EXAMPLE = Path(__file__).resolve().parents[1] / "docs/examples/podcast-library.toml"

# Hard-coded in scripts/podcasts_library_daily.sh (feed URL substring -> language)
# and scripts/enrich_podcasts.py (SHOW_HOSTS) before library.toml.
OLD_SHOWS = {
    "feeds.fireside.fm/sv101": "zh",
    "feeds.fireside.fm/latetalk": "zh",
    "dk4yh3pkpjp3": "zh",
    "q4dnrcam497p": "zh",
    "q9a6lueucj6a": "zh",
    "feed/podcast/1084089": "en",
    "rss.buzzsprout.com/2630998": "en",
    "destinations/2292818": "en",
}
OLD_HOSTS = {
    "feed/podcast/1084089": (
        "swyx (Shawn Wang; editor of Latent Space, organizer of the AI Engineer "
        "conference, AIE) and Alessio Fanelli (partner at Kernel Labs)"
    ),
    "rss.buzzsprout.com/2630998": "Brian Heater",
    "destinations/2292818": "Claire Asher",
}


def test_example_config_has_the_old_hard_coded_shows() -> None:
    shows = L.load_shows(EXAMPLE)
    assert len(shows) == len(OLD_SHOWS)
    for fragment, language in OLD_SHOWS.items():
        (show,) = [s for s in shows if fragment in s.feed]
        assert show.language == language
        assert L.host_hints(shows, show.feed) == OLD_HOSTS.get(fragment, "")


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "library.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no [[show]] entries"),
        ("[[show]\n", "library.toml"),
        ('title = "x"\n', "expected only [[show]] tables"),
        ('[show]\nfeed = "https://a/rss"\nlanguage = "zh"\n', "expected only [[show]] tables"),
        ('[[show]]\nlanguage = "zh"\n', "feed (the RSS URL) is required"),
        ('[[show]]\nfeed = "https://a/rss"\nlanguage = "fr"\n', "language must be one of zh, en"),
        (
            '[[show]]\nfeed = "https://a/rss"\nlanguage = "zh"\nhost = "A"\n',
            "unknown keys ['host']",
        ),
        ('[[show]]\nfeed = "https://a/rss"\nlanguage = "zh"\nhosts = ["A"]\n', "must be strings"),
        (
            '[[show]]\nfeed = "https://a/feed"\nlanguage = "zh"\n'
            '[[show]]\nfeed = "https://a/feed2"\nlanguage = "en"\n',
            "feed https://a/feed also matches https://a/feed2",
        ),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(L.LibraryError, match=re.escape(message)):
        L.load_shows(write(tmp_path, text))


def test_missing_config(tmp_path: Path) -> None:
    with pytest.raises(L.LibraryError, match="podcast-library.toml"):
        L.load_shows(tmp_path / "library.toml")
    assert L.load_shows(tmp_path / "library.toml", missing_ok=True) == []


def test_show_fields_and_matching(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[show]]\nname = " A "\nfeed = " https://a/rss "\nlanguage = "zh"\nhosts = "Ann"\n'
        '[[show]]\nfeed = "https://b/rss"\nlanguage = "en"\n',
    )
    shows = L.load_shows(path)
    assert shows == [
        L.Show(feed="https://a/rss", language="zh", name="A", hosts="Ann"),
        L.Show(feed="https://b/rss", language="en"),
    ]
    assert L.host_hints(shows, "https://a/rss") == "Ann"
    assert L.host_hints(shows, "https://b/rss") == ""
    assert L.host_hints(shows, None) == ""


class FakeShell:
    """Records the update's commands instead of running them; ``run <id>`` for
    an episode's transcription. Each command takes ``delay`` seconds."""

    def __init__(self, ok_envs=("a.env", "b.env"), codes=None, refresh_failures=0, delay=0.0):
        self.ok_envs = set(ok_envs)
        self.codes = codes or {}
        self.refresh_failures = refresh_failures
        self.delay = delay
        self.calls: list[tuple[str, str | None]] = []
        self.languages: dict[str, str] = {}
        self.guard = threading.Lock()
        self.running = self.most_running = 0

    def check(self, cmd, env, timeout):
        self.calls.append(("remote-check " + env["MEDIA_ASR_REMOTE_ENV"], None))
        assert cmd == [str(L.REMOTE), "remote-check"] and timeout == 90
        return env["MEDIA_ASR_REMOTE_ENV"] in self.ok_envs

    def stream(self, cmd, env, drop=None, echo=True):
        if cmd[0] == str(L.REMOTE):
            assert drop is L.PROGRESS
            if cmd[1] == "run":
                assert cmd[2] == "--ids" and cmd[4] == "--language" and not echo
                assert cmd[6:] == ["--diarize", "--diarization-backend", "pyannote"]
                name = f"run {cmd[3]}"
                self.languages[cmd[3]] = cmd[5]
            else:
                name = cmd[1]
        else:
            assert cmd[:3] == L.MEDIA_ASR
            name = cmd[3]
        assert env["MEDIA_ASR_DATA"] and env["PYTHONUNBUFFERED"] == "1"
        with self.guard:
            self.calls.append((name, env.get("MEDIA_ASR_REMOTE_ENV")))
            self.running += 1
            self.most_running = max(self.most_running, self.running)
        time.sleep(self.delay)
        with self.guard:
            self.running -= 1
        if name == "refresh" and self.refresh_failures:
            self.refresh_failures -= 1
            return 0, "Show A: 0 new / 5 checked\nfeed failed: https://b/rss: timeout\n"
        code = self.codes.get(name, 0)
        if isinstance(code, Exception):
            raise code
        return code, f"output of {name}\nline 2\n"

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


SHOWS = [
    L.Show(feed="https://a/rss", language="zh", name="A"),
    L.Show(feed="https://b/rss", language="en", name="B"),
]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "media.db")
    store.add_feed("https://a/rss", "A", 3)
    store.add_feed("https://b/rss", "B", 3)
    yield store
    store.close()


def episodes(store: Store, feed: str, count: int, *, first: int = 0) -> list[str]:
    """``count`` pending episodes of ``feed``."""
    return [
        store.add_item(
            source=f"{feed}/{n}.mp3", source_type="podcast", title=f"Ep {n}", feed_url=feed
        )[0]
        for n in range(first, first + count)
    ]


def test_update_runs_the_steps_in_order(store: Store) -> None:
    (a,), (b,) = episodes(store, "https://a/rss", 1), episodes(store, "https://b/rss", 1)
    shell, sleeps = FakeShell(), []
    status = L.update(
        store, SHOWS, gpu_env="a.env", fallback_gpu_env="b.env", shell=shell, sleep=sleeps.append
    )
    assert status == 0
    assert shell.calls == [
        ("remote-check a.env", None),
        ("refresh", "a.env"),
        (f"run {a}", "a.env"),
        (f"run {b}", "a.env"),
        ("rediarize", "a.env"),
        ("enrich", "a.env"),
        ("correct-terms", "a.env"),
        ("stats", "a.env"),
    ]
    assert shell.languages == {a: "zh", b: "en"}
    assert sleeps == []


def test_update_falls_back_and_keeps_going_after_failures(store: Store, capsys) -> None:
    item, _ = store.add_item(
        source="https://a/1.mp3", source_type="podcast", title="Ep 1", feed_url="https://a/rss"
    )
    store.set_status(item, "failed", error="download failed")
    shell = FakeShell(ok_envs={"b.env"}, codes={f"run {item}": 1, "enrich": 1}, refresh_failures=3)
    sleeps: list[int] = []
    status = L.update(
        store, SHOWS, gpu_env="a.env", fallback_gpu_env="b.env", shell=shell, sleep=sleeps.append
    )
    assert status == 1
    assert shell.names() == [
        "remote-check a.env",
        "remote-check b.env",
        "refresh",
        "refresh",
        "refresh",
        f"run {item}",
        "rediarize",
        "enrich",
        "correct-terms",
        "stats",
    ]
    assert {env for _, env in shell.calls[2:]} == {"b.env"}
    assert sleeps == [30, 60]
    out = capsys.readouterr().out
    assert "gpu: a.env failed remote-check\ngpu: b.env, 1 episode(s) at a time\n" in out
    assert "WARN: some feeds failed 3 times" in out
    assert f"[1/1] FAILED (exit 1) in 0m00s: {item} Ep 1\n    output of run {item}\n" in out
    assert f"FAILED episodes:\n{item}  Ep 1\n" in out
    assert "end status=1" in out


def test_update_retries_refresh_until_it_passes(store: Store) -> None:
    (item,) = episodes(store, "https://a/rss", 1)
    shell, sleeps = FakeShell(refresh_failures=1), []
    assert L.update(store, SHOWS, shell=shell, sleep=sleeps.append) == 0
    assert shell.names()[:3] == ["refresh", "refresh", f"run {item}"]
    assert sleeps == [30]


def test_jobs_transcribe_each_episode_once_in_its_show_language(store: Store, capsys) -> None:
    zh = episodes(store, "https://a/rss", 4)
    en = episodes(store, "https://b/rss", 3)
    shell = FakeShell(delay=0.05)
    assert L.update(store, SHOWS, gpu_env="a.env", jobs=3, shell=shell, sleep=None) == 0
    runs = [name.split()[1] for name in shell.names() if name.startswith("run ")]
    assert sorted(runs) == sorted(zh + en)  # every episode, none twice
    assert shell.languages == {**dict.fromkeys(zh, "zh"), **dict.fromkeys(en, "en")}
    assert shell.most_running == 3
    out = capsys.readouterr().out
    assert "gpu: a.env, 3 episode(s) at a time\n" in out
    assert "7 episode(s), up to 3 at a time\n" in out and "transcribed 7, failed 0, in " in out
    for item in zh + en:  # one block per episode: its result line, then its output
        assert re.search(rf"\] done in 0m00s: {item} Ep \d\n    output of run {item}\n", out)


def test_one_failed_episode_does_not_stop_the_others(store: Store, capsys) -> None:
    items = episodes(store, "https://a/rss", 5)
    shell = FakeShell(
        codes={f"run {items[1]}": 1, f"run {items[3]}": OSError("media-asr-remote: not found")},
        delay=0.02,
    )
    assert L.update(store, SHOWS, gpu_env="a.env", jobs=2, shell=shell, sleep=None) == 1
    assert sorted(n.split()[1] for n in shell.names() if n.startswith("run ")) == sorted(items)
    assert shell.names()[-4:] == ["rediarize", "enrich", "correct-terms", "stats"]
    out = capsys.readouterr().out
    assert f"FAILED (exit 1) in 0m00s: {items[1]} Ep 1\n" in out
    assert f"FAILED (exit 1) in 0m00s: {items[3]} Ep 3\n    media-asr-remote: not found\n" in out
    assert "transcribed 3, failed 2, in " in out and "end status=1" in out


def test_the_fallback_host_gets_one_episode_at_a_time(store: Store) -> None:
    items = episodes(store, "https://a/rss", 4)
    shell = FakeShell(ok_envs={"b.env"}, delay=0.02)
    status = L.update(
        store, SHOWS, gpu_env="a.env", fallback_gpu_env="b.env", jobs=3, shell=shell, sleep=None
    )
    assert status == 0
    assert [n for n in shell.names() if n.startswith("run ")] == [f"run {i}" for i in items]
    assert shell.most_running == 1


def test_without_a_gpu_host_only_the_model_steps_run(store: Store, capsys) -> None:
    episodes(store, "https://a/rss", 2)
    shell = FakeShell(ok_envs=())
    status = L.update(
        store, SHOWS, gpu_env="a.env", fallback_gpu_env="b.env", jobs=3, shell=shell, sleep=None
    )
    assert status == 1
    assert shell.names() == [
        "remote-check a.env",
        "remote-check b.env",
        "refresh",
        "enrich",
        "correct-terms",
        "stats",
    ]
    out = capsys.readouterr().out
    assert "error: no GPU host passed remote-check" in out
    assert "== transcribe\nskipped: no GPU host (2 episode(s) waiting)\n" in out
    assert "re-diarize episodes where one label merged several people\nskipped: no GPU host" in out


def test_the_preferred_host_alone_has_no_fallback(store: Store) -> None:
    episodes(store, "https://a/rss", 1)
    shell = FakeShell(ok_envs=())
    assert L.update(store, SHOWS, gpu_env="a.env", shell=shell, sleep=None) == 1
    assert shell.names()[:2] == ["remote-check a.env", "refresh"]
    assert not [n for n in shell.names() if n.startswith("run ") or n == "rediarize"]


def test_an_episode_two_shows_match_is_transcribed_once(store: Store) -> None:
    (item,) = episodes(store, "https://a/rss", 1)
    shows = [*SHOWS, L.Show(feed="a/rss", language="en")]
    shell = FakeShell()
    assert L.update(store, shows, shell=shell, sleep=None) == 0
    assert [n for n in shell.names() if n.startswith("run ")] == [f"run {item}"]
    assert shell.languages == {item: "zh"}


def test_each_show_queues_at_most_run_limit_episodes(store: Store, monkeypatch) -> None:
    monkeypatch.setattr(L, "RUN_LIMIT", 2)
    episodes(store, "https://a/rss", 3)
    episodes(store, "https://b/rss", 1)
    shell = FakeShell()
    assert L.update(store, SHOWS, shell=shell, sleep=None) == 0
    assert len([n for n in shell.names() if n.startswith("run ")]) == 3


@pytest.mark.parametrize("failing", ["refresh", "rediarize", "correct-terms"])
def test_update_exit_code_follows_any_failed_step(store: Store, failing: str) -> None:
    shell = FakeShell(codes={failing: 2})
    status = L.update(store, SHOWS, shell=shell, sleep=lambda seconds: None)
    assert status == 1
    assert [name for name, _ in shell.calls][-1] == "stats"


def test_update_skips_shows_that_are_not_subscribed(tmp_path: Path, capsys) -> None:
    store = Store(tmp_path / "media.db")
    store.add_feed("https://a/rss", "A", 3)
    (a,) = episodes(store, "https://a/rss", 1)
    episodes(store, "https://b/rss", 1)  # a show that is not subscribed
    shell = FakeShell()
    assert L.update(store, SHOWS, shell=shell, sleep=lambda seconds: None) == 1
    assert [n for n in shell.names() if n.startswith("run ")] == [f"run {a}"]
    assert "WARN: B is not subscribed: media-asr add https://b/rss" in capsys.readouterr().out
    store.close()


def test_failed_items_listed_only_when_failed(store: Store, capsys) -> None:
    item, _ = store.add_item(
        source="https://b/1.mp3", source_type="podcast", title="Ep", feed_url="https://b/rss"
    )
    store.save_result(
        item,
        segments=[Segment(0.0, 1.0, "hi")],
        duration=1.0,
        language="en",
        model="m",
        transcript_path="t.md",
    )
    assert L.update(store, SHOWS, shell=FakeShell(), sleep=lambda seconds: None) == 0
    assert "FAILED" not in capsys.readouterr().out


def test_update_runs_once_per_library(store: Store, capsys) -> None:
    shell = FakeShell()
    with file_lock(store.path.parent / ".library-update.lock"):
        assert L.update(store, SHOWS, shell=shell, sleep=lambda seconds: None) == 1
    assert shell.calls == []
    assert "another library-update holds" in capsys.readouterr().out
    assert L.update(store, SHOWS, shell=shell, sleep=lambda seconds: None) == 0


def test_library_update_passes_jobs_and_workers(monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(L, "load_shows", lambda: SHOWS)
    monkeypatch.setattr(L, "update", lambda store, shows, **options: seen.update(options) or 0)
    args = cli.build_parser().parse_args(["library-update", "--jobs", "3", "--gpu-env", "a.env"])
    assert cli.cmd_library_update(None, args) == 0
    assert (seen["jobs"], seen["workers"], seen["gpu_env"]) == (3, 4, "a.env")
    assert cli.build_parser().parse_args(["library-update"]).jobs == 1
    assert cli.build_parser().parse_args(["rediarize"]).workers == 1


@pytest.mark.parametrize(
    "argv", [["library-update", "--jobs", "0"], ["rediarize", "--workers", "0"]]
)
def test_job_and_worker_counts_must_be_at_least_one(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)
