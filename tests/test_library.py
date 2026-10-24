from __future__ import annotations

import re
from pathlib import Path

import pytest

from media_asr import library as L
from media_asr.db import Segment, Store

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
    """Records the update's commands instead of running them."""

    def __init__(self, check=True, codes=None, refresh_failures=0):
        self.check_ok = check
        self.codes = codes or {}
        self.refresh_failures = refresh_failures
        self.calls: list[tuple[str, str | None]] = []

    def check(self, cmd, env, timeout):
        self.calls.append(("remote-check " + env["MEDIA_ASR_REMOTE_ENV"], None))
        assert cmd == [str(L.REMOTE), "remote-check"] and timeout == 90
        return self.check_ok

    def stream(self, cmd, env, drop=None):
        if cmd[0] == str(L.REMOTE):
            name = " ".join(cmd[1:2] + cmd[4:5])  # "run <feed>"
            assert drop is L.PROGRESS
        else:
            assert cmd[:3] == L.MEDIA_ASR
            name = cmd[3]
        self.calls.append((name, env.get("MEDIA_ASR_REMOTE_ENV")))
        assert env["MEDIA_ASR_DATA"] and env["PYTHONUNBUFFERED"] == "1"
        if name == "refresh" and self.refresh_failures:
            self.refresh_failures -= 1
            return 0, "Show A: 0 new / 5 checked\nfeed failed: https://b/rss: timeout\n"
        return self.codes.get(name, 0), ""


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


def test_update_runs_the_steps_in_order(store: Store) -> None:
    shell, sleeps = FakeShell(), []
    status = L.update(
        store, SHOWS, gpu_env="a.env", fallback_gpu_env="b.env", shell=shell, sleep=sleeps.append
    )
    assert status == 0
    assert shell.calls == [
        ("remote-check a.env", None),
        ("refresh", "a.env"),
        ("run https://a/rss", "a.env"),
        ("run https://b/rss", "a.env"),
        ("rediarize", "a.env"),
        ("enrich", "a.env"),
        ("correct-terms", "a.env"),
        ("stats", "a.env"),
    ]
    assert sleeps == []


def test_update_falls_back_and_keeps_going_after_failures(store: Store, capsys) -> None:
    item, _ = store.add_item(
        source="https://a/1.mp3", source_type="podcast", title="Ep 1", feed_url="https://a/rss"
    )
    store.set_status(item, "failed", error="download failed")
    shell = FakeShell(check=False, codes={"run https://a/rss": 1, "enrich": 1}, refresh_failures=3)
    sleeps: list[int] = []
    status = L.update(
        store, SHOWS, gpu_env="a.env", fallback_gpu_env="b.env", shell=shell, sleep=sleeps.append
    )
    assert status == 1
    assert [name for name, _ in shell.calls] == [
        "remote-check a.env",
        "refresh",
        "refresh",
        "refresh",
        "run https://a/rss",
        "run https://b/rss",
        "rediarize",
        "enrich",
        "correct-terms",
        "stats",
    ]
    assert {env for _, env in shell.calls[1:]} == {"b.env"}
    assert sleeps == [30, 60]
    out = capsys.readouterr().out
    assert "gpu: a.env failed remote-check\ngpu: b.env\n" in out
    assert "WARN: some feeds failed 3 times" in out
    assert f"FAILED episodes:\n{item}  Ep 1\n" in out
    assert "end status=1" in out


def test_update_retries_refresh_until_it_passes(store: Store) -> None:
    shell, sleeps = FakeShell(refresh_failures=1), []
    assert L.update(store, SHOWS, shell=shell, sleep=sleeps.append) == 0
    assert [name for name, _ in shell.calls][:3] == ["refresh", "refresh", "run https://a/rss"]
    assert sleeps == [30]


@pytest.mark.parametrize("failing", ["refresh", "rediarize", "correct-terms"])
def test_update_exit_code_follows_any_failed_step(store: Store, failing: str) -> None:
    shell = FakeShell(codes={failing: 2})
    status = L.update(store, SHOWS, shell=shell, sleep=lambda seconds: None)
    assert status == 1
    assert [name for name, _ in shell.calls][-1] == "stats"


def test_update_skips_shows_that_are_not_subscribed(tmp_path: Path, capsys) -> None:
    store = Store(tmp_path / "media.db")
    store.add_feed("https://a/rss", "A", 3)
    shell = FakeShell()
    assert L.update(store, SHOWS, shell=shell, sleep=lambda seconds: None) == 1
    assert "run https://b/rss" not in [name for name, _ in shell.calls]
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
