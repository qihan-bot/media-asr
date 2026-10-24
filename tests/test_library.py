from __future__ import annotations

import re
from pathlib import Path

import pytest

from media_asr import library as L

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
