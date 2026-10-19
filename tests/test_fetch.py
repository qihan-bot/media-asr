from __future__ import annotations

from pathlib import Path

from media_asr.fetch import _feed_items, _parse_duration, resolve_source


RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>Older episode</title>
      <enclosure url="https://example.com/old.mp3" type="audio/mpeg" length="1"/>
      <itunes:duration>12:34</itunes:duration>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Newest episode</title>
      <enclosure url="https://example.com/new.mp3" type="audio/mpeg" length="1"/>
      <itunes:duration>1:02:03</itunes:duration>
      <pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def test_parse_duration() -> None:
    assert _parse_duration("46:13") == 2773
    assert _parse_duration("1:02:03") == 3723
    assert _parse_duration("3723") == 3723
    assert _parse_duration(None) is None


def test_feed_items_sorted_and_limited() -> None:
    specs, title = _feed_items(RSS, "https://example.com/feed.xml", latest=1)
    assert title == "Test Podcast"
    assert len(specs) == 1
    item = specs[0]
    assert item.title == "Newest episode"
    assert item.audio_url == "https://example.com/new.mp3"
    assert item.duration == 3723
    assert item.feed_url == "https://example.com/feed.xml"
    assert item.published_at.startswith("2024-01-02")


def test_resolve_local_file(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"not really audio")
    result = resolve_source(str(audio))
    assert len(result.specs) == 1
    assert result.specs[0].source_type == "file"
    assert result.specs[0].title == "sample"
    assert result.specs[0].extra["local"] is True