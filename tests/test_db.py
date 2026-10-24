from __future__ import annotations

import json
from pathlib import Path

from media_asr.db import Segment, Store


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "media.db")


def test_add_and_search(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    item_id, created = store.add_item(
        source="https://example.com/ep1",
        source_type="podcast",
        title="本地语音识别测试",
    )
    assert created
    assert store.get_item(item_id)["status"] == "pending"

    store.save_result(
        item_id,
        segments=[
            Segment(0.0, 4.0, "今天我们聊本地语音识别框架。", speaker="SPEAKER_00"),
            Segment(5.0, 9.0, "Speech recognition runs on the CPU.", speaker="SPEAKER_01"),
        ],
        duration=9.0,
        language="zh",
        model="test",
        transcript_path="/tmp/transcript.md",
    )

    assert store.get_item(item_id)["status"] == "done"
    chinese = store.search("语音识别")
    assert chinese and chinese[0]["id"] == item_id
    assert chinese[0]["speaker"] == "SPEAKER_00"
    short = store.search("语音")
    assert short and short[0]["id"] == item_id
    english = store.search("recognition")
    assert english and english[0]["id"] == item_id

    stats = store.stats()
    assert stats["segments"] == 2
    assert stats["counts"]["done"] == 1

    store.delete_item(item_id)
    assert store.get_item(item_id) is None
    assert store.search("语音识别") == []
    store.close()

def test_add_item_dedupes_same_title_in_same_feed(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    feed = "https://feed.example/show/rss"
    first, created = store.add_item(
        source="https://old-host.example/ep1.mp3", source_type="podcast", title="Ep 1", feed_url=feed
    )
    assert created
    again, created = store.add_item(
        source="https://new-host.example/ep1.mp3", source_type="podcast", title="Ep 1", feed_url=feed
    )
    assert (again, created) == (first, False)
    other, created = store.add_item(
        source="https://new-host.example/ep1-b.mp3", source_type="podcast", title="Ep 1",
        feed_url="https://feed.example/other/rss",
    )
    assert created and other != first


def test_untranscribed_episode_takes_the_new_download_address(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    feed = "https://show.example/rss"
    first, _ = store.add_item(
        source="https://old.example/expired.mp3",
        source_type="podcast",
        title="Ep",
        feed_url=feed,
        meta_json=json.dumps({"audio_url": "https://old.example/expired.mp3"}),
    )
    store.set_status(first, "failed")
    again, created = store.add_item(
        source="https://new.example/current.mp3",
        source_type="podcast",
        title="Ep",
        feed_url=feed,
        meta_json=json.dumps({"audio_url": "https://new.example/current.mp3"}),
    )
    row = store.get_item(first)
    assert (again, created) == (first, False)
    assert row["source"] == "https://new.example/current.mp3"
    assert json.loads(row["meta_json"])["audio_url"] == "https://new.example/current.mp3"
    assert row["status"] == "failed"  # still retried by run --retry-failed
    # A transcribed episode keeps the address it was transcribed from.
    store.save_result(
        first, segments=[], duration=1.0, language="en", model="m", transcript_path="t.md"
    )
    store.add_item(
        source="https://newer.example/x.mp3", source_type="podcast", title="Ep", feed_url=feed
    )
    assert store.get_item(first)["source"] == "https://new.example/current.mp3"
    store.close()
