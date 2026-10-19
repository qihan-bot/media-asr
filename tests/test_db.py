from __future__ import annotations

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