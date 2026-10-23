from __future__ import annotations

import argparse
from pathlib import Path

from media_asr import cli
from media_asr.db import Store


def test_run_feed_contains_selects_only_matching_feed(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "media.db")
    zh_id, _ = store.add_item(
        source="https://cdn.example/zh.m4a",
        source_type="podcast",
        title="zh",
        feed_url="https://feed.example/zh-show/rss",
    )
    store.add_item(
        source="https://cdn.example/en.mp3",
        source_type="podcast",
        title="en",
        feed_url="https://feed.example/en-show/rss",
    )
    seen: list[str] = []
    monkeypatch.setattr(
        cli, "_process_item", lambda store, item, *a, **k: seen.append(item["id"]) or True
    )
    parser = argparse.ArgumentParser()
    cli.add_run_options(parser)
    args = parser.parse_args(["--feed-contains", "zh-show"])
    args.backend = "local"
    monkeypatch.setattr(cli, "Transcriber", lambda *a, **k: object())
    assert cli.cmd_run(store, args) == 0
    assert seen == [zh_id]
