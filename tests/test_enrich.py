from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from media_asr import enrich as E
from media_asr.db import Store
from media_asr.render import render_all
from media_asr.transcribe import TranscriptSegment
from media_asr.util import item_lock


class PunctuatingLLM:
    """Adds punctuation, but "corrects" one word: that segment must be kept as is."""

    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system: str, user: str, max_tokens: int) -> dict:
        self.calls += 1
        lines = [line.replace("teh", "the").capitalize() + "." for line in user.split("\n")]
        return {"lines": lines}


def test_punctuation_never_changes_words() -> None:
    texts = ["hello world", "teh cat", "  ", "it is 3 pm"]
    out, accepted = E.punctuate(PunctuatingLLM(), texts, "en", workers=2)
    assert out == ["Hello world.", "teh cat", "  ", "It is 3 pm."]
    assert accepted == 2


def test_norm_ignores_punctuation_case_and_width() -> None:
    assert E.norm("Ｈello, World！") == E.norm("hello world")
    assert E.norm("你好，世界。") == "你好世界"


class NamingLLM:
    """Punctuates every line and names SPEAKER_0i as <name><i> with high confidence."""

    model = "fake"

    def __init__(self, name: str, mark: str = ".") -> None:
        self.name, self.mark = name, mark

    def chat(self, system: str, user: str, max_tokens: int) -> dict:
        if "label podcast speakers" in system:
            names = {
                f"SPEAKER_0{i}": {"name": f"{self.name}{i}", "confidence": "high"} for i in range(2)
            }
            return {"speakers": names}
        return {"lines": [line + self.mark for line in user.split("\n")]}


class LockedDB:
    """A connection whose index writes fail, as when SQLite is locked."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __getattr__(self, name: str):
        return getattr(self.conn, name)

    def __enter__(self):
        return self.conn.__enter__()

    def __exit__(self, *exc):
        return self.conn.__exit__(*exc)

    def executemany(self, *args):
        raise sqlite3.OperationalError("database is locked")


def make_library(tmp_path: Path, texts=("hello world", "second line")) -> tuple[Store, Path, str]:
    store = Store(tmp_path / "media.db")
    items_dir = tmp_path / "items"
    item, _ = store.add_item(source="https://a/1.mp3", source_type="podcast", title="Ep")
    segments = [
        TranscriptSegment(i * 10.0, i * 10.0 + 10, text, f"SPEAKER_0{i % 2}")
        for i, text in enumerate(texts)
    ]
    render_all(
        out_dir=items_dir / item,
        item_id=item,
        title="Ep",
        source="https://a/1.mp3",
        duration=20.0,
        language="en",
        model="asr",
        transcribed_at="t",
        segments=segments,
    )
    store.save_result(
        item,
        segments=segments,
        duration=20.0,
        language="en",
        model="asr",
        transcript_path=str(items_dir / item / "transcript.md"),
    )
    return store, items_dir, item


def state(store: Store, items_dir: Path, item: str) -> tuple[list, list, list]:
    """(speakers on disk, in named_base.json, in the index)."""
    named = json.loads((items_dir / item / "transcript_named.json").read_text())["segments"]
    base = json.loads((items_dir / item / "named_base.json").read_text())
    rows = store.segments_for(item)
    return [s["speaker"] for s in named], [s["speaker"] for s in base], [r["speaker"] for r in rows]


def test_failed_forced_redo_leaves_the_item_to_be_redone(tmp_path: Path) -> None:
    store, items_dir, item = make_library(tmp_path)
    assert E.run(store.conn, NamingLLM("Alice"), [], items_dir=items_dir) == 0
    (items_dir / item / "terms.json").write_text("{}")
    # The forced redo fails at the index: neither marker may survive it.
    assert (
        E.run(LockedDB(store.conn), NamingLLM("Bob", "!"), [], force=True, items_dir=items_dir) == 1
    )
    assert not (items_dir / item / "enrich.json").exists()
    assert not (items_dir / item / "terms.json").exists()
    # So the next normal run enriches it again, and everything agrees.
    assert E.run(store.conn, NamingLLM("Carol", "?"), [], items_dir=items_dir) == 0
    assert state(store, items_dir, item) == (["Carol0", "Carol1"],) * 3
    store.close()


def test_commits_of_two_runs_never_mix(tmp_path: Path) -> None:
    store, items_dir, item = make_library(tmp_path)
    row = store.get_item(item)
    alice = E.enrich_item(row, {}, {}, [], NamingLLM("Alice"), 1, items_dir)
    bob = E.enrich_item(row, {}, {}, [], NamingLLM("Bob", "!"), 1, items_dir)
    assert E.commit(store.conn, bob, items_dir, force=True)
    assert E.commit(store.conn, alice, items_dir, force=True)
    assert state(store, items_dir, item) == (["Alice0", "Alice1"],) * 3
    # Without --force, a result for an item another run already finished is dropped.
    assert not E.commit(store.conn, bob, items_dir)
    assert state(store, items_dir, item) == (["Alice0", "Alice1"],) * 3
    store.close()


def test_commit_refuses_a_transcript_that_changed(tmp_path: Path) -> None:
    store, items_dir, item = make_library(tmp_path)
    result = E.enrich_item(store.get_item(item), {}, {}, [], NamingLLM("Alice"), 1, items_dir)
    transcript = items_dir / item / "transcript.json"
    transcript.write_text(transcript.read_text().replace("hello world", "hello there"))
    with pytest.raises(RuntimeError, match="transcript.json changed"):
        E.commit(store.conn, result, items_dir)
    assert not (items_dir / item / "enrich.json").exists()
    store.close()


def test_commit_waits_for_the_item_lock(tmp_path: Path) -> None:
    store, items_dir, item = make_library(tmp_path)
    result = E.enrich_item(store.get_item(item), {}, {}, [], NamingLLM("Alice"), 1, items_dir)
    conn = sqlite3.connect(tmp_path / "media.db", check_same_thread=False)
    worker = threading.Thread(target=E.commit, args=(conn, result, items_dir))
    with item_lock(items_dir / item):
        worker.start()
        worker.join(0.3)
        assert worker.is_alive() and not (items_dir / item / "enrich.json").exists()
    worker.join(5)
    assert not worker.is_alive() and (items_dir / item / "enrich.json").exists()
    conn.close()
    store.close()
