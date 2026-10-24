"""scripts/migrate_allin.py on a small All-In archive."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import migrate_allin as M  # noqa: E402

from media_asr import enrich as E  # noqa: E402
from media_asr import voiceprint as V  # noqa: E402
from media_asr.db import Segment, Store  # noqa: E402
from media_asr.render import render_all  # noqa: E402
from media_asr.transcribe import TranscriptSegment  # noqa: E402
from media_asr.util import file_lock  # noqa: E402

FEED = "https://allinchamathjason.libsyn.com/rss"
VOICES = {"SPEAKER_00": [1.0, 0.05, 0.0], "SPEAKER_01": [0.0, 1.0, 0.05], "SPEAKER_02": [0, 0, 1.0]}
LLM = {
    "SPEAKER_00": {"name": "A Host", "confidence": "high"},
    "SPEAKER_01": {"name": "B Host", "confidence": "high"},
    "SPEAKER_02": {"name": "Guest 1", "confidence": "low"},
}
SHOWS = (
    '[[show]]\nfeed = "https://allinchamathjason.libsyn.com/rss"\nlanguage = "en"\n'
    'hosts = "A Host and B Host"\nvoiceprints = "voiceprints/all-in.npz"\n'
)


def diarized(item_dir: Path, item: str, title: str) -> list[dict]:
    """A diarized run: A Host 200 s, B Host 200 s, a 5 s clip; returns its segments."""
    segs = [
        TranscriptSegment(0.0, 200.0, "a talks", "SPEAKER_00"),
        TranscriptSegment(200.0, 400.0, "b talks", "SPEAKER_01"),
        TranscriptSegment(400.0, 405.0, "a clip", "SPEAKER_02"),
    ]
    run = item_dir / "runs" / "diarized"
    render_all(
        out_dir=run,
        item_id=item,
        title=title,
        source="s",
        duration=405.0,
        language="en",
        model="whisper (remote)",
        transcribed_at="t",
        segments=segs,
    )
    (run / "run.json").write_text(json.dumps({"model": "whisper (remote)", "device": "GPU"}))
    return json.loads((run / "transcript.json").read_text())["segments"]


def maps(run: Path, segments: list[dict]) -> None:
    fp = M.labels_fingerprint(segments)
    (run / "speaker_map.llm.json").write_text(json.dumps({"speakers": LLM, "labels_fp": fp}))
    final = {
        "SPEAKER_00": {
            **LLM["SPEAKER_00"],
            "voice": "match",
            "voice_best": "A Host",
            "voice_score": 0.999,
            "voice_margin": 0.95,
            "voice_seconds": 199.6,
            "share": 0.4938,
        },
        "SPEAKER_01": {
            **LLM["SPEAKER_01"],
            "voice": "match",
            "voice_best": "B Host",
            "voice_score": 0.999,
            "voice_margin": 0.95,
            "voice_seconds": 199.6,
            "share": 0.4938,
        },
        "SPEAKER_02": {
            **LLM["SPEAKER_02"],
            "voice": "insufficient",
            "voice_seconds": 4.6,
            "share": 0.0123,
        },
    }
    voiceprint = {
        "applied_at": "t",
        "threshold": 0.65,
        "margin": 0.1,
        "templates": ["A Host", "B Host"],
    }
    (run / "speaker_map.json").write_text(
        json.dumps({"speakers": final, "labels_fp": fp, "voiceprint": voiceprint})
    )


def embeddings(path: Path, segments: list[dict], fingerprint: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        emb=np.array([VOICES[s["speaker"]] for s in segments], np.float32),
        seg=np.arange(len(segments)),
        samples=np.array([int((s["end"] - s["start"] - 0.4) * 16000) for s in segments]),
        fp=np.array(fingerprint or V.segments_fingerprint(segments)),
    )


def make_archive(root: Path, count: int = 4) -> list[str]:
    """``count`` All-In episodes (diarized, mapped, embedded), the YouTube copy,
    an All-In episode never diarized and another show's episode."""
    store = Store(root / "media.db")
    store.add_feed(FEED, "All-In", 104)
    ids = []
    for n in range(count):
        item, _ = store.add_item(
            source=f"https://allin/{n}.mp3",
            source_type="podcast",
            title=f"All-In {n}",
            feed_url=FEED,
            published_at=f"2026-10-0{n + 1}T00:00:00+00:00",
        )
        ids.append(item)
    store.conn.execute(
        "INSERT INTO items (id, source, source_type, title, status, created_at, updated_at, published_at) "
        "VALUES (?, 'https://www.youtube.com/watch?v=x', 'video', 'All-In on YouTube', 'pending', 't', 't', '2026-10-09T00:00:00+00:00')",
        (next(iter(M.EXTRA_IDS)),),
    )
    store.conn.commit()
    ids.append(next(iter(M.EXTRA_IDS)))
    never, _ = store.add_item(
        source="https://allin/never.mp3",
        source_type="podcast",
        title="Never diarized",
        feed_url=FEED,
    )
    other, _ = store.add_item(
        source="https://other/1.mp3",
        source_type="podcast",
        title="Other show",
        feed_url="https://other/rss",
    )
    for item in [*ids, never, other]:
        item_dir = root / "items" / item
        item_dir.mkdir(parents=True)
        (item_dir / "meta.json").write_text(json.dumps({"id": item, "feed_url": FEED}))
        store.save_result(
            item,
            segments=[Segment(0, 405, "primary transcript")],
            duration=405.0,
            language="en",
            model="m",
            transcript_path=str(item_dir / "transcript.md"),
        )
        if item == never:
            continue
        segments = diarized(item_dir, item, store.get_item(item)["title"])
        if item != other:
            maps(item_dir / "runs" / "diarized", segments)
            embeddings(root / "speaker_embeddings" / f"{item}.npz", segments)
    store.close()
    return ids


def files(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def dirs(tmp_path: Path):
    archive, library = tmp_path / "data", tmp_path / "data-podcasts"
    ids = make_archive(archive)
    Store(library / "media.db").close()
    (library / "library.toml").write_text(SHOWS)
    return archive, library, ids


def migrate(archive: Path, library: Path, *flags: str, capsys) -> tuple[int, dict]:
    code = M.main(["--archive", str(archive), "--library", str(library), *flags])
    return code, json.loads(capsys.readouterr().out)


def test_a_dry_run_reports_and_writes_nothing(dirs, capsys) -> None:
    archive, library, ids = dirs
    before = files(archive), files(library)
    code, summary = migrate(archive, library, capsys=capsys)
    assert code == 0 and summary["mode"] == "dry run"
    assert (summary["archive_episodes"], summary["migrate"], summary["conflicts"]) == (5, 5, [])
    assert [reason for _, reason in summary["skipped"]] == ["no diarized transcript"]
    assert summary["segments"] == 15 and summary["voiceprint_from"] == {"embeddings": 5}
    assert summary["references"] == ["A Host", "B Host"]
    assert summary["voiceprint_coverage"] == {
        "labels": {"match": 10, "insufficient": 5},
        "speaking_time_named_by_voice": 0.9877,
    }
    assert summary["agreement_with_archive"]["labels"] == {"same": 15}
    assert "library_toml" not in summary
    assert (files(archive), files(library)) == before


def test_apply_moves_the_episodes_keeping_their_ids(dirs, capsys) -> None:
    archive, library, ids = dirs
    before = files(archive)
    code, summary = migrate(archive, library, "--apply", capsys=capsys)
    assert code == 0 and summary["applied"] == {"episodes": 5, "segments": 15}
    assert files(archive) == before
    db = sqlite3.connect(library / "media.db")
    db.row_factory = sqlite3.Row
    rows = {r["id"]: r for r in db.execute("SELECT * FROM items")}
    assert sorted(rows) == sorted(ids)
    for item in ids:
        run = archive / "items" / item / "runs" / "diarized"
        target = library / "items" / item
        for name in M.RUN_FILES:
            assert (target / name).read_bytes() == (run / name).read_bytes()
        assert rows[item]["status"] == "done" and rows[item]["transcript_path"] == str(
            target / "transcript.md"
        )
        speakers = [
            r[0]
            for r in db.execute(
                "SELECT speaker FROM segments WHERE item_id = ? ORDER BY idx", (item,)
            )
        ]
        assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
        marker = json.loads((target / "voiceprint.json").read_text())
        assert (
            marker["transcript_sha256"]
            == hashlib.sha256((target / "transcript.json").read_bytes()).hexdigest()
        )
        assert marker["speakers"]["SPEAKER_00"]["name"] == "A Host"
    assert rows[ids[-1]]["feed_url"] is None  # the YouTube copy stays outside the feed
    assert [r[0] for r in db.execute("SELECT url FROM feeds")] == [FEED]
    assert sorted(V.load_references(library / "voiceprints/all-in.npz")) == ["A Host", "B Host"]
    db.close()


def test_apply_again_changes_nothing_and_finishes_an_interrupted_run(dirs, capsys) -> None:
    archive, library, ids = dirs
    migrate(archive, library, "--apply", capsys=capsys)
    (library / "items" / ids[0] / "voiceprint.json").unlink()  # as if interrupted there
    before = files(library)
    code, summary = migrate(archive, library, "--apply", capsys=capsys)
    assert code == 0
    assert (summary["migrate"], summary["already_in_library"]) == (0, 5)
    assert summary["applied"] == {"voiceprints added": 1}
    assert "built from" not in summary["references_file"]
    after = files(library)

    def kept(tree: dict) -> dict:  # all but the marker added, the database and the locks
        return {
            k: v
            for k, v in tree.items()
            if "media.db" not in k and not k.endswith(("voiceprint.json", ".lock"))
        }

    assert kept(after) == kept(before)
    db = sqlite3.connect(library / "media.db")
    assert db.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 5
    assert db.execute("SELECT COUNT(*) FROM segments").fetchone()[0] == 15
    db.close()


def test_an_episode_with_another_transcript_is_a_conflict(dirs, capsys) -> None:
    archive, library, ids = dirs
    migrate(archive, library, "--apply", capsys=capsys)
    transcript = library / "items" / ids[1] / "transcript.json"
    transcript.write_text(transcript.read_text().replace("b talks", "b spoke"))
    code, summary = migrate(archive, library, "--apply", capsys=capsys)
    assert code == 1 and summary["conflicts"] == [ids[1]]
    assert "b spoke" in transcript.read_text()


def test_without_fitting_embeddings_the_speaker_map_gives_the_voices(dirs, capsys) -> None:
    archive, library, ids = dirs
    run = archive / "items" / ids[0] / "runs" / "diarized"
    segments = json.loads((run / "transcript.json").read_text())["segments"]
    embeddings(archive / "speaker_embeddings" / f"{ids[0]}.npz", segments, fingerprint="older")
    code, summary = migrate(archive, library, "--apply", capsys=capsys)
    assert code == 0 and summary["voiceprint_from"] == {"embeddings": 4, "speaker_map": 1}
    marker = json.loads((library / "items" / ids[0] / "voiceprint.json").read_text())
    assert marker["from"] == "speaker_map.json" and marker["references"] == ["A Host", "B Host"]
    assert (
        marker["speakers"]["SPEAKER_01"]["voice"] == "match"
        and marker["speakers"]["SPEAKER_01"]["name"] == "B Host"
    )
    assert marker["speakers"]["SPEAKER_02"]["voice"] == "insufficient"


def test_apply_waits_for_no_library_update(dirs, capsys) -> None:
    archive, library, _ = dirs
    with file_lock(library / ".library-update.lock"):
        assert M.main(["--archive", str(archive), "--library", str(library), "--apply"]) == 1
    assert "library-update holds" in capsys.readouterr().err
    assert not (library / "items").exists()


def test_without_the_all_in_show_the_summary_warns(dirs, capsys) -> None:
    archive, library, _ = dirs
    (library / "library.toml").write_text('[[show]]\nfeed = "https://other/rss"\nlanguage = "en"\n')
    _, summary = migrate(archive, library, capsys=capsys)
    assert summary["library_toml"].startswith("WARN: add the All-In show")


class Namer:
    model = "fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, system: str, user: str, max_tokens: int) -> dict:
        if "label podcast speakers" in system:
            self.prompts.append(user)
            return {
                "speakers": {
                    "SPEAKER_02": {"name": "C Guest", "role": "guest", "confidence": "high"}
                }
            }
        return {"lines": user.split("\n")}


def test_enrich_then_names_the_migrated_episodes_by_voice(dirs, capsys, monkeypatch) -> None:
    archive, library, ids = dirs
    migrate(archive, library, "--apply", capsys=capsys)
    store = Store(library / "media.db")
    llm = Namer()
    from media_asr.library import load_shows

    assert (
        E.run(store.conn, llm, load_shows(library / "library.toml"), items_dir=library / "items")
        == 0
    )
    assert len(llm.prompts) == 5 and all("Label only: SPEAKER_02" in p for p in llm.prompts)
    named = json.loads((library / "items" / ids[0] / "transcript_named.json").read_text())[
        "segments"
    ]
    assert [s["speaker"] for s in named] == ["A Host", "B Host", "C Guest"]
    roles = json.loads((library / "items" / ids[0] / "enrich.json").read_text())["speakers"]
    assert (roles["SPEAKER_00"]["role"], roles["SPEAKER_02"]["role"]) == ("host", "guest")
    store.close()
