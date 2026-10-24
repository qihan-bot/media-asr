"""Voiceprint naming: decisions, references, the voiceprint step and enrich."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from media_asr import enrich as E
from media_asr import voiceprint as V
from media_asr.db import Store
from media_asr.library import Show, file_digest
from media_asr.render import render_all
from media_asr.transcribe import TranscriptSegment

# Three voices in three dimensions; references for two of them.
A, B, C = np.eye(3)
REFERENCES = {"A Host": A, "B Host": B}
SHOW = Show(feed="https://a/rss", language="en", hosts="A Host and B Host", voiceprints="refs.npz")


def label(vec, seconds: float) -> dict:
    return {
        "share": 0.1,
        "seconds": seconds,
        "centroid": V.unit(np.asarray(vec, float)),
        "purity": 1.0,
    }


def test_a_clear_voice_matches_and_a_short_label_needs_more() -> None:
    near_a = [0.7, 0.0, 0.71]  # score ~0.70: enough at 60 s, not at 20 s
    out = V.judge(
        {"S0": label([1, 0.1, 0], 120), "S1": label(near_a, 60), "S2": label(near_a, 20)},
        REFERENCES,
    )
    assert out["S0"]["voice"] == "match" and out["S0"]["name"] == "A Host"
    assert out["S0"]["confidence"] == "high" and out["S0"]["score"] == 0.995
    assert out["S1"]["voice"] == "match" and out["S1"]["confidence"] == "medium"
    assert out["S2"]["voice"] == "no-match" and out["S2"]["best"] == "A Host"
    assert "name" not in out["S2"]


def test_little_speech_is_not_judged() -> None:
    out = V.judge(
        {
            "S0": label([1, 0, 0], 5),
            "S1": {"share": 0.0, "seconds": 0.0, "centroid": None, "purity": None},
        },
        REFERENCES,
    )
    assert out == {
        "S0": {"voice": "insufficient", "share": 0.1, "seconds": 5},
        "S1": {"voice": "insufficient", "share": 0.0, "seconds": 0.0},
    }


def test_a_voice_between_two_references_is_no_match() -> None:
    out = V.judge({"S0": label([1, 1, 0], 100)}, REFERENCES)  # 0.71 each: no margin
    assert out["S0"]["voice"] == "no-match" and out["S0"]["margin"] == 0.0


def test_fingerprint_is_the_archives() -> None:
    # speaker_common.segments_fingerprint of the All-In archive gave this.
    segs = [{"start": 0.0, "end": 5.12345}, {"start": 5.12345, "end": 9.0}]
    assert V.segments_fingerprint(segs) == "7b077d589a253df6"


def test_label_voices_weights_by_seconds_embedded() -> None:
    segs = [
        {"start": 0, "end": 10, "speaker": "S0"},
        {"start": 10, "end": 30, "speaker": "S0"},
        {"start": 30, "end": 40, "speaker": "S1"},
        {"start": 40, "end": 41, "speaker": "S1"},  # too short: not embedded
    ]
    emb = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    voices = V.label_voices(segs, emb, seg=[0, 1, 2], samples=[16000 * 10, 16000 * 30, 16000 * 5])
    assert voices["S0"]["seconds"] == 40 and voices["S1"]["seconds"] == 5
    assert np.allclose(voices["S0"]["centroid"], V.unit(np.array([10.0, 30.0, 0])))
    assert voices["S0"]["share"] == pytest.approx(30 / 41)


def seeds(name: str, vec, episodes: int, seconds: float = 200, confidence: str = "high") -> list:
    return [
        {
            "id": f"{name}{n}",
            "labels": {
                "S0": {
                    "llm": {"name": name, "confidence": confidence},
                    "seconds": seconds,
                    "centroid": V.unit(np.asarray(vec, float)),
                }
            },
        }
        for n in range(episodes)
    ]


def test_references_need_four_episodes_of_confident_long_labels() -> None:
    episodes = (
        seeds("A Host", [1, 0.1, 0], 4)
        + seeds("B Host", [0, 1, 0.1], 5)
        + seeds("Short Guest", [0.1, 0, 1], 6, seconds=60)  # too little speech
        + seeds("Unsure Guest", [0.1, 0.1, 1], 6, confidence="medium")
        + seeds("Rare Guest", [0, 0.1, 1], 3)  # three episodes only
        + seeds("Guest 1", [0, 0, 1], 9)  # a placeholder is nobody
    )
    references, info = V.build_references(episodes)
    assert sorted(references) == ["A Host", "B Host"]
    assert info["seed_counts"] == {"A Host": (4, 4), "B Host": (5, 5)}


def test_references_drop_seeds_that_sound_like_someone_else() -> None:
    wrong = {
        "id": "x",
        "labels": {
            "S9": {
                "llm": {"name": "A Host", "confidence": "high"},
                "seconds": 300,
                "centroid": V.unit(np.array([0, 1.0, 0.05])),
            }
        },
    }
    references, info = V.build_references(
        seeds("A Host", [1, 0, 0], 4) + seeds("B Host", [0, 1, 0], 4) + [wrong]
    )
    # seed_counts are the last round's (as in voiceprint_relabel.py): x went in round one.
    assert info["seed_counts"]["A Host"] == (4, 4)
    assert ("x", "S9") not in info["seeds"]["A Host"] and len(info["seeds"]["A Host"]) == 4
    assert float(references["A Host"] @ A) > 0.99


def test_references_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "voiceprints" / "show.npz"
    V.save_references(path, REFERENCES, {"built_from": "test"})
    loaded = V.load_references(path)
    assert sorted(loaded) == ["A Host", "B Host"] and np.allclose(loaded["A Host"], A)
    V.save_references(path, {"A Host": A}, {})
    with pytest.raises(ValueError, match="two or more"):
        V.load_references(path)


# --- the voiceprint step -------------------------------------------------------------


def episode(store: Store, items_dir: Path, n: int, feed: str = "https://a/rss") -> str:
    """A done, diarized episode: SPEAKER_00 (60 s), SPEAKER_01 (60 s), SPEAKER_02 (5 s)."""
    item, _ = store.add_item(
        source=f"{feed}/{n}.mp3", source_type="podcast", title=f"Ep {n}", feed_url=feed
    )
    segs = [
        TranscriptSegment(0.0, 60.0, "hello there", "SPEAKER_00"),
        TranscriptSegment(60.0, 120.0, "hi to you", "SPEAKER_01"),
        TranscriptSegment(120.0, 125.0, "an ad", "SPEAKER_02"),
    ]
    render_all(
        out_dir=items_dir / item,
        item_id=item,
        title=f"Ep {n}",
        source="s",
        duration=125.0,
        language="en",
        model="asr",
        transcribed_at="t",
        segments=segs,
    )
    store.save_result(
        item, segments=segs, duration=125.0, language="en", model="asr", transcript_path="t.md"
    )
    return item


class Embedder:
    """The GPU host stand-in: SPEAKER_00 sounds like A, SPEAKER_01 like C, SPEAKER_02 like B."""

    voices = {"SPEAKER_00": [1, 0.05, 0], "SPEAKER_01": [0.05, 0.1, 1], "SPEAKER_02": [0, 1, 0]}

    def __init__(self, fail=(), meanwhile=None) -> None:
        self.calls: list[str] = []
        self.fail, self.meanwhile = set(fail), meanwhile

    def __call__(self, row, segments, fingerprint):
        self.calls.append(row["id"])
        if row["id"] in self.fail:
            raise RuntimeError("remote failed")
        if self.meanwhile:
            self.meanwhile(row)
        emb = np.array([self.voices[s["speaker"]] for s in segments], float)
        samples = [int((s["end"] - s["start"] - 0.4) * 16000) for s in segments]
        return emb, list(range(len(segments))), samples, fingerprint


@pytest.fixture
def library(tmp_path: Path):
    store = Store(tmp_path / "media.db")
    items_dir = tmp_path / "items"
    V.save_references(tmp_path / "refs.npz", REFERENCES, {})
    yield store, items_dir, tmp_path
    store.close()


def run(store, items_dir, data_dir, embed, **kwargs) -> int:
    return V.run(store.conn, [SHOW], embed=embed, items_dir=items_dir, data_dir=data_dir, **kwargs)


def test_the_step_writes_a_marker_for_the_transcript_it_judged(library, capsys) -> None:
    store, items_dir, data_dir = library
    item = episode(store, items_dir, 1)
    other = episode(store, items_dir, 2, feed="https://other/rss")  # a show without voiceprints
    (items_dir / item / "enrich.json").write_text("{}")
    (items_dir / item / "terms.json").write_text("{}")
    embed = Embedder()
    assert run(store, items_dir, data_dir, embed) == 0
    assert embed.calls == [item]
    marker = json.loads((items_dir / item / "voiceprint.json").read_text())
    assert marker["transcript_sha256"] == file_digest(items_dir / item / "transcript.json")
    assert marker["references"] == ["A Host", "B Host"]
    speakers = marker["speakers"]
    assert speakers["SPEAKER_00"]["voice"] == "match" and speakers["SPEAKER_00"]["name"] == "A Host"
    assert (
        speakers["SPEAKER_01"]["voice"] == "no-match"
        and len(speakers["SPEAKER_01"]["centroid"]) == 3
    )
    assert speakers["SPEAKER_02"]["voice"] == "insufficient"
    # The names may change: enrich and correct-terms run again.
    assert not (items_dir / item / "enrich.json").exists()
    assert not (items_dir / item / "terms.json").exists()
    assert not (items_dir / other / "voiceprint.json").exists()
    assert "1/3 by voice: SPEAKER_00=A Host" in capsys.readouterr().out
    # Done once: the next run has nothing to do, until the transcript changes.
    assert run(store, items_dir, data_dir, embed) == 0 and embed.calls == [item]
    transcript = items_dir / item / "transcript.json"
    transcript.write_text(transcript.read_text().replace("hello there", "hello here"))
    assert run(store, items_dir, data_dir, embed) == 0 and embed.calls == [item, item]
    assert run(store, items_dir, data_dir, embed, force=True) == 0 and len(embed.calls) == 3


def test_one_failed_episode_does_not_stop_the_others(library, capsys) -> None:
    store, items_dir, data_dir = library
    first, second = episode(store, items_dir, 1), episode(store, items_dir, 2)
    assert run(store, items_dir, data_dir, Embedder(fail={first})) == 1
    assert not (items_dir / first / "voiceprint.json").exists()
    assert (items_dir / second / "voiceprint.json").exists()
    assert f"FAIL {first}" in capsys.readouterr().out


def test_no_marker_for_a_transcript_that_changed_meanwhile(library, capsys) -> None:
    store, items_dir, data_dir = library
    item = episode(store, items_dir, 1)

    def retranscribed(row) -> None:
        path = items_dir / row["id"] / "transcript.json"
        path.write_text(path.read_text().replace("hello there", "hello here"))

    assert run(store, items_dir, data_dir, Embedder(meanwhile=retranscribed)) == 1
    assert not (items_dir / item / "voiceprint.json").exists()
    assert "transcript.json changed while voiceprinting" in capsys.readouterr().out


def test_embeddings_of_another_segmentation_are_refused(library, capsys) -> None:
    store, items_dir, data_dir = library
    item = episode(store, items_dir, 1)

    def stale(row, segments, fingerprint):
        return Embedder()(row, segments, "0123456789abcdef")

    assert run(store, items_dir, data_dir, stale) == 1
    assert not (items_dir / item / "voiceprint.json").exists()
    assert "another segmentation" in capsys.readouterr().out


def test_missing_references_fail_the_shows_episodes(library, capsys) -> None:
    store, items_dir, data_dir = library
    episode(store, items_dir, 1)
    (data_dir / "refs.npz").unlink()
    assert run(store, items_dir, data_dir, Embedder()) == 1
    assert "refs.npz" in capsys.readouterr().out


def test_an_episode_without_speaker_labels_is_skipped(library) -> None:
    store, items_dir, data_dir = library
    item = episode(store, items_dir, 1)
    transcript = items_dir / item / "transcript.json"
    data = json.loads(transcript.read_text())
    for seg in data["segments"]:
        seg.pop("speaker")
    transcript.write_text(json.dumps(data))
    embed = Embedder()
    assert run(store, items_dir, data_dir, embed) == 0 and embed.calls == []


# --- enrich with voiceprints -------------------------------------------------------------


class Namer:
    """Echo punctuation; names labels as given, recording the speaker prompts."""

    model = "fake"

    def __init__(self, names: dict) -> None:
        self.names, self.prompts = names, []

    def chat(self, system: str, user: str, max_tokens: int) -> dict:
        if "label podcast speakers" in system:
            self.prompts.append(user)
            return {"speakers": {k: dict(v) for k, v in self.names.items()}}
        return {"lines": user.split("\n")}


def voiced(library) -> tuple:
    store, items_dir, data_dir = library
    item = episode(store, items_dir, 1)
    assert run(store, items_dir, data_dir, Embedder()) == 0
    return store, items_dir, item


def named(items_dir: Path, item: str) -> dict:
    return json.loads((items_dir / item / "enrich.json").read_text())["speakers"]


def test_enrich_keeps_voiceprint_names_and_asks_only_about_the_rest(library) -> None:
    store, items_dir, item = voiced(library)
    llm = Namer(
        {
            "SPEAKER_00": {"name": "B Host", "confidence": "high"},  # ignored: matched by voice
            "SPEAKER_01": {"name": "Carol Guest", "role": "guest", "confidence": "high"},
            "SPEAKER_02": {"name": None, "role": "other", "confidence": "low"},
        }
    )
    assert E.run(store.conn, llm, [SHOW], items_dir=items_dir) == 0
    (prompt,) = llm.prompts
    assert prompt.endswith(
        "Already identified by voice; keep these and do not label them again:\n"
        "SPEAKER_00: A Host\nLabel only: SPEAKER_01, SPEAKER_02\n"
    )
    speakers = named(items_dir, item)
    assert speakers["SPEAKER_00"] == {
        "name": "A Host",
        "role": "host",
        "confidence": "high",
        "evidence": "voiceprint 1.00 (margin 0.95)",
        "method": "voiceprint",
    }
    assert (
        speakers["SPEAKER_01"]["name"] == "Carol Guest" and "method" not in speakers["SPEAKER_01"]
    )
    assert speakers["SPEAKER_02"]["name"] == "Speaker 1"
    lines = json.loads((items_dir / item / "transcript_named.json").read_text())["segments"]
    assert [s["speaker"] for s in lines] == ["A Host", "Carol Guest", "Speaker 1"]


def test_enrich_rejects_a_reference_name_the_voice_contradicts(library) -> None:
    store, items_dir, item = voiced(library)
    # SPEAKER_01 sounds like nobody in the references (closest: B Host, 0.10).
    llm = Namer({"SPEAKER_01": {"name": "A Host", "role": "host", "confidence": "high"}})
    assert E.run(store.conn, llm, [SHOW], items_dir=items_dir) == 0
    rejected = named(items_dir, item)["SPEAKER_01"]
    assert rejected["name"] == "Speaker 1" and rejected["role"] == "other"
    assert rejected["evidence"] == (
        "voice does not match A Host (closest B Host 0.10); model name rejected"
    )


def test_enrich_keeps_a_reference_name_the_voice_leans_to(library) -> None:
    store, items_dir, item = voiced(library)
    # SPEAKER_01 matches nobody, but leans to B Host: the model's B Host stands.
    llm = Namer({"SPEAKER_01": {"name": "B Host", "role": "host", "confidence": "medium"}})
    assert E.run(store.conn, llm, [SHOW], items_dir=items_dir) == 0
    assert named(items_dir, item)["SPEAKER_01"]["name"] == "B Host"
    # A label with too little speech is not judged by voice: the model's name stands.
    llm = Namer({"SPEAKER_02": {"name": "B Host", "role": "host", "confidence": "high"}})
    assert E.run(store.conn, llm, [SHOW], force=True, items_dir=items_dir) == 0
    assert named(items_dir, item)["SPEAKER_02"]["name"] == "B Host"


def test_enrich_skips_the_model_when_every_label_is_known(library) -> None:
    store, items_dir, item = voiced(library)
    path = items_dir / item / "voiceprint.json"
    marker = json.loads(path.read_text())
    for rec in marker["speakers"].values():
        rec.update(voice="match", name="B Host", confidence="medium", score=0.7, margin=0.2)
    path.write_text(json.dumps(marker))
    llm = Namer({})
    assert E.run(store.conn, llm, [SHOW], items_dir=items_dir) == 0
    assert llm.prompts == []
    assert {s["name"] for s in named(items_dir, item).values()} == {"B Host"}


def test_enrich_refuses_a_voiceprint_of_another_transcript(library, capsys) -> None:
    store, items_dir, item = voiced(library)
    transcript = items_dir / item / "transcript.json"
    transcript.write_text(transcript.read_text().replace("hello there", "hello here"))
    assert E.run(store.conn, Namer({}), [SHOW], items_dir=items_dir) == 1
    assert not (items_dir / item / "enrich.json").exists()
    assert "voiceprint.json is for another transcript" in capsys.readouterr().out


def test_enrich_without_voiceprints_asks_as_before(library) -> None:
    store, items_dir, data_dir = library
    item = episode(store, items_dir, 1)
    llm = Namer({"SPEAKER_00": {"name": "B Host", "role": "host", "confidence": "high"}})
    assert E.run(store.conn, llm, [SHOW], items_dir=items_dir) == 0
    (prompt,) = llm.prompts
    assert "identified by voice" not in prompt and prompt.endswith('"short reason"}}}\n')
    assert named(items_dir, item)["SPEAKER_00"]["name"] == "B Host"


def test_a_voiceprint_written_while_enriching_wins(library) -> None:
    # Review 0.0.4 #1: names computed before the voiceprint landed were committed
    # over it, and the done marker kept the voiceprint names out for good.
    store, items_dir, data_dir = library
    item = episode(store, items_dir, 1)
    namer = Namer({"SPEAKER_00": {"name": "B Host", "role": "host", "confidence": "high"}})
    late = E.enrich_item(store.get_item(item), {}, {}, [SHOW], namer, 1, items_dir)
    assert late["voiceprint"] is None
    assert run(store, items_dir, data_dir, Embedder()) == 0  # lands meanwhile
    with pytest.raises(RuntimeError, match="voiceprint.json changed while enriching"):
        E.commit(store.conn, late, items_dir)
    assert not (items_dir / item / "enrich.json").exists()
    assert E.run(store.conn, namer, [SHOW], items_dir=items_dir) == 0
    assert named(items_dir, item)["SPEAKER_00"]["name"] == "A Host"


def test_a_voiceprint_redone_while_enriching_fails_the_commit(library) -> None:
    store, items_dir, item = voiced(library)
    stale = E.enrich_item(store.get_item(item), {}, {}, [SHOW], Namer({}), 1, items_dir)
    path = items_dir / item / "voiceprint.json"  # redone meanwhile, with another result
    marker = json.loads(path.read_text())
    marker["speakers"]["SPEAKER_01"].update(voice="match", name="B Host", confidence="medium")
    path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="voiceprint.json changed"):
        E.commit(store.conn, stale, items_dir)
    fresh = E.enrich_item(store.get_item(item), {}, {}, [SHOW], Namer({}), 1, items_dir)
    assert E.commit(store.conn, fresh, items_dir)
