from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import embed_speakers  # noqa: E402
import repair_speakers  # noqa: E402
import voiceprint_relabel  # noqa: E402
from normalize_names import fix  # noqa: E402
from speaker_common import segments_fingerprint  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sachs, what do you think?", "Sacks, what do you think?"),
        ("david sachs is right", "david sacks is right"),
        ("to build on Sax's point", "to build on Sacks's point"),
        ("Freeberg?", "Friedberg?"),
        ("Shamath Palihapiteo", "Chamath Palihapitiya"),
        ("Goldman Sachs said", "Goldman Sachs said"),
        ("Goldman  Sachs said", "Goldman  Sachs said"),
        ("Goldman Sachs group", "Goldman Sachs group"),
        ("Jeffrey Sachs said", "Jeffrey Sachs said"),
        ("Saks Global filed", "Saks Global filed"),
        ("is that the sax do you remember", "is that the sax do you remember"),
        ("not the sax apocalypse", "not the sax apocalypse"),
    ],
)
def test_normalize_names(text: str, expected: str) -> None:
    assert fix(text, collections.Counter()) == expected


def test_normalize_does_not_count_correct_spelling() -> None:
    counter: collections.Counter = collections.Counter()
    fix("Chamath Palihapitiya", counter)
    assert not counter


def test_valid_npz_rejects_stale_and_truncated(tmp_path: Path) -> None:
    segs = [{"start": 0.0, "end": 5.0}, {"start": 5.0, "end": 9.0}]
    fp = segments_fingerprint(segs)
    good = tmp_path / "good.npz"
    np.savez(good, emb=np.zeros((2, 256), np.float32), seg=np.array([0, 1]), samples=np.array([1, 2]), fp=np.array(fp))
    assert embed_speakers.valid_npz(good, fp)
    assert not embed_speakers.valid_npz(good, segments_fingerprint(segs[:1]))
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    assert not embed_speakers.valid_npz(empty, fp)
    old = tmp_path / "old.npz"
    np.savez(old, emb=np.zeros((2, 256), np.float32), seg=np.array([0, 1]))
    assert not embed_speakers.valid_npz(old, fp)


def _episode(labels: dict) -> dict:
    return {"labels": labels}


def _label(vec, seconds: float, name: str, conf: str = "high") -> dict:
    return {
        "llm": {"name": name, "confidence": conf},
        "share": 0.1,
        "seconds": seconds,
        "centroid": voiceprint_relabel.unit(np.asarray(vec, dtype=float)),
        "purity": 1.0,
    }


TEMPLATES = {
    "A Host": voiceprint_relabel.unit(np.array([1.0, 0.0, 0.0])),
    "B Host": voiceprint_relabel.unit(np.array([0.0, 1.0, 0.0])),
}


def test_decide_overrides_wrong_llm_name() -> None:
    out = voiceprint_relabel.decide(_episode({"S0": _label([1, 0.1, 0], 120, "B Host")}), TEMPLATES, 0.65, 0.10)
    assert out["S0"]["name"] == "A Host" and out["S0"]["method"] == "voice"


def test_decide_short_label_needs_stricter_score() -> None:
    vec = [0.7, 0.0, 0.71]  # score ~0.70: passes the normal bar, not the short-label bar
    long = voiceprint_relabel.decide(_episode({"S0": _label(vec, 60, "Guest 1")}), TEMPLATES, 0.65, 0.10)
    short = voiceprint_relabel.decide(_episode({"S0": _label(vec, 20, "Guest 1")}), TEMPLATES, 0.65, 0.10)
    assert long["S0"]["name"] == "A Host"
    assert short["S0"]["name"] == "Guest 1"


def test_decide_weak_same_name_is_kept_and_mismatch_is_rejected() -> None:
    weak = voiceprint_relabel.decide(_episode({"S0": _label([0.5, 0, 0.87], 60, "A Host")}), TEMPLATES, 0.65, 0.10)
    assert weak["S0"]["name"] == "A Host" and weak["S0"]["voice"] == "weak"
    reject = voiceprint_relabel.decide(_episode({"S0": _label([0.5, 0, 0.87], 60, "B Host")}), TEMPLATES, 0.65, 0.10)
    assert reject["S0"]["name"] == "Unidentified 1"


def test_decide_insufficient_keeps_llm_name() -> None:
    out = voiceprint_relabel.decide(_episode({"S0": _label([0, 1, 0], 5, "A Host")}), TEMPLATES, 0.65, 0.10)
    assert out["S0"]["name"] == "A Host" and out["S0"]["voice"] == "insufficient"


def test_repair_guest_numbers_do_not_collide(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "items" / "x" / "runs" / "diarized"
    run.mkdir(parents=True)
    segs = [{"start": i, "end": i + 1, "text": "hi", "speaker": f"SPEAKER_0{i}"} for i in range(3)]
    (run / "transcript.json").write_text(json.dumps({"title": "t", "segments": segs}))
    from speaker_common import labels_fingerprint

    mapping = {"speakers": {f"SPEAKER_0{i}": {"name": f"SPEAKER_0{i}", "confidence": "low"} for i in range(3)},
               "labels_fp": labels_fingerprint(segs)}
    (run / "speaker_map.json").write_text(json.dumps(mapping))
    monkeypatch.setattr(repair_speakers, "call", lambda *a: {"speakers": {"SPEAKER_00": {"name": "Guest 1"}}})
    assert repair_speakers.process(run, None, "key") == "done"
    names = [v["name"] for v in json.loads((run / "speaker_map.json").read_text())["speakers"].values()]
    assert sorted(names) == ["Guest 1", "Guest 2", "Guest 3"]
    assert (run / "transcript_named.md").exists()


def test_repair_fallback_parser_recovers_truncated_json(monkeypatch) -> None:
    content = '{"speakers": {"SPEAKER_00": {"name": "Jason Calacanis", "confidence": "high"}, "SPEAKER_01": {"na'

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": content}}]}

    class Client:
        def post(self, *a, **k) -> Response:
            return Response()

    result = repair_speakers.call(Client(), "key", "prompt")
    assert result["speakers"]["SPEAKER_00"]["name"] == "Jason Calacanis"


def test_stale_llm_map_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from speaker_common import labels_fingerprint

    item = tmp_path / "items" / "ep"
    run = item / "runs" / "diarized"
    run.mkdir(parents=True)
    (item / "meta.json").write_text(json.dumps({"feed_url": "https://allinchamathjason.libsyn.com/rss"}))
    segs = [{"start": 0.0, "end": 5.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 9.0, "text": "b", "speaker": "SPEAKER_01"}]
    (run / "transcript.json").write_text(json.dumps({"segments": segs}))
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    np.savez(emb_dir / "ep.npz", emb=np.ones((2, 3), np.float32), seg=np.array([0, 1]),
             samples=np.array([16000, 16000]), fp=np.array(segments_fingerprint(segs)))
    monkeypatch.setattr(voiceprint_relabel, "EMB_DIR", emb_dir)
    speakers = {"SPEAKER_00": {"name": "A"}, "SPEAKER_01": {"name": "B"}}
    swapped = [dict(segs[0], speaker="SPEAKER_01"), dict(segs[1], speaker="SPEAKER_00")]
    (run / "speaker_map.json").write_text(json.dumps({"speakers": speakers, "labels_fp": labels_fingerprint(swapped)}))
    assert voiceprint_relabel.load_episode(item) is None
    (run / "speaker_map.json").write_text(json.dumps({"speakers": speakers, "labels_fp": labels_fingerprint(segs)}))
    assert voiceprint_relabel.load_episode(item) is not None


def test_worker_rebuilds_stale_remote_output(tmp_path: Path) -> None:
    import embed_speakers_worker

    stale = tmp_path / "x.npz"
    np.savez(stale, emb=np.zeros((1, 256), np.float32), seg=np.array([0]), samples=np.array([1]), fp=np.array("old"))
    assert not embed_speakers_worker.valid_output(stale, "new")
    assert embed_speakers_worker.valid_output(stale, "old")
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, emb=np.zeros((1, 256), np.float32), seg=np.array([0]))
    assert not embed_speakers_worker.valid_output(legacy, "old")


def test_map_force_refresh_failure_is_retried(tmp_path: Path, monkeypatch) -> None:
    import map_speakers

    run = tmp_path / "runs" / "diarized"
    run.mkdir(parents=True)
    segs = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}]
    (run / "transcript.json").write_text(json.dumps({"title": "t", "segments": segs}))
    (run / "speaker_map.json").write_text(json.dumps({"speakers": {"SPEAKER_00": {"name": "Old"}}}))
    monkeypatch.setattr(map_speakers, "request_mapping",
                        lambda *a: {"speakers": {"SPEAKER_00": {"name": "Jason Calacanis", "confidence": "high"}}})

    def boom(**kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(map_speakers, "render_all", boom)
    with pytest.raises(RuntimeError):
        map_speakers.process_run(run, None, "key", force=True)
    assert not (run / "speaker_map.json").exists()  # no stale completion marker
    assert (run / "speaker_map.prev.json").exists()
    monkeypatch.undo()
    monkeypatch.setattr(map_speakers, "request_mapping",
                        lambda *a: {"speakers": {"SPEAKER_00": {"name": "Jason Calacanis", "confidence": "high"}}})
    assert map_speakers.process_run(run, None, "key", force=False) == "done"
    assert json.loads((run / "speaker_map.json").read_text())["speakers"]["SPEAKER_00"]["name"] == "Jason Calacanis"


def test_worker_and_driver_agree_on_bad_shape(tmp_path: Path) -> None:
    import embed_speakers_worker

    bad = tmp_path / "bad.npz"
    np.savez(bad, emb=np.zeros(3, np.float32), seg=np.array([0, 1, 2]), samples=np.array([1, 1, 1]), fp=np.array("fp"))
    assert not embed_speakers_worker.valid_output(bad, "fp")
    assert not embed_speakers.valid_npz(bad, "fp")


def test_repair_refuses_stale_map(tmp_path: Path, monkeypatch) -> None:
    from speaker_common import labels_fingerprint

    run = tmp_path / "runs" / "diarized"
    run.mkdir(parents=True)
    segs = [{"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01"}]
    old = [dict(segs[0], speaker="SPEAKER_01"), dict(segs[1], speaker="SPEAKER_00")]
    (run / "transcript.json").write_text(json.dumps({"title": "t", "segments": segs}))
    stale = {"speakers": {"SPEAKER_00": {"name": "Alice"}, "SPEAKER_01": {"name": "SPEAKER_01"}},
             "labels_fp": labels_fingerprint(old)}
    (run / "speaker_map.json").write_text(json.dumps(stale))
    monkeypatch.setattr(repair_speakers, "call", lambda *a: {"speakers": {"SPEAKER_01": {"name": "Bob"}}})
    with pytest.raises(RuntimeError, match="does not match"):
        repair_speakers.process(run, None, "key")
    assert json.loads((run / "speaker_map.json").read_text())["labels_fp"] == labels_fingerprint(old)
