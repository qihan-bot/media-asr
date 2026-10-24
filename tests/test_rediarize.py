from __future__ import annotations

import json
import subprocess
from pathlib import Path

from media_asr import cli
from media_asr import enrich as E
from media_asr import rediarize as R
from media_asr.db import Segment, Store
from media_asr.render import render_all
from media_asr.transcribe import TranscriptSegment

# title -> (largest label share, model verdict)
EPISODES = {
    "hard merge": (0.95, {"merged": True, "people": 2, "reason": "asks and answers"}),
    "three people": (0.80, {"merged": True, "people": 3, "reason": "two guests"}),
    "long answer": (0.85, {"merged": True, "people": 2, "reason": "guest talks a lot"}),
    "one voice": (0.95, {"merged": False, "people": 1, "reason": "monologue"}),
    "balanced": (0.60, None),  # under --share: never checked
}


def segments(share: float) -> list[dict]:
    first = round(share * 100)
    return [
        {"start": 0.0, "end": float(first), "text": "a", "speaker": "SPEAKER_00"},
        {"start": float(first), "end": 100.0, "text": "b", "speaker": "SPEAKER_01"},
    ]


class FakeLLM:
    def __init__(self) -> None:
        self.titles: list[str] = []

    def chat(self, system: str, prompt: str, max_tokens: int) -> dict:
        title = prompt.split("Episode title: ")[1].split("\n")[0]
        self.titles.append(title)
        return dict(EPISODES[title][1])


def make_library(tmp_path: Path) -> tuple[Store, Path, dict[str, str]]:
    store = Store(tmp_path / "media.db")
    items_dir = tmp_path / "items"
    ids = {}
    for title, (share, _) in EPISODES.items():
        item, _ = store.add_item(
            source=f"https://a/{title}.mp3", source_type="podcast", title=title
        )
        store.save_result(
            item,
            segments=[Segment(0, 1, "a")],
            duration=100.0,
            language="zh",
            model="m",
            transcript_path="t.md",
        )
        (items_dir / item).mkdir(parents=True)
        (items_dir / item / "transcript.json").write_text(json.dumps({"segments": segments(share)}))
        (items_dir / item / "enrich.json").write_text("{}")
        ids[title] = item
    return store, items_dir, ids


def test_rediarize_reruns_hard_merges_and_three_people(tmp_path: Path) -> None:
    store, items_dir, ids = make_library(tmp_path)
    reruns = []

    def rerun(item_id: str, language: str, num_speakers: int):
        reruns.append((item_id, language, num_speakers))
        transcript = items_dir / item_id / "transcript.json"
        transcript.write_text(json.dumps({"segments": segments(0.55)}))
        return subprocess.CompletedProcess([], 0, "", "")

    llm = FakeLLM()
    assert R.run(store.conn, llm, items_dir=items_dir, rerun=rerun) == 0
    assert sorted(llm.titles) == ["hard merge", "long answer", "one voice", "three people"]
    assert sorted(reruns) == sorted([(ids["hard merge"], "zh", 3), (ids["three people"], "zh", 4)])
    for title in ("hard merge", "three people"):
        record = json.loads((items_dir / ids[title] / "rediarize.json").read_text())
        assert record["share_after"] == 0.55 and record["label"] == "SPEAKER_00"
        assert not (items_dir / ids[title] / "enrich.json").exists()
    for title in ("long answer", "one voice"):
        record = json.loads((items_dir / ids[title] / "rediarize.json").read_text())
        assert "share_after" not in record and record["merged"] == EPISODES[title][1]["merged"]
        assert (items_dir / ids[title] / "enrich.json").exists()
    assert not (items_dir / ids["balanced"] / "rediarize.json").exists()

    # Marked items are never checked again.
    again = FakeLLM()
    assert R.run(store.conn, again, items_dir=items_dir, rerun=rerun) == 0
    assert again.titles == []
    store.close()


def test_rediarize_failed_rerun_and_dry_run_leave_no_marker(tmp_path: Path) -> None:
    store, items_dir, ids = make_library(tmp_path)

    def broken(item_id: str, language: str, num_speakers: int):
        return subprocess.CompletedProcess([], 1, "", "ssh: connect failed")

    assert R.run(store.conn, FakeLLM(), items_dir=items_dir, dry_run=True, rerun=broken) == 0
    assert not list(items_dir.glob("*/rediarize.json"))
    assert R.run(store.conn, FakeLLM(), items_dir=items_dir, rerun=broken) == 1
    assert not (items_dir / ids["hard merge"] / "rediarize.json").exists()
    assert (items_dir / ids["hard merge"] / "enrich.json").exists()
    assert (items_dir / ids["long answer"] / "rediarize.json").exists()
    store.close()


def test_rediarize_shards_split_the_items(tmp_path: Path) -> None:
    store, items_dir, _ = make_library(tmp_path)
    seen = []
    for shard in ("0/2", "1/2"):
        llm = FakeLLM()
        R.run(store.conn, llm, items_dir=items_dir, dry_run=True, shard=shard)
        seen.append(set(llm.titles))
    assert not seen[0] & seen[1]
    assert seen[0] | seen[1] == {"hard merge", "three people", "long answer", "one voice"}
    store.close()


class FakeRemote:
    """Remote full pipeline stand-in: fails, or writes a new two-speaker transcript."""

    mode, model, user, host = "full", "fake", "fake", "fake"

    def __init__(self, fail: bool) -> None:
        self.fail = fail

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def run_full(self, item, item_dir, **kwargs):
        if self.fail:
            raise RuntimeError("remote failed")
        segs = [
            TranscriptSegment(0.0, 10.0, "new transcript", "SPEAKER_00"),
            TranscriptSegment(10.0, 20.0, "new second speaker", "SPEAKER_01"),
        ]
        render_all(
            out_dir=item_dir,
            item_id=item["id"],
            title="Ep",
            source=item["source"],
            duration=20.0,
            language="en",
            model="asr-2",
            transcribed_at="t",
            segments=segs,
        )
        return {"segments": segs, "language": "en", "duration": 20.0, "model_label": "asr-2"}


class Punctuator:
    model = "fake"

    def chat(self, system: str, user: str, max_tokens: int) -> dict:
        if "label podcast speakers" in system:
            return {"speakers": {}}
        return {"lines": [line + "." for line in user.split("\n")]}


def test_a_new_transcript_makes_later_steps_run_again(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "media.db")
    items_dir = tmp_path / "items"
    item, _ = store.add_item(source="https://a/1.mp3", source_type="podcast", title="Ep")
    segs = [TranscriptSegment(0.0, 10.0, "hello world", "SPEAKER_00")]
    render_all(
        out_dir=items_dir / item,
        item_id=item,
        title="Ep",
        source="https://a/1.mp3",
        duration=10.0,
        language="en",
        model="asr",
        transcribed_at="t",
        segments=segs,
    )
    store.save_result(
        item, segments=segs, duration=10.0, language="en", model="asr", transcript_path="t.md"
    )
    assert E.run(store.conn, Punctuator(), [], items_dir=items_dir) == 0
    (items_dir / item / "terms.json").write_text("{}")  # as if corrected
    monkeypatch.setattr(cli, "ITEMS_DIR", items_dir)
    remote = FakeRemote(fail=True)
    monkeypatch.setattr(cli.RemoteGPU, "from_args", lambda args: remote)
    parser = cli.build_parser()

    def rerun(item_id: str, language: str, num_speakers: int):
        args = parser.parse_args(
            ["run", "--ids", item_id, "--backend", "remote", "--remote-mode", "full", "--diarize"]
        )
        return subprocess.CompletedProcess([], cli.cmd_run(store, args, item_ids=args.ids), "", "")

    class Merged:
        def chat(self, *args) -> dict:
            return {"merged": True, "people": 2, "reason": "asks and answers"}

    # The re-run fails: the item is failed and none of the later markers survive.
    assert R.run(store.conn, Merged(), items_dir=items_dir, rerun=rerun) == 1
    assert store.get_item(item)["status"] == "failed"
    assert not [
        m
        for m in ("rediarize.json", "enrich.json", "terms.json")
        if (items_dir / item / m).exists()
    ]
    # The daily retry transcribes it again; every later step then runs on the new transcript.
    remote.fail = False
    args = parser.parse_args(
        ["run", "--retry-failed", "--backend", "remote", "--remote-mode", "full", "--diarize"]
    )
    assert cli.cmd_run(store, args) == 0
    assert R.run(store.conn, Merged(), items_dir=items_dir, rerun=rerun) == 0  # balanced now
    assert E.run(store.conn, Punctuator(), [], items_dir=items_dir) == 0
    named = json.loads((items_dir / item / "transcript_named.json").read_text())["segments"]
    assert [s["text"] for s in named] == ["new transcript.", "new second speaker."]
    assert [r["text"] for r in store.segments_for(item)] == [s["text"] for s in named]
    assert not (items_dir / item / "terms.json").exists()  # correct-terms runs again
    store.close()


def test_one_bad_episode_does_not_stop_the_others(tmp_path: Path, capsys) -> None:
    store = Store(tmp_path / "media.db")
    items_dir = tmp_path / "items"
    ids = []
    for n in range(5):
        item, _ = store.add_item(source=f"https://a/{n}.mp3", source_type="podcast", title=f"ep{n}")
        store.save_result(
            item,
            segments=[Segment(0, 1, "a")],
            duration=100.0,
            language="en",
            model="m",
            transcript_path="t.md",
        )
        (items_dir / item).mkdir(parents=True)
        (items_dir / item / "transcript.json").write_text(json.dumps({"segments": segments(0.95)}))
        ids.append(item)
    (items_dir / ids[2] / "transcript.json").write_text("{not json")
    answers = {
        "ep0": {"merged": False, "people": "two", "reason": "x"},  # the review's case
        "ep1": {"merged": "false", "people": 1},  # a string, not a boolean
        "ep3": {"merged": True, "people": 3, "reason": "panel"},  # the re-run raises
        "ep4": {"merged": False, "people": 1, "reason": "monologue"},
    }

    class LLM:
        def chat(self, system: str, prompt: str, max_tokens: int) -> dict:
            return answers[prompt.split("Episode title: ")[1].split("\n")[0]]

    def rerun(item_id: str, language: str, num_speakers: int):
        raise FileNotFoundError("media-asr-remote")

    assert R.run(store.conn, LLM(), items_dir=items_dir, rerun=rerun) == 1
    marked = [n for n, item in enumerate(ids) if (items_dir / item / "rediarize.json").exists()]
    assert marked == [4]
    out = capsys.readouterr().out
    assert out.count("FAIL") == 4
    assert "people is not a number: 'two'" in out and "malformed verdict" in out
    store.close()


def one_episode(tmp_path: Path, share: float = 0.95) -> tuple[Store, Path, str]:
    store = Store(tmp_path / "media.db")
    items_dir = tmp_path / "items"
    item, _ = store.add_item(source="https://a/1.mp3", source_type="podcast", title="Ep")
    store.save_result(
        item,
        segments=[Segment(0, 1, "a")],
        duration=100.0,
        language="en",
        model="m",
        transcript_path="t.md",
    )
    (items_dir / item).mkdir(parents=True)
    (items_dir / item / "transcript.json").write_text(json.dumps({"segments": segments(share)}))
    return store, items_dir, item


class Verdict:
    """Answers with a fixed verdict, after letting ``meanwhile`` play another run."""

    def __init__(self, verdict: dict, meanwhile=lambda: None) -> None:
        self.verdict, self.meanwhile = verdict, meanwhile

    def chat(self, *args) -> dict:
        self.meanwhile()
        return dict(self.verdict)


HARD_MERGE = {"merged": True, "people": 2, "reason": "asks and answers"}


def test_no_rerun_on_a_verdict_about_an_older_transcript(tmp_path: Path, capsys) -> None:
    store, items_dir, item = one_episode(tmp_path)
    transcript = items_dir / item / "transcript.json"

    def retranscribed() -> None:  # another run wrote a new transcript meanwhile
        transcript.write_text(json.dumps({"segments": segments(0.55)}))

    reruns = []
    llm = Verdict(HARD_MERGE, retranscribed)
    assert R.run(store.conn, llm, items_dir=items_dir, rerun=lambda *a: reruns.append(a)) == 1
    assert reruns == [] and not (items_dir / item / "rediarize.json").exists()
    assert "transcript.json changed meanwhile" in capsys.readouterr().out
    store.close()


def test_no_rerun_when_another_run_decided_meanwhile(tmp_path: Path) -> None:
    store, items_dir, item = one_episode(tmp_path)
    marker = items_dir / item / "rediarize.json"
    llm = Verdict(HARD_MERGE, lambda: marker.write_text('{"by": "other run"}'))
    reruns = []
    assert R.run(store.conn, llm, items_dir=items_dir, rerun=lambda *a: reruns.append(a)) == 0
    assert reruns == [] and json.loads(marker.read_text()) == {"by": "other run"}
    store.close()


def test_newer_state_after_the_rerun_is_kept(tmp_path: Path, monkeypatch) -> None:
    store, items_dir, item = one_episode(tmp_path)
    item_dir = items_dir / item
    marker = item_dir / "rediarize.json"

    def rerun(item_id: str, language: str, num_speakers: int):
        (item_dir / "transcript.json").write_text(json.dumps({"segments": segments(0.6)}))
        marker.write_text('{"by": "other run"}')  # another run checked the new transcript
        return subprocess.CompletedProcess([], 0, "", "")

    assert R.run(store.conn, Verdict(HARD_MERGE), items_dir=items_dir, rerun=rerun) == 0
    assert json.loads(marker.read_text()) == {"by": "other run"}

    # Another run re-transcribes between the re-run and the marker: no marker.
    marker.unlink()
    (item_dir / "transcript.json").write_text(json.dumps({"segments": segments(0.95)}))

    def rerun_only(item_id: str, language: str, num_speakers: int):
        (item_dir / "transcript.json").write_text(json.dumps({"segments": segments(0.6)}))
        return subprocess.CompletedProcess([], 0, "", "")

    real_lock, entries = R.item_lock, []

    def lock(path: Path):
        entries.append(path)
        if len(entries) == 2:  # the lock after the re-run
            (item_dir / "transcript.json").write_text(json.dumps({"segments": segments(0.7)}))
        return real_lock(path)

    monkeypatch.setattr(R, "item_lock", lock)
    assert R.run(store.conn, Verdict(HARD_MERGE), items_dir=items_dir, rerun=rerun_only) == 0
    assert len(entries) == 2 and not marker.exists()
    store.close()
