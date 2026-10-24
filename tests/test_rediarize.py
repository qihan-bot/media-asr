from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from media_asr import cli
from media_asr import enrich as E
from media_asr import rediarize as R
from media_asr.db import Segment, Store
from media_asr.library import file_digest
from media_asr.render import render_all
from media_asr.transcribe import TranscriptSegment
from media_asr.util import file_lock, item_lock

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
        return file_digest(transcript)

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
        raise RuntimeError("ssh: connect failed")

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

    def rerun(item_id: str, language: str, num_speakers: int) -> str:
        return cli._rerun_item(store, item_id, language, num_speakers)

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
    assert "people is not a count of at least 1: 'two'" in out and "malformed verdict" in out
    store.close()


def label_share(transcript: Path) -> float:
    """Largest label's share of a transcript.json."""
    return R.label_shares(json.loads(transcript.read_text())["segments"]).most_common(1)[0][1]


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


def rerun_episode(tmp_path: Path, monkeypatch, remote: FakeRemote):
    """An episode that needs a re-run, and the real in-process re-run on ``remote``."""
    store, items_dir, item = one_episode(tmp_path)
    monkeypatch.setattr(cli, "ITEMS_DIR", items_dir)
    monkeypatch.setattr(cli.RemoteGPU, "from_args", lambda args: remote)

    def rerun(item_id: str, language: str, num_speakers: int) -> str:
        return cli._rerun_item(store, item_id, language, num_speakers)

    return store, items_dir / item, rerun


def test_no_other_run_gets_in_between_check_and_rerun(tmp_path: Path, monkeypatch) -> None:
    # Review r3, before_rerun: another run takes the episode right after the
    # check; the stale verdict then re-ran and overwrote its newer transcript.
    events: list[str] = []

    class Remote(FakeRemote):
        def run_full(self, item, item_dir, **kwargs):
            events.append(f"rerun with {kwargs['num_speakers']} speakers")
            return super().run_full(item, item_dir, **kwargs)

    store, item_dir, rerun = rerun_episode(tmp_path, monkeypatch, Remote(fail=False))
    marker, transcript = item_dir / "rediarize.json", item_dir / "transcript.json"
    seen: dict = {}

    def another_run() -> None:
        with item_lock(item_dir):  # every writer honors the item lock
            seen["marker"] = json.loads(marker.read_text())
            events.append("other run")
            transcript.write_text(json.dumps({"segments": segments(0.3)}))
            marker.write_text('{"by": "other run"}')

    other = threading.Thread(target=another_run)
    real_lock = R.item_lock

    @contextmanager
    def lock_then_other_run(path: Path):
        with real_lock(path) as held:
            other.start()  # wants the episode now, right after the check
            yield held

    monkeypatch.setattr(R, "item_lock", lock_then_other_run)
    assert R.run(store.conn, Verdict(HARD_MERGE), items_dir=item_dir.parent, rerun=rerun) == 0
    other.join(5)
    assert events == ["rerun with 3 speakers", "other run"]
    # It got the episode only after this run's marker, and its newer state stands.
    assert seen["marker"]["share_after"] == 0.5
    assert json.loads(marker.read_text()) == {"by": "other run"}
    assert label_share(transcript) == 0.7
    store.close()


def test_marker_records_the_transcript_this_rerun_wrote(tmp_path: Path, monkeypatch) -> None:
    # Review r3, after_rerun: another run wrote a newer transcript and
    # enrich.json after the re-run; its share was recorded as the re-run's and
    # its enrich.json deleted.
    def another_run() -> None:
        with item_lock(item_dir):
            transcript.write_text(json.dumps({"segments": segments(0.99)}))
            (item_dir / "enrich.json").write_text('{"by": "other run"}')

    other = threading.Thread(target=another_run)

    class Remote(FakeRemote):
        def run_full(self, item, item_dir, **kwargs):
            with file_lock(item_dir / ".lock", wait=False) as free:
                assert not free  # the re-run holds the episode's lock
            output = super().run_full(item, item_dir, **kwargs)
            other.start()  # wants the episode right after the re-run's output
            time.sleep(0.2)
            return output

    store, item_dir, rerun = rerun_episode(tmp_path, monkeypatch, Remote(fail=False))
    transcript = item_dir / "transcript.json"
    assert R.run(store.conn, Verdict(HARD_MERGE), items_dir=item_dir.parent, rerun=rerun) == 0
    other.join(5)
    record = json.loads((item_dir / "rediarize.json").read_text())
    assert record["share_before"] == 0.95 and record["share_after"] == 0.5
    assert json.loads((item_dir / "enrich.json").read_text()) == {"by": "other run"}
    assert label_share(transcript) == 0.99
    store.close()


def test_rerun_needs_a_configured_gpu(tmp_path: Path, monkeypatch) -> None:
    store, items_dir, item = one_episode(tmp_path)
    monkeypatch.delenv("MEDIA_ASR_REMOTE_HOST", raising=False)
    with pytest.raises(RuntimeError, match="media-asr-remote rediarize"):
        cli._rerun_item(store, item, "en", 3)
    store.close()


@pytest.mark.parametrize("people", [None, False, True, [], 0, -1, "2", 2.5])
@pytest.mark.parametrize("merged", [True, False])
def test_invalid_people_fails_the_episode(tmp_path: Path, merged: bool, people) -> None:
    store, items_dir, item = one_episode(tmp_path)
    reruns = []
    llm = Verdict({"merged": merged, "people": people, "reason": "x"})
    assert R.run(store.conn, llm, items_dir=items_dir, rerun=lambda *a: reruns.append(a)) == 1
    assert reruns == [] and not (items_dir / item / "rediarize.json").exists()
    store.close()


def test_checks_run_in_parallel_and_reruns_one_at_a_time(tmp_path: Path, capsys) -> None:
    guard = threading.Lock()
    outputs, peaks = [], []
    for workers in (1, 4):
        store, items_dir, ids = make_library(tmp_path / str(workers))
        running = {"check": 0, "rerun": 0}
        peak = {"check": 0, "rerun": 0}

        def enter(kind: str) -> None:
            with guard:
                running[kind] += 1
                peak[kind] = max(peak[kind], running[kind])
            time.sleep(0.05)
            with guard:
                running[kind] -= 1

        class SlowLLM(FakeLLM):
            def chat(self, system: str, prompt: str, max_tokens: int) -> dict:
                enter("check")
                return super().chat(system, prompt, max_tokens)

        reruns = []

        def rerun(item_id: str, language: str, num_speakers: int):
            enter("rerun")
            reruns.append(item_id)
            transcript = items_dir / item_id / "transcript.json"
            transcript.write_text(json.dumps({"segments": segments(0.55)}))
            return file_digest(transcript)

        assert R.run(store.conn, SlowLLM(), items_dir=items_dir, rerun=rerun, workers=workers) == 0
        assert sorted(reruns) == sorted([ids["hard merge"], ids["three people"]])
        marked = {t for t, item in ids.items() if (items_dir / item / "rediarize.json").exists()}
        assert marked == {"hard merge", "three people", "long answer", "one voice"}
        outputs.append(capsys.readouterr().out)
        peaks.append(peak)
        store.close()
    assert peaks == [{"check": 1, "rerun": 1}, {"check": 4, "rerun": 1}]
    assert outputs[0] == outputs[1]  # verdicts are reported and acted on in order
