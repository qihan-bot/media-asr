from __future__ import annotations

import json
import subprocess
from pathlib import Path

from media_asr import rediarize as R
from media_asr.db import Segment, Store

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
