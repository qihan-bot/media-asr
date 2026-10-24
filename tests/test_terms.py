from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("wordfreq")
from media_asr import llm  # noqa: E402
from media_asr import terms as C  # noqa: E402
from media_asr.db import Segment, Store  # noqa: E402

SPEAKERS = {"罗福莉", "张小珺", "泓君"}


@pytest.mark.parametrize(
    ("wrong", "right", "ok"),
    [
        ("罗福利", "罗福莉", True),  # same syllable count
        ("红军", "泓君", True),  # common word, but a misheard speaker name
        ("开始临界", "开始连接", True),
        ("Skading Law", "Scaling Law", True),
        ("VOA", "VLA", True),  # short tokens: 0.67 >= 0.65
        ("福利", "罗福莉", False),  # syllable count changes, common word
        ("小俊", "张小珺", False),  # expands a nickname
        ("skew", "SKU", False),  # common English word
        ("Junark", "SF Compute", False),  # does not sound alike
        ("OE", "o1", False),  # short token, too loose
    ],
)
def test_sound_problem(wrong: str, right: str, ok: bool) -> None:
    assert (C.sound_problem(wrong, right, SPEAKERS, "high") is None) is ok


def test_sound_problem_reasons() -> None:
    assert C.sound_problem("Junark", "SF Compute", SPEAKERS, "high") == "spelling too different"
    assert C.sound_problem("小俊", "张小珺", SPEAKERS, "high") == (
        "Chinese pair changes the syllable count"
    )
    assert C.sound_problem("GPT五", "GPT-5", SPEAKERS, "medium") == (
        "mixed-script pair without high confidence"
    )
    assert C.sound_problem("GPT五", "GPT-5", SPEAKERS, "high") is None


def test_select_tiers_and_filters() -> None:
    text = "罗福利说 Skading Law 和 Skading Law，还有福利。"
    fixes = [
        {"wrong": "罗福利", "right": "罗福莉", "confidence": "medium"},
        {"wrong": "Skading Law", "right": "Scaling Law", "confidence": "high"},
        {"wrong": "福利", "right": "罗福莉", "confidence": "high"},
        {"wrong": "不存在", "right": "存在", "confidence": "high"},
    ]
    accepted, rejected = C.select(fixes, text, "罗福莉 访谈", SPEAKERS)
    got = {(a["wrong"], a["tier"], a["count"]) for a in accepted}
    assert got == {("罗福利", "reference", 1), ("Skading Law", "model", 2)}
    assert {r["wrong"] for r in rejected} == {"福利", "不存在"}


@pytest.mark.parametrize(
    ("fix", "reference", "outcome"),
    [
        ({"right": "Scaling Law", "confidence": "medium"}, "scaling law", "reference"),
        ({"right": "Scaling Law", "confidence": "high"}, "", "model"),
        ({"right": "Scaling Law", "confidence": "medium"}, "", "not high confidence"),
        ({"right": "Scaling Law", "confidence": "low"}, "scaling law", "not high confidence"),
        ({"right": "Scaling Law", "confidence": "high"}, "Scaling Law", "reference"),
        ({"right": "Skading Law", "confidence": "high"}, "", "empty or too short"),
        ({"right": "", "confidence": "high"}, "", "empty or too short"),
    ],
)
def test_select_tier_rules(fix: dict, reference: str, outcome: str) -> None:
    accepted, rejected = C.select(
        [{"wrong": "Skading Law", **fix}], "Skading Law.", reference, set()
    )
    if accepted:
        assert accepted[0]["tier"] == outcome
    else:
        assert rejected[0]["rejected"] == outcome


def test_select_drops_ambiguous_and_duplicate_pairs() -> None:
    fixes = [
        {"wrong": "Calcode", "right": "Claude Code", "confidence": "high"},
        {"wrong": "Calcode", "right": "Claude Code", "confidence": "high"},
        {"wrong": "Skading", "right": "Scaling", "confidence": "high"},
        {"wrong": "Skading", "right": "Scalding", "confidence": "high"},
    ]
    accepted, rejected = C.select(fixes, "Calcode and Skading", "", set())
    assert [(a["wrong"], a["count"]) for a in accepted] == [("Calcode", 1)]
    assert [r["rejected"] for r in rejected] == ["ambiguous: several corrections"] * 2


def test_ascii_pattern_matches_whole_words_only() -> None:
    assert C.pattern("VOA").sub("VLA", "VOA and VOAX") == "VLA and VOAX"


class Response:
    def __init__(self, content: str, finish_reason: str):
        self.payload = {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
        }

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


class TruncatingClient:
    """DeepSeek stand-in: an answer for more than ``fits`` lines is cut off at
    max_tokens; otherwise one fix per "Skading Law" line."""

    def __init__(self, fits: int):
        self.fits = fits
        self.chunks: list[list[str]] = []

    def post(self, url, *, headers, json, timeout):
        assert json["max_tokens"] == C.MAX_TOKENS
        prompt = json["messages"][1]["content"]
        lines = re.search(r"Transcript chunk:\n(.*?)\n\nList only", prompt, re.S)[1].split("\n")
        self.chunks.append(lines)
        if len(lines) > self.fits:
            return Response('{"fixes": [{"wrong": "Skading Law", "right": "Scal', "length")
        fixes = [
            {"wrong": "Skading Law", "right": "Scaling Law", "confidence": "high"}
            for line in lines
            if "Skading Law" in line
        ]
        return Response(json_dumps({"fixes": fixes}), "stop")

    def close(self) -> None:
        pass


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def make_item(tmp_path: Path, texts: list[str]) -> tuple[Store, Path, str]:
    store = Store(tmp_path / "media.db")
    item, _ = store.add_item(
        source="https://a/1.mp3", source_type="podcast", title="Ep 1", feed_url="https://a/rss"
    )
    segments = [Segment(float(i), i + 1.0, text, speaker="Host") for i, text in enumerate(texts)]
    store.save_result(
        item, segments=segments, duration=9.0, language="en", model="m", transcript_path="t.md"
    )
    items_dir = tmp_path / "items"
    item_dir = items_dir / item
    item_dir.mkdir(parents=True)
    base = [
        {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker} for s in segments
    ]
    (item_dir / "named_base.json").write_text(json_dumps(base))
    named = {
        "id": item,
        "title": "Ep 1",
        "source": "https://a/1.mp3",
        "duration": 9.0,
        "language": "en",
        "model": "whisper + deepseek-flash punctuation/speaker names",
        "segments": base,
    }
    (item_dir / "transcript_named.json").write_text(json_dumps(named))
    (item_dir / "enrich.json").write_text("{}")
    return store, items_dir, item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


def test_cut_off_answer_splits_the_chunk_instead_of_failing(tmp_path: Path) -> None:
    texts = ["Skading Law one.", "Nothing here.", "Skading Law two.", "More Skading Law."]
    store, items_dir, item = make_item(tmp_path, texts)
    client = TruncatingClient(fits=2)
    with llm.LLM("key", client=client) as model:
        assert C.run(store.conn, model, items_dir=items_dir) == 0
    # 3 attempts on the cut-off 4-line chunk, then one call per half.
    assert [len(chunk) for chunk in client.chunks] == [4, 4, 4, 2, 2]
    report = json.loads((items_dir / item / "terms.json").read_text())
    assert [(f["wrong"], f["count"]) for f in report["applied"]] == [("Skading Law", 3)]
    assert report["replacements"] == 3
    named = json.loads((items_dir / item / "transcript_named.json").read_text())
    assert [s["text"] for s in named["segments"]][0] == "Scaling Law one."
    assert (
        named["model"]
        == "whisper + deepseek-flash punctuation/speaker names + term correction (deepseek-flash)"
    )
    assert [row["text"] for row in store.segments_for(item)][3] == "More Scaling Law."
    store.close()


def test_single_line_cut_off_fails_the_episode(tmp_path: Path, capsys) -> None:
    store, items_dir, item = make_item(tmp_path, ["Skading Law one.", "Skading Law two."])
    with llm.LLM("key", client=TruncatingClient(fits=0)) as model:
        assert C.run(store.conn, model, items_dir=items_dir) == 1
    assert not (items_dir / item / "terms.json").exists()
    assert "FAIL" in capsys.readouterr().out
    store.close()
