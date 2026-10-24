from __future__ import annotations

import pytest

pytest.importorskip("wordfreq")
from media_asr import terms as C  # noqa: E402

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
