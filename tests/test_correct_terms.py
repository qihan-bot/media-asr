from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("wordfreq")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import correct_terms as C  # noqa: E402

SPEAKERS = {"罗福莉", "张小珺", "泓君"}


@pytest.mark.parametrize(
    ("wrong", "right", "ok"),
    [
        ("罗福利", "罗福莉", True),  # same syllable count
        ("红军", "泓君", True),  # common word, but a misheard speaker name
        ("开始临界", "开始连接", True),
        ("Skading Law", "Scaling Law", True),
        ("VOA", "VLA", True),
        ("福利", "罗福莉", False),  # syllable count changes, common word
        ("小俊", "张小珺", False),  # expands a nickname
        ("skew", "SKU", False),  # common English word
        ("Junark", "SF Compute", False),  # does not sound alike
        ("OE", "o1", False),  # short token, too loose
    ],
)
def test_sound_problem(wrong: str, right: str, ok: bool) -> None:
    assert (C.sound_problem(wrong, right, SPEAKERS, "high") is None) is ok


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


def test_ascii_pattern_matches_whole_words_only() -> None:
    assert C.pattern("VOA").sub("VLA", "VOA and VOAX") == "VLA and VOAX"
