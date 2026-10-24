from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("wordfreq")
pytest.importorskip("pypinyin")
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
        # pinyin: character pairs need a toneless similarity >= 0.4; three or
        # more syllables tolerate one unlike pair, two syllables none
        ("余昊", "宇树", False),  # hao / shu 0.33
        ("人物智能", "具身智能", False),  # ren / ju and wu / shen: two unlike
        ("巨星智能", "具身智能", True),  # xing / shen 0.25: the one tolerated
        ("波斯文动力", "波士顿动力", True),  # wen / dun 0.33: the one tolerated
        ("巨声", "具身", True),  # sheng / shen 0.89
        ("冷量运输", "冷链运输", True),  # liang / lian 0.89
        ("付费一变不强", "付费意愿不强", True),  # bian / yuan 0.5
        ("舍负千科", "舍甫琴科", True),  # qian / qin 0.86
        # English spelling similarity >= 0.55 (was 0.5)
        ("Codecs", "Claude Code", False),  # 0.5
        ("Druckner", "Druckenmiller", True),  # 0.76
        ("Chairpoint", "Checkpoint", True),  # 0.7
    ],
)
def test_sound_problem(wrong: str, right: str, ok: bool) -> None:
    assert (C.sound_problem(wrong, right, SPEAKERS, "high") is None) is ok


def test_sound_problem_reasons() -> None:
    assert C.sound_problem("余昊", "宇树", SPEAKERS, "high") == "pinyin too different"
    assert C.sound_problem("Codecs", "Claude Code", SPEAKERS, "high") == "spelling too different"
    assert C.sound_problem("小俊", "张小珺", SPEAKERS, "high") == (
        "Chinese pair changes the syllable count"
    )
    assert C.sound_problem("GPT五", "GPT-5", SPEAKERS, "medium") == (
        "mixed-script pair without high confidence"
    )
    assert C.sound_problem("GPT五", "GPT-5", SPEAKERS, "high") is None


def test_pinyin_rule_uses_every_reading_and_only_chinese_pairs() -> None:
    assert {"pu", "piao"} <= C.readings("朴")
    # pypinyin reads 质朴 in context as "zhi piao"; 智谱 (Zhipu) is "zhi pu".
    assert not C.pinyin_mismatch("质朴", "智谱")
    assert not C.pinyin_mismatch("第3代", "第三代")  # the digit is not compared
    assert C.pinyin_mismatch("余昊", "宇树")


def test_pinyin_rule_tolerates_one_unlike_syllable_in_three_or_more() -> None:
    assert not C.pinyin_mismatch("达蒙院", "达摩院")  # meng / mo, 1 of 3
    assert C.pinyin_mismatch("地基学习", "深度学习")  # di / shen, ji / du: 2 of 4
    assert C.pinyin_mismatch("余昊3", "宇树3")  # 2 syllables: a digit does not count


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
    # 5 attempts on the cut-off 4-line chunk, then one call per half.
    assert [len(chunk) for chunk in client.chunks] == [4, 4, 4, 4, 4, 2, 2]
    report = json.loads((items_dir / item / "terms.json").read_text())
    assert [(f["wrong"], f["count"]) for f in report["applied"]] == [("Skading Law", 3)]
    assert report["replacements"] == 3
    named = json.loads((items_dir / item / "transcript_named.json").read_text())
    assert [s["text"] for s in named["segments"]][0] == "Scaling Law one."
    assert (
        named["model"]
        == "whisper + deepseek-flash punctuation/speaker names + term correction (deepseek-v4.1-flash-tp4)"
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


class FixedClient:
    """DeepSeek stand-in that proposes the same fixes for every chunk."""

    def __init__(self, fixes: list[tuple[str, str]]):
        self.fixes = [{"wrong": w, "right": r, "confidence": "high"} for w, r in fixes]

    def post(self, url, *, headers, json, timeout):
        return Response(json_dumps({"fixes": self.fixes}), "stop")

    def close(self) -> None:
        pass


def correct(tmp_path: Path, texts: list[str], fixes: list[tuple[str, str]]) -> tuple[dict, list]:
    """Run correct-terms on one item; return terms.json and the corrected texts,
    after checking that files and search index agree."""
    store, items_dir, item = make_item(tmp_path, texts)
    with llm.LLM("key", client=FixedClient(fixes)) as model:
        assert C.run(store.conn, model, items_dir=items_dir) == 0
    report = json.loads((items_dir / item / "terms.json").read_text())
    named = json.loads((items_dir / item / "transcript_named.json").read_text())
    out = [s["text"] for s in named["segments"]]
    assert [row["text"] for row in store.segments_for(item)] == out
    assert report["replacements"] == sum(f["count"] for f in report["applied"])
    store.close()
    return report, out


def test_corrections_never_rescan_their_own_output(tmp_path: Path) -> None:
    # Live library, buzzsprout 2b4e61a8: "Rota AI" became "Rhoda AI AI".
    fixes = [("Rota AI", "Rhoda AI"), ("Rhoda", "Rhoda AI")]
    report, out = correct(tmp_path, ["Rota AI and Rhoda."], fixes)
    assert out == ["Rhoda AI and Rhoda AI."]
    assert {f["wrong"]: f["count"] for f in report["applied"]} == {"Rota AI": 1, "Rhoda": 1}


def test_already_correct_text_is_left_alone(tmp_path: Path) -> None:
    report, out = correct(
        tmp_path, ["Rhoda AI builds robots; Rhoda ships."], [("Rhoda", "Rhoda AI")]
    )
    assert out == ["Rhoda AI builds robots; Rhoda AI ships."]
    assert report["applied"][0]["count"] == 1


def test_audit_counts_replacements_actually_made(tmp_path: Path) -> None:
    report, out = correct(
        tmp_path, ["冷量运输。冷量运输"], [("冷量运输", "冷链运输"), ("冷量", "冷链")]
    )
    assert out == ["冷链运输。冷链运输"]
    assert [(f["wrong"], f["count"]) for f in report["applied"]] == [("冷量运输", 2)]
    assert report["replacements"] == 2
    (unused,) = [r for r in report["rejected"] if r["wrong"] == "冷量"]
    assert unused["rejected"] == "every occurrence is part of a longer correction or a correct form"
    assert "count" not in unused and "tier" not in unused


@pytest.mark.parametrize(
    ("wrong", "right"),
    [("Mathbb", r"\mathbb"), ("nvdia-smi", r"\nvidia-smi"), ("Grupe", r"Grupe\1")],
)
def test_right_form_is_inserted_literally(tmp_path: Path, wrong: str, right: str) -> None:
    report, out = correct(tmp_path, [f"{wrong} and {wrong}."], [(wrong, right)])
    assert out == [f"{right} and {right}."]
    assert report["applied"][0]["right"] == right and report["applied"][0]["count"] == 2


def test_apply_fixes_is_one_pass_longest_first() -> None:
    segments = [{"text": "Open Cloud Moment, Open Cloud"}, {"text": "no match"}]
    accepted = [
        {"wrong": "Open Cloud", "right": "OpenClaw", "tier": "model", "count": 2},
        {"wrong": "Open Cloud Moment", "right": "OpenClaw Moment", "tier": "model", "count": 1},
    ]
    applied, unused = C.apply_fixes(segments, accepted)
    assert [s["text"] for s in segments] == ["OpenClaw Moment, OpenClaw", "no match"]
    assert {f["wrong"]: f["count"] for f in applied} == {"Open Cloud": 1, "Open Cloud Moment": 1}
    assert unused == [] and C.apply_fixes(segments, []) == ([], [])


class LockedDB:
    """A connection whose index writes fail, as when SQLite is locked."""

    def __init__(self, conn) -> None:
        self.conn = conn

    def __getattr__(self, name: str):
        return getattr(self.conn, name)

    def __enter__(self):
        return self.conn.__enter__()

    def __exit__(self, *exc):
        return self.conn.__exit__(*exc)

    def executemany(self, *args):
        raise sqlite3.OperationalError("database is locked")


def test_failed_forced_correction_leaves_the_item_to_be_redone(tmp_path: Path) -> None:
    store, items_dir, item = make_item(tmp_path, ["Skading Law."])
    fixes = [("Skading Law", "Scaling Law")]
    with llm.LLM("key", client=FixedClient(fixes)) as model:
        assert C.run(store.conn, model, items_dir=items_dir) == 0
        assert C.run(LockedDB(store.conn), model, force=True, items_dir=items_dir) == 1
        assert not (items_dir / item / "terms.json").exists()
        assert (items_dir / item / "enrich.json").exists()
        assert C.run(store.conn, model, items_dir=items_dir) == 0
    named = json.loads((items_dir / item / "transcript_named.json").read_text())["segments"]
    assert [s["text"] for s in named] == [r["text"] for r in store.segments_for(item)]
    assert named[0]["text"] == "Scaling Law."
    store.close()


def test_correction_refuses_a_named_base_that_changed(tmp_path: Path) -> None:
    store, items_dir, item = make_item(tmp_path, ["Skading Law."])
    row = store.get_item(item)
    with llm.LLM("key", client=FixedClient([("Skading Law", "Scaling Law")])) as model:
        result = C.correct_item(row, {}, {}, {}, model, items_dir)
    base = items_dir / item / "named_base.json"
    base.write_text(base.read_text().replace("Skading", "Skating"))
    with pytest.raises(RuntimeError, match="enriched again"):
        C.commit(store.conn, result, items_dir)
    assert not (items_dir / item / "terms.json").exists()
    store.close()
