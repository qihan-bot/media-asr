from __future__ import annotations

from media_asr.remote_scripts.remote_worker import is_hallucination


def test_normal_subscription_text_is_kept() -> None:
    assert not is_hallucination("公司的订阅收入增长百分之二十")


def test_credit_lines_are_dropped() -> None:
    assert is_hallucination("字幕志愿者 杨茜茜")
    assert is_hallucination("請不吝點贊訂閱轉發打賞支持明鏡與點點欄目")
    assert is_hallucination("Subtitles by the community")