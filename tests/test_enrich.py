from __future__ import annotations

from media_asr import enrich as E


class PunctuatingLLM:
    """Adds punctuation, but "corrects" one word: that segment must be kept as is."""

    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system: str, user: str, max_tokens: int) -> dict:
        self.calls += 1
        lines = [line.replace("teh", "the").capitalize() + "." for line in user.split("\n")]
        return {"lines": lines}


def test_punctuation_never_changes_words() -> None:
    texts = ["hello world", "teh cat", "  ", "it is 3 pm"]
    out, accepted = E.punctuate(PunctuatingLLM(), texts, "en", workers=2)
    assert out == ["Hello world.", "teh cat", "  ", "It is 3 pm."]
    assert accepted == 2


def test_norm_ignores_punctuation_case_and_width() -> None:
    assert E.norm("Ｈello, World！") == E.norm("hello world")
    assert E.norm("你好，世界。") == "你好世界"
