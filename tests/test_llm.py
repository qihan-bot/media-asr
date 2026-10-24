from __future__ import annotations

import pytest

from media_asr import llm as L


class Response:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.content = content
        self.finish_reason = finish_reason

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        message = {"content": self.content}
        return {"choices": [{"message": message, "finish_reason": self.finish_reason}]}


class Client:
    """Stands in for httpx.Client: answers posts from a list."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.payloads: list[dict] = []

    def post(self, url, *, headers, json, timeout):
        assert url == L.API_URL and headers["Authorization"] == "Bearer key"
        self.payloads.append(dict(json))
        return self.answers.pop(0)

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch) -> None:
    monkeypatch.setattr(L.time, "sleep", lambda seconds: None)


def test_chat_retries_sampled_and_parses_fenced_json() -> None:
    client = Client(Response("not json"), Response('```json\n{"ok": 1}\n```'))
    assert L.LLM("key", model="m", client=client).chat("system", "user", 100) == {"ok": 1}
    assert [p["temperature"] for p in client.payloads] == [0, 0.3]
    first = client.payloads[0]
    assert first["model"] == "m" and first["max_tokens"] == 100
    assert first["response_format"] == {"type": "json_object"}


def test_chat_reports_an_answer_cut_off_at_max_tokens() -> None:
    client = Client(*[Response('{"fixes": [{"wrong": "Skad', "length")] * 3)
    with pytest.raises(L.TruncatedResponse, match="DeepSeek request failed"):
        L.LLM("key", client=client).chat("system", "user", 10)
    assert len(client.payloads) == 3


def test_chat_other_failures_are_not_truncation() -> None:
    client = Client(*[Response("{bad", "stop")] * 3)
    with pytest.raises(RuntimeError) as info:
        L.LLM("key", client=client).chat("system", "user", 10)
    assert not isinstance(info.value, L.TruncatedResponse)


def test_complete_json_at_the_limit_is_accepted() -> None:
    client = Client(Response('{"fixes": []}', "length"))
    assert L.LLM("key", client=client).chat("system", "user", 10) == {"fixes": []}


def test_model_from_environment_and_direct_connection(monkeypatch) -> None:
    monkeypatch.setenv("MEDIA_ASR_LLM_MODEL", "deepseek-pro")
    assert L.LLM("key", client=Client()).model == "deepseek-pro"
    monkeypatch.delenv("MEDIA_ASR_LLM_MODEL")
    with L.LLM("key") as llm:
        assert llm.model == "deepseek-flash"
        assert llm.client.trust_env is False
