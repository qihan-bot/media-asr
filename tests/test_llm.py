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
        assert url == L.DEFAULT_URL and headers["Authorization"] == "Bearer key"
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
    client = Client(*[Response('{"fixes": [{"wrong": "Skad', "length")] * 5)
    with pytest.raises(L.TruncatedResponse, match="LLM request failed"):
        L.LLM("key", client=client).chat("system", "user", 10)
    assert len(client.payloads) == 5


def test_chat_other_failures_are_not_truncation() -> None:
    client = Client(*[Response("{bad", "stop")] * 5)
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
        assert llm.model == "deepseek-v4.1-flash-tp4"
        assert llm.client.trust_env is False


@pytest.mark.parametrize(
    "url",
    [
        "https://api.deepseek.com/chat/completions",
        "http://8.8.8.8/v1/chat/completions",
        "http://example.com/v1",
        "http://169.254.1.1/v1",
    ],
)
def test_cloud_or_public_endpoints_are_refused(monkeypatch, url: str) -> None:
    monkeypatch.setenv("MEDIA_ASR_LLM_URL", url)
    with pytest.raises(L.EndpointNotAllowed):
        L.LLM("key", client=Client())


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.100.161:8889/v1/chat/completions",
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
    ],
)
def test_lan_endpoints_are_allowed(monkeypatch, url: str) -> None:
    monkeypatch.setenv("MEDIA_ASR_LLM_URL", url)
    assert L.LLM("key", client=Client()).url == url


def test_concurrency_is_capped_across_threads(monkeypatch) -> None:
    import threading

    monkeypatch.setattr(L, "_SLOTS", threading.BoundedSemaphore(2))
    active, peak, lock = [0], [0], threading.Lock()

    class Slow:
        def post(self, *a, **k):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            threading.Event().wait(0.05)  # time.sleep is patched out in this module
            with lock:
                active[0] -= 1
            return Response('{"ok": 1}', "stop")

        def close(self):
            pass

    llm = L.LLM("key", client=Slow())
    threads = [threading.Thread(target=llm.chat, args=("s", "u", 10)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2
