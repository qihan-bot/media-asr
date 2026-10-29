"""Chat client shared by the podcast library steps.

Default: the local DeepSeek on the DGX (OpenAI-compatible). User rule 2026-10-24: library LLM work runs on
the LAN model, never the DeepSeek cloud API. ``MEDIA_ASR_LLM_URL`` / ``MEDIA_ASR_LLM_MODEL`` override.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
import time
from urllib.parse import urlsplit

import httpx

DEFAULT_URL = "http://192.168.31.161:8889/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4.1-flash-tp4"
ATTEMPTS = 5
# Requests in flight to the shared model from this process, however the callers nest their thread pools.
_SLOTS = threading.BoundedSemaphore(
    max(1, int(os.environ.get("MEDIA_ASR_LLM_MAX_CONCURRENCY", "4")))
)


class EndpointNotAllowed(ValueError):
    """The configured endpoint is not on this machine or the LAN."""


def check_endpoint(url: str) -> str:
    """Accept only loopback or private-network endpoints (user rule: no cloud LLM for the library)."""
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return url
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise EndpointNotAllowed(
            f"LLM endpoint must be a LAN IP or localhost, not {host!r}"
        ) from None
    if not (address.is_private or address.is_loopback) or address.is_link_local:
        raise EndpointNotAllowed(f"LLM endpoint must be a LAN IP or localhost, not {host!r}")
    return url


class TruncatedResponse(RuntimeError):
    """The answer hit ``max_tokens`` before its JSON was complete."""


def text_answer(content: str | None, finish_reason: str | None = None) -> str:
    """A plain-text answer: cut off at ``max_tokens`` or empty counts as failed."""
    if finish_reason == "length":
        raise TruncatedResponse("answer cut off at max_tokens")
    if not (content or "").strip():
        raise ValueError("empty answer")
    return content


def parse_json(content: str, finish_reason: str | None = None):
    """Parse a JSON answer, tolerating a Markdown code fence around it.

    DeepSeek on the DGX sometimes stops (finish_reason "stop") one or two closing
    braces short of a nested object; only that case, an answer that ends early,
    is repaired by appending the missing closers.
    """
    text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if finish_reason == "length":
            raise TruncatedResponse(f"answer cut off at max_tokens: {exc}") from exc
        if exc.pos >= len(text):
            for suffix in ("}", "}}", "]}", "}]}", "}}}"):
                try:
                    return json.loads(text + suffix)
                except json.JSONDecodeError:
                    pass
        raise


class LLM:
    """DeepSeek chat in JSON mode, reached directly (``trust_env=False``: no proxy).

    Endpoint ``MEDIA_ASR_LLM_URL`` (default the DGX), model ``MEDIA_ASR_LLM_MODEL``
    (default ``deepseek-v4.1-flash-tp4``).
    """

    def __init__(self, key: str, *, model: str | None = None, client: httpx.Client | None = None):
        self.key = key
        self.model = model or os.environ.get("MEDIA_ASR_LLM_MODEL") or DEFAULT_MODEL
        self.url = check_endpoint(os.environ.get("MEDIA_ASR_LLM_URL") or DEFAULT_URL)
        self.client = client or httpx.Client(trust_env=False)

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.close()

    def chat(self, system: str, user: str, max_tokens: int):
        """Return the parsed JSON answer.

        Five attempts: greedy first, then sampled at temperature 0.3, because
        greedy decoding repeats a malformed answer. Raises RuntimeError after the
        last failure, or TruncatedResponse if that answer was cut off at
        ``max_tokens``.
        """
        return self._ask(system, user, max_tokens, json_mode=True)

    def chat_text(self, system: str, user: str, max_tokens: int) -> str:
        """Return the answer as plain text, without JSON mode (e.g. a translated Markdown section).

        The DGX server accepts ``response_format`` but does not enforce it, so a long Markdown answer
        often arrives unwrapped; ask for text when text is the answer. Same endpoint check, cap on
        requests in flight and attempts as ``chat``; an empty answer is retried.
        """
        return self._ask(system, user, max_tokens, json_mode=False)

    def _ask(self, system: str, user: str, max_tokens: int, *, json_mode: bool):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(3 * attempt)
            payload["temperature"] = 0 if attempt == 0 else 0.3
            try:
                with _SLOTS:
                    response = self.client.post(
                        check_endpoint(self.url),
                        headers={"Authorization": f"Bearer {self.key}"},
                        json=payload,
                        # Fail fast on an unreachable host; a slow answer still gets 300 s.
                        timeout=httpx.Timeout(300, connect=10),
                    )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                if json_mode:
                    return parse_json(choice["message"]["content"], choice.get("finish_reason"))
                return text_answer(choice["message"].get("content"), choice.get("finish_reason"))
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last = exc
        error = TruncatedResponse if isinstance(last, TruncatedResponse) else RuntimeError
        raise error(f"LLM request failed: {last}")
