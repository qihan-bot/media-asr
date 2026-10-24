"""DeepSeek chat client shared by the podcast library steps."""

from __future__ import annotations

import json
import os
import re
import time

import httpx

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"


class TruncatedResponse(RuntimeError):
    """The answer hit ``max_tokens`` before its JSON was complete."""


def parse_json(content: str, finish_reason: str | None = None):
    """Parse a JSON answer, tolerating a Markdown code fence around it."""
    text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if finish_reason == "length":
            raise TruncatedResponse(f"answer cut off at max_tokens: {exc}") from exc
        raise


class LLM:
    """DeepSeek chat in JSON mode, reached directly (``trust_env=False``: no proxy).

    The model comes from ``MEDIA_ASR_LLM_MODEL`` (default ``deepseek-flash``).
    """

    def __init__(self, key: str, *, model: str | None = None, client: httpx.Client | None = None):
        self.key = key
        self.model = model or os.environ.get("MEDIA_ASR_LLM_MODEL") or DEFAULT_MODEL
        self.client = client or httpx.Client(trust_env=False)

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.close()

    def chat(self, system: str, user: str, max_tokens: int):
        """Return the parsed JSON answer.

        Three attempts: greedy first, then sampled at temperature 0.3, because
        greedy decoding repeats a malformed answer. Raises RuntimeError after the
        last failure, or TruncatedResponse if that answer was cut off at
        ``max_tokens``.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        last: Exception | None = None
        for attempt in range(3):
            if attempt:
                time.sleep(3 * attempt)
            payload["temperature"] = 0 if attempt == 0 else 0.3
            try:
                response = self.client.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {self.key}"},
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                return parse_json(choice["message"]["content"], choice.get("finish_reason"))
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last = exc
        error = TruncatedResponse if isinstance(last, TruncatedResponse) else RuntimeError
        raise error(f"DeepSeek request failed: {last}")
