"""Minimal OpenAI-compatible chat client.

Works with any server that speaks ``POST {base}/chat/completions``: OpenAI, Groq,
OpenRouter, Together, vLLM, llama.cpp ``llama-server``, Ollama (``/v1``) and so on.
Configured only through the environment: OPENAI_BASE_URL, OPENAI_API_KEY, MODEL.
Nothing in tests or CI talks to a network; they use the mock judge / mock models.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol

import httpx


class ChatClient(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0, seed: int | None = None) -> str: ...


class LLMError(RuntimeError):
    pass


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)

    @classmethod
    def from_env(cls, model: str | None = None) -> OpenAICompatClient:
        base = os.environ.get("OPENAI_BASE_URL")
        if not base:
            raise LLMError("OPENAI_BASE_URL is not set (use --llm mock for the offline default)")
        name = model or os.environ.get("MODEL")
        if not name:
            raise LLMError("MODEL is not set")
        return cls(base, os.environ.get("OPENAI_API_KEY", ""), name)

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0, seed: int | None = None) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if seed is not None:
            body["seed"] = seed
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.post(f"{self.base_url}/chat/completions", json=body)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return str(resp.json()["choices"][0]["message"]["content"] or "")
            except (httpx.HTTPError, LLMError, KeyError, IndexError) as exc:
                last = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(8.0, 0.5 * 2**attempt))
        raise LLMError(f"chat completion failed after {self.max_retries} attempts: {last}")


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates ```json fences and prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(candidate[start : i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("reply JSON is not an object")
                return obj
    raise ValueError("unterminated JSON object in reply")
