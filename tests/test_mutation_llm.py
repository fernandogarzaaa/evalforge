"""Targeted tests for evalforge.llm (mutation-testing survivors): retry policy, backoff, defaults."""

from __future__ import annotations

import httpx
import pytest

from evalforge.llm import LLMError, OpenAICompatClient, extract_json


def _client(handler, **kw) -> OpenAICompatClient:  # type: ignore[no-untyped-def]
    return OpenAICompatClient("http://x", "", "m", client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


@pytest.fixture()
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    out: list[float] = []
    monkeypatch.setattr("evalforge.llm.time.sleep", out.append)
    return out


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(code: int, sleeps: list[float]) -> None:
    codes = [code, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        c = codes.pop(0)
        return httpx.Response(c, json={"choices": [{"message": {"content": "ok"}}]})

    assert _client(handler).complete([]) == "ok"
    assert codes == [] and sleeps == [0.5]


def test_backoff_is_exponential_capped_and_skips_last_sleep(sleeps: list[float]) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(LLMError, match=r"failed after 7 attempts: HTTP 503"):
        _client(handler, max_retries=7).complete([])
    assert len(calls) == 7
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]


def test_default_retry_count_is_three(sleeps: list[float]) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    c = _client(handler)
    assert c.max_retries == 3
    with pytest.raises(LLMError, match="after 3 attempts"):
        c.complete([])
    assert len(calls) == 3 and sleeps == [0.5, 1.0]


def test_malformed_body_is_retried_then_reported(sleeps: list[float]) -> None:
    bodies = [{"choices": []}, {"nope": 1}, {"choices": [{"message": {"content": 42}}]}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bodies.pop(0))

    assert _client(handler).complete([]) == "42"
    assert len(sleeps) == 2

    with pytest.raises(LLMError, match="after 1 attempts"):
        _client(lambda r: httpx.Response(200, json={}), max_retries=1).complete([])
    assert len(sleeps) == 2  # a single attempt never sleeps


def test_client_errors_are_raised_as_llm_error(sleeps: list[float]) -> None:
    with pytest.raises(LLMError, match="404"):
        _client(lambda r: httpx.Response(404), max_retries=2).complete([])


def test_default_client_settings() -> None:
    c = OpenAICompatClient("http://h/v1//", "sk", "mod")
    assert c.name == "mod" and c.model == "mod" and c.base_url == "http://h/v1"
    assert c.client.timeout.read == 60.0 and c.client.headers["authorization"] == "Bearer sk"
    c2 = OpenAICompatClient("http://h", "", "mod", timeout=5)
    assert c2.client.timeout.read == 5 and "authorization" not in c2.client.headers


def test_temperature_default_and_zero_seed() -> None:
    import json

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    _client(handler).complete([{"role": "user", "content": "q"}], seed=0)
    assert seen[0]["temperature"] == 0.0 and seen[0]["seed"] == 0


def test_extract_json_variants() -> None:
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('prose ```json {"a": {"b": 2}} ``` more {"z": 0}') == {"a": {"b": 2}}
    assert extract_json('x {"a": 1} {"b": 2}') == {"a": 1}
    assert extract_json('{"a": "{not a brace}"}') == {"a": "{not a brace}"}
    assert extract_json('{"a": "\\\\", "b": "}"}') == {"a": "\\", "b": "}"}
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("[1, 2]")
    with pytest.raises(ValueError, match="unterminated"):
        extract_json('{"a": "}')


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_status_is_reported_as_http_code(code: int, sleeps: list[float]) -> None:
    with pytest.raises(LLMError, match=f"after 1 attempts: HTTP {code}$"):
        _client(lambda r: httpx.Response(code), max_retries=1).complete([])


def test_extract_json_empty_key_and_trailing_text() -> None:
    assert extract_json('{"": 1}') == {"": 1}
    assert extract_json('{"a": 1}x') == {"a": 1}
    assert extract_json('{"a": 1}}') == {"a": 1}
