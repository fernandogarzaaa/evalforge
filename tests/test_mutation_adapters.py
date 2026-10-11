"""Targeted tests for evalforge.adapters (mutation-testing survivors)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evalforge import adapters
from evalforge.adapters import detect_format, otel_span_to_raw, read_sqlite


def _attr(key: str, **value: object) -> dict:
    return {"key": key, "value": value}


def test_otel_first_prompt_index_and_unset_fields() -> None:
    span = {
        "spanId": "s",
        "attributes": [
            _attr("gen_ai.request.model", stringValue="m"),
            _attr("gen_ai.prompt.0.content", stringValue="hello"),
        ],
    }
    raw = otel_span_to_raw(span)
    assert raw is not None
    assert raw["user_prompt"] == "hello" and raw["system_prompt"] == ""
    assert raw["timestamp"] is None  # startTimeUnixNano missing = unset
    assert raw["latency_ms"] == 0.0
    assert raw["prompt_tokens"] == 0 and raw["completion_tokens"] == 0
    assert raw["feature"] == "unknown" and raw["feedback"] is None and raw["content_filtered"] is False


def test_otel_latency_conversion_and_clamp() -> None:
    base = {"attributes": [_attr("gen_ai.response.model", stringValue="m")]}
    one_s = otel_span_to_raw({**base, "startTimeUnixNano": "1000000000", "endTimeUnixNano": "2000000000"})
    assert one_s is not None and one_s["latency_ms"] == 1000.0
    assert one_s["timestamp"] == "1970-01-01T00:00:01.000Z"
    tiny = otel_span_to_raw({**base, "startTimeUnixNano": "1000000000", "endTimeUnixNano": "1000500000"})
    assert tiny is not None and tiny["latency_ms"] == 0.5
    backwards = otel_span_to_raw({**base, "startTimeUnixNano": "2000000000", "endTimeUnixNano": "1000000000"})
    assert backwards is not None and backwards["latency_ms"] == 0.0
    no_end = otel_span_to_raw({**base, "startTimeUnixNano": "5"})
    assert no_end is not None and no_end["latency_ms"] == 0.0 and no_end["timestamp"] is not None


def test_read_sqlite_closes_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "x.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE llm_logs (id TEXT)")
    con.execute("INSERT INTO llm_logs VALUES ('a')")
    con.commit()
    con.close()
    closed: list[bool] = []
    real = sqlite3.connect

    class Spy:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self.inner = inner

        def __setattr__(self, name: str, value: object) -> None:
            if name == "inner":
                object.__setattr__(self, name, value)
            else:
                setattr(self.inner, name, value)

        def execute(self, *a: object):  # type: ignore[no-untyped-def]
            return self.inner.execute(*a)  # type: ignore[arg-type]

        def close(self) -> None:
            closed.append(True)
            self.inner.close()

    monkeypatch.setattr(adapters.sqlite3, "connect", lambda p: Spy(real(p)))
    assert [r["id"] for r in read_sqlite(db)] == ["a"]
    assert closed == [True]


def test_detect_format_reads_exactly_4096_chars(tmp_path: Path) -> None:
    marker = '"resourceSpans"'
    inside = tmp_path / "inside.json"
    inside.write_text(" " * (4096 - len(marker)) + marker + ": []", encoding="utf-8")
    assert detect_format(inside) == "otel"
    outside = tmp_path / "outside.json"
    outside.write_text(" " * (4097 - len(marker)) + marker + ": []", encoding="utf-8")
    assert detect_format(outside) == "jsonl"


def test_otel_start_of_one_nanosecond_is_a_real_timestamp() -> None:
    raw = otel_span_to_raw({"startTimeUnixNano": "1", "attributes": [_attr("gen_ai.request.model", stringValue="m")]})
    assert raw is not None and raw["timestamp"] == "1970-01-01T00:00:00.000Z"
