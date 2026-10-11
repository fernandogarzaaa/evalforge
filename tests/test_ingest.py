from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evalforge.adapters import detect_format, otel_span_to_raw, read_any, read_jsonl, read_otel, read_sqlite
from evalforge.ingest import IngestStats, file_sha256, ingest_file, normalise
from evalforge.redact import Redactor
from evalforge.store import Store
from evalforge.synth import to_otel_span
from tests.conftest import raw


def write_jsonl(path: Path, rows: list) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n\n")
    return path


def test_read_jsonl_handles_bad_lines(tmp_path: Path) -> None:
    p = write_jsonl(tmp_path / "a.jsonl", [raw(), "{not json", "[1, 2]"])
    rows = list(read_jsonl(p))
    assert rows[0]["request_id"] == "r1"
    assert rows[1] == {"_parse_error": "line 2: Expecting property name enclosed in double quotes"}
    assert rows[2] == {"_parse_error": "line 3: not an object"}
    assert len(rows) == 3


def test_otel_round_trip_matches_jsonl_fields() -> None:
    r = raw(feedback="down", confidence=0.5, content_filtered=True)
    span = to_otel_span(r)
    out = otel_span_to_raw(span, "svc")
    assert out is not None
    assert out["request_id"] == "r1"
    assert out["timestamp"] == "2026-09-14T10:00:00.000Z"
    assert out["feature"] == "support_bot"
    assert out["model"] == "gpt-4o-mini"
    assert out["system_prompt"] == "You are a support bot."
    assert out["user_prompt"] == r["user_prompt"]
    assert out["response"] == r["response"]
    assert out["latency_ms"] == pytest.approx(800.0)
    assert out["prompt_tokens"] == 20 and out["completion_tokens"] == 18
    assert out["session_id"] == "s1"
    assert out["feedback"] == "down"
    assert out["content_filtered"] is True
    assert out["confidence"] == 0.5
    assert out["metadata"]["trace_id"] == span["traceId"]


def test_otel_defaults_and_skips() -> None:
    span = {
        "spanId": "x",
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "1500000000",
        "attributes": [
            {"key": "gen_ai.request.model", "value": {"stringValue": "m"}},
            {"key": "gen_ai.usage.prompt_tokens", "value": {"intValue": "4"}},
            {"key": "gen_ai.usage.completion_tokens", "value": {"intValue": "6"}},
            {"key": "gen_ai.prompt.0.content", "value": {"stringValue": "hi"}},
            {"key": "weird", "value": {"arrayValue": {}}},
        ],
    }
    out = otel_span_to_raw(span, "code-assistant")
    assert out is not None
    assert out["feature"] == "code-assistant"
    assert out["user_prompt"] == "hi" and out["system_prompt"] == ""
    assert out["prompt_tokens"] == 4 and out["completion_tokens"] == 6
    assert out["latency_ms"] == pytest.approx(500.0)
    assert out["feedback"] is None and out["response"] == "" and out["content_filtered"] is False
    assert otel_span_to_raw({"attributes": []}) is None
    assert otel_span_to_raw({**span, "startTimeUnixNano": "0", "endTimeUnixNano": "0"})["timestamp"] is None
    no_service = otel_span_to_raw(span)
    assert no_service is not None and no_service["feature"] == "unknown"
    resp_model = {
        **span,
        "attributes": span["attributes"]
        + [
            {"key": "gen_ai.response.model", "value": {"stringValue": "m-2024"}},
            {"key": "app.feature", "value": {"stringValue": "f"}},
        ],
    }
    out2 = otel_span_to_raw(resp_model, "svc")
    assert out2 is not None and out2["model"] == "m-2024" and out2["feature"] == "f"


def test_read_otel_file(tmp_path: Path) -> None:
    doc = {
        "resourceSpans": [
            {
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "svc"}}]},
                "scopeSpans": [{"spans": [to_otel_span(raw()), {"spanId": "noise", "attributes": []}]}],
            }
        ]
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    rows = list(read_otel(p))
    assert len(rows) == 1 and rows[0]["request_id"] == "r1"
    assert detect_format(p) == "otel"


def test_read_sqlite_and_mapping(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE llm_logs (id TEXT, ts TEXT, feature TEXT, model TEXT, prompt TEXT, edited INTEGER)")
    con.execute("INSERT INTO llm_logs VALUES ('a', '2026-09-14T00:00:00Z', 'f', 'm', 'p', 1)")
    con.commit()
    con.close()
    rows = list(read_sqlite(db, column_map={"id": "request_id", "ts": "timestamp", "prompt": "user_prompt"}))
    assert rows == [
        {
            "request_id": "a",
            "timestamp": "2026-09-14T00:00:00Z",
            "feature": "f",
            "model": "m",
            "user_prompt": "p",
            "edited": True,
        }
    ]
    with pytest.raises(ValueError):
        list(read_sqlite(db, table="x; DROP TABLE y"))
    assert detect_format(db) == "sqlite"
    assert list(read_any(db))[0]["id"] == "a"


def test_detect_and_read_any(tmp_path: Path) -> None:
    j = write_jsonl(tmp_path / "a.jsonl", [raw()])
    assert detect_format(j) == "jsonl"
    other = tmp_path / "b.log"
    other.write_text(json.dumps(raw()) + "\n")
    assert detect_format(other) == "jsonl"
    assert list(read_any(other))[0]["request_id"] == "r1"
    with pytest.raises(ValueError):
        read_any(other, "csv")


def test_normalise_validates_dedups_and_redacts(store: Store) -> None:
    rows = [
        raw(user_prompt="email me at a@b.co"),
        raw(),
        raw(request_id="r2"),
        raw(request_id="r3", latency_ms=-1),
        {"_parse_error": "x"},
        raw(request_id="r4", user_prompt="other"),
    ]
    records, stats = normalise(rows, store, Redactor())
    # r2 shares content with the second row, but that row was dropped by id, so r2 is new.
    assert [r.request_id for r in records] == ["r1", "r2", "r4"]
    assert records[0].user_prompt == "email me at [EMAIL]"
    assert stats.read == 6 and stats.invalid == 2 and stats.duplicates == 1
    assert stats.errors["unparseable"] == 1
    assert stats.redactions == {"email": 1}
    assert "fingerprint" in records[0].metadata


def test_duplicate_detection_uses_pre_redaction_content(store: Store) -> None:
    a = raw(request_id="a", user_prompt="me: x@y.com")
    b = raw(request_id="b", user_prompt="me: z@w.com")  # different people, same redacted text
    records, stats = normalise([a, b], store)
    assert len(records) == 2 and stats.duplicates == 0
    store.add_logs(records)
    again, stats2 = normalise([raw(request_id="c", user_prompt="me: x@y.com")], store)
    assert again == [] and stats2.duplicates == 1


def test_ingest_file_is_idempotent(tmp_path: Path, store: Store) -> None:
    p = write_jsonl(tmp_path / "logs.jsonl", [raw(), raw(request_id="r2", user_prompt="b"), "garbage"])
    s1 = ingest_file(p, store)
    assert (s1.read, s1.stored, s1.invalid) == (3, 2, 1)
    assert store.count_logs() == 2
    assert ingest_file(p, store).to_dict()["read"] == 0
    forced = ingest_file(p, store, force=True)
    assert forced.duplicates == 2 and forced.stored == 0
    assert len(file_sha256(p)) == 64


def test_stats_merge() -> None:
    a, b = (
        IngestStats(read=1, invalid=1, duplicates=1, stored=1),
        IngestStats(read=2, invalid=3, duplicates=4, stored=5),
    )
    a.errors["x"] = 1
    b.errors["x"] = 2
    b.redactions["email"] = 3
    a.merge(b)
    assert a.to_dict() == {
        "read": 3,
        "invalid": 4,
        "duplicates": 5,
        "stored": 6,
        "errors": {"x": 3},
        "redactions": {"email": 3},
    }
