from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalforge.schema import LogRecord, parse_timestamp, validate
from tests.conftest import raw, rec


def test_valid_record_round_trips() -> None:
    record, errors = validate(raw())
    assert errors == []
    assert record is not None
    assert record.request_id == "r1"
    assert record.timestamp == "2026-09-14T10:00:00Z"
    assert record.latency_ms == 800.0
    assert record.prompt_tokens == 20 and record.completion_tokens == 18
    assert record.session_id == "s1"
    assert record.feedback is None and record.confidence is None
    assert record.edited is False and record.content_filtered is False
    assert LogRecord(**record.to_dict()) == record


def test_strips_ids_and_normalises_timestamp_offset() -> None:
    record, _ = validate(raw(request_id="  r9 ", feature=" sql ", model=" m ", timestamp="2026-09-14T12:00:00+02:00"))
    assert record is not None
    assert (record.request_id, record.feature, record.model) == ("r9", "sql", "m")
    assert record.timestamp == "2026-09-14T10:00:00Z"


def test_naive_timestamp_is_utc() -> None:
    assert parse_timestamp("2026-09-14T10:00:00") == datetime(2026, 9, 14, 10, tzinfo=timezone.utc)
    assert parse_timestamp(" 2026-09-14T10:00:00Z ") == datetime(2026, 9, 14, 10, tzinfo=timezone.utc)


@pytest.mark.parametrize("field", ["request_id", "timestamp", "feature", "model", "user_prompt"])
def test_required_fields(field: str) -> None:
    for bad in (None, "", "   ", 5):
        record, errors = validate(raw(**{field: bad}))
        assert record is None
        assert f"{field} is required" in errors


def test_missing_required_field_entirely() -> None:
    r = raw()
    del r["user_prompt"]
    record, errors = validate(r)
    assert record is None and errors == ["user_prompt is required"]


@pytest.mark.parametrize(
    "over,message",
    [
        ({"timestamp": "yesterday"}, "timestamp is not ISO-8601"),
        ({"latency_ms": -1}, "latency_ms must be a non-negative number"),
        ({"latency_ms": "fast"}, "latency_ms must be a non-negative number"),
        ({"latency_ms": True}, "latency_ms must be a non-negative number"),
        ({"prompt_tokens": -3}, "prompt_tokens must be a non-negative integer"),
        ({"prompt_tokens": 2.5}, "prompt_tokens must be a non-negative integer"),
        ({"completion_tokens": True}, "completion_tokens must be a non-negative integer"),
        ({"completion_tokens": "9"}, "completion_tokens must be a non-negative integer"),
        ({"feedback": "meh"}, "feedback must be 'up', 'down' or null"),
        ({"confidence": 1.5}, "confidence must be within [0, 1]"),
        ({"confidence": -0.1}, "confidence must be within [0, 1]"),
        ({"confidence": True}, "confidence must be within [0, 1]"),
        ({"confidence": "high"}, "confidence must be within [0, 1]"),
        ({"session_id": 12}, "session_id must be a string"),
        ({"response": 5}, "response must be a string"),
        ({"system_prompt": ["x"]}, "system_prompt must be a string"),
    ],
)
def test_field_errors(over: dict, message: str) -> None:
    record, errors = validate(raw(**over))
    assert record is None
    assert errors == [message]


def test_boundaries_are_accepted() -> None:
    record, errors = validate(raw(latency_ms=0, prompt_tokens=0, completion_tokens=0, confidence=0.0, feedback="up"))
    assert errors == [] and record is not None
    record, errors = validate(raw(confidence=1.0, feedback="down", prompt_tokens=7.0))
    assert errors == [] and record is not None
    assert record.prompt_tokens == 7 and isinstance(record.prompt_tokens, int)
    assert record.confidence == 1.0 and record.feedback == "down"


def test_non_dict_and_none_values() -> None:
    assert validate([1, 2]) == (None, ["record is not an object"])  # type: ignore[arg-type]
    record, errors = validate(raw(response=None, system_prompt=None))
    assert errors == [] and record is not None
    assert record.response == "" and record.system_prompt == ""


def test_unknown_keys_land_in_metadata() -> None:
    record, _ = validate(raw(trace="abc", metadata={"team": "x"}, edited=1, content_filtered=1))
    assert record is not None
    assert record.metadata == {"team": "x", "trace": "abc"}
    assert record.edited is True and record.content_filtered is True


def test_is_error_and_fingerprint() -> None:
    assert rec().is_error is False
    assert rec(response="  ").is_error is True
    assert rec(content_filtered=True).is_error is True
    a, b = rec(), rec(request_id="other")
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 64
    for change in (
        {"response": "x"},
        {"user_prompt": "y"},
        {"model": "m2"},
        {"feature": "f2"},
        {"system_prompt": "s2"},
        {"timestamp": "2026-09-14T10:00:01Z"},
    ):
        assert rec(**change).fingerprint() != a.fingerprint()


def test_defaults_for_optional_fields() -> None:
    minimal = {k: raw()[k] for k in ("request_id", "timestamp", "feature", "model", "user_prompt")}
    record, errors = validate(minimal)
    assert errors == [] and record is not None
    assert record.latency_ms == 0.0 and record.prompt_tokens == 0 and record.completion_tokens == 0
    assert record.edited is False and record.content_filtered is False and record.response == ""
    assert record.metadata == {}
    bare = LogRecord("i", "t", "f", "m", "p")
    assert (bare.latency_ms, bare.prompt_tokens, bare.completion_tokens) == (0.0, 0, 0)
    assert bare.edited is False and bare.content_filtered is False and bare.session_id is None
    assert bare.response == "" and bare.system_prompt == "" and bare.metadata == {}
