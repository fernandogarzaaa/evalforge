"""Targeted tests for evalforge.ingest (mutation-testing survivors)."""

from __future__ import annotations

from evalforge.ingest import IngestStats, normalise
from evalforge.redact import Redactor
from evalforge.store import Store
from tests.conftest import raw


def test_ingest_stats_start_at_zero() -> None:
    assert IngestStats().to_dict() == {
        "read": 0,
        "invalid": 0,
        "duplicates": 0,
        "stored": 0,
        "errors": {},
        "redactions": {},
    }


def test_validation_errors_are_counted_by_message(store: Store) -> None:
    no_model = raw(request_id="a")
    del no_model["model"]
    bad_latency = raw(request_id="b", latency_ms="x")
    also_no_model = raw(request_id="c", user_prompt="other")
    del also_no_model["model"]
    out, st = normalise([no_model, bad_latency, also_no_model, {"_parse_error": "line 4: x"}], store)
    assert out == [] and st.invalid == 4 and st.read == 4
    assert dict(st.errors) == {
        "model is required": 2,
        "latency_ms must be a non-negative number": 1,
        "unparseable": 1,
    }


def test_same_content_new_request_id_is_a_duplicate(store: Store) -> None:
    out, st = normalise([raw(request_id="a"), raw(request_id="b"), raw(request_id="a", user_prompt="different")], store)
    assert [r.request_id for r in out] == ["a"] and st.duplicates == 2


def test_redaction_counts_are_per_call_with_a_shared_redactor(store: Store) -> None:
    red = Redactor()
    red.redact("a@b.com c@d.com e@f.com")  # pre-existing counts on the shared redactor
    _, st = normalise([raw(request_id="x", user_prompt="mail me at z@y.com")], store, red)
    assert dict(st.redactions) == {"email": 1}
    assert red.counts["email"] == 4
