"""Targeted tests for evalforge.store (mutation-testing survivors)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evalforge.store import Store, decode_vector, encode_vector
from tests.conftest import rec
from tests.test_pipeline_store import make_case


def test_file_store_uses_wal_and_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "e.db"
    s = Store(path)
    assert s.path == str(path) and path.exists()
    assert s.con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert s.con.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    s.close()
    mem = Store()
    assert mem.path == ":memory:"
    assert mem.con.execute("PRAGMA journal_mode").fetchone()[0] == "memory"
    assert not Path(":memory:").exists()


def test_vector_codec_is_float32_little_endian() -> None:
    enc = encode_vector([1.0, -2.5])
    assert enc == "AACAPwAAIMA="
    assert decode_vector(enc) == [1.0, -2.5]
    assert decode_vector(encode_vector([0.1]))[0] == pytest.approx(0.1, abs=1e-7)
    assert decode_vector(encode_vector([0.1]))[0] != 0.1  # stored as float32


def test_add_logs_day_counts_and_order(store: Store) -> None:
    a = rec(request_id="b", timestamp="2026-09-14T10:00:00Z")
    b = rec(request_id="a", timestamp="2026-09-14T10:00:00Z", user_prompt="other")
    c = rec(request_id="c", timestamp="2026-09-15T00:00:01Z", user_prompt="third")
    assert store.add_logs([a, b, c]) == 3
    assert store.add_logs([a]) == 0  # ignored duplicate does not count
    assert store.log_days() == ["2026-09-14", "2026-09-15"]
    assert [r.request_id for r in store.logs_for_day("2026-09-14")] == ["a", "b"]
    assert store.count_logs() == 3
    row = store.con.execute("SELECT fingerprint, feature, model FROM logs WHERE request_id='a'").fetchone()
    assert row["fingerprint"] == b.fingerprint() and row["feature"] == b.feature and row["model"] == b.model


def test_add_logs_prefers_stored_fingerprint(store: Store) -> None:
    r = rec(request_id="x")
    r.metadata["fingerprint"] = "custom"
    store.add_logs([r])
    assert store.has_log("nope", "custom") and store.has_log("x", "zzz") and not store.has_log("nope", "zzz")


def test_files_and_categories(store: Store) -> None:
    assert not store.file_seen("h")
    store.mark_file("h", "p", {"read": 1})
    store.mark_file("h", "p2", {"read": 2})
    assert store.file_seen("h")
    assert json.loads(store.con.execute("SELECT stats FROM ingested_files").fetchone()[0]) == {"read": 2}
    first = store.upsert_category(None, "pw", [1.0, 0.0], 3, "2026-09-14")
    second = store.upsert_category(None, "refund", [0.0, 1.0], 2, "2026-09-14")
    assert (first, second) == (1, 2)
    assert store.upsert_category(first, "ignored", [0.5, 0.5], 9, "2026-09-20") == first
    cats = store.categories()
    assert cats[0] == {"id": 1, "name": "pw", "centroid": [0.5, 0.5], "size": 9, "created_day": "2026-09-14"}
    assert cats[1]["name"] == "refund"


def test_cases_filters_order_and_embeddings(store: Store) -> None:
    c1 = make_case("b", "x", status="active")
    c2 = make_case("a", "y", status="pending_review")
    c3 = make_case("c", "z", status="rejected")
    c0 = {**make_case("z", "w", status="active"), "created_day": "2026-09-13"}
    for c, e in ((c1, [1.0, 0.0]), (c2, [0.0, 1.0]), (c3, [1.0, 1.0]), (c0, [0.5, 0.0])):
        store.add_case(c, e)
    assert [c["id"] for c in store.cases()] == ["z", "a", "b", "c"]
    assert [c["id"] for c in store.cases("active")] == ["z", "b"]
    assert [c["id"] for c in store.cases(["rejected", "pending_review"])] == ["a", "c"]
    assert store.cases([]) == []
    ids, embs = store.case_embeddings()
    assert ids == ["a", "b", "z"] and embs == [[0.0, 1.0], [1.0, 0.0], [0.5, 0.0]]
    ids2, _ = store.case_embeddings(["rejected"])
    assert ids2 == ["c"]
    row = store.con.execute("SELECT * FROM cases WHERE id='b'").fetchone()
    assert row["confidence"] == 0.5 and row["category"] == "c" and row["polarity"] == "positive"
    assert row["first_seen"] == c1["first_seen"] and row["created_day"] == "2026-09-14"


def test_update_case_keeps_embedding_and_requires_existing(store: Store) -> None:
    store.add_case(make_case("a", "x"), [0.25, 0.75])
    case = store.get_case("a")
    assert case is not None
    case["status"] = "active"
    store.update_case(case)
    assert store.case_embeddings(["active"]) == (["a"], [[0.25, 0.75]])
    assert store.get_case("missing") is None
    with pytest.raises(KeyError):
        store.update_case(make_case("missing", "x"))


def test_reviews_runs_and_evals(store: Store) -> None:
    store.add_review("a", "ana", "approve", None, None, "t1")
    store.add_review("a", "ben", "edit", {"quality": 2}, "n", "t2")
    store.add_review("a", "cy", "edit", {}, None, "t3")
    rv = store.reviews()
    assert [r["reviewer"] for r in rv] == ["ana", "ben", "cy"]
    assert rv[0]["labels"] is None and json.loads(rv[1]["labels"]) == {"quality": 2} and rv[2]["labels"] is None
    assert rv[1]["note"] == "n" and rv[1]["created_at"] == "t2" and rv[1]["case_id"] == "a"
    store.add_pipeline_run("d2", {"x": 1})
    store.add_pipeline_run("d1", {"x": 2})
    assert store.pipeline_runs() == [{"day": "d2", "x": 1}, {"day": "d1", "x": 2}]
    assert store.add_eval_run("n", "m", "t", {"r": 1}) == 1
    assert store.add_eval_run("n2", "m2", "t2", {"r": 2}) == 2
    ev = store.eval_runs()
    assert ev[0] == {"id": 1, "name": "n", "model": "m", "created_at": "t", "report": {"r": 1}}
    assert ev[1]["report"] == {"r": 2}


def test_tx_rolls_back_on_error(store: Store) -> None:
    with pytest.raises(RuntimeError), store.tx() as con:
        con.execute("INSERT INTO pipeline_runs (day, stats) VALUES ('d', '{}')")
        raise RuntimeError("boom")
    assert store.pipeline_runs() == []
    assert np.asarray(store.case_embeddings()[1]).size == 0


def test_close_releases_the_connection(tmp_path: Path) -> None:
    import sqlite3

    s = Store(tmp_path / "c.db")
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s.count_logs()
