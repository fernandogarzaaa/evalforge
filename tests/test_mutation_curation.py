"""Targeted tests for evalforge.curation (mutation-testing survivors)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from evalforge.curation import apply_edits, review, review_queue, reviewer_stats, similar_cases
from evalforge.store import Store
from tests.test_pipeline_store import make_case


def test_review_defaults_timestamp_to_now(store: Store) -> None:
    store.add_case(make_case("a", "reset password"), [1.0, 0.0])
    review(store, "a", "ana", "approve")
    at = store.reviews()[0]["created_at"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", at)


def test_review_queue_treats_missing_confidence_as_zero(store: Store) -> None:
    store.add_case(make_case("a", "x", conf=0.0001), [1.0, 0.0])
    none_case = make_case("b", "y")
    none_case["confidence"] = None
    store.add_case(none_case, [0.0, 1.0])
    assert [c["id"] for c in review_queue(store)] == ["b", "a"]


@pytest.mark.parametrize("q", [1, 5, "3"])
def test_apply_edits_quality_bounds_inclusive(q: object) -> None:
    out = apply_edits(make_case("a", "x"), {"quality": q})
    assert out["observed"]["quality"] == int(q)  # type: ignore[call-overload]


@pytest.mark.parametrize("q", [0, 6])
def test_apply_edits_quality_out_of_range(q: int) -> None:
    with pytest.raises(ValueError, match="quality must be 1-5"):
        apply_edits(make_case("a", "x"), {"quality": q})


def test_edit_without_edits_is_an_approval(store: Store) -> None:
    store.add_case(make_case("a", "x"), [1.0, 0.0])
    out = review(store, "a", "ana", "edit", at="t")
    assert out["status"] == "active" and out["label_source"] == "human"
    assert store.reviews()[0]["labels"] is None


def test_similar_cases_default_k_and_rounding(store: Store) -> None:
    store.add_case(make_case("t", "reset password"), [1.0, 0.0])
    for i in range(5):
        store.add_case(make_case(f"c{i}", f"reset password {'x' * (i + 2)}", status="active"), [1.0, 0.0])

    class Fixed:
        name = "f"

        def embed(self, texts):  # type: ignore[no-untyped-def]
            return np.asarray([[1.0, 0.0] if t == "reset password" else [1.0, float(len(t) - 15) / 7] for t in texts])

    out = similar_cases(store, "t", embedder=Fixed())
    assert len(out) == 3 and [c["id"] for c in out] == ["c0", "c1", "c2"]
    sim = 1 / np.sqrt(1 + (2 / 7) ** 2)
    assert out[0]["similarity"] == round(float(sim), 4) and len(str(out[0]["similarity"]).split(".")[1]) <= 4


def test_reviewer_stats_kappa_and_rounding() -> None:
    reviews = []
    a_votes = ["reject", "approve", "approve", "edit", "approve", "approve"]
    b_votes = ["reject", "approve", "reject", "approve", "approve", "approve"]
    for i, (a, b) in enumerate(zip(a_votes, b_votes)):
        reviews.append({"reviewer": "ana", "case_id": f"c{i}", "decision": a, "created_at": "2026-09-14T01:00:00Z"})
        reviews.append({"reviewer": "ben", "case_id": f"c{i}", "decision": b, "created_at": None})
    s = reviewer_stats(reviews)
    assert s["reviews"] == 12 and s["per_reviewer"] == {"ana": 6, "ben": 6}
    assert s["per_reviewer_per_day"] == {"ana": {"2026-09-14": 6}, "ben": {"": 6}}
    # ana: R A A A A A  ben: R A R A A A -> po = 5/6, pe = (1*2 + 5*4)/36 = 22/36, kappa = 8/14 = 0.571428...
    assert s["kappa"] == {"ana~ben": 0.5714}
    same = reviewer_stats([{**r, "decision": "approve" if r["decision"] == "edit" else r["decision"]} for r in reviews])
    assert same["kappa"] == {"ana~ben": 0.5714}  # edit counts as accept
