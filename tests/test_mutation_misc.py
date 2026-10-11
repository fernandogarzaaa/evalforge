"""Targeted tests for small survivors in sampling, dedup and labeling."""

from __future__ import annotations

from evalforge.dedup import coverage
from evalforge.judge import Label
from evalforge.labeling import aggregate
from evalforge.sampling import allocate


def test_allocate_single_slot_goes_to_largest_stratum() -> None:
    assert allocate({"a": 5, "b": 3}, 1) == {"a": 1, "b": 0}
    assert allocate({"a": 5, "b": 3}, 0) == {"a": 0, "b": 0}
    assert allocate({"a": 5, "b": 3}, -1) == {"a": 0, "b": 0}


def test_coverage_ratio_rounds_to_four_places() -> None:
    cases = [{"category": "a"}] * 2 + [{"category": "b"}]
    cov = coverage(cases, ["a", "b", "c"], target=2)
    assert cov["covered_ratio"] == 0.3333 and cov["sparse"] == ["b", "c"]


def test_aggregate_confidence_rounds_to_four_places() -> None:
    runs = [
        Label(quality=5, behavior="answer", difficulty="simple"),
        Label(quality=5, behavior="answer", difficulty="simple"),
        Label(quality=5, behavior="refuse", difficulty="simple"),
    ]
    assert aggregate(runs).confidence == 0.8333
