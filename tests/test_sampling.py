from __future__ import annotations

from collections import Counter

import pytest

from evalforge.sampling import allocate, sample, signal_boosted, signal_weight, stratified, stratum_key, uniform
from tests.conftest import rec


def records(n: int = 100) -> list:
    out = []
    for i in range(n):
        out.append(
            rec(
                request_id=f"r{i}",
                feature="a" if i < 80 else "b",
                model="m1" if i % 2 else "m2",
                response="" if i % 25 == 0 else "ok answer here",
                feedback="down" if i % 10 == 0 else None,
            )
        )
    return out


def test_uniform_is_seeded_and_bounded() -> None:
    rs = records()
    a, b = uniform(rs, 10, seed=1), uniform(rs, 10, seed=1)
    assert [r.request_id for r in a] == [r.request_id for r in b]
    assert len(a) == 10 and len({r.request_id for r in a}) == 10
    assert uniform(rs, 10, seed=2) != a
    assert uniform(rs, 500) == rs and uniform(rs, 100) == rs


def test_stratum_key() -> None:
    assert stratum_key(rec(feature="f", model="m")) == "f|m|ok"
    assert stratum_key(rec(feature="f", model="m", response="")) == "f|m|error"


def test_allocate() -> None:
    assert allocate({"a": 80, "b": 20}, 10) == {"a": 8, "b": 2}
    assert allocate({"a": 98, "b": 1, "c": 1}, 10) == {"a": 8, "b": 1, "c": 1}
    assert allocate({"a": 5, "b": 5}, 20) == {"a": 5, "b": 5}
    assert allocate({"a": 5, "b": 5}, 10) == {"a": 5, "b": 5}
    assert allocate({"a": 50, "b": 30, "c": 20}, 2) == {"a": 1, "b": 1, "c": 0}
    assert allocate({"a": 0, "b": 0}, 3) == {"a": 0, "b": 0}
    assert allocate({"a": 4}, 0) == {"a": 0}
    assert allocate({"a": 3, "b": 0}, 2) == {"a": 2, "b": 0}
    out = allocate({"a": 10, "b": 10, "c": 10}, 7)
    assert sum(out.values()) == 7 and sorted(out.values()) == [2, 2, 3]
    assert out == {"a": 3, "b": 2, "c": 2}
    assert allocate({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}


def test_stratified_covers_every_stratum() -> None:
    rs = records()
    s = stratified(rs, 12, seed=0)
    assert len(s) == 12
    strata = {stratum_key(r) for r in rs}
    assert {stratum_key(r) for r in s} == strata
    assert stratified(rs, 12, seed=0) == s


def test_signal_weight() -> None:
    r = rec(request_id="x")
    assert signal_weight(r, set(), set()) == 1.0
    assert signal_weight(rec(request_id="x", feedback="down"), set(), set()) == 5.0
    assert signal_weight(r, {"x"}, set()) == 4.0
    assert signal_weight(rec(request_id="x", content_filtered=True), set(), set()) == 3.0
    assert signal_weight(rec(request_id="x", edited=True), set(), set()) == 3.0
    assert signal_weight(r, set(), {"x"}) == 3.0
    assert signal_weight(rec(request_id="x", feedback="up"), set(), set()) == 1.0


def test_signal_boosted_oversamples_signals() -> None:
    rs = [rec(request_id=f"r{i}", feedback="down" if i < 50 else None) for i in range(500)]
    counts: Counter[bool] = Counter()
    for seed in range(20):
        s = signal_boosted(rs, 50, seed=seed)
        assert len(s) == 50 and len({r.request_id for r in s}) == 50
        counts.update(r.feedback == "down" for r in s)
    share = counts[True] / sum(counts.values())
    assert share > 0.25  # population share is 0.10
    assert signal_boosted(rs, 600) == rs
    retried = {f"r{i}" for i in range(450, 500)}
    s = signal_boosted(rs, 50, seed=0, retried=retried, slow={"r60"})
    assert sum(r.request_id in retried for r in s) > 5


def test_sample_dispatch() -> None:
    rs = records()
    assert sample(rs, 5, "uniform", 3) == uniform(rs, 5, 3)
    assert sample(rs, 5, "stratified", 3) == stratified(rs, 5, 3)
    assert sample(rs, 5, "signal", 3) == signal_boosted(rs, 5, 3)
    with pytest.raises(ValueError, match="unknown sampling mode"):
        sample(rs, 5, "fancy")


def test_allocate_edges() -> None:
    assert allocate({"a": 13, "b": 7}, 5) == {"a": 3, "b": 2}  # largest remainder, not largest quota
    assert allocate({"a": 10, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert allocate({"a": 5, "b": 5}, 1) == {"a": 1, "b": 0}
    assert allocate({"a": 4}, -1) == {"a": 0}
    assert allocate({"a": 3, "b": 2}, 5) == {"a": 3, "b": 2}


def test_default_seeds_are_zero() -> None:
    rs = records()
    assert uniform(rs, 7) == uniform(rs, 7, seed=0)
    assert stratified(rs, 7) == stratified(rs, 7, seed=0)
    assert signal_boosted(rs, 7) == signal_boosted(rs, 7, seed=0)
    assert sample(rs, 7) == signal_boosted(rs, 7, 0)
    assert signal_boosted(rs, len(rs)) == rs


def test_signal_boosted_prefers_heavy_weights_deterministically() -> None:
    rs = [rec(request_id="light")] + [rec(request_id=f"h{i}", feedback="down", edited=True) for i in range(3)]
    picks = Counter(r.request_id for seed in range(200) for r in signal_boosted(rs, 1, seed=seed))
    assert picks["light"] < 20
