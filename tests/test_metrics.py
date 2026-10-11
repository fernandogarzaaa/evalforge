from __future__ import annotations

import pytest

from evalforge.metrics import cohen_kappa, fleiss_style_agreement, freshness, precision_recall, purity

# ---------------------------------------------------------------- metrics


def test_cohen_kappa() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert cohen_kappa(["a", "b", "a", "a"], ["a", "a", "a", "b"]) == pytest.approx(-1 / 3)
    assert cohen_kappa([], []) == 0.0
    assert cohen_kappa(["a", "a", "a"], ["a", "a", "b"]) == 0.0
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])
    assert cohen_kappa(["y", "y", "n", "n", "y"], ["y", "n", "n", "n", "y"]) == pytest.approx(0.6154, abs=1e-4)


def test_agreement_pr_purity() -> None:
    assert fleiss_style_agreement([["a", "a", "a"], ["a", "b", "b"]]) == pytest.approx((1 + 1 / 3) / 2)
    assert fleiss_style_agreement([["a"]]) == 1.0 and fleiss_style_agreement([]) == 0.0
    assert fleiss_style_agreement([["a", "b"]]) == 0.0
    pr = precision_recall({1, 2, 3, 4}, {3, 4, 5})
    assert pr == {"precision": 0.5, "recall": 0.6667, "f1": 0.5714, "tp": 2, "predicted": 4, "actual": 3}
    assert precision_recall([], [1])["precision"] == 0.0 and precision_recall([1], [])["recall"] == 0.0
    assert precision_recall([1], [2])["f1"] == 0.0
    assert purity([0, 0, 1, 1, -1], ["a", "b", "c", "c", "z"]) == pytest.approx(0.75)
    assert purity([-1], ["a"]) == 0.0


def test_freshness() -> None:
    cases = [
        {"first_seen": "2026-09-30T00:00:00Z"},
        {"first_seen": "2026-08-01T00:00:00Z"},
        {"first_seen": None},
        {"first_seen": "2026-09-01T00:00:00Z"},
    ]
    assert freshness(cases, "2026-10-01T00:00:00Z") == pytest.approx(2 / 3)
    assert freshness(cases, "2026-10-01T00:00:00Z", days=100) == 1.0
    assert freshness([], "2026-10-01T00:00:00Z") == 0.0


def test_metric_edges() -> None:
    assert cohen_kappa(["x", "y"], ["x", "x"]) == 0.0
    assert cohen_kappa(["x", "y", "y", "y"], ["x", "x", "x", "x"]) == 0.0
    assert cohen_kappa(["x", "y", "z", "z"], ["x", "x", "z", "y"]) == pytest.approx(0.2727, abs=1e-4)
    pr = precision_recall({1, 2}, {1, 3})
    assert pr["f1"] == 0.5
    assert precision_recall({1, 2, 3}, {1})["precision"] == 0.3333
    assert precision_recall({1}, {1, 2, 3})["recall"] == 0.3333
    assert precision_recall({1, 2, 3}, {1})["f1"] == 0.5
    assert precision_recall({1, 2, 4}, {1, 2, 3, 5, 6, 7})["f1"] == 0.4444
    assert purity([0, 0, 0], ["a", "a", "b"]) == pytest.approx(2 / 3)
    assert purity([0, 0, 1, 1, 1, 1], ["a", "b", "c", "c", "c", "d"]) == pytest.approx(4 / 6)
    assert freshness([{"first_seen": "2026-09-01T00:00:00Z"}], "2026-10-01T00:00:00Z") == 1.0
    assert freshness([{"first_seen": "2026-09-01T00:00:00Z"}], "2026-10-01T00:00:01Z") == 0.0
