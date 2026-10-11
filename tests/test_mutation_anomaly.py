"""Targeted tests for evalforge.anomaly (mutation-testing survivors): boundaries and constants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalforge.anomaly import (
    FLAG_WEIGHTS,
    compute_flags,
    detect_retries,
    flag_score,
    latency_outliers,
    length_outliers,
    outliers_by_group,
    robust_z,
    rowwise_cosine,
)
from tests.conftest import rec


def test_robust_z_constant_and_mad() -> None:
    # median 3, MAD 1 -> z = 0.6745 * (v - 3)
    assert robust_z([1, 2, 3, 4, 5]) == pytest.approx([-1.349, -0.6745, 0.0, 0.6745, 1.349])
    assert robust_z([10.0]) == [0.0]
    # MAD zero but values differ (majority equal): still all zero
    assert robust_z([1, 1, 1, 9]) == [0.0, 0.0, 0.0, 0.0]
    out = robust_z([0, 0, 2, 2])  # median 1, MAD 1
    assert out == pytest.approx([-0.6745, -0.6745, 0.6745, 0.6745]) and all(isinstance(v, float) for v in out)


def test_outlier_threshold_is_strict_and_two_sided() -> None:
    rs = [rec(request_id=f"r{i}", latency_ms=float(v)) for i, v in enumerate([1, 2, 3, 4, 5, 50, -44])]
    zs = robust_z([r.latency_ms for r in rs])
    hi = zs[5]
    assert outliers_by_group(rs, lambda r: r.latency_ms, threshold=hi) == set()
    assert outliers_by_group(rs, lambda r: r.latency_ms, threshold=hi - 1e-9) == {"r5", "r6"}
    # default threshold 3.5
    assert outliers_by_group(rs, lambda r: r.latency_ms) == {"r5", "r6"}

    def grp(x: float) -> list:
        return [rec(request_id=f"n{i}", latency_ms=float(v)) for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, x])]

    # z(16) = 4.22, z(13) = 3.20: the default 3.5 flags the first and not the second
    assert outliers_by_group(grp(16), lambda r: r.latency_ms) == {"n7"}
    assert outliers_by_group(grp(13), lambda r: r.latency_ms) == set()


def test_outliers_group_by_custom_key() -> None:
    rs = [rec(request_id=f"r{i}", model="m1" if i < 6 else "m2", latency_ms=float(10 + i % 3)) for i in range(12)]
    rs.append(rec(request_id="x", model="m2", latency_ms=1000.0))
    assert outliers_by_group(rs, lambda r: r.latency_ms, group=lambda r: r.model) == {"x"}


def test_latency_outliers_uses_log_scale_and_default() -> None:
    base = [rec(request_id=f"r{i}", latency_ms=float(v)) for i, v in enumerate([100, 110, 120, 130, 140])]
    # 300ms is a big linear outlier (z=6.1) but on a log scale z = 0.6745*(log1p(300)-log1p(120))/(log1p(130)-log1p(120)) ~ 7.6;
    # 170ms: linear z = 0.6745*50/10 = 3.37 (not flagged); log z ~ 0.6745*0.348/0.080 = 2.9 (not flagged)
    assert latency_outliers(base + [rec(request_id="s", latency_ms=170.0)]) == set()
    assert latency_outliers(base + [rec(request_id="s", latency_ms=300.0)]) == {"s"}
    zs = robust_z([math.log1p(v) for v in [100, 110, 120, 130, 140, 300]])
    assert latency_outliers(base + [rec(request_id="s", latency_ms=300.0)], threshold=zs[-1]) == set()


def test_length_outliers_short_rule_boundary() -> None:
    rs = [rec(request_id=f"r{i}", response="alpha beta gamma delta") for i in range(6)]
    rs.append(rec(request_id="two", response="too short"))
    rs.append(rec(request_id="three", response="just three words"))
    found = length_outliers(rs, threshold=1e9)
    assert found == {"two"}
    # the log-length outlier test applies to non-error records only, default threshold 3.5
    rs2 = [rec(request_id=f"r{i}", response="w " * (10 + i % 3)) for i in range(9)]
    rs2.append(rec(request_id="huge", response="w " * 60))
    rs2.append(rec(request_id="err", response="   ", content_filtered=False))
    assert length_outliers(rs2) == {"huge"}


def test_rowwise_cosine_clamps_tiny_norms() -> None:
    a = np.array([[1e-13, 0.0]])
    b = np.array([[1.0, 0.0]])
    assert rowwise_cosine(a, b)[0] == pytest.approx(0.1, rel=1e-3)
    assert rowwise_cosine(b, a)[0] == pytest.approx(0.1, rel=1e-3)
    assert rowwise_cosine(np.array([[3.0, 4.0]]), np.array([[3.0, 4.0]]))[0] == pytest.approx(1.0)


def _pair(gap_s: int, p1: str = "reset password", p2: str = "reset password") -> list:
    return [
        rec(request_id="a", session_id="s", timestamp="2026-09-14T10:00:00Z", user_prompt=p1),
        rec(
            request_id="b",
            session_id="s",
            timestamp=f"2026-09-14T10:{gap_s // 60:02d}:{gap_s % 60:02d}Z",
            user_prompt=p2,
        ),
    ]


def test_retry_window_is_inclusive_with_120s_default() -> None:
    assert detect_retries(_pair(120)) == {"a"}
    assert detect_retries(_pair(121)) == set()
    assert detect_retries(_pair(0)) == {"a"}
    assert detect_retries(_pair(30), window_s=30) == {"a"}
    assert detect_retries(_pair(31), window_s=30) == set()


def test_retry_similarity_is_inclusive_with_default() -> None:
    from evalforge.embed import HashingEmbedder

    e = HashingEmbedder()
    p1, p2 = "reset my password now", "password link broken today"
    sim = float(rowwise_cosine(e.embed([p1]), e.embed([p2]))[0])
    assert 0 < sim < 1
    assert detect_retries(_pair(10, p1, p2), min_similarity=sim) == {"a"}
    assert detect_retries(_pair(10, p1, p2), min_similarity=sim + 1e-4) == set()
    # default 0.1: unrelated prompts are not retries, related ones are
    assert detect_retries(_pair(10, "cancel my plan", "what is a regex")) == set()


def test_retries_sorted_by_time_and_custom_embedder() -> None:
    rs = list(reversed(_pair(60)))

    class Ones:
        name = "ones"
        calls = 0

        def embed(self, texts):  # type: ignore[no-untyped-def]
            Ones.calls += 1
            return np.ones((len(texts), 3))

    assert detect_retries(rs, embedder=Ones()) == {"a"} and Ones.calls == 2
    assert detect_retries([rec(request_id="z", session_id="")]) == set()


def test_compute_flags_low_confidence_boundary_and_retry_detection() -> None:
    rs = [rec(request_id="lo", confidence=0.6999), rec(request_id="eq", confidence=0.7)]
    f = compute_flags(rs, retried_ids=set())
    assert f == {"lo": ["low_confidence"], "eq": []}
    # retried_ids=None runs detection; an explicit empty set disables it
    pair = _pair(5)
    assert compute_flags(pair)["a"] == ["retried"]
    assert compute_flags(pair, retried_ids=set())["a"] == []
    assert compute_flags(pair, novel_ids=None, retried_ids=set())["b"] == []


def test_compute_flags_length_and_latency_order() -> None:
    rs = [rec(request_id=f"r{i}", latency_ms=100.0 + i, response="alpha beta gamma delta") for i in range(8)]
    rs.append(rec(request_id="x", latency_ms=90000.0, response="no"))
    f = compute_flags(rs, retried_ids=set())
    assert f["x"] == ["length_outlier", "latency_outlier"] and f["r0"] == []


def test_flag_weights_exact() -> None:
    assert FLAG_WEIGHTS == {
        "negative_feedback": 3.0,
        "retried": 3.0,
        "filtered": 2.5,
        "novel": 2.5,
        "edited": 2.0,
        "low_confidence": 2.0,
        "length_outlier": 1.5,
        "latency_outlier": 1.0,
    }
    assert flag_score(["unknown_a", "unknown_b"]) == 2.0
    assert flag_score(f for f in ["retried"]) == 3.0
