"""Small, dependency-free metrics used by the pipeline, the curation queue and the benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from datetime import timedelta
from typing import Any

from evalforge.schema import parse_timestamp


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters. 1.0 when both raters are constant and identical."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if expected == 1.0:  # both raters used one identical label throughout, so observed == 1 too
        return 1.0
    return (observed - expected) / (1 - expected)


def fleiss_style_agreement(runs: Sequence[Sequence[Hashable]]) -> float:
    """Mean pairwise exact agreement across k raters (per item, averaged over items)."""
    if not runs:
        return 0.0
    total = 0.0
    for labels in runs:
        k = len(labels)
        if k < 2:
            total += 1.0
            continue
        pairs = k * (k - 1) / 2
        same = sum(1 for i in range(k) for j in range(i + 1, k) if labels[i] == labels[j])
        total += same / pairs
    return total / len(runs)


def precision_recall(predicted: Iterable[Hashable], actual: Iterable[Hashable]) -> dict[str, float]:
    p, t = set(predicted), set(actual)
    tp = len(p & t)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "predicted": len(p),
        "actual": len(t),
    }


def purity(clusters: Sequence[Hashable], truth: Sequence[Hashable]) -> float:
    """Share of points whose cluster's majority true class matches their own (noise excluded)."""
    pairs = Counter((c, t) for c, t in zip(clusters, truth) if c != -1)
    majority: dict[Hashable, int] = {}
    for (c, _), count in pairs.items():
        majority[c] = max(majority.get(c, 0), count)
    n = sum(pairs.values())
    return sum(majority.values()) / n if n else 0.0


def freshness(cases: Iterable[dict[str, Any]], now: str, days: int = 30) -> float:
    """Share of cases whose source interaction is from the last ``days`` days before ``now``."""
    items = [c for c in cases if c.get("first_seen")]
    if not items:
        return 0.0
    cutoff = parse_timestamp(now) - timedelta(days=days)
    return sum(parse_timestamp(c["first_seen"]) >= cutoff for c in items) / len(items)
