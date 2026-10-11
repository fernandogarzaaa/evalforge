"""Near-duplicate filtering against the existing eval set and coverage tracking (Phase 3.4)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evalforge.embed import cosine_matrix

DEFAULT_THRESHOLD = 0.92


@dataclass
class DedupResult:
    keep: list[bool]
    duplicate_of: list[str | None]  # id of the existing case (or earlier candidate) it duplicates
    max_similarity: list[float]


def dedup(
    candidates: np.ndarray,
    candidate_ids: Sequence[str],
    existing: np.ndarray,
    existing_ids: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> DedupResult:
    """Drop candidates whose cosine similarity to an existing case, or to an earlier kept
    candidate in the same batch, is strictly greater than ``threshold``."""
    n = candidates.shape[0]
    keep = [True] * n
    dup_of: list[str | None] = [None] * n
    best_sim = [0.0] * n
    if n == 0:
        return DedupResult(keep, dup_of, best_sim)
    if existing.size:
        sims = cosine_matrix(candidates, existing)
        for i in range(n):
            j = int(np.argmax(sims[i]))
            best_sim[i] = float(sims[i, j])
            if sims[i, j] > threshold:
                keep[i] = False
                dup_of[i] = existing_ids[j]
    within = cosine_matrix(candidates, candidates)
    kept: list[int] = []
    for i in range(n):
        if not keep[i]:
            continue
        if kept:
            prev = np.asarray(kept)
            j = int(prev[np.argmax(within[i, prev])])
            s = float(within[i, j])
            best_sim[i] = max(best_sim[i], s)
            if s > threshold:
                keep[i] = False
                dup_of[i] = candidate_ids[j]
                continue
        kept.append(i)
    return DedupResult(keep, dup_of, best_sim)


def coverage(cases: Iterable[dict[str, Any]], categories: Iterable[str], target: int = 5) -> dict[str, Any]:
    """Per-category active case counts, the sparse categories, and the share that meet ``target``."""
    counts: Counter[str] = Counter(c.get("category") or "unclustered" for c in cases)
    cats = sorted(set(categories) | set(counts))
    per = {c: counts.get(c, 0) for c in cats}
    sparse = [c for c in cats if per[c] < target]
    covered = len(cats) - len(sparse)
    return {
        "per_category": per,
        "sparse": sparse,
        "target": target,
        "covered_ratio": round(covered / len(cats), 4) if cats else 0.0,
    }


def coverage_boost(category: str | None, per_category: dict[str, int], target: int = 5, weight: float = 2.0) -> float:
    """Priority bonus for candidates from under-represented categories (0 once ``target`` is met)."""
    have = per_category.get(category or "unclustered", 0)
    return weight * max(0, target - have) / target
