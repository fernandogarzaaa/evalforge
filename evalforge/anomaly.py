"""Edge-case and anomaly signals (Phase 2.2).

Each signal is cheap, explainable and independent so reviewers can see *why* an
interaction was picked: novel (HDBSCAN noise), low model confidence, immediate
rephrased retry, response-length outlier, latency outlier, content filter, negative
feedback and user edits.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from evalforge.embed import Embedder, HashingEmbedder
from evalforge.schema import LogRecord, parse_timestamp

FLAG_WEIGHTS = {
    "negative_feedback": 3.0,
    "retried": 3.0,
    "filtered": 2.5,
    "novel": 2.5,
    "edited": 2.0,
    "low_confidence": 2.0,
    "length_outlier": 1.5,
    "latency_outlier": 1.0,
}


def robust_z(values: Sequence[float]) -> list[float]:
    """Median/MAD z-scores; 0 for every value when the spread is zero."""
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad == 0:
        return [0.0] * len(values)
    return [float(v) for v in 0.6745 * (arr - med) / mad]


def outliers_by_group(
    records: Sequence[LogRecord],
    value: Callable[[LogRecord], float],
    threshold: float = 3.5,
    group: Callable[[LogRecord], str] = lambda r: r.feature,
) -> set[str]:
    groups: dict[str, list[LogRecord]] = defaultdict(list)
    for r in records:
        groups[group(r)].append(r)
    out: set[str] = set()
    for members in groups.values():
        zs = robust_z([value(r) for r in members])
        out.update(r.request_id for r, z in zip(members, zs) if abs(z) > threshold)
    return out


def length_outliers(records: Sequence[LogRecord], threshold: float = 3.5) -> set[str]:
    """Unusually long or short responses (log word count, per feature). Filtered/empty responses excluded."""
    candidates = [r for r in records if not r.is_error]
    found = outliers_by_group(candidates, lambda r: math.log1p(len(r.response.split())), threshold)
    found.update(r.request_id for r in candidates if len(r.response.split()) < 3)
    return found


def latency_outliers(records: Sequence[LogRecord], threshold: float = 3.5) -> set[str]:
    return outliers_by_group(records, lambda r: math.log1p(r.latency_ms), threshold)


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.asarray(np.sum(an * bn, axis=1))


def detect_retries(
    records: Iterable[LogRecord], window_s: float = 120.0, min_similarity: float = 0.1, embedder: Embedder | None = None
) -> set[str]:
    """Ids of interactions the user immediately re-asked in the same session (the retried original)."""
    by_session: dict[str, list[LogRecord]] = defaultdict(list)
    for r in records:
        if r.session_id:
            by_session[r.session_id].append(r)
    pairs: list[tuple[LogRecord, LogRecord]] = []
    for items in by_session.values():
        items.sort(key=lambda r: r.timestamp)
        for a, b in zip(items, items[1:]):
            gap = (parse_timestamp(b.timestamp) - parse_timestamp(a.timestamp)).total_seconds()
            if 0 <= gap <= window_s:
                pairs.append((a, b))
    if not pairs:
        return set()
    emb = embedder or HashingEmbedder()
    ea = emb.embed([a.user_prompt for a, _ in pairs])
    eb = emb.embed([b.user_prompt for _, b in pairs])
    sims = rowwise_cosine(ea, eb)
    return {a.request_id for (a, _), s in zip(pairs, sims) if s >= min_similarity}


def compute_flags(
    records: Sequence[LogRecord],
    novel_ids: set[str] | None = None,
    retried_ids: set[str] | None = None,
    low_confidence: float = 0.7,
) -> dict[str, list[str]]:
    novel_ids = novel_ids or set()
    retried_ids = retried_ids if retried_ids is not None else detect_retries(records)
    long_short = length_outliers(records)
    slow = latency_outliers(records)
    flags: dict[str, list[str]] = {}
    for r in records:
        f: list[str] = []
        if r.request_id in novel_ids:
            f.append("novel")
        if r.confidence is not None and r.confidence < low_confidence:
            f.append("low_confidence")
        if r.request_id in retried_ids:
            f.append("retried")
        if r.request_id in long_short:
            f.append("length_outlier")
        if r.request_id in slow:
            f.append("latency_outlier")
        if r.content_filtered:
            f.append("filtered")
        if r.feedback == "down":
            f.append("negative_feedback")
        if r.edited:
            f.append("edited")
        flags[r.request_id] = f
    return flags


def flag_score(flags: Iterable[str]) -> float:
    return sum(FLAG_WEIGHTS.get(f, 1.0) for f in flags)
