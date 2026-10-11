"""Sampling strategies (Phase 1.3): uniform, stratified and signal-boosted."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence

from evalforge.schema import LogRecord

MODES = ("uniform", "stratified", "signal")


def uniform(records: Sequence[LogRecord], n: int, seed: int = 0) -> list[LogRecord]:
    if n >= len(records):
        return list(records)
    return random.Random(seed).sample(list(records), n)


def stratum_key(record: LogRecord) -> str:
    return f"{record.feature}|{record.model}|{'error' if record.is_error else 'ok'}"


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation (largest remainder), then at least one slot per non-empty stratum,
    taken from the currently largest allocation."""
    total = sum(sizes.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in sizes}
    if n >= total:
        return dict(sizes)
    keys = sorted(k for k, v in sizes.items() if v > 0)
    quotas = {k: n * sizes[k] / total for k in keys}
    alloc = {k: int(math.floor(quotas[k])) for k in keys}
    left = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), -sizes[k], k))[:left]:
        alloc[k] += 1
    for k in sorted(keys, key=lambda k: (-sizes[k], k)):
        if alloc[k] == 0:
            donor = max(keys, key=lambda d: (alloc[d], sizes[d]))
            if alloc[donor] <= 1:
                break
            alloc[donor] -= 1
            alloc[k] = 1
    return {k: alloc.get(k, 0) for k in sizes}


def stratified(
    records: Sequence[LogRecord], n: int, seed: int = 0, key: Callable[[LogRecord], str] = stratum_key
) -> list[LogRecord]:
    groups: dict[str, list[LogRecord]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    alloc = allocate({k: len(v) for k, v in groups.items()}, n)
    rng = random.Random(seed)
    out: list[LogRecord] = []
    for k in sorted(groups):
        out.extend(rng.sample(groups[k], alloc[k]))
    return out


def signal_weight(record: LogRecord, retried: set[str], slow: set[str]) -> float:
    w = 1.0
    if record.feedback == "down":
        w += 4.0
    if record.request_id in retried:
        w += 3.0
    if record.content_filtered:
        w += 2.0
    if record.edited:
        w += 2.0
    if record.request_id in slow:
        w += 2.0
    return w


def signal_boosted(
    records: Sequence[LogRecord], n: int, seed: int = 0, retried: set[str] | None = None, slow: set[str] | None = None
) -> list[LogRecord]:
    """Weighted sampling without replacement (Efraimidis-Spirakis): keys u^(1/w), take the top n."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    retried = retried or set()
    slow = slow or set()
    # log(u) / w orders exactly like u ** (1 / w); 1 - random() keeps u in (0, 1].
    keyed = [(math.log(1.0 - rng.random()) / signal_weight(r, retried, slow), i) for i, r in enumerate(records)]
    keyed.sort(reverse=True)
    return [records[i] for _, i in keyed[:n]]


def sample(
    records: Sequence[LogRecord],
    n: int,
    mode: str = "signal",
    seed: int = 0,
    retried: set[str] | None = None,
    slow: set[str] | None = None,
) -> list[LogRecord]:
    if mode == "uniform":
        return uniform(records, n, seed)
    if mode == "stratified":
        return stratified(records, n, seed)
    if mode == "signal":
        return signal_boosted(records, n, seed, retried, slow)
    raise ValueError(f"unknown sampling mode: {mode} (choose from {', '.join(MODES)})")
