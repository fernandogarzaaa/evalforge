"""Multi-run labeling with confidence-based routing (Phase 3.3).

Each candidate is labeled ``k`` times (temperature > 0 for a real LLM; seeded noise
for the mock). Agreement between runs is the confidence: unanimous labels go straight
into the dataset, anything contested goes to the human review queue.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from evalforge.judge import Judge, Label


def quality_bucket(q: int) -> str:
    return "good" if q >= 4 else "bad" if q <= 2 else "mixed"


def polarity(q: int) -> str:
    """High-quality interactions are positive cases (keep doing this); the rest are negative (fix this)."""
    return "positive" if q >= 4 else "negative"


@dataclass
class AggregatedLabel:
    label: Label
    confidence: float
    runs: list[Label]
    route: str  # "auto" | "review"


def _mode(values: Sequence[str]) -> str:
    counts = Counter(values)
    return sorted(counts.items(), key=lambda kv: (-kv[1], values.index(kv[0])))[0][0]


def aggregate(runs: Sequence[Label], threshold: float = 0.85) -> AggregatedLabel:
    """Combine runs: median quality, modal behaviour/difficulty; confidence = mean agreement rate."""
    if not runs:
        raise ValueError("no label runs to aggregate")
    behaviors = [r.behavior for r in runs]
    buckets = [quality_bucket(r.quality) for r in runs]
    beh = _mode(behaviors)
    bucket = _mode(buckets)
    agree_b = behaviors.count(beh) / len(runs)
    agree_q = buckets.count(bucket) / len(runs)
    confidence = round((agree_b + agree_q) / 2, 4)
    qualities = sorted(r.quality for r in runs if quality_bucket(r.quality) == bucket)
    median_q = qualities[(len(qualities) - 1) // 2]
    # Take assertions and reference from the first run that agrees with the consensus.
    base = next((r for r in runs if r.behavior == beh and quality_bucket(r.quality) == bucket), runs[0])
    label = Label(
        quality=median_q,
        behavior=beh,
        difficulty=_mode([r.difficulty for r in runs]),
        must_contain=list(base.must_contain),
        must_not_contain=list(base.must_not_contain),
        reference=base.reference,
        rubric=list(base.rubric),
        rationale=base.rationale,
    )
    return AggregatedLabel(label, confidence, list(runs), "auto" if confidence >= threshold else "review")


def label_with_agreement(
    judge: Judge,
    prompt: str,
    response: str,
    k: int = 3,
    feedback: str | None = None,
    key: str = "",
    threshold: float = 0.85,
) -> AggregatedLabel:
    runs = [judge.label(prompt, response, run_index=i, feedback=feedback, key=key) for i in range(k)]
    return aggregate(runs, threshold)
