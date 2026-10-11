"""Human-in-the-loop curation (Phase 5): review queue actions and reviewer statistics.

All logic lives here so the Streamlit dashboard stays a thin, untested-by-design view.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from evalforge.embed import Embedder, HashingEmbedder, cosine_matrix
from evalforge.labeling import polarity
from evalforge.metrics import cohen_kappa
from evalforge.store import Store

DECISIONS = ("approve", "edit", "reject")
EDITABLE = ("behavior", "quality", "difficulty", "must_contain", "must_not_contain", "reference", "rubric", "category")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def review_queue(store: Store) -> list[dict[str, Any]]:
    """Pending cases, least confident first."""
    return sorted(store.cases("pending_review"), key=lambda c: (c.get("confidence") or 0.0, c["id"]))


def apply_edits(case: dict[str, Any], edits: dict[str, Any]) -> dict[str, Any]:
    unknown = set(edits) - set(EDITABLE)
    if unknown:
        raise ValueError(f"not editable: {sorted(unknown)}")
    case = {**case, "expected": dict(case["expected"]), "assertions": dict(case.get("assertions", {}))}
    for key, value in edits.items():
        if key in ("behavior", "reference", "rubric"):
            case["expected"][key] = value
        elif key in ("must_contain", "must_not_contain"):
            case["assertions"][key] = list(value)
        elif key == "quality":
            q = int(value)
            if not 1 <= q <= 5:
                raise ValueError("quality must be 1-5")
            case["observed"] = {**case.get("observed", {}), "quality": q}
            case["polarity"] = polarity(q)
        else:
            case[key] = value
    return case


def review(
    store: Store,
    case_id: str,
    reviewer: str,
    decision: str,
    edits: dict[str, Any] | None = None,
    note: str | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """Approve, edit (then approve) or reject a case and record the review."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    case = store.get_case(case_id)
    if case is None:
        raise KeyError(case_id)
    if decision == "edit":
        case = apply_edits(case, edits or {})
    case["status"] = "rejected" if decision == "reject" else "active"
    if decision != "reject":
        case["label_source"] = "human"
    if note:
        case["notes"] = [*case.get("notes", []), {"by": reviewer, "note": note}]
    store.update_case(case)
    store.add_review(case_id, reviewer, decision, edits, note, at or _now())
    return case


def similar_cases(store: Store, case_id: str, k: int = 3, embedder: Embedder | None = None) -> list[dict[str, Any]]:
    """Closest active cases to show reviewers as context."""
    target = store.get_case(case_id)
    if target is None:
        raise KeyError(case_id)
    active = [c for c in store.cases("active") if c["id"] != case_id]
    if not active:
        return []
    emb = embedder or HashingEmbedder()
    sims = cosine_matrix(emb.embed([target["input"]["prompt"]]), emb.embed([c["input"]["prompt"] for c in active]))[0]
    order = np.argsort(-sims)[:k]
    return [{**active[i], "similarity": round(float(sims[i]), 4)} for i in order]


def reviewer_stats(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Throughput per reviewer and Cohen's kappa on cases that two reviewers both decided."""
    per = Counter(r["reviewer"] for r in reviews)
    per_day: dict[str, Counter[str]] = defaultdict(Counter)
    for r in reviews:
        per_day[r["reviewer"]][str(r.get("created_at") or "")[:10]] += 1
    by_case: dict[str, dict[str, str]] = defaultdict(dict)
    for r in reviews:
        verdict = "reject" if r["decision"] == "reject" else "accept"
        by_case[r["case_id"]][r["reviewer"]] = verdict
    pairs: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for votes in by_case.values():
        names = sorted(votes)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = pairs.setdefault((names[i], names[j]), ([], []))
                a.append(votes[names[i]])
                b.append(votes[names[j]])
    kappas = {f"{x}~{y}": round(cohen_kappa(a, b), 4) for (x, y), (a, b) in pairs.items()}
    return {
        "reviews": len(reviews),
        "per_reviewer": dict(per),
        "per_reviewer_per_day": {k: dict(v) for k, v in per_day.items()},
        "kappa": kappas,
    }
