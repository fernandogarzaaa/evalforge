"""Regression detection between two eval reports (Phase 4.2)."""

from __future__ import annotations

from typing import Any


def compare(
    prev: dict[str, Any], curr: dict[str, Any], score_threshold: float = 0.2, category_threshold: float = 0.05
) -> dict[str, Any]:
    """New failures, new passes, large score moves and category-level pass-rate shifts.

    Only cases present in both runs are compared case-by-case; cases added since the
    previous run are reported separately so dataset growth is never mistaken for a regression.
    """
    before = {r["id"]: r for r in prev["results"]}
    after = {r["id"]: r for r in curr["results"]}
    shared = sorted(set(before) & set(after))
    new_failures = [i for i in shared if before[i]["passed"] and not after[i]["passed"]]
    new_passes = [i for i in shared if not before[i]["passed"] and after[i]["passed"]]
    score_changes = [
        {
            "id": i,
            "before": before[i]["score"],
            "after": after[i]["score"],
            "delta": round(after[i]["score"] - before[i]["score"], 4),
        }
        for i in shared
        if abs(round(after[i]["score"] - before[i]["score"], 4)) >= score_threshold
    ]
    shifts = []
    for cat in sorted(set(prev["by_category"]) & set(curr["by_category"])):
        delta = round(curr["by_category"][cat]["pass_rate"] - prev["by_category"][cat]["pass_rate"], 4)
        if abs(delta) >= category_threshold:
            shifts.append(
                {
                    "category": cat,
                    "before": prev["by_category"][cat]["pass_rate"],
                    "after": curr["by_category"][cat]["pass_rate"],
                    "delta": delta,
                }
            )
    return {
        "previous_model": prev.get("model"),
        "current_model": curr.get("model"),
        "compared": len(shared),
        "added_cases": sorted(set(after) - set(before)),
        "new_failures": new_failures,
        "new_passes": new_passes,
        "score_changes": score_changes,
        "category_shifts": shifts,
        "regressed": bool(new_failures) or any(s["delta"] < 0 for s in shifts),
    }
