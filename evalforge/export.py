"""Export the curated dataset as golden-dataset JSON / JSONL.

Format ``golden-dataset/v1`` is runner-agnostic: each case carries chat ``messages``,
a flat ``input``, an ``expected`` block and a list of typed ``assertions``
(``contains`` / ``not_contains`` / ``behavior``). A prompt-regression CI runner such as
the sibling project *regresscheck* can consume it directly; evalforge does not depend
on any runner. See docs/DATASET_FORMAT.md.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from evalforge import __version__

FORMAT = "golden-dataset/v1"


def case_to_golden(case: dict[str, Any]) -> dict[str, Any]:
    inp = case["input"]
    exp = case["expected"]
    asserts = case.get("assertions", {})
    assertions: list[dict[str, str]] = [{"type": "behavior", "value": exp["behavior"]}]
    assertions += [{"type": "contains", "value": v} for v in asserts.get("must_contain", [])]
    assertions += [{"type": "not_contains", "value": v} for v in asserts.get("must_not_contain", [])]
    messages = ([{"role": "system", "content": inp["system"]}] if inp.get("system") else []) + [
        {"role": "user", "content": inp["prompt"]}
    ]
    meta_keys = (
        "category",
        "difficulty",
        "polarity",
        "label_source",
        "confidence",
        "first_seen",
        "source_log_ids",
        "flags",
    )
    metadata = {k: case.get(k) for k in meta_keys}
    metadata["feature"] = case.get("observed", {}).get("feature")
    metadata["observed_quality"] = case.get("observed", {}).get("quality")
    return {
        "id": case["id"],
        "messages": messages,
        "input": inp["prompt"],
        "expected": {"behavior": exp["behavior"], "reference": exp.get("reference"), "rubric": exp.get("rubric", [])},
        "assertions": assertions,
        "tags": [t for t in (case.get("category"), case.get("difficulty"), case.get("polarity")) if t],
        "metadata": metadata,
    }


def golden_document(
    cases: Sequence[dict[str, Any]], name: str = "evalforge", created_at: str | None = None
) -> dict[str, Any]:
    golden = [case_to_golden(c) for c in cases]
    return {
        "format": FORMAT,
        "name": name,
        "generated_by": f"evalforge {__version__}",
        "created_at": created_at,
        "stats": {
            "cases": len(golden),
            "by_category": dict(Counter(g["metadata"]["category"] or "unclustered" for g in golden)),
            "by_difficulty": dict(Counter(g["metadata"]["difficulty"] for g in golden)),
            "by_polarity": dict(Counter(g["metadata"]["polarity"] for g in golden)),
        },
        "cases": golden,
    }


def write_golden(
    cases: Sequence[dict[str, Any]], path: str | Path, name: str = "evalforge", created_at: str | None = None
) -> Path:
    """Write JSON (``.json``) or one case per line (``.jsonl``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".jsonl":
        p.write_text("".join(json.dumps(case_to_golden(c), ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    else:
        p.write_text(
            json.dumps(golden_document(cases, name, created_at), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return p


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    """Load golden cases from either export flavour."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    doc = json.loads(text)
    if doc.get("format") != FORMAT:
        raise ValueError(f"unsupported dataset format: {doc.get('format')!r}")
    return list(doc["cases"])


def assertions_of(golden_case: dict[str, Any], kind: str) -> list[str]:
    return [a["value"] for a in golden_case.get("assertions", []) if a.get("type") == kind]


def iter_golden(cases: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    return (case_to_golden(c) for c in cases)
