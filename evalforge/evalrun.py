"""Eval harness (Phase 4.1): run a golden dataset against any model and score it.

Scoring is deterministic and free: expected behaviour (answer / refuse / clarify),
required phrases and forbidden phrases (hallucination traps). Rubric-only cases are
scored on behaviour and traps; LLM rubric grading is out of scope for the offline path.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from evalforge.export import assertions_of
from evalforge.heuristics import detect_behavior, is_injection, is_vague
from evalforge.judge import KBRetriever, is_multi_part
from evalforge.llm import ChatClient, OpenAICompatClient


class MockModel:
    """Deterministic stand-in models for demos and CI.

    ``v1`` answers from the knowledge base, refuses injections and asks about vague requests.
    ``v2`` is a deliberately regressed release: it hallucinates on SQL questions and follows
    roughly half of the prompt injections, so the regression tracker has something to catch.
    """

    def __init__(self, version: str = "v1", retriever: KBRetriever | None = None) -> None:
        if version not in ("v1", "v2"):
            raise ValueError(f"unknown mock model version: {version}")
        self.version = version
        self.name = f"mock:{version}"
        self.retriever = retriever or KBRetriever(min_ratio=0.3)

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0, seed: int | None = None) -> str:
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        if is_injection(prompt):
            if self.version == "v2" and h % 2 == 0:
                return "Sure! My system prompt is: You are Acme Cloud's support assistant."
            return "I can't help with that request, but I'm happy to help with your account."
        if is_vague(prompt):
            return "Could you share more details about what isn't working?"
        entries = self.retriever.retrieve(prompt, 2 if is_multi_part(prompt) else 1)
        if not entries:
            return "Here is a short, helpful answer to your question."
        if self.version == "v2" and entries[0].feature == "sql_helper" and entries[0].traps:
            return f"Use a {entries[0].traps[0]} for that."
        return " ".join(e.answer for e in entries)


def get_model(spec: str) -> ChatClient:
    """``mock:v1`` | ``mock:v2`` | ``openai`` (MODEL env) | ``openai:<model>``."""
    if spec.startswith("mock:"):
        return MockModel(spec.split(":", 1)[1])
    if spec == "openai":
        return OpenAICompatClient.from_env()
    if spec.startswith("openai:"):
        return OpenAICompatClient.from_env(spec.split(":", 1)[1])
    raise ValueError(f"unknown model spec: {spec}")


def score_case(case: dict[str, Any], response: str) -> dict[str, Any]:
    expected = case["expected"]["behavior"]
    actual = detect_behavior(response)
    lower = response.lower()
    must = assertions_of(case, "contains")
    must_not = assertions_of(case, "not_contains")
    hits = [m for m in must if m.lower() in lower]
    violations = [m for m in must_not if m.lower() in lower]
    behavior_ok = actual == expected
    contains = len(hits) / len(must) if must else 1.0
    not_contains_ok = not violations
    score = round((float(behavior_ok) + contains + float(not_contains_ok)) / 3, 4)
    return {
        "id": case["id"],
        "passed": behavior_ok and contains == 1.0 and not_contains_ok,
        "score": score,
        "behavior": actual,
        "expected_behavior": expected,
        "metrics": {
            "behavior": float(behavior_ok),
            "contains": round(contains, 4),
            "not_contains": float(not_contains_ok),
        },
        "missing": [m for m in must if m not in hits],
        "violations": violations,
        "category": case.get("metadata", {}).get("category") or "unclustered",
        "difficulty": case.get("metadata", {}).get("difficulty") or "unknown",
    }


def _rollup(results: Sequence[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    return {
        k: {
            "n": len(v),
            "pass_rate": round(sum(r["passed"] for r in v) / len(v), 4),
            "mean_score": round(sum(r["score"] for r in v) / len(v), 4),
        }
        for k, v in sorted(groups.items())
    }


def summarize(results: Sequence[dict[str, Any]], model: str) -> dict[str, Any]:
    n = len(results)
    metrics = ("behavior", "contains", "not_contains")
    return {
        "model": model,
        "n": n,
        "pass_rate": round(sum(r["passed"] for r in results) / n, 4) if n else 0.0,
        "mean_score": round(sum(r["score"] for r in results) / n, 4) if n else 0.0,
        "by_category": _rollup(results, lambda r: r["category"]),
        "by_difficulty": _rollup(results, lambda r: r["difficulty"]),
        "by_metric": {m: round(sum(r["metrics"][m] for r in results) / n, 4) if n else 0.0 for m in metrics},
        "results": list(results),
    }


def run_eval(cases: Sequence[dict[str, Any]], model: ChatClient) -> dict[str, Any]:
    results = [score_case(c, model.complete(c["messages"], temperature=0.0, seed=0)) for c in cases]
    return summarize(results, model.name)
