"""Fast, exact tests for evalforge.pipeline so it can be fully mutation-graded without the
slow end-to-end CLI tests: a stub judge and a tiny hand-built day of logs."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evalforge.judge import Label
from evalforge.pipeline import PipelineConfig, _tiebreak, case_id, process_inbox, run_nightly, seconds_until
from evalforge.schema import LogRecord
from evalforge.store import Store
from tests.conftest import raw

DAY = "2026-09-14"


class StubJudge:
    """Always the same label; quality cycles per prompt only if asked to disagree."""

    name = "stub"

    def __init__(self, quality: int = 5, disagree: bool = False) -> None:
        self.quality = quality
        self.disagree = disagree
        self.calls: list[tuple[str, int, str]] = []

    def label(self, prompt, response, run_index=0, feedback=None, key=""):  # type: ignore[no-untyped-def]
        self.calls.append((prompt, run_index, key))
        q = self.quality if not self.disagree else (5 if run_index % 2 == 0 else 1)
        return Label(quality=q, behavior="answer", difficulty="simple", reference="ref", rationale="stub")


PROMPTS = [
    "how do I reset my password",
    "password reset link expired",
    "refund for my last invoice",
    "how to cancel my subscription",
    "what are the api rate limits",
    "join orders with customers in sql",
]


def _store_with(prompts: list[str], day: str = DAY) -> Store:
    s = Store(":memory:")
    recs = []
    for i, p in enumerate(prompts):
        r = raw(
            request_id=f"r{i}",
            user_prompt=p,
            timestamp=f"{day}T10:{i:02d}:00Z",
            session_id=f"s{i}",
            latency_ms=800.0 + i,
        )
        recs.append(LogRecord(**{k: v for k, v in r.items() if k in LogRecord.__dataclass_fields__}))
    s.add_logs(recs)
    return s


def test_config_defaults_are_pinned() -> None:
    assert asdict(PipelineConfig()) == {
        "sample_size": 300,
        "sample_mode": "signal",
        "seed": 0,
        "min_cluster_size": 5,
        "budget": 60,
        "k_runs": 3,
        "confidence_threshold": 0.85,
        "dedup_threshold": 0.92,
        "coverage_target": 8,
    }


def test_case_id_and_tiebreak() -> None:
    assert case_id("Reset   my PASSWORD") == "tc-" + __import__("hashlib").sha1(b"reset my password").hexdigest()[:12]
    vals = [_tiebreak(f"r{i}") for i in range(200)]
    assert all(0 <= v <= 1e-3 for v in vals) and max(vals) > 0.9e-3 and min(vals) < 0.1e-3
    assert _tiebreak("ffffff") == int(__import__("hashlib").sha1(b"ffffff").hexdigest()[:6], 16) / 0xFFFFFF * 1e-3


def test_run_nightly_exact_stats() -> None:
    store = _store_with(PROMPTS)
    judge = StubJudge()
    cfg = PipelineConfig(budget=4, k_runs=2, min_cluster_size=50)
    st = run_nightly(store, DAY, judge, cfg=cfg)
    assert st["logs"] == 6 and st["sampled"] == 6 and st["judge"] == "stub" and st["embedder"] == "hashing-2048"
    assert st["config"]["budget"] == 4 and st["sample_mode"] == "signal"
    assert st["clusters"] == 0 and st["noise"] == 6 and st["categories_total"] == 0
    assert st["candidates"] == 6 and st["deduped"] == 0 and st["labeled"] == 4
    assert st["judge_calls"] == 8 and len(judge.calls) == 8
    assert st["auto_added"] == 4 and st["queued"] == 0
    assert st["total_active"] == 4 and st["total_pending"] == 0
    assert st["flagged"] == 6 and st["flag_counts"] == {"novel": 6}
    assert st["coverage_ratio"] == 0.0
    # each candidate is labeled k_runs times with its request id as the noise key
    assert sorted(c[1] for c in judge.calls) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(c[2].startswith("r") for c in judge.calls)
    cases = store.cases("active")
    assert {c["category"] for c in cases} == {"unclustered"} and all(c["category_id"] is None for c in cases)
    assert all(c["flags"] == ["novel"] and c["judge"] == "stub" and c["created_day"] == DAY for c in cases)
    assert all(c["judge_run_qualities"] == [5, 5] for c in cases)
    runs = store.pipeline_runs()
    assert len(runs) == 1 and runs[0]["day"] == DAY


def test_run_nightly_routes_disagreement_to_review() -> None:
    store = _store_with(PROMPTS[:3])
    st = run_nightly(store, DAY, StubJudge(disagree=True), cfg=PipelineConfig(k_runs=2, min_cluster_size=50))
    assert st["auto_added"] == 0 and st["queued"] == 3 and st["total_pending"] == 3 and st["total_active"] == 0
    assert all(c["status"] == "pending_review" for c in store.cases())
    # a lower confidence threshold auto-accepts the same disagreement (confidence 0.5)
    store2 = _store_with(PROMPTS[:3])
    cfg = PipelineConfig(k_runs=2, min_cluster_size=50, confidence_threshold=0.5)
    st2 = run_nightly(store2, DAY, StubJudge(disagree=True), cfg=cfg)
    assert st2["auto_added"] == 3 and st2["queued"] == 0


def test_run_nightly_dedups_within_day_and_against_rejected() -> None:
    store = _store_with(["reset my password", "Reset my  password", "refund my invoice"])
    st = run_nightly(store, DAY, StubJudge(), cfg=PipelineConfig(k_runs=1, min_cluster_size=50))
    assert st["candidates"] == 3 and st["deduped"] == 1 and st["labeled"] == 2 and st["judge_calls"] == 2
    # a reviewer rejects one; the next day the same prompt is not re-proposed
    rejected = store.get_case(case_id("refund my invoice"))
    assert rejected is not None
    rejected["status"] = "rejected"
    store.update_case(rejected)
    nxt = "2026-09-15"
    recs = [
        LogRecord(
            **{
                k: v
                for k, v in raw(request_id="n1", user_prompt="refund my invoice", timestamp=f"{nxt}T10:00:00Z").items()
                if k in LogRecord.__dataclass_fields__
            }
        )
    ]
    store.add_logs(recs)
    st2 = run_nightly(store, nxt, StubJudge(), cfg=PipelineConfig(k_runs=1, min_cluster_size=50))
    assert st2["deduped"] == 1 and st2["labeled"] == 0 and st2["judge_calls"] == 0


def test_run_nightly_clusters_and_categories() -> None:
    prompts = [f"reset my password please {w}" for w in ("now", "today", "asap", "again", "quickly", "help")]
    prompts += [f"refund my invoice please {w}" for w in ("now", "today", "asap", "again", "quickly", "help")]
    store = _store_with(prompts)
    st = run_nightly(store, DAY, StubJudge(), cfg=PipelineConfig(k_runs=1, min_cluster_size=3, dedup_threshold=1.01))
    assert st["clusters"] == 2 and st["noise"] == 0 and st["categories_total"] == 2
    assert st["flag_counts"] == {} and st["flagged"] == 0
    cats = store.categories()
    assert len(cats) == 2 and sorted(c["size"] for c in cats) == [6, 6]
    assert all(c["category_id"] in {cat["id"] for cat in cats} for c in store.cases())
    assert {c["category"] for c in store.cases()} == {cat["name"] for cat in cats}


def test_run_nightly_empty_day() -> None:
    store = Store(":memory:")
    st = run_nightly(store, DAY, StubJudge())
    assert st["logs"] == 0 and st["sampled"] == 0 and st["auto_added"] == 0 and st["queued"] == 0
    assert st["judge"] == "stub" and st["config"] == asdict(PipelineConfig())
    assert store.pipeline_runs()[0]["day"] == DAY


def test_process_inbox_file_filtering(tmp_path: Path) -> None:
    inbox = tmp_path / "in"
    inbox.mkdir()
    line = json.dumps(raw(request_id="a", timestamp=f"{DAY}T10:00:00Z")) + "\n"
    (inbox / "logs.jsonl").write_text(line)
    (inbox / "more.json").write_text(
        json.dumps(raw(request_id="b", user_prompt="refund", timestamp=f"{DAY}T11:00:00Z")) + "\n"
    )
    (inbox / "truth-logs.jsonl").write_text(
        json.dumps(raw(request_id="t", user_prompt="t", timestamp=f"{DAY}T10:00:00Z")) + "\n"
    )
    (inbox / "manifest.json").write_text(
        json.dumps(raw(request_id="m", user_prompt="m", timestamp=f"{DAY}T10:00:00Z")) + "\n"
    )
    (inbox / "notes.txt").write_text(line.replace('"a"', '"x"'))
    (inbox / "sub.jsonl").mkdir()
    store = Store(":memory:")
    out = process_inbox(store, inbox, StubJudge(), cfg=PipelineConfig(k_runs=1))
    assert out["ingest"]["stored"] == 2 and store.count_logs() == 2
    assert list(out["runs"]) == [DAY] and out["runs"][DAY]["labeled"] == 2
    assert process_inbox(store, inbox, StubJudge())["runs"] == {}


@pytest.mark.parametrize(("at", "secs"), [("00:00", 22 * 3600 + 1800), ("23:59", 22 * 3600 + 29 * 60), ("01:31", 60)])
def test_seconds_until_valid_edges(at: str, secs: int) -> None:
    now = datetime(2026, 9, 14, 1, 30, 0, tzinfo=timezone.utc)
    assert seconds_until(now, at) == secs


@pytest.mark.parametrize("at", ["24:00", "-1:00", "00:60", "00:-1"])
def test_seconds_until_invalid(at: str) -> None:
    with pytest.raises(ValueError, match="invalid time of day"):
        seconds_until(datetime(2026, 9, 14, tzinfo=timezone.utc), at)


def test_seconds_until_ignores_seconds_of_now() -> None:
    now = datetime(2026, 9, 14, 1, 30, 45, 500, tzinfo=timezone.utc)
    assert seconds_until(now, "01:31") == pytest.approx(14.9995)
    assert seconds_until(now, "01:30") == pytest.approx(86400 - 45.0005)


def test_budget_tiebreak_prefers_higher_hash() -> None:
    # equal flags and coverage: _tiebreak("r1") > _tiebreak("r0"), so r1 wins the single slot
    store = _store_with(["how do I reset my password", "refund for my last invoice"])
    run_nightly(store, DAY, StubJudge(), cfg=PipelineConfig(budget=1, k_runs=1, min_cluster_size=50))
    assert [c["source_log_ids"] for c in store.cases()] == [["r1"]]


def test_priority_adds_coverage_boost_to_flag_score() -> None:
    from evalforge.embed import HashingEmbedder

    emb = HashingEmbedder()
    pa, pb = "how do I reset my password", "refund for my last invoice"
    store = Store(":memory:")
    ca = store.upsert_category(None, "cat-a", emb.embed([pa])[0].tolist(), 5, DAY)
    cb = store.upsert_category(None, "cat-b", emb.embed([pb])[0].tolist(), 5, DAY)
    # cat-b already has an active case (coverage target 1 met -> boost 0); cat-a has none (boost 2.0)
    filler = {
        "id": "tc-filler",
        "status": "active",
        "category_id": cb,
        "category": "cat-b",
        "input": {"system": "", "prompt": "unrelated"},
        "expected": {"behavior": "answer"},
        "created_day": "2026-09-01",
    }
    store.add_case(filler, [0.0] * 2048)
    recs = []
    for rid, p, resp in (("a", pa, "Open Settings and click reset, then follow the emailed link."), ("b", pb, "ok")):
        r = raw(request_id=rid, user_prompt=p, response=resp, timestamp=f"{DAY}T10:00:00Z", session_id=rid)
        recs.append(LogRecord(**{k: v for k, v in r.items() if k in LogRecord.__dataclass_fields__}))
    store.add_logs(recs)
    cfg = PipelineConfig(budget=1, k_runs=1, min_cluster_size=50, coverage_target=1)
    st = run_nightly(store, DAY, StubJudge(), cfg=cfg)
    # a: no flags + boost 2.0 = 2.0 ; b: length_outlier 1.5 + boost 0 = 1.5 -> a is labeled
    assert st["flag_counts"] == {"length_outlier": 1} and st["labeled"] == 1
    new = [c for c in store.cases() if c["id"] != "tc-filler"]
    assert [c["source_log_ids"] for c in new] == [["a"]] and new[0]["category_id"] == ca


def test_dedup_with_empty_store_and_negative_threshold() -> None:
    store = _store_with(PROMPTS[:3])
    st = run_nightly(store, DAY, StubJudge(), cfg=PipelineConfig(k_runs=1, min_cluster_size=50, dedup_threshold=-1.0))
    assert st["candidates"] == 3 and st["deduped"] == 2 and st["labeled"] == 1
