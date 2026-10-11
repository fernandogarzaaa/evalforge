"""The nightly pipeline: sample -> cluster -> flag -> select -> dedup -> label -> route -> store."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from evalforge.anomaly import compute_flags, detect_retries, flag_score, latency_outliers
from evalforge.cluster import UNCLUSTERED, assign_categories, cluster_embeddings
from evalforge.dedup import coverage, coverage_boost, dedup
from evalforge.embed import Embedder, HashingEmbedder
from evalforge.ingest import IngestStats, ingest_file
from evalforge.judge import Judge, MockJudge
from evalforge.labeling import label_with_agreement, polarity
from evalforge.redact import Redactor
from evalforge.sampling import sample
from evalforge.schema import LogRecord
from evalforge.store import Store


@dataclass
class PipelineConfig:
    sample_size: int = 300
    sample_mode: str = "signal"
    seed: int = 0
    min_cluster_size: int = 5
    budget: int = 60  # max candidates sent to the judge per night
    k_runs: int = 3
    confidence_threshold: float = 0.85
    dedup_threshold: float = 0.92
    coverage_target: int = 8


def case_id(prompt: str) -> str:
    return "tc-" + hashlib.sha1(" ".join(prompt.lower().split()).encode()).hexdigest()[:12]


def _tiebreak(request_id: str) -> float:
    return int(hashlib.sha1(request_id.encode()).hexdigest()[:6], 16) / 0xFFFFFF * 1e-3


def run_nightly(
    store: Store,
    day: str,
    judge: Judge | None = None,
    embedder: Embedder | None = None,
    cfg: PipelineConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PipelineConfig()
    judge = judge or MockJudge()
    embedder = embedder or HashingEmbedder()
    records = store.logs_for_day(day)
    stats: dict[str, Any] = {
        "logs": len(records),
        "judge": judge.name,
        "embedder": embedder.name,
        "config": asdict(cfg),
    }
    if not records:
        stats.update(sampled=0, auto_added=0, queued=0)
        store.add_pipeline_run(day, stats)
        return stats

    retried_all = detect_retries(records, embedder=embedder)
    slow_all = latency_outliers(records)
    sampled = sample(records, cfg.sample_size, cfg.sample_mode, cfg.seed, retried_all, slow_all)
    prompts = [r.user_prompt for r in sampled]
    x = embedder.embed(prompts)

    labels = cluster_embeddings(x, cfg.min_cluster_size, seed=cfg.seed)
    assignment = assign_categories(prompts, x, labels, store.categories())
    cat_ids: list[int | None] = list(assignment.category_ids)
    for update in assignment.updates:
        cid = store.upsert_category(update["id"], update["name"], update["centroid"], update["size"], day)
        for m in update["members"]:
            cat_ids[m] = cid
    names = {c["id"]: c["name"] for c in store.categories()}
    cat_names = [names.get(c, UNCLUSTERED) if c is not None else UNCLUSTERED for c in cat_ids]
    novel = {r.request_id for r, c in zip(sampled, cat_ids) if c is None}
    flags = compute_flags(sampled, novel_ids=novel, retried_ids=retried_all & {r.request_id for r in sampled})

    active = store.cases("active")
    cov = coverage(active, names.values(), cfg.coverage_target)
    priority = [
        flag_score(flags[r.request_id])
        + coverage_boost(cat, cov["per_category"], cfg.coverage_target)
        + _tiebreak(r.request_id)
        for r, cat in zip(sampled, cat_names)
    ]
    ranked = sorted(range(len(sampled)), key=lambda i: -priority[i])

    # Dedup before labeling so a duplicate never costs a judge call, and before applying the
    # budget so duplicates do not eat it. Rejected cases count as existing: the pipeline must
    # not keep re-proposing what a reviewer already threw out.
    ex_ids, ex_emb = store.case_embeddings(("active", "pending_review", "rejected"))
    existing = np.asarray(ex_emb, dtype=np.float32) if ex_emb else np.zeros((0, x.shape[1]), dtype=np.float32)
    result = dedup(x[ranked], [case_id(prompts[i]) for i in ranked], existing, ex_ids, cfg.dedup_threshold)
    selected = [i for i, keep in zip(ranked, result.keep) if keep][: cfg.budget]
    deduped = sum(1 for keep in result.keep if not keep)

    added = queued = 0
    for i in selected:
        rec = sampled[i]
        agg = label_with_agreement(
            judge, rec.user_prompt, rec.response, cfg.k_runs, rec.feedback, rec.request_id, cfg.confidence_threshold
        )
        case = build_case(
            rec,
            agg.label.to_dict(),
            agg.confidence,
            agg.route,
            cat_ids[i],
            cat_names[i],
            flags[rec.request_id],
            day,
            judge.name,
            [r.quality for r in agg.runs],
        )
        store.add_case(case, x[i].tolist())
        added += agg.route == "auto"
        queued += agg.route == "review"

    final_active = store.cases("active")
    stats.update(
        {
            "sampled": len(sampled),
            "sample_mode": cfg.sample_mode,
            "clusters": assignment.n_clusters,
            "noise": assignment.n_noise,
            "categories_total": len(names),
            "flagged": sum(1 for f in flags.values() if f),
            "flag_counts": _count_flags(flags.values()),
            "candidates": len(ranked),
            "deduped": deduped,
            "labeled": len(selected),
            "judge_calls": len(selected) * cfg.k_runs,
            "auto_added": added,
            "queued": queued,
            "total_active": len(final_active),
            "total_pending": len(store.cases("pending_review")),
            "coverage_ratio": coverage(final_active, names.values(), cfg.coverage_target)["covered_ratio"],
        }
    )
    store.add_pipeline_run(day, stats)
    return stats


def _count_flags(all_flags: Iterable[list[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for fl in all_flags:
        for f in fl:
            out[f] = out.get(f, 0) + 1
    return dict(sorted(out.items()))


def build_case(
    rec: LogRecord,
    label: dict[str, Any],
    confidence: float,
    route: str,
    category_id: int | None,
    category: str,
    flags: list[str],
    day: str,
    judge_name: str,
    run_qualities: list[int],
) -> dict[str, Any]:
    return {
        "id": case_id(rec.user_prompt),
        "status": "active" if route == "auto" else "pending_review",
        "category_id": category_id,
        "category": category,
        "difficulty": label["difficulty"],
        "polarity": polarity(label["quality"]),
        "label_source": "auto",
        "confidence": confidence,
        "input": {"system": rec.system_prompt, "prompt": rec.user_prompt},
        "observed": {"response": rec.response, "model": rec.model, "feature": rec.feature, "quality": label["quality"]},
        "expected": {"behavior": label["behavior"], "reference": label["reference"], "rubric": label["rubric"]},
        "assertions": {"must_contain": label["must_contain"], "must_not_contain": label["must_not_contain"]},
        "rationale": label["rationale"],
        "flags": flags,
        "source_log_ids": [rec.request_id],
        "first_seen": rec.timestamp,
        "created_day": day,
        "judge": judge_name,
        "judge_run_qualities": run_qualities,
        "notes": [],
    }


def process_inbox(
    store: Store,
    inbox: str | Path,
    judge: Judge | None = None,
    embedder: Embedder | None = None,
    cfg: PipelineConfig | None = None,
    redactor: Redactor | None = None,
) -> dict[str, Any]:
    """Ingest every new file in ``inbox`` and run the nightly pipeline for each day not yet processed."""
    totals = IngestStats()
    files = sorted(
        p
        for p in Path(inbox).iterdir()
        if p.is_file() and p.suffix in (".jsonl", ".json") and not p.name.startswith(("truth-", "manifest"))
    )
    for path in files:
        totals.merge(ingest_file(path, store, redactor))
    done = {r["day"] for r in store.pipeline_runs()}
    runs = {day: run_nightly(store, day, judge, embedder, cfg) for day in store.log_days() if day not in done}
    return {"ingest": totals.to_dict(), "runs": runs}


def seconds_until(now: datetime, at: str) -> float:
    """Seconds from ``now`` until the next ``HH:MM`` (same timezone as ``now``)."""
    hh, mm = (int(v) for v in at.split(":"))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"invalid time of day: {at}")
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()
