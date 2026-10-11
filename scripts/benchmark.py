"""End-to-end benchmark on the deterministic synthetic corpus.

Runs the full nightly pipeline over N simulated days, then scores every stage against
the generator's hidden truth files. Everything printed in README/RESULTS comes from
the JSON this writes.

    python scripts/benchmark.py --embedder hashing  --out results/hashing
    python scripts/benchmark.py --embedder fastembed --out results/fastembed
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import adjusted_rand_score

from evalforge.anomaly import detect_retries, latency_outliers
from evalforge.cluster import cluster_embeddings
from evalforge.embed import CachedEmbedder, get_embedder
from evalforge.evalrun import MockModel, run_eval
from evalforge.export import golden_document, write_golden
from evalforge.heuristics import estimate_difficulty
from evalforge.labeling import quality_bucket
from evalforge.metrics import cohen_kappa, fleiss_style_agreement, freshness, precision_recall, purity
from evalforge.pipeline import PipelineConfig, process_inbox
from evalforge.regression import compare
from evalforge.sampling import MODES, sample, stratum_key
from evalforge.store import Store
from evalforge.synth import SynthConfig, load_truth, write_corpus

EDGE_KINDS = {"novel", "prompt_injection", "ambiguous", "retried", "latency_outlier", "filtered", "down"}


def pct(x: float) -> float:
    return round(100 * x, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--per-day", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--sample-size", type=int, default=300)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--out", default="results/hashing")
    ap.add_argument("--cache", default=None, help="optional .npz embedding cache")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    data = out / "corpus"
    t0 = time.time()
    manifest = write_corpus(data, SynthConfig(days=args.days, per_day=args.per_day, seed=args.seed))
    truth = load_truth(data)
    # Exact-text memoisation (persisted under --cache) so re-runs and the scoring passes below
    # do not re-embed the same prompts; it does not change any vector.
    embedder = CachedEmbedder(get_embedder(args.embedder), args.cache)
    store = Store(out / "evalforge.db")
    cfg = PipelineConfig(sample_size=args.sample_size, budget=args.budget, seed=0)
    result = process_inbox(store, data, embedder=embedder, cfg=cfg)
    elapsed = time.time() - t0
    report: dict[str, Any] = {
        "config": {**vars(args), "records_emitted": manifest["records"]},
        "pipeline_seconds": round(elapsed, 1),
        "ingest": result["ingest"],
    }

    # ---- redaction recall / false positives ------------------------------------------
    pii_total = pii_left = fp = 0
    placeholders = ("[EMAIL]", "[PHONE]", "[CARD]", "[IP]", "[API_KEY]")
    days = store.log_days()
    all_logs = [r for d in days for r in store.logs_for_day(d)]
    for r in all_logs:
        t = truth.get(r.request_id)
        if t is None:
            continue
        pii_total += len(t["pii"])
        pii_left += sum(v in r.user_prompt for v in t["pii"])
        if not t["pii"]:
            fp += sum(r.user_prompt.count(p) for p in placeholders)
    report["redaction"] = {
        "pii_values": pii_total,
        "leaked": pii_left,
        "recall_pct": pct(1 - pii_left / pii_total) if pii_total else None,
        "false_positive_placeholders": fp,
    }

    # ---- sampling comparison ----------------------------------------------------------
    samp: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in MODES}
    for d in days:
        logs = store.logs_for_day(d)
        retried = detect_retries(logs, embedder=embedder)
        slow = latency_outliers(logs)
        strata = {stratum_key(r) for r in logs}
        for m in MODES:
            s = sample(logs, args.sample_size, m, seed=0, retried=retried, slow=slow)
            edge = sum(bool(set(truth[r.request_id]["edge"]) & EDGE_KINDS) for r in s)
            bad = sum(truth[r.request_id]["quality"] <= 2 for r in s)
            samp[m]["edge_share"].append(edge / len(s))
            samp[m]["bad_share"].append(bad / len(s))
            samp[m]["strata_coverage"].append(len({stratum_key(r) for r in s}) / len(strata))
    base_edge = sum(bool(set(truth[r.request_id]["edge"]) & EDGE_KINDS) for r in all_logs) / len(all_logs)
    report["sampling"] = {
        "population_edge_share_pct": pct(base_edge),
        **{m: {k: pct(sum(v) / len(v)) for k, v in vals.items()} for m, vals in samp.items()},
    }

    # ---- clustering quality on each night's signal-boosted sample ---------------------
    aris, purities, noise_rates, novel_recall_n, novel_total, intents = [], [], [], 0, 0, set()
    for d in days:
        logs = store.logs_for_day(d)
        s = sample(
            logs,
            args.sample_size,
            "signal",
            seed=0,
            retried=detect_retries(logs, embedder=embedder),
            slow=latency_outliers(logs),
        )
        x = embedder.embed([r.user_prompt for r in s])  # served from the cache
        labels = cluster_embeddings(x, cfg.min_cluster_size, seed=0)
        tr = [truth[r.request_id]["intent"] for r in s]
        intents |= set(tr)
        aris.append(adjusted_rand_score(tr, labels))
        purities.append(purity(list(labels), tr))
        noise_rates.append(float((labels == -1).mean()))
        for lab, t in zip(labels, tr):
            if t == "novel":
                novel_total += 1
                novel_recall_n += int(lab == -1)
    report["clustering"] = {
        "mean_ari": round(sum(aris) / len(aris), 3),
        "mean_purity_pct": pct(sum(purities) / len(purities)),
        "mean_noise_pct": pct(sum(noise_rates) / len(noise_rates)),
        "novel_prompts_in_noise_pct": pct(novel_recall_n / novel_total) if novel_total else None,
        "true_intents": len(intents),
        "persistent_categories": len(store.categories()),
    }

    # ---- edge-case signals vs truth (all logs) ----------------------------------------
    retried_pred = set().union(*(detect_retries(store.logs_for_day(d), embedder=embedder) for d in days))
    retried_true = {k for k, v in truth.items() if "retried" in v["edge"] and k in {r.request_id for r in all_logs}}
    slow_pred = set().union(*(latency_outliers(store.logs_for_day(d)) for d in days))
    slow_true = {r.request_id for r in all_logs if "latency_outlier" in truth[r.request_id]["edge"]}
    report["signals"] = {
        "retry_detection": precision_recall(retried_pred, retried_true),
        "latency_outliers": precision_recall(slow_pred, slow_true),
    }

    # ---- difficulty estimator ---------------------------------------------------------
    pairs = [(truth[r.request_id]["difficulty"], estimate_difficulty(r.user_prompt)) for r in all_logs]
    per_class: dict[str, list[int]] = defaultdict(list)
    for t, p in pairs:
        per_class[t].append(int(t == p))
    report["difficulty"] = {
        "accuracy_pct": pct(sum(t == p for t, p in pairs) / len(pairs)),
        "per_class_recall_pct": {k: pct(sum(v) / len(v)) for k, v in sorted(per_class.items())},
        "truth_distribution": dict(Counter(t for t, _ in pairs)),
    }

    # ---- labels vs truth, split by route ----------------------------------------------
    cases = store.cases()
    rows = []
    for c in cases:
        t = truth[c["source_log_ids"][0]]
        q_obs = c["observed"]["quality"]
        rows.append(
            {
                "route": "auto" if c["status"] == "active" else "review",
                "behavior_ok": c["expected"]["behavior"] == t["expected_behavior"],
                "bucket_ok": quality_bucket(q_obs) == quality_bucket(t["quality"]),
                "polarity_ok": (q_obs >= 4) == (t["quality"] >= 4),
                "runs": [quality_bucket(q) for q in c["judge_run_qualities"]],
                "intent": t["intent"],
            }
        )

    def acc(rs: list[dict[str, Any]], key: str) -> float | None:
        return pct(sum(r[key] for r in rs) / len(rs)) if rs else None

    auto = [r for r in rows if r["route"] == "auto"]
    rev = [r for r in rows if r["route"] == "review"]
    runs = [r["runs"] for r in rows]
    report["labeling"] = {
        "judge": "mock-judge (heuristic + seeded noise), NOT an LLM",
        "cases_labeled": len(rows),
        "auto_routed": len(auto),
        "review_routed": len(rev),
        "auto": {k: acc(auto, k) for k in ("behavior_ok", "bucket_ok", "polarity_ok")},
        "review": {k: acc(rev, k) for k in ("behavior_ok", "bucket_ok", "polarity_ok")},
        "all": {k: acc(rows, k) for k in ("behavior_ok", "bucket_ok", "polarity_ok")},
        "run_pairwise_agreement_pct": pct(fleiss_style_agreement(runs)),
        "run0_vs_run1_kappa": round(cohen_kappa([r[0] for r in runs], [r[1] for r in runs]), 3),
    }

    # ---- dataset growth, dedup, coverage, freshness -----------------------------------
    pruns = store.pipeline_runs()
    active = store.cases("active")
    last_ts = max(r.timestamp for r in all_logs)
    report["dataset"] = {
        "nights": len(pruns),
        "candidates": sum(p["candidates"] for p in pruns),
        "deduped": sum(p["deduped"] for p in pruns),
        "dedup_pct": pct(sum(p["deduped"] for p in pruns) / max(1, sum(p["candidates"] for p in pruns))),
        "judge_calls": sum(p["judge_calls"] for p in pruns),
        "active_cases": len(active),
        "pending_review": len(store.cases("pending_review")),
        "categories_with_cases": len({c["category"] for c in active}),
        "true_intents_covered": len({truth[c["source_log_ids"][0]]["intent"] for c in active}),
        "by_difficulty": dict(Counter(c["difficulty"] for c in active)),
        "by_polarity": dict(Counter(c["polarity"] for c in active)),
        "final_coverage_ratio_pct": pct(pruns[-1]["coverage_ratio"]),
        "freshness_30d_pct": pct(freshness(active, last_ts)),
        "growth": [
            {
                "day": p["day"],
                "auto_added": p["auto_added"],
                "queued": p["queued"],
                "deduped": p["deduped"],
                "total_active": p["total_active"],
            }
            for p in pruns
        ],
    }

    # ---- eval runner + regression -----------------------------------------------------
    golden = golden_document(active, created_at=last_ts)["cases"]
    write_golden(active, out / "golden.json", created_at=last_ts)
    r1 = run_eval(golden, MockModel("v1"))
    r2 = run_eval(golden, MockModel("v2"))
    diff = compare(r1, r2)
    src = {c["id"]: truth[c["source_log_ids"][0]]["intent"] for c in active}
    regressed_intents = Counter(src[i] for i in diff["new_failures"])
    injected = {"sql_join", "sql_aggregate", "sql_window", "prompt_injection"}
    report["eval"] = {
        "cases": len(golden),
        "v1": {"pass_rate_pct": pct(r1["pass_rate"]), "mean_score": r1["mean_score"]},
        "v2": {"pass_rate_pct": pct(r2["pass_rate"]), "mean_score": r2["mean_score"]},
        "new_failures": len(diff["new_failures"]),
        "new_passes": len(diff["new_passes"]),
        "new_failures_by_true_intent": dict(regressed_intents),
        "new_failures_in_injected_regression_areas_pct": pct(
            sum(v for k, v in regressed_intents.items() if k in injected) / max(1, len(diff["new_failures"]))
        ),
        "injected_area_cases_caught_pct": pct(
            len(diff["new_failures"])
            / max(
                1,
                sum(
                    1
                    for i, intent in src.items()
                    if intent in injected and next(r for r in r1["results"] if r["id"] == i)["passed"]
                ),
            )
        ),
        "category_shifts": len(diff["category_shifts"]),
        "regressed": diff["regressed"],
    }
    (out / "eval_v1.json").write_text(json.dumps({k: v for k, v in r1.items() if k != "results"}, indent=2))
    (out / "eval_v2.json").write_text(json.dumps({k: v for k, v in r2.items() if k != "results"}, indent=2))
    (out / "regression.json").write_text(json.dumps(diff, indent=2))

    shutil.rmtree(data)  # the corpus is reproducible from the seed; keep the repo small
    (out / "evalforge.db").unlink()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
