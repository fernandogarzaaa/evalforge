"""Command-line interface: ``evalforge <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalforge import __version__
from evalforge.curation import review, review_queue, reviewer_stats
from evalforge.dedup import coverage
from evalforge.embed import get_embedder
from evalforge.evalrun import get_model, run_eval
from evalforge.export import load_golden, write_golden
from evalforge.ingest import IngestStats, ingest_file
from evalforge.judge import Judge, LLMJudge, MockJudge
from evalforge.llm import OpenAICompatClient
from evalforge.metrics import freshness
from evalforge.pipeline import PipelineConfig, process_inbox, run_nightly, seconds_until
from evalforge.redact import Redactor
from evalforge.regression import compare
from evalforge.store import Store
from evalforge.synth import SynthConfig, write_corpus

DEFAULT_DB = os.environ.get("EVALFORGE_DB", "evalforge.db")


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def make_judge(kind: str) -> Judge:
    if kind == "mock":
        return MockJudge()
    if kind == "openai":
        return LLMJudge(OpenAICompatClient.from_env())
    raise SystemExit(f"unknown --llm {kind!r} (mock | openai)")


def _config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        sample_size=args.sample_size,
        sample_mode=args.sample_mode,
        budget=args.budget,
        k_runs=args.k_runs,
        confidence_threshold=args.confidence,
        dedup_threshold=args.dedup,
        seed=args.seed,
    )


def cmd_synth(args: argparse.Namespace) -> int:
    _print(write_corpus(args.out, SynthConfig(days=args.days, per_day=args.per_day, seed=args.seed)))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    store = Store(args.db)
    redactor = Redactor.from_config(args.redact_config)
    total = IngestStats()
    for path in args.paths:
        total.merge(ingest_file(path, store, redactor, args.format, force=args.force))
    _print(total.to_dict())
    return 0


def cmd_nightly(args: argparse.Namespace) -> int:
    store = Store(args.db)
    stats = run_nightly(store, args.day, make_judge(args.llm), get_embedder(args.embedder), _config(args))
    _print(stats)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store = Store(args.db)
    out = process_inbox(
        store,
        args.inbox,
        make_judge(args.llm),
        get_embedder(args.embedder),
        _config(args),
        Redactor.from_config(args.redact_config),
    )
    _print(
        {
            "ingest": out["ingest"],
            "nights": {d: {k: v for k, v in s.items() if k != "config"} for d, s in out["runs"].items()},
        }
    )
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    """Long-running nightly scheduler (used by docker-compose). ``--once`` runs immediately and exits."""
    while True:
        if not args.once:
            wait = seconds_until(datetime.now(timezone.utc), args.at)
            print(f"next run in {wait / 3600:.2f} h (at {args.at} UTC)", flush=True)
            time.sleep(wait)
        cmd_run(args)
        if args.once:
            return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = Store(args.db)
    statuses = ["active", "pending_review"] if args.include_pending else ["active"]
    cases = store.cases(statuses)
    write_golden(cases, args.out, name=args.name, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"wrote {len(cases)} cases to {args.out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cases = load_golden(args.dataset)
    report = run_eval(cases, get_model(args.model))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    if args.db:
        Store(args.db).add_eval_run(
            args.name or args.model, report["model"], datetime.now(timezone.utc).isoformat(timespec="seconds"), report
        )
    _print({k: v for k, v in report.items() if k != "results"})
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    prev = json.loads(Path(args.previous).read_text())
    curr = json.loads(Path(args.current).read_text())
    diff = compare(prev, curr, args.score_threshold, args.category_threshold)
    _print(diff)
    return 1 if args.fail_on_regression and diff["regressed"] else 0


def cmd_review(args: argparse.Namespace) -> int:
    store = Store(args.db)
    if args.action == "list":
        _print(
            [
                {
                    "id": c["id"],
                    "confidence": c["confidence"],
                    "category": c["category"],
                    "prompt": c["input"]["prompt"][:100],
                    "behavior": c["expected"]["behavior"],
                    "quality": c["observed"]["quality"],
                }
                for c in review_queue(store)[: args.limit]
            ]
        )
        return 0
    if args.action == "stats":
        _print(reviewer_stats(store.reviews()))
        return 0
    if not args.case_id:
        raise SystemExit("--case-id is required")
    edits = json.loads(args.edits) if args.edits else None
    case = review(store, args.case_id, args.reviewer, args.action, edits, args.note)
    print(f"{case['id']} -> {case['status']}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    store = Store(args.db)
    active = store.cases("active")
    days = store.log_days()
    _print(
        {
            "logs": store.count_logs(),
            "active_cases": len(active),
            "pending_review": len(store.cases("pending_review")),
            "rejected": len(store.cases("rejected")),
            "coverage": coverage(active, [c["name"] for c in store.categories()], args.target),
            "freshness_30d": round(freshness(active, days[-1] + "T23:59:59Z"), 4) if days else 0.0,
            "growth": [
                {
                    "day": r["day"],
                    "total_active": r.get("total_active"),
                    "auto_added": r.get("auto_added"),
                    "queued": r.get("queued"),
                }
                for r in store.pipeline_runs()
            ],
        }
    )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Synthetic logs -> pipeline -> golden export -> eval v1 vs v2 -> regression report, all offline."""
    root = Path(args.dir)
    data = root / "logs"
    write_corpus(data, SynthConfig(days=args.days, per_day=args.per_day, seed=args.seed))
    db = root / "evalforge.db"
    if db.exists():
        db.unlink()
    store = Store(db)
    out = process_inbox(store, data, MockJudge(), get_embedder("hashing"), PipelineConfig())
    golden = root / "golden.json"
    active = store.cases("active")
    write_golden(active, golden, created_at="demo")
    cases = load_golden(golden)
    r1, r2 = run_eval(cases, get_model("mock:v1")), run_eval(cases, get_model("mock:v2"))
    diff = compare(r1, r2)
    _print(
        {
            "logs_stored": out["ingest"]["stored"],
            "invalid": out["ingest"]["invalid"],
            "duplicates": out["ingest"]["duplicates"],
            "redactions": out["ingest"]["redactions"],
            "growth": [(d, s["total_active"]) for d, s in out["runs"].items()],
            "active_cases": len(active),
            "pending_review": len(store.cases("pending_review")),
            "golden": str(golden),
            "eval": {"mock:v1": r1["pass_rate"], "mock:v2": r2["pass_rate"]},
            "regression": {
                "new_failures": len(diff["new_failures"]),
                "regressed": diff["regressed"],
                "category_shifts": [s["category"] for s in diff["category_shifts"]],
            },
        }
    )
    return 0


def _pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--llm", default=os.environ.get("EVALFORGE_LLM", "mock"), help="mock (default) | openai")
    p.add_argument("--embedder", default=None, help="hashing (default) | fastembed | openai")
    p.add_argument("--sample-size", type=int, default=300)
    p.add_argument("--sample-mode", default="signal", choices=("uniform", "stratified", "signal"))
    p.add_argument("--budget", type=int, default=60, help="max candidates labeled per night")
    p.add_argument("--k-runs", type=int, default=3, help="judge runs per candidate (agreement = confidence)")
    p.add_argument("--confidence", type=float, default=0.85, help="auto-accept threshold")
    p.add_argument("--dedup", type=float, default=0.92, help="cosine threshold for near-duplicates")
    p.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalforge", description=__doc__)
    ap.add_argument("--version", action="version", version=f"evalforge {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write the deterministic synthetic log corpus")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--per-day", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("ingest", help="ingest JSONL / OTLP-JSON / SQLite logs")
    p.add_argument("paths", nargs="+")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--format", choices=("jsonl", "otel", "sqlite"))
    p.add_argument("--redact-config", help="JSON file with extra/disabled redaction rules")
    p.add_argument("--force", action="store_true", help="re-read files already ingested")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("nightly", help="run the pipeline for one day already in the store")
    _pipeline_args(p)
    p.add_argument("--day", required=True, help="YYYY-MM-DD")
    p.set_defaults(func=cmd_nightly)

    for name, func, helptext in (
        ("run", cmd_run, "ingest an inbox directory and process every new day"),
        ("schedule", cmd_schedule, "run `run` every night at --at (UTC)"),
    ):
        p = sub.add_parser(name, help=helptext)
        _pipeline_args(p)
        p.add_argument("--inbox", required=True)
        p.add_argument("--redact-config")
        if name == "schedule":
            p.add_argument("--at", default="02:00")
            p.add_argument("--once", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("export", help="export the dataset as golden-dataset JSON or JSONL")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--out", default="golden.json")
    p.add_argument("--name", default="evalforge")
    p.add_argument("--include-pending", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("eval", help="run a golden dataset against a model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="mock:v1", help="mock:v1 | mock:v2 | openai | openai:<model>")
    p.add_argument("--out")
    p.add_argument("--db", help="also record the run in this store")
    p.add_argument("--name")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("compare", help="regression report between two eval reports")
    p.add_argument("previous")
    p.add_argument("current")
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--category-threshold", type=float, default=0.05)
    p.add_argument("--fail-on-regression", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("review", help="work the human review queue")
    p.add_argument("action", choices=("list", "approve", "edit", "reject", "stats"))
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--case-id")
    p.add_argument("--reviewer", default=os.environ.get("USER", "reviewer"))
    p.add_argument("--edits", help='JSON, e.g. {"behavior": "refuse"}')
    p.add_argument("--note")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("stats", help="dataset size, coverage, freshness and growth")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--target", type=int, default=8)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("demo", help="end-to-end offline demo on synthetic logs")
    p.add_argument("--dir", default="demo-out")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--per-day", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, sqlite3.Error) as exc:
        # bad paths, unreadable/invalid JSON, unusable --db: one line, exit 2
        print(f"evalforge {args.command}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
