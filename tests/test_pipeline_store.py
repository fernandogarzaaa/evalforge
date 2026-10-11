from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest

from evalforge.cli import main
from evalforge.cluster import (
    UNCLUSTERED,
    assign_categories,
    cluster_embeddings,
    name_cluster,
    representative,
    top_terms,
)
from evalforge.curation import apply_edits, review, review_queue, reviewer_stats, similar_cases
from evalforge.embed import (
    CachedEmbedder,
    HashingEmbedder,
    OpenAIEmbedder,
    cosine_matrix,
    get_embedder,
    normalise_text,
)
from evalforge.pipeline import PipelineConfig, build_case, case_id, process_inbox, run_nightly, seconds_until
from evalforge.store import Store, decode_vector, encode_vector
from evalforge.synth import SynthConfig, generate_day, load_truth, write_corpus
from tests.conftest import rec

# ---------------------------------------------------------------- embeddings


def test_hashing_embedder_is_stable_and_ignores_pii_and_chitchat() -> None:
    e = HashingEmbedder()
    a = e.embed(["How do I reset my password? [EMAIL]"])
    b = e.embed(["Hi team, how do I reset my password? thanks"])
    assert e.embed([]).shape == (0, 2048)
    assert cosine_matrix(a, b)[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)
    assert normalise_text("Mail [EMAIL] NOW") == "mail   now"
    assert e.name == "hashing-2048"


def test_cosine_matrix_edges() -> None:
    assert cosine_matrix(np.zeros((0, 3)), np.ones((2, 3))).shape == (0, 2)
    m = cosine_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[3.0, 0.0], [1.0, 1.0]]))
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(0.7071, abs=1e-4) and m[1, 0] == 0.0


def test_cached_embedder(tmp_path: Path) -> None:
    class Counting:
        name = "c"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def embed(self, texts: list[str]) -> np.ndarray:
            self.seen.extend(texts)
            return np.asarray([[len(t), 1.0] for t in texts], dtype=np.float32).reshape(len(texts), 2)

    inner = Counting()
    path = str(tmp_path / "c.npz")
    c = CachedEmbedder(inner, path)
    out = c.embed(["aa", "b", "aa"])
    assert out.tolist() == [[2, 1], [1, 1], [2, 1]] and inner.seen == ["aa", "b"]
    c.embed(["b"])
    assert inner.seen == ["aa", "b"] and c.name == "c"
    again = CachedEmbedder(Counting(), path)
    assert again.embed(["aa"]).tolist() == [[2, 1]] and again.inner.seen == []  # type: ignore[attr-defined]
    assert again.embed([]).shape == (0, 2)


def test_openai_embedder_batches_and_normalises() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"data": data})

    e = OpenAIEmbedder("http://x/v1/", "k", "emb", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = e.embed([f"t{i}" for i in range(130)])
    assert out.shape == (130, 2) and out[0].tolist() == pytest.approx([0.6, 0.8])
    assert [len(c["input"]) for c in calls] == [128, 2] and calls[0]["model"] == "emb"
    assert e.name == "openai:emb"


def test_get_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALFORGE_EMBEDDER", raising=False)
    assert isinstance(get_embedder(), HashingEmbedder)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    assert isinstance(get_embedder("openai"), OpenAIEmbedder)
    with pytest.raises(ValueError):
        get_embedder("word2vec")


# ---------------------------------------------------------------- clustering

PROMPTS = (
    [
        "How do I reset my password?",
        "reset my password please",
        "password reset link not working",
        "I forgot my password, reset it",
        "how to reset password for my account",
        "reset password help",
    ]
    + [
        "How do I get a refund?",
        "refund my last invoice",
        "I want a refund for my invoice",
        "can I get a refund on my invoice",
        "refund invoice please",
        "refund for the invoice",
    ]
    + ["Write a haiku about the moon"]
)


def test_cluster_and_assign_categories() -> None:
    e = HashingEmbedder()
    x = e.embed(PROMPTS)
    found = cluster_embeddings(x, min_cluster_size=3)
    assert len(set(found[:6])) == 1 and len(set(found[6:12])) == 1 and found[0] != found[6] and found[0] != -1
    labels = np.array([0] * 6 + [1] * 6 + [-1])
    first = assign_categories(PROMPTS, x, labels, [])
    assert first.n_clusters == 2 and first.n_noise == 1
    assert all(u["id"] is None for u in first.updates) and first.category_ids == [None] * 13
    names = sorted(u["name"] for u in first.updates)
    assert any("refund" in n for n in names) and any("password" in n for n in names)
    existing = [
        {"id": i + 1, "name": u["name"], "centroid": u["centroid"], "size": u["size"]}
        for i, u in enumerate(first.updates)
    ]
    second = assign_categories(PROMPTS, x, labels, existing)
    assert sorted(u["id"] for u in second.updates) == [1, 2]
    assert all(u["size"] == 12 for u in second.updates)
    assert second.category_ids[-1] is None
    # a noise point that strongly matches an existing category joins it
    dup_noise = assign_categories(
        ["reset my password please"], e.embed(["reset my password please"]), np.array([-1]), existing
    )
    assert dup_noise.category_ids[0] in (1, 2)
    strict = assign_categories(PROMPTS, x, labels, existing, match_threshold=1.01)
    assert all(u["id"] is None for u in strict.updates)


def test_two_clusters_merging_into_one_category_accumulate() -> None:
    e = HashingEmbedder()
    x = e.embed(PROMPTS[:12])
    labels = np.array([0] * 6 + [1] * 6)
    existing = [{"id": 7, "name": "all", "centroid": x.mean(axis=0).tolist(), "size": 10}]
    out = assign_categories(PROMPTS[:12], x, labels, existing, match_threshold=0.0)
    assert [u["size"] for u in out.updates] == [16, 22]
    assert existing[0]["size"] == 10  # caller's list is not mutated


def test_cluster_small_inputs_and_naming() -> None:
    assert cluster_embeddings(np.ones((2, 4)), 5).tolist() == [-1, -1]
    assert top_terms(["Reset my password", "password reset link", "hi thanks"], k=2) == ["password", "reset"]
    assert name_cluster(["the and of"]) == "misc"
    assert name_cluster(["sql join orders", "join orders"]) == "join / orders / sql"
    x = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert representative(["a", "b", "c"], x) == "b"
    assert UNCLUSTERED == "unclustered"


# ---------------------------------------------------------------- store + curation


def test_vector_codec() -> None:
    v = [0.5, -1.25, 3.0]
    assert decode_vector(encode_vector(v)) == v


def make_case(cid: str, prompt: str, status: str = "pending_review", conf: float = 0.5) -> dict:
    return {
        "id": cid,
        "status": status,
        "category": "c",
        "difficulty": "simple",
        "polarity": "positive",
        "label_source": "auto",
        "confidence": conf,
        "input": {"system": "", "prompt": prompt},
        "observed": {"response": "r", "quality": 5},
        "expected": {"behavior": "answer", "reference": None, "rubric": []},
        "assertions": {"must_contain": [], "must_not_contain": []},
        "first_seen": "2026-09-14T00:00:00Z",
        "created_day": "2026-09-14",
    }


def test_store_round_trips(store: Store, tmp_path: Path) -> None:
    store.add_case(make_case("a", "reset password"), [1.0, 0.0])
    store.add_case(make_case("b", "refund", "active"), [0.0, 1.0])
    assert store.get_case("a")["input"]["prompt"] == "reset password"
    assert store.get_case("zzz") is None
    assert [c["id"] for c in store.cases("active")] == ["b"]
    assert [c["id"] for c in store.cases(["active", "pending_review"])] == ["a", "b"]
    assert len(store.cases()) == 2
    ids, embs = store.case_embeddings()
    assert ids == ["a", "b"] and embs == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(KeyError):
        store.update_case(make_case("nope", "x"))
    cid = store.upsert_category(None, "n", [1.0], 3, "d")
    assert store.upsert_category(cid, "n", [2.0], 5, "d") == cid
    assert store.categories() == [{"id": cid, "name": "n", "centroid": [2.0], "size": 5, "created_day": "d"}]
    rid = store.add_eval_run("n", "m", "t", {"pass_rate": 1})
    assert rid == 1 and store.eval_runs()[0]["report"] == {"pass_rate": 1}
    store.add_pipeline_run("2026-09-14", {"x": 1})
    assert store.pipeline_runs() == [{"day": "2026-09-14", "x": 1}]
    assert store.add_logs([rec(), rec()]) == 1 and store.count_logs() == 1
    assert store.log_days() == ["2026-09-14"] and store.logs_for_day("2026-09-14")[0].request_id == "r1"
    assert store.has_log("r1", "x") and store.has_log("zz", rec().fingerprint()) and not store.has_log("zz", "x")
    disk = Store(tmp_path / "d" / "e.db")
    disk.add_logs([rec()])
    disk.close()
    assert Store(tmp_path / "d" / "e.db").count_logs() == 1


def test_review_flow(store: Store) -> None:
    store.add_case(make_case("a", "reset password", conf=0.6), [1.0, 0.0])
    store.add_case(make_case("b", "refund invoice", conf=0.4), [0.0, 1.0])
    store.add_case(make_case("c", "reset my password now", "active"), [1.0, 0.1])
    assert [c["id"] for c in review_queue(store)] == ["b", "a"]
    approved = review(store, "a", "ana", "approve", note="looks right", at="2026-09-15T00:00:00Z")
    assert approved["status"] == "active" and approved["label_source"] == "human"
    assert approved["notes"] == [{"by": "ana", "note": "looks right"}]
    rejected = review(store, "b", "ana", "reject", at="2026-09-15T00:00:00Z")
    assert rejected["status"] == "rejected" and rejected["label_source"] == "auto"
    edited = review(
        store, "c", "ben", "edit", {"behavior": "refuse", "quality": 2, "must_contain": ["x"], "category": "new"}
    )
    assert edited["expected"]["behavior"] == "refuse" and edited["polarity"] == "negative"
    assert edited["observed"]["quality"] == 2 and edited["assertions"]["must_contain"] == ["x"]
    assert edited["category"] == "new" and store.get_case("c")["expected"]["behavior"] == "refuse"
    assert review_queue(store) == []
    with pytest.raises(ValueError):
        review(store, "a", "x", "maybe")
    with pytest.raises(KeyError):
        review(store, "missing", "x", "approve")
    with pytest.raises(ValueError, match="not editable"):
        apply_edits(make_case("z", "p"), {"id": "hack"})
    with pytest.raises(ValueError, match="1-5"):
        apply_edits(make_case("z", "p"), {"quality": 9})
    sims = similar_cases(store, "a", k=1)
    assert [s["id"] for s in sims] == ["c"] and sims[0]["similarity"] > 0.5
    with pytest.raises(KeyError):
        similar_cases(store, "missing")
    reviews = store.reviews()
    assert [r["decision"] for r in reviews] == ["approve", "reject", "edit"]
    assert json.loads(reviews[2]["labels"])["behavior"] == "refuse"


def test_similar_cases_empty(store: Store) -> None:
    store.add_case(make_case("a", "p"), [1.0])
    assert similar_cases(store, "a") == []


def test_reviewer_stats() -> None:
    reviews = [
        {"case_id": "1", "reviewer": "a", "decision": "approve", "created_at": "2026-09-15T01:00:00"},
        {"case_id": "1", "reviewer": "b", "decision": "edit", "created_at": "2026-09-15T02:00:00"},
        {"case_id": "2", "reviewer": "a", "decision": "reject", "created_at": "2026-09-16T01:00:00"},
        {"case_id": "2", "reviewer": "b", "decision": "approve", "created_at": "2026-09-16T01:00:00"},
        {"case_id": "3", "reviewer": "a", "decision": "reject", "created_at": None},
        {"case_id": "3", "reviewer": "b", "decision": "reject", "created_at": None},
    ]
    s = reviewer_stats(reviews)
    assert s["reviews"] == 6 and s["per_reviewer"] == {"a": 3, "b": 3}
    assert s["per_reviewer_per_day"]["a"] == {"2026-09-15": 1, "2026-09-16": 1, "": 1}
    assert s["kappa"] == {"a~b": pytest.approx(0.4)}
    assert reviewer_stats([])["kappa"] == {}


# ---------------------------------------------------------------- synthetic data + pipeline


def test_synth_is_deterministic_and_has_truth(tmp_path: Path) -> None:
    cfg = SynthConfig(days=1, per_day=200)
    a, ta = generate_day(cfg, 0)
    b, tb = generate_day(cfg, 0)
    assert a == b and ta == tb
    assert len(ta) >= 180
    m = write_corpus(tmp_path, cfg)
    assert m["days"] == ["2026-09-14"] and m["records"] == len(a)
    truth = load_truth(tmp_path)
    assert len(truth) == len(ta)
    intents = {t["intent"] for t in ta}
    assert {"prompt_injection", "ambiguous", "reset_password"} <= intents
    assert any(t["pii"] for t in ta) and any("retried" in t["edge"] for t in ta)
    assert (tmp_path / "otel-2026-09-14.json").exists() and (tmp_path / "logs-2026-09-14.jsonl").exists()


def test_case_id_and_build_case() -> None:
    assert case_id("Reset  Password") == case_id("reset password") and case_id("a") != case_id("b")
    assert case_id("x").startswith("tc-") and len(case_id("x")) == 15
    label = {
        "quality": 2,
        "behavior": "answer",
        "difficulty": "simple",
        "must_contain": ["a"],
        "must_not_contain": ["b"],
        "reference": "ref",
        "rubric": [],
        "rationale": "why",
    }
    c = build_case(rec(), label, 0.9, "auto", 3, "cat", ["retried"], "2026-09-14", "j", [2, 2, 3])
    assert c["status"] == "active" and c["polarity"] == "negative" and c["category_id"] == 3
    assert c["assertions"] == {"must_contain": ["a"], "must_not_contain": ["b"]}
    assert c["source_log_ids"] == ["r1"] and c["first_seen"] == "2026-09-14T10:00:00Z"
    assert c["judge_run_qualities"] == [2, 2, 3] and c["label_source"] == "auto"
    assert build_case(rec(), label, 0.5, "review", None, "u", [], "d", "j", [])["status"] == "pending_review"


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    write_corpus(tmp_path / "in", SynthConfig(days=2, per_day=250))
    store = Store(tmp_path / "e.db")
    out = process_inbox(store, tmp_path / "in", cfg=PipelineConfig(sample_size=150, budget=30))
    assert out["ingest"]["stored"] > 400 and out["ingest"]["invalid"] > 0 and out["ingest"]["duplicates"] > 0
    assert list(out["runs"]) == ["2026-09-14", "2026-09-15"]
    day1 = out["runs"]["2026-09-14"]
    assert day1["sampled"] == 150 and day1["labeled"] <= 30
    assert day1["judge_calls"] == day1["labeled"] * 3
    assert day1["auto_added"] + day1["queued"] == day1["labeled"]
    assert day1["total_active"] == day1["auto_added"]
    assert len(store.cases()) == day1["labeled"] + out["runs"]["2026-09-15"]["labeled"]
    ids = [c["id"] for c in store.cases()]
    assert len(ids) == len(set(ids))
    again = process_inbox(store, tmp_path / "in")
    assert again["runs"] == {} and again["ingest"]["read"] == 0
    empty = run_nightly(store, "2030-01-01")
    assert empty["logs"] == 0 and empty["sampled"] == 0


def test_seconds_until() -> None:
    now = datetime(2026, 9, 14, 1, 30, tzinfo=timezone.utc)
    assert seconds_until(now, "02:00") == 1800
    assert seconds_until(now, "01:30") == 86400
    assert seconds_until(now, "01:00") == 84600
    with pytest.raises(ValueError):
        seconds_until(now, "25:00")
    with pytest.raises(ValueError):
        seconds_until(now, "10:60")


# ---------------------------------------------------------------- CLI


@pytest.mark.slow
def test_cli_flow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, db = tmp_path / "logs", str(tmp_path / "e.db")
    assert main(["synth", "--out", str(data), "--days", "2", "--per-day", "200"]) == 0
    assert json.loads(capsys.readouterr().out)["days"] == ["2026-09-14", "2026-09-15"]
    assert main(["ingest", str(data / "logs-2026-09-14.jsonl"), "--db", db]) == 0
    assert json.loads(capsys.readouterr().out)["stored"] > 0
    assert main(["nightly", "--db", db, "--day", "2026-09-14", "--budget", "10"]) == 0
    capsys.readouterr()
    assert main(["run", "--db", db, "--inbox", str(data), "--budget", "10"]) == 0
    assert '"nights"' in capsys.readouterr().out
    golden = tmp_path / "g.json"
    assert main(["export", "--db", db, "--out", str(golden), "--include-pending"]) == 0
    assert "cases to" in capsys.readouterr().out
    r1, r2 = tmp_path / "r1.json", tmp_path / "r2.json"
    assert main(["eval", "--dataset", str(golden), "--model", "mock:v1", "--out", str(r1), "--db", db]) == 0
    assert main(["eval", "--dataset", str(golden), "--model", "mock:v2", "--out", str(r2)]) == 0
    assert len(Store(db).eval_runs()) == 1
    capsys.readouterr()
    assert main(["compare", str(r1), str(r1), "--fail-on-regression"]) == 0
    assert json.loads(capsys.readouterr().out)["regressed"] is False
    assert main(["review", "list", "--db", db]) == 0
    queue = json.loads(capsys.readouterr().out)
    if queue:
        assert main(["review", "approve", "--db", db, "--case-id", queue[0]["id"], "--reviewer", "t"]) == 0
        assert "-> active" in capsys.readouterr().out
    assert main(["review", "stats", "--db", db]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["review", "approve", "--db", db])
    assert main(["stats", "--db", db]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["logs"] > 0 and len(stats["growth"]) == 2
    with pytest.raises(SystemExit):
        main(["--version"])


@pytest.mark.slow
def test_cli_demo_and_compare_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--dir", str(tmp_path / "d"), "--days", "2", "--per-day", "250"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["active_cases"] > 0 and out["eval"]["mock:v1"] > out["eval"]["mock:v2"]
    assert out["regression"]["regressed"] is True
    prev = {"model": "a", "results": [{"id": "x", "passed": True, "score": 1.0}], "by_category": {}}
    curr = {"model": "b", "results": [{"id": "x", "passed": False, "score": 0.0}], "by_category": {}}
    (tmp_path / "p.json").write_text(json.dumps(prev))
    (tmp_path / "c.json").write_text(json.dumps(curr))
    assert main(["compare", str(tmp_path / "p.json"), str(tmp_path / "c.json"), "--fail-on-regression"]) == 1
    assert main(["compare", str(tmp_path / "p.json"), str(tmp_path / "c.json")]) == 0


def test_cli_llm_choice(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown --llm"):
        main(["nightly", "--db", str(tmp_path / "x.db"), "--day", "2026-09-14", "--llm", "magic"])
