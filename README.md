# evalforge

[![CI](https://github.com/fernandogarzaaa/evalforge/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/evalforge/actions/workflows/ci.yml)

**Turn production LLM logs into a growing, labeled, deduplicated evaluation dataset.**

evalforge reads the logs your LLM features already write (JSONL, OpenTelemetry GenAI
spans, or a database table), redacts PII, and every night picks the interactions most
worth turning into test cases: novel requests, retries, thumbs-down, latency and length
outliers, content-filter hits, prompt injections. It clusters traffic into stable
categories, labels candidates with an LLM judge run *k* times (agreement = confidence),
routes contested labels to a human review queue, drops near-duplicates of cases you
already have, and exports a runner-agnostic `golden-dataset/v1` file. A built-in eval
harness and regression tracker then tell you which model release broke which category.

Everything runs offline by default (hashing embedder, mock judge, mock models), so the
full pipeline, the tests and CI cost nothing and need no keys. Swap in a real
OpenAI-compatible endpoint when you want real labels.

> Status: v0.1.0, not published to PyPI. Read [Known gaps](#known-gaps) before relying on it.

---

## Contents

- [Setup](#setup)
- [Five-minute tour](#five-minute-tour)
- [CLI reference](#cli-reference)
- [Log format](#log-format)
- [Architecture and decisions](#architecture-and-decisions)
- [Results](#results)
- [Golden dataset format](#golden-dataset-format)
- [Testing and quality gates](#testing-and-quality-gates)
- [Known gaps](#known-gaps)

## Setup

Python 3.10+.

```bash
git clone https://github.com/fernandogarzaaa/evalforge.git
cd evalforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + pytest/ruff/mypy/testteeth
pip install -e ".[semantic]"       # optional: local ONNX sentence embeddings (fastembed)
pip install -e ".[dashboard]"      # optional: Streamlit review dashboard
```

Core dependencies are deliberately small: `numpy`, `scikit-learn` (>= 1.3 for
`sklearn.cluster.HDBSCAN`) and `httpx`.

### Using a real LLM (optional)

All network access goes through one OpenAI-compatible client configured from the
environment. Any server that speaks `POST {base}/chat/completions` works: OpenAI, Groq,
OpenRouter, Together, vLLM, llama.cpp `llama-server`, Ollama (`/v1`).

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=...            # never committed; read from env only
export MODEL=gpt-4o-mini             # judge / eval model
export EMBED_MODEL=text-embedding-3-small   # only for --embedder openai
evalforge run --inbox logs/ --llm openai
```

## Five-minute tour

```bash
# One command, fully offline: synthesize 3 days of traffic, run the nightly pipeline,
# export golden.json, evaluate mock:v1 and mock:v2 against it, print the regression.
evalforge demo --dir demo-out

# The same thing step by step:
evalforge synth --out data/synthetic --days 14 --per-day 600      # deterministic corpus + hidden truth
evalforge run --inbox data/synthetic --db evalforge.db             # ingest + process every new day
evalforge stats --db evalforge.db                                  # size, coverage, freshness, growth
evalforge review list --db evalforge.db                            # contested labels, least confident first
evalforge review approve --db evalforge.db --case-id tc-0123abcd4567
evalforge export --db evalforge.db --out golden.json
evalforge eval --dataset golden.json --model mock:v1 --out v1.json
evalforge eval --dataset golden.json --model mock:v2 --out v2.json
evalforge compare v1.json v2.json --fail-on-regression             # non-zero exit if v2 regressed
```

## CLI reference

| Command | What it does | Key options |
|---|---|---|
| `synth` | Write the deterministic synthetic log corpus plus truth files | `--out --days --per-day --seed` |
| `ingest` | Ingest JSONL / OTLP-JSON / SQLite logs into the store (validate, redact, dedup) | `paths... --format --redact-config --force` |
| `nightly` | Run the pipeline for one day already in the store | `--day YYYY-MM-DD` + pipeline options |
| `run` | Ingest an inbox directory and process every day not yet processed | `--inbox` + pipeline options |
| `schedule` | Run `run` every night at a UTC time (simple in-process loop) | `--at 02:00 --once` |
| `export` | Export active cases as `golden-dataset/v1` JSON (`.json`) or JSONL (`.jsonl`) | `--out --name --include-pending` |
| `eval` | Run a golden dataset against a model and score it | `--dataset --model mock:v1\|mock:v2\|openai[:model] --out --db` |
| `compare` | Regression report between two eval reports | `--score-threshold --category-threshold --fail-on-regression` |
| `review` | Work the human review queue | `list\|approve\|edit\|reject\|stats --case-id --edits --note --reviewer` |
| `stats` | Dataset size, per-category coverage, freshness, growth | `--target` |
| `demo` | End-to-end offline demo on synthetic logs | `--dir --days --per-day` |

Pipeline options shared by `nightly`, `run` and `schedule`:

| Option | Default | Meaning |
|---|---|---|
| `--db` | `evalforge.db` | SQLite store |
| `--llm` | `mock` | Judge backend: `mock` or `openai` |
| `--embedder` | `hashing` | `hashing`, `fastembed` or `openai` |
| `--sample-size` | 300 | Interactions sampled per night |
| `--sample-mode` | `signal` | `uniform`, `stratified` or `signal` |
| `--budget` | 60 | Max candidates sent to the judge per night (cost cap) |
| `--k-runs` | 3 | Judge runs per candidate; agreement between runs is the confidence |
| `--confidence` | 0.85 | Auto-accept threshold; below it a case goes to review |
| `--dedup` | 0.92 | Cosine threshold for near-duplicates |
| `--seed` | 0 | Sampling seed |

Redaction can be extended or narrowed with `--redact-config rules.json` (extra regex
rules, disabled built-ins). The dashboard runs with
`EVALFORGE_DB=evalforge.db streamlit run evalforge/dashboard.py`.

## Log format

Every source is normalised into one `LogRecord` (`evalforge/schema.py`):

| Field | Required | Notes |
|---|---|---|
| `request_id` | yes | |
| `timestamp` | yes | ISO-8601; naive values are treated as UTC |
| `feature`, `model` | yes | Which product surface and model produced it |
| `user_prompt` | yes | |
| `response`, `system_prompt` | no | Empty response = error/filtered |
| `latency_ms`, `prompt_tokens`, `completion_tokens` | no | Non-negative |
| `session_id` | no | Needed for retry detection |
| `feedback` | no | `"up"`, `"down"` or null |
| `edited`, `content_filtered` | no | Booleans |
| `confidence` | no | e.g. mean token probability, if your provider exposes it |
| `metadata` | no | Free-form |

Adapters (`evalforge/adapters.py`): one JSON object per line; OTLP/JSON trace exports
whose spans carry GenAI semantic-convention attributes; or any SQLite table with a
`column_map`. Invalid rows are counted per reason and skipped, never fatal. Re-delivered
rows with a fresh request id are dropped by content fingerprint; whole files already
ingested (same SHA-256) are skipped unless `--force`.

## Architecture and decisions

Full write-up with the pipeline diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
adapters -> schema validate -> PII redact -> fingerprint dedup -> SQLite store
                                                                     |
nightly:  sample -> embed -> HDBSCAN -> persistent categories -> anomaly flags
          -> prioritise (flags + coverage boost) -> budget cut -> near-dup filter
          -> judge x k -> agreement routing -> active cases | review queue
export:   golden-dataset/v1  ->  eval harness  ->  regression compare
```

The decisions that shape it, and why:

| Decision | Rationale |
|---|---|
| **Redact at ingest, before the store** | PII never reaches disk, embeddings, the judge or the exported dataset. Credit cards are Luhn-checked so order numbers are not masked. |
| **SQLite as the store** | Zero infrastructure, one file, transactional, fine for millions of rows. The access layer is small enough to port to Postgres/ClickHouse later. |
| **Signal-boosted sampling (Efraimidis-Spirakis)** | Uniform sampling spends the budget on easy traffic. Weighting by anomaly flags lifts edge cases from 24% of traffic to ~34% of the sample while stratification keeps every feature/model stratum represented. |
| **HDBSCAN, not k-means** | The number of interaction types is unknown, and HDBSCAN labels low-density points as noise; that noise *is* the "novel request" signal. SVD first for speed and stability. |
| **Persistent categories by centroid matching** | Cluster ids are meaningless across nights; mapping clusters onto stable categories makes coverage and growth trackable over time. |
| **Cheap, explainable signals** | Novelty, low confidence, rephrased retry, length/latency robust-z outliers (median/MAD), content filter, thumbs-down, user edits. Each flag is stored with the case so reviewers see *why* it was picked. |
| **Hashing embedder by default** | Stateless and identical on every machine with no model download. Embeddings are persisted and compared across nights, so stability beats semantic depth for the default. fastembed/OpenAI are drop-ins when better recall is worth it (see Results). |
| **Judge run k times; agreement = confidence** | A single LLM label has no error bar. Unanimous labels go straight in; contested ones go to a human. This is what makes auto-accepted labels more accurate than reviewed ones in the results below. |
| **Judge grounded in source data** | For factual questions the judge gets retrieved knowledge-base entries, including known-wrong claims, so it checks facts instead of vibes. |
| **Budget cap per night + coverage boost** | Judge calls are the cost driver; the budget bounds spend, and under-covered categories get priority so the set broadens instead of piling onto popular intents. |
| **Mock judge and mock models** | The whole system, its tests and CI run deterministically with no keys or cost. Mock accuracy is reported as exactly that, never as LLM accuracy. |
| **Runner-agnostic export** | `golden-dataset/v1` carries chat messages plus typed assertions, so any prompt-regression runner can consume it; evalforge depends on none. |
| **Deterministic scoring in the eval harness** | Behaviour (answer/refuse/clarify), required phrases and forbidden phrases (hallucination traps). Reproducible and free; LLM rubric grading is out of scope for the offline path. |
| **Synthetic corpus with hidden truth** | Public chat logs (LMSYS-Chat-1M, WildChat) are gated and carry no ground truth for quality, failure mode or PII. A seeded generator gives realistic mess *and* answers to score against. |

## Results

All numbers come from `scripts/benchmark.py` on the deterministic synthetic corpus:
14 simulated days x 600 interactions/day (seed 7; 7,749 records emitted, including
malformed and re-delivered rows), sample size 300/night, judge budget 60/night, k = 3.
Raw output: [`results/hashing/report.json`](results/hashing/report.json) and
[`results/fastembed/report.json`](results/fastembed/report.json). Reproduce with:

```bash
python scripts/benchmark.py --embedder hashing   --out results/hashing
python scripts/benchmark.py --embedder fastembed --out results/fastembed   # needs .[semantic]
```

**Important caveat:** the labels come from the *mock* judge (a heuristic with seeded
noise), and the models under test are `mock:v1` and a deliberately regressed `mock:v2`.
These numbers validate the plumbing and the routing logic against known truth. They are
not a measurement of how well GPT-4o or any real LLM labels production traffic.

### Ingest and redaction

| Metric | Value |
|---|---|
| Records read / invalid / re-delivered duplicates / stored | 7,760 / 138 / 238 / 7,384 |
| PII values planted (email, phone, card, IPv4) | 703 |
| Leaked after redaction | **0 (100% recall)** |
| False-positive placeholders | 0 |

### Pipeline stages (embedder-dependent)

| Metric | hashing (default) | fastembed (bge-small, ONNX) |
|---|---|---|
| Clustering: mean ARI vs true intent | 0.303 | **0.365** |
| Clustering: mean purity | 86.3% | **92.8%** |
| Clustering: mean noise share | 14.2% | 14.3% |
| Novel prompts landing in HDBSCAN noise | 30.7% | **93.2%** |
| Persistent categories (18 true intents) | 68 | 35 |
| Retry detection P / R | 1.00 / 0.86 | 1.00 / **0.98** |
| Latency outliers P / R | 0.875 / 0.933 | 0.875 / 0.933 |
| Difficulty estimate accuracy | 96.8% | 96.8% |
| Pipeline wall time (14 nights, CPU) | 4.7 s | 27.5 s (embedding cache warm) |

Sampling (identical for both embedders except the signal row):

| Mode | Edge-case share of sample | Bad-response share | Strata covered |
|---|---|---|---|
| Population | 24.4% | n/a | n/a |
| Uniform | 24.2% | 15.0% | 94.4% |
| Stratified | 23.9% | 14.7% | 100% |
| Signal-boosted (default) | **33.6%** (fastembed 33.8%) | **21.1%** (21.3%) | 98.7% (98.0%) |

### Labeling and routing (mock judge)

| Metric | hashing | fastembed |
|---|---|---|
| Candidates labeled | 278 | 201 |
| Auto-accepted / sent to review | 233 / 45 | 164 / 37 |
| Auto-accepted: behaviour / quality bucket / polarity correct | 99.1% / 95.3% / 96.6% | 98.8% / 93.3% / 94.5% |
| Routed to review: behaviour / bucket / polarity correct | 95.6% / 68.9% / 71.1% | 94.6% / 67.6% / 67.6% |
| Judge run-to-run pairwise agreement | 94.2% | 92.7% |
| Cohen's kappa, run 0 vs run 1 | 0.875 | 0.868 |

The routing does its job: labels the judge disagreed with itself on are much more often
wrong (bucket accuracy ~68% vs ~95%), and those are exactly the ones sent to humans.

### Dataset growth over 14 nights

| Metric | hashing | fastembed |
|---|---|---|
| Candidates considered / dropped as near-duplicates | 4,200 / 3,836 (91.3%) | 4,200 / 3,959 (94.3%) |
| Judge calls | 834 | 603 |
| Active cases / pending review | 233 / 45 | 164 / 37 |
| Categories with cases / true intents covered | 50 / 18 of 18 | 32 / 18 of 18 |
| Categories meeting the coverage target (8) | 13.4% | 22.2% |
| Difficulty mix (simple/moderate/hard/adversarial) | 123 / 57 / 43 / 10 | 105 / 33 / 22 / 4 |
| Polarity (positive/negative) | 129 / 104 | 78 / 86 |
| Freshness (source within 30 days) | 100% | 100% |

New auto-accepted cases per night (hashing): 55, 50, 36, 11, 14, 16, 11, 7, 9, 6, 7, 5, 3, 3.
fastembed: 54, 44, 11, 10, 6, 9, 3, 5, 8, 5, 6, 2, 1, 0. Growth flattens because the
synthetic corpus is built from a finite set of templates (see Known gaps). The semantic
embedder recognises paraphrases as duplicates sooner, so it saturates faster and spends
28% fewer judge calls for full intent coverage.

### Regression tracking (mock:v1 -> mock:v2)

`mock:v2` is deliberately regressed: it hallucinates on SQL questions with known traps
and follows roughly half of prompt injections.

| Metric | hashing dataset | fastembed dataset |
|---|---|---|
| Cases | 233 | 164 |
| Pass rate v1 -> v2 | 95.7% -> 90.1% | 97.0% -> 89.0% |
| Mean score v1 -> v2 | 0.986 -> 0.949 | 0.990 -> 0.937 |
| New failures / new passes | 13 / 0 | 13 / 0 |
| New failures inside the injected regression areas | **100%** | **100%** |
| Injected-area cases (passing on v1) caught failing on v2 | 68.4% | 81.2% |
| Categories flagged with a pass-rate shift | 8 | 6 |
| `regressed` verdict | true | true |

Zero false alarms: every new failure is in an area where a regression was planted.

## Golden dataset format

`evalforge export` writes `golden-dataset/v1` (spec: [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md)).
A document is `{format, name, generated_by, created_at, stats, cases[]}`; each case is:

```json
{
  "id": "tc-09781f631c18",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "How do I upgrade to Starter? thx"}],
  "input": "How do I upgrade to Starter? thx",
  "expected": {"behavior": "answer", "reference": "Upgrade from Billing > Plan > Change plan; ...", "rubric": []},
  "assertions": [
    {"type": "behavior", "value": "answer"},
    {"type": "contains", "value": "prorated"},
    {"type": "not_contains", "value": "full price again"}
  ],
  "tags": ["starter / money / plan", "simple", "negative"],
  "metadata": {"category": "...", "difficulty": "simple", "polarity": "negative", "label_source": "auto",
               "confidence": 1.0, "first_seen": "2026-09-14T20:58:00Z", "source_log_ids": ["..."],
               "flags": ["low_confidence", "latency_outlier", "negative_feedback"]}
}
```

The format is runner-agnostic. It is designed to feed a prompt-regression CI runner such
as the sibling project [regresscheck](https://github.com/fernandogarzaaa/regresscheck)
directly, but evalforge has no dependency on it, and the built-in `evalforge eval` /
`compare` commands consume the same file. A sample export from the benchmark is at
[`results/hashing/golden.json`](results/hashing/golden.json).

## Testing and quality gates

```bash
ruff check . && ruff format --check .
mypy                       # strict on evalforge/ (disallow_untyped_defs)
pytest -q                  # 189 tests, all offline, ~1 min
testteeth run              # mutation testing (configured in pyproject.toml)
```

CI (`.github/workflows/ci.yml`) runs lint, format, mypy, the full test suite and a
wheel/sdist build on Python 3.10 and 3.12 for every push and PR.

### Mutation scores (testteeth 0.2.0)

Mutation testing measures whether the tests actually fail when the code is broken.
Modules graded so far (killed / total mutants):

| Module | Score | Killed / total | Scope of the run |
|---|---|---|---|
| `store.py` | **86.8%** | 46 / 53 | all operators; `test_ingest.py` + `test_pipeline_store.py` (excluding the end-to-end test) |
| `anomaly.py` | **81.7%** | 67 / 82 | all operators; `test_signals_judge.py` |
| `judge.py` | 81.0% (partial) | 64 / 79 | comparison, return-value, arithmetic, boolean, remove-call only; the 73 `constant` mutants (thresholds and weights) did not fit the run budget and are **ungraded** |
| `llm.py` | 61.3% | 38 / 62 | all operators; `test_signals_judge.py` |
| `embed.py` | 60.6% | 40 / 66 | all operators; embedder unit tests only. 11 mutants are uncovered, mostly the optional `FastEmbedEmbedder` path, which no test exercises |
| `cluster.py` | 58.6% | 34 / 58 | all operators; cluster unit tests (excluding the end-to-end test) |
| `pipeline.py` | 54.2% (sampled) | 13 / 24 | evenly sampled 24 of 77 mutants (`--max-mutants 24`); full grading needs the slow end-to-end test per mutant |

Survivors cluster around tuning constants (thresholds, minimum sizes, epsilons) and
boundary comparisons (`>=` vs `>`), which the current tests do not pin. They are the
first place to add tests. The remaining modules in `[tool.testteeth] paths` (schema,
redact, sampling, dedup, labeling, heuristics, metrics, regression, export, evalrun,
curation, adapters, ingest) were graded during development; their scores are not
re-recorded here.

## Known gaps

Being explicit about what is not done or not proven:

- **No named-entity redaction.** Redaction is regex + checksum (email, phone, card,
  IPv4, API keys). Names, street addresses and free-text identifiers pass through. Add an
  NER pass (e.g. Presidio/spaCy) before using this on real customer data.
- **The real-LLM judge has only been tested against a fake transport.** `LLMJudge` and
  `OpenAICompatClient` are covered by unit tests using a fake client and
  `httpx.MockTransport` (parsing, retries, fallbacks). They have not been run against a
  live model, so real label accuracy, cost and latency are unmeasured.
- **The Streamlit dashboard is untested.** It is a thin view over `evalforge.curation`
  (which is tested), but the UI itself has no tests and has not been exercised end to end.
- **The synthetic template corpus saturates in about two weeks.** New cases per night
  fall from ~55 to ~0-3 by day 14 because the generator draws from a finite set of
  templates. Long-horizon growth curves on real traffic are unknown.
- **Docker is untested locally.** The `Dockerfile` has not been built or run in this
  environment.
- All headline accuracy numbers come from the mock judge on synthetic data, as stated in
  [Results](#results).
- `schedule` is a simple in-process loop; production deployments should use cron, a
  Kubernetes CronJob or an orchestrator calling `evalforge run`.

## License

MIT. See [LICENSE](LICENSE).
