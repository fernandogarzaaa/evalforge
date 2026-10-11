# evalforge architecture

This document is for engineers changing evalforge. The README covers usage; this covers
how the pieces fit, why they are shaped this way, and where to extend them.

## Data flow

```
                ┌──────────── ingest (evalforge/ingest.py) ────────────┐
 JSONL ─┐       │                                                      │
 OTLP  ─┼─ adapters.py ─► schema.validate ─► redact.Redactor ─► fingerprint dedup ─► store.Store (SQLite)
 SQLite ┘       │   (reshape only)   (per-reason error counts)   (Luhn, rules)   (re-delivery)  │
                └──────────────────────────────────────────────────────┘
                                                                        │ one calendar day at a time
 nightly (evalforge/pipeline.py: run_nightly / process_inbox)           ▼
   1. sample          sampling.py   uniform | stratified | signal (Efraimidis-Spirakis weighted)
   2. embed           embed.py      hashing (default) | fastembed | openai, optional exact-text cache
   3. cluster         cluster.py    SVD -> HDBSCAN; -1 = noise = "novel"
   4. categorise      cluster.py    map run-local clusters to persistent categories by centroid cosine
   5. flag            anomaly.py    novel, low_confidence, retry, length/latency outlier, filtered, thumbs-down, edited
   6. prioritise      pipeline.py   flag_score + coverage_boost (under-covered categories first)
   7. budget cut      pipeline.py   top --budget candidates per night (judge cost cap)
   8. near-dup filter dedup.py      cosine >= --dedup vs existing cases and earlier picks
   9. label x k       judge.py      MockJudge (offline) | LLMJudge (OpenAI-compatible), grounded in KB facts
  10. route           labeling.py   agreement across k runs = confidence; >= threshold -> active, else review
  11. persist         store.py      cases, categories, flags, embeddings (float32 base64), run history

 curation   curation.py (+ thin dashboard.py)   review queue, approve/edit/reject, reviewer kappa
 export     export.py    golden-dataset/v1 JSON or JSONL
 eval       evalrun.py   run cases against mock:v1 | mock:v2 | openai[:model]; deterministic scoring
 regression regression.py new failures / passes, score moves, category pass-rate shifts
```

## Module map

| Module | Responsibility | Tested |
|---|---|---|
| `schema.py` | `LogRecord`, validation, timestamp parsing, content fingerprint | yes |
| `adapters.py` | JSONL, OTLP/JSON GenAI spans, SQLite tables -> raw dicts; format detection | yes |
| `ingest.py` | validate -> redact -> dedup -> store; per-file SHA-256 skip | yes |
| `redact.py` | Regex rules + Luhn check; JSON config to add/disable rules; recall helper | yes |
| `sampling.py` | Three sampling modes, largest-remainder stratum allocation | yes |
| `embed.py` | Hashing / fastembed / OpenAI embedders, cache, cosine | yes (fastembed path not in the suite; exercised by the benchmark) |
| `cluster.py` | HDBSCAN clustering, naming, representatives, persistent categories | yes |
| `anomaly.py` | Robust-z outliers, retry detection, flag computation and scoring | yes |
| `heuristics.py` | Difficulty, expected behaviour, behaviour detection, injection/vagueness | yes |
| `dedup.py` | Near-duplicate filter, coverage stats, coverage boost | yes |
| `judge.py` | `Label`, `MockJudge`, `LLMJudge`, KB retrieval, reply parsing | yes (LLM path via fake transport only) |
| `labeling.py` | k-run aggregation, confidence, polarity, quality buckets | yes |
| `llm.py` | Minimal OpenAI-compatible chat client with retries | yes (MockTransport) |
| `pipeline.py` | Nightly orchestration and inbox processing | yes |
| `store.py` | SQLite schema and access layer | yes |
| `curation.py` | Review queue actions and reviewer statistics | yes |
| `dashboard.py` | Streamlit view over curation/store | **no** |
| `export.py` | golden-dataset/v1 writer/reader | yes |
| `evalrun.py` | Eval harness, mock models, scoring | yes |
| `regression.py` | Report comparison | yes |
| `synth.py`, `kb.py` | Synthetic "Acme Cloud" traffic generator and its knowledge base | via benchmark/tests |
| `metrics.py` | kappa, agreement, precision/recall, purity, freshness | yes |
| `cli.py` | argparse front end | end-to-end tests (marked `slow`) |

## Key decisions

### PII is redacted before anything is stored
Nothing downstream (store, embeddings, judge prompts, exports) ever sees raw PII, so no
later stage needs its own scrubbing. Card numbers must pass Luhn, which stops order
numbers and IDs from being masked. Redaction is regex-based only; there is no
named-entity pass (see Known gaps in the README).

### SQLite
Zero infrastructure, transactional, one file to back up, comfortably millions of rows.
The access layer in `store.py` is small and SQL-plain on purpose so it can be ported to
Postgres or ClickHouse when a single file stops being enough. Vectors are stored as
float32 little-endian base64 (exact, ~5x smaller than JSON floats).

### Sampling spends the budget where failures are
Signal-boosted sampling weights each interaction by its anomaly flags and samples
without replacement with Efraimidis-Spirakis keys (`u^(1/w)`). On the benchmark it lifts
the edge-case share from 24.4% of traffic to 33.6% of the sample and the bad-response
share from 15.0% to 21.1%, while still covering 98.7% of feature/model strata.
Stratified sampling is available when exact stratum coverage matters more (100%).

### HDBSCAN plus persistent categories
k-means would need the number of intents up front and forces every point into a cluster.
HDBSCAN finds an unknown number of dense groups and labels the rest noise, and noise is
the novelty signal. Run-local cluster ids are mapped onto persistent categories by
centroid similarity so coverage and growth are tracked across nights.

With the default hashing embedder, clusters are purer than they look from ARI alone
(86.3% purity, ARI 0.303) but intents fragment into more categories (68 for 18 true
intents) and only 30.7% of truly novel prompts land in noise. With fastembed
(bge-small) purity rises to 92.8%, ARI to 0.365, categories consolidate to 35, and 93.2%
of novel prompts land in noise.

### Hashing embedder is the default anyway
Embeddings are persisted and compared across nights (case dedup, category centroids), so
the default must give the same vector for the same text on every machine forever, with
no download. The hashing embedder does; model-based embedders can change with a model
upgrade. fastembed is the recommended upgrade when a ~70 MB one-time download is fine:
it is free, offline and deterministic for a pinned model.

### Labels carry an error bar
Each candidate is labeled k = 3 times (temperature > 0 for a real LLM, seeded noise for
the mock). Agreement is the confidence. Unanimous labels are auto-accepted; contested
ones go to the review queue, least confident first. On the benchmark, auto-accepted
labels have 95.3% quality-bucket accuracy versus 68.9% for the routed ones (hashing),
which is the evidence that agreement is a usable confidence signal. Judge run-to-run
Cohen's kappa is 0.875.

### The judge checks facts against source data
For factual questions the judge receives retrieved knowledge-base entries, including
known-wrong claims ("traps"). In production the KB is your docs/help-centre index.
Unparseable LLM replies fall back to the heuristic label and say so in the rationale.

### Cost is bounded
Judge calls are the main cost. `--budget` caps candidates per night; `--k-runs`
multiplies it. Near-duplicate filtering happens before labeling so already-covered
prompts are never paid for: 91.3% (hashing) / 94.3% (fastembed) of candidates were
dropped this way over 14 nights, leaving 834 / 603 judge calls in total.

### Offline-first
The mock judge and mock models make the pipeline, its 189 tests and CI deterministic and
free. The mock judge's accuracy is reported as the accuracy of a heuristic on synthetic
data, never as LLM accuracy.

### Runner-agnostic export
`golden-dataset/v1` (see `DATASET_FORMAT.md`) is plain chat messages plus typed
assertions, consumable by the built-in harness, by regresscheck, or by any other
prompt-regression runner, with no code dependency in either direction.

## Benchmark summary (current mock:v2)

From `results/{hashing,fastembed}/report.json`, 14 nights x 600 interactions, seed 7:

| | hashing | fastembed |
|---|---|---|
| Redaction recall / false positives | 100% / 0 | 100% / 0 |
| Cluster purity / ARI | 86.3% / 0.303 | 92.8% / 0.365 |
| Novel prompts in noise | 30.7% | 93.2% |
| Retry detection P / R | 1.00 / 0.86 | 1.00 / 0.98 |
| Latency outliers P / R | 0.875 / 0.933 | 0.875 / 0.933 |
| Difficulty accuracy | 96.8% | 96.8% |
| Active cases / pending review | 233 / 45 | 164 / 37 |
| Judge calls | 834 | 603 |
| Pass rate mock:v1 -> mock:v2 | 95.7% -> 90.1% | 97.0% -> 89.0% |
| New failures (all in injected areas) | 13 (100%) | 13 (100%) |
| Injected-area cases caught | 68.4% | 81.2% |

## Extending

- **New log source:** add a reader in `adapters.py` that yields raw dicts; validation,
  redaction and dedup are applied for free in `ingest.py`.
- **New anomaly signal:** add it to `anomaly.compute_flags` and give it a weight in
  `flag_score` / `sampling.signal_weight`. Keep it cheap and explainable.
- **New judge:** implement the `Judge` protocol (`label(prompt, response, run_index,
  feedback, key) -> Label`).
- **New embedder:** implement `Embedder` (`name`, `embed(texts) -> np.ndarray`) and
  register it in `get_embedder`. Changing the embedder of an existing store invalidates
  stored vectors; start a new store or re-embed.
- **Bigger store:** port `store.Store` methods; nothing else touches SQL.
