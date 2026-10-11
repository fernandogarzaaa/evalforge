# golden-dataset/v1

The export format written by `evalforge export` and read by `evalforge eval`. It is
runner-agnostic: a prompt-regression runner such as
[regresscheck](https://github.com/fernandogarzaaa/regresscheck) can consume it directly,
and evalforge has no dependency on any runner.

## Document (`.json`)

```json
{
  "format": "golden-dataset/v1",
  "name": "evalforge",
  "generated_by": "evalforge 0.1.0",
  "created_at": "2026-09-27T23:52:19.666000Z",
  "stats": {"cases": 233, "by_category": {"...": 5}, "by_difficulty": {"simple": 123}, "by_polarity": {"negative": 104}},
  "cases": [ /* Case objects */ ]
}
```

`.jsonl` exports contain one Case object per line and no envelope. `load_golden` reads
both flavours.

## Case

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable id: `tc-` + first 12 hex of SHA-1 of the whitespace-normalised, lower-cased prompt |
| `messages` | list of `{role, content}` | Chat input: optional `system` message, then the `user` message |
| `input` | string | The user prompt alone, for runners that want a flat string |
| `expected.behavior` | `answer` \| `refuse` \| `clarify` | What a well-behaved model should do |
| `expected.reference` | string \| null | Reference answer from source data, when one exists |
| `expected.rubric` | list of strings | Free-text grading criteria (not scored by the offline harness) |
| `assertions` | list of `{type, value}` | Machine-checkable expectations, see below |
| `tags` | list of strings | `[category, difficulty, polarity]` |
| `metadata` | object | Provenance, see below |

### Assertion types

| `type` | `value` | Passes when |
|---|---|---|
| `behavior` | `answer` / `refuse` / `clarify` | The detected behaviour of the response equals `value` |
| `contains` | string | The response contains `value` (case-insensitive) |
| `not_contains` | string | The response does not contain `value` (case-insensitive); used for hallucination traps and leaked system prompts |

`evalforge eval` scores each case as the mean of three parts (behaviour match, share of
`contains` satisfied, no `not_contains` violation); a case passes only when all three are
fully satisfied. Runners may ignore unknown assertion types, so new types can be added
without a version bump; changing the meaning of an existing field requires `v2`.

### Metadata

| Key | Meaning |
|---|---|
| `category` | Persistent category name (top terms), or null = unclustered |
| `difficulty` | `simple` / `moderate` / `hard` / `adversarial` |
| `polarity` | `positive` (high-quality interaction: keep doing this) / `negative` (fix this) |
| `label_source` | `auto` (judge agreement above threshold) or `human` (approved/edited in review) |
| `confidence` | Judge agreement across k runs, 0-1 |
| `first_seen` | ISO-8601 UTC timestamp of the source interaction |
| `source_log_ids` | Request ids of the production interactions the case came from |
| `flags` | Anomaly signals that selected it (e.g. `novel`, `retried`, `negative_feedback`) |
| `feature` | Product surface that produced the source interaction |
| `observed_quality` | Judge quality score of the observed production response |
