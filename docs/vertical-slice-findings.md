# Phase 9a vertical-slice findings

## Status

The live run has not been recorded yet. The harness, scripted conversation, and deterministic fake-model test are implemented; a recorded finding needs `MEMORY_WEAVE_LIVE=1`, `ANTHROPIC_API_KEY`, the `live` and `local-models` optional dependencies, and a chosen Anthropic model ID.

## Run metadata

| Field | Value |
| --- | --- |
| Model ID | Not run |
| Prompt version | `v1` |
| Runs | 3 fresh SQLite stores when run |
| Script | `examples/vertical_slice.py` |

## Results to record

| Measure | Recorded result |
| --- | --- |
| Write attempts | Not measured |
| `invalid_subject` responses | Not measured |
| `entity_ambiguous` responses | Not measured |
| Evidence-not-supported responses | Not measured |
| Downgrades | Not measured |
| Active attributes for the same answer-style preference across three runs | Not measured |
| Evidence-quote match rate | Not measured |
| Preference-applicable turns that searched memory | Not measured |
| Topically adjacent ordinary turns that searched memory | Not measured |
| Topically adjacent ordinary turns with a non-empty memory result | Not measured |

## How to read the first run

One stable attribute set across all completed runs means the serving model produces usable current-fact keys for the same preference. The report records each run's active attributes and the number of runs that contributed one, so a missing or superseded record cannot make stability look better than it is. A low evidence-quote match rate or many downgrades means the tool contract is too difficult for the serving model to satisfy reliably. The evidence rate uses the new claim's audit provenance, including when a claim reinforces an existing record.

The search-turn rates count turns with at least one search, not raw tool calls. Raw call counts are included separately to expose repeated searches within one turn. The four ordinary turns are deliberately topically adjacent to stored Python, deployment-checklist, and answer-format memories, without needing those memories to answer the question. A non-empty ordinary-turn result is the false-retrieval signal. If the active answer-style attribute set varies across runs, add a system-proposed attribute vocabulary to LLD section 18 before Phase 10 begins.

## Run command

```bash
MEMORY_WEAVE_LIVE=1 ANTHROPIC_API_KEY=... uv run --extra live --extra local-models python examples/vertical_slice.py --model <anthropic-model-id>
```

Copy the JSON report into the results table, replace `Not run` in the metadata, and add a short interpretation of the observed numbers before considering the phase's live finding complete. A failed run remains in `failed_runs` while the report retains every completed run.

When using `--database-dir`, the harness creates a fresh `attempt-*` child directory for every invocation. The report includes that directory as `artifact_dir`, which preserves partial-run stores and makes a failed live run inspectable.
