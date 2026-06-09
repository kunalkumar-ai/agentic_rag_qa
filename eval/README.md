# Eval Harness

Measurement comes before optimization. Every change to the pipeline should be justified by a number on this harness.

## Files

| File | Purpose |
|---|---|
| `golden_dataset.jsonl` | Hand-curated questions + ground-truth. One JSON object per line. |
| `run_eval.py` | Runs the dataset through `run_agent`, computes metrics, writes timestamped results. |
| `results/` | Timestamped result JSONs. Commit them — these are the artifact. |

## Dataset schema

Each line in `golden_dataset.jsonl`:

```json
{
  "id": 1,
  "question": "...",
  "expected_companies": ["tesla"],
  "expected_years": ["2024"],
  "category": "single-company-single-year",
  "ground_truth_answer": "Short factual answer the system should produce.",
  "ground_truth_parent_ids": ["parent_id_a", "parent_id_b"],
  "notes": "Optional. Why this question, what it's testing."
}
```

### Categories to cover

- `single-company-single-year`
- `multi-company-single-year`
- `single-company-multi-year`
- `multi-company-multi-year`
- `out-of-corpus` — e.g. asking about Honda. Expected behavior: refuse / state corpus scope.
- `ambiguous` — references that need disambiguation.
- `numeric-comparison` — requires exact numbers across years/companies.
- `adversarial` — hostile phrasing, trick questions, conflicting sources.

### Writing a question

For each question you need:

- **`question`** — what you're asking
- **`category`** + **`difficulty`** — for bucketed reporting
- **`expected_companies`** / **`expected_years`** — which filings the agent should route to. Empty lists for out-of-corpus questions.
- **`should_refuse`** — `true` if the right behavior is to say "not in corpus"
- **`ground_truth_answer`** — one-paragraph reference answer (used by future LLM-judge correctness; not used by current metrics)
- **`must_mention`** — list of specific strings (numbers, entity names, key phrases) the correct answer must contain. This is the sharp signal.

You don't need to label chunk IDs. All retrieval-quality signal comes from routing + faithfulness + fact coverage, computed automatically from the agent trace.

## Running

```bash
# Run full dataset
python3 -m eval.run_eval

# Smoke test (first 3 questions)
python3 -m eval.run_eval --limit 3

# Custom dataset path
python3 -m eval.run_eval --dataset eval/golden_dataset.jsonl
```

Output: a timestamped JSON in `eval/results/` containing per-question records and an aggregate summary.

## Metrics

Computed automatically from the agent trace (no chunk labeling needed):

- **routing** — agent called `retrieve` with the expected companies and years.
- **fact_coverage** — fraction of `must_mention` strings found in the answer.
- **refusal** — for `should_refuse: true` items, the answer explicitly states the limitation.
- **faithfulness** — every citation in the answer points to a (company, year) that was actually retrieved. Hallucinated citations are flagged.
- **latency_ms_p50 / p95** — end-to-end agent latency.
- **iterations_used / max_iterations_hit / errors** — agent health.

Stubbed (TODO):

- **correctness** — LLM-as-judge against `ground_truth_answer`.
- **cost_usd / input_tokens / output_tokens** — needs token instrumentation in `agent.py`.

## Discipline

Every PR / commit that changes the pipeline:

1. Run `python3 -m eval.run_eval`.
2. Commit the new results JSON alongside the code change.
3. Add one line to the commit message: `eval: recall 0.62 → 0.71, p95 4200ms → 3800ms`.

If a number gets worse, justify it. If a number improves, you have a data point for your resume.
