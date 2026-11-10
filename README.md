# Temporal Knowledge System

An offline, deterministic knowledge-retrieval system for answering credit-union website questions across multiple historical snapshots. Built for the interface.ai knowledge-system assessment.

## What it does

- **Temporal answers:** `answer(question, as_of=None)` selects the latest available snapshot on or before the requested date; otherwise it uses the newest snapshot.
- **Rate and tier disambiguation:** parses rate tables and distinguishes terms, balances, product variants, APR, APY and other qualifiers.
- **Conflict resolution:** prioritizes authoritative rate sheets and relevant product pages over stale promotional, calculator, and historical content; `conflicts()` reports detected contradictions.
- **Evidence and abstention:** returns supporting sources when available and avoids unsupported numerical claims.
- **Efficient retrieval:** uses a standard-library BM25-based text retrieval approach alongside structured extraction for rates, branch information and other facts.

## Submission entry point

The assessed implementation is in **`submission/`**. Its `solution.py` exposes `build()`, which returns an object implementing:

```python
system = build()
system.ingest(snapshot)
answer = system.answer(question, as_of=None)
conflicts = system.conflicts()
```

Supporting modules are `kb_parse.py`, `kb_rates.py`, `kb_text.py`, and `kb_conflicts.py`. See [submission/NOTES.md](submission/NOTES.md) for the implementation approach, precedence rules, and limitations.

## Run the public evaluation

Requires Python 3.9+ and no third-party packages or API keys.

1. Obtain the five original website snapshot ZIP files from the **private participant kit**. Place them in a local `snapshots_zip/` directory at the repository root. **Do not commit or publish them.**
2. From the repository root, run:

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

The harness extracts the snapshots locally and evaluates the submission. The supplied self-test fixtures in `harness/selftest/` use synthetic `example-cu.test` content.

## Verified public evaluation

The implementation was rerun locally on all five supplied snapshots and all 40 public questions on November 10, 2026.

| Metric | Result |
| --- | ---: |
| Overall public evaluation score | **94.57%** |
| Questions | 40 |
| Evaluation errors | 0 |
| Stale-trap hits | 0 |
| Paraphrase consistency | 100% |
| Repeat consistency | 100% |
| 95th-percentile answer time | 0.29 s |
| Snapshot ingestion time | 9.1 s |

These are **public evaluation results only**, not a claim about the hidden evaluation. The weakest public categories were list completeness and historical `as_of` questions (75% each).

## Repository structure

```text
submission/     Assessed solution and technical notes
harness/        Evaluation contract, scorer and synthetic self-tests
eval/           Public practice questions
results/        Saved evaluation reports
tools/          Additional paraphrase checks
starter/        Provided baseline for comparison
NOTES.md        Detailed design and precedence decisions
```

## Data and security

The real credit-union website snapshots are supplied for this assessment only and are intentionally **excluded** from this public repository. No credentials, paid APIs or live website calls are required. Never commit `snapshots/`, `snapshots_zip/`, `.env` or other secrets.
