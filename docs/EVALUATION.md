# Evaluation

An LLM system without an evaluation suite is a system whose quality nobody can
defend. This one measures three things independently, because they fail
independently.

---

## Running it

```bash
reposage index .
python -m evals.run_evals --repo reposage --output evals/REPORT.md
```

Useful flags:

```bash
--limit 5              # smoke test on the first five cases
--no-judge             # retrieval metrics only: free, deterministic, fast
--no-ablation          # skip the retrieval ablation
--json out.json        # machine-readable results
--min-recall 0.55      # exit non-zero below this (CI gate)
--min-pass-rate 0.60   # exit non-zero below this (CI gate)
```

---

## What is measured

### Retrieval quality

Deterministic, requires no judging, runs on every relevant pull request.

| Metric | Question it answers |
| --- | --- |
| **hit rate** | Did we retrieve *any* correct file? |
| **recall@k** | What fraction of the expected files did we retrieve? |
| **precision@k** | What fraction of what we retrieved was relevant? |
| **MRR** | How high up was the first correct file? |
| **nDCG** | Are correct files ranked near the top, not just present? |

Ground truth is at **file** granularity, not chunk granularity. Chunk boundaries
move whenever the chunker is tuned, and a dataset that must be rewritten after
every improvement is a dataset nobody maintains.

Path matching allows suffixes, so a case can name `auth/jwt.py` and match
`src/auth/jwt.py`.

### Answer quality

| Metric | How |
| --- | --- |
| **correctness** | LLM judge, 1-5, anchored rubric |
| **groundedness** | LLM judge, 1-5, anchored rubric |
| **completeness** | LLM judge, 1-5, anchored rubric |
| **citation validity** | Objective: fraction of citations resolving to indexed files |
| **fact coverage** | Objective: fraction of required facts present in the answer |

The judge design tries hard to be useful rather than to pretend it is ground
truth:

- **Separate axes.** A fluent answer can be complete and wrong; a correct answer
  can be poorly grounded. Collapsing these into one number hides exactly the
  failure you want to see.
- **An explicit rubric.** Every point on the 1-5 scale is defined, which
  measurably reduces judge variance compared with an unanchored "rate this".
- **Reasoning before scores.** Scores produced after written justification are
  better calibrated than scores produced before it.
- **Temperature zero, fast model.** Judging is classification. Determinism beats
  eloquence.

Judge scores are always reported *alongside* the objective metrics, never
instead of them.

### Operational cost

Latency, tokens and dollars per answer. An accuracy win that triples the bill is
a trade-off, not an improvement, and the report shows both.

---

## The ablation

Every configuration runs the same questions through the **production retrieval
path**, selected by the `mode` parameter on `HybridRetriever.retrieve`. Nothing
is reimplemented for the benchmark, so the numbers describe the shipping system.

| Configuration | What it isolates |
| --- | --- |
| dense only | The naive RAG baseline |
| lexical only | What BM25 alone contributes |
| hybrid + RRF | The value of fusing both retrievers |
| hybrid + rerank | The value of the listwise reranker |
| full pipeline | Reranking plus neighbour expansion |

This localises regressions. If recall drops after a chunker change, the ablation
shows whether the loss is in dense recall, lexical recall, or the reranker's
ordering.

---

## The dataset

`evals/datasets/golden.jsonl`, one JSON object per line:

```json
{
  "id": "hybrid-retrieval",
  "question": "How does retrieval combine semantic and keyword search?",
  "expected_paths": ["src/reposage/index/retriever.py", "src/reposage/index/fusion.py"],
  "expected_symbols": ["HybridRetriever", "reciprocal_rank_fusion"],
  "must_mention": ["reciprocal rank fusion", "BM25"],
  "reference_answer": "Dense and BM25 run in parallel, then fuse with RRF ...",
  "category": "retrieval",
  "difficulty": "medium"
}
```

| Field | Purpose |
| --- | --- |
| `expected_paths` | Ground truth for every retrieval metric |
| `must_mention` | Blunt but useful guard against fluent answers that omit the point |
| `reference_answer` | Given to the judge; improves calibration substantially |
| `category`, `difficulty` | Slice results to find *where* quality is weak |

The dataset targets RepoSage itself, so anyone who clones the repository can
reproduce a run without external setup. `tests/test_evals.py` asserts that every
path in the dataset still exists, so ground truth cannot silently rot as the code
moves.

### Adding cases

Aim for questions a new engineer would actually ask. Good cases have a clear
locus in the code, a checkable answer, and ground truth you would defend in
review. Cover the categories that already exist, and add hard cases deliberately:
a suite everything passes has stopped measuring anything.

---

## Reading a report

The generated `evals/REPORT.md` contains:

1. **Headline table.** Pass rate, judge scores, citation validity, cost.
2. **Ablation table and bar chart.** Attribution of retrieval quality.
3. **Per-case table.** Every case with its scores, so regressions are traceable
   to a specific question.
4. **Failures section.** For each weak answer: the judge's reasoning, any claims
   flagged as unsupported, and the expected files that were never retrieved.

That last section is the most useful part. "Expected files never retrieved" tells
you immediately whether a bad answer was a retrieval failure or a reasoning
failure, which is the first thing you need to know and the thing a single
aggregate score can never tell you.

---

## Known limitations

- **The judge shares a family with the system under test.** Same-family judges
  correlate with their own outputs. The objective metrics are the counterweight,
  which is why they are never dropped.
- **The dataset is small.** Sixteen cases catch large regressions reliably and
  small ones noisily. Treat single-point movements as noise.
- **Self-referential ground truth.** Evaluating on RepoSage's own code makes the
  suite reproducible for anyone, at the cost of some generality. Adding a second
  dataset against a well-known external repository is the obvious next step.
