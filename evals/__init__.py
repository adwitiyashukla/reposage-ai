"""Evaluation harness.

An LLM system without an evaluation suite is a system whose quality nobody can
defend. This package measures three things that matter independently:

* **Retrieval quality** - did we find the right code? Measured with recall@k,
  MRR and nDCG against a golden set of question-to-file mappings. These are
  deterministic and free: no model calls, so they run on every commit.
* **Answer quality** - is the answer correct, grounded and complete? Measured
  by an LLM judge against a rubric, plus objective citation validity.
* **Cost and latency** - what does an answer actually cost in tokens and
  seconds?

The harness also runs ablations, so improvements can be attributed to a specific
component rather than asserted.
"""

from evals.dataset import EvalCase, load_dataset
from evals.metrics import RetrievalMetrics, aggregate_retrieval, evaluate_retrieval
from evals.report import render_report

__all__ = [
    "EvalCase",
    "RetrievalMetrics",
    "aggregate_retrieval",
    "evaluate_retrieval",
    "load_dataset",
    "render_report",
]
