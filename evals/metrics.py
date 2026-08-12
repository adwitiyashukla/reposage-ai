from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass(slots=True)
class RetrievalMetrics:
    case_id: str = ""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    hit: bool = False
    first_relevant_rank: int | None = None
    retrieved: int = 0
    expected: int = 0
    matched_paths: list[str] = field(default_factory=list)
    missed_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "recall@k": round(self.recall_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "hit": self.hit,
            "first_relevant_rank": self.first_relevant_rank,
            "retrieved": self.retrieved,
            "expected": self.expected,
            "missed_paths": self.missed_paths,
        }


def _normalise(path: str) -> str:
    return path.strip().lstrip("./").replace("\\", "/").lower()


def _matches(retrieved: str, expected: str) -> bool:
    r, e = _normalise(retrieved), _normalise(expected)
    return r == e or r.endswith("/" + e) or e.endswith("/" + r)


def evaluate_retrieval(
    case_id: str, retrieved_paths: list[str], expected_paths: list[str], k: int | None = None
) -> RetrievalMetrics:
    if not expected_paths:
        return RetrievalMetrics(case_id=case_id, retrieved=len(retrieved_paths))

    ranked: list[str] = []
    for path in retrieved_paths:
        if path not in ranked:
            ranked.append(path)
    if k:
        ranked = ranked[:k]

    relevance = [
        1 if any(_matches(path, expected) for expected in expected_paths) else 0 for path in ranked
    ]
    matched = [
        expected for expected in expected_paths if any(_matches(path, expected) for path in ranked)
    ]
    missed = [expected for expected in expected_paths if expected not in matched]

    first_rank = next((i + 1 for i, rel in enumerate(relevance) if rel), None)
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected_paths), len(ranked) or 1)))

    return RetrievalMetrics(
        case_id=case_id,
        recall_at_k=len(matched) / len(expected_paths),
        precision_at_k=(sum(relevance) / len(ranked)) if ranked else 0.0,
        mrr=(1.0 / first_rank) if first_rank else 0.0,
        ndcg=(dcg / ideal) if ideal else 0.0,
        hit=bool(matched),
        first_relevant_rank=first_rank,
        retrieved=len(ranked),
        expected=len(expected_paths),
        matched_paths=matched,
        missed_paths=missed,
    )


def aggregate_retrieval(results: list[RetrievalMetrics]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "cases": len(results),
        "hit_rate": round(mean(1.0 if r.hit else 0.0 for r in results), 4),
        "recall@k": round(mean(r.recall_at_k for r in results), 4),
        "precision@k": round(mean(r.precision_at_k for r in results), 4),
        "mrr": round(mean(r.mrr for r in results), 4),
        "ndcg": round(mean(r.ndcg for r in results), 4),
        "mean_rank_of_first_hit": round(
            mean([r.first_relevant_rank for r in results if r.first_relevant_rank] or [0]), 2
        ),
    }


def citation_validity(citations: list[Any], indexed_paths: set[str]) -> dict[str, float]:
    if not citations:
        return {"citations": 0, "valid_rate": 0.0}
    valid = sum(1 for c in citations if getattr(c, "path", "") in indexed_paths)
    return {"citations": len(citations), "valid_rate": round(valid / len(citations), 4)}


def keyword_coverage(answer: str, required: list[str]) -> float:
    if not required:
        return 1.0
    lowered = answer.lower()
    return round(sum(1 for term in required if term.lower() in lowered) / len(required), 4)
