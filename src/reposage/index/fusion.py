from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class FusedResult:
    doc_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def retrievers(self) -> tuple[str, ...]:
        return tuple(sorted(self.ranks))

    @property
    def agreement(self) -> int:
        return len(self.ranks)


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[FusedResult]:
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    positions: dict[str, dict[str, int]] = defaultdict(dict)

    for retriever, ordered_ids in rankings.items():
        weight = weights.get(retriever, 1.0)
        for rank, doc_id in enumerate(ordered_ids):
            scores[doc_id] += weight / (k + rank + 1)
            positions[doc_id][retriever] = rank

    fused = [
        FusedResult(doc_id=doc_id, score=score, ranks=positions[doc_id])
        for doc_id, score in scores.items()
    ]
    fused.sort(
        key=lambda r: (-r.score, -r.agreement, min(r.ranks.values(), default=10**6), r.doc_id)
    )
    return fused[:top_n] if top_n else fused
