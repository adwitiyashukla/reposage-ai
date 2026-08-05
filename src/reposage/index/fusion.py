"""Rank fusion.

Dense and lexical retrievers produce scores on incompatible scales: cosine
similarity is bounded in [-1, 1] and clusters tightly, BM25 is unbounded and
varies with corpus statistics. Normalising them into a weighted sum requires
per-corpus calibration that silently rots as the corpus changes.

Reciprocal Rank Fusion sidesteps the problem by discarding magnitudes and using
only ordinal position:

    score(d) = sum over retrievers of  weight / (k + rank(d))

It has no tunable per-corpus parameters, is robust to outlier scores, and
rewards documents that several independent retrievers agree on. ``k`` (60 by
convention) damps the influence of the very top ranks so a single retriever
cannot dominate the fused list.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class FusedResult:
    """A fused document with the per-retriever ranks that produced it."""

    doc_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def retrievers(self) -> tuple[str, ...]:
        return tuple(sorted(self.ranks))

    @property
    def agreement(self) -> int:
        """How many independent retrievers surfaced this document."""
        return len(self.ranks)


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[FusedResult]:
    """Fuse several ranked id lists into one.

    Args:
        rankings: retriever name -> ranked document ids, best first.
        k: smoothing constant; larger values flatten the contribution curve.
        weights: optional per-retriever multipliers, defaulting to 1.0.
        top_n: truncate the fused output.
    """
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
    # Ties broken by agreement, then by best single rank, then by id for determinism.
    fused.sort(
        key=lambda r: (-r.score, -r.agreement, min(r.ranks.values(), default=10**6), r.doc_id)
    )
    return fused[:top_n] if top_n else fused
