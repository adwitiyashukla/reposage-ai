"""The hybrid retrieval pipeline.

For each query the retriever runs three stages:

1. **Recall.** Dense search (semantic) and BM25 (lexical) run over every query
   variant the planner produced. All query embeddings are computed in a single
   batched call, so N sub-queries cost one round trip rather than N.
2. **Fusion.** Ranked lists are combined with reciprocal rank fusion, then
   optionally boosted for paths the planner flagged as likely.
3. **Precision.** The fused shortlist is diversified so no single file can
   monopolise the context window, then reranked by an LLM that reads the query
   and candidate together.

Everything is instrumented. :class:`RetrievalDebug` records what each stage
contributed, which is what makes a bad answer diagnosable instead of mysterious.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from reposage.config import Settings, get_settings
from reposage.index.fusion import reciprocal_rank_fusion
from reposage.index.reranker import LLMReranker
from reposage.index.store import RepoIndex
from reposage.logging_setup import get_logger
from reposage.models import ScoredChunk
from reposage.observability import current_tracer

if TYPE_CHECKING:  # pragma: no cover
    from reposage.llm.client import LLMClient

log = get_logger(__name__)

DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 0.85
PATH_HINT_BOOST = 1.25
MAX_CHUNKS_PER_FILE = 4
NEIGHBOUR_RADIUS = 1


@dataclass
class RetrievalDebug:
    """Per-stage accounting for one retrieval call."""

    queries: list[str] = field(default_factory=list)
    dense_hits: int = 0
    lexical_hits: int = 0
    fused_candidates: int = 0
    after_diversity: int = 0
    reranked: bool = False
    neighbours_added: int = 0
    elapsed_ms: float = 0.0
    files_touched: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "dense_hits": self.dense_hits,
            "lexical_hits": self.lexical_hits,
            "fused_candidates": self.fused_candidates,
            "after_diversity": self.after_diversity,
            "reranked": self.reranked,
            "neighbours_added": self.neighbours_added,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "files_touched": self.files_touched[:20],
        }


class HybridRetriever:
    """Dense + lexical retrieval with fusion, diversification and reranking."""

    def __init__(
        self,
        index: RepoIndex,
        client: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.index = index
        self.client = client
        self.settings = settings or get_settings()
        self.reranker = LLMReranker(client)

    async def retrieve(
        self,
        queries: list[str] | str,
        *,
        top_k: int | None = None,
        path_hints: list[str] | None = None,
        rerank: bool | None = None,
        expand_neighbours: bool = True,
        mode: str = "hybrid",
    ) -> tuple[list[ScoredChunk], RetrievalDebug]:
        """Retrieve the most relevant chunks for one or more query phrasings.

        ``mode`` selects which retrievers participate: ``hybrid`` (default),
        ``dense`` or ``lexical``. The single-retriever modes exist so the
        evaluation harness can run ablations against the exact production code
        path rather than a reimplementation of it.
        """
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in queries if q and q.strip()][:8]
        if not queries:
            return [], RetrievalDebug()

        top_k = top_k or self.settings.top_k
        rerank = self.settings.enable_rerank if rerank is None else rerank
        candidate_k = self.settings.candidate_k
        debug = RetrievalDebug(queries=list(queries))
        tracer = current_tracer()

        with tracer.span("retrieval.hybrid", queries=len(queries), top_k=top_k) as span:
            started = asyncio.get_event_loop().time()

            dense_task = (
                asyncio.create_task(self._dense(queries, candidate_k))
                if mode in ("hybrid", "dense")
                else None
            )
            lexical = (
                await asyncio.to_thread(self._lexical, queries, candidate_k)
                if mode in ("hybrid", "lexical")
                else []
            )
            dense = await dense_task if dense_task is not None else []

            rankings: dict[str, list[str]] = {}
            for name, ranked in (("dense", dense), ("lexical", lexical)):
                for i, ids in enumerate(ranked):
                    if ids:
                        rankings[f"{name}[{i}]"] = ids
            debug.dense_hits = sum(len(r) for r in dense)
            debug.lexical_hits = sum(len(r) for r in lexical)

            if not rankings:
                span.set(result="empty")
                return [], debug

            weights = {
                key: (DENSE_WEIGHT if key.startswith("dense") else LEXICAL_WEIGHT)
                for key in rankings
            }
            fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k, weights=weights)
            debug.fused_candidates = len(fused)

            scored = self._to_scored(fused, path_hints)
            shortlist = self._diversify(scored, limit=max(top_k * 3, candidate_k))
            debug.after_diversity = len(shortlist)

            if rerank and len(shortlist) > top_k:
                selected = await self.reranker.rerank(queries[0], shortlist, top_k)
                debug.reranked = True
            else:
                selected = shortlist[:top_k]

            if expand_neighbours:
                before = len(selected)
                selected = self._add_neighbours(selected, limit=top_k + 4)
                debug.neighbours_added = len(selected) - before

            selected.sort(key=lambda c: -c.final_score)
            debug.files_touched = sorted({c.chunk.path for c in selected})
            debug.elapsed_ms = (asyncio.get_event_loop().time() - started) * 1000
            span.set(
                returned=len(selected),
                files=len(debug.files_touched),
                reranked=debug.reranked,
                mode=mode,
            )
        return selected, debug

    # ------------------------------------------------------------- retrievers
    async def _dense(self, queries: list[str], k: int) -> list[list[str]]:
        """Embed every query in one batch, then run a single batched matmul."""
        if len(self.index.vectors) == 0:
            return []
        vectors = await self.client.embed(queries, task_type="RETRIEVAL_QUERY")
        usable = [(i, v) for i, v in enumerate(vectors) if v]
        if not usable:
            return []
        matrix = np.asarray([v for _, v in usable], dtype=np.float32)
        results = self.index.vectors.search_many(matrix, k)
        ordered: list[list[str]] = [[] for _ in queries]
        for (original_index, _), hits in zip(usable, results, strict=True):
            ordered[original_index] = [doc_id for doc_id, _ in hits]
        return ordered

    def _lexical(self, queries: list[str], k: int) -> list[list[str]]:
        if len(self.index.lexical) == 0:
            return []
        return [[doc_id for doc_id, _ in self.index.lexical.search(q, k)] for q in queries]

    # -------------------------------------------------------------- refinement
    def _to_scored(self, fused: list[Any], path_hints: list[str] | None) -> list[ScoredChunk]:
        hints = [h.lower().strip("/") for h in (path_hints or []) if h]
        scored: list[ScoredChunk] = []
        for result in fused:
            chunk = self.index.get(result.doc_id)
            if chunk is None:
                continue
            score = result.score
            if hints and any(hint in chunk.path.lower() for hint in hints):
                score *= PATH_HINT_BOOST
            dense_ranks = [v for key, v in result.ranks.items() if key.startswith("dense")]
            lexical_ranks = [v for key, v in result.ranks.items() if key.startswith("lexical")]
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    dense_rank=min(dense_ranks) if dense_ranks else None,
                    lexical_rank=min(lexical_ranks) if lexical_ranks else None,
                    retrievers=result.retrievers,
                )
            )
        scored.sort(key=lambda c: -c.score)
        return scored

    @staticmethod
    def _diversify(scored: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        """Cap chunks per file so one large file cannot crowd out the answer.

        Overflow is not discarded: it is appended after the diversified head, so
        a genuinely single-file answer still gets the depth it needs.
        """
        per_file: dict[str, int] = defaultdict(int)
        primary: list[ScoredChunk] = []
        overflow: list[ScoredChunk] = []
        for candidate in scored:
            path = candidate.chunk.path
            if per_file[path] < MAX_CHUNKS_PER_FILE:
                per_file[path] += 1
                primary.append(candidate)
            else:
                overflow.append(candidate)
            if len(primary) >= limit:
                break
        return (primary + overflow)[:limit]

    def _add_neighbours(self, selected: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        """Pull in immediately adjacent chunks to repair split context.

        A retrieved function often depends on the constant or helper defined
        directly above it. Adding neighbours is far cheaper than a second
        retrieval round and fixes most "the answer was almost there" failures.
        """
        chosen = {c.chunk.chunk_id for c in selected}
        additions: list[ScoredChunk] = []
        for candidate in list(selected):
            if len(selected) + len(additions) >= limit:
                break
            siblings = self.index.chunks_for_path(candidate.chunk.path)
            try:
                position = next(
                    i for i, c in enumerate(siblings) if c.chunk_id == candidate.chunk.chunk_id
                )
            except StopIteration:  # pragma: no cover - defensive
                continue
            for offset in range(-NEIGHBOUR_RADIUS, NEIGHBOUR_RADIUS + 1):
                neighbour_index = position + offset
                if offset == 0 or not (0 <= neighbour_index < len(siblings)):
                    continue
                neighbour = siblings[neighbour_index]
                if neighbour.chunk_id in chosen:
                    continue
                chosen.add(neighbour.chunk_id)
                additions.append(
                    ScoredChunk(
                        chunk=neighbour,
                        score=candidate.score * 0.35,
                        rerank_score=(candidate.final_score * 0.35),
                        retrievers=("neighbour",),
                    )
                )
        return selected + additions

    # ------------------------------------------------------------------ misc
    def context_budget(
        self, chunks: list[ScoredChunk], max_tokens: int = 60_000
    ) -> list[ScoredChunk]:
        """Trim a chunk list to fit a token budget, highest scoring first."""
        kept: list[ScoredChunk] = []
        used = 0
        for candidate in chunks:
            cost = candidate.chunk.token_estimate
            if used + cost > max_tokens:
                continue
            kept.append(candidate)
            used += cost
        return kept
