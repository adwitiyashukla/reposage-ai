from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from reposage.logging_setup import get_logger
from reposage.models import ScoredChunk
from reposage.observability import current_tracer

if TYPE_CHECKING:
    from reposage.llm.client import LLMClient

log = get_logger(__name__)

_WINDOW = 20
_PREVIEW_CHARS = 900

_SYSTEM = """You are a precision retrieval reranker for source code.
You score how well each candidate snippet answers a developer's question about a codebase.

Scoring rubric (0-10):
  9-10  Directly contains the answer: the exact implementation, definition or configuration asked about.
  6-8   Strongly relevant: calls, defines or configures the thing asked about, or is required to understand it.
  3-5   Topically related but does not answer the question.
  1-2   Same domain, wrong subject.
  0     Irrelevant.

Judge substance, not surface keyword overlap. A snippet that merely mentions the
term scores low; a snippet that implements the behaviour scores high.
Return a score for every candidate id you were given."""


class _Score(BaseModel):
    id: int
    score: float = Field(ge=0.0, le=10.0)


class _ScoreList(BaseModel):
    scores: list[_Score] = Field(default_factory=list)


class LLMReranker:
    def __init__(self, client: LLMClient, window: int = _WINDOW) -> None:
        self.client = client
        self.window = window

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if len(candidates) <= 1:
            return candidates[:top_k]

        tracer = current_tracer()
        with tracer.span("retrieval.rerank", candidates=len(candidates), top_k=top_k) as span:
            windows = [
                candidates[i : i + self.window] for i in range(0, len(candidates), self.window)
            ]
            results = await asyncio.gather(
                *(
                    self._score_window(query, w, offset=i * self.window)
                    for i, w in enumerate(windows)
                ),
                return_exceptions=True,
            )

            scored_any = False
            for window, result in zip(windows, results, strict=True):
                if isinstance(result, BaseException):
                    log.warning("rerank.window_failed", error=str(result)[:160])
                    continue
                scored_any = True
                for candidate, score in zip(window, result, strict=True):
                    candidate.rerank_score = score

            if not scored_any:
                span.set(status="degraded", reason="all windows failed")
                return candidates[:top_k]

            for candidate in candidates:
                if candidate.rerank_score is None:
                    candidate.rerank_score = min(5.0, candidate.score * 100)

            ordered = sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))
            span.set(
                windows=len(windows),
                top_score=round(ordered[0].rerank_score or 0.0, 2),
                promoted=sum(
                    1 for i, c in enumerate(ordered[:top_k]) if candidates.index(c) >= top_k
                ),
            )
            return ordered[:top_k]

    async def _score_window(
        self, query: str, window: list[ScoredChunk], offset: int
    ) -> list[float]:
        from reposage.llm.client import ModelTier

        blocks = []
        for local_id, candidate in enumerate(window):
            chunk = candidate.chunk
            preview = chunk.content[:_PREVIEW_CHARS]
            if len(chunk.content) > _PREVIEW_CHARS:
                preview += "\n... (truncated)"
            blocks.append(
                f"--- candidate id={local_id} ---\n"
                f"path: {chunk.location}\n"
                f"symbol: {chunk.qualified_name} ({chunk.kind.value})\n"
                f"{preview}"
            )

        prompt = (
            f"Developer question:\n{query}\n\n"
            f"Candidates ({len(window)}):\n\n" + "\n\n".join(blocks) + "\n\n"
            f"Score every candidate id from 0 to {len(window) - 1}."
        )
        parsed = await self.client.structured(
            prompt,
            _ScoreList,
            tier=ModelTier.FAST,
            system=_SYSTEM,
            temperature=0.0,
            max_output_tokens=1024,
        )
        scores = [0.0] * len(window)
        for item in parsed.scores:
            if 0 <= item.id < len(window):
                scores[item.id] = float(item.score)
        _ = offset
        return scores
