"""Retrieval node: run the hybrid engine and merge results into state."""

from __future__ import annotations

from reposage.agents.state import AgentDeps, AgentState
from reposage.logging_setup import get_logger
from reposage.models import ScoredChunk
from reposage.observability import current_tracer

log = get_logger(__name__)


async def retriever_node(state: AgentState, deps: AgentDeps) -> dict:
    """Fetch context for the current query set.

    On a refinement round this runs again with the critic's follow-up queries and
    merges the new chunks with what we already had, keeping the better score for
    any chunk both rounds found. Merging rather than replacing matters: the
    refinement is meant to fill a gap, not to discard the context that was
    already working.
    """
    tracer = current_tracer()
    queries = state.get("active_queries") or [state["question"]]
    existing: list[ScoredChunk] = list(state.get("retrieved") or [])

    with tracer.span("agent.retrieve", round=state.get("refinements", 0)) as span:
        try:
            fresh, debug = await deps.retriever.retrieve(
                queries,
                top_k=deps.settings.top_k,
                path_hints=state.get("path_hints"),
            )
        except Exception as exc:
            log.warning("retrieve.failed", error=str(exc)[:200])
            span.set(status="error", error=str(exc)[:160])
            return {"errors": [f"retrieval failed: {exc}"], "retrieval_debug": []}

        merged: dict[str, ScoredChunk] = {c.chunk.chunk_id: c for c in existing}
        added = 0
        for candidate in fresh:
            key = candidate.chunk.chunk_id
            current = merged.get(key)
            if current is None:
                merged[key] = candidate
                added += 1
            elif candidate.final_score > current.final_score:
                merged[key] = candidate

        combined = sorted(merged.values(), key=lambda c: -c.final_score)
        # Keep the window bounded so repeated refinement cannot grow the prompt
        # without limit.
        ceiling = deps.settings.top_k * 2 + 6
        combined = combined[:ceiling]

        span.set(
            queries=len(queries),
            retrieved=len(fresh),
            new=added,
            total=len(combined),
            files=len({c.chunk.path for c in combined}),
        )
        tracer.log(
            "retrieve.done",
            files=sorted({c.chunk.path for c in combined})[:12],
            total=len(combined),
        )
        return {"retrieved": combined, "retrieval_debug": [debug.as_dict()]}
