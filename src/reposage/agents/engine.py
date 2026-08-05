"""The public agent API.

:class:`CodebaseAgent` wraps graph construction, tracing and result assembly so
callers write three lines rather than thirty. It exposes two shapes:

* :meth:`ask` - await a complete :class:`AgentAnswer`. Used by the CLI, the
  evaluation harness and the batch API.
* :meth:`astream` - async-iterate structured events while the run is in flight.
  Used by the web UI to render the agent's reasoning as it happens.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from reposage.agents.graph import build_graph, describe_graph
from reposage.agents.state import AgentDeps, initial_state
from reposage.config import Settings, get_settings
from reposage.index.retriever import HybridRetriever
from reposage.index.store import RepoIndex
from reposage.llm.client import LLMClient, get_client
from reposage.logging_setup import get_logger
from reposage.models import AgentAnswer, Citation
from reposage.observability import Tracer, use_tracer

log = get_logger(__name__)

_DRAIN_INTERVAL = 0.05


class CodebaseAgent:
    """Answers questions about one indexed repository."""

    def __init__(
        self,
        index: RepoIndex,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or get_client(self.settings)
        self.index = index
        self.retriever = HybridRetriever(index, self.client, self.settings)
        self.deps = AgentDeps(client=self.client, retriever=self.retriever, settings=self.settings)
        self.graph = build_graph(self.deps)

    # ------------------------------------------------------------------ ask
    async def ask(
        self,
        question: str,
        *,
        stream_tokens: bool = False,
        tracer: Tracer | None = None,
    ) -> AgentAnswer:
        """Run the full graph and return a complete answer."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        owns_tracer = tracer is None
        tracer = tracer or Tracer()
        started = time.perf_counter()

        async def _execute() -> dict[str, Any]:
            with tracer.span("agent.run", repo=self.index.metadata.name):
                return await self.graph.ainvoke(
                    initial_state(question, self.deps, stream_tokens=stream_tokens),
                    config={"recursion_limit": self.settings.max_agent_steps * 4},
                )

        if owns_tracer:
            with use_tracer(tracer):
                final = await _execute()
        else:
            final = await _execute()

        answer = AgentAnswer(
            question=question.strip(),
            answer=final.get("draft", "").strip(),
            citations=list(final.get("citations") or []),
            confidence=float(final.get("confidence", 0.0)),
            plan=final.get("plan"),
            critique=final.get("critique"),
            usage=tracer.usage,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            refinement_rounds=int(final.get("refinements", 0)),
            retrieved_paths=sorted({c.chunk.path for c in (final.get("retrieved") or [])}),
        )
        log.info(
            "agent.answered",
            repo=self.index.metadata.name,
            confidence=answer.confidence,
            citations=len(answer.citations),
            refinements=answer.refinement_rounds,
            seconds=answer.elapsed_seconds,
            cost_usd=answer.usage.cost_usd,
        )
        return answer

    # --------------------------------------------------------------- stream
    async def astream(self, question: str) -> AsyncIterator[dict[str, Any]]:
        """Yield trace events as the agent works, then a terminal result event.

        The run executes in a background task while this coroutine drains the
        tracer's event queue, so the caller sees planning, retrieval and tokens
        arrive live rather than after the fact.
        """
        tracer = Tracer()
        queue = tracer.subscribe()

        async def _run() -> AgentAnswer:
            with use_tracer(tracer):
                return await self.ask(question, stream_tokens=True, tracer=tracer)

        task = asyncio.create_task(_run())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_DRAIN_INTERVAL)
                    yield event.to_dict()
                except (asyncio.TimeoutError, TimeoutError):
                    if task.done() and queue.empty():
                        break
            answer = await task
            yield {
                "type": "final",
                "name": "answer",
                "attributes": {
                    "answer": answer.answer,
                    "citations": [c.model_dump() for c in answer.citations],
                    "confidence": answer.confidence,
                    "usage": answer.usage.model_dump(),
                    "elapsed_seconds": answer.elapsed_seconds,
                    "refinement_rounds": answer.refinement_rounds,
                    "retrieved_paths": answer.retrieved_paths,
                    "plan": answer.plan.model_dump() if answer.plan else None,
                    "critique": answer.critique.model_dump() if answer.critique else None,
                    "trace": tracer.waterfall(),
                    "summary": tracer.summary(),
                },
            }
        except Exception as exc:  # pragma: no cover - surfaced to the client
            log.exception("agent.stream_failed")
            if not task.done():
                task.cancel()
            yield {
                "type": "error",
                "name": "agent_failed",
                "attributes": {"error": str(exc)[:600], "error_type": type(exc).__name__},
            }
        finally:
            tracer.unsubscribe(queue)

    # ----------------------------------------------------------------- misc
    async def batch(self, questions: list[str], concurrency: int = 3) -> list[AgentAnswer]:
        """Answer several questions with bounded parallelism (used by evals)."""
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _one(question: str) -> AgentAnswer:
            async with semaphore:
                return await self.ask(question)

        return await asyncio.gather(*(_one(q) for q in questions))

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index.describe(),
            "graph": describe_graph(),
            "settings": self.settings.fingerprint(),
        }

    @staticmethod
    def format_citations(citations: list[Citation]) -> str:
        return "\n".join(
            f"- {c.label}" + (f"  ({c.symbol})" if c.symbol else "") for c in citations
        )
