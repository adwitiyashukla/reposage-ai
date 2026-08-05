"""Planning node: question plus repository map, out comes a retrieval plan."""

from __future__ import annotations

from reposage.agents.prompts import PLANNER_SYSTEM, PLANNER_USER
from reposage.agents.state import AgentDeps, AgentState
from reposage.llm.client import ModelTier
from reposage.logging_setup import get_logger
from reposage.models import QueryPlan, SubQuestion
from reposage.observability import current_tracer

log = get_logger(__name__)

_MAP_BUDGET = 14_000


async def planner_node(state: AgentState, deps: AgentDeps) -> dict:
    """Decompose the question into targeted search queries.

    Planning is the cheapest place to add retrieval quality: one fast-model call
    turns a vague question into several well-formed queries covering both
    identifier-style and descriptive vocabulary, which is what the hybrid
    retriever needs to do its job.

    A planning failure is never fatal. We fall back to using the raw question as
    a single query, which is exactly what a naive RAG system would have done.
    """
    tracer = current_tracer()
    question = state["question"]

    with tracer.span("agent.plan") as span:
        repo_map = state.get("repo_map", "")
        if len(repo_map) > _MAP_BUDGET:
            repo_map = repo_map[:_MAP_BUDGET] + "\n... (repository map truncated)"

        prompt = PLANNER_USER.format(
            repo_name=state.get("repo_name", "unknown"),
            languages=state.get("languages", "unknown"),
            repo_map=repo_map,
            question=question,
        )
        try:
            plan = await deps.client.structured(
                prompt,
                QueryPlan,
                tier=ModelTier.FAST,
                system=PLANNER_SYSTEM,
                temperature=0.15,
                max_output_tokens=2048,
            )
        except Exception as exc:
            log.warning("plan.failed", error=str(exc)[:200])
            span.set(status="fallback", error=str(exc)[:160])
            fallback = QueryPlan(
                intent="explain",
                restated_question=question,
                sub_questions=[SubQuestion(question=question, search_queries=[question])],
            )
            return {
                "plan": fallback,
                "active_queries": [question],
                "path_hints": [],
                "errors": [f"planner fell back to the raw question: {exc}"],
            }

        queries = plan.all_queries or [question]
        # Keyword hints are appended as their own query so exact identifiers get
        # a dedicated BM25 shot rather than being diluted inside a long phrase.
        if plan.keyword_hints:
            queries.append(" ".join(plan.keyword_hints[:6]))

        span.set(
            intent=plan.intent,
            sub_questions=len(plan.sub_questions),
            queries=len(queries),
            path_hints=len(plan.path_hints),
        )
        tracer.log(
            "plan.ready",
            intent=plan.intent,
            queries=queries[:6],
            paths=plan.path_hints[:5],
        )
        return {
            "plan": plan,
            "active_queries": queries[:8],
            "path_hints": plan.path_hints,
        }
