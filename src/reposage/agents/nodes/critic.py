from __future__ import annotations

from reposage.agents.prompts import CRITIC_SYSTEM, CRITIC_USER, format_context_summary
from reposage.agents.state import AgentDeps, AgentState
from reposage.llm.client import ModelTier
from reposage.logging_setup import get_logger
from reposage.models import Critique
from reposage.observability import current_tracer

log = get_logger(__name__)


async def critic_node(state: AgentState, deps: AgentDeps) -> dict:
    tracer = current_tracer()
    draft = state.get("draft", "")
    if not draft or not deps.settings.enable_critic:
        return {"critique": Critique(verdict="accept", confidence=0.6)}

    with tracer.span("agent.critique", round=state.get("refinements", 0)) as span:
        retrieved = state.get("retrieved") or []
        prompt = CRITIC_USER.format(
            question=state["question"],
            context_summary=format_context_summary(retrieved),
            draft=draft[:12_000],
        )
        try:
            critique = await deps.client.structured(
                prompt,
                Critique,
                tier=ModelTier.FAST,
                system=CRITIC_SYSTEM,
                temperature=0.0,
                max_output_tokens=1536,
            )
        except Exception as exc:
            log.warning("critique.failed", error=str(exc)[:200])
            span.set(status="skipped", error=str(exc)[:160])
            return {
                "critique": Critique(verdict="accept", confidence=0.5),
                "errors": [f"critic unavailable, accepting draft: {exc}"],
            }

        exhausted = state.get("refinements", 0) >= deps.settings.max_refinements
        if critique.needs_refinement and exhausted:
            critique.verdict = "accept"
            tracer.log("critique.budget_exhausted", refinements=state.get("refinements", 0))

        span.set(
            verdict=critique.verdict,
            grounded=critique.grounded,
            complete=critique.complete,
            confidence=round(critique.confidence, 2),
            issues=len(critique.issues),
        )
        tracer.log("critique.done", verdict=critique.verdict, issues=critique.issues[:4])

        update: dict = {"critique": critique, "confidence": critique.confidence}
        if critique.needs_refinement:
            update["active_queries"] = critique.follow_up_queries[:5]
            update["refinements"] = state.get("refinements", 0) + 1
        return update
