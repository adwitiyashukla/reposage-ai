"""Analysis node: turn retrieved code into a grounded, cited draft answer."""

from __future__ import annotations

from reposage.agents.prompts import ANALYST_SYSTEM, ANALYST_USER, format_context
from reposage.agents.state import AgentDeps, AgentState
from reposage.llm.client import ModelTier
from reposage.logging_setup import get_logger
from reposage.observability import current_tracer

log = get_logger(__name__)


async def analyst_node(state: AgentState, deps: AgentDeps) -> dict:
    """Write the answer from retrieved context only.

    Two generation modes share one prompt. Interactive callers stream tokens so
    the UI can render the answer as it is written; batch callers (CLI,
    evaluation) use the cached completion path, which makes reruns free and
    keeps eval scores reproducible.
    """
    tracer = current_tracer()
    retrieved = state.get("retrieved") or []

    plan = state.get("plan")
    if not retrieved:
        # Structural questions ("how many services are there?") are answerable
        # from the repository map alone, and the planner marks them as such.
        if plan is not None and not plan.needs_retrieval and state.get("repo_map"):
            return await _answer_from_map(state, deps)
        return {
            "draft": (
                "I could not retrieve any code relevant to this question. "
                "The repository may not contain this functionality, or the index "
                "may need to be rebuilt."
            ),
            "confidence": 0.0,
        }

    with tracer.span("agent.analyse", chunks=len(retrieved)) as span:
        plan_block = ""
        if plan and plan.sub_questions:
            bullets = "\n".join(f"- {sq.question}" for sq in plan.sub_questions)
            plan_block = f"Sub-questions to cover:\n{bullets}\n\n"

        context = format_context(retrieved)
        prompt = ANALYST_USER.format(
            repo_name=state.get("repo_name", "the repository"),
            question=state["question"],
            plan_block=plan_block,
            num_chunks=len(retrieved),
            num_files=len({c.chunk.path for c in retrieved}),
            context=context,
        )

        if state.get("stream_tokens"):
            pieces: list[str] = []
            async for token in deps.client.stream(
                prompt,
                tier=ModelTier.DEEP,
                system=ANALYST_SYSTEM,
                temperature=0.15,
                max_output_tokens=4096,
            ):
                pieces.append(token)
                tracer.token(token)
            draft = "".join(pieces)
        else:
            response = await deps.client.complete(
                prompt,
                tier=ModelTier.DEEP,
                system=ANALYST_SYSTEM,
                temperature=0.15,
                max_output_tokens=4096,
            )
            draft = response.text

        draft = draft.strip()
        span.set(chars=len(draft), context_chars=len(context))
        if not draft:
            return {
                "draft": "The model returned an empty response. Try rephrasing the question.",
                "errors": ["analyst returned empty output"],
                "confidence": 0.0,
            }
        return {"draft": draft}


async def _answer_from_map(state: AgentState, deps: AgentDeps) -> dict:
    """Answer a structural question directly from the repository map."""
    response = await deps.client.complete(
        f"Repository: {state.get('repo_name')}\n\n{state.get('repo_map', '')[:20_000]}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer using only the repository map above. Be specific and concise. "
        "If the map does not contain enough information, say so.",
        tier=ModelTier.FAST,
        system=ANALYST_SYSTEM,
        temperature=0.1,
        max_output_tokens=1536,
    )
    current_tracer().log("analyse.from_repo_map")
    return {"draft": response.text.strip()}
