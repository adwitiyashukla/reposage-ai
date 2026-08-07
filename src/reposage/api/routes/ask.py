"""Question answering, batch and streaming."""

from __future__ import annotations

import orjson
from fastapi import APIRouter, Header, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from reposage.agents.engine import CodebaseAgent
from reposage.api.demo import visitor_id
from reposage.api.deps import get_state
from reposage.api.schemas import AskRequest, AskResponse
from reposage.logging_setup import get_logger
from reposage.observability import Tracer, use_tracer

log = get_logger(__name__)
router = APIRouter(tags=["ask"])

# Visitors may supply their own key to bypass the shared demo budget.
API_KEY_HEADER = "x-reposage-key"


async def _resolve_agent(
    request: Request, repo: str, own_key: str | None
) -> tuple[CodebaseAgent | None, str | None, dict | None]:
    """Pick the agent for this request and evaluate the demo budget.

    Returns ``(agent, visitor, refusal)``. The refusal is returned rather than
    raised because the two callers need it in different shapes: the JSON
    endpoint turns it into a 429, while the SSE endpoint must open the stream
    successfully and deliver it as an event. A stream that fails its handshake
    gives the browser no way to read why.
    """
    state = get_state()
    try:
        index = await state.get_index(repo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    own_key = (own_key or "").strip()
    if own_key:
        state.budget.record_own_key()
        return CodebaseAgent(index, state.client_for_key(own_key), state.settings), None, None

    if not state.settings.has_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    if not state.settings.demo_mode:
        return await state.get_agent(repo), None, None

    visitor = visitor_id(
        request.client.host if request.client else None,
        request.headers.get("x-forwarded-for"),
        request.headers.get("user-agent"),
    )
    decision = state.budget.check(visitor)
    if not decision.allowed:
        return None, None, {
            "detail": decision.reason,
            "scope": decision.scope,
            "needs_own_key": decision.needs_own_key,
            "retry_after_seconds": decision.retry_after_seconds,
        }
    return await state.get_agent(repo), visitor, None


@router.post("/ask", response_model=AskResponse, summary="Ask a question about a repository")
async def ask(
    payload: AskRequest,
    request: Request,
    x_reposage_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> AskResponse:
    state = get_state()
    agent, visitor, refusal = await _resolve_agent(request, payload.repo, x_reposage_key)
    if refusal is not None:
        raise HTTPException(
            status_code=429,
            detail=refusal["detail"],
            headers={
                "retry-after": str(refusal["retry_after_seconds"]),
                "x-demo-scope": refusal["scope"],
                "x-demo-needs-own-key": "1" if refusal["needs_own_key"] else "0",
            },
        )
    assert agent is not None

    tracer = Tracer()
    try:
        with use_tracer(tracer):
            answer = await agent.ask(payload.question, tracer=tracer)
    except Exception as exc:
        log.exception("ask.failed", repo=payload.repo)
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc
    if visitor:
        state.budget.consume(visitor)

    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        citations=answer.citations,
        confidence=answer.confidence,
        refinement_rounds=answer.refinement_rounds,
        retrieved_paths=answer.retrieved_paths,
        usage=answer.usage,
        elapsed_seconds=answer.elapsed_seconds,
        trace=tracer.waterfall(),
    )


@router.get("/ask/stream", summary="Ask with live agent trace over SSE")
async def ask_stream(
    request: Request,
    repo: str = Query(description="Index id"),
    q: str = Query(min_length=3, description="The question"),
    key: str | None = Query(default=None, description="Optional caller-supplied API key"),
) -> EventSourceResponse:
    """Stream the agent's reasoning and answer tokens as they are produced.

    Uses GET because the browser ``EventSource`` API cannot issue a POST, which
    is also why an optional key arrives as a query parameter here rather than a
    header. That is acceptable only because the demo is the sole caller that
    uses it and the value is the caller's own credential.
    """
    state = get_state()
    agent, visitor, refusal = await _resolve_agent(request, repo, key)
    if visitor:
        state.budget.consume(visitor)

    async def publisher():
        # A refused request still opens a healthy stream and explains itself, so
        # the browser can render guidance instead of a bare connection failure.
        if refusal is not None:
            yield {"event": "limit", "data": orjson.dumps(refusal).decode()}
            return
        async for event in agent.astream(q):
            yield {
                "event": event.get("type", "message"),
                "data": orjson.dumps(event).decode(),
            }

    return EventSourceResponse(publisher())


@router.get("/source/{repo}", summary="Read an indexed file for citation display")
async def read_source(repo: str, path: str = Query(description="Repository-relative path")) -> dict:
    """Return the reconstructed content of an indexed file.

    Serving from the index rather than the filesystem keeps the endpoint safe by
    construction: only paths that were indexed can be read, so no traversal is
    possible regardless of what the caller sends.
    """
    state = get_state()
    try:
        index = await state.get_index(repo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    chunks = index.chunks_for_path(path)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"'{path}' is not present in index '{repo}'")
    return {
        "path": path,
        "language": chunks[0].language,
        "segments": [
            {
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol": c.qualified_name,
                "kind": c.kind.value,
                "content": c.content,
            }
            for c in chunks
        ],
    }
