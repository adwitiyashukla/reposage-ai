"""Question answering, batch and streaming."""

from __future__ import annotations

import orjson
from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from reposage.api.deps import get_state
from reposage.api.schemas import AskRequest, AskResponse
from reposage.logging_setup import get_logger
from reposage.observability import Tracer, use_tracer

log = get_logger(__name__)
router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse, summary="Ask a question about a repository")
async def ask(request: AskRequest) -> AskResponse:
    state = get_state()
    if not state.settings.has_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")
    try:
        agent = await state.get_agent(request.repo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    tracer = Tracer()
    try:
        with use_tracer(tracer):
            answer = await agent.ask(request.question, tracer=tracer)
    except Exception as exc:
        log.exception("ask.failed", repo=request.repo)
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc

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
    repo: str = Query(description="Index id"),
    q: str = Query(min_length=3, description="The question"),
) -> EventSourceResponse:
    """Stream the agent's reasoning and answer tokens as they are produced.

    Uses GET because the browser ``EventSource`` API cannot issue a POST.
    """
    state = get_state()
    if not state.settings.has_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")
    try:
        agent = await state.get_agent(repo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def publisher():
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
