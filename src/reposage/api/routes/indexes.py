"""Index lifecycle: list, build, inspect and delete."""

from __future__ import annotations

import asyncio
import time

import orjson
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from reposage.api.deps import get_state
from reposage.api.schemas import IndexRequest, IndexResponse, IndexSummary
from reposage.index.store import RepoIndex
from reposage.ingest.pipeline import IngestionPipeline
from reposage.logging_setup import get_logger
from reposage.observability import Tracer, use_tracer

log = get_logger(__name__)
router = APIRouter(prefix="/indexes", tags=["indexes"])

_DEMO_LOCKED = (
    "Indexing is disabled on the public demo. Indexing a repository costs "
    "hundreds of embedding requests and several minutes, so the demo ships with "
    "a pre-built index. Run RepoSage locally to index your own repositories: "
    "https://github.com/adwitiyashukla/reposage-ai"
)


def _reject_if_demo() -> None:
    """Write operations are unavailable on the hosted demo."""
    if get_state().settings.demo_mode:
        raise HTTPException(status_code=403, detail=_DEMO_LOCKED)


@router.get("", summary="List every index on disk")
async def list_all() -> dict:
    return {"indexes": get_state().catalogue()}


@router.get("/{name}", summary="Describe one index")
async def describe(name: str) -> dict:
    state = get_state()
    try:
        index = await state.get_index(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return index.describe()


@router.delete("/{name}", summary="Delete an index")
async def delete(name: str) -> dict:
    _reject_if_demo()
    state = get_state()
    removed = RepoIndex.delete(name, state.settings)
    state.invalidate(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No index named '{name}'")
    return {"deleted": name}


@router.post("", response_model=IndexResponse, summary="Build an index synchronously")
async def build(request: IndexRequest) -> IndexResponse:
    """Clone, chunk, embed and persist a repository.

    Long-running. Prefer the streaming variant for anything a user is watching.
    """
    _reject_if_demo()
    state = get_state()
    if not state.settings.has_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    started = time.perf_counter()
    tracer = Tracer()
    try:
        with use_tracer(tracer):
            ingestion = await IngestionPipeline(state.settings).run(
                request.source, branch=request.branch, refresh=request.refresh
            )
            index = await RepoIndex.build(ingestion, state.client, settings=state.settings)
            await asyncio.to_thread(index.save, state.settings.index_dir)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("index.build_failed", source=request.source)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    state.register(index)
    metadata = index.metadata
    return IndexResponse(
        index=IndexSummary(
            id=index.index_id,
            name=metadata.name,
            commit=metadata.commit[:8],
            files=metadata.num_files,
            chunks=metadata.num_chunks,
            indexed_at=metadata.indexed_at,
            languages=list(metadata.languages)[:6],
        ),
        stats=index.stats.as_dict(),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        usage=tracer.usage,
    )


@router.get("/stream/build", summary="Build an index with live progress over SSE")
async def build_streaming(
    source: str, branch: str | None = None, refresh: bool = False
) -> EventSourceResponse:
    """Same work as POST /indexes, with progress events as they happen."""
    _reject_if_demo()
    state = get_state()
    if not state.settings.has_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    async def publisher():
        tracer = Tracer()
        queue = tracer.subscribe()

        async def _work():
            with use_tracer(tracer):
                ingestion = await IngestionPipeline(state.settings).run(
                    source, branch=branch, refresh=refresh
                )
                index = await RepoIndex.build(ingestion, state.client, settings=state.settings)
                await asyncio.to_thread(index.save, state.settings.index_dir)
                state.register(index)
                return index

        task = asyncio.create_task(_work())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.25)
                    yield {"event": "progress", "data": orjson.dumps(event.to_dict()).decode()}
                except (asyncio.TimeoutError, TimeoutError):
                    if task.done() and queue.empty():
                        break
            index = await task
            yield {
                "event": "complete",
                "data": orjson.dumps(
                    {
                        "index": index.index_id,
                        "name": index.metadata.name,
                        "files": index.metadata.num_files,
                        "chunks": index.metadata.num_chunks,
                        "stats": index.stats.as_dict(),
                        "usage": tracer.usage.model_dump(),
                    }
                ).decode(),
            }
        except Exception as exc:
            log.exception("index.stream_failed", source=source)
            yield {"event": "error", "data": orjson.dumps({"error": str(exc)[:500]}).decode()}
        finally:
            tracer.unsubscribe(queue)

    return EventSourceResponse(publisher())
