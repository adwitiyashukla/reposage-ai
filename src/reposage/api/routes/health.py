"""Health, readiness and introspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from reposage import __version__
from reposage.agents.graph import describe_graph
from reposage.agents.prompts import PROMPT_VERSION
from reposage.api.deps import get_state
from reposage.api.schemas import HealthResponse
from reposage.ingest.chunker import available_grammars, treesitter_available

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and configuration")
async def health() -> HealthResponse:
    """Report version, configuration and index count without calling the model."""
    state = get_state()
    return HealthResponse(
        status="ok" if state.settings.has_api_key else "degraded",
        version=__version__,
        prompt_version=PROMPT_VERSION,
        indexes=len(state.catalogue()),
        llm={
            "configured": state.settings.has_api_key,
            "fast_model": state.settings.fast_model,
            "deep_model": state.settings.deep_model,
            "embed_model": state.settings.embed_model,
        },
        settings={
            **state.settings.fingerprint(),
            "treesitter": treesitter_available(),
            "grammars": available_grammars(),
        },
    )


@router.get("/ready", summary="Readiness probe including a live model call")
async def ready() -> dict:
    """Verify the model is reachable. Used by container orchestrators."""
    state = get_state()
    if not state.settings.has_api_key:
        return {"ready": False, "reason": "GEMINI_API_KEY is not configured"}
    result = await state.client.healthcheck()
    return {"ready": bool(result.get("ok")), **result}


@router.get("/graph", summary="Agent graph as Mermaid source")
async def graph() -> dict:
    return {"format": "mermaid", "source": describe_graph()}


@router.get("/stats", summary="Cumulative token usage and cache statistics")
async def stats() -> dict:
    return get_state().client.stats()
