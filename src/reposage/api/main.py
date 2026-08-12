from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from reposage import __version__
from reposage.api.deps import get_state, reset_state
from reposage.api.routes import ask, health, indexes, review
from reposage.config import get_settings
from reposage.logging_setup import configure_logging, get_logger

log = get_logger(__name__)

DESCRIPTION = """
Agentic code intelligence for any Git repository.

**Index** a repository, **ask** architectural questions and get answers with
verified line-level citations, or **review** a pull request diff.

Retrieval is hybrid (dense + BM25 + reciprocal rank fusion + LLM reranking) over
AST-aware chunks. Answering runs a LangGraph agent that plans, retrieves,
drafts, self-critiques and refines before returning.
"""

TAGS = [
    {"name": "system", "description": "Health, readiness and configuration."},
    {"name": "indexes", "description": "Build, inspect and delete repository indexes."},
    {"name": "ask", "description": "Question answering with citations."},
    {"name": "review", "description": "Automated pull-request review."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()
    state = get_state()
    log.info(
        "api.startup",
        version=__version__,
        indexes=len(state.catalogue()),
        api_key_configured=settings.has_api_key,
    )
    if not settings.has_api_key:
        log.warning("api.no_api_key", hint="Set GEMINI_API_KEY in .env to enable model calls.")
    yield
    await reset_state()
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RepoSage",
        description=DESCRIPTION,
        version=__version__,
        openapi_tags=TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
        if not request.url.path.startswith(("/assets", "/static")):
            log.debug(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round(elapsed_ms, 1),
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("http.unhandled", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": str(exc)[:400],
                "type": type(exc).__name__,
            },
        )

    app.include_router(health.router, prefix="/api")
    app.include_router(indexes.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    app.include_router(review.router, prefix="/api")

    _mount_web(app)
    return app


def _mount_web(app: FastAPI) -> None:
    web_dir = Path(__file__).resolve().parent.parent / "web"
    index_html = web_dir / "index.html"
    if not index_html.exists():
        log.warning("api.web_missing", path=str(web_dir))
        return

    assets = web_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(str(index_html))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)


app = create_app()
