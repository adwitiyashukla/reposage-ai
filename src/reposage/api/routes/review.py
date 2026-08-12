from __future__ import annotations

from fastapi import APIRouter, HTTPException

from reposage.api.deps import get_state
from reposage.api.schemas import ReviewRequest, ReviewResponse
from reposage.index.retriever import HybridRetriever
from reposage.logging_setup import get_logger
from reposage.observability import Tracer, use_tracer
from reposage.review.reviewer import PullRequestReviewer

log = get_logger(__name__)
router = APIRouter(tags=["review"])


@router.post("/review", response_model=ReviewResponse, summary="Review a unified diff")
async def review(request: ReviewRequest) -> ReviewResponse:
    state = get_state()
    if not state.settings.has_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    retriever: HybridRetriever | None = None
    if request.repo:
        try:
            index = await state.get_index(request.repo)
            retriever = HybridRetriever(index, state.client, state.settings)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    reviewer = PullRequestReviewer(state.client, retriever, state.settings)
    tracer = Tracer()
    try:
        with use_tracer(tracer):
            report = await reviewer.review(
                request.diff,
                title=request.title,
                description=request.description,
                min_confidence=request.min_confidence,
            )
    except Exception as exc:
        log.exception("review.failed")
        raise HTTPException(status_code=500, detail=f"Review failed: {exc}") from exc

    return ReviewResponse(
        summary=report.summary,
        findings=[f.model_dump(mode="json") for f in report.sorted_findings()],
        files_reviewed=report.files_reviewed,
        blocking=len(report.blocking),
        usage=tracer.usage,
        elapsed_seconds=report.elapsed_seconds,
    )
