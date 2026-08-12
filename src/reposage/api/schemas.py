from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from reposage.models import Citation, Usage


class IndexRequest(BaseModel):
    source: str = Field(
        description="Repository to index: a GitHub URL, an owner/repo shorthand, or a local path.",
        examples=["tiangolo/fastapi", "https://github.com/psf/requests.git", "./my-project"],
    )
    branch: str | None = None
    refresh: bool = Field(default=False, description="Discard any cached clone and re-fetch.")


class IndexSummary(BaseModel):
    id: str
    name: str
    commit: str = ""
    files: int = 0
    chunks: int = 0
    indexed_at: str = ""
    languages: list[str] = Field(default_factory=list)


class IndexResponse(BaseModel):
    index: IndexSummary
    stats: dict[str, Any] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    usage: Usage = Field(default_factory=Usage)


class AskRequest(BaseModel):
    repo: str = Field(description="Index id, as returned by GET /api/indexes.")
    question: str = Field(min_length=3, max_length=4000)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    refinement_rounds: int = 0
    retrieved_paths: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0.0
    trace: list[dict[str, Any]] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    diff: str = Field(min_length=1, description="A unified diff.")
    title: str = "Untitled change"
    description: str = ""
    repo: str | None = Field(
        default=None,
        description="Optional index id used to ground the review in repository context.",
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ReviewResponse(BaseModel):
    summary: str
    findings: list[dict[str, Any]] = Field(default_factory=list)
    files_reviewed: int = 0
    blocking: int = 0
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0.0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    prompt_version: str
    indexes: int = 0
    llm: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    type: str = "error"
