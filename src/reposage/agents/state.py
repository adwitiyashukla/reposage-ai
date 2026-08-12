from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, TypedDict

from reposage.models import Citation, Critique, QueryPlan, ScoredChunk

if TYPE_CHECKING:
    from reposage.config import Settings
    from reposage.index.retriever import HybridRetriever
    from reposage.llm.client import LLMClient


class AgentState(TypedDict, total=False):
    question: str
    repo_name: str
    repo_map: str
    languages: str
    stream_tokens: bool

    plan: QueryPlan | None
    active_queries: list[str]
    path_hints: list[str]

    retrieved: list[ScoredChunk]
    retrieval_debug: Annotated[list[dict], operator.add]

    draft: str
    critique: Critique | None
    refinements: int
    confidence: float
    citations: list[Citation]

    errors: Annotated[list[str], operator.add]


@dataclass(slots=True)
class AgentDeps:
    client: LLMClient
    retriever: HybridRetriever
    settings: Settings

    @property
    def repo_map(self) -> str:
        return self.retriever.index.repo_map

    @property
    def repo_name(self) -> str:
        return self.retriever.index.metadata.name


def initial_state(question: str, deps: AgentDeps, *, stream_tokens: bool = False) -> AgentState:
    metadata = deps.retriever.index.metadata
    languages = ", ".join(list(metadata.languages)[:8]) or "unknown"
    return AgentState(
        question=question.strip(),
        repo_name=metadata.name,
        repo_map=deps.repo_map,
        languages=languages,
        stream_tokens=stream_tokens,
        plan=None,
        active_queries=[],
        path_hints=[],
        retrieved=[],
        retrieval_debug=[],
        draft="",
        critique=None,
        refinements=0,
        confidence=0.0,
        citations=[],
        errors=[],
    )
