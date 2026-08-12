from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkKind(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    MODULE = "module"
    IMPORTS = "imports"
    CONFIG = "config"
    DOCUMENTATION = "documentation"
    BLOCK = "block"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NIT = "nit"

    @property
    def rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3, "nit": 4}[self.value]

    @property
    def label(self) -> str:
        return {
            "critical": "CRITICAL",
            "high": "HIGH",
            "medium": "MEDIUM",
            "low": "LOW",
            "nit": "NIT",
        }[self.value]


@dataclass(slots=True)
class Chunk:
    path: str
    content: str
    start_line: int
    end_line: int
    language: str = "text"
    kind: ChunkKind = ChunkKind.BLOCK
    symbol: str | None = None
    parent_symbol: str | None = None
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            digest = hashlib.sha1(
                f"{self.path}:{self.start_line}:{self.end_line}:{self.content[:256]}".encode(
                    "utf-8", errors="ignore"
                )
            ).hexdigest()[:16]
            self.chunk_id = digest

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def qualified_name(self) -> str:
        if self.symbol and self.parent_symbol:
            return f"{self.parent_symbol}.{self.symbol}"
        return self.symbol or self.path

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.content) // 4)

    def render(self, with_header: bool = True) -> str:
        if not with_header:
            return self.content
        header = f"// {self.location}"
        if self.symbol:
            header += f"  [{self.kind.value}: {self.qualified_name}]"
        return f"{header}\n{self.content}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        payload = dict(data)
        payload["kind"] = ChunkKind(payload.get("kind", "block"))
        return cls(**payload)


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    retrievers: tuple[str, ...] = ()

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def provenance(self) -> str:
        parts = []
        if self.dense_rank is not None:
            parts.append(f"dense#{self.dense_rank + 1}")
        if self.lexical_rank is not None:
            parts.append(f"bm25#{self.lexical_rank + 1}")
        if self.rerank_score is not None:
            parts.append(f"rerank={self.rerank_score:.2f}")
        return " + ".join(parts) or "unknown"


class Citation(BaseModel):
    path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    excerpt: str = ""

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens

    def merge(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            cache_hits=self.cache_hits + other.cache_hits,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
        )


class SubQuestion(BaseModel):
    question: str
    search_queries: list[str] = Field(default_factory=list)
    rationale: str = ""


class QueryPlan(BaseModel):
    intent: str = "explain"
    restated_question: str = ""
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    keyword_hints: list[str] = Field(default_factory=list)
    path_hints: list[str] = Field(default_factory=list)
    needs_retrieval: bool = True

    @property
    def all_queries(self) -> list[str]:
        seen: dict[str, None] = {}
        for sq in self.sub_questions:
            for q in sq.search_queries or [sq.question]:
                if q.strip():
                    seen.setdefault(q.strip(), None)
        if not seen and self.restated_question:
            seen[self.restated_question] = None
        return list(seen)


class Critique(BaseModel):
    grounded: bool = True
    complete: bool = True
    confidence: float = 0.8
    issues: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    verdict: str = "accept"

    @property
    def needs_refinement(self) -> bool:
        return self.verdict == "refine" and bool(self.follow_up_queries)


class AgentAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    plan: QueryPlan | None = None
    critique: Critique | None = None
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0.0
    refinement_rounds: int = 0
    retrieved_paths: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    path: str
    line: int | None = None
    severity: Severity = Severity.MEDIUM
    title: str = ""
    body: str = ""
    category: str = "correctness"
    suggestion: str | None = None
    confidence: float = 0.7

    def to_markdown(self) -> str:
        loc = f"`{self.path}`" + (f" line {self.line}" if self.line else "")
        out = [
            f"**{self.severity.label} - {self.title}**",
            "",
            f"{loc} - _{self.category}_",
            "",
            self.body,
        ]
        if self.suggestion:
            out += ["", "```suggestion", self.suggestion, "```"]
        return "\n".join(out)


class ReviewReport(BaseModel):
    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = ""
    files_reviewed: int = 0
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0.0

    @property
    def blocking(self) -> list[ReviewFinding]:
        return [f for f in self.findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]

    def sorted_findings(self) -> list[ReviewFinding]:
        return sorted(self.findings, key=lambda f: (f.severity.rank, f.path, f.line or 0))


@dataclass
class RepoMetadata:
    repo_id: str
    source: str
    name: str
    commit: str = ""
    branch: str = ""
    indexed_at: str = ""
    num_files: int = 0
    num_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    embed_model: str = ""
    total_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepoMetadata:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
