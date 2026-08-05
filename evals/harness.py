"""The evaluation runner.

Two things happen here.

**Retrieval ablation.** The same golden questions are run through progressively
richer retrieval configurations, using the production code path rather than a
reimplementation. That turns "hybrid search and reranking help" from a claim
into a measurement, and it localises regressions: if recall drops after a
chunker change, the ablation shows whether the loss is in dense recall, lexical
recall, or the reranker's ordering.

**Answer evaluation.** The full agent answers each question and the result is
scored on grounded metrics (citation validity, required-fact coverage) and by an
LLM judge. Cost and latency are recorded per case, because an accuracy win that
triples the bill is a trade-off, not an improvement.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from evals.dataset import EvalCase
from evals.judges import JudgeVerdict, judge_answer
from evals.metrics import (
    RetrievalMetrics,
    aggregate_retrieval,
    citation_validity,
    evaluate_retrieval,
    keyword_coverage,
)
from reposage.agents.engine import CodebaseAgent
from reposage.agents.prompts import PROMPT_VERSION
from reposage.config import Settings, get_settings
from reposage.index.retriever import HybridRetriever
from reposage.index.store import RepoIndex
from reposage.llm.client import LLMClient
from reposage.logging_setup import get_logger
from reposage.observability import Tracer, use_tracer

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Variant:
    """One retrieval configuration in the ablation."""

    name: str
    mode: str = "hybrid"
    rerank: bool = False
    neighbours: bool = False
    description: str = ""


ABLATION: tuple[Variant, ...] = (
    Variant(
        "dense only", "dense", False, False, "Embedding similarity alone. The naive RAG baseline."
    ),
    Variant("lexical only", "lexical", False, False, "BM25 over code-aware tokens. No embeddings."),
    Variant(
        "hybrid + RRF", "hybrid", False, False, "Both retrievers fused by reciprocal rank fusion."
    ),
    Variant("hybrid + rerank", "hybrid", True, False, "Fusion, then LLM listwise reranking."),
    Variant("full pipeline", "hybrid", True, True, "Reranking plus adjacent-chunk expansion."),
)


@dataclass
class VariantResult:
    name: str
    description: str
    aggregate: dict[str, float] = field(default_factory=dict)
    per_case: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    case_id: str
    question: str
    category: str
    answer: str = ""
    confidence: float = 0.0
    citations: int = 0
    citation_valid_rate: float = 0.0
    fact_coverage: float = 1.0
    retrieval: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    refinements: int = 0
    elapsed_seconds: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    error: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.judge.get("passed")) and not self.error


@dataclass
class EvalReport:
    """Everything one evaluation run produced."""

    repo: str
    generated_at: str
    prompt_version: str
    config: dict[str, Any] = field(default_factory=dict)
    dataset_size: int = 0
    ablation: list[VariantResult] = field(default_factory=list)
    answers: list[AnswerResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    total_seconds: float = 0.0
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "generated_at": self.generated_at,
            "prompt_version": self.prompt_version,
            "config": self.config,
            "dataset_size": self.dataset_size,
            "ablation": [asdict(v) for v in self.ablation],
            "answers": [asdict(a) for a in self.answers],
            "summary": self.summary,
            "total_seconds": round(self.total_seconds, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


class EvalRunner:
    """Runs the ablation and the answer suite against one index."""

    def __init__(
        self,
        index: RepoIndex,
        client: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.index = index
        self.client = client
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever(index, client, self.settings)
        self.indexed_paths = set(index.paths())

    # ------------------------------------------------------------- ablation
    async def run_ablation(
        self, cases: list[EvalCase], variants: tuple[Variant, ...] = ABLATION
    ) -> list[VariantResult]:
        """Score each retrieval configuration over the whole dataset."""
        results: list[VariantResult] = []
        for variant in variants:
            tracer = Tracer()
            started = time.perf_counter()
            per_case: list[RetrievalMetrics] = []

            with use_tracer(tracer):
                for case in cases:
                    try:
                        chunks, _ = await self.retriever.retrieve(
                            case.question,
                            top_k=self.settings.top_k,
                            rerank=variant.rerank,
                            expand_neighbours=variant.neighbours,
                            mode=variant.mode,
                        )
                        paths: list[str] = []
                        for scored in chunks:
                            if scored.chunk.path not in paths:
                                paths.append(scored.chunk.path)
                    except Exception as exc:
                        log.warning("ablation.case_failed", case=case.id, error=str(exc)[:160])
                        paths = []
                    per_case.append(
                        evaluate_retrieval(
                            case.id, paths, case.expected_paths, k=self.settings.top_k
                        )
                    )

            results.append(
                VariantResult(
                    name=variant.name,
                    description=variant.description,
                    aggregate=aggregate_retrieval(per_case),
                    per_case=[m.as_dict() for m in per_case],
                    elapsed_seconds=round(time.perf_counter() - started, 2),
                    usage=tracer.usage.model_dump(),
                )
            )
            log.info(
                "ablation.variant_done",
                variant=variant.name,
                **{
                    k: v
                    for k, v in results[-1].aggregate.items()
                    if k in ("hit_rate", "recall@k", "mrr")
                },
            )
        return results

    # --------------------------------------------------------------- answers
    async def run_answers(
        self, cases: list[EvalCase], *, judge: bool = True, concurrency: int = 1
    ) -> list[AnswerResult]:
        """Answer every case with the full agent and score the results."""
        agent = CodebaseAgent(self.index, self.client, self.settings)
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _one(case: EvalCase) -> AnswerResult:
            async with semaphore:
                result = AnswerResult(
                    case_id=case.id, question=case.question, category=case.category
                )
                tracer = Tracer()
                try:
                    with use_tracer(tracer):
                        answer = await agent.ask(case.question, tracer=tracer)
                except Exception as exc:
                    log.warning("answers.case_failed", case=case.id, error=str(exc)[:200])
                    result.error = str(exc)[:300]
                    return result

                result.answer = answer.answer
                result.confidence = answer.confidence
                result.refinements = answer.refinement_rounds
                result.elapsed_seconds = answer.elapsed_seconds
                result.cost_usd = answer.usage.cost_usd
                result.tokens = answer.usage.total_tokens

                validity = citation_validity(answer.citations, self.indexed_paths)
                result.citations = int(validity["citations"])
                result.citation_valid_rate = validity["valid_rate"]
                result.fact_coverage = keyword_coverage(answer.answer, case.must_mention)
                result.retrieval = evaluate_retrieval(
                    case.id, answer.retrieved_paths, case.expected_paths, k=self.settings.top_k * 2
                ).as_dict()

                if judge:
                    verdict: JudgeVerdict = await judge_answer(
                        self.client,
                        case.question,
                        answer.answer,
                        case.reference_answer,
                        [c.label for c in answer.citations],
                    )
                    result.judge = {
                        "correctness": verdict.correctness,
                        "groundedness": verdict.groundedness,
                        "completeness": verdict.completeness,
                        "overall": verdict.overall,
                        "passed": verdict.passed,
                        "hallucinations": verdict.hallucinations,
                        "reasoning": verdict.reasoning[:600],
                    }
                return result

        return await asyncio.gather(*(_one(case) for case in cases))

    # ------------------------------------------------------------------ all
    async def run(
        self, cases: list[EvalCase], *, ablation: bool = True, judge: bool = True
    ) -> EvalReport:
        started = time.perf_counter()
        report = EvalReport(
            repo=self.index.metadata.name,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            prompt_version=PROMPT_VERSION,
            config={
                **self.settings.fingerprint(),
                "index_chunks": len(self.index),
                "index_files": len(self.indexed_paths),
                "commit": self.index.metadata.commit[:8],
            },
            dataset_size=len(cases),
        )
        if ablation:
            report.ablation = await self.run_ablation(cases)
        report.answers = await self.run_answers(cases, judge=judge)
        report.summary = summarise(report)
        report.total_seconds = round(time.perf_counter() - started, 2)
        report.total_cost_usd = round(
            sum(v.usage.get("cost_usd", 0.0) for v in report.ablation)
            + sum(a.cost_usd for a in report.answers),
            6,
        )
        return report


def summarise(report: EvalReport) -> dict[str, Any]:
    """Headline numbers, including the lift the full pipeline gives over naive RAG."""
    answers = [a for a in report.answers if not a.error]
    judged = [a for a in answers if a.judge]

    summary: dict[str, Any] = {
        "answered": len(answers),
        "failed": len(report.answers) - len(answers),
        "pass_rate": round(mean([1.0 if a.passed else 0.0 for a in judged]), 4) if judged else 0.0,
        "mean_confidence": round(mean([a.confidence for a in answers]), 4) if answers else 0.0,
        "mean_citations": round(mean([a.citations for a in answers]), 2) if answers else 0.0,
        "citation_valid_rate": round(mean([a.citation_valid_rate for a in answers]), 4)
        if answers
        else 0.0,
        "fact_coverage": round(mean([a.fact_coverage for a in answers]), 4) if answers else 0.0,
        "answer_recall@k": round(mean([a.retrieval.get("recall@k", 0.0) for a in answers]), 4)
        if answers
        else 0.0,
        "mean_latency_s": round(mean([a.elapsed_seconds for a in answers]), 2) if answers else 0.0,
        "mean_cost_usd": round(mean([a.cost_usd for a in answers]), 6) if answers else 0.0,
        "mean_tokens": int(mean([a.tokens for a in answers])) if answers else 0,
        "total_refinements": sum(a.refinements for a in answers),
    }
    if judged:
        summary.update(
            {
                "judge_correctness": round(mean([a.judge["correctness"] for a in judged]), 2),
                "judge_groundedness": round(mean([a.judge["groundedness"] for a in judged]), 2),
                "judge_completeness": round(mean([a.judge["completeness"] for a in judged]), 2),
                "judge_overall": round(mean([a.judge["overall"] for a in judged]), 2),
                "hallucination_cases": sum(1 for a in judged if a.judge.get("hallucinations")),
            }
        )
    if report.ablation:
        by_name = {v.name: v.aggregate.get("recall@k", 0.0) for v in report.ablation}
        baseline = by_name.get("dense only", 0.0)
        best = by_name.get("full pipeline", 0.0)
        summary["retrieval_lift_vs_dense_only"] = (
            round((best - baseline) / baseline, 4) if baseline else None
        )
        summary["best_variant"] = max(by_name, key=lambda k: by_name[k]) if by_name else None
    return summary
