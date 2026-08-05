"""The pull-request review agent.

The thing that separates a useful automated reviewer from a noise generator is
context. A model shown only a diff will confidently flag a missing null check
that is guaranteed by a caller three files away, or miss that a renamed method
has six other call sites. So before reviewing each file this agent retrieves the
surrounding code from the indexed repository using the identifiers the diff
touches, and reviews the change against that.

The second thing is discipline about output. Findings are filtered by model
confidence, anchored to lines that actually changed, and deduplicated, because a
review with three real findings gets read and one with thirty speculative ones
gets muted.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from reposage.agents.prompts import (
    REVIEW_SUMMARY_SYSTEM,
    REVIEWER_SYSTEM,
    REVIEWER_USER,
    format_context,
)
from reposage.config import Settings, get_settings
from reposage.llm.client import LLMClient, ModelTier
from reposage.logging_setup import get_logger
from reposage.models import ReviewFinding, ReviewReport, Severity
from reposage.observability import current_tracer
from reposage.review.diff import DiffFile, ParsedDiff, parse_unified_diff

if TYPE_CHECKING:  # pragma: no cover
    from reposage.index.retriever import HybridRetriever

log = get_logger(__name__)

MIN_CONFIDENCE = 0.5
MAX_FINDINGS_PER_FILE = 6
MAX_TOTAL_FINDINGS = 25
CONTEXT_CHUNKS = 6


class _Finding(BaseModel):
    """Wire format for one finding, before validation and anchoring."""

    line: int | None = Field(default=None, description="Line number in the new file")
    severity: str = "medium"
    category: str = "correctness"
    title: str = ""
    body: str = ""
    suggestion: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class _FileReview(BaseModel):
    findings: list[_Finding] = Field(default_factory=list)
    verdict: str = "ok"


class PullRequestReviewer:
    """Reviews a unified diff, optionally grounded in an indexed repository."""

    def __init__(
        self,
        client: LLMClient,
        retriever: HybridRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ API
    async def review(
        self,
        diff_text: str,
        *,
        title: str = "Untitled change",
        description: str = "",
        concurrency: int = 3,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> ReviewReport:
        """Review a unified diff and return findings plus a summary."""
        started = time.perf_counter()
        tracer = current_tracer()
        diff = parse_unified_diff(diff_text)
        targets = diff.reviewable

        with tracer.span("review.run", reviewing=len(targets), **diff.stats()) as span:
            if not targets:
                return ReviewReport(
                    summary="No reviewable changes were found in this diff.",
                    files_reviewed=0,
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                )

            semaphore = asyncio.Semaphore(max(1, concurrency))

            async def _one(diff_file: DiffFile) -> list[ReviewFinding]:
                async with semaphore:
                    return await self._review_file(diff_file, title, description, min_confidence)

            results = await asyncio.gather(*(_one(f) for f in targets), return_exceptions=True)

            findings: list[ReviewFinding] = []
            for diff_file, result in zip(targets, results, strict=True):
                if isinstance(result, BaseException):
                    log.warning("review.file_failed", path=diff_file.path, error=str(result)[:200])
                    continue
                findings.extend(result)

            findings = _deduplicate(findings)[:MAX_TOTAL_FINDINGS]
            summary = await self._summarise(findings, diff, title)
            span.set(
                findings=len(findings),
                blocking=sum(
                    1 for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)
                ),
            )

        report = ReviewReport(
            findings=findings,
            summary=summary,
            files_reviewed=len(targets),
            usage=self.client.usage,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        log.info(
            "review.complete",
            files=len(targets),
            findings=len(findings),
            blocking=len(report.blocking),
            seconds=report.elapsed_seconds,
        )
        return report

    # -------------------------------------------------------------- internals
    async def _review_file(
        self,
        diff_file: DiffFile,
        title: str,
        description: str,
        min_confidence: float,
    ) -> list[ReviewFinding]:
        tracer = current_tracer()
        with tracer.span("review.file", path=diff_file.path, additions=diff_file.additions) as span:
            context_block = await self._related_context(diff_file)
            prompt = REVIEWER_USER.format(
                title=title,
                description_block=f"Description:\n{description.strip()[:2000]}\n\n"
                if description.strip()
                else "",
                file_list=diff_file.path,
                context_block=context_block,
                diff=diff_file.render(),
            )
            try:
                review = await self.client.structured(
                    prompt,
                    _FileReview,
                    tier=ModelTier.DEEP,
                    system=REVIEWER_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=3072,
                )
            except Exception as exc:
                log.warning("review.model_failed", path=diff_file.path, error=str(exc)[:200])
                span.set(status="error")
                return []

            valid = _validate(review.findings, diff_file, min_confidence)
            span.set(raw=len(review.findings), kept=len(valid))
            return valid[:MAX_FINDINGS_PER_FILE]

    async def _related_context(self, diff_file: DiffFile) -> str:
        """Retrieve repository code related to the identifiers this diff touches."""
        if self.retriever is None:
            return ""
        terms = diff_file.search_terms()
        if not terms:
            return ""
        queries = [
            " ".join(terms[:6]),
            f"{diff_file.path} {' '.join(terms[:3])}",
        ]
        try:
            chunks, _ = await self.retriever.retrieve(
                queries,
                top_k=CONTEXT_CHUNKS,
                path_hints=[diff_file.path.rsplit("/", 1)[0]] if "/" in diff_file.path else None,
                rerank=False,
                expand_neighbours=False,
            )
        except Exception as exc:
            log.debug("review.context_failed", path=diff_file.path, error=str(exc)[:160])
            return ""
        # Exclude the file under review: the diff already shows it, and repeating
        # it wastes context and invites comments on unchanged lines.
        chunks = [c for c in chunks if c.chunk.path != diff_file.path][:CONTEXT_CHUNKS]
        if not chunks:
            return ""
        return (
            "Related code from the repository (for context only, do not review it):\n\n"
            + format_context(chunks, max_chars=18_000)
            + "\n\n"
        )

    async def _summarise(self, findings: list[ReviewFinding], diff: ParsedDiff, title: str) -> str:
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
        listing = (
            "\n".join(
                f"- [{f.severity.value}] {f.path}:{f.line or '?'} {f.title}" for f in findings[:20]
            )
            or "No findings."
        )
        prompt = (
            f"Pull request: {title}\n"
            f"Diff: {diff.stats()['files']} files, +{diff.additions} / -{diff.deletions} lines\n"
            f"Findings by severity: {counts or 'none'}\n\n"
            f"Findings:\n{listing}\n\n"
            "Write the review summary."
        )
        try:
            response = await self.client.complete(
                prompt,
                tier=ModelTier.FAST,
                system=REVIEW_SUMMARY_SYSTEM,
                temperature=0.2,
                max_output_tokens=512,
            )
            return response.text.strip()
        except Exception as exc:
            log.debug("review.summary_failed", error=str(exc)[:160])
            if not findings:
                return "No issues found in this change."
            return f"Found {len(findings)} issue(s) across {diff.stats()['reviewable']} file(s)."


def _validate(
    raw_findings: list[_Finding], diff_file: DiffFile, min_confidence: float
) -> list[ReviewFinding]:
    """Anchor findings to changed lines and drop low-confidence noise."""
    changed = diff_file.changed_line_numbers
    validated: list[ReviewFinding] = []

    for raw in raw_findings:
        if raw.confidence < min_confidence:
            continue
        if not (raw.title or raw.body):
            continue
        try:
            severity = Severity(raw.severity.strip().lower())
        except ValueError:
            severity = Severity.MEDIUM

        line = raw.line
        if line is not None and changed and line not in changed:
            # The model is often a line or two off. Snap to the nearest changed
            # line when it is close, otherwise drop the anchor and comment at
            # file level rather than pointing somewhere misleading.
            nearest = min(changed, key=lambda candidate: abs(candidate - line))
            line = nearest if abs(nearest - line) <= 3 else None

        validated.append(
            ReviewFinding(
                path=diff_file.path,
                line=line,
                severity=severity,
                title=raw.title.strip()[:200] or "Issue",
                body=raw.body.strip(),
                category=raw.category.strip().lower()[:40] or "correctness",
                suggestion=raw.suggestion,
                confidence=raw.confidence,
            )
        )
    return sorted(validated, key=lambda f: (f.severity.rank, -f.confidence))


def _deduplicate(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    """Collapse findings that repeat the same point at the same location."""
    seen: dict[tuple[str, int, str], ReviewFinding] = {}
    for finding in findings:
        key = (finding.path, finding.line or -1, finding.title.lower()[:60])
        existing = seen.get(key)
        if existing is None or finding.confidence > existing.confidence:
            seen[key] = finding
    return sorted(seen.values(), key=lambda f: (f.severity.rank, f.path, f.line or 0))
