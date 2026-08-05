"""Finalisation node: extract and validate citations, score confidence."""

from __future__ import annotations

import re

from reposage.agents.state import AgentDeps, AgentState
from reposage.logging_setup import get_logger
from reposage.models import Citation
from reposage.observability import current_tracer

log = get_logger(__name__)

# Matches [src/auth/jwt.py:12-48] and the single-line form [src/auth/jwt.py:12].
_CITATION = re.compile(r"\[([^\[\]\s]+?\.[A-Za-z0-9_]+):(\d+)(?:\s*-\s*(\d+))?\]")

# The highest confidence the system will ever report. See _score.
_CONFIDENCE_CEILING = 0.97


async def finalizer_node(state: AgentState, deps: AgentDeps) -> dict:
    """Resolve inline citation markers against the index and score the answer.

    Citations are *verified*, not merely parsed: a marker pointing at a file that
    is not in the index is dropped and counted against confidence. That closes
    the most common way a grounded-looking answer is still wrong.
    """
    tracer = current_tracer()
    draft = state.get("draft", "")

    with tracer.span("agent.finalise") as span:
        index = deps.retriever.index
        known_paths = set(index.paths())
        retrieved = state.get("retrieved") or []
        retrieved_paths = {c.chunk.path for c in retrieved}

        citations: list[Citation] = []
        seen: set[tuple[str, int, int]] = set()
        invalid = 0

        for match in _CITATION.finditer(draft):
            path, start_raw, end_raw = match.group(1), match.group(2), match.group(3)
            start = int(start_raw)
            end = int(end_raw) if end_raw else start
            if end < start:
                start, end = end, start

            resolved = path if path in known_paths else _resolve_path(path, known_paths)
            if resolved is None:
                invalid += 1
                continue
            key = (resolved, start, end)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    path=resolved,
                    start_line=start,
                    end_line=end,
                    symbol=_symbol_at(index, resolved, start, end),
                    excerpt=_excerpt(index, resolved, start, end),
                )
            )

        confidence = _score(state, citations, invalid, retrieved_paths)
        span.set(
            citations=len(citations),
            invalid_citations=invalid,
            confidence=round(confidence, 3),
        )
        if invalid:
            tracer.log("finalise.invalid_citations", count=invalid)
        return {"confidence": confidence, "citations": citations}


def _resolve_path(path: str, known: set[str]) -> str | None:
    """Accept a suffix match, which is how models usually shorten paths."""
    candidates = [p for p in known if p.endswith(path) or p.endswith("/" + path)]
    return min(candidates, key=len) if len(candidates) >= 1 else None


def _symbol_at(index, path: str, start: int, end: int) -> str | None:
    for chunk in index.chunks_for_path(path):
        if chunk.start_line <= start <= chunk.end_line and chunk.symbol:
            return chunk.qualified_name
    _ = end
    return None


def _excerpt(index, path: str, start: int, end: int, max_lines: int = 6) -> str:
    for chunk in index.chunks_for_path(path):
        if chunk.start_line <= start <= chunk.end_line:
            offset = max(0, start - chunk.start_line)
            body = chunk.content.splitlines()
            window = body[offset : offset + min(max_lines, end - start + 1)]
            return "\n".join(window)[:600]
    return ""


def _score(
    state: AgentState,
    citations: list[Citation],
    invalid: int,
    retrieved_paths: set[str],
) -> float:
    """Blend the critic's judgement with objective grounding signals.

    The critic is a language model judging itself, so its confidence alone is not
    trustworthy. We temper it with things we can verify: whether citations exist,
    whether they resolve, and whether they point at code we actually retrieved.
    """
    critique = state.get("critique")
    base = critique.confidence if critique else 0.5
    if critique:
        if not critique.grounded:
            base *= 0.55
        if not critique.complete:
            base *= 0.8

    if not citations:
        base *= 0.6
    else:
        cited_paths = {c.path for c in citations}
        overlap = len(cited_paths & retrieved_paths) / len(cited_paths)
        base *= 0.7 + 0.3 * overlap
        base = min(1.0, base * (1.0 + 0.04 * min(len(citations), 5)))

    if invalid:
        base *= max(0.4, 1.0 - 0.18 * invalid)

    if state.get("errors"):
        base *= 0.9

    # Never report certainty. The critic routinely returns 1.0, and an answer
    # built from a partial view of a codebase is never certain: retrieval may
    # simply have missed the file that contradicts it. Capping below 1.0 keeps
    # the number honest and preserves headroom for ranking answers.
    return round(max(0.0, min(_CONFIDENCE_CEILING, base)), 3)
