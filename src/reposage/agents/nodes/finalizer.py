"""Finalisation node: extract and validate citations, score confidence."""

from __future__ import annotations

import re

from reposage.agents.state import AgentDeps, AgentState
from reposage.logging_setup import get_logger
from reposage.models import Citation
from reposage.observability import current_tracer

log = get_logger(__name__)

# Models group references far more often than the prompt asks them to, and the
# grouped forms are the ones that carry the most evidence:
#
#   [src/auth/jwt.py:12-48]                          one reference
#   [src/auth/jwt.py:12]                             a single line
#   [src/auth/jwt.py:12-48, src/api.py:3-9]          two files at once
#   [src/auth/jwt.py:12-48, 90-104]                  two ranges in the same file
#
# Matching only the first form discarded roughly half the citations in practice,
# which then read as a low-evidence answer and depressed the confidence score.
# So brackets are located first, then split on commas, with a bare range
# inheriting the path from the reference before it.
_BRACKET = re.compile(r"\[([^\[\]]{3,400}?)\]")
_REF = re.compile(r"^\s*([^\s:]+?\.[A-Za-z0-9_]+):(\d+)(?:\s*[-\u2013]\s*(\d+))?\s*$")
_BARE_RANGE = re.compile(r"^\s*(\d+)(?:\s*[-\u2013]\s*(\d+))?\s*$")


def parse_citation_markers(text: str) -> list[tuple[str, int, int]]:
    """Extract ``(path, start, end)`` triples from inline citation markers."""
    found: list[tuple[str, int, int]] = []
    for bracket in _BRACKET.finditer(text or ""):
        last_path: str | None = None
        parts = bracket.group(1).split(",")
        for part in parts:
            if match := _REF.match(part):
                path, start_raw, end_raw = match.groups()
                last_path = path
            elif last_path and (match := _BARE_RANGE.match(part)):
                start_raw, end_raw = match.groups()
                path = last_path
            else:
                # A bracket holding prose is not a citation; abandon the group
                # so "[see below, line 4]" cannot inherit an unrelated path.
                last_path = None
                continue
            start = int(start_raw)
            end = int(end_raw) if end_raw else start
            found.append((path, min(start, end), max(start, end)))
    return found


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

        for path, start, end in parse_citation_markers(draft):
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
        # Proportional to how much of the evidence failed, not to the raw count:
        # a long answer with twenty good citations and two bad ones is not
        # meaningfully less trustworthy than one with ten and none.
        share_bad = invalid / max(1, invalid + len(citations))
        base *= max(0.55, 1.0 - 0.6 * share_bad)

    if state.get("errors"):
        base *= 0.9

    # Never report certainty. The critic routinely returns 1.0, and an answer
    # built from a partial view of a codebase is never certain: retrieval may
    # simply have missed the file that contradicts it. Capping below 1.0 keeps
    # the number honest and preserves headroom for ranking answers.
    return round(max(0.0, min(_CONFIDENCE_CEILING, base)), 3)
