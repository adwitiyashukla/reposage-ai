"""Lightweight, dependency-free tracing for agent runs.

Every meaningful operation opens a :class:`Span`. Spans nest, carry arbitrary
attributes, and record token usage, so a completed run yields both a waterfall
timeline and an exact cost breakdown. Subscribers can attach to a tracer to
receive events as they happen, which is what powers the live UI stream.

This is deliberately not OpenTelemetry: OTel would add a heavy dependency and a
collector to run, and we only need in-process, single-run visibility. The event
schema is intentionally OTel-shaped so exporting later is a small change.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from reposage.models import Usage


class EventType(str, Enum):
    SPAN_START = "span_start"
    SPAN_END = "span_end"
    LOG = "log"
    TOKEN = "token"
    ERROR = "error"
    RESULT = "result"


@dataclass(slots=True)
class TraceEvent:
    """A single point-in-time record in a run's timeline."""

    type: EventType
    name: str
    timestamp: float = field(default_factory=time.time)
    span_id: str = ""
    parent_id: str | None = None
    depth: int = 0
    duration_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "timestamp": self.timestamp,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """A timed unit of work."""

    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    depth: int = 0
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000.0

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


class Tracer:
    """Collects spans and usage for one logical run.

    A tracer is cheap: create one per request. Attach subscribers to stream
    events to a websocket or SSE channel while the run is still in flight.
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[TraceEvent] = []
        self.spans: list[Span] = []
        self.usage = Usage()
        self._stack: list[Span] = []
        self._subscribers: list[asyncio.Queue] = []
        self._started = time.perf_counter()

    # ------------------------------------------------------------- streaming
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _emit(self, event: TraceEvent) -> None:
        self.events.append(event)
        for queue in list(self._subscribers):
            # A subscriber that cannot keep up loses events rather than
            # blocking the run that is producing them.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    # ----------------------------------------------------------------- spans
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        span = Span(
            name=name,
            parent_id=parent.span_id if parent else None,
            depth=len(self._stack),
            attributes=dict(attributes),
        )
        self._stack.append(span)
        self.spans.append(span)
        self._emit(
            TraceEvent(
                type=EventType.SPAN_START,
                name=name,
                span_id=span.span_id,
                parent_id=span.parent_id,
                depth=span.depth,
                attributes=dict(attributes),
            )
        )
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.set(error=f"{type(exc).__name__}: {exc}")
            self._emit(
                TraceEvent(
                    type=EventType.ERROR,
                    name=name,
                    span_id=span.span_id,
                    parent_id=span.parent_id,
                    depth=span.depth,
                    attributes={"error": str(exc), "error_type": type(exc).__name__},
                )
            )
            raise
        finally:
            span.end = time.perf_counter()
            self._stack.pop()
            self._emit(
                TraceEvent(
                    type=EventType.SPAN_END,
                    name=name,
                    span_id=span.span_id,
                    parent_id=span.parent_id,
                    depth=span.depth,
                    duration_ms=span.duration_ms,
                    attributes={**span.attributes, "status": span.status},
                )
            )

    # ------------------------------------------------------------- reporting
    def log(self, message: str, **attributes: Any) -> None:
        parent = self._stack[-1] if self._stack else None
        self._emit(
            TraceEvent(
                type=EventType.LOG,
                name=message,
                span_id=parent.span_id if parent else "",
                parent_id=parent.parent_id if parent else None,
                depth=len(self._stack),
                attributes=attributes,
            )
        )

    def token(self, text: str) -> None:
        """Stream a partial generation token to subscribers."""
        self._emit(TraceEvent(type=EventType.TOKEN, name="token", attributes={"text": text}))

    def result(self, payload: dict[str, Any]) -> None:
        self._emit(TraceEvent(type=EventType.RESULT, name="result", attributes=payload))

    def record_usage(self, usage: Usage) -> None:
        self.usage = self.usage.merge(usage)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._started

    def waterfall(self) -> list[dict[str, Any]]:
        """Flat, ordered view of the run suitable for rendering a timeline."""
        return [
            {
                "name": s.name,
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "depth": s.depth,
                "duration_ms": round(s.duration_ms, 2),
                "status": s.status,
                "attributes": s.attributes,
            }
            for s in self.spans
        ]

    def summary(self) -> dict[str, Any]:
        by_name: dict[str, dict[str, float]] = {}
        for s in self.spans:
            entry = by_name.setdefault(s.name, {"count": 0, "total_ms": 0.0})
            entry["count"] += 1
            entry["total_ms"] += s.duration_ms
        return {
            "run_id": self.run_id,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "spans": len(self.spans),
            "usage": self.usage.model_dump(),
            "by_operation": {
                k: {"count": int(v["count"]), "total_ms": round(v["total_ms"], 1)}
                for k, v in sorted(by_name.items(), key=lambda kv: -kv[1]["total_ms"])
            },
        }


_NULL = Tracer(run_id="null")
_current: contextvars.ContextVar[Tracer] = contextvars.ContextVar("reposage_tracer", default=_NULL)


def current_tracer() -> Tracer:
    """The tracer bound to the current execution context."""
    return _current.get()


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    """Bind ``tracer`` for the duration of the block."""
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)
