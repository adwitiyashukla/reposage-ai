"""Provider-agnostic contracts.

Everything above this module talks to :class:`LLMProvider`, never to a vendor
SDK. Swapping Gemini for Groq, OpenRouter or a local Ollama server is a
configuration change, not a code change.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Base class for provider failures."""


class TransientLLMError(LLMError):
    """Retryable failure: timeout, 5xx, connection reset."""


class RateLimitError(TransientLLMError):
    """429 from the provider. Retried with a longer backoff."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ContentBlockedError(LLMError):
    """The provider refused to answer. Not retryable."""


@dataclass(slots=True)
class Message:
    role: str  # "user" | "model"
    content: str


@dataclass(slots=True)
class LLMResponse:
    """A completed generation with usage attached."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    cached: bool = False
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal surface every provider must implement."""

    name: str

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        json_mode: bool = False,
        history: list[Message] | None = None,
    ) -> LLMResponse: ...

    async def stream(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        history: list[Message] | None = None,
    ) -> AsyncIterator[str]: ...

    async def embed(
        self,
        texts: list[str],
        *,
        model: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        dimensions: int | None = None,
    ) -> list[list[float]]: ...

    async def aclose(self) -> None: ...
