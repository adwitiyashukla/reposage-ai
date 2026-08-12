from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    pass


class TransientLLMError(LLMError):
    pass


class RateLimitError(TransientLLMError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ContentBlockedError(LLMError):
    pass


@dataclass(slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class LLMResponse:
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
