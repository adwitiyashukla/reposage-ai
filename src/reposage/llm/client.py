from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from reposage.config import Settings, get_settings
from reposage.llm.base import (
    ContentBlockedError,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    RateLimitError,
    TransientLLMError,
)
from reposage.llm.cache import ResponseCache
from reposage.llm.gemini import GeminiProvider, extract_json
from reposage.llm.pricing import estimate_cost
from reposage.llm.ratelimit import ConcurrencyGuard, TokenBucket
from reposage.logging_setup import get_logger
from reposage.models import Usage
from reposage.observability import current_tracer

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelTier(str, Enum):
    FAST = "fast"
    DEEP = "deep"


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider | None = None,
        settings: Settings | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider: LLMProvider = provider or GeminiProvider(
            self.settings.gemini_api_key, timeout=self.settings.request_timeout
        )
        self.cache = cache or ResponseCache(
            self.settings.cache_dir,
            ttl_seconds=self.settings.cache_ttl_seconds,
            enabled=self.settings.enable_cache,
        )
        self._bucket = TokenBucket(self.settings.max_rpm)
        self._embed_bucket = TokenBucket(self.settings.embed_rpm)
        self._guard = ConcurrencyGuard(self.settings.max_concurrency)
        self._embed_guard = ConcurrencyGuard(self.settings.embed_concurrency)
        self.usage = Usage()

    def model_for(self, tier: ModelTier) -> str:
        return self.settings.deep_model if tier is ModelTier.DEEP else self.settings.fast_model

    def _record(self, response: LLMResponse, *, cached: bool = False) -> None:
        cost = (
            0.0
            if cached
            else estimate_cost(response.model, response.prompt_tokens, response.completion_tokens)
        )
        usage = Usage(
            prompt_tokens=0 if cached else response.prompt_tokens,
            completion_tokens=0 if cached else response.completion_tokens,
            llm_calls=0 if cached else 1,
            cache_hits=1 if cached else 0,
            cost_usd=cost,
        )
        self.usage = self.usage.merge(usage)
        current_tracer().record_usage(usage)

    async def _retrying(
        self,
        operation: str,
        func: Any,
        *args: Any,
        _bucket: TokenBucket | None = None,
        _guard: ConcurrencyGuard | None = None,
        _cost: int = 1,
        **kwargs: Any,
    ) -> Any:
        attempts = max(1, self.settings.max_retries)
        bucket = _bucket or self._bucket
        guard = _guard or self._guard
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                await bucket.acquire(_cost)
                async with guard:
                    return await func(*args, **kwargs)
            except RateLimitError as exc:
                last = exc
                last_resort = min(90.0, 4.0 * (2**attempt))
                delay = (exc.retry_after or last_resort) * (0.5 + random.random() * 0.5)
                log.warning(
                    "llm.rate_limited",
                    operation=operation,
                    attempt=attempt + 1,
                    sleep=round(delay, 2),
                )
            except TransientLLMError as exc:
                last = exc
                delay = min(20.0, 0.75 * (2**attempt)) * (0.5 + random.random() * 0.5)
                log.warning(
                    "llm.transient_error",
                    operation=operation,
                    attempt=attempt + 1,
                    error=str(exc)[:200],
                    sleep=round(delay, 2),
                )
            if attempt < attempts - 1:
                await asyncio.sleep(delay)
        raise LLMError(f"{operation} failed after {attempts} attempts: {last}") from last

    async def complete(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.FAST,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        json_mode: bool = False,
        history: list[Message] | None = None,
        cacheable: bool = True,
    ) -> LLMResponse:
        model = self.model_for(tier)
        key = ResponseCache.generation_key(
            provider=self.provider.name,
            model=model,
            prompt=prompt,
            system=system or "",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
            history=[(m.role, m.content) for m in (history or [])],
        )
        if cacheable and (hit := self.cache.get(key)) is not None:
            response = LLMResponse(**hit)
            response.cached = True
            self._record(response, cached=True)
            current_tracer().log("llm.cache_hit", model=model, chars=len(response.text))
            return response

        started = time.perf_counter()
        with current_tracer().span("llm.generate", model=model, tier=tier.value) as span:
            response: LLMResponse = await self._retrying(
                "generate",
                self.provider.generate,
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                json_mode=json_mode,
                history=history,
            )
            response.latency_ms = (time.perf_counter() - started) * 1000
            span.set(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=round(response.latency_ms, 1),
                finish_reason=response.finish_reason,
            )
        self._record(response)
        if cacheable:
            payload = {
                "text": response.text,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "finish_reason": response.finish_reason,
            }
            self.cache.set(key, payload)
        return response

    async def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        tier: ModelTier = ModelTier.FAST,
        system: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
        repair_attempts: int = 1,
    ) -> T:
        instruction = (
            f"{prompt}\n\n"
            "Respond with a single JSON object and nothing else. No prose, no markdown fences.\n"
            f"It must conform to this JSON schema:\n{_compact_schema(schema)}"
        )
        response = await self.complete(
            instruction,
            tier=tier,
            system=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=True,
        )
        text = response.text
        for attempt in range(repair_attempts + 1):
            try:
                return schema.model_validate(extract_json(text))
            except (ValidationError, ValueError) as exc:
                if attempt >= repair_attempts:
                    current_tracer().log("llm.structured_failed", schema=schema.__name__)
                    raise LLMError(
                        f"{schema.__name__} could not be parsed after {attempt + 1} attempts: {exc}"
                    ) from exc
                current_tracer().log("llm.structured_repair", schema=schema.__name__)
                repair = await self.complete(
                    "Your previous response was not valid for the required schema.\n\n"
                    f"Previous response:\n{text[:4000]}\n\n"
                    f"Validation error:\n{str(exc)[:1500]}\n\n"
                    f"Required schema:\n{_compact_schema(schema)}\n\n"
                    "Return only the corrected JSON object.",
                    tier=tier,
                    temperature=0.0,
                    json_mode=True,
                    max_output_tokens=max_output_tokens,
                )
                text = repair.text
        raise LLMError("unreachable")

    async def stream(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.DEEP,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        model = self.model_for(tier)
        tracer = current_tracer()
        emitted = 0
        with tracer.span("llm.stream", model=model, tier=tier.value) as span:
            await self._bucket.acquire()
            async with self._guard:
                async for piece in self.provider.stream(
                    prompt,
                    model=model,
                    system=system,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ):
                    emitted += len(piece)
                    yield piece
            span.set(chars=emitted)
        approx_completion = max(1, emitted // 4)
        approx_prompt = max(1, len(prompt) // 4)
        self._record(
            LLMResponse(
                text="",
                model=model,
                prompt_tokens=approx_prompt,
                completion_tokens=approx_completion,
            )
        )

    async def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        dimensions: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        model = self.settings.embed_model
        if dimensions is None:
            dimensions = self.settings.embed_dimensions
        results: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = ResponseCache.embedding_key(model, f"{task_type}:{dimensions}", text)
            hit = self.cache.get(key)
            if hit is not None:
                results[i] = hit
            else:
                pending.append((i, text))

        if pending:
            batch_size = max(1, self.settings.embed_batch_size)
            with current_tracer().span(
                "llm.embed", model=model, texts=len(texts), misses=len(pending)
            ) as span:
                for start in range(0, len(pending), batch_size):
                    batch = pending[start : start + batch_size]
                    vectors = await self._retrying(
                        "embed",
                        self.provider.embed,
                        [t for _, t in batch],
                        _bucket=self._embed_bucket,
                        _guard=self._embed_guard,
                        _cost=len(batch),
                        model=model,
                        task_type=task_type,
                        dimensions=dimensions,
                    )
                    if len(vectors) != len(batch):
                        raise LLMError(
                            f"embedding count mismatch: got {len(vectors)}, expected {len(batch)}"
                        )
                    for (index, text), vector in zip(batch, vectors, strict=True):
                        results[index] = vector
                        self.cache.set(
                            ResponseCache.embedding_key(model, f"{task_type}:{dimensions}", text),
                            vector,
                        )
                span.set(batches=(len(pending) + batch_size - 1) // batch_size)

            approx_tokens = sum(len(t) // 4 for _, t in pending)
            usage = Usage(
                embedding_tokens=approx_tokens,
                llm_calls=1,
                cost_usd=estimate_cost(model, approx_tokens, 0),
            )
            self.usage = self.usage.merge(usage)
            current_tracer().record_usage(usage)

        hits = len(texts) - len(pending)
        if hits:
            current_tracer().log("embed.cache_hits", hits=hits, total=len(texts))
        return [r if r is not None else [] for r in results]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed([text], task_type="RETRIEVAL_QUERY")
        return vectors[0] if vectors else []

    async def healthcheck(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = await self.complete(
                "Reply with exactly: ok", tier=ModelTier.FAST, max_output_tokens=8, cacheable=False
            )
            return {
                "ok": True,
                "model": response.model,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "reply": response.text.strip()[:40],
            }
        except (LLMError, ContentBlockedError) as exc:
            return {"ok": False, "error": str(exc)[:300]}

    def stats(self) -> dict[str, Any]:
        return {
            "usage": self.usage.model_dump(),
            "cache": self.cache.stats(),
            "peak_concurrency": self._guard.peak,
            "models": {"fast": self.settings.fast_model, "deep": self.settings.deep_model},
        }

    async def aclose(self) -> None:
        await self.provider.aclose()
        self.cache.close()


def _compact_schema(schema: type[BaseModel]) -> str:
    import orjson

    raw = schema.model_json_schema()
    raw.pop("title", None)
    raw.pop("$defs", None) if not raw.get("$defs") else None
    return orjson.dumps(raw).decode()


_CLIENT: LLMClient | None = None


def get_client(settings: Settings | None = None) -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient(settings=settings)
    return _CLIENT


async def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
        _CLIENT = None
