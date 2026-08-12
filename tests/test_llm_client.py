from __future__ import annotations

import httpx
import orjson
import pytest
from pydantic import BaseModel

from reposage.llm.base import LLMError, RateLimitError, TransientLLMError
from reposage.llm.cache import ResponseCache
from reposage.llm.client import LLMClient, ModelTier
from reposage.llm.gemini import GeminiProvider, extract_json
from reposage.llm.pricing import estimate_cost, lookup
from reposage.llm.ratelimit import TokenBucket


def _gemini_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
        },
    }


class TestGeminiProvider:
    async def test_generate_parses_text_and_usage(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = orjson.loads(request.content)
            assert body["contents"][0]["parts"][0]["text"] == "hello"
            assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
            return httpx.Response(200, json=_gemini_response("hi", 12, 3))

        provider = GeminiProvider("k", transport=httpx.MockTransport(handler))
        response = await provider.generate("hello", model="gemini-2.0-flash", system="be terse")
        assert response.text == "hi"
        assert response.prompt_tokens == 12 and response.completion_tokens == 3
        await provider.aclose()

    async def test_json_mode_sets_the_response_mime_type(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = orjson.loads(request.content)
            assert body["generationConfig"]["responseMimeType"] == "application/json"
            return httpx.Response(200, json=_gemini_response("{}"))

        provider = GeminiProvider("k", transport=httpx.MockTransport(handler))
        await provider.generate("x", model="m", json_mode=True)
        await provider.aclose()

    async def test_rate_limit_raises_a_retryable_error(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(429, text="quota", headers={"retry-after": "2"})
        )
        provider = GeminiProvider("k", transport=transport)
        with pytest.raises(RateLimitError) as info:
            await provider.generate("x", model="m")
        assert info.value.retry_after == 2.0
        await provider.aclose()

    async def test_server_errors_are_retryable(self):
        provider = GeminiProvider(
            "k", transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
        )
        with pytest.raises(TransientLLMError):
            await provider.generate("x", model="m")
        await provider.aclose()

    async def test_bad_api_key_is_not_retryable(self):
        provider = GeminiProvider(
            "k",
            transport=httpx.MockTransport(lambda r: httpx.Response(400, text="API key not valid")),
        )
        with pytest.raises(LLMError, match="API key"):
            await provider.generate("x", model="m")
        await provider.aclose()

    async def test_embeddings_are_batched_into_one_request(self):
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests = orjson.loads(request.content)["requests"]
            seen.append(len(requests))
            return httpx.Response(
                200, json={"embeddings": [{"values": [0.1, 0.2]} for _ in requests]}
            )

        provider = GeminiProvider("k", transport=httpx.MockTransport(handler))
        vectors = await provider.embed(["a", "b", "c"], model="text-embedding-004")
        assert seen == [3] and len(vectors) == 3
        await provider.aclose()

    def test_missing_key_fails_fast_with_a_useful_message(self):
        with pytest.raises(LLMError, match=r"aistudio\.google\.com"):
            GeminiProvider("")


class TestJSONRecovery:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('Sure!\n```json\n{"a": 2}\n```\nDone.') == {"a": 2}

    def test_json_embedded_in_prose(self):
        assert extract_json('Here you go: {"a": 3} hope that helps') == {"a": 3}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"a": "}{"}') == {"a": "}{"}

    def test_arrays(self):
        assert extract_json("prefix [1, 2, 3] suffix") == [1, 2, 3]

    def test_unrecoverable_input_raises(self):
        with pytest.raises(ValueError):
            extract_json("no json here at all")


class _Shape(BaseModel):
    name: str
    count: int


class TestClientBehaviour:
    async def test_retries_then_succeeds(self, settings):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, json=_gemini_response("recovered"))

        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider("k", transport=httpx.MockTransport(handler)),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        response = await client.complete("x", tier=ModelTier.FAST)
        assert response.text == "recovered" and attempts["n"] == 3
        await client.aclose()

    async def test_gives_up_after_the_retry_budget(self, settings):
        settings.max_retries = 2
        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider(
                "k", transport=httpx.MockTransport(lambda r: httpx.Response(503))
            ),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        with pytest.raises(LLMError, match="after 2 attempts"):
            await client.complete("x")
        await client.aclose()

    async def test_cache_prevents_a_second_request(self, settings, tmp_path):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=_gemini_response("cached body"))

        settings.enable_cache = True
        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider("k", transport=httpx.MockTransport(handler)),
            settings=settings,
            cache=ResponseCache(tmp_path / "cache", enabled=True),
        )
        first = await client.complete("same prompt")
        second = await client.complete("same prompt")
        assert calls["n"] == 1
        assert second.cached and second.text == first.text
        assert client.usage.cache_hits == 1
        await client.aclose()

    async def test_structured_output_validates_into_the_model(self, settings):
        payload = orjson.dumps({"name": "widget", "count": 3}).decode()
        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider(
                "k",
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, json=_gemini_response(payload))
                ),
            ),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        result = await client.structured("give me a shape", _Shape)
        assert result.name == "widget" and result.count == 3
        await client.aclose()

    async def test_structured_output_repairs_malformed_json(self, settings):
        replies = ["this is not json", orjson.dumps({"name": "fixed", "count": 1}).decode()]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_gemini_response(replies.pop(0)))

        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider("k", transport=httpx.MockTransport(handler)),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        result = await client.structured("shape", _Shape, repair_attempts=1)
        assert result.name == "fixed" and not replies
        await client.aclose()

    async def test_embeddings_reuse_the_cache_for_repeated_text(self, settings, tmp_path):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            requests = orjson.loads(request.content)["requests"]
            calls["n"] += len(requests)
            return httpx.Response(200, json={"embeddings": [{"values": [0.5]} for _ in requests]})

        settings.enable_cache = True
        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider("k", transport=httpx.MockTransport(handler)),
            settings=settings,
            cache=ResponseCache(tmp_path / "embcache", enabled=True),
        )
        await client.embed(["alpha", "beta"])
        await client.embed(["alpha", "gamma"])
        assert calls["n"] == 3
        await client.aclose()


class TestPricing:
    def test_longest_prefix_match_wins(self):
        assert lookup("gemini-2.0-flash-lite-001").input_per_mtok == 0.075
        assert lookup("gemini-2.0-flash-001").input_per_mtok == 0.10

    def test_unknown_models_fall_back(self):
        assert lookup("some-future-model").note.startswith("unknown")

    def test_cost_scales_with_tokens(self):
        assert estimate_cost("gemini-2.0-flash", 1_000_000, 0) == pytest.approx(0.10)
        assert estimate_cost("gemini-2.0-flash", 0, 0) == 0.0


class TestRateLimiting:
    async def test_disabled_bucket_never_waits(self):
        assert await TokenBucket(0).acquire() == 0.0

    async def test_bucket_throttles_once_the_burst_is_spent(self):
        bucket = TokenBucket(rate_per_minute=6000, burst=2)
        assert await bucket.acquire() == 0.0
        assert await bucket.acquire() == 0.0
        assert await bucket.acquire() > 0.0


class TestQuotaShaping:
    async def test_a_cost_larger_than_capacity_is_clamped(self):
        bucket = TokenBucket(rate_per_minute=10, burst=4)
        assert await bucket.acquire(999) == 0.0

    async def test_cost_is_deducted_in_full(self):
        bucket = TokenBucket(rate_per_minute=6000, burst=10)
        await bucket.acquire(8)
        assert await bucket.acquire(2) == 0.0
        assert await bucket.acquire(1) > 0.0

    async def test_embedding_uses_its_own_bucket(self, settings):
        settings.max_rpm = 6000
        settings.embed_rpm = 6000
        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider(
                "k",
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, json={"embeddings": [{"values": [0.1]}] * 3})
                ),
            ),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        assert client._bucket is not client._embed_bucket
        await client.embed(["a", "b", "c"])
        await client.aclose()

    async def test_default_burst_stays_inside_a_rolling_window(self):
        bucket = TokenBucket(rate_per_minute=100)
        assert bucket.capacity == 20
        assert bucket.capacity + bucket.rate_per_minute <= 120

    async def test_rate_limit_backoff_can_outlast_a_full_window(self, settings):
        settings.max_retries = 6
        delays: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider(
                "k", transport=httpx.MockTransport(lambda r: httpx.Response(429, text="quota"))
            ),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        import asyncio as _asyncio

        original, _asyncio.sleep = _asyncio.sleep, fake_sleep
        try:
            with pytest.raises(LLMError):
                await client.complete("x")
        finally:
            _asyncio.sleep = original
        await client.aclose()
        assert sum(delays) >= 60.0, f"total backoff {sum(delays):.1f}s cannot span a quota window"

    async def test_embedding_is_serialised(self, settings):
        import asyncio as _asyncio

        settings.embed_concurrency = 1
        settings.embed_batch_size = 1
        settings.embed_rpm = 0
        settings.ensure_dirs()

        in_flight = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await _asyncio.sleep(0.01)
            in_flight -= 1
            return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]})

        client = LLMClient(
            provider=GeminiProvider("k", transport=httpx.MockTransport(handler)),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        await client.embed([f"chunk {i}" for i in range(6)])
        await client.aclose()
        assert peak == 1, f"expected serialised embedding, saw {peak} concurrent requests"

    async def test_generation_concurrency_is_unaffected(self, settings):
        settings.embed_concurrency = 1
        settings.max_concurrency = 8
        settings.ensure_dirs()
        client = LLMClient(
            provider=GeminiProvider(
                "k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
            ),
            settings=settings,
            cache=ResponseCache(settings.cache_dir, enabled=False),
        )
        assert client._guard is not client._embed_guard
        await client.aclose()
