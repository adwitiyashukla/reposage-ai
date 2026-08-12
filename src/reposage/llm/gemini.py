from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
import orjson

from reposage.llm.base import (
    ContentBlockedError,
    LLMError,
    LLMResponse,
    Message,
    RateLimitError,
    TransientLLMError,
)
from reposage.logging_setup import get_logger

log = get_logger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

_SAFETY_OFF = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 120.0,
        max_connections: int = 16,
        api_root: str = API_ROOT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Create a free key at "
                "https://aistudio.google.com/apikey and add it to your .env file."
            )
        self.api_root = api_root.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=8),
            headers={
                "x-goog-api-key": api_key,
                "content-type": "application/json",
                "user-agent": "reposage/1.0",
            },
            transport=transport,
        )

    @staticmethod
    def _model_path(model: str) -> str:
        return model if model.startswith("models/") else f"models/{model}"

    @staticmethod
    def _build_contents(prompt: str, history: list[Message] | None) -> list[dict[str, Any]]:
        contents = [{"role": m.role, "parts": [{"text": m.content}]} for m in (history or [])]
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        return contents

    def _build_body(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_output_tokens: int,
        json_mode: bool,
        history: list[Message] | None,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.95,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        body: dict[str, Any] = {
            "contents": self._build_contents(prompt, history),
            "generationConfig": generation_config,
            "safetySettings": _SAFETY_OFF,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @staticmethod
    def _raise_for_status(response: httpx.Response, body: str) -> None:
        status = response.status_code
        if status == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                f"Gemini rate limit hit: {body[:300]}",
                retry_after=float(retry_after) if retry_after else None,
            )
        if status in (408, 500, 502, 503, 504):
            raise TransientLLMError(f"Gemini {status}: {body[:300]}")
        if status == 400 and "API key not valid" in body:
            raise LLMError("Gemini rejected the API key. Check GEMINI_API_KEY in your .env file.")
        if status >= 400:
            raise LLMError(f"Gemini {status}: {body[:500]}")

    @staticmethod
    def _extract(payload: dict[str, Any], model: str) -> LLMResponse:
        candidates = payload.get("candidates") or []
        if not candidates:
            feedback = payload.get("promptFeedback", {})
            raise ContentBlockedError(f"Gemini returned no candidates. Feedback: {feedback}")
        candidate = candidates[0]
        finish = candidate.get("finishReason", "STOP")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text and finish in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT"):
            raise ContentBlockedError(f"Gemini stopped early: finishReason={finish}")
        usage = payload.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=int(usage.get("promptTokenCount", 0)),
            completion_tokens=int(usage.get("candidatesTokenCount", 0)),
            finish_reason=str(finish).lower(),
            raw=payload,
        )

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
    ) -> LLMResponse:
        url = f"{self.api_root}/{self._model_path(model)}:generateContent"
        body = self._build_body(prompt, system, temperature, max_output_tokens, json_mode, history)
        try:
            response = await self._client.post(url, content=orjson.dumps(body))
        except httpx.TimeoutException as exc:
            raise TransientLLMError(f"Gemini request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"Gemini transport error: {exc}") from exc
        text = response.text
        self._raise_for_status(response, text)
        return self._extract(orjson.loads(text), model)

    async def stream(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        history: list[Message] | None = None,
    ) -> AsyncIterator[str]:
        url = f"{self.api_root}/{self._model_path(model)}:streamGenerateContent?alt=sse"
        body = self._build_body(prompt, system, temperature, max_output_tokens, False, history)
        try:
            async with self._client.stream("POST", url, content=orjson.dumps(body)) as response:
                if response.status_code >= 400:
                    raw = (await response.aread()).decode("utf-8", errors="replace")
                    self._raise_for_status(response, raw)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        continue
                    for candidate in chunk.get("candidates", []):
                        for part in (candidate.get("content") or {}).get("parts", []):
                            if piece := part.get("text"):
                                yield piece
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"Gemini stream failed: {exc}") from exc

    async def embed(
        self,
        texts: list[str],
        *,
        model: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        dimensions: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        model_path = self._model_path(model)
        url = f"{self.api_root}/{model_path}:batchEmbedContents"
        requests = []
        for text in texts:
            request: dict[str, Any] = {
                "model": model_path,
                "content": {"parts": [{"text": text[:36_000]}]},
                "taskType": task_type,
            }
            if dimensions:
                request["outputDimensionality"] = dimensions
            requests.append(request)
        try:
            response = await self._client.post(url, content=orjson.dumps({"requests": requests}))
        except httpx.TimeoutException as exc:
            raise TransientLLMError(f"Embedding request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"Embedding transport error: {exc}") from exc
        raw = response.text
        self._raise_for_status(response, raw)
        payload = orjson.loads(raw)
        return [item.get("values", []) for item in payload.get("embeddings", [])]

    async def aclose(self) -> None:
        await self._client.aclose()


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        pass

    if match := _JSON_FENCE.search(text):
        try:
            return orjson.loads(match.group(1).strip())
        except orjson.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return orjson.loads(text[start : index + 1])
                    except orjson.JSONDecodeError:
                        break
    raise ValueError(f"could not parse JSON from response: {text[:200]!r}")


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
