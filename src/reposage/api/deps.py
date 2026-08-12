from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from reposage.agents.engine import CodebaseAgent
from reposage.api.demo import DemoBudget
from reposage.config import Settings, get_settings
from reposage.index.store import RepoIndex, list_indexes
from reposage.llm.client import LLMClient, get_client
from reposage.llm.gemini import GeminiProvider
from reposage.logging_setup import get_logger

log = get_logger(__name__)


class AppState:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: LLMClient | None = None
        self._indexes: dict[str, RepoIndex] = {}
        self._agents: dict[str, CodebaseAgent] = {}
        self._lock = asyncio.Lock()
        self.budget = DemoBudget(
            daily_limit=self.settings.demo_daily_budget,
            visitor_limit=self.settings.demo_visitor_budget,
        )
        self._byo: dict[str, LLMClient] = {}

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client(self.settings)
        return self._client

    async def get_index(self, name: str) -> RepoIndex:
        async with self._lock:
            if name not in self._indexes:
                log.info("state.loading_index", name=name)
                self._indexes[name] = await asyncio.to_thread(
                    RepoIndex.load_by_name, name, self.settings
                )
            return self._indexes[name]

    async def get_agent(self, name: str) -> CodebaseAgent:
        index = await self.get_index(name)
        async with self._lock:
            if name not in self._agents:
                self._agents[name] = CodebaseAgent(index, self.client, self.settings)
            return self._agents[name]

    def client_for_key(self, api_key: str) -> LLMClient:
        digest = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        if digest not in self._byo:
            if len(self._byo) >= 32:
                self._byo.pop(next(iter(self._byo)))
            self._byo[digest] = LLMClient(
                provider=GeminiProvider(api_key, timeout=self.settings.request_timeout),
                settings=self.settings,
            )
        return self._byo[digest]

    def register(self, index: RepoIndex) -> None:
        self._indexes[index.index_id] = index
        self._agents.pop(index.index_id, None)

    def invalidate(self, name: str) -> None:
        self._indexes.pop(name, None)
        self._agents.pop(name, None)

    def catalogue(self) -> list[dict[str, Any]]:
        return list_indexes(self.settings)

    async def aclose(self) -> None:
        for client in self._byo.values():
            await client.aclose()
        self._byo.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_STATE: AppState | None = None


def get_state() -> AppState:
    global _STATE
    if _STATE is None:
        _STATE = AppState()
    return _STATE


async def reset_state() -> None:
    global _STATE
    if _STATE is not None:
        await _STATE.aclose()
        _STATE = None
