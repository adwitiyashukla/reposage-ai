"""Shared application state.

Indexes are expensive to load (a NumPy matrix plus BM25 postings) and entirely
read-only once built, so they are cached per process behind a lock. Without the
lock two concurrent first-requests for the same repository would each pay the
full load cost.
"""

from __future__ import annotations

import asyncio
from typing import Any

from reposage.agents.engine import CodebaseAgent
from reposage.config import Settings, get_settings
from reposage.index.store import RepoIndex, list_indexes
from reposage.llm.client import LLMClient, get_client
from reposage.logging_setup import get_logger

log = get_logger(__name__)


class AppState:
    """Process-wide resources shared across requests."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: LLMClient | None = None
        self._indexes: dict[str, RepoIndex] = {}
        self._agents: dict[str, CodebaseAgent] = {}
        self._lock = asyncio.Lock()

    @property
    def client(self) -> LLMClient:
        """Construct the model client on first use.

        Deferring this lets the server start without credentials and serve the
        UI, the docs and ``/api/health`` in a degraded state, which is far more
        useful to a first-time user than a crash on boot.
        """
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

    def register(self, index: RepoIndex) -> None:
        """Make a freshly built index available without a reload."""
        self._indexes[index.index_id] = index
        self._agents.pop(index.index_id, None)

    def invalidate(self, name: str) -> None:
        self._indexes.pop(name, None)
        self._agents.pop(name, None)

    def catalogue(self) -> list[dict[str, Any]]:
        return list_indexes(self.settings)

    async def aclose(self) -> None:
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
