from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any

import orjson

from reposage.logging_setup import get_logger

log = get_logger(__name__)

try:
    from diskcache import Cache as _DiskCache

    _HAS_DISKCACHE = True
except ImportError:
    _DiskCache = None
    _HAS_DISKCACHE = False


def _stable_hash(payload: dict[str, Any]) -> str:
    blob = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(blob).hexdigest()


class _MemoryCache:
    def __init__(self, max_items: int = 4096) -> None:
        self._data: dict[str, Any] = {}
        self._max = max_items

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any, expire: float | None = None) -> None:
        if len(self._data) >= self._max:
            self._data.pop(next(iter(self._data)))
        self._data[key] = value

    def close(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class ResponseCache:
    def __init__(self, directory: Path, ttl_seconds: int = 604_800, enabled: bool = True) -> None:
        self.enabled = enabled
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
        if not enabled:
            self._backend: Any = _MemoryCache(max_items=1)
            return
        if _HAS_DISKCACHE:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self._backend = _DiskCache(str(directory), size_limit=2 * 1024**3)
                return
            except Exception as exc:
                log.warning("cache.disk_unavailable", error=str(exc))
        self._backend = _MemoryCache()

    @staticmethod
    def generation_key(**parts: Any) -> str:
        return "gen:" + _stable_hash(parts)

    @staticmethod
    def embedding_key(model: str, task_type: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        return f"emb:{model}:{task_type}:{digest}"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        value = self._backend.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            self._backend.set(key, value, expire=self.ttl)
        except Exception as exc:
            log.debug("cache.write_failed", error=str(exc))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "backend": "disk" if _HAS_DISKCACHE and self.enabled else "memory",
        }

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._backend.close()
