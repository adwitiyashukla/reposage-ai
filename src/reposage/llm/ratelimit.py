from __future__ import annotations

import asyncio
import time


class TokenBucket:
    def __init__(self, rate_per_minute: int, burst: int | None = None) -> None:
        self.rate_per_minute = rate_per_minute
        self.capacity = burst if burst is not None else max(1, rate_per_minute // 5)
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate_per_minute > 0

    async def acquire(self, amount: int = 1) -> float:
        if not self.enabled:
            return 0.0
        amount = min(amount, self.capacity)
        waited = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                refill = (now - self._updated) * (self.rate_per_minute / 60.0)
                self._tokens = min(self.capacity, self._tokens + refill)
                self._updated = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                deficit = amount - self._tokens
                delay = deficit / (self.rate_per_minute / 60.0)
            delay = min(max(delay, 0.01), 30.0)
            waited += delay
            await asyncio.sleep(delay)


class ConcurrencyGuard:
    def __init__(self, limit: int) -> None:
        self._sem = asyncio.Semaphore(max(1, limit))
        self.in_flight = 0
        self.peak = 0

    async def __aenter__(self) -> ConcurrencyGuard:
        await self._sem.acquire()
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.in_flight -= 1
        self._sem.release()
