"""Public demo guardrails.

A hosted demo runs on the maintainer's API key, which means anyone who finds the
URL can spend it. The free tier is finite, so an unguarded demo is one crawler
away from being permanently broken, which is worse than having no demo at all.

Three mechanisms, in order of how much they matter:

1. **A global daily budget.** The shared key answers a bounded number of
   questions per UTC day. This is the backstop that makes the worst case
   survivable.
2. **A per-visitor budget.** A sliding hourly window per visitor stops one
   person consuming the whole day in a minute, without needing accounts.
3. **Bring-your-own-key.** When the shared budget is spent the demo does not
   simply break: visitors can supply their own key and continue, and those
   requests bypass the budget entirely because they cost the host nothing.

State is in-process and deliberately so. A demo runs as a single container, and
adding Redis to rate-limit a portfolio demo would be architecture theatre.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from reposage.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Outcome of a budget check, including why it was refused."""

    allowed: bool
    reason: str = ""
    scope: str = ""  # "global" | "visitor"
    remaining_today: int = 0
    remaining_for_visitor: int = 0
    retry_after_seconds: int = 0

    @property
    def needs_own_key(self) -> bool:
        """A spent global budget is the case a visitor can fix with their key."""
        return not self.allowed and self.scope == "global"


class DemoBudget:
    """Global daily and per-visitor hourly limits for the shared key."""

    def __init__(
        self, daily_limit: int = 200, visitor_limit: int = 5, window_seconds: int = 3600
    ) -> None:
        self.daily_limit = daily_limit
        self.visitor_limit = visitor_limit
        self.window_seconds = window_seconds
        self._day = self._today()
        self._used_today = 0
        self._visitors: dict[str, deque[float]] = defaultdict(deque)
        self.total_served = 0
        self.total_refused = 0
        self.byo_key_requests = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            log.info("demo.budget_reset", previous_day=self._day, used=self._used_today)
            self._day = today
            self._used_today = 0

    def _prune(self, visitor: str, now: float) -> deque[float]:
        history = self._visitors[visitor]
        cutoff = now - self.window_seconds
        while history and history[0] < cutoff:
            history.popleft()
        return history

    # ------------------------------------------------------------------ API
    def check(self, visitor: str) -> BudgetDecision:
        """Test whether ``visitor`` may spend one unit of the shared budget."""
        self._roll_day()
        now = time.time()
        history = self._prune(visitor, now)

        if len(history) >= self.visitor_limit:
            oldest = history[0]
            retry = max(1, int(self.window_seconds - (now - oldest)))
            self.total_refused += 1
            return BudgetDecision(
                allowed=False,
                scope="visitor",
                reason=(
                    f"You have used this demo's limit of {self.visitor_limit} questions "
                    f"per hour. It resets shortly, or you can supply your own API key."
                ),
                remaining_today=max(0, self.daily_limit - self._used_today),
                remaining_for_visitor=0,
                retry_after_seconds=retry,
            )

        if self._used_today >= self.daily_limit:
            self.total_refused += 1
            return BudgetDecision(
                allowed=False,
                scope="global",
                reason=(
                    "The shared demo budget for today is spent. Add your own free "
                    "Gemini API key to keep exploring, or try again tomorrow."
                ),
                remaining_today=0,
                remaining_for_visitor=max(0, self.visitor_limit - len(history)),
                retry_after_seconds=_seconds_until_utc_midnight(),
            )

        return BudgetDecision(
            allowed=True,
            remaining_today=max(0, self.daily_limit - self._used_today),
            remaining_for_visitor=max(0, self.visitor_limit - len(history)),
        )

    def consume(self, visitor: str) -> None:
        """Record one answered question against the shared budget."""
        self._roll_day()
        self._used_today += 1
        self.total_served += 1
        self._visitors[visitor].append(time.time())
        if len(self._visitors) > 5000:  # bound memory on a long-running demo
            self._evict_stale()

    def record_own_key(self) -> None:
        """A request served on a visitor's own key costs the host nothing."""
        self.byo_key_requests += 1

    def _evict_stale(self) -> None:
        cutoff = time.time() - self.window_seconds
        stale = [k for k, v in self._visitors.items() if not v or v[-1] < cutoff]
        for key in stale:
            self._visitors.pop(key, None)

    def status(self) -> dict[str, object]:
        self._roll_day()
        return {
            "daily_limit": self.daily_limit,
            "used_today": self._used_today,
            "remaining_today": max(0, self.daily_limit - self._used_today),
            "visitor_limit_per_hour": self.visitor_limit,
            "served": self.total_served,
            "refused": self.total_refused,
            "own_key_requests": self.byo_key_requests,
            "resets_in_seconds": _seconds_until_utc_midnight(),
        }


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=23, minute=59, second=59)
    return max(1, int((midnight - now).total_seconds()))


def visitor_id(client_host: str | None, forwarded_for: str | None, user_agent: str | None) -> str:
    """A stable, non-identifying handle for one visitor.

    Behind a proxy the real address is the first entry of X-Forwarded-For.
    The result is hashed because a demo has no business retaining raw IPs.
    """
    address = (forwarded_for or "").split(",")[0].strip() or (client_host or "unknown")
    digest = hashlib.sha256(f"{address}|{(user_agent or '')[:120]}".encode()).hexdigest()
    return digest[:20]
