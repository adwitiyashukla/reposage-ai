"""Hosted-demo guardrails.

The demo runs on the maintainer's key, so these limits are the only thing
between a public URL and an exhausted quota. They are worth testing properly.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from reposage.api import deps
from reposage.api.demo import DemoBudget, visitor_id
from reposage.api.main import create_app
from reposage.index.store import RepoIndex
from reposage.ingest.pipeline import IngestionPipeline
from reposage.observability import Tracer, use_tracer


class TestBudget:
    def test_allows_within_both_limits(self):
        budget = DemoBudget(daily_limit=10, visitor_limit=3)
        assert budget.check("alice").allowed

    def test_visitor_limit_refuses_the_fourth_question(self):
        budget = DemoBudget(daily_limit=100, visitor_limit=3)
        for _ in range(3):
            assert budget.check("alice").allowed
            budget.consume("alice")
        decision = budget.check("alice")
        assert not decision.allowed and decision.scope == "visitor"

    def test_one_visitor_cannot_block_another(self):
        budget = DemoBudget(daily_limit=100, visitor_limit=2)
        for _ in range(2):
            budget.consume("alice")
        assert not budget.check("alice").allowed
        assert budget.check("bob").allowed

    def test_daily_limit_refuses_everyone(self):
        budget = DemoBudget(daily_limit=3, visitor_limit=100)
        for i in range(3):
            budget.consume(f"visitor-{i}")
        decision = budget.check("someone-new")
        assert not decision.allowed and decision.scope == "global"

    def test_a_spent_daily_budget_is_fixable_with_an_own_key(self):
        """The distinction matters: the UI only offers the key box when supplying
        a key would actually help."""
        budget = DemoBudget(daily_limit=1, visitor_limit=100)
        budget.consume("a")
        assert budget.check("b").needs_own_key

        hourly = DemoBudget(daily_limit=100, visitor_limit=1)
        hourly.consume("a")
        assert not hourly.check("a").needs_own_key

    def test_visitor_window_expires(self):
        budget = DemoBudget(daily_limit=100, visitor_limit=1, window_seconds=1)
        budget.consume("alice")
        assert not budget.check("alice").allowed
        time.sleep(1.1)
        assert budget.check("alice").allowed

    def test_daily_counter_rolls_over(self, monkeypatch):
        budget = DemoBudget(daily_limit=1, visitor_limit=100)
        budget.consume("a")
        assert not budget.check("b").allowed

        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        monkeypatch.setattr(DemoBudget, "_today", staticmethod(lambda: tomorrow.strftime("%Y-%m-%d")))
        assert budget.check("b").allowed

    def test_own_key_requests_are_counted_separately(self):
        budget = DemoBudget(daily_limit=1, visitor_limit=1)
        budget.record_own_key()
        budget.record_own_key()
        assert budget.status()["own_key_requests"] == 2
        assert budget.status()["used_today"] == 0

    def test_visitor_table_is_bounded(self):
        budget = DemoBudget(daily_limit=10**9, visitor_limit=10**9, window_seconds=0)
        for i in range(5200):
            budget.consume(f"v{i}")
        assert len(budget._visitors) <= 5000

    def test_status_reports_remaining(self):
        budget = DemoBudget(daily_limit=10, visitor_limit=2)
        budget.consume("a")
        status = budget.status()
        assert status["used_today"] == 1 and status["remaining_today"] == 9


class TestVisitorIdentity:
    def test_is_stable_for_the_same_visitor(self):
        assert visitor_id("1.2.3.4", None, "Mozilla") == visitor_id("1.2.3.4", None, "Mozilla")

    def test_differs_across_visitors(self):
        assert visitor_id("1.2.3.4", None, "Mozilla") != visitor_id("5.6.7.8", None, "Mozilla")

    def test_prefers_the_forwarded_address_behind_a_proxy(self):
        """Hosted platforms terminate TLS upstream, so the socket address is the
        proxy and would collapse every visitor into one bucket."""
        direct = visitor_id("10.0.0.1", "203.0.113.9, 10.0.0.1", "UA")
        same_client = visitor_id("10.0.0.2", "203.0.113.9, 10.0.0.2", "UA")
        assert direct == same_client

    def test_does_not_retain_the_raw_address(self):
        handle = visitor_id("203.0.113.9", None, "UA")
        assert "203.0.113.9" not in handle and len(handle) == 20


@pytest.fixture
async def demo_client(sample_repo, settings, client, monkeypatch):
    """An app running in demo mode over a small pre-built index."""
    settings.demo_mode = True
    settings.demo_daily_budget = 3
    settings.demo_visitor_budget = 2
    with use_tracer(Tracer()):
        ingestion = await IngestionPipeline(settings).run(str(sample_repo))
        index = await RepoIndex.build(ingestion, client, settings=settings)
        index.save(settings.index_dir)
    settings.demo_index = index.index_id

    state = deps.AppState(settings)
    state._client = client
    state.register(index)
    monkeypatch.setattr(deps, "_STATE", state)
    monkeypatch.setattr(deps, "get_state", lambda: state)
    for module in ("health", "indexes", "ask", "review"):
        monkeypatch.setattr(f"reposage.api.routes.{module}.get_state", lambda: state)

    with TestClient(create_app()) as test_client:
        yield test_client, index


class TestDemoEndpoints:
    def test_demo_status_is_advertised(self, demo_client):
        http, index = demo_client
        payload = http.get("/api/demo").json()
        assert payload["enabled"] and payload["index"] == index.index_id
        assert payload["budget"]["daily_limit"] == 3

    def test_indexing_is_refused(self, demo_client):
        http, _ = demo_client
        response = http.post("/api/indexes", json={"source": "psf/requests"})
        assert response.status_code == 403
        assert "disabled on the public demo" in response.json()["detail"]

    def test_streaming_index_build_is_refused(self, demo_client):
        http, _ = demo_client
        assert http.get("/api/indexes/stream/build", params={"source": "psf/requests"}).status_code == 403

    def test_deleting_an_index_is_refused(self, demo_client):
        http, index = demo_client
        assert http.delete(f"/api/indexes/{index.index_id}").status_code == 403

    def test_questions_are_answered_until_the_visitor_limit(self, demo_client):
        http, index = demo_client
        body = {"repo": index.index_id, "question": "How does token validation work?"}
        assert http.post("/api/ask", json=body).status_code == 200
        assert http.post("/api/ask", json=body).status_code == 200
        refused = http.post("/api/ask", json=body)
        assert refused.status_code == 429
        assert "retry-after" in refused.headers

    def test_reading_indexed_source_stays_available(self, demo_client):
        """Citation viewing is read-only and must keep working when the budget
        is spent, otherwise a refused answer becomes unverifiable."""
        http, index = demo_client
        path = next(p for p in index.paths() if p.endswith(".py"))
        assert http.get(f"/api/source/{index.index_id}", params={"path": path}).status_code == 200

    def test_own_key_bypasses_the_budget(self, demo_client, monkeypatch, provider):
        """A visitor paying with their own key costs the host nothing, so the
        shared budget must not apply to them."""
        http, index = demo_client
        # Substitute the provider so no real credential or socket is needed.
        monkeypatch.setattr(deps, "GeminiProvider", lambda key, **kw: provider)
        body = {"repo": index.index_id, "question": "How does token validation work?"}
        for _ in range(2):
            http.post("/api/ask", json=body)
        assert http.post("/api/ask", json=body).status_code == 429
        # The same visitor, now paying their own way.
        assert http.post("/api/ask", json=body, headers={"x-reposage-key": "visitor-key"}).status_code == 200

    def test_a_refused_stream_still_opens_and_explains_itself(self, demo_client):
        """EventSource cannot read the body of a failed handshake, so a refusal
        delivered as an HTTP error is invisible to the browser. It must arrive
        as an event on a healthy stream instead."""
        http, index = demo_client
        params = {"repo": index.index_id, "q": "How does token validation work?"}
        for _ in range(2):
            http.get("/api/ask/stream", params=params)

        response = http.get("/api/ask/stream", params=params)
        assert response.status_code == 200
        assert "event: limit" in response.text
        assert "needs_own_key" in response.text

    def test_streaming_consumes_the_budget(self, demo_client):
        http, index = demo_client
        params = {"repo": index.index_id, "q": "How does token validation work?"}
        before = http.get("/api/demo").json()["budget"]["used_today"]
        http.get("/api/ask/stream", params=params)
        after = http.get("/api/demo").json()["budget"]["used_today"]
        assert after == before + 1
