"""HTTP surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reposage.api import deps
from reposage.api.main import create_app
from reposage.index.store import RepoIndex
from reposage.ingest.pipeline import IngestionPipeline
from reposage.observability import Tracer, use_tracer


@pytest.fixture
async def app_client(sample_repo, settings, client, monkeypatch):
    """An app wired to the fake model and a freshly built index."""
    with use_tracer(Tracer()):
        ingestion = await IngestionPipeline(settings).run(str(sample_repo))
        index = await RepoIndex.build(ingestion, client, settings=settings)
        index.save(settings.index_dir)

    state = deps.AppState(settings)
    state._client = client
    state.register(index)
    monkeypatch.setattr(deps, "_STATE", state)
    monkeypatch.setattr(deps, "get_state", lambda: state)

    for module in ("health", "indexes", "ask", "review"):
        monkeypatch.setattr(f"reposage.api.routes.{module}.get_state", lambda: state)

    with TestClient(create_app()) as test_client:
        yield test_client, index


class TestSystemRoutes:
    def test_health_reports_configuration(self, app_client):
        client, _ = app_client
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert payload["version"]
        assert "fast_model" in payload["llm"]

    def test_graph_returns_mermaid(self, app_client):
        client, _ = app_client
        payload = client.get("/api/graph").json()
        assert payload["format"] == "mermaid" and "critique" in payload["source"]

    def test_ui_is_served(self, app_client):
        client, _ = app_client
        response = client.get("/")
        assert response.status_code == 200 and "RepoSage" in response.text

    def test_openapi_schema_is_valid(self, app_client):
        client, _ = app_client
        schema = client.get("/openapi.json").json()
        assert "/api/ask" in schema["paths"]


class TestIndexRoutes:
    def test_lists_indexes(self, app_client):
        client, index = app_client
        ids = [entry["id"] for entry in client.get("/api/indexes").json()["indexes"]]
        assert index.index_id in ids

    def test_describes_one_index(self, app_client):
        client, index = app_client
        payload = client.get(f"/api/indexes/{index.index_id}").json()
        assert payload["chunks"] == len(index)
        assert payload["vector_store"]["backend"] == "numpy-exact"

    def test_unknown_index_is_404(self, app_client):
        client, _ = app_client
        assert client.get("/api/indexes/nope").status_code == 404


class TestAskRoutes:
    def test_answers_with_citations(self, app_client):
        client, index = app_client
        response = client.post(
            "/api/ask", json={"repo": index.index_id, "question": "How does token validation work?"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"]
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["trace"]

    def test_short_questions_are_rejected(self, app_client):
        client, index = app_client
        assert (
            client.post("/api/ask", json={"repo": index.index_id, "question": "x"}).status_code
            == 422
        )

    def test_unknown_repo_is_404(self, app_client):
        client, _ = app_client
        assert (
            client.post("/api/ask", json={"repo": "nope", "question": "anything here"}).status_code
            == 404
        )

    def test_source_endpoint_returns_segments(self, app_client):
        client, index = app_client
        path = next(p for p in index.paths() if p.endswith(".py"))
        payload = client.get(f"/api/source/{index.index_id}", params={"path": path}).json()
        assert payload["segments"] and payload["path"] == path

    def test_source_endpoint_rejects_unindexed_paths(self, app_client):
        client, index = app_client
        response = client.get(f"/api/source/{index.index_id}", params={"path": "../../etc/passwd"})
        assert response.status_code == 404


class TestReviewRoute:
    def test_reviews_a_diff(self, app_client, sample_diff):
        client, index = app_client
        response = client.post(
            "/api/review",
            json={"diff": sample_diff, "title": "Fix SQL injection", "repo": index.index_id},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["files_reviewed"] == 2
        assert payload["findings"]
        assert "src/api.py" in {f["path"] for f in payload["findings"]}

    def test_empty_diff_is_rejected(self, app_client):
        client, _ = app_client
        assert client.post("/api/review", json={"diff": ""}).status_code == 422
