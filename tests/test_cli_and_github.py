"""CLI surface and GitHub review posting."""

from __future__ import annotations

import httpx
import orjson
import pytest
from typer.testing import CliRunner

from reposage import __version__
from reposage.cli import app
from reposage.models import ReviewFinding, ReviewReport, Severity
from reposage.review.github import (
    MARKER,
    GitHubClient,
    GitHubError,
    PullRequestRef,
    ref_from_environment,
    render_findings,
    render_summary,
)

runner = CliRunner()


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run the CLI from an empty directory.

    Settings load `.env` relative to the working directory, so without this a
    developer's real key leaks into the test and `doctor` makes a live call.
    Tests must not depend on whoever is running them.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REPOSAGE_DATA_DIR", str(tmp_path / ".reposage"))
    from reposage.config import reset_settings_cache

    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


class TestCLI:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0 and __version__ in result.stdout

    def test_help_lists_every_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("index", "ask", "list", "review", "serve", "doctor", "eval"):
            assert command in result.stdout

    def test_list_is_empty_on_a_fresh_install(self, isolated_cwd, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0 and "No indexes yet" in result.stdout

    def test_missing_api_key_gives_actionable_guidance(self, isolated_cwd, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        result = runner.invoke(app, ["ask", "-r", "anything", "a question here"])
        assert result.exit_code == 2
        assert "aistudio.google.com" in result.stdout

    def test_review_requires_a_source(self, isolated_cwd, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        result = runner.invoke(app, ["review"])
        assert result.exit_code == 2 and "--diff or --pr" in result.stdout

    def test_doctor_reports_diagnostics(self, isolated_cwd, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "tree-sitter" in result.stdout and "GEMINI_API_KEY" in result.stdout


@pytest.fixture
def report() -> ReviewReport:
    return ReviewReport(
        summary="One security issue needs fixing before merge.",
        files_reviewed=2,
        findings=[
            ReviewFinding(
                path="src/api.py",
                line=11,
                severity=Severity.HIGH,
                category="security",
                title="SQL built by string interpolation",
                body="User input reaches the query.",
                suggestion='row = db.query("SELECT ...", user_id)',
                confidence=0.92,
            ),
            ReviewFinding(
                path="src/api.py",
                line=None,
                severity=Severity.LOW,
                category="tests",
                title="No test for the not-found path",
                body="Add one.",
                confidence=0.6,
            ),
        ],
    )


class TestRendering:
    def test_summary_carries_the_idempotency_marker(self, report):
        assert render_summary(report).startswith(MARKER)

    def test_summary_shows_severity_counts(self, report):
        body = render_summary(report)
        assert "1 high" in body and "1 low" in body

    def test_unanchored_findings_appear_in_the_summary(self, report):
        assert "No test for the not-found path" in render_summary(report)

    def test_clean_review_says_so(self):
        assert "No issues found" in render_summary(ReviewReport(summary="Looks good."))

    def test_findings_render_with_suggestions(self, report):
        assert "```suggestion" in render_findings(report)

    def test_blocking_findings_are_identified(self, report):
        assert len(report.blocking) == 1


class TestGitHubClient:
    def test_a_token_is_required(self):
        with pytest.raises(GitHubError, match="token is required"):
            GitHubClient("")

    async def test_fetches_a_diff_with_the_right_accept_header(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["accept"] == "application/vnd.github.v3.diff"
            return httpx.Response(200, text="diff --git a/a b/a\n")

        client = GitHubClient("t", transport=httpx.MockTransport(handler))
        assert "diff --git" in await client.get_diff(PullRequestRef("o", "r", 1))
        await client.aclose()

    async def test_posts_inline_comments(self, report):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(orjson.loads(request.content))
            return httpx.Response(200, json={"id": 1})

        client = GitHubClient("t", transport=httpx.MockTransport(handler))
        await client.post_review(PullRequestRef("o", "r", 7), report)
        assert captured["event"] == "COMMENT"
        assert len(captured["comments"]) == 1  # only the anchored finding
        assert captured["comments"][0]["line"] == 11
        assert captured["comments"][0]["side"] == "RIGHT"
        await client.aclose()

    async def test_requests_changes_when_asked_and_blocking_exists(self, report):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(orjson.loads(request.content))
            return httpx.Response(200, json={"id": 1})

        client = GitHubClient("t", transport=httpx.MockTransport(handler))
        await client.post_review(
            PullRequestRef("o", "r", 7), report, request_changes_on_blocking=True
        )
        assert captured["event"] == "REQUEST_CHANGES"
        await client.aclose()

    async def test_rejected_inline_comments_degrade_to_a_summary(self, report):
        """GitHub 422s the whole review if one comment is off-diff. Losing the
        review entirely would be worse than losing the inline placement."""
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = orjson.loads(request.content)
            calls.append(payload)
            if "comments" in payload:
                return httpx.Response(422, text="line must be part of the diff")
            return httpx.Response(200, json={"id": 2})

        client = GitHubClient("t", transport=httpx.MockTransport(handler))
        await client.post_review(PullRequestRef("o", "r", 7), report)
        assert len(calls) == 2
        assert "comments" not in calls[1]
        assert "SQL built by string interpolation" in calls[1]["body"]
        await client.aclose()

    async def test_summary_comment_is_updated_not_duplicated(self):
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.method == "GET":
                return httpx.Response(200, json=[{"id": 55, "body": f"{MARKER}\nold"}])
            return httpx.Response(200, json={"id": 55})

        client = GitHubClient("t", transport=httpx.MockTransport(handler))
        await client.upsert_summary_comment(PullRequestRef("o", "r", 7), "new body")
        assert requests[-1][0] == "PATCH" and "comments/55" in requests[-1][1]
        await client.aclose()

    async def test_api_errors_surface_clearly(self):
        client = GitHubClient(
            "t", transport=httpx.MockTransport(lambda r: httpx.Response(404, text="Not Found"))
        )
        with pytest.raises(GitHubError, match="404"):
            await client.get_pull_request(PullRequestRef("o", "r", 999))
        await client.aclose()


class TestActionsEnvironment:
    def test_reads_the_repository_and_pr_number(self, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "adwitiyashukla/reposage")
        monkeypatch.setenv("PR_NUMBER", "42")
        ref = ref_from_environment()
        assert ref and ref.slug == "adwitiyashukla/reposage#42"

    def test_falls_back_to_the_event_payload(self, monkeypatch, tmp_path):
        event = tmp_path / "event.json"
        event.write_bytes(orjson.dumps({"pull_request": {"number": 7}}))
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.delenv("PR_NUMBER", raising=False)
        monkeypatch.delenv("GITHUB_PR_NUMBER", raising=False)
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
        ref = ref_from_environment()
        assert ref and ref.number == 7

    def test_returns_none_outside_actions(self, monkeypatch):
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("PR_NUMBER", raising=False)
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        assert ref_from_environment() is None


class TestConsoleEncoding:
    """Windows defaults redirected output to cp1252, which cannot encode the
    box-drawing and severity glyphs the CLI prints. Left unhandled it is not a
    garbled character, it is an unhandled exception mid-command."""

    def test_configure_is_idempotent_and_safe(self):
        from reposage.logging_setup import configure_console_encoding

        configure_console_encoding()
        configure_console_encoding()

    def test_tolerates_streams_without_reconfigure(self, monkeypatch):
        import io
        import sys as _sys

        from reposage.logging_setup import configure_console_encoding

        monkeypatch.setattr(_sys, "stdout", io.StringIO())
        configure_console_encoding()

    def test_severity_glyphs_survive_a_cp1252_roundtrip(self):
        """Rendering must not depend on the terminal's code page."""
        for severity in Severity:
            rendered = ReviewFinding(
                path="a.py", line=1, severity=severity, title="t", body="b"
            ).to_markdown()
            assert rendered.encode("utf-8", errors="replace").decode("utf-8")
