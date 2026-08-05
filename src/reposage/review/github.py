"""Minimal GitHub REST client for pull-request review.

Only four operations are needed, so a full SDK would be dead weight. What this
client does add is the operational behaviour a bot needs to be tolerable:

* **Idempotent summaries.** Every summary carries a hidden marker comment. On a
  re-run the existing comment is edited rather than a new one appended, so a PR
  with twelve pushes has one review comment, not twelve.
* **Graceful inline degradation.** GitHub rejects an entire review if any single
  inline comment targets a line outside the diff. Rather than losing the review,
  a 422 falls back to posting the findings in the summary body.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import orjson

from reposage.logging_setup import get_logger
from reposage.models import ReviewReport, Severity

log = get_logger(__name__)

API_ROOT = "https://api.github.com"
MARKER = "<!-- reposage-review -->"


class GitHubError(RuntimeError):
    """A GitHub API call failed."""


@dataclass(slots=True)
class PullRequestRef:
    owner: str
    repo: str
    number: int

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}#{self.number}"


class GitHubClient:
    """Thin async wrapper over the endpoints a review bot needs."""

    def __init__(
        self,
        token: str,
        *,
        api_root: str = API_ROOT,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not token:
            raise GitHubError(
                "A GitHub token is required. In Actions pass secrets.GITHUB_TOKEN; "
                "locally set GITHUB_TOKEN in your .env file."
            )
        self.api_root = api_root.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "authorization": f"Bearer {token}",
                "accept": "application/vnd.github+json",
                "x-github-api-version": "2022-11-28",
                "user-agent": "reposage-review/1.0",
            },
            transport=transport,
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.api_root}{path}"
        response = await self._client.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise GitHubError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
        return response

    # ------------------------------------------------------------------ read
    async def get_pull_request(self, ref: PullRequestRef) -> dict[str, Any]:
        response = await self._request("GET", f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}")
        return response.json()

    async def get_diff(self, ref: PullRequestRef) -> str:
        """Fetch the raw unified diff for a pull request."""
        response = await self._client.get(
            f"{self.api_root}/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}",
            headers={"accept": "application/vnd.github.v3.diff"},
        )
        if response.status_code >= 400:
            raise GitHubError(f"diff fetch failed {response.status_code}: {response.text[:300]}")
        return response.text

    # ----------------------------------------------------------------- write
    async def post_review(
        self,
        ref: PullRequestRef,
        report: ReviewReport,
        *,
        inline: bool = True,
        request_changes_on_blocking: bool = False,
    ) -> dict[str, Any]:
        """Publish the review, degrading to a summary-only comment if needed."""
        body = render_summary(report)
        comments = _inline_comments(report) if inline else []
        event = "REQUEST_CHANGES" if request_changes_on_blocking and report.blocking else "COMMENT"
        payload: dict[str, Any] = {"body": body, "event": event}
        if comments:
            payload["comments"] = comments

        try:
            response = await self._request(
                "POST",
                f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/reviews",
                content=orjson.dumps(payload),
            )
            log.info("github.review_posted", pr=ref.slug, inline=len(comments), review_event=event)
            return response.json()
        except GitHubError as exc:
            if "422" not in str(exc) or not comments:
                raise
            log.warning("github.inline_rejected", pr=ref.slug, detail=str(exc)[:200])

        fallback = f"{body}\n\n{render_findings(report)}"
        response = await self._request(
            "POST",
            f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/reviews",
            content=orjson.dumps({"body": fallback, "event": "COMMENT"}),
        )
        log.info("github.review_posted", pr=ref.slug, inline=0, review_event="COMMENT", degraded=True)
        return response.json()

    async def upsert_summary_comment(self, ref: PullRequestRef, body: str) -> dict[str, Any]:
        """Create or update the single RepoSage comment on this PR."""
        marked = f"{MARKER}\n{body}"
        existing = await self._find_marked_comment(ref)
        if existing is not None:
            response = await self._request(
                "PATCH",
                f"/repos/{ref.owner}/{ref.repo}/issues/comments/{existing}",
                content=orjson.dumps({"body": marked}),
            )
            log.info("github.comment_updated", pr=ref.slug, comment_id=existing)
        else:
            response = await self._request(
                "POST",
                f"/repos/{ref.owner}/{ref.repo}/issues/{ref.number}/comments",
                content=orjson.dumps({"body": marked}),
            )
            log.info("github.comment_created", pr=ref.slug)
        return response.json()

    async def _find_marked_comment(self, ref: PullRequestRef) -> int | None:
        response = await self._request(
            "GET",
            f"/repos/{ref.owner}/{ref.repo}/issues/{ref.number}/comments",
            params={"per_page": 100},
        )
        for comment in response.json():
            if MARKER in (comment.get("body") or ""):
                return int(comment["id"])
        return None

    async def aclose(self) -> None:
        await self._client.aclose()


def _inline_comments(report: ReviewReport) -> list[dict[str, Any]]:
    """Findings that carry a concrete line become inline review comments."""
    comments: list[dict[str, Any]] = []
    for finding in report.sorted_findings():
        if finding.line is None:
            continue
        comments.append(
            {
                "path": finding.path,
                "line": finding.line,
                "side": "RIGHT",
                "body": finding.to_markdown(),
            }
        )
    return comments


def render_summary(report: ReviewReport) -> str:
    """The markdown block posted at the top of the review."""
    counts: dict[str, int] = {}
    for finding in report.findings:
        counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1

    lines = [f"{MARKER}", "## RepoSage review", "", report.summary or "_No summary produced._", ""]
    if counts:
        badges = "  ".join(
            f"{Severity(sev).emoji} **{count} {sev}**"
            for sev, count in sorted(counts.items(), key=lambda kv: Severity(kv[0]).rank)
        )
        lines += [badges, ""]
    else:
        lines += ["No issues found.", ""]

    unanchored = [f for f in report.sorted_findings() if f.line is None]
    if unanchored:
        lines += ["### File-level notes", ""]
        lines += [f"{f.to_markdown()}\n" for f in unanchored]

    lines += [
        "---",
        f"<sub>Reviewed {report.files_reviewed} file(s) in {report.elapsed_seconds:.1f}s "
        f"| {report.usage.total_tokens:,} tokens | ${report.usage.cost_usd:.4f} "
        f"| <a href='https://github.com/adwitiyashukla/reposage-ai'>RepoSage</a></sub>",
    ]
    return "\n".join(lines)


def render_findings(report: ReviewReport) -> str:
    """All findings as markdown, used when inline comments are rejected."""
    if not report.findings:
        return "_No findings._"
    return "\n\n".join(
        f"### `{f.path}`" + (f" line {f.line}" if f.line else "") + f"\n\n{f.to_markdown()}"
        for f in report.sorted_findings()
    )


def ref_from_environment() -> PullRequestRef | None:
    """Build a pull-request reference from GitHub Actions environment variables."""
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repository:
        return None
    owner, repo = repository.split("/", 1)

    number = os.environ.get("PR_NUMBER") or os.environ.get("GITHUB_PR_NUMBER")
    if not number:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and os.path.exists(event_path):
            try:
                with open(event_path, "rb") as handle:
                    event = orjson.loads(handle.read())
                number = str(
                    event.get("pull_request", {}).get("number") or event.get("number") or ""
                )
            except Exception as exc:  # pragma: no cover
                log.debug("github.event_parse_failed", error=str(exc)[:160])
    if not number or not str(number).isdigit():
        return None
    return PullRequestRef(owner=owner, repo=repo, number=int(number))
