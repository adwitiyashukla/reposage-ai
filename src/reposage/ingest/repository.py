"""Repository acquisition.

Accepts a local path, a full Git URL, or GitHub shorthand (``owner/repo``) and
normalises all three into a checked-out working tree plus the commit SHA that
identifies it. Clones are shallow and single-branch because we only ever read
the current tree, and cached across runs so re-indexing is a fetch rather than a
full download.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from reposage.logging_setup import get_logger

log = get_logger(__name__)

_SHORTHAND = re.compile(r"^[\w.\-]+/[\w.\-]+$")
_GIT_URL = re.compile(r"^(https?://|git@|ssh://|git://)")
_SLUG_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


class RepositoryError(RuntimeError):
    """Raised when a repository cannot be acquired."""


@dataclass(slots=True)
class RepositorySource:
    """A resolved, on-disk repository ready to be walked."""

    repo_id: str
    name: str
    origin: str
    path: Path
    commit: str = ""
    branch: str = ""
    is_local: bool = False

    @property
    def short_commit(self) -> str:
        return self.commit[:8] if self.commit else "working-tree"

    @property
    def web_url(self) -> str | None:
        """Best-effort browse URL so citations can deep-link to GitHub."""
        origin = self.origin
        if origin.startswith("git@"):
            origin = origin.replace(":", "/").replace("git@", "https://")
        if origin.startswith("http") and "github.com" in origin:
            base = origin.removesuffix(".git")
            ref = self.commit or self.branch or "HEAD"
            return f"{base}/blob/{ref}"
        return None


async def _run_git(*args: str, cwd: Path | None = None, timeout: float = 300.0) -> str:
    """Run git and return stdout, raising a readable error on failure."""
    if shutil.which("git") is None:
        raise RepositoryError("git is not installed or not on PATH.")
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        process.kill()
        raise RepositoryError(f"git {' '.join(args[:2])} timed out after {timeout:.0f}s") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:400]
        raise RepositoryError(f"git {' '.join(args[:2])} failed: {detail}")
    return stdout.decode("utf-8", errors="replace").strip()


def normalise_source(source: str) -> tuple[str, str]:
    """Return ``(git_url, display_name)`` for any accepted source form."""
    source = source.strip().rstrip("/")
    if _SHORTHAND.match(source):
        return f"https://github.com/{source}.git", source
    if _GIT_URL.match(source):
        name = source.removesuffix(".git").rstrip("/")
        parts = [p for p in name.replace(":", "/").split("/") if p]
        display = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        return source, display
    raise RepositoryError(f"Unrecognised repository source: {source!r}")


def _slug(name: str) -> str:
    return _SLUG_SAFE.sub("-", name).strip("-").lower() or "repo"


async def resolve_repository(
    source: str,
    cache_dir: Path,
    *,
    branch: str | None = None,
    refresh: bool = False,
    depth: int = 1,
) -> RepositorySource:
    """Materialise ``source`` on disk and return its identity."""
    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_dir():
        return await _resolve_local(candidate)

    url, display = normalise_source(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _slug(display)

    if target.exists() and refresh:
        shutil.rmtree(target, ignore_errors=True)

    if target.exists() and (target / ".git").exists():
        log.info("repo.refresh", path=str(target))
        try:
            await _run_git("fetch", "--depth", str(depth), "origin", cwd=target)
            head = await _run_git("rev-parse", "--abbrev-ref", "origin/HEAD", cwd=target)
            await _run_git("reset", "--hard", head.strip() or "origin/HEAD", cwd=target)
        except RepositoryError as exc:
            log.warning("repo.refresh_failed", error=str(exc)[:200])
    else:
        shutil.rmtree(target, ignore_errors=True)
        args = ["clone", "--depth", str(depth), "--single-branch", "--no-tags", "--quiet"]
        if branch:
            args += ["--branch", branch]
        args += [url, str(target)]
        log.info("repo.clone", url=url, depth=depth)
        await _run_git(*args, timeout=900.0)

    commit = await _safe_git("rev-parse", "HEAD", cwd=target)
    current_branch = await _safe_git("rev-parse", "--abbrev-ref", "HEAD", cwd=target)
    return RepositorySource(
        repo_id=f"{_slug(display)}@{commit[:8] or 'head'}",
        name=display,
        origin=url,
        path=target,
        commit=commit,
        branch=current_branch,
    )


async def _resolve_local(path: Path) -> RepositorySource:
    resolved = path.resolve()
    commit = await _safe_git("rev-parse", "HEAD", cwd=resolved)
    branch = await _safe_git("rev-parse", "--abbrev-ref", "HEAD", cwd=resolved)
    origin = await _safe_git("config", "--get", "remote.origin.url", cwd=resolved)
    name = resolved.name
    return RepositorySource(
        repo_id=f"{_slug(name)}@{commit[:8] or 'local'}",
        name=name,
        origin=origin or str(resolved),
        path=resolved,
        commit=commit,
        branch=branch,
        is_local=True,
    )


async def _safe_git(*args: str, cwd: Path) -> str:
    """Git call that returns an empty string instead of raising."""
    try:
        return await _run_git(*args, cwd=cwd, timeout=30.0)
    except RepositoryError:
        return ""
