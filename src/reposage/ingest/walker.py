"""Repository traversal.

Real repositories are mostly noise: lockfiles, minified bundles, vendored
dependencies, build output, fixtures. Indexing them costs embedding budget and,
worse, dilutes retrieval quality. This module applies layered filtering and,
when a repository still exceeds the file budget, keeps the most informative
files rather than an arbitrary prefix of the directory listing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from reposage.ingest.languages import LanguageSpec, get_spec
from reposage.logging_setup import get_logger

log = get_logger(__name__)

try:
    import pathspec

    _HAS_PATHSPEC = True
except ImportError:  # pragma: no cover
    pathspec = None  # type: ignore[assignment]
    _HAS_PATHSPEC = False


EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        ".vs",
        "node_modules",
        "bower_components",
        "jspm_packages",
        "venv",
        ".venv",
        "env",
        ".env",
        "virtualenv",
        "site-packages",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        "out",
        "target",
        "bin",
        "obj",
        ".next",
        ".nuxt",
        ".svelte-kit",
        "vendor",
        "third_party",
        "thirdparty",
        "externals",
        "coverage",
        "htmlcov",
        ".nyc_output",
        "migrations",
        "locale",
        "locales",
        "i18n",
        ".terraform",
        ".serverless",
        ".gradle",
        ".m2",
        "fixtures",
        "__snapshots__",
        "testdata",
        ".reposage",
        ".cache",
        "tmp",
        "temp",
    }
)

EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "poetry.lock",
        "pdm.lock",
        "uv.lock",
        "pipfile.lock",
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "go.sum",
        "mix.lock",
        "flake.lock",
        "packages.lock.json",
    }
)

EXCLUDED_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".map",
    ".bundle.js",
    ".chunk.js",
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".a",
    ".o",
    ".class",
    ".jar",
    ".war",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".bmp",
    ".tiff",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".webm",
    ".ogg",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".ttf",
    ".woff",
    ".woff2",
    ".eot",
    ".otf",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
    ".pkl",
    ".npy",
    ".npz",
    ".lock",
    ".log",
    ".bin",
    ".dat",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)

# Files whose presence explains how a project is structured and operated.
HIGH_VALUE_NAMES: frozenset[str] = frozenset(
    {
        "readme.md",
        "architecture.md",
        "contributing.md",
        "design.md",
        "pyproject.toml",
        "setup.py",
        "package.json",
        "cargo.toml",
        "go.mod",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "makefile",
        "main.py",
        "app.py",
        "__main__.py",
        "cli.py",
        "server.py",
        "index.js",
        "index.ts",
        "main.go",
        "main.rs",
        "main.java",
    }
)


@dataclass(slots=True)
class SourceFile:
    """One file that survived filtering, with its text already loaded."""

    path: Path
    rel_path: str
    language: str
    spec: LanguageSpec
    content: str
    size_bytes: int
    num_lines: int

    @property
    def is_code(self) -> bool:
        return self.spec.is_code


def _looks_binary(sample: bytes) -> bool:
    """Null bytes are the single most reliable text/binary discriminator."""
    return b"\x00" in sample


def _load_gitignore(root: Path) -> object | None:
    if not _HAS_PATHSPEC:
        return None
    patterns: list[str] = []
    for candidate in (root / ".gitignore", root / ".git" / "info" / "exclude"):
        if candidate.is_file():
            try:
                patterns.extend(candidate.read_text(encoding="utf-8", errors="ignore").splitlines())
            except OSError:  # pragma: no cover
                continue
    if not patterns:
        return None
    try:
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    except Exception:  # pragma: no cover - malformed gitignore
        return None


def _importance(rel_path: str, spec: LanguageSpec, num_lines: int) -> float:
    """Heuristic ranking used only when a repository exceeds the file budget.

    Favours real source code, shallow paths, entry points and documentation,
    and penalises tests and generated-looking files.
    """
    name = Path(rel_path).name.lower()
    score = 10.0 if spec.is_code else 4.0
    if name in HIGH_VALUE_NAMES:
        score += 12.0
    depth = rel_path.count("/")
    score += max(0.0, 5.0 - depth * 1.2)
    if 20 <= num_lines <= 800:
        score += 3.0
    elif num_lines > 3000:
        score -= 4.0
    lowered = rel_path.lower()
    if any(marker in lowered for marker in ("test", "spec", "mock", "example", "sample", "demo")):
        score -= 5.0
    if any(marker in lowered for marker in ("generated", "_pb2", ".pb.", "autogen")):
        score -= 8.0
    if spec.name in ("markdown", "rst") and depth <= 1:
        score += 4.0
    return score


def walk_repository(
    root: Path,
    *,
    max_file_bytes: int = 400_000,
    max_files: int = 4_000,
    respect_gitignore: bool = True,
    include_globs: list[str] | None = None,
) -> list[SourceFile]:
    """Return the files worth indexing, capped at ``max_files`` by importance."""
    root = root.resolve()
    ignore_spec = _load_gitignore(root) if respect_gitignore else None
    include_matcher = None
    if include_globs and _HAS_PATHSPEC:
        include_matcher = pathspec.PathSpec.from_lines("gitwildmatch", include_globs)

    collected: list[tuple[float, SourceFile]] = []
    skipped = {"binary": 0, "too_large": 0, "ignored": 0, "unreadable": 0, "excluded": 0}

    for file_path in _iter_files(root):
        rel = file_path.relative_to(root).as_posix()
        name = file_path.name.lower()

        if name in EXCLUDED_FILENAMES or rel.lower().endswith(EXCLUDED_SUFFIXES):
            skipped["excluded"] += 1
            continue
        if ignore_spec is not None and ignore_spec.match_file(rel):  # type: ignore[union-attr]
            skipped["ignored"] += 1
            continue
        if include_matcher is not None and not include_matcher.match_file(rel):
            skipped["excluded"] += 1
            continue

        try:
            stat = file_path.stat()
        except OSError:
            skipped["unreadable"] += 1
            continue
        if stat.st_size > max_file_bytes or stat.st_size == 0:
            skipped["too_large"] += 1
            continue

        try:
            raw = file_path.read_bytes()
        except OSError:
            skipped["unreadable"] += 1
            continue
        if _looks_binary(raw[:8192]):
            skipped["binary"] += 1
            continue

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")

        # A very long average line length means minified or generated content.
        num_lines = content.count("\n") + 1
        if num_lines and len(content) / num_lines > 400:
            skipped["excluded"] += 1
            continue

        spec = get_spec(file_path)
        source = SourceFile(
            path=file_path,
            rel_path=rel,
            language=spec.name,
            spec=spec,
            content=content,
            size_bytes=stat.st_size,
            num_lines=num_lines,
        )
        collected.append((_importance(rel, spec, num_lines), source))

    collected.sort(key=lambda item: (-item[0], item[1].rel_path))
    truncated = len(collected) > max_files
    selected = [source for _, source in collected[:max_files]]

    log.info(
        "walk.complete",
        root=str(root),
        found=len(collected),
        selected=len(selected),
        truncated=truncated,
        **skipped,
    )
    # Restore path order so downstream output is deterministic and readable.
    selected.sort(key=lambda f: f.rel_path)
    return selected


def _iter_files(root: Path) -> Iterator[Path]:
    """Depth-first walk that prunes excluded directories in place."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = [
            d for d in dirnames if d.lower() not in EXCLUDED_DIRS and not d.startswith(".git")
        ]
        current = Path(dirpath)
        for filename in filenames:
            yield current / filename
