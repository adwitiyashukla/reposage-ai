from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reposage.config import Settings, get_settings
from reposage.ingest.chunker import ASTChunker, treesitter_available
from reposage.ingest.repository import RepositorySource, resolve_repository
from reposage.ingest.walker import SourceFile, walk_repository
from reposage.logging_setup import get_logger
from reposage.models import Chunk, ChunkKind, RepoMetadata
from reposage.observability import current_tracer

log = get_logger(__name__)


@dataclass(slots=True)
class IngestionResult:
    repo: RepositorySource
    metadata: RepoMetadata
    chunks: list[Chunk]
    repo_map: str
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def run(
        self,
        source: str,
        *,
        branch: str | None = None,
        refresh: bool = False,
        include_globs: list[str] | None = None,
    ) -> IngestionResult:
        tracer = current_tracer()
        self.settings.ensure_dirs()

        with tracer.span("ingest.resolve", source=source):
            repo = await resolve_repository(
                source, self.settings.repo_dir, branch=branch, refresh=refresh
            )

        with tracer.span("ingest.walk", path=str(repo.path)) as span:
            files = await asyncio.to_thread(
                walk_repository,
                repo.path,
                max_file_bytes=self.settings.max_file_bytes,
                max_files=self.settings.max_files,
                include_globs=include_globs,
            )
            span.set(files=len(files))

        if not files:
            raise ValueError(
                f"No indexable files found in {repo.name}. "
                "The repository may be empty or contain only excluded file types."
            )

        with tracer.span("ingest.chunk", files=len(files)) as span:
            chunks, chunk_stats = await asyncio.to_thread(self._chunk_all, files)
            span.set(**chunk_stats)

        repo_map = build_repo_map(files, chunks)
        languages = Counter(f.language for f in files)
        metadata = RepoMetadata(
            repo_id=repo.repo_id,
            source=repo.origin,
            name=repo.name,
            commit=repo.commit,
            branch=repo.branch,
            indexed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            num_files=len(files),
            num_chunks=len(chunks),
            languages=dict(languages.most_common()),
            embed_model=self.settings.embed_model,
            total_lines=sum(f.num_lines for f in files),
        )

        stats = {
            **chunk_stats,
            "treesitter_available": treesitter_available(),
            "languages": dict(languages.most_common(12)),
            "avg_chunk_lines": round(
                sum(c.end_line - c.start_line + 1 for c in chunks) / max(1, len(chunks)), 1
            ),
            "chunk_kinds": dict(Counter(c.kind.value for c in chunks).most_common()),
        }
        log.info(
            "ingest.complete",
            repo=repo.name,
            files=len(files),
            chunks=len(chunks),
            ast_coverage=stats.get("ast_coverage"),
        )
        return IngestionResult(
            repo=repo, metadata=metadata, chunks=chunks, repo_map=repo_map, stats=stats
        )

    def _chunk_all(self, files: list[SourceFile]) -> tuple[list[Chunk], dict[str, Any]]:
        chunker = ASTChunker(
            max_lines=self.settings.chunk_max_lines,
            overlap_lines=self.settings.chunk_overlap_lines,
        )
        chunks: list[Chunk] = []
        for source_file in files:
            try:
                chunks.extend(
                    chunker.chunk(source_file.rel_path, source_file.content, source_file.spec)
                )
            except Exception as exc:
                log.warning("chunk.failed", path=source_file.rel_path, error=str(exc)[:160])
        return chunks, chunker.stats.as_dict()


_TREE_BUDGET = 260
_SYMBOLS_PER_FILE = 6


def build_repo_map(files: list[SourceFile], chunks: list[Chunk]) -> str:
    symbols: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        if chunk.symbol and chunk.kind in (
            ChunkKind.CLASS,
            ChunkKind.FUNCTION,
            ChunkKind.INTERFACE,
            ChunkKind.MODULE,
        ):
            bucket = symbols[chunk.path]
            if chunk.symbol not in bucket and len(bucket) < _SYMBOLS_PER_FILE:
                bucket.append(chunk.symbol)

    by_directory: dict[str, list[SourceFile]] = defaultdict(list)
    for source_file in files:
        directory = str(Path(source_file.rel_path).parent).replace("\\", "/")
        by_directory[directory if directory != "." else "(root)"].append(source_file)

    ranked = sorted(
        by_directory.items(),
        key=lambda item: (-sum(f.num_lines for f in item[1]), item[0]),
    )

    lines: list[str] = [
        f"REPOSITORY MAP  |  {len(files)} files, "
        f"{sum(f.num_lines for f in files):,} lines, {len(chunks)} indexed chunks",
        "",
    ]
    emitted = 0
    for directory, entries in ranked:
        if emitted >= _TREE_BUDGET:
            lines.append(
                f"... {len(ranked) - ranked.index((directory, entries))} more directories omitted"
            )
            break
        lines.append(f"{directory}/")
        entries.sort(key=lambda f: -f.num_lines)
        for source_file in entries[:14]:
            if emitted >= _TREE_BUDGET:
                break
            names = symbols.get(source_file.rel_path, [])
            suffix = f"  -> {', '.join(names)}" if names else ""
            lines.append(
                f"  {Path(source_file.rel_path).name} ({source_file.num_lines} lines){suffix}"
            )
            emitted += 1
        if len(entries) > 14:
            lines.append(f"  ... {len(entries) - 14} more files")
        lines.append("")
    return "\n".join(lines)
