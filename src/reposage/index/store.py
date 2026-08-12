from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson

from reposage.config import Settings, get_settings
from reposage.index.lexical import BM25Index
from reposage.index.vector_store import NumpyVectorStore
from reposage.ingest.pipeline import IngestionPipeline, IngestionResult
from reposage.logging_setup import get_logger
from reposage.models import Chunk, RepoMetadata
from reposage.observability import current_tracer

if TYPE_CHECKING:
    from reposage.llm.client import LLMClient

log = get_logger(__name__)

MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.jsonl"
REPO_MAP_FILE = "repo_map.txt"
SCHEMA_VERSION = 1


def slugify(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in name.lower())
    return safe.strip("-") or "repo"


@dataclass
class IndexStats:
    chunks: int = 0
    files: int = 0
    vectors: int = 0
    vocabulary: int = 0
    embed_seconds: float = 0.0
    build_seconds: float = 0.0
    ingestion: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "files": self.files,
            "vectors": self.vectors,
            "vocabulary": self.vocabulary,
            "embed_seconds": round(self.embed_seconds, 2),
            "build_seconds": round(self.build_seconds, 2),
            "ingestion": self.ingestion or {},
        }


class RepoIndex:
    def __init__(
        self,
        metadata: RepoMetadata,
        chunks: list[Chunk] | None = None,
        vectors: NumpyVectorStore | None = None,
        lexical: BM25Index | None = None,
        repo_map: str = "",
    ) -> None:
        self.metadata = metadata
        self.chunks: dict[str, Chunk] = {c.chunk_id: c for c in (chunks or [])}
        self.vectors = vectors or NumpyVectorStore()
        self.lexical = lexical or BM25Index()
        self.repo_map = repo_map
        self.stats = IndexStats()

    @classmethod
    async def build(
        cls,
        ingestion: IngestionResult,
        client: LLMClient,
        *,
        settings: Settings | None = None,
    ) -> RepoIndex:
        settings = settings or get_settings()
        tracer = current_tracer()
        started = asyncio.get_event_loop().time()

        index = cls(
            metadata=ingestion.metadata,
            chunks=ingestion.chunks,
            repo_map=ingestion.repo_map,
        )
        chunk_list = list(index.chunks.values())
        if not chunk_list:
            raise ValueError("Nothing to index: ingestion produced zero chunks.")

        with tracer.span("index.embed", chunks=len(chunk_list)) as span:
            embed_started = asyncio.get_event_loop().time()
            texts = [render_for_embedding(c) for c in chunk_list]
            vectors = await client.embed(texts, task_type="RETRIEVAL_DOCUMENT")
            embed_seconds = asyncio.get_event_loop().time() - embed_started

            keep_ids: list[str] = []
            keep_vectors: list[list[float]] = []
            for chunk, vector in zip(chunk_list, vectors, strict=True):
                if vector:
                    keep_ids.append(chunk.chunk_id)
                    keep_vectors.append(vector)
            if not keep_vectors:
                raise ValueError(
                    "Embedding returned no vectors. Check GEMINI_API_KEY and network access."
                )
            index.vectors.add(keep_ids, np.asarray(keep_vectors, dtype=np.float32))
            span.set(embedded=len(keep_ids), dim=index.vectors.dim, seconds=round(embed_seconds, 2))

        with tracer.span("index.lexical", chunks=len(chunk_list)) as span:
            await asyncio.to_thread(
                index.lexical.build,
                [c.chunk_id for c in chunk_list],
                [render_for_lexical(c) for c in chunk_list],
            )
            span.set(vocabulary=len(index.lexical.vocabulary))

        index.stats = IndexStats(
            chunks=len(chunk_list),
            files=ingestion.metadata.num_files,
            vectors=len(index.vectors),
            vocabulary=len(index.lexical.vocabulary),
            embed_seconds=embed_seconds,
            build_seconds=asyncio.get_event_loop().time() - started,
            ingestion=ingestion.stats,
        )
        log.info("index.built", repo=index.metadata.name, **index.stats.as_dict())
        return index

    @classmethod
    async def build_from_source(
        cls,
        source: str,
        client: LLMClient,
        *,
        settings: Settings | None = None,
        branch: str | None = None,
        refresh: bool = False,
    ) -> RepoIndex:
        settings = settings or get_settings()
        ingestion = await IngestionPipeline(settings).run(source, branch=branch, refresh=refresh)
        return await cls.build(ingestion, client, settings=settings)

    @property
    def index_id(self) -> str:
        return slugify(self.metadata.name)

    def save(self, root: Path) -> Path:
        directory = root / self.index_id
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / CHUNKS_FILE).open("wb") as handle:
            for chunk in self.chunks.values():
                handle.write(orjson.dumps(chunk.to_dict()) + b"\n")

        self.vectors.save(directory)
        self.lexical.save(directory)
        (directory / REPO_MAP_FILE).write_text(self.repo_map, encoding="utf-8")
        (directory / MANIFEST_FILE).write_bytes(
            orjson.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "metadata": self.metadata.to_dict(),
                    "stats": self.stats.as_dict(),
                    "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                option=orjson.OPT_INDENT_2,
            )
        )
        log.info("index.saved", path=str(directory), chunks=len(self.chunks))
        return directory

    @classmethod
    def load(cls, directory: Path) -> RepoIndex:
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"No index manifest at {manifest_path}")
        manifest = orjson.loads(manifest_path.read_bytes())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Index at {directory} uses schema v{manifest.get('schema_version')}, "
                f"this build expects v{SCHEMA_VERSION}. Re-index to upgrade."
            )

        chunks: list[Chunk] = []
        chunks_path = directory / CHUNKS_FILE
        if chunks_path.exists():
            with chunks_path.open("rb") as handle:
                chunks = [Chunk.from_dict(orjson.loads(line)) for line in handle if line.strip()]

        repo_map_path = directory / REPO_MAP_FILE
        index = cls(
            metadata=RepoMetadata.from_dict(manifest["metadata"]),
            chunks=chunks,
            vectors=NumpyVectorStore.load(directory),
            lexical=BM25Index.load(directory),
            repo_map=repo_map_path.read_text(encoding="utf-8") if repo_map_path.exists() else "",
        )
        raw_stats = manifest.get("stats", {})
        index.stats = IndexStats(
            **{k: v for k, v in raw_stats.items() if k in IndexStats.__annotations__}
        )
        return index

    @classmethod
    def load_by_name(cls, name: str, settings: Settings | None = None) -> RepoIndex:
        settings = settings or get_settings()
        directory = settings.index_dir / slugify(name)
        if not directory.exists():
            available = ", ".join(i["id"] for i in list_indexes(settings)) or "none"
            raise FileNotFoundError(
                f"No index named '{name}'. Available indexes: {available}. "
                f"Build one with: reposage index <repo>"
            )
        return cls.load(directory)

    @staticmethod
    def delete(name: str, settings: Settings | None = None) -> bool:
        settings = settings or get_settings()
        directory = settings.index_dir / slugify(name)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            return True
        return False

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def paths(self) -> list[str]:
        return sorted({c.path for c in self.chunks.values()})

    def chunks_for_path(self, path: str) -> list[Chunk]:
        return sorted(
            (c for c in self.chunks.values() if c.path == path), key=lambda c: c.start_line
        )

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.index_id,
            "metadata": self.metadata.to_dict(),
            "chunks": len(self.chunks),
            "files": len(self.paths()),
            "vector_store": self.vectors.stats(),
            "lexical": self.lexical.stats(),
            "stats": self.stats.as_dict(),
        }

    def __len__(self) -> int:
        return len(self.chunks)


def render_for_embedding(chunk: Chunk) -> str:
    header = f"{chunk.path} | {chunk.kind.value}: {chunk.qualified_name}"
    return f"{header}\n{chunk.content[:8000]}"


def render_for_lexical(chunk: Chunk) -> str:
    parts = [chunk.path, chunk.qualified_name or "", chunk.symbol or "", chunk.content]
    return "\n".join(p for p in parts if p)


def list_indexes(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    root = settings.index_dir
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir()):
        manifest_path = directory / MANIFEST_FILE
        if not directory.is_dir() or not manifest_path.exists():
            continue
        try:
            manifest = orjson.loads(manifest_path.read_bytes())
        except Exception:
            continue
        metadata = manifest.get("metadata", {})
        found.append(
            {
                "id": directory.name,
                "name": metadata.get("name", directory.name),
                "commit": (metadata.get("commit") or "")[:8],
                "chunks": metadata.get("num_chunks", 0),
                "files": metadata.get("num_files", 0),
                "indexed_at": metadata.get("indexed_at", ""),
                "languages": list(metadata.get("languages", {}))[:5],
                "path": str(directory),
            }
        )
    return found
