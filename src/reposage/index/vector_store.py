from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from reposage.logging_setup import get_logger

log = get_logger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def save(self, directory: Path) -> None: ...

    @classmethod
    def load(cls, directory: Path) -> VectorStore: ...

    def __len__(self) -> int: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class NumpyVectorStore:
    VECTORS_FILE = "vectors.npy"
    IDS_FILE = "vector_ids.txt"
    LEGACY_IDS_FILE = "vector_ids.npy"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError(f"id/vector length mismatch: {len(ids)} vs {len(vectors)}")
        if not ids:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {matrix.shape}")
        if self.dim is None:
            self.dim = matrix.shape[1]
        elif matrix.shape[1] != self.dim:
            raise ValueError(f"dimension mismatch: expected {self.dim}, got {matrix.shape[1]}")

        matrix = _l2_normalise(matrix)
        self._matrix = matrix if self._matrix is None else np.vstack([self._matrix, matrix])
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None or not self._ids:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"query dimension {vector.shape[0]} != index dimension {self._matrix.shape[1]}"
            )
        norm = float(np.linalg.norm(vector)) or 1e-12
        scores = self._matrix @ (vector / norm)

        k = min(k, scores.shape[0])
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in top]

    def search_many(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        if self._matrix is None or not self._ids:
            return [[] for _ in range(len(queries))]
        batch = _l2_normalise(np.asarray(queries, dtype=np.float32))
        scores = batch @ self._matrix.T
        k = min(k, scores.shape[1])
        results: list[list[tuple[str, float]]] = []
        for row in scores:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            results.append([(self._ids[int(i)], float(row[int(i)])) for i in top])
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._matrix is None:
            return
        np.save(directory / self.VECTORS_FILE, self._matrix)
        (directory / self.IDS_FILE).write_text("\n".join(self._ids), encoding="utf-8")
        log.debug("vector_store.saved", vectors=len(self._ids), dim=self.dim)

    @classmethod
    def load(cls, directory: Path) -> NumpyVectorStore:
        store = cls()
        vectors_path = directory / cls.VECTORS_FILE
        ids_path = directory / cls.IDS_FILE
        legacy_path = directory / cls.LEGACY_IDS_FILE

        if not vectors_path.exists() or not (ids_path.exists() or legacy_path.exists()):
            return store

        matrix = np.load(vectors_path).astype(np.float32, copy=False)
        store._matrix = matrix
        if ids_path.exists():
            store._ids = ids_path.read_text(encoding="utf-8").splitlines()
        else:
            store._ids = [str(i) for i in np.load(legacy_path, allow_pickle=True).tolist()]
            log.info("vector_store.legacy_ids", path=str(legacy_path))
        store.dim = int(matrix.shape[1]) if matrix.size else None
        return store

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def memory_mb(self) -> float:
        return round(self._matrix.nbytes / 1024**2, 2) if self._matrix is not None else 0.0

    def stats(self) -> dict[str, object]:
        return {
            "backend": "numpy-exact",
            "vectors": len(self._ids),
            "dim": self.dim,
            "memory_mb": self.memory_mb,
        }
