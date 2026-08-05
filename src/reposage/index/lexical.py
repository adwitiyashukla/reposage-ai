"""BM25 lexical retrieval with a code-aware tokenizer.

Dense retrieval alone is weak on exactly the queries developers ask most: exact
identifier lookups. Embeddings map ``ProcessPaymentIntent`` and
``handle_payment`` to nearby points, which is helpful for concept search and
actively harmful when the user typed a specific symbol they want to find.

Two decisions make the lexical side pull its weight:

* **Identifier splitting.** ``getUserByID`` indexes as the whole token *and* as
  ``get``, ``user``, ``by``, ``id``. A query for either form now matches, which
  a naive whitespace tokenizer cannot do.
* **A hand-rolled sparse index.** An inverted index with NumPy postings scores a
  query in a single vectorised pass per term. It also serialises to two small
  files, so an index round-trips without a database.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson

from reposage.logging_setup import get_logger

log = get_logger(__name__)

_SPLIT = re.compile(r"[^A-Za-z0-9_]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")

# Terms so common in source code that they carry no discriminative signal.
STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "is",
        "it",
        "for",
        "on",
        "with",
        "as",
        "by",
        "at",
        "be",
        "this",
        "that",
        "from",
        "are",
        "was",
        "if",
        "else",
        "return",
        "self",
        "true",
        "false",
        "none",
        "null",
        "nil",
        "def",
        "var",
        "let",
        "const",
        "new",
        "import",
        "class",
        "function",
        "func",
        "public",
        "private",
        "static",
        "void",
        "int",
        "str",
        "string",
        "bool",
        "float",
        "list",
        "dict",
        "type",
        "value",
        "data",
    }
)


def tokenize_code(text: str, *, min_length: int = 2, max_tokens: int = 4000) -> list[str]:
    """Tokenize source text, expanding compound identifiers into their parts."""
    tokens: list[str] = []
    for raw in _SPLIT.split(text):
        if not raw:
            continue
        lowered = raw.lower()
        if len(lowered) >= min_length and lowered not in STOPWORDS:
            tokens.append(lowered)
        # Only decompose genuinely compound identifiers.
        if len(raw) > 3 and ("_" in raw or _CAMEL.search(raw)):
            for piece in _CAMEL.sub(" ", raw).replace("_", " ").split():
                for part in _DIGIT_BOUNDARY.sub(" ", piece).split():
                    lowered_part = part.lower()
                    if len(lowered_part) >= min_length and lowered_part not in STOPWORDS:
                        tokens.append(lowered_part)
        if len(tokens) >= max_tokens:
            break
    return tokens


class BM25Index:
    """Okapi BM25 over an inverted index with NumPy postings."""

    POSTINGS_FILE = "bm25.npz"
    META_FILE = "bm25_meta.json"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avg_doc_length: float = 0.0
        self.vocabulary: dict[str, int] = {}
        # term_id -> (document indices, term frequencies)
        self.postings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)

    # ----------------------------------------------------------------- build
    def build(self, doc_ids: list[str], documents: list[str]) -> None:
        """Construct the index. Rebuilding is cheap enough to prefer over updates."""
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must be the same length")
        self.doc_ids = list(doc_ids)
        num_docs = len(documents)
        if num_docs == 0:
            return

        raw_postings: dict[int, dict[int, int]] = defaultdict(dict)
        lengths = np.zeros(num_docs, dtype=np.float32)
        vocabulary: dict[str, int] = {}

        for doc_index, document in enumerate(documents):
            tokens = tokenize_code(document)
            lengths[doc_index] = len(tokens) or 1
            counts: dict[int, int] = defaultdict(int)
            for token in tokens:
                term_id = vocabulary.get(token)
                if term_id is None:
                    term_id = len(vocabulary)
                    vocabulary[token] = term_id
                counts[term_id] += 1
            for term_id, count in counts.items():
                raw_postings[term_id][doc_index] = count

        self.vocabulary = vocabulary
        self.doc_lengths = lengths
        self.avg_doc_length = float(lengths.mean())

        idf = np.zeros(len(vocabulary), dtype=np.float32)
        postings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for term_id, doc_counts in raw_postings.items():
            docs = np.fromiter(doc_counts.keys(), dtype=np.int32, count=len(doc_counts))
            freqs = np.fromiter(doc_counts.values(), dtype=np.float32, count=len(doc_counts))
            order = np.argsort(docs)
            postings[term_id] = (docs[order], freqs[order])
            df = len(doc_counts)
            idf[term_id] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

        self.postings = postings
        self.idf = idf
        log.debug("bm25.built", documents=num_docs, vocabulary=len(vocabulary))

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 40) -> list[tuple[str, float]]:
        """Top-``k`` ``(doc_id, score)`` pairs for ``query``."""
        if not self.doc_ids or not self.postings:
            return []
        terms = tokenize_code(query)
        if not terms:
            return []

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        length_norm = self.k1 * (
            1 - self.b + self.b * (self.doc_lengths / max(self.avg_doc_length, 1e-6))
        )
        matched = 0
        for term in set(terms):
            term_id = self.vocabulary.get(term)
            if term_id is None:
                continue
            docs, freqs = self.postings[term_id]
            contribution = self.idf[term_id] * (freqs * (self.k1 + 1)) / (freqs + length_norm[docs])
            np.add.at(scores, docs, contribution)
            matched += 1
        if matched == 0:
            return []

        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in top if scores[int(i)] > 0]

    # ----------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if not self.postings:
            return
        term_ids = np.array(sorted(self.postings), dtype=np.int32)
        offsets = np.zeros(len(term_ids) + 1, dtype=np.int64)
        doc_parts, freq_parts = [], []
        for position, term_id in enumerate(term_ids):
            docs, freqs = self.postings[int(term_id)]
            doc_parts.append(docs)
            freq_parts.append(freqs)
            offsets[position + 1] = offsets[position] + len(docs)
        np.savez_compressed(
            directory / self.POSTINGS_FILE,
            term_ids=term_ids,
            offsets=offsets,
            docs=np.concatenate(doc_parts) if doc_parts else np.zeros(0, dtype=np.int32),
            freqs=np.concatenate(freq_parts) if freq_parts else np.zeros(0, dtype=np.float32),
            idf=self.idf,
            doc_lengths=self.doc_lengths,
        )
        (directory / self.META_FILE).write_bytes(
            orjson.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "avg_doc_length": self.avg_doc_length,
                    "doc_ids": self.doc_ids,
                    "vocabulary": self.vocabulary,
                }
            )
        )

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        index = cls()
        postings_path = directory / cls.POSTINGS_FILE
        meta_path = directory / cls.META_FILE
        if not postings_path.exists() or not meta_path.exists():
            return index

        meta = orjson.loads(meta_path.read_bytes())
        index.k1 = meta["k1"]
        index.b = meta["b"]
        index.avg_doc_length = meta["avg_doc_length"]
        index.doc_ids = meta["doc_ids"]
        index.vocabulary = meta["vocabulary"]

        with np.load(postings_path) as payload:
            term_ids = payload["term_ids"]
            offsets = payload["offsets"]
            docs = payload["docs"]
            freqs = payload["freqs"]
            index.idf = payload["idf"]
            index.doc_lengths = payload["doc_lengths"]
        index.postings = {
            int(term_id): (docs[offsets[i] : offsets[i + 1]], freqs[offsets[i] : offsets[i + 1]])
            for i, term_id in enumerate(term_ids)
        }
        return index

    def __len__(self) -> int:
        return len(self.doc_ids)

    def stats(self) -> dict[str, object]:
        return {
            "backend": "bm25-okapi",
            "documents": len(self.doc_ids),
            "vocabulary": len(self.vocabulary),
            "avg_doc_length": round(self.avg_doc_length, 1),
        }
