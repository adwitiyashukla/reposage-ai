"""Lexical index, vector store and rank fusion."""

from __future__ import annotations

import numpy as np
import pytest

from reposage.index.fusion import reciprocal_rank_fusion
from reposage.index.lexical import BM25Index, tokenize_code
from reposage.index.vector_store import NumpyVectorStore


class TestTokenizer:
    def test_splits_compound_identifiers(self):
        tokens = tokenize_code("getUserByID")
        assert "getuserbyid" in tokens
        assert {"get", "user", "id"} <= set(tokens)

    def test_splits_snake_case(self):
        assert {"verify_jwt", "verify", "jwt"} <= set(tokenize_code("verify_jwt"))

    def test_drops_stopwords_and_single_characters(self):
        tokens = tokenize_code("the a return self x")
        assert "the" not in tokens and "return" not in tokens and "x" not in tokens

    def test_separates_digits_from_letters(self):
        assert "sha" in tokenize_code("sha256Hash")


class TestBM25:
    @pytest.fixture
    def index(self) -> BM25Index:
        index = BM25Index()
        index.build(
            ["a", "b", "c"],
            [
                "def verify_jwt(token): check the signature and expiry",
                "class UserRepository: fetch users from the database",
                "def render_template(name): produce html output",
            ],
        )
        return index

    def test_exact_identifier_wins(self, index: BM25Index):
        results = index.search("verify_jwt", k=3)
        assert results and results[0][0] == "a"

    def test_decomposed_query_still_matches(self, index: BM25Index):
        assert next(doc for doc, _ in index.search("verify jwt signature", k=3)) == "a"

    def test_unknown_terms_return_nothing(self, index: BM25Index):
        assert index.search("kubernetes helm chart", k=5) == []

    def test_empty_index_is_safe(self):
        assert BM25Index().search("anything", k=5) == []

    def test_round_trips_through_disk(self, index: BM25Index, tmp_path):
        index.save(tmp_path)
        restored = BM25Index.load(tmp_path)
        assert len(restored) == len(index)
        assert restored.vocabulary == index.vocabulary
        assert restored.search("verify_jwt", k=2) == index.search("verify_jwt", k=2)


class TestVectorStore:
    def test_nearest_neighbour_is_the_identical_vector(self):
        store = NumpyVectorStore()
        vectors = np.eye(4, dtype=np.float32)
        store.add(["a", "b", "c", "d"], vectors)
        assert store.search(vectors[2], k=1)[0][0] == "c"

    def test_scores_are_cosine_similarities(self):
        store = NumpyVectorStore()
        store.add(["a"], np.array([[3.0, 4.0]], dtype=np.float32))
        _, score = store.search(np.array([6.0, 8.0], dtype=np.float32), k=1)[0]
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_batched_search_matches_single_search(self):
        rng = np.random.default_rng(7)
        store = NumpyVectorStore()
        store.add([str(i) for i in range(40)], rng.normal(size=(40, 16)).astype(np.float32))
        queries = rng.normal(size=(3, 16)).astype(np.float32)
        batched = store.search_many(queries, k=5)
        for query, expected in zip(queries, batched, strict=True):
            assert [doc for doc, _ in store.search(query, k=5)] == [doc for doc, _ in expected]

    def test_dimension_mismatch_is_rejected(self):
        store = NumpyVectorStore()
        store.add(["a"], np.zeros((1, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="dimension"):
            store.add(["b"], np.zeros((1, 16), dtype=np.float32))

    def test_round_trips_through_disk(self, tmp_path):
        store = NumpyVectorStore()
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(12, 32)).astype(np.float32)
        store.add([f"id{i}" for i in range(12)], vectors)
        store.save(tmp_path)
        restored = NumpyVectorStore.load(tmp_path)
        assert len(restored) == 12 and restored.dim == 32
        assert restored.search(vectors[3], k=1)[0][0] == "id3"

    def test_empty_store_returns_nothing(self):
        assert NumpyVectorStore().search(np.zeros(4, dtype=np.float32), k=3) == []


class TestFusion:
    def test_documents_found_by_both_retrievers_rank_highest(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "lexical": ["c", "a", "d"]})
        assert fused[0].doc_id == "a"
        assert fused[0].agreement == 2

    def test_provenance_records_every_rank(self):
        fused = reciprocal_rank_fusion({"dense": ["x"], "lexical": ["y", "x"]})
        entry = next(f for f in fused if f.doc_id == "x")
        assert entry.ranks == {"dense": 0, "lexical": 1}

    def test_weights_shift_the_ordering(self):
        rankings = {"dense": ["a", "b"], "lexical": ["b", "a"]}
        boosted = reciprocal_rank_fusion(rankings, weights={"lexical": 5.0})
        assert boosted[0].doc_id == "b"

    def test_output_is_deterministic(self):
        rankings = {"dense": ["a", "b", "c"], "lexical": ["c", "b", "a"]}
        first = [f.doc_id for f in reciprocal_rank_fusion(rankings)]
        second = [f.doc_id for f in reciprocal_rank_fusion(rankings)]
        assert first == second

    def test_empty_input_is_safe(self):
        assert reciprocal_rank_fusion({}) == []
