from __future__ import annotations

import pytest
from evals.dataset import DEFAULT_DATASET, load_dataset
from evals.metrics import (
    aggregate_retrieval,
    citation_validity,
    evaluate_retrieval,
    keyword_coverage,
)

from reposage.models import Citation


class TestRetrievalMetrics:
    def test_perfect_retrieval(self):
        result = evaluate_retrieval("c1", ["a.py", "b.py"], ["a.py", "b.py"])
        assert result.recall_at_k == 1.0 and result.mrr == 1.0 and result.hit

    def test_complete_miss(self):
        result = evaluate_retrieval("c1", ["x.py"], ["a.py"])
        assert result.recall_at_k == 0.0 and not result.hit
        assert result.missed_paths == ["a.py"]

    def test_mrr_reflects_the_rank_of_the_first_hit(self):
        result = evaluate_retrieval("c1", ["x.py", "y.py", "a.py"], ["a.py"])
        assert result.mrr == pytest.approx(1 / 3)
        assert result.first_relevant_rank == 3

    def test_suffix_paths_match(self):
        assert evaluate_retrieval("c1", ["src/auth/jwt.py"], ["auth/jwt.py"]).hit

    def test_duplicates_do_not_inflate_precision(self):
        result = evaluate_retrieval("c1", ["a.py", "a.py", "b.py"], ["a.py"])
        assert result.retrieved == 2

    def test_ndcg_rewards_earlier_hits(self):
        early = evaluate_retrieval("c1", ["a.py", "z.py"], ["a.py"]).ndcg
        late = evaluate_retrieval("c1", ["z.py", "a.py"], ["a.py"]).ndcg
        assert early > late

    def test_no_ground_truth_is_scored_as_neutral(self):
        assert evaluate_retrieval("c1", ["a.py"], []).recall_at_k == 0.0

    def test_aggregation(self):
        results = [
            evaluate_retrieval("a", ["a.py"], ["a.py"]),
            evaluate_retrieval("b", ["z.py"], ["b.py"]),
        ]
        aggregate = aggregate_retrieval(results)
        assert aggregate["cases"] == 2 and aggregate["hit_rate"] == 0.5


class TestAnswerMetrics:
    def test_citation_validity(self):
        citations = [
            Citation(path="a.py", start_line=1, end_line=2),
            Citation(path="ghost.py", start_line=1, end_line=2),
        ]
        assert citation_validity(citations, {"a.py"})["valid_rate"] == 0.5

    def test_no_citations_scores_zero(self):
        assert citation_validity([], {"a.py"})["valid_rate"] == 0.0

    def test_keyword_coverage(self):
        assert keyword_coverage("uses tree-sitter for parsing", ["tree-sitter", "parsing"]) == 1.0
        assert keyword_coverage("uses tree-sitter", ["tree-sitter", "bm25"]) == 0.5
        assert keyword_coverage("anything", []) == 1.0

    def test_coverage_is_case_insensitive(self):
        assert keyword_coverage("Uses BM25 ranking", ["bm25"]) == 1.0


class TestGoldenDataset:
    def test_dataset_loads(self):
        cases = load_dataset()
        assert len(cases) >= 10

    def test_case_ids_are_unique(self):
        ids = [case.id for case in load_dataset()]
        assert len(ids) == len(set(ids))

    def test_every_case_is_well_formed(self):
        for case in load_dataset():
            assert case.question.strip(), case.id
            assert case.expected_paths, f"{case.id} has no ground truth"
            assert case.reference_answer.strip(), case.id

    def test_expected_paths_exist_in_this_repository(self):
        root = DEFAULT_DATASET.parent.parent.parent
        missing = [
            (case.id, path)
            for case in load_dataset()
            for path in case.expected_paths
            if not (root / path).exists()
        ]
        assert not missing, f"dataset references files that no longer exist: {missing}"

    def test_limit_is_honoured(self):
        assert len(load_dataset(limit=3)) == 3
