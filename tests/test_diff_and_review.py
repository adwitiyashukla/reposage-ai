"""Unified diff parsing and review finding validation."""

from __future__ import annotations

import pytest

from reposage.models import ReviewFinding, Severity
from reposage.review.diff import LineType, parse_unified_diff
from reposage.review.reviewer import _deduplicate, _Finding, _validate


class TestDiffParsing:
    def test_files_and_counts(self, sample_diff: str):
        diff = parse_unified_diff(sample_diff)
        assert diff.stats() == {"files": 2, "reviewable": 2, "additions": 6, "deletions": 1}

    def test_new_file_is_flagged(self, sample_diff: str):
        assert parse_unified_diff(sample_diff).get("docs/README.md").is_new

    def test_added_lines_map_to_new_file_numbers(self, sample_diff: str):
        api = parse_unified_diff(sample_diff).get("src/api.py")
        assert api.changed_line_numbers == {11, 12, 13, 14}

    def test_removed_lines_keep_old_numbers(self, sample_diff: str):
        api = parse_unified_diff(sample_diff).get("src/api.py")
        removed = api.removed_lines
        assert len(removed) == 1 and removed[0].old_line == 11 and removed[0].new_line is None

    def test_context_lines_carry_both_numbers(self, sample_diff: str):
        api = parse_unified_diff(sample_diff).get("src/api.py")
        context = [ln for ln in api.hunks[0].lines if ln.type is LineType.CONTEXT]
        assert context and all(ln.old_line and ln.new_line for ln in context)

    def test_position_lookup_matches_hunk_offset(self, sample_diff: str):
        api = parse_unified_diff(sample_diff).get("src/api.py")
        assert api.position_for_line(11) == 3
        assert api.position_for_line(9999) is None

    def test_search_terms_are_ranked_identifiers(self, sample_diff: str):
        terms = parse_unified_diff(sample_diff).get("src/api.py").search_terms()
        assert "user_id" in terms and len(terms) <= 12

    def test_malformed_input_does_not_raise(self):
        assert parse_unified_diff("").files == []
        assert parse_unified_diff("not a diff at all\njust text").files == []

    def test_deleted_file_is_not_reviewable(self):
        diff = parse_unified_diff(
            "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n"
            "--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-one\n-two\n"
        )
        assert diff.files[0].is_deleted and diff.reviewable == []

    def test_binary_file_is_skipped(self):
        diff = parse_unified_diff(
            "diff --git a/logo.png b/logo.png\nBinary files a/logo.png and b/logo.png differ\n"
        )
        assert diff.files[0].is_binary and diff.reviewable == []


class TestFindingValidation:
    @pytest.fixture
    def diff_file(self, sample_diff: str):
        return parse_unified_diff(sample_diff).get("src/api.py")

    def test_low_confidence_findings_are_dropped(self, diff_file):
        raw = [_Finding(line=11, title="Maybe", body="Not sure", confidence=0.2)]
        assert _validate(raw, diff_file, 0.5) == []

    def test_anchor_snaps_to_a_nearby_changed_line(self, diff_file):
        raw = [_Finding(line=15, title="Off by two", body="x", confidence=0.9)]
        assert _validate(raw, diff_file, 0.5)[0].line == 14

    def test_far_anchor_becomes_file_level(self, diff_file):
        raw = [_Finding(line=900, title="Way off", body="x", confidence=0.9)]
        assert _validate(raw, diff_file, 0.5)[0].line is None

    def test_unknown_severity_defaults_to_medium(self, diff_file):
        raw = [_Finding(line=11, severity="catastrophic", title="t", body="b", confidence=0.9)]
        assert _validate(raw, diff_file, 0.5)[0].severity is Severity.MEDIUM

    def test_findings_sort_by_severity(self, diff_file):
        raw = [
            _Finding(line=11, severity="nit", title="style", body="b", confidence=0.9),
            _Finding(line=12, severity="critical", title="crash", body="b", confidence=0.9),
        ]
        assert _validate(raw, diff_file, 0.5)[0].severity is Severity.CRITICAL

    def test_duplicates_collapse_keeping_the_confident_one(self):
        findings = [
            ReviewFinding(path="a.py", line=1, title="Same issue", body="x", confidence=0.6),
            ReviewFinding(path="a.py", line=1, title="Same issue", body="y", confidence=0.9),
        ]
        deduplicated = _deduplicate(findings)
        assert len(deduplicated) == 1 and deduplicated[0].confidence == 0.9


def test_severity_ordering_and_markdown():
    assert Severity.CRITICAL.rank < Severity.HIGH.rank < Severity.NIT.rank
    markdown = ReviewFinding(
        path="a.py",
        line=4,
        severity=Severity.HIGH,
        title="Bug",
        body="Explanation.",
        suggestion="fixed = True",
    ).to_markdown()
    assert "HIGH" in markdown and "a.py" in markdown and "```suggestion" in markdown
