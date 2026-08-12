from __future__ import annotations

import pytest

from reposage.ingest.chunker import ASTChunker, available_grammars, chunk_file, treesitter_available
from reposage.models import ChunkKind
from tests.conftest import SAMPLE_PYTHON

needs_grammar = pytest.mark.skipif(
    not treesitter_available(), reason="tree-sitter grammars are not installed"
)


@needs_grammar
def test_python_chunks_align_to_declarations():
    chunks = chunk_file("auth/jwt.py", SAMPLE_PYTHON)
    symbols = {c.symbol for c in chunks if c.symbol}
    assert "TokenValidator" in symbols
    assert "build_validator" in symbols
    assert all(c.start_line <= c.end_line for c in chunks)
    assert all(c.path == "auth/jwt.py" for c in chunks)


@needs_grammar
def test_large_class_splits_into_members_with_a_skeleton():
    chunker = ASTChunker(max_lines=12, overlap_lines=2)
    chunks = chunker.chunk("auth/jwt.py", SAMPLE_PYTHON)
    methods = [c for c in chunks if c.parent_symbol == "TokenValidator"]
    assert methods, "expected the oversized class to be split per method"
    assert {"verify_jwt", "refresh"} <= {c.symbol for c in methods}
    assert chunker.stats.skeletons >= 1


@needs_grammar
def test_context_header_carries_path_and_imports():
    chunks = chunk_file("auth/jwt.py", SAMPLE_PYTHON)
    body = next(c for c in chunks if c.symbol == "build_validator")
    assert body.content.startswith("// file: auth/jwt.py (python)")
    assert "import os" in body.content.split("\n")[1]


@needs_grammar
def test_javascript_declarations_are_recognised():
    source = (
        "import express from 'express';\n\n"
        "export function createServer(port) {\n  return express().listen(port);\n}\n\n"
        "export class Router {\n  dispatch(path) { return path; }\n}\n"
    )
    symbols = {c.symbol for c in chunk_file("web/app.js", source) if c.symbol}
    assert "createServer" in symbols or "Router" in symbols


def test_unknown_language_falls_back_without_raising():
    chunks = chunk_file("data/notes.xyz", "line one\n" * 400)
    assert chunks
    assert all(c.kind is ChunkKind.BLOCK for c in chunks)


def test_markdown_splits_on_headings():
    doc = "# Title\n\nIntro paragraph here.\n\n## Install\n\nRun the installer now.\n\n## Usage\n\nCall the function.\n"
    chunks = chunk_file("README.md", doc)
    assert all(c.kind is ChunkKind.DOCUMENTATION for c in chunks)
    assert {"Install", "Usage"} <= {c.symbol for c in chunks if c.symbol}


def test_empty_and_trivial_input_is_safe():
    assert chunk_file("empty.py", "") == []
    assert chunk_file("tiny.py", "x=1") == []


def test_chunk_ids_are_deterministic_and_unique():
    first = chunk_file("auth/jwt.py", SAMPLE_PYTHON)
    second = chunk_file("auth/jwt.py", SAMPLE_PYTHON)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_windows_respect_the_line_budget():
    chunker = ASTChunker(max_lines=30, overlap_lines=5)
    chunks = chunker.chunk("big.txt", "\n".join(f"line {i}" for i in range(400)))
    assert chunks
    assert max(c.end_line - c.start_line + 1 for c in chunks) <= 30


def test_grammar_registry_reports_what_is_loadable():
    grammars = available_grammars()
    assert isinstance(grammars, list)
    if treesitter_available():
        assert "python" in grammars
