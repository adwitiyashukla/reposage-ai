from __future__ import annotations

import pytest

from reposage.agents.engine import CodebaseAgent
from reposage.agents.graph import route_after_critique, route_after_plan
from reposage.index.store import RepoIndex, list_indexes, render_for_embedding
from reposage.ingest.pipeline import IngestionPipeline, build_repo_map
from reposage.ingest.walker import walk_repository
from reposage.models import Critique, QueryPlan
from reposage.observability import Tracer, use_tracer


class TestWalker:
    def test_excludes_dependency_directories(self, sample_repo):
        paths = {f.rel_path for f in walk_repository(sample_repo)}
        assert not any(p.startswith("node_modules/") for p in paths)

    def test_excludes_lockfiles_and_binaries(self, sample_repo):
        paths = {f.rel_path for f in walk_repository(sample_repo)}
        assert "package-lock.json" not in paths
        assert "logo.png" not in paths

    def test_keeps_real_source_and_docs(self, sample_repo):
        paths = {f.rel_path for f in walk_repository(sample_repo)}
        assert {"auth/jwt.py", "web/app.js", "README.md"} <= paths

    def test_detects_languages(self, sample_repo):
        by_path = {f.rel_path: f.language for f in walk_repository(sample_repo)}
        assert by_path["auth/jwt.py"] == "python"
        assert by_path["web/app.js"] == "javascript"
        assert by_path["README.md"] == "markdown"

    def test_file_budget_is_respected(self, sample_repo):
        assert len(walk_repository(sample_repo, max_files=2)) == 2


class TestIngestion:
    async def test_produces_chunks_and_a_repo_map(self, sample_repo, settings):
        with use_tracer(Tracer()):
            result = await IngestionPipeline(settings).run(str(sample_repo))
        assert result.num_chunks > 0
        assert result.metadata.num_files >= 3
        assert "REPOSITORY MAP" in result.repo_map
        assert "python" in result.metadata.languages

    async def test_empty_directory_raises_a_clear_error(self, tmp_path, settings):
        empty = tmp_path / "empty-repo"
        empty.mkdir()
        with pytest.raises(ValueError, match="No indexable files"), use_tracer(Tracer()):
            await IngestionPipeline(settings).run(str(empty))

    def test_repo_map_lists_symbols(self, sample_repo, settings):
        from reposage.ingest.chunker import ASTChunker

        files = walk_repository(sample_repo)
        chunker = ASTChunker()
        chunks = [c for f in files for c in chunker.chunk(f.rel_path, f.content, f.spec)]
        repo_map = build_repo_map(files, chunks)
        assert "auth/" in repo_map and "jwt.py" in repo_map


class TestIndexPersistence:
    @pytest.fixture
    async def built_index(self, sample_repo, settings, client) -> RepoIndex:
        with use_tracer(Tracer()):
            ingestion = await IngestionPipeline(settings).run(str(sample_repo))
            return await RepoIndex.build(ingestion, client, settings=settings)

    async def test_index_contains_vectors_and_vocabulary(self, built_index):
        assert len(built_index) > 0
        assert len(built_index.vectors) == len(built_index)
        assert len(built_index.lexical.vocabulary) > 0

    async def test_round_trips_through_disk(self, built_index, settings):
        path = built_index.save(settings.index_dir)
        restored = RepoIndex.load(path)
        assert len(restored) == len(built_index)
        assert len(restored.vectors) == len(built_index.vectors)
        assert restored.metadata.name == built_index.metadata.name
        assert restored.repo_map == built_index.repo_map

    async def test_listing_reports_the_saved_index(self, built_index, settings):
        built_index.save(settings.index_dir)
        entries = list_indexes(settings)
        assert entries and entries[0]["chunks"] == len(built_index)

    async def test_delete_removes_it(self, built_index, settings):
        built_index.save(settings.index_dir)
        assert RepoIndex.delete(built_index.index_id, settings)
        assert list_indexes(settings) == []

    async def test_missing_index_error_names_the_alternatives(self, settings):
        settings.ensure_dirs()
        with pytest.raises(FileNotFoundError, match="reposage index"):
            RepoIndex.load_by_name("does-not-exist", settings)

    async def test_embedding_text_includes_the_symbol(self, built_index):
        chunk = next(c for c in built_index.chunks.values() if c.symbol)
        assert chunk.qualified_name in render_for_embedding(chunk)


class TestGraphRouting:
    def test_plan_skips_retrieval_when_the_map_suffices(self):
        assert route_after_plan({"plan": QueryPlan(needs_retrieval=False)}) == "analyse"

    def test_plan_retrieves_by_default(self):
        assert route_after_plan({"plan": QueryPlan(needs_retrieval=True)}) == "retrieve"
        assert route_after_plan({}) == "retrieve"

    def test_critique_loops_only_with_concrete_follow_ups(self):
        refine = Critique(verdict="refine", follow_up_queries=["where is X defined"])
        assert route_after_critique({"critique": refine}) == "retrieve"
        assert route_after_critique({"critique": Critique(verdict="refine")}) == "finalise"
        assert route_after_critique({"critique": Critique(verdict="accept")}) == "finalise"


class TestAgentEndToEnd:
    @pytest.fixture
    async def agent(self, sample_repo, settings, client) -> CodebaseAgent:
        with use_tracer(Tracer()):
            ingestion = await IngestionPipeline(settings).run(str(sample_repo))
            index = await RepoIndex.build(ingestion, client, settings=settings)
        return CodebaseAgent(index, client, settings)

    async def test_answers_with_verified_citations(self, agent):
        tracer = Tracer()
        with use_tracer(tracer):
            answer = await agent.ask("How does token validation work?", tracer=tracer)
        assert answer.answer
        assert answer.citations, "expected at least one resolvable citation"
        assert all(c.path in set(agent.index.paths()) for c in answer.citations)

    async def test_hallucinated_citations_are_discarded(self, agent):
        tracer = Tracer()
        with use_tracer(tracer):
            answer = await agent.ask("How does token validation work?", tracer=tracer)
        assert all(c.path != "does/not/exist.py" for c in answer.citations)
        assert answer.confidence < 1.0

    async def test_usage_and_trace_are_recorded(self, agent):
        tracer = Tracer()
        with use_tracer(tracer):
            answer = await agent.ask("How does token validation work?", tracer=tracer)
        assert answer.usage.llm_calls > 0
        assert answer.elapsed_seconds >= 0
        span_names = {span["name"] for span in tracer.waterfall()}
        assert {"agent.plan", "agent.retrieve", "agent.analyse", "agent.finalise"} <= span_names

    async def test_streaming_emits_events_then_a_final_result(self, agent):
        events = [event async for event in agent.astream("How does token validation work?")]
        assert events[-1]["type"] == "final"
        assert events[-1]["attributes"]["answer"]
        assert any(event["type"] == "token" for event in events)
        assert any(event["type"] == "span_start" for event in events)

    async def test_empty_question_is_rejected(self, agent):
        with pytest.raises(ValueError, match="must not be empty"):
            await agent.ask("   ")


class TestConfidenceCalibration:
    def test_confidence_never_reaches_certainty(self):
        from reposage.agents.nodes.finalizer import _CONFIDENCE_CEILING, _score
        from reposage.models import Citation, Critique

        citations = [Citation(path=f"f{i}.py", start_line=1, end_line=2) for i in range(8)]
        state = {
            "critique": Critique(grounded=True, complete=True, confidence=1.0, verdict="accept"),
            "errors": [],
        }
        score = _score(state, citations, invalid=0, retrieved_paths={c.path for c in citations})
        assert score == _CONFIDENCE_CEILING < 1.0

    def test_invalid_citations_reduce_confidence(self):
        from reposage.agents.nodes.finalizer import _score
        from reposage.models import Citation, Critique

        citations = [Citation(path="a.py", start_line=1, end_line=2)]
        state = {"critique": Critique(confidence=0.9), "errors": []}
        clean = _score(state, citations, 0, {"a.py"})
        dirty = _score(state, citations, 3, {"a.py"})
        assert dirty < clean

    def test_ungrounded_verdict_is_penalised(self):
        from reposage.agents.nodes.finalizer import _score
        from reposage.models import Citation, Critique

        citations = [Citation(path="a.py", start_line=1, end_line=2)]
        grounded = _score(
            {"critique": Critique(grounded=True, confidence=0.9)}, citations, 0, {"a.py"}
        )
        floating = _score(
            {"critique": Critique(grounded=False, confidence=0.9)}, citations, 0, {"a.py"}
        )
        assert floating < grounded


class TestCitationParsing:
    def test_single_reference(self):
        from reposage.agents.nodes.finalizer import parse_citation_markers

        assert parse_citation_markers("see [src/a.py:12-48]") == [("src/a.py", 12, 48)]

    def test_single_line_reference(self):
        from reposage.agents.nodes.finalizer import parse_citation_markers

        assert parse_citation_markers("[src/a.py:12]") == [("src/a.py", 12, 12)]

    def test_two_files_in_one_bracket(self):
        from reposage.agents.nodes.finalizer import parse_citation_markers

        assert parse_citation_markers("[src/a.py:3-6, docs/b.md:148-151]") == [
            ("src/a.py", 3, 6),
            ("docs/b.md", 148, 151),
        ]

    def test_second_range_inherits_the_path(self):
        from reposage.agents.nodes.finalizer import parse_citation_markers

        assert parse_citation_markers("[src/a.py:16-18, 45-63]") == [
            ("src/a.py", 16, 18),
            ("src/a.py", 45, 63),
        ]

    def test_prose_in_brackets_is_not_a_citation(self):
        from reposage.agents.nodes.finalizer import parse_citation_markers

        assert parse_citation_markers("[see below, line 4]") == []

    def test_a_bare_range_cannot_inherit_across_prose(self):
        from reposage.agents.nodes.finalizer import parse_citation_markers

        found = parse_citation_markers("[src/a.py:1-2] and later [note, 40-50]")
        assert found == [("src/a.py", 1, 2)]

    def test_reversed_ranges_are_normalised(self):
        from reposage.agents.nodes.finalizer import parse_citation_markers

        assert parse_citation_markers("[src/a.py:48-12]") == [("src/a.py", 12, 48)]

    def test_invalid_citations_are_penalised_proportionally(self):
        from reposage.agents.nodes.finalizer import _score
        from reposage.models import Citation, Critique

        many = [Citation(path=f"f{i}.py", start_line=1, end_line=2) for i in range(20)]
        few = [Citation(path=f"f{i}.py", start_line=1, end_line=2) for i in range(10)]
        state = {"critique": Critique(confidence=0.9), "errors": []}
        long_answer = _score(state, many, invalid=2, retrieved_paths={c.path for c in many})
        short_answer = _score(state, few, invalid=0, retrieved_paths={c.path for c in few})
        assert long_answer >= short_answer * 0.9
