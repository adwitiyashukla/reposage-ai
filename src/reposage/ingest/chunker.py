from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from reposage.ingest.languages import LanguageSpec, get_spec
from reposage.logging_setup import get_logger
from reposage.models import Chunk, ChunkKind

log = get_logger(__name__)

_MIN_CHUNK_CHARS = 24
_MAX_CONTEXT_IMPORT_LINES = 12


def _body_chars(content: str) -> int:
    body = [
        line for line in content.splitlines() if not line.startswith(("// file:", "// imports:"))
    ]
    return len("\n".join(body).strip())


_GRAMMARS: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "bash": ("tree_sitter_bash", "language"),
}


@lru_cache(maxsize=32)
def _get_parser(language: str) -> Any | None:
    entry = _GRAMMARS.get(language)
    if entry is None:
        return None
    module_name, factory = entry
    try:
        import importlib

        from tree_sitter import Language, Parser

        module = importlib.import_module(module_name)
        return Parser(Language(getattr(module, factory)()))
    except Exception as exc:
        log.debug("chunker.parser_unavailable", language=language, error=str(exc)[:140])
        return None


def treesitter_available() -> bool:
    try:
        import tree_sitter
    except ImportError:
        return False
    return _get_parser("python") is not None


def available_grammars() -> list[str]:
    return sorted(name for name in _GRAMMARS if _get_parser(name) is not None)


_KIND_BY_NODE: dict[str, ChunkKind] = {
    "function_definition": ChunkKind.FUNCTION,
    "function_declaration": ChunkKind.FUNCTION,
    "generator_function_declaration": ChunkKind.FUNCTION,
    "function_item": ChunkKind.FUNCTION,
    "method_definition": ChunkKind.METHOD,
    "method_declaration": ChunkKind.METHOD,
    "class_definition": ChunkKind.CLASS,
    "class_declaration": ChunkKind.CLASS,
    "class_specifier": ChunkKind.CLASS,
    "struct_item": ChunkKind.CLASS,
    "struct_specifier": ChunkKind.CLASS,
    "struct_declaration": ChunkKind.CLASS,
    "record_declaration": ChunkKind.CLASS,
    "impl_item": ChunkKind.CLASS,
    "type_declaration": ChunkKind.INTERFACE,
    "interface_declaration": ChunkKind.INTERFACE,
    "trait_item": ChunkKind.INTERFACE,
    "trait_declaration": ChunkKind.INTERFACE,
    "type_alias_declaration": ChunkKind.INTERFACE,
    "enum_declaration": ChunkKind.INTERFACE,
    "enum_item": ChunkKind.INTERFACE,
    "enum_specifier": ChunkKind.INTERFACE,
    "module": ChunkKind.MODULE,
    "mod_item": ChunkKind.MODULE,
    "namespace_definition": ChunkKind.MODULE,
    "object_definition": ChunkKind.CLASS,
}

_NAME_FIELDS = ("name", "declarator", "pattern", "type", "alias")
_IDENTIFIER_TYPES = {
    "identifier",
    "type_identifier",
    "field_identifier",
    "property_identifier",
    "constant",
    "scoped_identifier",
    "dotted_name",
    "name",
}


@dataclass(slots=True)
class ChunkingStats:
    files: int = 0
    chunks: int = 0
    ast_files: int = 0
    fallback_files: int = 0
    skeletons: int = 0

    @property
    def ast_coverage(self) -> float:
        return self.ast_files / self.files if self.files else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "chunks": self.chunks,
            "ast_files": self.ast_files,
            "fallback_files": self.fallback_files,
            "skeleton_chunks": self.skeletons,
            "ast_coverage": round(self.ast_coverage, 4),
        }


class ASTChunker:
    def __init__(self, max_lines: int = 120, overlap_lines: int = 15) -> None:
        self.max_lines = max(20, max_lines)
        self.overlap_lines = max(0, min(overlap_lines, self.max_lines // 2))
        self.stats = ChunkingStats()

    def chunk(self, rel_path: str, content: str, spec: LanguageSpec | None = None) -> list[Chunk]:
        spec = spec or get_spec(rel_path)
        self.stats.files += 1

        if spec.name in ("markdown", "rst"):
            chunks = self._chunk_markdown(rel_path, content, spec)
            self.stats.fallback_files += 1
        else:
            chunks = []
            if spec.tree_sitter:
                parser = _get_parser(spec.tree_sitter)
                if parser is not None:
                    try:
                        chunks = self._chunk_with_ast(rel_path, content, spec, parser)
                        self.stats.ast_files += 1
                    except Exception as exc:
                        log.debug("chunker.ast_failed", path=rel_path, error=str(exc)[:160])
                        chunks = []
            if not chunks:
                chunks = self._chunk_lines(rel_path, content, spec)
                self.stats.fallback_files += 1

        chunks = [c for c in chunks if _body_chars(c.content) >= _MIN_CHUNK_CHARS]
        self.stats.chunks += len(chunks)
        return chunks

    def _chunk_with_ast(
        self, rel_path: str, content: str, spec: LanguageSpec, parser: Any
    ) -> list[Chunk]:
        source = content.encode("utf-8", errors="replace")
        tree = parser.parse(source)
        root = tree.root_node
        lines = content.splitlines()
        if not lines:
            return []

        header = self._file_context(rel_path, spec, root, lines)
        chunks: list[Chunk] = []
        covered: set[int] = set()

        for node in root.children:
            if node.type not in spec.declaration_nodes:
                continue
            target = self._unwrap(node)
            start, end = target.start_point[0], target.end_point[0]
            if end < start:
                continue
            covered.update(range(start, end + 1))
            symbol = self._symbol_name(target)
            kind = _KIND_BY_NODE.get(target.type, ChunkKind.BLOCK)
            span = end - start + 1

            if span > self.max_lines and target.type in spec.container_nodes:
                chunks.extend(
                    self._split_container(rel_path, lines, target, spec, symbol, kind, header)
                )
            elif span > self.max_lines * 2:
                chunks.extend(
                    self._window(rel_path, lines, start, end, spec, kind, symbol, header, None)
                )
            else:
                chunks.append(
                    self._make(rel_path, lines, start, end, spec, kind, symbol, None, header)
                )

        leftovers = self._uncovered_ranges(len(lines), covered, lines)
        for start, end in leftovers:
            chunks.extend(
                self._window(
                    rel_path, lines, start, end, spec, ChunkKind.MODULE, None, header, None
                )
            )

        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    def _split_container(
        self,
        rel_path: str,
        lines: list[str],
        node: Any,
        spec: LanguageSpec,
        parent: str | None,
        kind: ChunkKind,
        header: str,
    ) -> list[Chunk]:
        members: list[Any] = []
        stack = list(node.children)
        while stack:
            child = stack.pop(0)
            if child.type in _KIND_BY_NODE and child.type not in ("class_definition",):
                members.append(child)
            elif child.type in (
                "block",
                "class_body",
                "declaration_list",
                "field_declaration_list",
                "body",
            ):
                stack = list(child.children) + stack
        members = [m for m in members if m.end_point[0] > m.start_point[0]]

        chunks: list[Chunk] = []
        if members:
            skeleton = self._skeleton(node, members, parent)
            if skeleton:
                self.stats.skeletons += 1
                chunks.append(
                    Chunk(
                        path=rel_path,
                        content=f"{header}{skeleton}",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=spec.name,
                        kind=kind,
                        symbol=parent,
                    )
                )
            for member in members:
                start, end = member.start_point[0], member.end_point[0]
                chunks.append(
                    self._make(
                        rel_path,
                        lines,
                        start,
                        end,
                        spec,
                        _KIND_BY_NODE.get(member.type, ChunkKind.METHOD),
                        self._symbol_name(member),
                        parent,
                        header,
                    )
                )
        else:
            chunks.extend(
                self._window(
                    rel_path,
                    lines,
                    node.start_point[0],
                    node.end_point[0],
                    spec,
                    kind,
                    parent,
                    header,
                    None,
                )
            )
        return chunks

    def _skeleton(self, node: Any, members: list[Any], parent: str | None) -> str:
        try:
            decl_line = node.text.decode("utf-8", errors="replace").splitlines()[0]
        except Exception:
            return ""
        parts = [decl_line.rstrip()]
        for member in members[:60]:
            try:
                first = member.text.decode("utf-8", errors="replace").splitlines()[0].strip()
            except Exception:
                continue
            parts.append(f"    {first}")
        if len(members) > 60:
            parts.append(f"    # ... and {len(members) - 60} more members")
        outline = "\n".join(parts)
        return f"# Outline of {parent or 'container'} ({len(members)} members)\n{outline}\n"

    def _chunk_lines(self, rel_path: str, content: str, spec: LanguageSpec) -> list[Chunk]:
        lines = content.splitlines()
        if not lines:
            return []
        header = f"// file: {rel_path} ({spec.name})\n"
        return self._window(
            rel_path, lines, 0, len(lines) - 1, spec, ChunkKind.BLOCK, None, header, None
        )

    def _chunk_markdown(self, rel_path: str, content: str, spec: LanguageSpec) -> list[Chunk]:
        lines = content.splitlines()
        if not lines:
            return []
        heading = re.compile(r"^(#{1,4})\s+(.*)$")
        sections: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            if match := heading.match(line):
                sections.append((i, match.group(2).strip()))
        header = f"// file: {rel_path} (documentation)\n"
        if not sections:
            return self._window(
                rel_path,
                lines,
                0,
                len(lines) - 1,
                spec,
                ChunkKind.DOCUMENTATION,
                None,
                header,
                None,
            )

        chunks: list[Chunk] = []
        if sections[0][0] > 0:
            chunks.extend(
                self._window(
                    rel_path,
                    lines,
                    0,
                    sections[0][0] - 1,
                    spec,
                    ChunkKind.DOCUMENTATION,
                    "preamble",
                    header,
                    None,
                )
            )
        for index, (start, title) in enumerate(sections):
            end = sections[index + 1][0] - 1 if index + 1 < len(sections) else len(lines) - 1
            if end < start:
                continue
            chunks.extend(
                self._window(
                    rel_path, lines, start, end, spec, ChunkKind.DOCUMENTATION, title, header, None
                )
            )
        return chunks

    def _window(
        self,
        rel_path: str,
        lines: list[str],
        start: int,
        end: int,
        spec: LanguageSpec,
        kind: ChunkKind,
        symbol: str | None,
        header: str,
        parent: str | None,
    ) -> list[Chunk]:
        span = end - start + 1
        if span <= self.max_lines:
            return [self._make(rel_path, lines, start, end, spec, kind, symbol, parent, header)]

        chunks: list[Chunk] = []
        cursor = start
        step = max(1, self.max_lines - self.overlap_lines)
        while cursor <= end:
            stop = min(cursor + self.max_lines - 1, end)
            stop = self._snap_boundary(lines, cursor, stop, end)
            chunks.append(
                self._make(rel_path, lines, cursor, stop, spec, kind, symbol, parent, header)
            )
            if stop >= end:
                break
            cursor = max(cursor + step, stop - self.overlap_lines + 1)
        return chunks

    @staticmethod
    def _snap_boundary(lines: list[str], start: int, proposed: int, hard_end: int) -> int:
        window = range(proposed, max(start + 5, proposed - 12), -1)
        for i in window:
            if i >= len(lines) or i > hard_end:
                continue
            if not lines[i].strip():
                return i
            if lines[i] and not lines[i][0].isspace():
                return max(start, i - 1)
        return proposed

    def _make(
        self,
        rel_path: str,
        lines: list[str],
        start: int,
        end: int,
        spec: LanguageSpec,
        kind: ChunkKind,
        symbol: str | None,
        parent: str | None,
        header: str,
    ) -> Chunk:
        body = "\n".join(lines[start : end + 1])
        return Chunk(
            path=rel_path,
            content=f"{header}{body}",
            start_line=start + 1,
            end_line=end + 1,
            language=spec.name,
            kind=kind,
            symbol=symbol,
            parent_symbol=parent,
        )

    def _file_context(self, rel_path: str, spec: LanguageSpec, root: Any, lines: list[str]) -> str:
        parts = [f"// file: {rel_path} ({spec.name})"]
        if spec.import_nodes:
            imports: list[str] = []
            for node in root.children:
                if node.type in spec.import_nodes:
                    text = node.text.decode("utf-8", errors="replace").strip()
                    if text and len(text) < 200:
                        imports.append(text)
                if len(imports) >= _MAX_CONTEXT_IMPORT_LINES:
                    break
            if imports:
                parts.append("// imports: " + "; ".join(imports).replace("\n", " ")[:600])
        return "\n".join(parts) + "\n"

    @staticmethod
    def _unwrap(node: Any) -> Any:
        for _ in range(3):
            if node.type in ("decorated_definition", "export_statement"):
                inner = [c for c in node.children if c.type in _KIND_BY_NODE]
                if inner:
                    node = inner[-1]
                    continue
            break
        return node

    @staticmethod
    def _symbol_name(node: Any) -> str | None:
        for field in _NAME_FIELDS:
            try:
                child = node.child_by_field_name(field)
            except Exception:
                child = None
            if child is not None:
                text = child.text.decode("utf-8", errors="replace").strip()
                text = text.split("(")[0].strip().lstrip("*&")
                if text and len(text) < 120:
                    return text
        for child in node.children:
            if child.type in _IDENTIFIER_TYPES:
                text = child.text.decode("utf-8", errors="replace").strip()
                if text:
                    return text[:120]
        return None

    @staticmethod
    def _uncovered_ranges(
        total_lines: int, covered: set[int], lines: list[str], merge_gap: int = 4
    ) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i in range(total_lines):
            if i not in covered and lines[i].strip():
                if start is None:
                    start = i
            elif start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, total_lines - 1))

        merged: list[tuple[int, int]] = []
        for run_start, run_end in runs:
            if merged and run_start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], run_end)
            else:
                merged.append((run_start, run_end))
        return merged


def chunk_file(
    rel_path: str, content: str, max_lines: int = 120, overlap_lines: int = 15
) -> list[Chunk]:
    return ASTChunker(max_lines=max_lines, overlap_lines=overlap_lines).chunk(rel_path, content)
