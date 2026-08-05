"""Language detection and per-language parsing metadata.

Extension mapping is deliberate rather than heuristic: guessing wrongly sends a
file to the wrong tree-sitter grammar, which silently degrades chunk quality
across the whole index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """How to treat one language during chunking."""

    name: str
    tree_sitter: str | None
    # Node types that represent a self-contained, retrievable declaration.
    declaration_nodes: tuple[str, ...] = ()
    # Node types that carry file-level context worth prepending to every chunk.
    import_nodes: tuple[str, ...] = ()
    # Node types whose children should be split individually when too large.
    container_nodes: tuple[str, ...] = ()
    is_code: bool = True


_PY = LanguageSpec(
    name="python",
    tree_sitter="python",
    declaration_nodes=("function_definition", "class_definition", "decorated_definition"),
    import_nodes=("import_statement", "import_from_statement"),
    container_nodes=("class_definition",),
)
_JS_DECLS = (
    "function_declaration",
    "generator_function_declaration",
    "class_declaration",
    "lexical_declaration",
    "variable_declaration",
    "export_statement",
    "method_definition",
)
_TS_DECLS = (*_JS_DECLS, "interface_declaration", "type_alias_declaration", "enum_declaration")

LANGUAGES: dict[str, LanguageSpec] = {
    ".py": _PY,
    ".pyi": _PY,
    ".js": LanguageSpec(
        "javascript", "javascript", _JS_DECLS, ("import_statement",), ("class_declaration",)
    ),
    ".jsx": LanguageSpec(
        "javascript", "javascript", _JS_DECLS, ("import_statement",), ("class_declaration",)
    ),
    ".mjs": LanguageSpec(
        "javascript", "javascript", _JS_DECLS, ("import_statement",), ("class_declaration",)
    ),
    ".cjs": LanguageSpec(
        "javascript", "javascript", _JS_DECLS, ("import_statement",), ("class_declaration",)
    ),
    ".ts": LanguageSpec(
        "typescript", "typescript", _TS_DECLS, ("import_statement",), ("class_declaration",)
    ),
    ".tsx": LanguageSpec("tsx", "tsx", _TS_DECLS, ("import_statement",), ("class_declaration",)),
    ".go": LanguageSpec(
        "go",
        "go",
        ("function_declaration", "method_declaration", "type_declaration", "const_declaration"),
        ("import_declaration",),
    ),
    ".rs": LanguageSpec(
        "rust",
        "rust",
        ("function_item", "struct_item", "enum_item", "impl_item", "trait_item", "mod_item"),
        ("use_declaration",),
        ("impl_item",),
    ),
    ".java": LanguageSpec(
        "java",
        "java",
        (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "method_declaration",
            "record_declaration",
        ),
        ("import_declaration",),
        ("class_declaration",),
    ),
    ".kt": LanguageSpec(
        "kotlin", "kotlin", ("function_declaration", "class_declaration"), ("import_header",)
    ),
    ".rb": LanguageSpec("ruby", "ruby", ("method", "class", "module"), (), ("class", "module")),
    ".php": LanguageSpec(
        "php",
        "php",
        ("function_definition", "class_declaration", "interface_declaration", "trait_declaration"),
        ("namespace_use_declaration",),
        ("class_declaration",),
    ),
    ".c": LanguageSpec(
        "c",
        "c",
        ("function_definition", "struct_specifier", "enum_specifier"),
        ("preproc_include",),
    ),
    ".h": LanguageSpec(
        "c", "c", ("function_definition", "struct_specifier", "declaration"), ("preproc_include",)
    ),
    ".cpp": LanguageSpec(
        "cpp",
        "cpp",
        ("function_definition", "class_specifier", "struct_specifier", "namespace_definition"),
        ("preproc_include",),
        ("class_specifier",),
    ),
    ".cc": LanguageSpec(
        "cpp",
        "cpp",
        ("function_definition", "class_specifier"),
        ("preproc_include",),
        ("class_specifier",),
    ),
    ".hpp": LanguageSpec(
        "cpp",
        "cpp",
        ("function_definition", "class_specifier"),
        ("preproc_include",),
        ("class_specifier",),
    ),
    ".cs": LanguageSpec(
        "csharp",
        "c_sharp",
        (
            "class_declaration",
            "interface_declaration",
            "method_declaration",
            "struct_declaration",
            "record_declaration",
        ),
        ("using_directive",),
        ("class_declaration",),
    ),
    ".scala": LanguageSpec(
        "scala",
        "scala",
        ("class_definition", "object_definition", "function_definition"),
        ("import_declaration",),
    ),
    ".swift": LanguageSpec(
        "swift",
        "swift",
        ("class_declaration", "function_declaration", "protocol_declaration"),
        ("import_declaration",),
    ),
    ".sh": LanguageSpec("shell", "bash", ("function_definition",)),
    ".bash": LanguageSpec("shell", "bash", ("function_definition",)),
    ".sql": LanguageSpec("sql", None),
    ".lua": LanguageSpec("lua", "lua", ("function_declaration",)),
    ".r": LanguageSpec("r", None),
    ".dart": LanguageSpec("dart", None),
    ".vue": LanguageSpec("vue", None),
    ".svelte": LanguageSpec("svelte", None),
    # Structured, non-code files still carry architectural signal.
    ".md": LanguageSpec("markdown", None, is_code=False),
    ".mdx": LanguageSpec("markdown", None, is_code=False),
    ".rst": LanguageSpec("rst", None, is_code=False),
    ".txt": LanguageSpec("text", None, is_code=False),
    ".json": LanguageSpec("json", None, is_code=False),
    ".yaml": LanguageSpec("yaml", None, is_code=False),
    ".yml": LanguageSpec("yaml", None, is_code=False),
    ".toml": LanguageSpec("toml", None, is_code=False),
    ".ini": LanguageSpec("ini", None, is_code=False),
    ".cfg": LanguageSpec("ini", None, is_code=False),
    ".xml": LanguageSpec("xml", None, is_code=False),
    ".html": LanguageSpec("html", None, is_code=False),
    ".css": LanguageSpec("css", None, is_code=False),
    ".scss": LanguageSpec("scss", None, is_code=False),
    ".graphql": LanguageSpec("graphql", None, is_code=False),
    ".proto": LanguageSpec("protobuf", None, is_code=False),
    ".tf": LanguageSpec("terraform", None, is_code=False),
}

# Extension-less files that matter for understanding how a project is built.
SPECIAL_FILENAMES: dict[str, LanguageSpec] = {
    "dockerfile": LanguageSpec("dockerfile", None, is_code=False),
    "makefile": LanguageSpec("make", None, is_code=False),
    "justfile": LanguageSpec("just", None, is_code=False),
    "procfile": LanguageSpec("text", None, is_code=False),
    "readme": LanguageSpec("markdown", None, is_code=False),
    "license": LanguageSpec("text", None, is_code=False),
    ".gitignore": LanguageSpec("text", None, is_code=False),
    ".env.example": LanguageSpec("text", None, is_code=False),
}

_UNKNOWN = LanguageSpec("text", None, is_code=False)


def get_spec(path: str | Path) -> LanguageSpec:
    """Resolve a path to its language spec, falling back to plain text."""
    p = Path(path)
    name = p.name.lower()
    if name in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[name]
    stem = p.stem.lower()
    if stem in SPECIAL_FILENAMES and p.suffix.lower() not in LANGUAGES:
        return SPECIAL_FILENAMES[stem]
    return LANGUAGES.get(p.suffix.lower(), _UNKNOWN)


def detect_language(path: str | Path) -> str:
    return get_spec(path).name


def is_code_language(path: str | Path) -> bool:
    return get_spec(path).is_code


def supported_extensions() -> set[str]:
    return set(LANGUAGES)
