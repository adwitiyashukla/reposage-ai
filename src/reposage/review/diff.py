from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+)$")


class LineType(str, Enum):
    CONTEXT = "context"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(slots=True)
class DiffLine:
    type: LineType
    content: str
    old_line: int | None
    new_line: int | None
    position: int

    @property
    def is_change(self) -> bool:
        return self.type is not LineType.CONTEXT


@dataclass(slots=True)
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    heading: str = ""
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def header(self) -> str:
        return f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"

    def render(self) -> str:
        prefix = {LineType.ADDED: "+", LineType.REMOVED: "-", LineType.CONTEXT: " "}
        body = "\n".join(f"{prefix[line.type]}{line.content}" for line in self.lines)
        return f"{self.header}{(' ' + self.heading) if self.heading else ''}\n{body}"


@dataclass(slots=True)
class DiffFile:
    path: str
    old_path: str | None = None
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    is_binary: bool = False
    hunks: list[DiffHunk] = field(default_factory=list)

    @property
    def added_lines(self) -> list[DiffLine]:
        return [ln for h in self.hunks for ln in h.lines if ln.type is LineType.ADDED]

    @property
    def removed_lines(self) -> list[DiffLine]:
        return [ln for h in self.hunks for ln in h.lines if ln.type is LineType.REMOVED]

    @property
    def additions(self) -> int:
        return len(self.added_lines)

    @property
    def deletions(self) -> int:
        return len(self.removed_lines)

    @property
    def changed_line_numbers(self) -> set[int]:
        return {ln.new_line for ln in self.added_lines if ln.new_line is not None}

    def position_for_line(self, new_line: int) -> int | None:
        for hunk in self.hunks:
            for line in hunk.lines:
                if line.new_line == new_line:
                    return line.position
        return None

    def render(self, max_chars: int = 24_000) -> str:
        header = f"--- a/{self.old_path or self.path}\n+++ b/{self.path}"
        body = "\n".join(h.render() for h in self.hunks)
        out = f"{header}\n{body}"
        return out if len(out) <= max_chars else out[:max_chars] + "\n... (diff truncated)"

    def search_terms(self) -> list[str]:
        terms: list[str] = []
        pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
        for line in self.added_lines + self.removed_lines:
            terms.extend(pattern.findall(line.content))
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [term for term, _ in ranked[:12]]


@dataclass(slots=True)
class ParsedDiff:
    files: list[DiffFile] = field(default_factory=list)

    @property
    def additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def deletions(self) -> int:
        return sum(f.deletions for f in self.files)

    @property
    def reviewable(self) -> list[DiffFile]:
        return [f for f in self.files if not f.is_deleted and not f.is_binary and f.hunks]

    def stats(self) -> dict[str, int]:
        return {
            "files": len(self.files),
            "reviewable": len(self.reviewable),
            "additions": self.additions,
            "deletions": self.deletions,
        }

    def get(self, path: str) -> DiffFile | None:
        return next((f for f in self.files if f.path == path), None)


def parse_unified_diff(text: str) -> ParsedDiff:
    parsed = ParsedDiff()
    current: DiffFile | None = None
    hunk: DiffHunk | None = None
    old_no = new_no = 0
    position = 0

    for raw in (text or "").splitlines():
        if match := _DIFF_GIT.match(raw):
            current = DiffFile(path=match.group(2), old_path=match.group(1))
            parsed.files.append(current)
            hunk = None
            continue

        if current is None:
            if raw.startswith("--- "):
                current = DiffFile(path="unknown", old_path=_strip_prefix(raw[4:]))
                parsed.files.append(current)
            continue

        if raw.startswith("new file mode"):
            current.is_new = True
            continue
        if raw.startswith("deleted file mode"):
            current.is_deleted = True
            continue
        if raw.startswith(("rename from", "rename to")):
            current.is_renamed = True
            continue
        if raw.startswith("Binary files") or raw.startswith("GIT binary patch"):
            current.is_binary = True
            continue
        if raw.startswith("--- "):
            stripped = _strip_prefix(raw[4:])
            current.old_path = None if stripped == "/dev/null" else stripped
            current.is_new = current.is_new or stripped == "/dev/null"
            continue
        if raw.startswith("+++ "):
            stripped = _strip_prefix(raw[4:])
            if stripped == "/dev/null":
                current.is_deleted = True
            elif current.path in ("unknown", ""):
                current.path = stripped
            continue

        if match := _HUNK_HEADER.match(raw):
            hunk = DiffHunk(
                old_start=int(match.group(1)),
                old_count=int(match.group(2) or 1),
                new_start=int(match.group(3)),
                new_count=int(match.group(4) or 1),
                heading=match.group(5).strip(),
            )
            current.hunks.append(hunk)
            old_no = hunk.old_start
            new_no = hunk.new_start
            position = 0
            continue

        if hunk is None:
            continue

        if raw.startswith("\\"):
            continue

        position += 1
        marker, content = (raw[0], raw[1:]) if raw else (" ", "")
        if marker == "+":
            hunk.lines.append(DiffLine(LineType.ADDED, content, None, new_no, position))
            new_no += 1
        elif marker == "-":
            hunk.lines.append(DiffLine(LineType.REMOVED, content, old_no, None, position))
            old_no += 1
        else:
            hunk.lines.append(DiffLine(LineType.CONTEXT, content, old_no, new_no, position))
            old_no += 1
            new_no += 1

    return parsed


def _strip_prefix(path: str) -> str:
    path = path.split("\t")[0].strip()
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[2:]
    return path
