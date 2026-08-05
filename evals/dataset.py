"""Golden dataset loading and validation.

A case pairs a realistic question with the files that must be retrieved to
answer it and the facts a correct answer must contain. Keeping ground truth at
file granularity (rather than chunk granularity) matters: chunk boundaries move
whenever the chunker is tuned, and a dataset that has to be rewritten every time
you improve the system is a dataset nobody maintains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "golden.jsonl"


@dataclass(slots=True)
class EvalCase:
    """One evaluated question."""

    id: str
    question: str
    expected_paths: list[str] = field(default_factory=list)
    expected_symbols: list[str] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)
    reference_answer: str = ""
    category: str = "general"
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalCase:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_paths": self.expected_paths,
            "expected_symbols": self.expected_symbols,
            "must_mention": self.must_mention,
            "reference_answer": self.reference_answer,
            "category": self.category,
            "difficulty": self.difficulty,
        }


def load_dataset(path: Path | None = None, limit: int | None = None) -> list[EvalCase]:
    """Read a JSONL dataset, skipping blanks and comment lines."""
    path = path or DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}. Expected a JSONL file with one case per line."
        )
    cases: list[EvalCase] = []
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith(b"//") or line.startswith(b"#"):
                continue
            try:
                cases.append(EvalCase.from_dict(orjson.loads(line)))
            except Exception as exc:  # pragma: no cover - dataset authoring error
                raise ValueError(f"{path}:{number} is not a valid case: {exc}") from exc
    if not cases:
        raise ValueError(f"{path} contains no cases.")
    return cases[:limit] if limit else cases


def save_dataset(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for case in cases:
            handle.write(orjson.dumps(case.to_dict()) + b"\n")
