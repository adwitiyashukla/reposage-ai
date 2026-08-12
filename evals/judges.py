from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from reposage.llm.client import ModelTier
from reposage.logging_setup import get_logger

if TYPE_CHECKING:
    from reposage.llm.client import LLMClient

log = get_logger(__name__)

JUDGE_SYSTEM = """You grade answers produced by a code-intelligence system.

You are given a question about a codebase, the answer under test, and where
available a reference answer written by a human who knows the code.

Write your reasoning first, then the scores.

Score three axes independently, each from 1 to 5.

CORRECTNESS - are the technical claims true?
  5  Every claim is accurate; names, paths and behaviour are right.
  4  Accurate overall; one trivial imprecision.
  3  Broadly right with one meaningful error or an unsupported leap.
  2  Several errors, or a central claim is wrong.
  1  Fundamentally wrong, or describes code that does not exist.

GROUNDEDNESS - is it anchored in the codebase rather than invented?
  5  Specific real symbols and paths, cited precisely.
  4  Well grounded; a citation or two is loose.
  3  Partly grounded; some claims float free of any evidence.
  2  Mostly generic; could describe any project.
  1  Hallucinated files, functions or APIs.

COMPLETENESS - does it answer what was asked?
  5  Fully answers every part, including edge cases worth knowing.
  4  Answers the question; a secondary aspect is thin.
  3  Answers the main thrust but omits a requested part.
  2  Touches the topic without answering it.
  1  Does not address the question.

Judge substance, not length or polish. A short precise answer beats a long vague
one. An answer that correctly says "this is not in the retrieved context" should
score 4-5 on correctness and groundedness, and low on completeness only if the
information was in fact available."""


class JudgeVerdict(BaseModel):
    reasoning: str = Field(default="", description="Written before the scores.")
    correctness: int = Field(default=3, ge=1, le=5)
    groundedness: int = Field(default=3, ge=1, le=5)
    completeness: int = Field(default=3, ge=1, le=5)
    hallucinations: list[str] = Field(
        default_factory=list, description="Specific invented claims, if any."
    )

    @property
    def overall(self) -> float:
        return round(
            (self.correctness * 0.45 + self.groundedness * 0.35 + self.completeness * 0.20), 3
        )

    @property
    def passed(self) -> bool:
        return (
            min(self.correctness, self.groundedness, self.completeness) >= 3 and self.overall >= 3.5
        )


async def judge_answer(
    client: LLMClient,
    question: str,
    answer: str,
    reference: str = "",
    citations: list[str] | None = None,
) -> JudgeVerdict:
    reference_block = (
        f"Reference answer (written by someone who knows the code):\n{reference}\n\n"
        if reference.strip()
        else "No reference answer is available. Judge on internal consistency and specificity.\n\n"
    )
    citation_block = (
        "Citations the answer provided:\n" + "\n".join(f"- {c}" for c in citations) + "\n\n"
        if citations
        else ""
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"{reference_block}"
        f"{citation_block}"
        f"Answer under test:\n{answer[:12_000]}\n\n"
        "Grade it."
    )
    try:
        return await client.structured(
            prompt,
            JudgeVerdict,
            tier=ModelTier.FAST,
            system=JUDGE_SYSTEM,
            temperature=0.0,
            max_output_tokens=1536,
        )
    except Exception as exc:
        log.warning("judge.failed", error=str(exc)[:200])
        return JudgeVerdict(
            reasoning=f"judge unavailable: {exc}", correctness=3, groundedness=3, completeness=3
        )
