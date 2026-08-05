"""Markdown report rendering.

The report is written to be committed. A reviewer opening the diff should be
able to see, without running anything, that recall moved from 0.62 to 0.81 and
what it cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from evals.harness import EvalReport


def _bar(value: float, width: int = 18) -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return "#" * filled + "." * (width - filled)


def render_report(report: EvalReport) -> str:
    """Render a full evaluation run as markdown."""
    summary = report.summary
    lines: list[str] = [
        "# RepoSage evaluation report",
        "",
        f"**Repository** `{report.repo}` @ `{report.config.get('commit', 'unknown')}`  ",
        f"**Generated** {report.generated_at}  ",
        f"**Prompt version** `{report.prompt_version}`  ",
        f"**Dataset** {report.dataset_size} cases  ",
        f"**Index** {report.config.get('index_files', 0):,} files / "
        f"{report.config.get('index_chunks', 0):,} chunks",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]

    headline = [
        ("Answer pass rate (LLM judge)", _pct(summary.get("pass_rate"))),
        ("Judge: correctness", _score(summary.get("judge_correctness"))),
        ("Judge: groundedness", _score(summary.get("judge_groundedness"))),
        ("Judge: completeness", _score(summary.get("judge_completeness"))),
        ("Retrieval recall@k (agent)", _pct(summary.get("answer_recall@k"))),
        ("Citation validity", _pct(summary.get("citation_valid_rate"))),
        ("Required-fact coverage", _pct(summary.get("fact_coverage"))),
        ("Mean citations per answer", _num(summary.get("mean_citations"))),
        ("Cases with hallucinations", str(summary.get("hallucination_cases", "n/a"))),
        ("Mean latency", f"{summary.get('mean_latency_s', 0):.1f}s"),
        ("Mean cost per answer", f"${summary.get('mean_cost_usd', 0):.5f}"),
        ("Mean tokens per answer", f"{summary.get('mean_tokens', 0):,}"),
        ("Total run cost", f"${report.total_cost_usd:.4f}"),
    ]
    lines += [f"| {name} | {value} |" for name, value in headline]

    if report.ablation:
        lines += [
            "",
            "## Retrieval ablation",
            "",
            "Every configuration runs the same questions through the production retrieval",
            "path, so the differences below are attributable to the components themselves.",
            "",
            "| Configuration | hit rate | recall@k | precision@k | MRR | nDCG | time |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for variant in report.ablation:
            aggregate = variant.aggregate
            lines.append(
                f"| **{variant.name}** | {_pct(aggregate.get('hit_rate'))} | "
                f"{_pct(aggregate.get('recall@k'))} | {_pct(aggregate.get('precision@k'))} | "
                f"{_num(aggregate.get('mrr'))} | {_num(aggregate.get('ndcg'))} | "
                f"{variant.elapsed_seconds:.1f}s |"
            )
        lift = summary.get("retrieval_lift_vs_dense_only")
        if lift is not None:
            lines += [
                "",
                f"Full pipeline improves recall@k by **{lift:+.1%}** over the dense-only baseline.",
            ]
        lines += ["", "```"]
        for variant in report.ablation:
            recall = variant.aggregate.get("recall@k", 0.0)
            lines.append(f"{variant.name:<18} {_bar(recall)} {recall:.3f}")
        lines += ["```"]

    lines += [
        "",
        "## Per-case results",
        "",
        "| Case | Category | Judge | Conf. | Recall@k | Citations | Valid | Latency | Cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for answer in report.answers:
        if answer.error:
            lines.append(
                f"| `{answer.case_id}` | {answer.category} | ERROR | - | - | - | - | - | - |"
            )
            continue
        judge = answer.judge or {}
        mark = "pass" if answer.passed else "fail"
        lines.append(
            f"| `{answer.case_id}` | {answer.category} | "
            f"{judge.get('overall', '-')} ({mark}) | {answer.confidence:.2f} | "
            f"{_pct(answer.retrieval.get('recall@k'))} | {answer.citations} | "
            f"{_pct(answer.citation_valid_rate)} | {answer.elapsed_seconds:.1f}s | "
            f"${answer.cost_usd:.5f} |"
        )

    problems = [a for a in report.answers if a.error or (a.judge and not a.judge.get("passed"))]
    if problems:
        lines += ["", "## Failures and weak answers", ""]
        for answer in problems:
            lines += [f"### `{answer.case_id}` - {answer.question}", ""]
            if answer.error:
                lines += [f"Run failed: `{answer.error}`", ""]
                continue
            judge = answer.judge
            lines += [
                f"Scores: correctness {judge['correctness']}/5, "
                f"groundedness {judge['groundedness']}/5, completeness {judge['completeness']}/5",
                "",
                f"> {judge.get('reasoning', '').strip()[:500]}",
                "",
            ]
            if judge.get("hallucinations"):
                lines += ["Flagged as unsupported:", ""]
                lines += [f"- {item}" for item in judge["hallucinations"]]
                lines += [""]
            missed = answer.retrieval.get("missed_paths") or []
            if missed:
                lines += ["Expected files that were never retrieved:", ""]
                lines += [f"- `{path}`" for path in missed]
                lines += [""]

    lines += [
        "",
        "## Configuration",
        "",
        "```json",
        _pretty(report.config),
        "```",
        "",
        f"Total wall time {report.total_seconds:.1f}s.",
        "",
        "<sub>Generated by `python -m evals.run_evals`.</sub>",
        "",
    ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "-"


def _num(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def _score(value: float | None) -> str:
    return f"{value:.2f} / 5" if isinstance(value, (int, float)) else "-"


def _pretty(payload: dict) -> str:
    import orjson

    return orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode()
