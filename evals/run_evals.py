from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import orjson

from evals.dataset import DEFAULT_DATASET, load_dataset
from evals.harness import EvalReport, EvalRunner
from evals.report import render_report
from reposage.config import get_settings
from reposage.index.store import RepoIndex
from reposage.llm.client import close_client, get_client
from reposage.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


async def run(
    repo: str,
    dataset_path: Path | None = None,
    limit: int | None = None,
    *,
    ablation: bool = True,
    judge: bool = True,
) -> EvalReport:
    settings = get_settings()
    if not settings.has_api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Evaluation needs model access; "
            "get a free key at https://aistudio.google.com/apikey"
        )
    cases = load_dataset(dataset_path, limit)
    index = RepoIndex.load_by_name(repo, settings)
    client = get_client(settings)
    try:
        log.info("eval.start", repo=repo, cases=len(cases), ablation=ablation, judge=judge)
        return await EvalRunner(index, client, settings).run(cases, ablation=ablation, judge=judge)
    finally:
        await close_client()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_evals",
        description="Evaluate RepoSage retrieval and answer quality against a golden dataset.",
    )
    parser.add_argument(
        "--repo", "-r", required=True, help="Index id to evaluate (see: reposage list)."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=None,
        help=f"JSONL dataset (default: {DEFAULT_DATASET.name}).",
    )
    parser.add_argument("--limit", "-n", type=int, default=None, help="Only run the first N cases.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("evals/REPORT.md"),
        help="Where to write the markdown report.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the raw results as JSON."
    )
    parser.add_argument("--no-ablation", action="store_true", help="Skip the retrieval ablation.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judging (free and fast).")
    parser.add_argument(
        "--min-recall", type=float, default=None, help="Fail if agent recall@k falls below this."
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="Fail if the judge pass rate falls below this.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO", False)

    report = asyncio.run(
        run(
            args.repo,
            args.dataset,
            args.limit,
            ablation=not args.no_ablation,
            judge=not args.no_judge,
        )
    )

    markdown = render_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"\nReport written to {args.output}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_bytes(orjson.dumps(report.to_dict(), option=orjson.OPT_INDENT_2))
        print(f"Raw results written to {args.json}")

    summary = report.summary
    print("\n" + "=" * 62)
    print(f"  cases            {report.dataset_size}")
    print(f"  pass rate        {summary.get('pass_rate', 0):.1%}")
    print(f"  recall@k         {summary.get('answer_recall@k', 0):.1%}")
    print(f"  citation valid   {summary.get('citation_valid_rate', 0):.1%}")
    print(f"  judge overall    {summary.get('judge_overall', 0)} / 5")
    print(f"  mean latency     {summary.get('mean_latency_s', 0):.1f}s")
    print(f"  total cost       ${report.total_cost_usd:.4f}")
    print("=" * 62)

    failures: list[str] = []
    if args.min_recall is not None and summary.get("answer_recall@k", 0.0) < args.min_recall:
        failures.append(f"recall@k {summary.get('answer_recall@k', 0):.3f} < {args.min_recall}")
    if args.min_pass_rate is not None and summary.get("pass_rate", 0.0) < args.min_pass_rate:
        failures.append(f"pass rate {summary.get('pass_rate', 0):.3f} < {args.min_pass_rate}")
    if failures:
        print("\nQuality gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


def run_cli(repo: str, dataset: Path | None, limit: int | None, output: Path | None) -> None:
    argv = ["--repo", repo]
    if dataset:
        argv += ["--dataset", str(dataset)]
    if limit:
        argv += ["--limit", str(limit)]
    if output:
        argv += ["--output", str(output)]
    raise SystemExit(main(argv))


if __name__ == "__main__":
    sys.exit(main())
