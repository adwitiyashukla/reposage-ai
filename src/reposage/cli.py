from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from reposage import __version__
from reposage.config import get_settings
from reposage.logging_setup import configure_console_encoding, configure_logging, get_logger

configure_console_encoding()

app = typer.Typer(
    name="reposage",
    help="Agentic code intelligence: index a repository, ask it questions, review its pull requests.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
log = get_logger(__name__)


def _bootstrap(verbose: bool = False) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level, settings.log_json)
    settings.ensure_dirs()


def _require_key() -> None:
    if not get_settings().has_api_key:
        console.print(
            "[bold red]GEMINI_API_KEY is not set.[/bold red]\n"
            "Get a free key at [link]https://aistudio.google.com/apikey[/link], then either\n"
            "  copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and fill it in, or\n"
            "  export it in your shell."
        )
        raise typer.Exit(code=2)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"reposage {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command("index", help="Clone, chunk, embed and persist a repository.")
def index_command(
    source: str = typer.Argument(..., help="GitHub URL, owner/repo, or a local path."),
    branch: str | None = typer.Option(None, "--branch", "-b", help="Branch to clone."),
    refresh: bool = typer.Option(False, "--refresh", help="Discard any cached clone."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _bootstrap(verbose)
    _require_key()

    async def run() -> None:
        from reposage.index.store import RepoIndex
        from reposage.ingest.pipeline import IngestionPipeline
        from reposage.llm.client import close_client, get_client
        from reposage.observability import Tracer, use_tracer

        settings = get_settings()
        client = get_client(settings)
        tracer = Tracer()
        try:
            with (
                console.status(f"[cyan]Ingesting {source}[/cyan]", spinner="dots"),
                use_tracer(tracer),
            ):
                ingestion = await IngestionPipeline(settings).run(
                    source, branch=branch, refresh=refresh
                )
            console.print(
                f"  walked [bold]{ingestion.metadata.num_files}[/bold] files "
                f"-> [bold]{ingestion.num_chunks}[/bold] chunks "
                f"(AST coverage {ingestion.stats.get('ast_coverage', 0):.0%})"
            )
            with (
                console.status("[cyan]Embedding and indexing[/cyan]", spinner="dots"),
                use_tracer(tracer),
            ):
                repo_index = await RepoIndex.build(ingestion, client, settings=settings)
                path = repo_index.save(settings.index_dir)

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_row("index id", f"[bold cyan]{repo_index.index_id}[/bold cyan]")
            table.add_row("commit", repo_index.metadata.commit[:8] or "working tree")
            table.add_row("files", f"{repo_index.metadata.num_files:,}")
            table.add_row("chunks", f"{repo_index.metadata.num_chunks:,}")
            table.add_row("languages", ", ".join(list(repo_index.metadata.languages)[:6]))
            table.add_row("vectors", f"{len(repo_index.vectors):,} x {repo_index.vectors.dim}")
            table.add_row("vocabulary", f"{len(repo_index.lexical.vocabulary):,} terms")
            table.add_row("tokens", f"{tracer.usage.total_tokens:,}")
            table.add_row("cost", f"${tracer.usage.cost_usd:.4f}")
            table.add_row("location", str(path))
            console.print(Panel(table, title="[green]Index ready[/green]", border_style="green"))
            console.print(
                f'\nAsk it something:\n  [cyan]reposage ask -r {repo_index.index_id} "how does X work?"[/cyan]'
            )
        finally:
            await close_client()

    asyncio.run(run())


@app.command("ask", help="Ask a question about an indexed repository.")
def ask_command(
    question: str = typer.Argument(..., help="What you want to know."),
    repo: str = typer.Option(..., "--repo", "-r", help="Index id (see: reposage list)."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    show_trace: bool = typer.Option(False, "--trace", help="Print the agent's span timeline."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _bootstrap(verbose)
    _require_key()

    async def run() -> None:
        from reposage.agents.engine import CodebaseAgent
        from reposage.index.store import RepoIndex
        from reposage.llm.client import close_client, get_client
        from reposage.observability import Tracer, use_tracer

        settings = get_settings()
        client = get_client(settings)
        tracer = Tracer()
        try:
            try:
                repo_index = RepoIndex.load_by_name(repo, settings)
            except FileNotFoundError as exc:
                console.print(f"[bold red]{exc}[/bold red]")
                raise typer.Exit(code=1) from exc

            agent = CodebaseAgent(repo_index, client, settings)
            with console.status("[cyan]Thinking[/cyan]", spinner="dots"), use_tracer(tracer):
                answer = await agent.ask(question, tracer=tracer)

            if json_output:
                console.print_json(json.dumps(answer.model_dump(mode="json")))
                return

            console.print()
            console.print(Markdown(answer.answer))
            console.print()

            if answer.citations:
                table = Table(title="Citations", show_header=True, header_style="bold cyan")
                table.add_column("Location")
                table.add_column("Symbol")
                for citation in answer.citations:
                    table.add_row(citation.label, citation.symbol or "-")
                console.print(table)

            colour = (
                "green"
                if answer.confidence >= 0.7
                else "yellow"
                if answer.confidence >= 0.45
                else "red"
            )
            console.print(
                f"\n[{colour}]confidence {answer.confidence:.0%}[/{colour}]  "
                f"[dim]{answer.elapsed_seconds:.1f}s | "
                f"{answer.usage.total_tokens:,} tokens | ${answer.usage.cost_usd:.4f} | "
                f"{answer.refinement_rounds} refinement(s)[/dim]"
            )
            if show_trace:
                console.print()
                trace_table = Table(title="Trace", show_header=True, header_style="bold cyan")
                trace_table.add_column("Span")
                trace_table.add_column("ms", justify="right")
                for span in tracer.waterfall():
                    trace_table.add_row(
                        "  " * span["depth"] + span["name"], f"{span['duration_ms']:.0f}"
                    )
                console.print(trace_table)
        finally:
            await close_client()

    asyncio.run(run())


@app.command("list", help="List every index available on disk.")
def list_command() -> None:
    _bootstrap()
    from reposage.index.store import list_indexes

    entries = list_indexes(get_settings())
    if not entries:
        console.print(
            "[yellow]No indexes yet.[/yellow] Build one with: [cyan]reposage index owner/repo[/cyan]"
        )
        return
    table = Table(title="Indexed repositories", header_style="bold cyan")
    for column in ("id", "name", "commit", "files", "chunks", "languages", "indexed at"):
        table.add_column(column)
    for entry in entries:
        table.add_row(
            entry["id"],
            entry["name"],
            entry["commit"],
            f"{entry['files']:,}",
            f"{entry['chunks']:,}",
            ", ".join(entry["languages"][:3]),
            entry["indexed_at"],
        )
    console.print(table)


@app.command("delete", help="Delete an index from disk.")
def delete_command(name: str = typer.Argument(..., help="Index id to delete.")) -> None:
    _bootstrap()
    from reposage.index.store import RepoIndex

    if RepoIndex.delete(name, get_settings()):
        console.print(f"[green]Deleted index '{name}'.[/green]")
    else:
        console.print(f"[yellow]No index named '{name}'.[/yellow]")
        raise typer.Exit(code=1)


@app.command("review", help="Review a diff or a GitHub pull request.")
def review_command(
    diff_file: Path | None = typer.Option(
        None, "--diff", "-d", help="Path to a unified diff, or - for stdin."
    ),
    pr: str | None = typer.Option(None, "--pr", help="Review a GitHub PR: owner/repo#123."),
    repo: str | None = typer.Option(
        None, "--repo", "-r", help="Index id used to ground the review."
    ),
    title: str = typer.Option("Local change", "--title"),
    post: bool = typer.Option(False, "--post", help="Post the review back to GitHub."),
    fail_on: str = typer.Option(
        "none",
        "--fail-on",
        help="Exit non-zero at this severity or above: critical, high, medium, none.",
    ),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _bootstrap(verbose)
    if not pr and not diff_file:
        console.print("[red]Provide either --diff or --pr.[/red]")
        raise typer.Exit(code=2)
    if pr and ("#" not in pr or "/" not in pr):
        console.print("[red]--pr must look like owner/repo#123[/red]")
        raise typer.Exit(code=2)
    _require_key()

    async def run() -> None:
        from reposage.index.retriever import HybridRetriever
        from reposage.index.store import RepoIndex
        from reposage.llm.client import close_client, get_client
        from reposage.models import Severity
        from reposage.observability import Tracer, use_tracer
        from reposage.review.github import GitHubClient, PullRequestRef
        from reposage.review.reviewer import PullRequestReviewer

        settings = get_settings()
        client = get_client(settings)
        github: GitHubClient | None = None
        ref: PullRequestRef | None = None
        try:
            if pr:
                slug, number = pr.rsplit("#", 1)
                owner, name = slug.split("/", 1)
                ref = PullRequestRef(owner=owner, repo=name, number=int(number))
                github = GitHubClient(settings.github_token)
                with console.status(f"[cyan]Fetching {pr}[/cyan]", spinner="dots"):
                    diff_text = await github.get_diff(ref)
                    metadata = await github.get_pull_request(ref)
                pr_title = metadata.get("title", title)
                description = metadata.get("body") or ""
            else:
                diff_text = (
                    sys.stdin.read()
                    if str(diff_file) == "-"
                    else diff_file.read_text(encoding="utf-8")
                )
                pr_title, description = title, ""

            retriever = None
            if repo:
                repo_index = RepoIndex.load_by_name(repo, settings)
                retriever = HybridRetriever(repo_index, client, settings)

            tracer = Tracer()
            reviewer = PullRequestReviewer(client, retriever, settings)
            with console.status("[cyan]Reviewing[/cyan]", spinner="dots"), use_tracer(tracer):
                report = await reviewer.review(diff_text, title=pr_title, description=description)

            if json_output:
                console.print_json(json.dumps(report.model_dump(mode="json")))
            else:
                console.print(Panel(report.summary or "No summary.", title="[cyan]Summary[/cyan]"))
                if report.findings:
                    for finding in report.sorted_findings():
                        colour = {
                            "critical": "red",
                            "high": "red",
                            "medium": "yellow",
                            "low": "blue",
                            "nit": "dim",
                        }[finding.severity.value]
                        location = f"{finding.path}" + (f":{finding.line}" if finding.line else "")
                        console.print(
                            f"\n[{colour}][bold]{finding.severity.value.upper()}[/bold][/{colour}] "
                            f"[dim]{finding.category}[/dim]  {location}\n"
                            f"  [bold]{finding.title}[/bold]\n  {finding.body}"
                        )
                        if finding.suggestion:
                            console.print(f"  [green]suggestion:[/green] {finding.suggestion}")
                else:
                    console.print("[green]No issues found.[/green]")
                console.print(
                    f"\n[dim]{report.files_reviewed} file(s) | {report.elapsed_seconds:.1f}s | "
                    f"{tracer.usage.total_tokens:,} tokens | ${tracer.usage.cost_usd:.4f}[/dim]"
                )

            if post and github and ref:
                await github.post_review(ref, report)
                console.print(f"[green]Posted review to {ref.slug}.[/green]")
            elif post:
                console.print("[yellow]--post requires --pr.[/yellow]")

            threshold = fail_on.strip().lower()
            if threshold != "none":
                try:
                    limit = Severity(threshold).rank
                except ValueError:
                    console.print(f"[red]Unknown severity '{fail_on}'.[/red]")
                    raise typer.Exit(code=2) from None
                if any(f.severity.rank <= limit for f in report.findings):
                    console.print(f"[red]Failing: findings at or above '{threshold}'.[/red]")
                    raise typer.Exit(code=1)
        finally:
            if github:
                await github.aclose()
            await close_client()

    asyncio.run(run())


@app.command("serve", help="Run the web UI and HTTP API.")
def serve_command(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    _bootstrap()
    import uvicorn

    console.print(
        Panel(
            f"UI       [link]http://{host}:{port}/[/link]\n"
            f"API docs [link]http://{host}:{port}/docs[/link]",
            title="[green]RepoSage[/green]",
            border_style="green",
        )
    )
    uvicorn.run(
        "reposage.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=get_settings().log_level.lower(),
    )


@app.command("doctor", help="Check configuration, optional grammars and model connectivity.")
def doctor_command() -> None:
    _bootstrap()
    from reposage.index.store import list_indexes
    from reposage.ingest.chunker import available_grammars, treesitter_available

    settings = get_settings()
    table = Table(title="RepoSage diagnostics", header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    ok = "[green]ok[/green]"
    warn = "[yellow]warn[/yellow]"
    bad = "[red]fail[/red]"

    table.add_row("version", ok, __version__)
    table.add_row("python", ok, sys.version.split()[0])
    table.add_row(
        "GEMINI_API_KEY",
        ok if settings.has_api_key else bad,
        "configured" if settings.has_api_key else "missing: https://aistudio.google.com/apikey",
    )
    grammars = available_grammars()
    table.add_row(
        "tree-sitter",
        ok if treesitter_available() else warn,
        f"{len(grammars)} grammars: {', '.join(grammars[:6])}"
        if grammars
        else "not installed; chunking falls back to line splitting (pip install -e '.[treesitter]')",
    )
    table.add_row("data dir", ok, str(settings.data_dir.resolve()))
    table.add_row("indexes", ok, f"{len(list_indexes(settings))} on disk")
    table.add_row(
        "GITHUB_TOKEN",
        ok if settings.github_token else warn,
        "configured" if settings.github_token else "not set (only needed for --pr review)",
    )
    console.print(table)

    if settings.has_api_key:

        async def probe() -> None:
            from reposage.llm.client import close_client, get_client

            client = get_client(settings)
            try:
                with console.status("[cyan]Contacting the model[/cyan]", spinner="dots"):
                    result = await client.healthcheck()
                if result.get("ok"):
                    console.print(
                        f"[green]Model reachable[/green] ({result['model']}, {result['latency_ms']:.0f} ms)"
                    )
                else:
                    console.print(f"[red]Model unreachable:[/red] {result.get('error')}")
            finally:
                await close_client()

        asyncio.run(probe())


@app.command("eval", help="Run the evaluation suite against an index.")
def eval_command(
    repo: str = typer.Option(..., "--repo", "-r", help="Index id to evaluate against."),
    dataset: Path | None = typer.Option(
        None, "--dataset", "-d", help="Path to a golden JSONL dataset."
    ),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Only run the first N cases."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write a markdown report here."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _bootstrap(verbose)
    _require_key()
    try:
        from evals.run_evals import run_cli
    except ImportError:
        console.print(
            "[yellow]The evaluation harness lives in the repository, not the installed package.[/yellow]\n"
            "Run it from a clone:\n"
            "  [cyan]git clone https://github.com/adwitiyashukla/reposage-ai && cd reposage-ai[/cyan]\n"
            "  [cyan]python -m evals.run_evals --repo " + repo + "[/cyan]"
        )
        raise typer.Exit(code=1) from None

    run_cli(repo=repo, dataset=dataset, limit=limit, output=output)


if __name__ == "__main__":
    app()
