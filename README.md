# RepoSage

Ask an unfamiliar codebase architectural questions and get answers with line-level citations you can
click. Then point the same index at a pull request and let it review the diff.

[![Live demo](https://img.shields.io/badge/demo-live%20on%20Spaces-yellow)](https://huggingface.co/spaces/adwitiyashukla/reposage)
[![CI](https://github.com/adwitiyashukla/reposage-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/reposage-ai/actions/workflows/ci.yml)
[![Evaluation report](https://img.shields.io/badge/evaluation-report-blue)](evals/REPORT.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

There is a live demo at
[huggingface.co/spaces/adwitiyashukla/reposage](https://huggingface.co/spaces/adwitiyashukla/reposage).
It answers questions about RepoSage's own source code, shipped pre-indexed, so there is nothing to
sign up for and nothing to wait for. Two questions worth trying: "How are source files split into
chunks, and why not fixed-size splitting?" and "How does the critic decide whether to refine an
answer?" The panel on the right shows the agent working, and every citation opens the exact lines.

## Why I did not build plain RAG over the files

The obvious version of this project takes an afternoon. Split every file into fixed-size text
windows, embed them, take the top five by cosine similarity, hand them to a model. I built that
first. It demos beautifully and the answers cite functions that do not exist.

The model was not the problem. Retrieval was, in three specific ways, and each one turned into a
piece of the design.

A window that cuts a function in half retrieves badly. The window that matches your query is often
not the window holding the logic that answers it, and the model cannot tell it is reading the second
half of something.

Embeddings cannot do exact lookup. `ProcessPaymentIntent` and `handle_payment` sit close together in
vector space, which is useful when you search by concept and wrong when you typed a specific symbol
and want that symbol.

One retrieval pass is a guess. If the first search misses, nothing notices, and the model writes a
fluent paragraph over whatever it happened to get.

So the project is three fixes to those three problems, plus a way to check whether the fixes
actually did anything.

## What it looks like when it runs

```bash
reposage index .
reposage ask -r reposage-ai "What are the stages of the agent graph and when does it loop?"
```

```
The graph has five nodes wired in agents/graph.py. After the critic runs,
route_after_critique sends the state back to retrieval when the critique asks
for a refinement, and to finalise otherwise
[src/reposage/agents/graph.py:34-40].

The loop cannot run forever. The critic checks the refinement count against
max_refinements first, and once the budget is spent it rewrites its own verdict
to accept [src/reposage/agents/nodes/critic.py:43-46].

confidence 97%   19.9s | $0.00629 | 22 citations
```

Every bracketed reference is a real file and a real line range, checked against the index before it
reaches you. Anything that does not resolve is dropped and pulls the confidence score down. In the
web UI you click a citation and the exact lines open.

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/architecture-light.svg">
    <img alt="RepoSage architecture: ingestion, hybrid index, and the self-correcting agent loop" src="docs/assets/architecture-light.svg" width="100%">
  </picture>
</p>

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) walks the whole path from question to answer. The three
sections below are the parts I would defend in an interview.

### Chunking, which is where most of the quality comes from

Files are parsed with tree-sitter and one chunk is emitted per declaration, so a retrieved chunk is
a whole function, class or method with a name and an exact line range. Node types are declared per
language in `ingest/languages.py`, covering 48 file extensions with 12 tree-sitter grammars. The
grammars come from wheels rather than being downloaded and compiled, so container builds are
reproducible and CI works with no network access.

Two details took longer than the rest of the chunker put together.

A 900-line class is not a useful retrieval unit, but neither is a class chopped into 40 methods with
no outline. Large containers get split per member and also summarised as a skeleton chunk: the
declaration line plus every member signature. "What does this class do?" retrieves the skeleton.
"How does it validate the token?" retrieves the method. Splitting only one way loses one of those.

Every chunk is prefixed with a short deterministic header naming its file, language and imports.
The usual solution is to ask a model to write a summary for each chunk, which costs one call per
chunk and gets expensive fast. The header gets most of the same benefit for free.

When there is no grammar, or the parse fails, the file falls back to line chunking that prefers to
end windows on blank lines and dedents, and Markdown splits on headings, so the pipeline never
hard-fails on a file it does not understand.

### Two searches instead of one

Dense embedding search and BM25 both run, over every query variant the planner wrote. I wrote the
BM25 index in the repository instead of pulling in a library, mainly so the tokenizer could be
code-aware: `getUserByID` is indexed as the whole token and as `get`, `user`, `by` and `id`, so a
query in either style matches. That is the entire reason lexical search earns its place here.

The two ranked lists merge with reciprocal rank fusion:

```
score(d) = sum over retrievers of  weight / (k + rank(d))
```

Cosine similarity is bounded and clustered tight; BM25 is unbounded and moves with corpus
statistics. Normalising them into a weighted sum needs per-corpus calibration that quietly goes
stale. RRF throws the magnitudes away and uses only position in the list, so there is nothing to
calibrate and documents several retrievers agree on rise. `k = 60` stops the top of one list from
dominating.

After fusion the shortlist is diversified, at most four chunks per file at the head, so one big file
cannot eat the context window. Then an LLM reranks the survivors listwise, reading the query and the
candidates together, which is the thing a bi-encoder structurally cannot do.

### The loop that catches its own mistakes

The agent is a LangGraph state machine with five nodes.

| Node | Model | What it does |
| --- | --- | --- |
| `plan` | fast | Turns the question into search queries using a compressed map of the repository |
| `retrieve` | none | Runs the hybrid pipeline and merges results with context already gathered |
| `analyse` | deep | Drafts a cited answer from the retrieved code |
| `critique` | fast | Audits grounding and completeness, writes follow-up queries |
| `finalise` | none | Resolves every citation against the index and scores confidence |

The planner sees the repository map before writing anything, so it produces queries naming real
files and symbols instead of generic vocabulary. It mixes identifier-style queries
(`verify_jwt signature expiry`) with descriptive ones (`how session cookies are signed`), because
the lexical side rewards the first kind and the dense side rewards the second. If planning fails,
the raw question becomes a single query, which is what naive RAG would have done anyway.

The critic is the main defence against a confident wrong answer. It reads a compact index of what
was retrieved rather than the full context, which keeps it cheap and makes it harder for the same
misleading text to fool both the analyst and its reviewer. It only asks for another round when it
can name specific follow-up queries, and a refinement merges with what was already working rather
than replacing it. The budget is bounded, so a run always ends.

Confidence is not the model marking its own homework. The critic's number is the starting point,
then it is adjusted by things that can be checked: whether there are citations at all, what share of
cited paths were actually retrieved, and how many markers failed to resolve. It is capped at 0.97,
because nothing here should report certainty.

## Reviewing pull requests with the same index

The review path is a second consumer of the index. It reviews a diff against the repository rather
than in isolation, which is what stops it flagging a missing null check that a caller three files
away already guarantees.

```bash
reposage review --diff my-change.diff --repo reposage-ai
git diff main | reposage review --diff - --repo reposage-ai
reposage review --pr owner/repo#42 --repo reposage-ai --post --fail-on high
```

As a GitHub Action:

```yaml
name: AI review
on: pull_request
permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: adwitiyashukla/reposage-ai/.github/actions/reposage-review@main
        with:
          gemini-api-key: ${{ secrets.GEMINI_API_KEY }}
          github-token: ${{ secrets.GITHUB_TOKEN }}
          fail-on: high
```

I spent more time filtering the output than producing it. Findings below a confidence threshold are
dropped, anchors snap to lines that actually changed or get removed if the model was more than three
lines off, duplicates collapse, and the summary comment is edited in place on a re-run instead of
piling up. A review with three real findings gets read. One with thirty speculative ones gets muted,
and then the good three are lost too.

I also wrote the unified diff parser instead of using a library, because a review comment is only
useful if it lands on the right line. Each line records its old-file number, its new-file number and
its position inside the hunk, which is exactly what the GitHub review API wants. If GitHub rejects
an inline comment anyway, the review degrades to a summary rather than disappearing.

## Putting it online

The demo runs on my own API key, so a public URL is one crawler away from being permanently broken.
`REPOSAGE_DEMO_MODE` turns the same application into a metered demo rather than a separate build.
Indexing and deletion return 403, since indexing costs hundreds of embedding requests and several
minutes, so the Space ships a pre-built index. A global daily budget bounds the worst case and a
per-visitor hourly window stops one caller spending the whole day. Anyone can paste their own free
key to skip the budget entirely, because those requests cost me nothing. Read-only routes stay open,
because refusing to serve source would make the citations unopenable and defeat the point of citing
them.

One thing I did not expect: a budget refusal has to arrive as an event on a stream that opened
successfully, not as an HTTP error. `EventSource` in the browser cannot read the body of a failed
handshake, so a 429 reaches the visitor as a bare connection failure with nothing to explain it.
There is a test called `test_a_refused_stream_still_opens_and_explains_itself` purely because I got
this wrong first.

Hugging Face also pauses a free Space after 48 hours without traffic. It wakes on the next visit so
the link never breaks, but the person most likely to arrive cold is someone following a link from my
CV. A scheduled workflow pings `/api/health` every 12 hours, which costs no API quota and keeps the
Space resident.

## Things I got wrong

Six things broke badly enough to need their own fix. These three are the ones worth writing about.

### Confidence said 26 percent on an answer that was correct

The citation parser matched `[path:start-end]` and nothing else. Models group their references far
more than the prompt asks, so a real answer was full of `[src/a.py:16-18, 45-63]` and
`[src/a.py:3-6, docs/b.md:148-151]`, and the parser silently threw away roughly half of them. Fewer
resolved citations means a lower confidence score, so a well-cited, correct answer came back
reporting 26 percent.

What made me look was two numbers disagreeing. The judge scored the answer 5 out of 5 for
groundedness while the system reported low confidence about the same text. Both could not be right.

Brackets are now split on commas, a bare range inherits the path in front of it, and the penalty for
invalid citations became proportional to the share of bad references rather than the raw count. One
bad reference out of twenty should barely register; the old code treated it like one out of two.

### Loading an index meant running whatever built it

The row labels for the vector matrix were stored as an object-dtype `.npy` file. NumPy can only read
that back by unpickling, and unpickling runs arbitrary code. An index is exactly the thing people
copy between machines and bake into Docker images, so I had built an artefact that executes on load.
It also broke the deployed demo outright.

Ids are now newline-delimited text in `vector_ids.txt`. Old indexes still load, so nobody has to pay
for a re-index, and three tests pin the behaviour down: `test_ids_are_stored_as_plain_text`,
`test_no_pickled_object_arrays_are_written` and `test_legacy_pickled_ids_still_load`.

There was a sequel. Hugging Face stores `.npy` through Git LFS, so copying the index out of the
Space repository yielded a 132-byte pointer file, and NumPy read that text and reported it as
pickled data. The image now pulls the index from a pinned commit and checks the NumPy magic bytes at
build time, so a broken index fails the build rather than the first visitor.

### Eight parallel embedding batches were slower than one

Indexing embeds in batches, and I ran eight of them concurrently because that is what you do to make
things faster. CI then started failing on 429s while indexing, having burned its whole retry budget.

Throughput was never set by parallelism. It is set by the token bucket in front of the API, and all
the concurrency did was turn smooth pacing into bursts that landed together inside the provider's
rolling window. Serialising the embedding requests costs nothing in throughput and makes the pacing
exact. Generation kept its own guard and its looser quota.

The two settings that came out of this were measured against the live API rather than guessed. The
per-minute embedding ceiling is counted per item, not per request, so a batch of 32 spends 32 of it.
`REPOSAGE_EMBED_RPM` defaults to 75 against a measured ceiling of 100, which leaves headroom for
retries, and `REPOSAGE_EMBED_CONCURRENCY` defaults to 1.

## How I decided it was good enough

Most portfolio RAG projects assert that they work. I wanted a number I could defend, so three things
are measured separately, because they fail separately: retrieval quality (recall@k, precision@k,
MRR, nDCG, hit rate, all free and deterministic), answer quality (an LLM judge on correctness,
groundedness and completeness, plus citation validity and required-fact coverage), and operational
cost (latency, tokens and dollars per answer).

```bash
reposage index .
python -m evals.run_evals --repo reposage-ai --output evals/REPORT.md
```

### The ablation, and the result that surprised me

Every configuration runs the same questions through the production retrieval path, selected by the
`mode` parameter on `HybridRetriever.retrieve`. Nothing is reimplemented for the benchmark, so the
numbers describe the shipping system. Measured on this repository, 81 files and 494 chunks:

| Configuration | recall@k | precision@k | MRR | nDCG | time |
| --- | --- | --- | --- | --- | --- |
| dense only | 80.6% | 22.7% | 0.611 | 0.575 | 1.6s |
| lexical only | 69.4% | 13.7% | 0.557 | 0.509 | 0.0s |
| hybrid + RRF | 80.6% | 18.3% | 0.778 | 0.679 | 0.0s |
| hybrid + rerank | 88.9% | 20.5% | 0.833 | 0.797 | 9.0s |

I expected fusion to be the win. It is not. Fusion does not move recall at all, 80.6% before and
after, and lifts MRR from 0.611 to 0.778. Then it clicked that this is exactly what it should do:
both retrievers were already finding the right files, so RRF was never adding files, it was fixing
where in the list they landed. Reranking is what turns better ordering into better recall, because
it promotes the chunk that answers the question over the chunk that merely mentions it. The
bottleneck was ranking quality, not raw recall, which is not what I would have guessed.

### Answer quality

From the committed report in [evals/REPORT.md](evals/REPORT.md):

| Metric | Value |
| --- | --- |
| Cases in the run | 6 of the 16 in the dataset |
| Cases that produced an answer | 5 of 6 |
| Answers the judge passed | 5 of 5 |
| Judge correctness, groundedness, completeness | 5.00 / 5 on each axis |
| Citation validity | 100.0% |
| Required-fact coverage | 90.0% |
| Answers containing a hallucination | 0 of 5 |
| Mean citations per answer | 16.2 |
| Mean latency | 32.6s |
| Mean cost per answer | $0.00792 |
| Mean tokens per answer | 29,225 |
| Total cost of the run | $0.0475 |

Those scores look perfect, so it is worth being clear about what they do not show. Five scored
answers is a small sample, and the sixth case never ran because the run hit a free-tier rate limit
partway through, which is also why the report covers 6 cases rather than all 16. The judge comes
from the same model family as the system it grades, and same-family judges tend to agree with their
own outputs, which is why the objective metrics sit next to the judge scores and are never reported
without them. The ground truth is my own repository, which makes the run reproducible for anyone who
clones it and also means the questions are not adversarial. Read the table as evidence that nothing
is obviously broken, not as proof the system is right in general.

The dataset is `evals/datasets/golden.jsonl`, 16 cases across architecture, retrieval, agents,
observability, infrastructure, ingestion, review and API, tagged easy, medium or hard. A test called
`test_expected_paths_exist_in_this_repository` asserts every path in the dataset still exists, which
stops ground truth rotting silently as the code moves. In CI, `--min-recall` and `--min-pass-rate`
turn the suite into a gate, so a pull request that degrades retrieval fails before it merges.
[docs/EVALUATION.md](docs/EVALUATION.md) covers the judge rubric and how to add cases.

Embeddings, not generation, are the binding constraint on a free tier. Indexing this repository
costs roughly 500 embedding requests and the free tier caps them per day as well as per minute, so
two or three full builds exhaust a day. When it runs out, indexing fails with an explicit quota
error instead of writing a partial index, and the content-addressed embedding cache means a retry
after the reset re-embeds only what changed.

## Running it

You need Python 3.10 or newer, Git, and a free
[Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/adwitiyashukla/reposage-ai.git
cd reposage-ai

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,treesitter]"

cp .env.example .env                                  # then add your GEMINI_API_KEY
reposage doctor                                       # checks config, grammars and connectivity
```

Index something and ask it a question:

```bash
reposage index .                       # index RepoSage itself
reposage list                          # shows the index id that was created
reposage ask -r reposage-ai "How does the critic decide to refine an answer?"
```

The index id comes from the directory or repository name and is printed when indexing finishes, so
use whatever `reposage index` reported.

For the web UI, which streams the agent's reasoning live and makes citations clickable:

```bash
reposage serve                         # http://127.0.0.1:8000
```

Or with Docker:

```bash
echo "GEMINI_API_KEY=your-key" > .env
docker compose up --build              # http://localhost:8000
```

Common tasks have Make targets: `make install`, `make check` for lint, types and tests, `make run`
for the dev server, `make eval` for the evaluation suite.

### The HTTP API

`reposage serve` exposes an OpenAPI-documented API at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Version, configuration, index count |
| `GET` | `/api/ready` | Readiness including a live model probe |
| `POST` | `/api/indexes` | Build an index |
| `GET` | `/api/indexes/stream/build` | Build with SSE progress |
| `GET` | `/api/indexes` | List indexes |
| `POST` | `/api/ask` | Ask a question |
| `GET` | `/api/ask/stream` | Ask with a live SSE agent trace |
| `GET` | `/api/source/{repo}` | Read an indexed file for citation display |
| `POST` | `/api/review` | Review a unified diff |
| `GET` | `/api/graph` | The agent graph as Mermaid |

### Configuration

Everything comes from environment variables or `.env`. The annotated list is in
[.env.example](.env.example); these are the ones worth knowing about.

| Variable | Default | Effect |
| --- | --- | --- |
| `GEMINI_API_KEY` | required | Model access |
| `REPOSAGE_FAST_MODEL` | `gemini-flash-lite-latest` | Planning, reranking, critique |
| `REPOSAGE_DEEP_MODEL` | `gemini-flash-latest` | Analysis and review |
| `REPOSAGE_TOP_K` | `12` | Chunks passed to the analyst |
| `REPOSAGE_CANDIDATE_K` | `40` | Candidates from each retriever |
| `REPOSAGE_ENABLE_RERANK` | `true` | LLM listwise reranking |
| `REPOSAGE_ENABLE_CRITIC` | `true` | Self-critique loop |
| `REPOSAGE_MAX_REFINEMENTS` | `2` | Retrieval retry budget |
| `REPOSAGE_CHUNK_MAX_LINES` | `120` | Chunk line budget |
| `REPOSAGE_EMBED_RPM` | `75` | Embedding rate cap, counted per item |
| `REPOSAGE_EMBED_CONCURRENCY` | `1` | Parallel embedding batches |
| `REPOSAGE_ENABLE_CACHE` | `true` | Disk cache for generations and embeddings |

## What is in the repo

```
src/reposage/
  config.py            Settings from the environment, validated once
  models.py            Domain types shared by every layer
  cli.py               Typer CLI mirroring the HTTP API
  ingest/              Traversal, tree-sitter chunking, cloning, the repository map
  index/               Vector store, BM25, fusion, reranker, hybrid retriever, persistence
  agents/              Prompts, the LangGraph state machine, and the five nodes
  review/              Diff parser, context-grounded reviewer, idempotent GitHub posting
  llm/                 Provider, disk cache, pricing table, rate limiting
  observability/       Spans, token accounting, live event streaming
  api/                 FastAPI app, routes, schemas, demo metering
  web/                 Single-file UI, no build step

evals/                 Dataset, metrics, judges, ablation harness, report generator
tests/                 194 tests, fully offline
docs/                  Architecture and evaluation write-ups
space/                 Hugging Face Space packaging
```

An index is a plain directory, no database and no server: a manifest with stats and a schema
version, the chunks as JSONL, the float32 vector matrix and its id list, the BM25 postings and
vocabulary, and the repository map. Loading an index built by an incompatible version raises with
instructions to re-index instead of failing obscurely later.

## Tests

```bash
pytest -q
```

194 tests, all offline. A deterministic fake provider stands in for the model, so the suite is fast,
free and reproducible, and CI needs no secrets. CI runs it on Python 3.10 through 3.13 on Linux,
plus 3.12 on Windows and macOS, and separately builds the Docker image and waits for the container
to report healthy.

A few that show what I thought was worth pinning down:

| Test | What it protects |
| --- | --- |
| `test_large_class_splits_into_members_with_a_skeleton` | The chunking decision retrieval quality rests on |
| `test_unknown_language_falls_back_without_raising` | A file with no grammar never kills an index build |
| `test_documents_found_by_both_retrievers_rank_highest` | The property RRF exists to give |
| `test_hallucinated_citations_are_discarded` | Citations that do not resolve never reach the user |
| `test_invalid_citations_are_penalised_proportionally` | The confidence bug above, pinned |
| `test_no_pickled_object_arrays_are_written` | The index format bug above, pinned |
| `test_embedding_is_serialised` | The rate limit bug above, pinned |
| `test_anchor_snaps_to_a_nearby_changed_line` | Review comments land on lines that actually changed |
| `test_expected_paths_exist_in_this_repository` | Evaluation ground truth cannot rot silently |

## Choices I would defend

REST instead of the vendor SDK. The Gemini provider talks to the HTTP API through `httpx` rather
than `google-genai`, which is one dependency instead of a transitive tree, full control over
timeouts, retry classification and SSE parsing, and nothing to fix when the SDK cuts a major
version. It cost about 150 lines of request shaping, tested with a mocked transport, and it sits
behind an `LLMProvider` protocol, so adding Groq, OpenRouter or a local Ollama server is a new
class rather than a rewrite.

Exact vector search instead of an ANN index. Dense retrieval is a brute-force cosine matmul over one
contiguous float32 matrix. At repository scale that is a few milliseconds, recall is perfect so a
retrieval regression can never be blamed on an approximate index, and it adds no dependency and no
service to run. Past a few hundred thousand chunks an ANN index wins, which is why the code is
written against a `VectorStore` protocol.

An LLM reranker instead of a cross-encoder. A hosted cross-encoder wants a GPU and a model download.
Listwise LLM scoring runs on a free tier, keeps the install light, and falls back to the fused
ordering if a window fails.

Tracing without OpenTelemetry. What I needed was in-process, single-run visibility, not a collector
to deploy. The event schema is deliberately OTel-shaped, so exporting later is a small change.

Indexing is a full rebuild. Incremental re-indexing over a commit range is the obvious improvement
and the embedding cache already makes rebuilds cheap, but a full rebuild is one code path instead of
two and I would rather have one that is correct.

There are things this shape cannot do well, and they follow from the same choices. Answer quality
tracks retrieval quality, so a question whose answer is spread thinly across many files ("where is
every place we retry?") is harder than one with a clear locus. The critic is a language model
judging a language model: it catches obvious ungrounded claims reliably and subtle ones
inconsistently, which is why confidence leans on verifiable citation signals rather than trusting
it. And repository history is not indexed, so "why was this changed?" is out of scope.

## License and stack

MIT. See [LICENSE](LICENSE).

Python 3.10+, LangGraph, FastAPI, Typer, tree-sitter, NumPy, httpx, Pydantic, structlog, pytest,
ruff, mypy, Docker, GitHub Actions.

Built by [Adwitiya Shukla](https://github.com/adwitiyashukla).
