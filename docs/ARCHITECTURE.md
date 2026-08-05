# Architecture

This document explains how a question becomes an answer, and why each stage
exists. It assumes you have read the README.

---

## 1. Ingestion

**Goal:** turn a repository into retrievable units that are worth retrieving.

### Acquisition

`ingest/repository.py` accepts three forms of input and normalises all of them
into a working tree plus a commit SHA:

| Input | Handling |
| --- | --- |
| `owner/repo` | Expanded to a GitHub HTTPS URL |
| Any Git URL | Used directly |
| A local path | Used in place, no clone |

Clones are shallow (`--depth 1`), single-branch and cached between runs, because
only the current tree is ever read. A second index of the same repository is a
fetch, not a download.

### Traversal

`ingest/walker.py` decides what is worth indexing. Real repositories are mostly
noise, and indexing noise costs embedding budget *and* dilutes retrieval quality,
which is the more expensive of the two.

Layered filtering:

1. Excluded directories are pruned during the walk, not filtered afterwards
   (`node_modules`, `.venv`, `dist`, `target`, `vendor`, `__pycache__`, ...).
2. Lockfiles, minified bundles, binaries and media are dropped by name and
   extension.
3. `.gitignore` is honoured via `pathspec`.
4. Files whose average line length exceeds 400 characters are dropped, which
   catches generated and minified content that slipped through.
5. Null-byte sampling rejects anything binary.

When a repository still exceeds `REPOSAGE_MAX_FILES`, files are ranked by an
importance heuristic rather than truncated arbitrarily. The heuristic favours
real source over data, shallow paths over deep ones, and recognised entry points
(`main.py`, `index.ts`, `Dockerfile`, `README.md`) over everything else, and
penalises tests, fixtures and anything that looks generated.

### Chunking

`ingest/chunker.py` is where retrieval quality is won or lost.

Each file is parsed with tree-sitter and one chunk is emitted per declaration,
using the node types declared per language in `ingest/languages.py`. Three
refinements matter:

**Container splitting.** A 900-line class is not a useful retrieval unit. Large
containers are split per member, and additionally summarised as a *skeleton*
chunk: the declaration line plus every member signature. Questions of the form
"what does this class do?" retrieve the skeleton; questions about specific
behaviour retrieve the method.

**Context enrichment.** Every chunk is prefixed with a deterministic header
naming its file, language and import block:

```
// file: src/auth/jwt.py (python)
// imports: import os; from typing import Optional
class TokenValidator:
    ...
```

This recovers most of the benefit of LLM-generated contextual retrieval without
spending a model call per chunk.

**Graceful degradation.** A missing grammar, a parse failure or an unfamiliar
language falls back to structure-aware line chunking that prefers to end windows
on blank lines and dedents. Markdown is split on heading boundaries. Scattered
module-level code (constants, loggers, guard clauses) is merged into a single
module-scope chunk rather than emitted as a dozen two-line fragments.

### The repository map

`ingest/pipeline.py` builds a compressed, symbol-annotated tree of the
repository:

```
REPOSITORY MAP  |  312 files, 48,201 lines, 4,118 indexed chunks

src/reposage/index/
  retriever.py (318 lines)  -> HybridRetriever, RetrievalDebug
  store.py (301 lines)      -> RepoIndex, IndexStats, list_indexes
  lexical.py (243 lines)    -> BM25Index, tokenize_code
```

The planner sees this before writing any query. It is the difference between a
plan built from generic vocabulary and one that names real files and symbols.

---

## 2. Indexing

`index/store.py` builds two indexes over the same chunks and persists both as
plain files. No database, no server: an index is a directory you can copy,
inspect and delete.

```
.reposage/indexes/<repo>/
├── manifest.json      metadata, stats, schema version
├── chunks.jsonl       one chunk per line
├── vectors.npy        float32 matrix, L2-normalised
├── vector_ids.npy     row -> chunk id
├── bm25.npz           compressed postings, idf, doc lengths
├── bm25_meta.json     vocabulary, doc ids, parameters
└── repo_map.txt
```

The manifest carries a schema version. Loading an index built by an incompatible
version raises with instructions to re-index, rather than failing obscurely
later.

**Dense side.** Chunks are embedded with the symbol path and kind prepended,
which measurably helps questions phrased in terms of names rather than
behaviour. Only cache misses reach the API, so re-indexing after a small change
costs close to nothing.

**Lexical side.** `index/lexical.py` implements Okapi BM25 over an inverted
index with NumPy postings. The tokenizer splits compound identifiers, so
`getUserByID` is indexed as the full token *and* as `get`, `user`, `by`, `id`.

---

## 3. Retrieval

`index/retriever.py` runs three stages.

### Recall

Dense and lexical search run over every query variant the planner produced. All
query embeddings are computed in one batched call, so N sub-queries cost one
round trip rather than N.

### Fusion

Ranked lists are merged with reciprocal rank fusion:

```
score(d) = sum over retrievers of  weight / (k + rank(d))
```

Cosine similarity is bounded and tightly clustered; BM25 is unbounded and varies
with corpus statistics. Normalising them into a weighted sum requires per-corpus
calibration that silently rots. RRF discards magnitudes entirely and uses only
ordinal position, so it has no corpus-specific parameters and rewards documents
several independent retrievers agree on. `k = 60` damps the influence of the very
top ranks so one retriever cannot dominate.

Paths the planner flagged as likely receive a modest score boost.

### Precision

**Diversification.** At most four chunks per file reach the head of the
shortlist, so one large file cannot monopolise the context window. Overflow is
appended rather than discarded, so a genuinely single-file answer still gets
depth.

**Reranking.** `index/reranker.py` scores candidates listwise against a rubric.
A bi-encoder scores query and chunk independently and therefore cannot model the
interaction between them: it finds text *about* the topic, not text that
*answers* the question. Windows are scored in parallel and any failed window
falls back to its fused ordering.

**Neighbour expansion.** Immediately adjacent chunks are pulled in at a
discounted score. A retrieved function often depends on a constant or helper
defined directly above it, and this is far cheaper than a second retrieval round.

---

## 4. The agent

`agents/graph.py` compiles a LangGraph state machine.

```
        ┌──────────────────────────────┐
        v                              │
 plan ──> retrieve ──> analyse ──> critique ──> finalise
   │                      ^            │
   └── map is enough ─────┘        refine
```

| Node | Model tier | Responsibility |
| --- | --- | --- |
| `plan` | fast | Decompose the question into search queries using the repository map |
| `retrieve` | none | Run the hybrid pipeline, merge with prior context |
| `analyse` | deep | Draft an answer from retrieved code, with inline citations |
| `critique` | fast | Audit grounding and completeness, emit follow-up queries |
| `finalise` | none | Resolve citations against the index, score confidence |

**Why the planner matters.** One fast-model call turns "how does auth work?" into
several well-formed queries that deliberately mix identifier-style
(`verify_jwt signature expiry`) and descriptive (`how session cookies are
signed`) vocabulary, because the lexical retriever rewards the former and the
dense retriever the latter. Planning failures are never fatal: the raw question
becomes a single query, which is exactly what naive RAG would have done.

**Why the critic matters.** It is the main defence against confident wrong
answers. It sees a compact index of what was retrieved rather than the full
context, which keeps the check cheap and makes it harder for the critic to be
seduced by the same text that misled the analyst. It only triggers a refinement
when it can name concrete follow-up queries; refining on a hunch wastes a round
trip.

**Why refinement merges rather than replaces.** A refinement is meant to fill a
gap, not discard context that was already working. New chunks are merged with
existing ones, keeping the better score for anything both rounds found, and the
combined window is capped so repeated refinement cannot grow the prompt without
bound.

**Confidence is not self-reported.** The critic's confidence is the starting
point, then tempered by things that can be verified: whether citations exist,
what fraction of cited paths were actually retrieved, and how many citation
markers failed to resolve against the index.

---

## 5. Observability

`observability/tracing.py` gives every run a span tree, token accounting and a
subscriber queue.

Spans nest, carry arbitrary attributes and record duration. Subscribers attach to
a tracer to receive events *as they happen*, which is what powers the live UI:
the agent run executes in a background task while the API drains the event queue
and republishes it as server-sent events.

This is deliberately not OpenTelemetry. OTel would add a heavy dependency and a
collector to run, and the requirement is in-process, single-run visibility. The
event schema is OTel-shaped so exporting later is a small change.

Token usage is recorded per call and converted to dollars by a per-model pricing
table matched by longest prefix, so `gemini-2.0-flash-001` resolves without an
exhaustive table. Cache hits are counted separately and cost nothing.

---

## 6. Review

`review/` is a separate consumer of the same index.

`diff.py` parses unified diffs in-tree rather than through a library, because a
review comment is only useful if it lands on the right line. Each line records
its old-file number, its new-file number and its position within the hunk, which
is exactly what the GitHub review API needs.

`reviewer.py` retrieves repository context around the identifiers each diff
touches before reviewing it, then filters the model's output: findings below a
confidence threshold are dropped, anchors are snapped to lines that actually
changed (or removed entirely if the model was more than three lines off), and
duplicates are collapsed. A review with three real findings gets read; one with
thirty speculative ones gets muted.

`github.py` posts idempotently. Every summary carries a hidden marker comment, so
a re-run edits the existing comment instead of appending a new one. If GitHub
rejects any inline comment with a 422, the review degrades to a summary rather
than being lost.

---

## Extension points

| To do this | Change this |
| --- | --- |
| Add a model provider | Implement `LLMProvider` in `llm/`, point `LLMClient` at it |
| Add an ANN backend | Implement `VectorStore` in `index/vector_store.py` |
| Support a new language | Add a `LanguageSpec` and grammar entry in `ingest/` |
| Change agent behaviour | Add a node in `agents/nodes/`, wire it in `graph.py` |
| Tune prompts | `agents/prompts.py`, then bump `PROMPT_VERSION` |
