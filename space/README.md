---
title: RepoSage
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Ask an AI agent about a codebase, with citations
tags:
  - rag
  - agents
  - langgraph
  - code-search
  - retrieval
---

# RepoSage

**Agentic code intelligence.** Ask architectural questions about a codebase and
get answers with citations that are verified against the index before you see
them. Click any citation to read the exact lines the answer was built from.

This Space is running RepoSage against **its own source code**, pre-indexed so
it answers immediately.

## Try asking

- How does the critic decide whether to refine an answer?
- Why is reciprocal rank fusion used instead of normalising and adding scores?
- How are source files split into chunks, and why not fixed-size splitting?
- Where does configuration come from and how is it validated?
- How does the system avoid hitting free-tier API rate limits?

## What is actually happening

Each question runs a five-stage agent, and the right-hand panel shows it live:

1. **Plan** reads a symbol-annotated map of the repository and decomposes your
   question into search queries in both identifier and natural-language form.
2. **Retrieve** runs dense vector search and BM25 in parallel, fuses the two
   ranked lists with reciprocal rank fusion, and diversifies per file.
3. **Rerank** scores the shortlist with an LLM that reads query and candidate
   together, which is what promotes the chunk that *answers* the question over
   the one that merely mentions it.
4. **Analyse** drafts the answer from retrieved code only.
5. **Critique** audits it for grounding and completeness and can send it back
   for another retrieval round with targeted follow-up queries.

Finally every citation is resolved against the index. Markers that do not
resolve are dropped and lower the reported confidence, which is capped below
certainty because an answer built from a partial view is never certain.

Measured on this repository, reranking lifts retrieval recall from 80.6% to
88.9% and MRR from 0.611 to 0.833. The full evaluation report, including every
failure, is in the repository.

## Demo limits

Indexing new repositories is disabled here: it costs hundreds of embedding
requests and several minutes. The shared API key is metered per visitor and per
day, and when the daily budget runs out you can paste your own free
[Gemini key](https://aistudio.google.com/apikey) to keep exploring. It stays in
your browser tab.

To index your own repositories, run it locally:

```bash
git clone https://github.com/adwitiyashukla/reposage-ai
cd reposage-ai && pip install -e ".[treesitter]"
reposage index your-org/your-repo
reposage serve
```

---

Source, architecture notes and evaluation methodology:
**[github.com/adwitiyashukla/reposage-ai](https://github.com/adwitiyashukla/reposage-ai)**
