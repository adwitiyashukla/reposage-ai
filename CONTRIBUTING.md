# Contributing

Thanks for taking a look.

## Setup

```bash
git clone https://github.com/adwitiyashukla/reposage-ai.git
cd reposage
make install          # venv + all extras
cp .env.example .env  # add your GEMINI_API_KEY
make check            # lint, type check, tests
```

## Before opening a pull request

```bash
make fmt      # ruff fix + format
make check    # must pass
```

The test suite is fully offline. A deterministic fake provider stands in for the
model, so tests need no API key and no network. If you add a feature that calls
a model, extend `FakeProvider` in `tests/conftest.py` rather than reaching for
the real API.

## What good changes look like

- **A behaviour change comes with a test.** The suite is the contract.
- **A retrieval change comes with numbers.** Run
  `python -m evals.run_evals --repo reposage` before and after, and put the
  ablation table in the pull request description. "This felt better" is not
  reviewable; "recall@k moved 0.75 to 0.81" is.
- **A prompt change bumps `PROMPT_VERSION`** in `agents/prompts.py`, so
  evaluation results stay attributable to an exact prompt revision.
- **A new dependency is justified in the description.** The dependency list is
  deliberately short and several things are implemented in-tree for that reason.

## Adding a language

1. Add a `LanguageSpec` to `LANGUAGES` in `src/reposage/ingest/languages.py`,
   naming the tree-sitter node types that represent declarations, imports and
   containers.
2. Add the grammar to `_GRAMMARS` in `src/reposage/ingest/chunker.py` and to the
   `treesitter` extra in `pyproject.toml`.
3. Add a chunking test in `tests/test_chunker.py`.

## Reporting bugs

Include the output of `reposage doctor`, the command you ran, and what you
expected. If it involves a specific repository, name it: reproducibility is most
of the fix.
