# Hugging Face Space image.
#
# RepoSage itself is installed from the GitHub repository rather than vendored,
# so this Space stays three small files plus a pre-built index and can never
# drift from the source it demonstrates.
#
# The index is baked into the image on purpose. Building it live would cost
# roughly 500 embedding requests and seven minutes, which is both quota-hostile
# and far longer than any visitor will wait.
FROM python:3.12-slim

# git is needed to install from the repository, and at runtime by the ingestion
# layer even though the demo never clones anything.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as UID 1000 and only $HOME is reliably writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Pinned to an exact commit rather than a branch. A moving ref lets Docker
# reuse its cached install layer, so a redeploy would silently ship whatever
# code the layer was built from. Bump this to redeploy.
ARG REPOSAGE_REF=d8785482d79fcdf8500fb30440d228d31ce05deb
RUN pip install --no-cache-dir --user \
    "reposage[treesitter] @ git+https://github.com/adwitiyashukla/reposage-ai.git@${REPOSAGE_REF}"

# The pre-built index is pulled from GitHub rather than committed to this
# Space. Hugging Face stores *.npy through Git LFS, so a COPY from the Space
# repository yields a 132-byte pointer file instead of the array, and NumPy
# reports that text as "pickled data" at runtime. GitHub serves the real bytes.
# Fetching at the pinned commit also keeps the index and the code in lockstep.
ARG INDEX_BASE=https://raw.githubusercontent.com/adwitiyashukla/reposage-ai
# Downloaded, then verified: a build that quietly produces a broken index only
# fails on a visitor's first question, which is the worst place to find out.
# The NumPy magic is a 0x93 byte followed by "NUMPY", so 8 bytes are read to
# cover the whole marker.
RUN set -eu; \
    dir="$HOME/data/indexes/reposage-ai"; \
    mkdir -p "$dir"; \
    for f in manifest.json chunks.jsonl vectors.npy vector_ids.txt \
             bm25.npz bm25_meta.json repo_map.txt; do \
        curl -fsSL --retry 3 -o "$dir/$f" "${INDEX_BASE}/${REPOSAGE_REF}/space/index/$f"; \
    done; \
    head -c 8 "$dir/vectors.npy" | grep -q NUMPY \
        || { echo "vectors.npy is not a NumPy array (Git LFS pointer?)"; exit 1; }; \
    python -c "import numpy,sys; a=numpy.load(sys.argv[1]); print('index vectors', a.shape, a.dtype)" "$dir/vectors.npy"; \
    test -s "$dir/vector_ids.txt"; \
    echo "index ready: $(wc -l < "$dir/vector_ids.txt") ids"

ENV REPOSAGE_DATA_DIR=$HOME/data \
    REPOSAGE_LOG_JSON=true \
    REPOSAGE_LOG_LEVEL=INFO \
    REPOSAGE_ENABLE_CACHE=true

# Demo mode: indexing endpoints are refused and the shared key is metered.
ENV REPOSAGE_DEMO_MODE=true \
    REPOSAGE_DEMO_INDEX=reposage-ai \
    REPOSAGE_DEMO_DAILY_BUDGET=150 \
    REPOSAGE_DEMO_VISITOR_BUDGET=4 \
    REPOSAGE_DEMO_REPO_URL=https://github.com/adwitiyashukla/reposage-ai

# One question is roughly six model calls; stay well inside the free tier.
ENV REPOSAGE_MAX_CONCURRENCY=2 \
    REPOSAGE_EMBED_CONCURRENCY=1 \
    REPOSAGE_EMBED_RPM=60

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "reposage.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
