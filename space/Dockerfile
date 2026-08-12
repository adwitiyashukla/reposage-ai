FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

ARG REPOSAGE_REF=d8785482d79fcdf8500fb30440d228d31ce05deb
RUN pip install --no-cache-dir --user \
    "reposage[treesitter] @ git+https://github.com/adwitiyashukla/reposage-ai.git@${REPOSAGE_REF}"

ARG INDEX_BASE=https://huggingface.co/datasets/adwitiyashukla/reposage-index/resolve/main
RUN set -eu; \
    dir="$HOME/data/indexes/reposage-ai"; \
    mkdir -p "$dir"; \
    for f in manifest.json chunks.jsonl vectors.npy vector_ids.txt \
             bm25.npz bm25_meta.json repo_map.txt; do \
        curl -fsSL --retry 3 -o "$dir/$f" "${INDEX_BASE}/$f"; \
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

ENV REPOSAGE_DEMO_MODE=true \
    REPOSAGE_DEMO_INDEX=reposage-ai \
    REPOSAGE_DEMO_DAILY_BUDGET=150 \
    REPOSAGE_DEMO_VISITOR_BUDGET=4 \
    REPOSAGE_DEMO_REPO_URL=https://github.com/adwitiyashukla/reposage-ai

ENV REPOSAGE_MAX_CONCURRENCY=2 \
    REPOSAGE_EMBED_CONCURRENCY=1 \
    REPOSAGE_EMBED_RPM=60

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "reposage.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
