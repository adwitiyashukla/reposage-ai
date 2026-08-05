# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build.
#
# The builder installs into a virtualenv which is then copied wholesale into a
# clean runtime image. Compilers and build headers never reach the final layer,
# which keeps the image small and removes a whole class of attack surface.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# Copy only what the dependency resolution needs first, so the expensive install
# layer is cached until the manifest itself changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[treesitter]"

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="RepoSage" \
      org.opencontainers.image.description="Agentic code intelligence: hybrid retrieval, multi-agent reasoning and automated PR review for any Git repository." \
      org.opencontainers.image.source="https://github.com/adwitiyashukla/reposage" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REPOSAGE_DATA_DIR=/data \
    REPOSAGE_LOG_JSON=true

# git is a runtime dependency: repositories are cloned, not vendored.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user. /data is a volume so indexes survive restarts.
RUN useradd --create-home --uid 10001 reposage \
    && mkdir -p /data \
    && chown -R reposage:reposage /data

COPY --from=builder /opt/venv /opt/venv

USER reposage
WORKDIR /home/reposage
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["uvicorn", "reposage.api.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
