"""Structured logging.

Human-readable colourised output for local development, single-line JSON in
containers so logs are directly ingestible by any log aggregator.
"""

from __future__ import annotations

import contextlib
import logging
import sys

import structlog

_CONFIGURED = False


def configure_console_encoding() -> None:
    """Force UTF-8 on stdout and stderr.

    Windows still defaults a redirected console to the legacy cp1252 code page,
    so any non-Latin-1 character (a box-drawing glyph from a Rich table, a
    severity emoji, a stray en dash in a docstring being echoed back) raises
    UnicodeEncodeError and takes the whole command down. Reconfiguring the
    streams costs nothing on platforms that were already UTF-8, and turns a
    crash into a replacement character on the one that was not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover - non-standard stream
            continue
        with contextlib.suppress(ValueError, OSError):  # pragma: no cover
            reconfigure(encoding="utf-8", errors="replace")


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    configure_console_encoding()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[return-value]
