from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from reposage.config import Settings
from reposage.llm.base import LLMResponse
from reposage.llm.cache import ResponseCache
from reposage.llm.client import LLMClient

SAMPLE_PYTHON = '''"""Authentication helpers."""

import os
from typing import Optional

MAX_RETRIES = 3
DEFAULT_ALGORITHM = "HS256"


class TokenValidator:
    """Validates signed session tokens."""

    def __init__(self, secret: str, algorithm: str = DEFAULT_ALGORITHM) -> None:
        self.secret = secret
        self.algorithm = algorithm

    def verify_jwt(self, token: str) -> Optional[dict]:
        """Verify a JWT and return its claims, or None when invalid."""
        parts = token.split(".")
        if len(parts) != 3:
            raise InvalidSignature("malformed token")
        header, payload, signature = parts
        if not self._check(header, payload, signature):
            return None
        return decode_payload(payload)

    def _check(self, header: str, payload: str, signature: str) -> bool:
        expected = sign(f"{header}.{payload}", self.secret, self.algorithm)
        return compare_digest(expected, signature)

    def refresh(self, token: str) -> str:
        """Issue a new token from a still-valid one."""
        claims = self.verify_jwt(token)
        if claims is None:
            raise InvalidSignature("cannot refresh an invalid token")
        return issue(claims, self.secret)


def build_validator(secret: Optional[str] = None) -> TokenValidator:
    """Construct a validator from the environment."""
    return TokenValidator(secret or os.environ["SESSION_SECRET"])
'''

SAMPLE_DIFF = """diff --git a/src/api.py b/src/api.py
index abc123..def456 100644
--- a/src/api.py
+++ b/src/api.py
@@ -10,7 +10,9 @@ class Handler:
     def get(self, user_id):
-        return db.query(f"SELECT * FROM users WHERE id={user_id}")
+        row = db.query("SELECT * FROM users WHERE id=%s", user_id)
+        if row is None:
+            raise NotFound(user_id)
+        return row

     def close(self):
         self.conn.close()
diff --git a/docs/README.md b/docs/README.md
new file mode 100644
--- /dev/null
+++ b/docs/README.md
@@ -0,0 +1,2 @@
+# Title
+Body text
"""


class FakeProvider:
    name = "fake"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.calls: list[str] = []
        self.embed_calls = 0

    async def generate(
        self,
        prompt,
        *,
        model,
        system=None,
        temperature=0.2,
        max_output_tokens=4096,
        json_mode=False,
        history=None,
    ) -> LLMResponse:
        self.calls.append((system or "")[:40])
        text = self._respond(system or "", prompt)
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
        )

    async def stream(self, prompt, *, model, system=None, **kwargs):
        response = await self.generate(prompt, model=model, system=system)
        for start in range(0, len(response.text), 20):
            yield response.text[start : start + 20]

    async def embed(self, texts, *, model, task_type="RETRIEVAL_DOCUMENT", dimensions=None):
        self.embed_calls += 1
        return [
            np.random.default_rng(abs(hash(text[:256])) % (2**31)).normal(size=self.dim).tolist()
            for text in texts
        ]

    async def aclose(self) -> None:
        return None

    @staticmethod
    def _respond(system: str, prompt: str) -> str:
        if "planning stage" in system:
            return json.dumps(
                {
                    "intent": "explain",
                    "restated_question": "How does token validation work?",
                    "sub_questions": [
                        {
                            "question": "How is a JWT verified?",
                            "search_queries": ["verify_jwt signature", "token validation"],
                            "rationale": "core",
                        }
                    ],
                    "keyword_hints": ["verify_jwt", "TokenValidator"],
                    "path_hints": ["auth"],
                    "needs_retrieval": True,
                }
            )
        if "reranker" in system:
            return json.dumps({"scores": [{"id": i, "score": 9.0 - i * 0.25} for i in range(24)]})
        if "audit draft answers" in system:
            return json.dumps(
                {
                    "grounded": True,
                    "complete": True,
                    "confidence": 0.85,
                    "issues": [],
                    "follow_up_queries": [],
                    "verdict": "accept",
                }
            )
        if "grade answers" in system:
            return json.dumps(
                {
                    "reasoning": "Accurate and well cited.",
                    "correctness": 5,
                    "groundedness": 5,
                    "completeness": 4,
                    "hallucinations": [],
                }
            )
        if "reviewing a pull request" in system:
            return json.dumps(
                {
                    "findings": [
                        {
                            "line": 11,
                            "severity": "high",
                            "category": "security",
                            "title": "SQL built by string interpolation",
                            "body": "The query interpolates user input directly.",
                            "suggestion": None,
                            "confidence": 0.9,
                        }
                    ],
                    "verdict": "changes",
                }
            )
        return (
            "Token validation lives in `TokenValidator` [auth/jwt.py:10-38]. "
            "The `verify_jwt` method splits the token and checks the signature "
            "[auth/jwt.py:17-25]. A reference to [does/not/exist.py:1-4] is deliberately invalid."
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        GEMINI_API_KEY="test-key",
        REPOSAGE_DATA_DIR=str(tmp_path / ".reposage"),
        REPOSAGE_ENABLE_CACHE=False,
        REPOSAGE_TOP_K=6,
        REPOSAGE_CANDIDATE_K=20,
        REPOSAGE_MAX_REFINEMENTS=1,
    )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def client(provider: FakeProvider, settings: Settings) -> LLMClient:
    settings.ensure_dirs()
    return LLMClient(
        provider=provider,
        settings=settings,
        cache=ResponseCache(settings.cache_dir, enabled=False),
    )


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample-repo"
    (root / "auth").mkdir(parents=True)
    (root / "web").mkdir()
    (root / "node_modules" / "junk").mkdir(parents=True)

    (root / "auth" / "jwt.py").write_text(SAMPLE_PYTHON, encoding="utf-8")
    (root / "web" / "app.js").write_text(
        "import express from 'express';\n\n"
        "export function createServer(port) {\n"
        "  const app = express();\n"
        "  app.get('/health', (req, res) => res.json({ ok: true }));\n"
        "  return app.listen(port);\n"
        "}\n\n"
        "export class Router {\n"
        "  constructor(routes) { this.routes = routes; }\n"
        "  dispatch(path) { return this.routes[path]; }\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Sample\n\nA tiny project.\n\n## Auth\n\nTokens are validated by TokenValidator.\n",
        encoding="utf-8",
    )
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}', encoding="utf-8")
    (root / "node_modules" / "junk" / "index.js").write_text(
        "module.exports = 1;", encoding="utf-8"
    )
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    return root


@pytest.fixture
def sample_diff() -> str:
    return SAMPLE_DIFF


def requires_live_api() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))
