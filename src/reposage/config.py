from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    github_token: str = Field(default="", alias="GITHUB_TOKEN")

    fast_model: str = Field(default="gemini-flash-lite-latest", alias="REPOSAGE_FAST_MODEL")
    deep_model: str = Field(default="gemini-flash-latest", alias="REPOSAGE_DEEP_MODEL")
    embed_model: str = Field(default="gemini-embedding-001", alias="REPOSAGE_EMBED_MODEL")
    embed_batch_size: int = Field(default=32, alias="REPOSAGE_EMBED_BATCH_SIZE")
    embed_dimensions: int = Field(default=768, alias="REPOSAGE_EMBED_DIMENSIONS")

    top_k: int = Field(default=12, alias="REPOSAGE_TOP_K")
    candidate_k: int = Field(default=40, alias="REPOSAGE_CANDIDATE_K")
    rrf_k: int = Field(default=60, alias="REPOSAGE_RRF_K")
    enable_rerank: bool = Field(default=True, alias="REPOSAGE_ENABLE_RERANK")

    max_agent_steps: int = Field(default=8, alias="REPOSAGE_MAX_AGENT_STEPS")
    max_refinements: int = Field(default=2, alias="REPOSAGE_MAX_REFINEMENTS")
    enable_critic: bool = Field(default=True, alias="REPOSAGE_ENABLE_CRITIC")

    max_file_bytes: int = Field(default=400_000, alias="REPOSAGE_MAX_FILE_BYTES")
    max_files: int = Field(default=4_000, alias="REPOSAGE_MAX_FILES")
    chunk_max_lines: int = Field(default=120, alias="REPOSAGE_CHUNK_MAX_LINES")
    chunk_overlap_lines: int = Field(default=15, alias="REPOSAGE_CHUNK_OVERLAP_LINES")

    demo_mode: bool = Field(default=False, alias="REPOSAGE_DEMO_MODE")
    demo_index: str = Field(default="", alias="REPOSAGE_DEMO_INDEX")
    demo_daily_budget: int = Field(default=200, alias="REPOSAGE_DEMO_DAILY_BUDGET")
    demo_visitor_budget: int = Field(default=5, alias="REPOSAGE_DEMO_VISITOR_BUDGET")
    demo_repo_url: str = Field(
        default="https://github.com/adwitiyashukla/reposage-ai",
        alias="REPOSAGE_DEMO_REPO_URL",
    )

    data_dir: Path = Field(default=Path(".reposage"), alias="REPOSAGE_DATA_DIR")
    enable_cache: bool = Field(default=True, alias="REPOSAGE_ENABLE_CACHE")
    cache_ttl_seconds: int = Field(default=604_800, alias="REPOSAGE_CACHE_TTL_SECONDS")
    log_level: str = Field(default="INFO", alias="REPOSAGE_LOG_LEVEL")
    log_json: bool = Field(default=False, alias="REPOSAGE_LOG_JSON")
    request_timeout: float = Field(default=120.0, alias="REPOSAGE_REQUEST_TIMEOUT")
    max_retries: int = Field(default=6, alias="REPOSAGE_MAX_RETRIES")
    max_concurrency: int = Field(default=8, alias="REPOSAGE_MAX_CONCURRENCY")
    max_rpm: int = Field(default=0, alias="REPOSAGE_MAX_RPM")
    embed_rpm: int = Field(default=75, alias="REPOSAGE_EMBED_RPM")
    embed_concurrency: int = Field(default=1, alias="REPOSAGE_EMBED_CONCURRENCY")

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: str | Path) -> Path:
        return Path(str(v)).expanduser()

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        return str(v).upper()

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key != "your-key-here")

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def repo_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir, self.repo_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def fingerprint(self) -> dict[str, object]:
        return {
            "fast_model": self.fast_model,
            "deep_model": self.deep_model,
            "embed_model": self.embed_model,
            "embed_dimensions": self.embed_dimensions,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "enable_rerank": self.enable_rerank,
            "enable_critic": self.enable_critic,
            "max_refinements": self.max_refinements,
            "chunk_max_lines": self.chunk_max_lines,
            "chunk_overlap_lines": self.chunk_overlap_lines,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
