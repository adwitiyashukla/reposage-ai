"""Token pricing tables and cost estimation.

Prices are USD per one million tokens and reflect published list prices for the
paid tier. Runs on a provider's free tier cost nothing in practice; we still
account for them so that the numbers reported in evaluations translate directly
to a production budget.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    note: str = ""


# Keys are matched by longest prefix, so "gemini-2.0-flash-001" resolves to the
# "gemini-2.0-flash" entry without needing an exhaustive table.
PRICING: dict[str, ModelPricing] = {
    # Rolling aliases. Priced as the model they currently resolve to.
    "gemini-flash-lite-latest": ModelPricing(0.10, 0.40),
    "gemini-flash-latest": ModelPricing(0.30, 2.50),
    "gemini-pro-latest": ModelPricing(1.25, 10.00),
    "gemini-3-flash": ModelPricing(0.30, 2.50),
    "gemini-3-pro": ModelPricing(1.25, 10.00),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00),
    "gemini-2.5-flash-lite": ModelPricing(0.10, 0.40),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-2.0-flash-lite": ModelPricing(0.075, 0.30),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),
    "text-embedding-004": ModelPricing(0.0, 0.0, "free tier"),
    "gemini-embedding-001": ModelPricing(0.15, 0.0),
    "gemini-embedding-2": ModelPricing(0.15, 0.0),
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
    "llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
}

_FALLBACK = ModelPricing(0.10, 0.40, "unknown model, assumed flash-class")


def lookup(model: str) -> ModelPricing:
    """Longest-prefix match so versioned model ids resolve correctly."""
    normalised = model.strip().lower().removeprefix("models/")
    best: tuple[int, ModelPricing] | None = None
    for key, pricing in PRICING.items():
        if normalised.startswith(key) and (best is None or len(key) > best[0]):
            best = (len(key), pricing)
    return best[1] if best else _FALLBACK


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    """USD cost of a single call, rounded to sub-cent precision."""
    pricing = lookup(model)
    cost = (
        prompt_tokens * pricing.input_per_mtok + completion_tokens * pricing.output_per_mtok
    ) / 1_000_000
    return round(cost, 8)
