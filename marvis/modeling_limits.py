"""Shared hard limits for governed modeling execution."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral


MIN_N_TRIALS = 1
MAX_N_TRIALS = 200


def normalize_n_trials(
    value: object,
    *,
    field: str = "n_trials",
    optional: bool = False,
) -> int | None:
    """Return a strict trial budget, rejecting coercion and unsafe bounds."""

    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{field} must be an integer between {MIN_N_TRIALS} and {MAX_N_TRIALS}"
        )
    normalized = int(value)
    if not MIN_N_TRIALS <= normalized <= MAX_N_TRIALS:
        raise ValueError(
            f"{field} must be an integer between {MIN_N_TRIALS} and {MAX_N_TRIALS}"
        )
    return normalized


def normalize_n_trials_by_recipe(value: object) -> dict[str, int]:
    """Validate every per-recipe budget with the same hard execution limit."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("n_trials_by_recipe must be an object")
    return {
        str(recipe): int(
            normalize_n_trials(
                budget,
                field=f"n_trials_by_recipe.{recipe}",
            )
        )
        for recipe, budget in value.items()
    }
