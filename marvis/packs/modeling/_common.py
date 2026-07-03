from __future__ import annotations

import hashlib
import numpy as np
import re
from dataclasses import asdict, is_dataclass
from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.defaults import DEFAULT_RANDOM_SEED
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.report_compute import BusinessColumns
from pathlib import Path
from typing import Any


MODEL_REPORT_SCORE_COL = "__model_score__"


SCORECARD_POINTS_COL = "__scorecard_points__"


PMML_SUPPORTED_ALGORITHMS = frozenset({"lr", "lgb", "xgb", "scorecard"})


CALIBRATION_PARAMS_KEY = "calibration"


SUPPORTED_MODELING_RECIPES = frozenset({
    "lgb",
    "xgb",
    "catboost",
    "lr",
    "scorecard",
    "mlp",
    "lgb_regressor",
    "lgb_multiclass",
    # SEL-6: seed-bagging ensemble -- deliberately NOT in BINARY_MODELING_RECIPES
    # (never joins the default multi-algorithm arena / tuning budget), an
    # explicit opt-in participant only.
    "ensemble",
})


BINARY_MODELING_RECIPES = frozenset({"lgb", "xgb", "catboost", "lr", "scorecard", "mlp"})


CONTINUOUS_MODELING_RECIPES = frozenset({"lgb_regressor"})


MULTICLASS_MODELING_RECIPES = frozenset({"lgb_multiclass"})


def _positive_int_or_none(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _nonnegative_float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0 else None


def _finite_float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _score_first(row: dict, keys: tuple[str, ...], *, minimize: bool = False) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            number = float(value)
            return -number if minimize else number
    return float("-inf")


def _is_metric_key(key: str) -> bool:
    return key.startswith(("train_", "test_", "oot_", "psi_", "weighted_")) or key == "overfit_flag"


def _snapshot_latest_model_meta(base_dir: Path) -> bytes | None:
    path = Path(base_dir) / "model_meta.json"
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _cleanup_unattached_artifact(artifact: ModelArtifact, base_dir: Path, meta_snapshot: bytes | None) -> None:
    base = Path(base_dir)
    for relative in (artifact.model_path, artifact.pmml_path, f"{artifact.id}.model_meta.json"):
        if not relative:
            continue
        try:
            _resolve_artifact_path(str(relative), base_dir=base).unlink(missing_ok=True)
        except OSError:
            pass
    latest = base / "model_meta.json"
    try:
        if meta_snapshot is None:
            latest.unlink(missing_ok=True)
        else:
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_bytes(meta_snapshot)
    except OSError:
        pass


def _resolve_artifact_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _effective_seed(inputs: dict, ctx) -> int:
    if inputs.get("seed") is not None:
        return int(inputs["seed"])
    if getattr(ctx, "seed", None) is not None:
        return int(ctx.seed)
    return DEFAULT_RANDOM_SEED


def _recipe_seed(seed: int, recipe: str) -> int:
    """Deterministic per-recipe seed derivation (TUNE-1): every recipe's search
    gets its own seed so trial sequences don't collide across algorithms, but the
    derivation is a pure function of (seed, recipe) — same base seed always
    reproduces the same per-recipe trial sequence."""
    digest = hashlib.sha256(f"{seed}:{recipe}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def _training_params(inputs: dict) -> dict:
    params = dict(inputs.get("params") or {})
    return {**params, **_training_control_params(inputs, params)}


def _training_control_params(inputs: dict, params: dict | None = None) -> dict:
    params = dict(params or {})
    controls = {}
    for key in ("sample_weight_col", "sample_weight_column", "weight_col"):
        value = inputs.get(key, params.get(key))
        if value not in (None, ""):
            controls["sample_weight_col"] = str(value).strip()
            break
    constraints = inputs.get(
        "monotone_constraints",
        inputs.get(
            "monotonic_constraints",
            params.get("monotone_constraints", params.get("monotonic_constraints")),
        ),
    )
    if constraints not in (None, ""):
        controls["monotone_constraints"] = constraints
    return controls


def _normalize_recipe_list(value) -> list[str]:
    recipes = _unique_strings(value if isinstance(value, list) else [value])
    if not recipes:
        recipes = ["lgb"]
    unsupported = [recipe for recipe in recipes if recipe not in SUPPORTED_MODELING_RECIPES]
    if unsupported:
        raise ModelingError(
            f"unsupported modeling recipe(s): {', '.join(unsupported)}; "
            f"available: {', '.join(sorted(SUPPORTED_MODELING_RECIPES))}"
        )
    _target_type_from_recipes(recipes)
    return recipes


def _target_type_from_recipes(recipes: list[str]) -> str:
    has_binary = any(recipe in BINARY_MODELING_RECIPES for recipe in recipes)
    has_continuous = any(recipe in CONTINUOUS_MODELING_RECIPES for recipe in recipes)
    has_multiclass = any(recipe in MULTICLASS_MODELING_RECIPES for recipe in recipes)
    family_count = sum(1 for flag in (has_binary, has_continuous, has_multiclass) if flag)
    if family_count > 1:
        raise ModelingError("binary, continuous, and multiclass recipes cannot be mixed in one modeling spec")
    if has_continuous:
        return "continuous"
    if has_multiclass:
        return "multiclass"
    return "binary"


def _normalize_modeling_target_type(value) -> str | None:
    target_type = str(value or "").strip().lower()
    if not target_type:
        return None
    if target_type not in {"binary", "continuous", "multiclass"}:
        raise ModelingError(f"unsupported target_type: {target_type}")
    return target_type


def _metric_policy_for_target_type(target_type: str) -> str:
    if target_type == "continuous":
        return "lower OOT RMSE, fallback lower test RMSE"
    if target_type == "multiclass":
        return "higher OOT macro-AUC, fallback higher test macro-AUC then lower logloss"
    # Binary champion selection uses test KS (overfit-penalized); OOT is reported only,
    # never used to pick a winner — mirrors tune_hyperparameters' policy (DOM-9).
    return "higher overfit-penalized test KS; OOT reported only, not used for selection"


def _eligible_algorithms(target_type: str) -> list[str]:
    if target_type == "continuous":
        return sorted(CONTINUOUS_MODELING_RECIPES)
    if target_type == "multiclass":
        return sorted(MULTICLASS_MODELING_RECIPES)
    return sorted(BINARY_MODELING_RECIPES)


def _disabled_algorithms(target_type: str) -> list[dict]:
    disabled = []
    eligible = set(_eligible_algorithms(target_type))
    for recipe in sorted(SUPPORTED_MODELING_RECIPES - eligible):
        disabled.append({
            "recipe": recipe,
            "reason": f"recipe target family does not match `{target_type}`",
        })
    return disabled


def _unique_strings(values) -> list[str]:
    return [value for value in dict.fromkeys(str(item).strip() for item in (values or [])) if value]


def _business_columns(payload: dict) -> BusinessColumns:
    return BusinessColumns(
        loan_month_col=_optional_str(payload.get("loan_month_col")),
        interest_rate_col=_optional_str(payload.get("interest_rate_col")),
        loan_amount_col=_optional_str(payload.get("loan_amount_col")),
        term_col=_optional_str(payload.get("term_col")),
        drawdown_amount_col=_optional_str(payload.get("drawdown_amount_col")),
        credit_limit_col=_optional_str(payload.get("credit_limit_col")),
        mob_observe_cols=tuple(str(item) for item in payload.get("mob_observe_cols") or ()),
    )


def _section_available(statuses, section: str) -> bool:
    return any(status.section == section and status.available for status in statuses)


def _unique_columns(values) -> list[str]:
    columns = []
    for value in values:
        if value and str(value) not in columns:
            columns.append(str(value))
    return columns


_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")


def _allowed_number_tokens(value) -> set[str]:
    tokens: set[str] = set()

    def visit(item) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            numeric = float(item)
            tokens.add(_format_number_token(numeric))
            tokens.add(str(item))
            return
        if isinstance(item, str):
            for match in _NUMBER_TOKEN_RE.finditer(item):
                tokens.add(match.group(0))

    visit(value)
    return {token for token in tokens if token}


def _number_token_allowed(token: str, allowed: set[str]) -> bool:
    if token in allowed:
        return True
    if token.endswith("%"):
        return False
    try:
        numeric = float(token)
    except ValueError:
        return False
    return _format_number_token(numeric) in allowed


def _format_number_token(value: float) -> str:
    return f"{value:.12g}"


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _jsonable(value: Any):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value: Any):
    """Like _jsonable, but additionally maps NaN/inf to None so the payload is strict
    JSON (no NaN/Infinity tokens) for the make_split sample analysis."""
    cleaned = _jsonable(value)
    return _strip_non_finite(cleaned)


def _strip_non_finite(value: Any):
    if isinstance(value, dict):
        return {key: _strip_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_non_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
