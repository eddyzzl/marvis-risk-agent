from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.artifact import persist_model_meta, write_artifact_file
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.recipes.common import (
    artifact_params,
    compute_model_metrics,
    split_modeling_frame,
)

#: Recipes a seed-bagging ensemble member can be trained with (SEL-6). Deliberately
#: the same BINARY_MODELING_RECIPES family minus scorecard/ensemble itself --
#: scorecard's WOE-encoded space score fn signature differs (points, not raw
#: probability) and averaging it with a probability-space member would be
#: apples-to-oranges; a scorecard-only ensemble gains nothing seed-bagging can't
#: already do via multi-seed scorecard runs compared manually.
_MEMBER_RECIPES = frozenset({"lgb", "xgb", "catboost", "lr", "mlp"})

#: Default seed-bagging member count (SEL-6 spec default N=5).
DEFAULT_ENSEMBLE_N_MEMBERS = 5

#: TrainConfig.params key selecting the member recipe every seed-bagging member
#: reuses (default "lgb") -- must be one of _MEMBER_RECIPES.
ENSEMBLE_BASE_RECIPE_PARAM_KEY = "base_recipe"

#: TrainConfig.params key overriding the member count (default DEFAULT_ENSEMBLE_N_MEMBERS).
ENSEMBLE_N_MEMBERS_PARAM_KEY = "n_members"

#: Platform-only params consumed by the ensemble wrapper itself -- never passed
#: through to the member recipe's own training params.
_ENSEMBLE_ONLY_PARAM_KEYS = frozenset({ENSEMBLE_BASE_RECIPE_PARAM_KEY, ENSEMBLE_N_MEMBERS_PARAM_KEY})


def train_ensemble(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    """Seed-bagging ensemble recipe (SEL-6): trains ``n_members`` independent
    copies of ``base_recipe`` (default lgb, N=5) at deterministically-derived
    seeds, reusing that recipe's own training function unchanged, then scores
    by averaging the members' predicted probabilities. A optional participant
    in multi-recipe comparisons (never in the default recipes list) --
    stability/variance reduction is the entire value proposition, at N-times
    the training cost of a single member.

    The artifact stores each member's own algorithm + relative model_path +
    an equal weight (1/N) rather than re-serializing a combined model object,
    so scoring-time replay (_ModelArtifactScorer) loads each member file the
    same way a standalone artifact of that algorithm would. PMML export is
    intentionally unsupported (see artifact.py/tools.py's PMML_SUPPORTED_ALGORITHMS) --
    an ensemble of heterogeneous sklearn/native-booster members has no single
    PMML pipeline representation.
    """
    base_recipe = str(config.params.get(ENSEMBLE_BASE_RECIPE_PARAM_KEY) or "lgb").strip()
    if base_recipe not in _MEMBER_RECIPES:
        raise ModelingError(
            f"unsupported ensemble base_recipe: {base_recipe}; available: {', '.join(sorted(_MEMBER_RECIPES))}"
        )
    n_members = int(config.params.get(ENSEMBLE_N_MEMBERS_PARAM_KEY) or DEFAULT_ENSEMBLE_N_MEMBERS)
    if n_members < 2:
        raise ModelingError(f"ensemble n_members must be at least 2: {n_members}")

    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )

    member_params = {
        str(key): value
        for key, value in dict(config.params).items()
        if str(key) not in _ENSEMBLE_ONLY_PARAM_KEYS
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    member_artifacts: list[ModelArtifact] = []
    member_scorers: list = []
    for index in range(n_members):
        member_seed = _member_seed(config.seed, index)
        member_config = replace(
            config,
            params=dict(member_params),
            seed=member_seed,
            recipe_id=base_recipe,
        )
        member_result = _train_member(base_recipe, backend, dataset_path, member_config, out_dir=out_dir)
        member_artifacts.append(member_result.artifact)
        member_scorers.append(_member_score_fn(base_recipe, member_result.artifact, out_dir))

    weights = [1.0 / n_members] * n_members

    def score_fn(data):
        predictions = np.array([scorer(data) for scorer in member_scorers], dtype=float)
        return np.average(predictions, axis=0, weights=weights)

    metrics = compute_model_metrics(
        score_fn,
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_ensemble_artifact(
        member_artifacts,
        weights,
        base_recipe=base_recipe,
        kind="seed_bagging",
        config=config,
        out_dir=out_dir,
        params=artifact_params(member_params, config),
    )
    feature_importance = _averaged_importance(member_artifacts)
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=feature_importance,
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _train_member(recipe: str, backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    """Delegate to the member recipe's own, unchanged training function (SEL-6:
    "复用现有 recipe 训练函数"). Deliberately a small local dispatcher rather than
    importing tools.py's _train_recipe -- tools.py imports every recipe module
    (including this one), so importing it back here would be circular."""
    if recipe == "lgb":
        from marvis.packs.modeling.recipes.lgb import train_lgb

        return train_lgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "xgb":
        from marvis.packs.modeling.recipes.xgb import train_xgb

        return train_xgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "catboost":
        from marvis.packs.modeling.recipes.catboost import train_catboost

        return train_catboost(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "lr":
        from marvis.packs.modeling.recipes.lr import train_lr

        return train_lr(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "mlp":
        from marvis.packs.modeling.recipes.mlp import train_mlp

        return train_mlp(backend, dataset_path, config, out_dir=out_dir)
    raise ModelingError(f"unsupported ensemble member recipe: {recipe}")


def _member_score_fn(recipe: str, artifact: ModelArtifact, out_dir: Path):
    """A ``DataFrame -> np.ndarray`` probability score function for one already-
    trained member artifact, loaded fresh from disk -- mirrors
    _ModelArtifactScorer.raw_score's per-algorithm dispatch (tools.py), kept
    local here to avoid importing tools.py (circular import, see _train_member)."""
    from marvis.packs.modeling.artifact import load_model

    model = load_model(artifact, base_dir=out_dir)
    features = list(artifact.feature_list)
    if recipe == "xgb" and not hasattr(model, "predict_proba"):
        import xgboost as xgb

        def score(data):
            matrix = xgb.DMatrix(data[features], feature_names=features)
            return np.asarray(model.predict(matrix), dtype=float)

        return score

    def score(data):
        return np.asarray(model.predict_proba(data[features])[:, 1], dtype=float)

    return score


def _member_seed(seed: int, index: int) -> int:
    """Deterministic per-member seed derivation (mirrors tools.py's _recipe_seed
    pattern): same base seed + same member index always reproduces the same
    member seed, but different members never share an RNG stream."""
    digest = hashlib.sha256(f"{int(seed)}:ensemble_member:{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def _save_ensemble_artifact(
    member_artifacts: list[ModelArtifact],
    weights: list[float],
    *,
    base_recipe: str,
    kind: str,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.joblib"
    payload = {
        "kind": kind,
        "base_recipe": base_recipe,
        "members": [
            {
                "algorithm": member.algorithm,
                "model_path": member.model_path,
                "artifact_id": member.id,
            }
            for member in member_artifacts
        ],
        "weights": list(weights),
    }
    write_artifact_file(out_dir, model_path, lambda path: joblib.dump(payload, path))
    feature_list = member_artifacts[0].feature_list if member_artifacts else tuple(config.features)
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="ensemble",
        model_path=model_path,
        pmml_path=None,
        feature_list=feature_list,
        params={
            **dict(params),
            "ensemble_kind": kind,
            "ensemble_base_recipe": base_recipe,
            "ensemble_member_count": len(member_artifacts),
            "ensemble_member_artifact_ids": [member.id for member in member_artifacts],
        },
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _averaged_importance(
    member_artifacts: list[ModelArtifact],
) -> tuple[tuple[str, float], ...]:
    """Best-effort feature importance for the ensemble: not currently populated
    from members (each member's importance already lives in its own artifact
    row via model_card/feature_importance if inspected directly) -- an ensemble
    of heterogeneous algorithms has no single shared importance scale to average
    honestly, so this returns empty rather than a misleading blended ranking."""
    return ()


__all__ = [
    "DEFAULT_ENSEMBLE_N_MEMBERS",
    "ENSEMBLE_BASE_RECIPE_PARAM_KEY",
    "ENSEMBLE_N_MEMBERS_PARAM_KEY",
    "train_ensemble",
]
