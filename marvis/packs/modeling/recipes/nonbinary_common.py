from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from marvis.data.errors import NanLabelNotConfirmedError
from marvis.packs.modeling.artifact import (
    persist_model_meta,
    points_direction_for_algorithm,
    score_direction_for_algorithm,
    write_artifact_file,
)
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.recipes.common import artifact_params


def resolve_multiclass_classes(target) -> tuple:
    values = [
        normalize_scalar(value)
        for value in target.tolist()
        if value is not None and not _is_nan(value)
    ]
    classes = tuple(sorted(set(values), key=lambda value: (type(value).__name__, str(value))))
    if len(classes) < 2:
        raise ModelingError("multiclass training requires at least two observed train classes")
    return classes


def encode_multiclass_target(target, classes: tuple) -> np.ndarray:
    class_to_index = {value: index for index, value in enumerate(classes)}
    try:
        return np.asarray(
            [class_to_index[normalize_scalar(value)] for value in target.tolist()],
            dtype=int,
        )
    except KeyError as exc:
        raise ModelingError(
            f"multiclass split contains a class absent from train: {exc.args[0]}"
        ) from exc


def resolve_multiclass_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    *,
    target_col: str,
    drop_nan_labels: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, bool, dict]:
    """Resolve missing labels without forcing class labels to be numeric.

    The shared binary/regression label gate deliberately rejects non-numeric
    labels.  Multiclass recipes, however, support stable string labels (for
    example ``low/medium/high``), so they need the same train/test/OOT
    confirmation policy with a class-safe missing-value mask.
    """

    splits: dict[str, pd.DataFrame] = {"train": train, "test": test}
    if oot is not None:
        splits["oot"] = oot

    masks: dict[str, np.ndarray] = {}
    by_split: dict[str, dict] = {}
    for role, rows in splits.items():
        mask = _multiclass_missing_mask(rows[target_col])
        masks[role] = mask
        by_split[role] = {"n_total": int(len(rows)), "n_nan": int(mask.sum())}

    required_nan = by_split["train"]["n_nan"] + by_split["test"]["n_nan"]
    if required_nan and not drop_nan_labels:
        raise NanLabelNotConfirmedError(
            target_col=target_col,
            n_total=int(len(train) + len(test)),
            n_nan=int(required_nan),
            scope="train/test",
            by_split=by_split,
        )

    train_clean = train.loc[~masks["train"]]
    test_clean = test.loc[~masks["test"]]
    total_dropped = int(required_nan)

    oot_clean = oot
    oot_has_labels = False
    if oot is not None:
        n_nan = by_split["oot"]["n_nan"]
        total = by_split["oot"]["n_total"]
        if total == 0 or n_nan == total:
            oot_has_labels = False
        elif n_nan == 0:
            oot_has_labels = True
        else:
            if not drop_nan_labels:
                raise NanLabelNotConfirmedError(
                    target_col=target_col,
                    n_total=int(total),
                    n_nan=int(n_nan),
                    scope="oot",
                    by_split=by_split,
                )
            oot_clean = oot.loc[~masks["oot"]]
            oot_has_labels = True
            total_dropped += int(n_nan)

    audit = {"by_split": by_split, "total_dropped": int(total_dropped)}
    return train_clean, test_clean, oot_clean, oot_has_labels, audit


def strict_json_classes(classes: tuple) -> list:
    return [normalize_scalar(value) for value in classes]


def save_joblib_nonbinary_artifact(
    model,
    *,
    algorithm: str,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
    classes: tuple | None = None,
    per_class: dict | None = None,
) -> ModelArtifact:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.joblib"
    write_artifact_file(out_dir, model_path, lambda path: joblib.dump(model, path))
    stored_params = artifact_params(dict(params), config)
    if classes is not None:
        stored_params["classes"] = strict_json_classes(classes)
    if per_class is not None:
        stored_params["per_class"] = per_class
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm=algorithm,
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=stored_params,
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
        score_direction=score_direction_for_algorithm(algorithm),
        points_direction=points_direction_for_algorithm(algorithm),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def sklearn_linear_importance(model, features: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    coefficients = np.asarray(estimator.coef_, dtype=float)
    if coefficients.ndim == 1:
        values = np.abs(coefficients)
    else:
        values = np.mean(np.abs(coefficients), axis=0)
    pairs = sorted(
        zip(features, values, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return tuple((feature, float(value)) for feature, value in pairs)


def xgb_importance(model, features: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    values = np.asarray(getattr(model, "feature_importances_", ()), dtype=float)
    if values.shape != (len(features),):
        return tuple((feature, 0.0) for feature in features)
    return tuple((feature, float(value)) for feature, value in zip(features, values, strict=True))


def normalize_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _is_nan(value) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False


def _multiclass_missing_mask(target) -> np.ndarray:
    mask = pd.isna(target).to_numpy(dtype=bool)
    values = target.to_numpy(dtype=object)
    for index, value in enumerate(values):
        if mask[index]:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            try:
                mask[index] = not np.isfinite(float(value))
            except (TypeError, ValueError):
                pass
    return mask


__all__ = [
    "encode_multiclass_target",
    "resolve_multiclass_classes",
    "resolve_multiclass_splits",
    "save_joblib_nonbinary_artifact",
    "sklearn_linear_importance",
    "strict_json_classes",
    "xgb_importance",
]
