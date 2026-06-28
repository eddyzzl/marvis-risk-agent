from __future__ import annotations

import uuid
from datetime import UTC, datetime
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.errors import ModelingError


_MODEL_SUFFIX = {
    "lgb": ".joblib",
    "lgb_regressor": ".txt",
    "xgb": ".joblib",
    "lr": ".joblib",
    "scorecard": ".joblib",
    "mlp": ".joblib",
}


def save_model(
    model,
    algorithm: str,
    out_dir: Path,
    *,
    feature_list,
    params,
    woe_maps=None,
) -> ModelArtifact:
    algorithm = str(algorithm)
    if algorithm not in _MODEL_SUFFIX:
        raise ModelingError(f"unsupported model algorithm: {algorithm}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}{_MODEL_SUFFIX[algorithm]}"
    target = out_dir / model_path
    if algorithm in {"lgb", "xgb"}:
        joblib.dump(model, target)
    elif algorithm == "lgb_regressor":
        model.save_model(target)
    elif algorithm == "scorecard" and not isinstance(model, dict):
        joblib.dump(
            {
                "model": model,
                "woe_maps": woe_maps or {},
                "params": dict(params),
                "features": tuple(feature_list),
            },
            target,
        )
    else:
        joblib.dump(model, target)
    return ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm=algorithm,
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(feature_list),
        params=dict(params),
        woe_maps=woe_maps,
        created_at=datetime.now(UTC).isoformat(),
    )


def load_model(artifact: ModelArtifact, *, base_dir: Path):
    path = Path(base_dir) / artifact.model_path
    if not path.exists():
        raise ModelingError(f"model file does not exist: {artifact.model_path}")
    if artifact.algorithm == "lgb":
        if path.suffix == ".joblib":
            return joblib.load(path)
        import lightgbm as lgb

        return lgb.Booster(model_file=path.as_posix())
    if artifact.algorithm == "lgb_regressor":
        import lightgbm as lgb

        return lgb.Booster(model_file=path.as_posix())
    if artifact.algorithm == "xgb":
        if path.suffix == ".joblib":
            return joblib.load(path)
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(path)
        return model
    if artifact.algorithm in {"lr", "scorecard", "mlp"}:
        return joblib.load(path)
    raise ModelingError(f"unsupported model algorithm: {artifact.algorithm}")


def export_pmml(
    artifact: ModelArtifact,
    dataset_path: Path,
    out_path: Path,
    *,
    base_dir: Path,
) -> Path:
    if artifact.algorithm not in {"lr", "lgb", "xgb", "scorecard"}:
        raise ModelingError(f"PMML export is not supported for algorithm: {artifact.algorithm}")
    try:
        from pypmml import Model
        from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
    except ImportError as exc:
        raise ModelingError("PMML export requires sklearn2pmml and pypmml") from exc

    model = load_model(artifact, base_dir=base_dir)
    schema_sample = _read_schema_sample(Path(dataset_path), list(artifact.feature_list))
    target_name = _target_name(Path(dataset_path), list(artifact.feature_list))
    if artifact.algorithm == "scorecard":
        pipeline = _scorecard_pmml_pipeline(
            model,
            list(artifact.feature_list),
            target_name,
            schema_sample,
        )
    else:
        pipeline = make_pmml_pipeline(
            model,
            active_fields=list(artifact.feature_list),
            target_fields=[target_name],
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sklearn2pmml(pipeline, out_path.as_posix(), with_repr=True)
    Model.load(out_path.as_posix())
    return out_path


def _scorecard_pmml_pipeline(
    model_payload,
    feature_list: list[str],
    target_name: str,
    sample_frame: pd.DataFrame,
):
    if not isinstance(model_payload, dict) or "model" not in model_payload or "woe_maps" not in model_payload:
        raise ModelingError("scorecard PMML export requires a saved scorecard payload with model and woe_maps")
    try:
        from sklearn.pipeline import FeatureUnion
        from sklearn2pmml.decoration import Alias
        from sklearn2pmml.pipeline import PMMLPipeline
        from sklearn2pmml.preprocessing import ExpressionTransformer
    except ImportError as exc:
        raise ModelingError("scorecard PMML export requires sklearn2pmml") from exc

    woe_maps = model_payload["woe_maps"]
    transforms = []
    for feature in feature_list:
        if feature not in woe_maps:
            raise ModelingError(f"scorecard WOE map missing feature: {feature}")
        name = _safe_transform_name(feature)
        transforms.append((
            name,
            Alias(ExpressionTransformer(_woe_expression(feature, woe_maps[feature])), f"{name}_woe"),
        ))
    union = FeatureUnion(transforms)
    union.fit(sample_frame[feature_list])
    pipeline = PMMLPipeline([
        ("woe", union),
        ("classifier", model_payload["model"]),
    ])
    pipeline.active_fields = list(feature_list)
    pipeline.target_fields = [target_name]
    return pipeline


def _woe_expression(feature: str, woe) -> str:
    edges = list(_woe_get(woe, "edges") or ())
    values = list(_woe_get(woe, "woe_by_bin") or ())
    if len(values) != max(0, len(edges) - 1):
        raise ModelingError(f"scorecard WOE map invalid for feature: {feature}")
    default = _pmml_number(_woe_get(woe, "na_woe"))
    if default is None:
        default = "0.0"
    if not values:
        inner = default
    else:
        thresholds = [float(value) for value in edges[1:-1] if math.isfinite(float(value))]
        inner = _pmml_number(values[-1]) or "0.0"
        for index in range(len(thresholds) - 1, -1, -1):
            inner = (
                f"({_pmml_number(values[index]) or '0.0'} "
                f"if X[{json.dumps(feature)}] < {_pmml_number(thresholds[index])} else {inner})"
            )
    return f"({default} if pandas.isnull(X[{json.dumps(feature)}]) else {inner})"


def _woe_get(woe, field: str):
    if isinstance(woe, dict):
        return woe.get(field)
    return getattr(woe, field, None)


def _pmml_number(value) -> str | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return f"{number:.17g}"


def _safe_transform_name(feature: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(feature)).strip("_")
    return f"woe_{safe or 'feature'}"


def _read_schema_sample(path: Path, columns: list[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns, nrows=100)
    raise ModelingError(f"unsupported dataset format for PMML export: {path.suffix}")


def _target_name(path: Path, feature_list: list[str]) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        columns = list(pd.read_parquet(path).columns)
    elif suffix == ".csv":
        columns = list(pd.read_csv(path, nrows=0).columns)
    else:
        return "target"
    features = set(feature_list)
    return next((column for column in columns if column not in features), "target")


__all__ = ["export_pmml", "load_model", "save_model"]
