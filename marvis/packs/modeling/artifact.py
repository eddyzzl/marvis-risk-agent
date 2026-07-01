from __future__ import annotations

import uuid
from datetime import UTC, datetime
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork, TransactionalArtifactStore
from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.errors import ModelingError


_MODEL_SUFFIX = {
    "lgb": ".pkl",
    "lgb_regressor": ".txt",
    "xgb": ".pkl",
    "catboost": ".pkl",
    "lr": ".joblib",
    "scorecard": ".joblib",
    "mlp": ".joblib",
}


def write_artifact_file(
    out_dir: Path,
    filename: str,
    writer,
    *,
    validator=None,
) -> Path:
    artifact = TransactionalArtifactStore(Path(out_dir)).stage(filename)
    try:
        writer(artifact.path)
        if validator is not None:
            validator(artifact.path)
        final_path = artifact.promote()
        artifact.commit()
        return final_path
    except Exception:
        artifact.rollback()
        raise


def save_model(
    model,
    algorithm: str,
    out_dir: Path,
    *,
    feature_list,
    params,
    woe_maps=None,
    scorecard_table=None,
) -> ModelArtifact:
    algorithm = str(algorithm)
    if algorithm not in _MODEL_SUFFIX:
        raise ModelingError(f"unsupported model algorithm: {algorithm}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}{_MODEL_SUFFIX[algorithm]}"
    if algorithm in {"lgb", "xgb", "catboost"}:
        write_artifact_file(out_dir, model_path, lambda path: joblib.dump(model, path))
    elif algorithm == "lgb_regressor":
        write_artifact_file(out_dir, model_path, model.save_model)
    elif algorithm == "scorecard" and not isinstance(model, dict):
        payload = {
            "model": model,
            "woe_maps": woe_maps or {},
            "params": dict(params),
            "features": tuple(feature_list),
            "scorecard_table": list(scorecard_table or []),
        }
        write_artifact_file(out_dir, model_path, lambda path: joblib.dump(payload, path))
    else:
        write_artifact_file(out_dir, model_path, lambda path: joblib.dump(model, path))
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm=algorithm,
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(feature_list),
        params=dict(params),
        woe_maps=woe_maps,
        created_at=datetime.now(UTC).isoformat(),
        scorecard_table=tuple(dict(item) for item in (scorecard_table or [])),
    )
    persist_model_meta(out_dir, artifact)
    return artifact


def persist_model_meta(
    out_dir: Path,
    artifact: ModelArtifact,
    *,
    config=None,
    uow: ArtifactUnitOfWork | None = None,
) -> Path:
    out_dir = Path(out_dir)
    meta = {
        "artifact_id": artifact.id,
        "algorithm": artifact.algorithm,
        "model_path": artifact.model_path,
        "pmml_path": artifact.pmml_path,
        "feature_list": list(artifact.feature_list),
        "params": _jsonable(artifact.params),
        "seed": _config_seed(config) if config is not None else _seed_from_params(artifact.params),
        "dataset_id": getattr(config, "dataset_id", None),
        "target_col": getattr(config, "target_col", None),
        "split_col": getattr(config, "split_col", None),
        "split_values": _jsonable(getattr(config, "split_values", None)),
        "target_type": getattr(config, "target_type", None),
        "recipe_id": getattr(config, "recipe_id", None),
        "scorecard_table": _jsonable(artifact.scorecard_table),
        "created_at": artifact.created_at,
    }
    meta_path = out_dir / f"{artifact.id}.model_meta.json"
    payload = json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str)
    if uow is not None:
        staged = [
            uow.stage_file(out_dir, meta_path.name),
            uow.stage_file(out_dir, "model_meta.json"),
        ]
        for artifact_file in staged:
            artifact_file.path.write_text(payload + "\n", encoding="utf-8")
        return meta_path

    store = TransactionalArtifactStore(out_dir)
    staged = [store.stage(meta_path.name), store.stage("model_meta.json")]
    try:
        for artifact_file in staged:
            artifact_file.path.write_text(payload + "\n", encoding="utf-8")
        for artifact_file in staged:
            artifact_file.promote()
        for artifact_file in staged:
            artifact_file.commit()
    except Exception:
        for artifact_file in reversed(staged):
            artifact_file.rollback()
        raise
    return meta_path


def _config_seed(config) -> int | None:
    value = getattr(config, "seed", None)
    return int(value) if value is not None else None


def _seed_from_params(params: dict) -> int | None:
    for key in ("seed", "random_state", "random_seed"):
        value = dict(params or {}).get(key)
        if value not in (None, ""):
            return int(value)
    return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def load_model(artifact: ModelArtifact, *, base_dir: Path):
    path = Path(base_dir) / artifact.model_path
    if not path.exists():
        raise ModelingError(f"model file does not exist: {artifact.model_path}")
    if artifact.algorithm == "lgb":
        if path.suffix in {".joblib", ".pkl"}:
            return joblib.load(path)
        import lightgbm as lgb

        return lgb.Booster(model_file=path.as_posix())
    if artifact.algorithm == "lgb_regressor":
        import lightgbm as lgb

        return lgb.Booster(model_file=path.as_posix())
    if artifact.algorithm == "xgb":
        if path.suffix in {".joblib", ".pkl"}:
            return joblib.load(path)
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(path)
        return model
    if artifact.algorithm in {"catboost", "lr", "scorecard", "mlp"}:
        return joblib.load(path)
    raise ModelingError(f"unsupported model algorithm: {artifact.algorithm}")


def export_pmml(
    artifact: ModelArtifact,
    dataset_path: Path,
    out_path: Path,
    *,
    base_dir: Path,
    target_col: str | None = None,
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
    target_name = _target_name(
        Path(dataset_path),
        list(artifact.feature_list),
        target_col=target_col,
        ignored_fields=_pmml_ignored_fields(artifact),
    )
    if artifact.algorithm == "scorecard":
        pipeline = _scorecard_pmml_pipeline(
            model,
            list(artifact.feature_list),
            target_name,
            schema_sample,
        )
    else:
        if not (hasattr(model, "fit") and (hasattr(model, "predict_proba") or hasattr(model, "predict"))):
            raise ModelingError(
                "PMML export requires a sklearn-compatible fitted model object; "
                "native LightGBM/XGBoost Booster artifacts require a dedicated JPMML export path."
            )
        pipeline = make_pmml_pipeline(
            model,
            active_fields=list(artifact.feature_list),
            target_fields=[target_name],
        )
    out_path = Path(out_path)
    return write_artifact_file(
        out_path.parent,
        out_path.name,
        lambda path: sklearn2pmml(pipeline, path.as_posix(), with_repr=True),
        validator=lambda path: Model.load(path.as_posix()),
    )


def _scorecard_pmml_pipeline(
    model_payload,
    feature_list: list[str],
    target_name: str,
    sample_frame: pd.DataFrame,
):
    validate_scorecard_pmml_payload(model_payload, feature_list=feature_list)
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


def validate_scorecard_pmml_payload(model_payload, *, feature_list: list[str]) -> None:
    if not isinstance(model_payload, dict) or "model" not in model_payload or "woe_maps" not in model_payload:
        raise ModelingError("scorecard PMML export requires a saved scorecard payload with model and woe_maps")
    woe_maps = model_payload["woe_maps"]
    if not isinstance(woe_maps, dict):
        raise ModelingError("scorecard PMML export requires woe_maps to be a mapping")
    for feature in feature_list:
        if feature not in woe_maps:
            raise ModelingError(f"scorecard WOE map missing feature: {feature}")
        _validate_woe_map(feature, woe_maps[feature])


def _validate_woe_map(feature: str, woe) -> None:
    edges = list(_woe_get(woe, "edges") or ())
    values = list(_woe_get(woe, "woe_by_bin") or ())
    if len(values) != max(0, len(edges) - 1):
        raise ModelingError(f"scorecard WOE map invalid for feature: {feature}")
    for edge in edges:
        try:
            float(edge)
        except (TypeError, ValueError) as exc:
            raise ModelingError(f"scorecard WOE map invalid for feature: {feature}") from exc
    for value in [*values, _woe_get(woe, "na_woe")]:
        if value is None:
            continue
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise ModelingError(f"scorecard WOE map invalid for feature: {feature}") from exc


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


def _target_name(
    path: Path,
    feature_list: list[str],
    *,
    target_col: str | None = None,
    ignored_fields: set[str] | None = None,
) -> str:
    explicit = str(target_col or "").strip()
    if explicit:
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        columns = list(pd.read_parquet(path).columns)
    elif suffix == ".csv":
        columns = list(pd.read_csv(path, nrows=0).columns)
    else:
        return "target"
    features = set(feature_list)
    ignored = set(ignored_fields or set())
    ignored.update({"split", "model_flag", "sample_weight", "weight"})
    return next((column for column in columns if column not in features and column not in ignored), "target")


def _pmml_ignored_fields(artifact: ModelArtifact) -> set[str]:
    params = dict(artifact.params or {})
    ignored = {str(params.get(key) or "").strip() for key in ("sample_weight_col", "split_col")}
    ignored.update(str(item or "").strip() for item in params.get("passthrough_cols") or [])
    return {item for item in ignored if item}


__all__ = [
    "export_pmml",
    "load_model",
    "persist_model_meta",
    "save_model",
    "validate_scorecard_pmml_payload",
    "write_artifact_file",
]
