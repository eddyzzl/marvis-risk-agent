from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib

from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.errors import ModelingError


_MODEL_SUFFIX = {
    "lgb": ".txt",
    "xgb": ".json",
    "lr": ".joblib",
    "scorecard": ".joblib",
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
    if algorithm == "lgb":
        model.save_model(target)
    elif algorithm == "xgb":
        model.save_model(target)
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
        import lightgbm as lgb

        return lgb.Booster(model_file=path.as_posix())
    if artifact.algorithm == "xgb":
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(path)
        return model
    if artifact.algorithm in {"lr", "scorecard"}:
        return joblib.load(path)
    raise ModelingError(f"unsupported model algorithm: {artifact.algorithm}")


def export_pmml(
    artifact: ModelArtifact,
    dataset_path: Path,
    out_path: Path,
    *,
    base_dir: Path,
) -> Path:
    del artifact, dataset_path, out_path, base_dir
    raise ModelingError("PMML export is not available until sklearn2pmml integration is installed")


__all__ = ["export_pmml", "load_model", "save_model"]
