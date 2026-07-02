from __future__ import annotations

import pandas as pd
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository
from marvis.feature.candidates import candidate_numeric_features
from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.training_dataset import TrainingDataset
from marvis.settings import build_settings
from pathlib import Path
from types import SimpleNamespace


MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"


class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.experiments = ExperimentStore(self.settings.db_path)
        self.modeling_repo = ModelingRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _resolve_feature_cols(
    runtime: _Runtime,
    dataset_id: str,
    features,
    *,
    target_col: str,
    split_col: str | None = None,
) -> list[str]:
    provided = _flatten_feature_cols(features)
    if provided:
        return provided
    dataset = runtime.registry.get(str(dataset_id))
    inferred = candidate_numeric_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=str(target_col),
        split_col=split_col,
    )
    if not inferred:
        raise ModelingError("未找到可用候选特征列;请检查拼接结果或指定特征列。")
    return inferred


def _flatten_feature_cols(features) -> list[str]:
    """Flatten a features input that may be a union of lists (FS-5): a workflow can pass
    ``features=[<base cols>, <$ref new_columns>]`` which resolves to nested lists; screen
    them together as one de-duplicated flat set (input order preserved)."""
    flat: list[str] = []
    seen: set[str] = set()
    for item in (features or []):
        candidates = item if isinstance(item, (list, tuple)) else [item]
        for candidate in candidates:
            name = str(candidate).strip()
            if name and name not in seen:
                seen.add(name)
                flat.append(name)
    return flat


def _artifact(runtime: _Runtime, artifact_id: str) -> ModelArtifact:
    artifact = runtime.modeling_repo.get_model_artifact(artifact_id)
    if artifact is None:
        raise ModelingError(f"model artifact not found: {artifact_id}")
    return artifact


def _artifact_base_dir(settings, task_id: str) -> Path:
    return Path(settings.tasks_dir) / task_id / MODELING_ARTIFACTS_DIR_NAME


def _cached_dataset_runtime(
    runtime: _Runtime,
    dataset_path: Path,
    *,
    frame: pd.DataFrame | None = None,
):
    dataset = (
        TrainingDataset(path=Path(dataset_path), frame=frame)
        if frame is not None
        else TrainingDataset.load(runtime.backend, dataset_path)
    )
    proxy = SimpleNamespace(**vars(runtime))
    proxy.backend = dataset.backend_adapter(runtime.backend)
    return proxy


def _artifact_model_base_dir(runtime: _Runtime, artifact: ModelArtifact) -> Path:
    experiment = runtime.experiments.get(artifact.experiment_id)
    return _artifact_base_dir(runtime.settings, experiment.task_id)
