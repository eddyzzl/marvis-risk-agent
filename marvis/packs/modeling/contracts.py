from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelRecipe:
    id: str
    algorithm: str
    default_params: dict[str, Any]
    param_space: dict[str, Any]
    requires_woe: bool


@dataclass(frozen=True)
class TrainConfig:
    dataset_id: str
    features: tuple[str, ...]
    target_col: str
    split_col: str
    split_values: dict[str, Any]
    params: dict[str, Any]
    seed: int
    early_stopping_rounds: int | None
    recipe_id: str | None = None
    scenario_id: str | None = None
    target_type: str = "binary"
    eval_metric: str = "ks_auc"


@dataclass(frozen=True)
class ModelMetrics:
    train_ks: float | None
    test_ks: float | None
    oot_ks: float | None
    train_auc: float | None
    test_auc: float | None
    oot_auc: float | None
    psi_test_vs_train: float | None
    psi_oot_vs_train: float | None
    overfit_train_test_gap: float
    overfit_train_oot_gap: float | None
    overfit_flag: bool
    train_rmse: float | None = None
    test_rmse: float | None = None
    oot_rmse: float | None = None
    train_mae: float | None = None
    test_mae: float | None = None
    oot_mae: float | None = None
    train_r2: float | None = None
    test_r2: float | None = None
    oot_r2: float | None = None


@dataclass(frozen=True)
class ModelArtifact:
    id: str
    experiment_id: str
    algorithm: str
    model_path: str
    pmml_path: str | None
    feature_list: tuple[str, ...]
    params: dict[str, Any]
    woe_maps: dict[str, Any] | None
    created_at: str


@dataclass(frozen=True)
class TrainResult:
    artifact: ModelArtifact
    metrics: ModelMetrics
    feature_importance: tuple[tuple[str, float], ...]
    experiment_id: str


@dataclass(frozen=True)
class Experiment:
    id: str
    task_id: str
    recipe_id: str
    config: TrainConfig
    metrics: ModelMetrics | None
    artifact_id: str | None
    status: str
    created_at: str


__all__ = [
    "Experiment",
    "ModelArtifact",
    "ModelMetrics",
    "ModelRecipe",
    "TrainConfig",
    "TrainResult",
]
