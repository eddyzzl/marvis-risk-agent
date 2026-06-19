from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from marvis.db import ModelingRepository
from marvis.packs.modeling.contracts import Experiment, TrainConfig, TrainResult


class ExperimentStore:
    def __init__(self, db_path: Path):
        self._repo = ModelingRepository(Path(db_path))

    def create(self, task_id: str, recipe_id: str, config: TrainConfig) -> str:
        experiment_id = f"experiment_{uuid.uuid4().hex}"
        self._repo.create_experiment(
            Experiment(
                id=experiment_id,
                task_id=task_id,
                recipe_id=recipe_id,
                config=config,
                metrics=None,
                artifact_id=None,
                status="created",
                created_at=datetime.now(UTC).isoformat(),
            )
        )
        return experiment_id

    def attach_result(self, experiment_id: str, result: TrainResult) -> None:
        self.get(experiment_id)
        artifact = replace(result.artifact, experiment_id=experiment_id)
        self._repo.create_model_artifact(artifact)
        self._repo.attach_experiment_result(
            experiment_id,
            metrics=result.metrics,
            artifact_id=artifact.id,
            status="trained",
        )

    def set_status(self, experiment_id: str, status: str) -> None:
        self._repo.set_experiment_status(experiment_id, status)

    def get(self, experiment_id: str) -> Experiment:
        experiment = self._repo.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        return experiment

    def list_for_task(self, task_id: str) -> list[Experiment]:
        return self._repo.list_experiments(task_id)

    def compare(self, experiment_ids: list[str]) -> dict:
        return {
            "experiments": [
                _comparison_row(self.get(experiment_id))
                for experiment_id in experiment_ids
            ]
        }


def _comparison_row(experiment: Experiment) -> dict:
    metrics = experiment.metrics
    return {
        "id": experiment.id,
        "recipe": experiment.recipe_id,
        "status": experiment.status,
        "artifact_id": experiment.artifact_id,
        "train_ks": None if metrics is None else metrics.train_ks,
        "test_ks": None if metrics is None else metrics.test_ks,
        "oot_ks": None if metrics is None else metrics.oot_ks,
        "train_auc": None if metrics is None else metrics.train_auc,
        "test_auc": None if metrics is None else metrics.test_auc,
        "oot_auc": None if metrics is None else metrics.oot_auc,
        "psi_test_vs_train": None if metrics is None else metrics.psi_test_vs_train,
        "psi_oot_vs_train": None if metrics is None else metrics.psi_oot_vs_train,
        "overfit_flag": None if metrics is None else metrics.overfit_flag,
    }


__all__ = ["ExperimentStore"]
