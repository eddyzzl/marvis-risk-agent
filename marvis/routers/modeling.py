from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

from marvis.api_task_helpers import get_task_or_404
from marvis.db import ModelingRepository, TaskRepository


router = APIRouter(prefix="/api", tags=["modeling"])

_LIST_MAX_LIMIT = 500


def _modeling_repo(request: Request) -> ModelingRepository:
    return ModelingRepository(request.app.state.settings.db_path)


def _task_repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _experiment_summary_payload(row: dict) -> dict:
    experiment = row["experiment"]
    metrics = experiment.metrics
    return {
        "id": experiment.id,
        "task_id": experiment.task_id,
        "task_model_name": row["task_model_name"],
        "task_model_version": row["task_model_version"],
        "task_type": row["task_type"],
        "task_algorithm": row["task_algorithm"],
        "recipe_id": experiment.recipe_id,
        "status": experiment.status,
        "created_at": experiment.created_at,
        "artifact_id": experiment.artifact_id,
        "train_ks": None if metrics is None else metrics.train_ks,
        "test_ks": None if metrics is None else metrics.test_ks,
        "oot_ks": None if metrics is None else metrics.oot_ks,
        "train_auc": None if metrics is None else metrics.train_auc,
        "test_auc": None if metrics is None else metrics.test_auc,
        "oot_auc": None if metrics is None else metrics.oot_auc,
    }


def _experiment_detail_payload(experiment, artifacts: list) -> dict:
    metrics = experiment.metrics
    return {
        "id": experiment.id,
        "task_id": experiment.task_id,
        "recipe_id": experiment.recipe_id,
        "status": experiment.status,
        "created_at": experiment.created_at,
        "artifact_id": experiment.artifact_id,
        "config": {
            "dataset_id": experiment.config.dataset_id,
            "features": list(experiment.config.features),
            "target_col": experiment.config.target_col,
            "split_col": experiment.config.split_col,
            "recipe_id": experiment.config.recipe_id,
            "scenario_id": experiment.config.scenario_id,
            "target_type": experiment.config.target_type,
            "eval_metric": experiment.config.eval_metric,
        },
        "metrics": None if metrics is None else _metrics_payload(metrics),
        "artifacts": [_artifact_summary_payload(artifact) for artifact in artifacts],
    }


def _metrics_payload(metrics) -> dict:
    return {
        "train_ks": metrics.train_ks,
        "test_ks": metrics.test_ks,
        "oot_ks": metrics.oot_ks,
        "train_auc": metrics.train_auc,
        "test_auc": metrics.test_auc,
        "oot_auc": metrics.oot_auc,
        "psi_test_vs_train": metrics.psi_test_vs_train,
        "psi_oot_vs_train": metrics.psi_oot_vs_train,
        "overfit_train_test_gap": metrics.overfit_train_test_gap,
        "overfit_train_oot_gap": metrics.overfit_train_oot_gap,
        "overfit_flag": metrics.overfit_flag,
    }


def _artifact_summary_payload(artifact) -> dict:
    return {
        "id": artifact.id,
        "experiment_id": artifact.experiment_id,
        "algorithm": artifact.algorithm,
        "model_path": artifact.model_path,
        "pmml_path": artifact.pmml_path,
        "feature_list": list(artifact.feature_list),
        "created_at": artifact.created_at,
        "score_direction": artifact.score_direction,
        "points_direction": artifact.points_direction,
    }


@router.get("/experiments")
def list_experiments_all(
    request: Request,
    response: Response,
    status: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """Cross-task read-only model registry (GAP-6): "which models have I
    delivered" across every task, without paging through the task list."""
    repo = _modeling_repo(request)
    bounded_limit = None if limit is None else max(1, min(int(limit), _LIST_MAX_LIMIT))
    bounded_offset = max(0, int(offset))
    query_limit = bounded_limit + 1 if bounded_limit is not None else None
    rows = repo.list_experiments_all(status=status, limit=query_limit, offset=bounded_offset)
    has_more = False
    if bounded_limit is not None and len(rows) > bounded_limit:
        has_more = True
        rows = rows[:bounded_limit]
    if bounded_limit is not None or bounded_offset:
        response.headers["X-Result-Limit"] = "" if bounded_limit is None else str(bounded_limit)
        response.headers["X-Result-Offset"] = str(bounded_offset)
        response.headers["X-Result-Has-More"] = "true" if has_more else "false"
    return [_experiment_summary_payload(row) for row in rows]


@router.get("/tasks/{task_id}/experiments")
def list_task_experiments(
    task_id: str,
    request: Request,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    """LT-13: limit/offset are optional so existing callers that omit them
    keep getting the full per-task experiment history (bounded by
    _LIST_MAX_LIMIT when a caller does opt in)."""
    get_task_or_404(_task_repo(request), task_id)
    repo = _modeling_repo(request)
    bounded_limit = None if limit is None else max(1, min(int(limit), _LIST_MAX_LIMIT))
    bounded_offset = max(0, int(offset))
    experiments = repo.list_experiments(task_id, limit=bounded_limit, offset=bounded_offset)
    total = repo.count_experiments(task_id)
    payload = {
        "experiments": [
            _experiment_detail_payload(
                experiment,
                repo.list_model_artifacts(experiment_id=experiment.id),
            )
            for experiment in experiments
        ]
    }
    if bounded_limit is not None or bounded_offset:
        payload["total"] = total
        payload["limit"] = bounded_limit
        payload["offset"] = bounded_offset
        payload["has_more"] = bounded_offset + len(experiments) < total
    return payload


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, request: Request) -> dict:
    repo = _modeling_repo(request)
    experiment = repo.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    artifacts = repo.list_model_artifacts(experiment_id=experiment_id)
    return _experiment_detail_payload(experiment, artifacts)
