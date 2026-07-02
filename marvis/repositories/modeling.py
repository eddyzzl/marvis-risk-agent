import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from marvis.db_schema import connect
from marvis.packs.modeling.contracts import (
    Experiment,
    ModelArtifact,
    ModelMetrics,
    TrainConfig,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ModelingRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def transaction(self):
        return connect(self.db_path)

    def create_experiment(self, experiment: Experiment) -> None:
        with connect(self.db_path) as conn:
            _insert_experiment_row(conn, experiment)

    def create_experiment_with_audit(self, experiment: Experiment, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _insert_experiment_row(conn, experiment)
            _write_audit_row(conn, **audit)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, recipe_id, config_json, metrics_json,
                       artifact_id, status, created_at
                  FROM experiments
                 WHERE id = ?
                """,
                (experiment_id,),
            ).fetchone()
        return None if row is None else _experiment_from_row(row)

    def list_experiments(self, task_id: str) -> list[Experiment]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, recipe_id, config_json, metrics_json,
                       artifact_id, status, created_at
                  FROM experiments
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (task_id,),
            ).fetchall()
        return [_experiment_from_row(row) for row in rows]

    def attach_experiment_result(
        self,
        experiment_id: str,
        *,
        metrics: ModelMetrics,
        artifact_id: str,
        status: str = "trained",
    ) -> None:
        with connect(self.db_path) as conn:
            _attach_experiment_result_row(
                conn,
                experiment_id,
                metrics=metrics,
                artifact_id=artifact_id,
                status=status,
            )

    def attach_experiment_result_with_artifact_and_audit(
        self,
        experiment_id: str,
        *,
        artifact: ModelArtifact,
        metrics: ModelMetrics,
        status: str = "trained",
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _insert_model_artifact_row(conn, artifact)
            _attach_experiment_result_row(
                conn,
                experiment_id,
                metrics=metrics,
                artifact_id=artifact.id,
                status=status,
            )
            _write_audit_row(conn, **audit)

    def set_experiment_status(self, experiment_id: str, status: str) -> None:
        with connect(self.db_path) as conn:
            _set_experiment_status_row(conn, experiment_id, status)

    def set_experiment_status_with_audit(
        self,
        experiment_id: str,
        status: str,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_experiment_status_row(conn, experiment_id, status)
            _write_audit_row(conn, **audit)

    def create_model_artifact(self, artifact: ModelArtifact) -> None:
        with connect(self.db_path) as conn:
            _insert_model_artifact_row(conn, artifact)

    def set_model_artifact_pmml_path(self, artifact_id: str, pmml_path: str) -> None:
        with connect(self.db_path) as conn:
            _set_model_artifact_pmml_path_row(conn, artifact_id, pmml_path)

    def set_model_artifact_pmml_path_with_audit(
        self,
        artifact_id: str,
        pmml_path: str,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _set_model_artifact_pmml_path_row(conn, artifact_id, pmml_path)
            _write_audit_row(conn, **audit)

    def set_model_artifact_params(self, artifact_id: str, params: dict) -> None:
        with connect(self.db_path) as conn:
            _set_model_artifact_params_row(conn, artifact_id, params)

    def set_model_artifact_params_with_audit(
        self,
        artifact_id: str,
        params: dict,
        *,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            self.set_model_artifact_params_with_audit_on_connection(
                conn,
                artifact_id,
                params,
                audit=audit,
            )

    def set_model_artifact_params_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        artifact_id: str,
        params: dict,
        *,
        audit: dict,
    ) -> None:
        _set_model_artifact_params_row(conn, artifact_id, params)
        _write_audit_row(conn, **audit)

    def get_model_artifact(self, artifact_id: str) -> ModelArtifact | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, experiment_id, algorithm, model_path, pmml_path,
                       feature_list_json, feature_importance_json, params_json, woe_maps_json,
                       scorecard_table_json, created_at
                  FROM model_artifacts
                 WHERE id = ?
                """,
                (artifact_id,),
            ).fetchone()
        return None if row is None else _model_artifact_from_row(row)

    def list_model_artifacts(
        self,
        *,
        experiment_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ModelArtifact]:
        bounded_limit = None if limit is None else max(1, int(limit))
        bounded_offset = max(0, int(offset))
        params: list[object] = []
        query = """
            SELECT id, experiment_id, algorithm, model_path, pmml_path,
                   feature_list_json, feature_importance_json, params_json, woe_maps_json,
                   scorecard_table_json, created_at
              FROM model_artifacts
        """
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            params.append(experiment_id)
        query += " ORDER BY created_at, id"
        if bounded_limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([bounded_limit, bounded_offset])
        with connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [_model_artifact_from_row(row) for row in rows]


def _experiment_insert_values(experiment: Experiment) -> tuple:
    return (
        experiment.id,
        experiment.task_id,
        experiment.recipe_id,
        _dump_json_any(_train_config_to_dict(experiment.config)),
        None if experiment.metrics is None else _dump_json_any(_model_metrics_to_dict(experiment.metrics)),
        experiment.artifact_id,
        experiment.status,
        experiment.created_at,
    )


def _insert_experiment_row(conn: sqlite3.Connection, experiment: Experiment) -> None:
    conn.execute(
        """
        INSERT INTO experiments(
            id, task_id, recipe_id, config_json, metrics_json,
            artifact_id, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _experiment_insert_values(experiment),
    )


def _attach_experiment_result_row(
    conn: sqlite3.Connection,
    experiment_id: str,
    *,
    metrics: ModelMetrics,
    artifact_id: str,
    status: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE experiments
           SET metrics_json = ?,
               artifact_id = ?,
               status = ?
         WHERE id = ?
        """,
        (
            _dump_json_any(_model_metrics_to_dict(metrics)),
            artifact_id,
            status,
            experiment_id,
        ),
    )
    if cursor.rowcount == 0:
        raise KeyError(experiment_id)


def _set_experiment_status_row(
    conn: sqlite3.Connection,
    experiment_id: str,
    status: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE experiments
           SET status = ?
         WHERE id = ?
        """,
        (status, experiment_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(experiment_id)


def _experiment_from_row(row: sqlite3.Row) -> Experiment:
    metrics_json = row["metrics_json"]
    return Experiment(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        recipe_id=str(row["recipe_id"]),
        config=_train_config_from_dict(_load_json_object(row["config_json"])),
        metrics=None if metrics_json is None else _model_metrics_from_dict(_load_json_object(metrics_json)),
        artifact_id=_optional_str(row["artifact_id"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
    )


def _model_artifact_insert_values(artifact: ModelArtifact) -> tuple:
    return (
        artifact.id,
        artifact.experiment_id,
        artifact.algorithm,
        artifact.model_path,
        artifact.pmml_path,
        _dump_json_any(list(artifact.feature_list)),
        _dump_json_any([list(item) for item in artifact.feature_importance]),
        _dump_json_any(artifact.params),
        None if artifact.woe_maps is None else _dump_json_any(artifact.woe_maps),
        _dump_json_any(list(artifact.scorecard_table)),
        artifact.created_at,
    )


def _insert_model_artifact_row(conn: sqlite3.Connection, artifact: ModelArtifact) -> None:
    conn.execute(
        """
        INSERT INTO model_artifacts(
            id, experiment_id, algorithm, model_path, pmml_path,
            feature_list_json, feature_importance_json, params_json, woe_maps_json,
            scorecard_table_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _model_artifact_insert_values(artifact),
    )


def _set_model_artifact_pmml_path_row(
    conn: sqlite3.Connection,
    artifact_id: str,
    pmml_path: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE model_artifacts
           SET pmml_path = ?
         WHERE id = ?
        """,
        (pmml_path, artifact_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(artifact_id)


def _set_model_artifact_params_row(
    conn: sqlite3.Connection,
    artifact_id: str,
    params: dict,
) -> None:
    cursor = conn.execute(
        """
        UPDATE model_artifacts
           SET params_json = ?
         WHERE id = ?
        """,
        (_dump_json_any(params), artifact_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(artifact_id)


def _model_artifact_from_row(row: sqlite3.Row) -> ModelArtifact:
    woe_maps_json = row["woe_maps_json"]
    return ModelArtifact(
        id=str(row["id"]),
        experiment_id=str(row["experiment_id"]),
        algorithm=str(row["algorithm"]),
        model_path=str(row["model_path"]),
        pmml_path=_optional_str(row["pmml_path"]),
        feature_list=tuple(str(item) for item in _load_json_array(row["feature_list_json"])),
        feature_importance=_feature_importance_from_json(row["feature_importance_json"]),
        params=_load_json_object(row["params_json"]),
        woe_maps=None if woe_maps_json is None else _load_json_object(woe_maps_json),
        created_at=str(row["created_at"]),
        scorecard_table=tuple(
            dict(item)
            for item in _load_json_array(row["scorecard_table_json"])
            if isinstance(item, dict)
        ),
    )


def _feature_importance_from_json(raw: str | None) -> tuple[tuple[str, float], ...]:
    pairs = []
    for item in _load_json_array(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        feature, importance = item
        try:
            pairs.append((str(feature), float(importance)))
        except (TypeError, ValueError):
            continue
    return tuple(pairs)


def _train_config_to_dict(config: TrainConfig) -> dict:
    payload = asdict(config)
    payload["features"] = list(config.features)
    return payload


def _train_config_from_dict(payload: dict) -> TrainConfig:
    return TrainConfig(
        dataset_id=str(payload["dataset_id"]),
        features=tuple(str(item) for item in payload.get("features") or ()),
        target_col=str(payload["target_col"]),
        split_col=str(payload["split_col"]),
        split_values=dict(payload.get("split_values") or {}),
        params=dict(payload.get("params") or {}),
        seed=int(payload["seed"]),
        early_stopping_rounds=_optional_int(payload.get("early_stopping_rounds")),
        recipe_id=_optional_str(payload.get("recipe_id")),
        scenario_id=_optional_str(payload.get("scenario_id")),
        target_type=str(payload.get("target_type") or "binary"),
        eval_metric=str(payload.get("eval_metric") or "ks_auc"),
        drop_nan_labels=_bool_from_payload(payload.get("drop_nan_labels"), default=False),
    )


def _model_metrics_to_dict(metrics: ModelMetrics) -> dict:
    return asdict(metrics)


def _model_metrics_from_dict(payload: dict) -> ModelMetrics:
    return ModelMetrics(
        train_ks=_optional_float(payload.get("train_ks")),
        test_ks=_optional_float(payload.get("test_ks")),
        oot_ks=_optional_float(payload.get("oot_ks")),
        train_auc=_optional_float(payload.get("train_auc")),
        test_auc=_optional_float(payload.get("test_auc")),
        oot_auc=_optional_float(payload.get("oot_auc")),
        psi_test_vs_train=_optional_float(payload.get("psi_test_vs_train")),
        psi_oot_vs_train=_optional_float(payload.get("psi_oot_vs_train")),
        overfit_train_test_gap=float(payload.get("overfit_train_test_gap") or 0.0),
        overfit_train_oot_gap=_optional_float(payload.get("overfit_train_oot_gap")),
        overfit_flag=bool(payload.get("overfit_flag")),
        weighted_train_ks=_optional_float(payload.get("weighted_train_ks")),
        weighted_test_ks=_optional_float(payload.get("weighted_test_ks")),
        weighted_oot_ks=_optional_float(payload.get("weighted_oot_ks")),
        weighted_train_auc=_optional_float(payload.get("weighted_train_auc")),
        weighted_test_auc=_optional_float(payload.get("weighted_test_auc")),
        weighted_oot_auc=_optional_float(payload.get("weighted_oot_auc")),
        weighted_psi_test_vs_train=_optional_float(payload.get("weighted_psi_test_vs_train")),
        weighted_psi_oot_vs_train=_optional_float(payload.get("weighted_psi_oot_vs_train")),
        train_rmse=_optional_float(payload.get("train_rmse")),
        test_rmse=_optional_float(payload.get("test_rmse")),
        oot_rmse=_optional_float(payload.get("oot_rmse")),
        train_mae=_optional_float(payload.get("train_mae")),
        test_mae=_optional_float(payload.get("test_mae")),
        oot_mae=_optional_float(payload.get("oot_mae")),
        train_r2=_optional_float(payload.get("train_r2")),
        test_r2=_optional_float(payload.get("test_r2")),
        oot_r2=_optional_float(payload.get("oot_r2")),
        train_macro_auc=_optional_float(payload.get("train_macro_auc")),
        test_macro_auc=_optional_float(payload.get("test_macro_auc")),
        oot_macro_auc=_optional_float(payload.get("oot_macro_auc")),
        train_logloss=_optional_float(payload.get("train_logloss")),
        test_logloss=_optional_float(payload.get("test_logloss")),
        oot_logloss=_optional_float(payload.get("oot_logloss")),
        train_accuracy=_optional_float(payload.get("train_accuracy")),
        test_accuracy=_optional_float(payload.get("test_accuracy")),
        oot_accuracy=_optional_float(payload.get("oot_accuracy")),
        test_ks_ci_low=_optional_float(payload.get("test_ks_ci_low")),
        test_ks_ci_high=_optional_float(payload.get("test_ks_ci_high")),
        test_ks_ci_std=_optional_float(payload.get("test_ks_ci_std")),
        oot_ks_ci_low=_optional_float(payload.get("oot_ks_ci_low")),
        oot_ks_ci_high=_optional_float(payload.get("oot_ks_ci_high")),
        oot_ks_ci_std=_optional_float(payload.get("oot_ks_ci_std")),
        ks_ci_n_boot=_optional_int(payload.get("ks_ci_n_boot")),
    )


def _write_audit_row(
    conn: sqlite3.Connection,
    *,
    kind: str,
    target_ref: str,
    actor: str = "system",
    inputs_hash: str | None = None,
    outcome: str | None = None,
    detail: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit(
            id, kind, actor, target_ref, inputs_hash, outcome,
            detail_json, at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            kind,
            actor,
            target_ref,
            inputs_hash,
            outcome,
            json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
            _now(),
        ),
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _bool_from_payload(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dump_json_any(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    value = json.loads(raw)
    return value if isinstance(value, dict) else {}


def _load_json_array(raw: str | None) -> list:
    if not raw:
        return []
    value = json.loads(raw)
    return value if isinstance(value, list) else []
