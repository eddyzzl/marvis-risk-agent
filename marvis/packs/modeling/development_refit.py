"""Binary LR final fitting on an authenticated complete development population.

No search, feature selection or held-out evaluation occurs in this producer.
The selected parent and its test metrics retain their original identities.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock
import numpy as np
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.authenticated_snapshot import authenticated_file_snapshot
from marvis.data.fitted_input_time import validate_fitted_input_time
from marvis.data.labels import resolve_labeled_frame
from marvis.data.preprocessing_evidence import (
    training_preprocessing_state_for_membership,
    verify_source_on_connection,
)
from marvis.feature.metrics import (
    feature_auc,
    feature_ks,
    weighted_feature_auc,
    weighted_feature_ks,
)
from marvis.packs.modeling.contracts import ModelMetrics, TrainResult
from marvis.packs.modeling.development_refit_baseline import development_baseline
from marvis.packs.modeling.development_refit_evidence import (
    KIND,
    ORIGIN,
    digest,
    members,
    parent_binding,
    receipt_id,
    validate_development_refit,
)
from marvis.packs.modeling.development_refit_recovery import (
    prepare_journal,
    reconcile_journal,
)
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.fold_selection import _ProjectedBackend
from marvis.packs.modeling.fold_tuning_evidence import validate_fold_experiment
from marvis.packs.modeling.producer_receipts import (
    capture_native_writes,
    finish_capture,
    load_receipt,
    metadata_hash,
)
from marvis.packs.modeling.recipes.common import (
    artifact_params,
    model_params,
    sample_weight_col,
    sample_weight_values,
)
from marvis.packs.modeling.recipes.lr import _lr_importance, _save_lr_model
from marvis.repositories.task_artifacts import stable_task_artifact_id

MAX_ROWS = 1_000_000
MAX_CELLS = 50_000_000


def _assert_not_frozen(conn, task_id):
    if conn.execute(
        "SELECT 1 FROM blind_oot_freezes f JOIN blind_oot_cohorts c ON c.id=f.cohort_id WHERE c.task_id=?",
        (task_id,),
    ).fetchone():
        raise ModelingError("development_refit_after_final_freeze_forbidden")


def _result(runtime, experiment, artifact, payload, pre_refit_metrics):
    # The producer's authenticated parent establishes selection/test evidence;
    # an internal caller's summary cannot replace the recorded metric values.
    parent = runtime.experiments.get(payload["parent"]["experiment_id"])
    parent_metrics = asdict(parent.metrics)
    return {
        "applied": True,
        "requested": True,
        "supported_profile": "binary_lr_native_development_refit.v1",
        "reason": "已按冻结参数拟合完整 development train+test；父候选的真实 test/选择指标保留，新模型无独立 test 或 OOT 评估。",
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "metrics": asdict(experiment.metrics),
        "metrics_before_refit": parent_metrics,
        "metrics_after_refit": asdict(experiment.metrics),
        "refit_holdout_metrics": {},
        "development_refit_receipt_id": receipt_id(artifact),
        "parent_experiment_id": payload["parent"]["experiment_id"],
        "fit_rows": payload["memberships"]["development_fit"]["rows"],
        "test_evaluation": "unavailable_consumed_as_development",
        "oot_evaluation": "not_performed",
    }


def development_refit(runtime, *, experiment, recipe, pre_refit_metrics):
    if recipe != "lr" or experiment.config.target_type != "binary":
        artifact = runtime.modeling_repo.get_model_artifact(experiment.artifact_id)
        if artifact is None:
            raise ModelingError("development_refit_parent_artifact_missing")
        validate_fold_experiment(
            runtime.registry, experiment.task_id, experiment, artifact
        )
        return {
            "applied": False,
            "requested": True,
            "supported_profile": "binary_lr_native_development_refit.v1",
            "reason": "当前原生 development refit 仅支持 binary LR；已保留原 fold-frozen 候选。",
        }
    directory = (
        Path(runtime.settings.tasks_dir) / experiment.task_id / "modeling_artifacts"
    )
    directory.mkdir(parents=True, exist_ok=True)
    if (
        directory.resolve(strict=True) != directory
        or runtime.registry.datasets_root
        != Path(runtime.settings.db_path).parent / "datasets"
    ):
        raise ModelingError("development_refit_workspace_profile_unsupported")
    lock = directory / ".development-refit.lock"
    if lock.is_symlink():
        raise ModelingError("development_refit_lock_invalid")
    with FileLock(str(lock), timeout=30, mode=0o600):
        parent = runtime.experiments.get(experiment.id)
        parent_artifact = runtime.modeling_repo.get_model_artifact(parent.artifact_id)
        if (
            parent_artifact is None
            or parent.status != "selected"
            or receipt_id(parent_artifact) is not None
        ):
            raise ModelingError("development_refit_selected_parent_required")
        native, _ = load_receipt(
            runtime.settings.db_path, parent, parent_artifact, directory
        )
        fold = validate_fold_experiment(
            runtime.registry, parent.task_id, parent, parent_artifact
        )
        if fold is None:
            raise ModelingError("development_refit_native_fold_required")
        parent_proof = parent_binding(parent, parent_artifact, native)
        identity = digest({"producer": ORIGIN, "parent": parent_proof})
        name = f"development_refit_{identity}.json"
        relative = f"tasks/{parent.task_id}/modeling_artifacts/{name}"
        ref_id = stable_task_artifact_id(
            task_id=parent.task_id, kind=KIND, path=relative
        )
        reconcile_journal(runtime, directory, identity, ref_id, parent.task_id)
        old = runtime.task_artifacts.get_for_task(parent.task_id, ref_id)
        if old:
            child = runtime.experiments.get(old["provenance"]["experiment_id"])
            artifact = runtime.modeling_repo.get_model_artifact(child.artifact_id)
            load_receipt(runtime.settings.db_path, child, artifact, directory)
            payload = validate_development_refit(
                runtime.settings.db_path, child, artifact, directory
            )
            return _result(runtime, child, artifact, payload, pre_refit_metrics)
        with runtime.experiments.transaction() as conn:
            _assert_not_frozen(conn, parent.task_id)
        source = runtime.registry.get(parent.config.dataset_id)
        if (
            source.task_id != parent.task_id
            or source.row_count > MAX_ROWS
            or source.row_count * len(parent.config.features) > MAX_CELLS
        ):
            raise ModelingError("development_refit_source_or_cell_budget_invalid")
        path = runtime.registry.resolve_verified_path(source.id)
        original = parent.config
        weight = sample_weight_col(original)
        columns = list(
            dict.fromkeys(
                [*original.features, original.target_col, *([weight] if weight else [])]
            )
        )
        with authenticated_file_snapshot(
            path,
            root=runtime.registry.datasets_root,
            expected_sha256=source.content_hash,
        ) as snapshot:
            parquet = pq.ParquetFile(snapshot)
            controls = _ProjectedBackend(
                parquet, np.arange(source.row_count), [original.split_col], {}
            ).read_frame(None)
            train = (
                controls[original.split_col]
                .eq(original.split_values["train"])
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            test = (
                controls[original.split_col]
                .eq(original.split_values["test"])
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            if not train.any() or not test.any() or (train & test).any():
                raise ModelingError("development_refit_disjoint_train_test_required")
            positions = np.flatnonzero(train | test)
            frame = _ProjectedBackend(parquet, positions, columns, {}).read_frame(None)
            frame, dropped = resolve_labeled_frame(
                frame,
                target_col=original.target_col,
                drop_nan_labels=original.drop_nan_labels,
            )
        actual = np.zeros(source.row_count, dtype=bool)
        actual[frame.index] = True
        # The union is the full label-eligible development population. Excluded
        # null labels remain explicit; they never become an invented holdout.
        train = train & actual
        test = test & actual
        if not train.any() or not test.any():
            raise ModelingError("development_refit_labeled_partitions_required")
        state = training_preprocessing_state_for_membership(
            runtime.registry, source.id, train_mask=actual
        )
        temporal = validate_fitted_input_time(
            runtime.registry,
            source.id,
            original.features,
            fit_mask=actual,
            evaluation_mask=np.zeros(source.row_count, dtype=bool),
            requires_labels=True,
            evaluation_roles=[],
            excluded_evaluation_roles=[
                "original_test_consumed_as_development",
                "oot_not_used_for_fit_or_selection",
            ],
        )
        temporal["development_refit_receipt_id"] = ref_id
        parameters = dict(original.params)
        for key in (
            "fold_tuning_evidence",
            "fold_tuning_evidence_ref",
            "fold_selection_plan",
        ):
            parameters.pop(key, None)
        parameters.update(
            refit_on_train_plus_test=True, fitted_input_time_evidence=temporal
        )
        config = replace(original, params=parameters, early_stopping_rounds=None)
        learner = model_params(parent_artifact.params)
        if parent_artifact.params.get("calibration"):
            raise ModelingError("development_refit_calibrated_parent_unsupported")
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(**learner)),
            ]
        )
        model.fit(
            frame[list(config.features)],
            frame[config.target_col].to_numpy(dtype=int),
            lr__sample_weight=sample_weight_values(frame, config),
        )
        scores = model.predict_proba(frame[list(config.features)])[:, 1]
        labels = frame[config.target_col].to_numpy(dtype=float)
        weights = sample_weight_values(frame, config)
        metrics = ModelMetrics(
            train_ks=feature_ks(scores, labels),
            train_auc=feature_auc(scores, labels),
            test_ks=None,
            test_auc=None,
            oot_ks=None,
            oot_auc=None,
            psi_test_vs_train=None,
            psi_oot_vs_train=None,
            overfit_train_test_gap=None,
            overfit_train_oot_gap=None,
            overfit_flag=None,
            weighted_train_ks=None
            if weights is None
            else weighted_feature_ks(scores, labels, weights),
            weighted_train_auc=None
            if weights is None
            else weighted_feature_auc(scores, labels, weights),
        )
        child = runtime.experiments.prepare(parent.task_id, recipe, config)
        uow = ArtifactUnitOfWork()
        staged_files = []
        journal = None
        try:
            with TemporaryDirectory(
                prefix=".development-refit-", dir=directory
            ) as scratch:
                with capture_native_writes() as writes:
                    artifact = _save_lr_model(
                        model, config, Path(scratch), artifact_params(learner, config)
                    )
                    artifact = replace(artifact, baseline_distributions=development_baseline(
                        frame, scores, config.features, source=source, membership=members(actual)))
                    from marvis.packs.modeling.artifact import persist_model_meta
                    persist_model_meta(Path(scratch), artifact, config=config)
                capture = finish_capture(writes, artifact, scratch)
                capture["directory"] = str(directory)
                for filename in (
                    artifact.model_path,
                    f"{artifact.id}.model_meta.json",
                    "model_meta.json",
                ):
                    staged = uow.stage_file(directory, filename)
                    staged.path.write_bytes((Path(scratch) / filename).read_bytes())
                    staged_files.append(staged)
            payload = {
                "schema_version": ORIGIN,
                "parent": parent_proof,
                "parent_fold_ref": original.params["fold_tuning_evidence_ref"],
                "experiment_id": child.id,
                "artifact_id": artifact.id,
                "model_metadata_sha256": metadata_hash(artifact),
                "baseline_sha256": digest(artifact.baseline_distributions),
                "baseline_scope": "development_fit",
                "config_sha256": digest(asdict(config)),
                "metrics_sha256": digest(asdict(metrics)),
                "features": list(config.features),
                "learner_params": learner,
                "source": {
                    "dataset_id": source.id,
                    "content_hash": source.content_hash,
                    "row_count": source.row_count,
                },
                "memberships": {
                    "original_train": members(train),
                    "original_test": members(test),
                    "development_fit": members(actual),
                    "excluded": members(~actual),
                },
                "preprocessing": {
                    "artifact_id": state.artifact_id,
                    "content_hash": state.content_hash,
                    "steps": state.steps,
                    "assurance": state.assurance,
                },
                "nan_labels_dropped": dropped,
                "independent_test_evaluation": "unavailable",
                "oot_evaluation": "not_performed",
                "historical_availability": "unknown",
            }
            raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            staged = uow.stage_file(directory, name)
            staged.path.write_bytes(raw)
            staged_files.append(staged)
            result = TrainResult(
                artifact=artifact,
                metrics=metrics,
                feature_importance=_lr_importance(model, config.features),
                experiment_id=child.id,
                nan_labels_dropped=dropped,
                producer_receipt=capture,
            )

            def commit(conn):
                conn.execute("BEGIN IMMEDIATE")
                _assert_not_frozen(conn, parent.task_id)
                fresh = runtime.experiments.get(parent.id)
                fresh_artifact = runtime.modeling_repo.get_model_artifact(
                    parent.artifact_id
                )
                fresh_native, _ = load_receipt(
                    runtime.settings.db_path, fresh, fresh_artifact, directory
                )
                if (
                    fresh.status != "selected"
                    or parent_binding(fresh, fresh_artifact, fresh_native)
                    != parent_proof
                ):
                    raise ModelingError(
                        "development_refit_parent_changed_before_publication"
                    )
                verify_source_on_connection(runtime.registry, conn, source)
                validate_fold_experiment(
                    runtime.registry, parent.task_id, fresh, fresh_artifact
                )
                runtime.experiments.create_on_connection(conn, child)
                attached = runtime.experiments.attach_result_on_connection(
                    conn, child.id, result
                )
                record = runtime.task_artifacts.register_on_connection(
                    conn,
                    task_id=parent.task_id,
                    kind=KIND,
                    path=relative,
                    content_hash=hashlib.sha256(raw).hexdigest(),
                    origin_tool=ORIGIN,
                    provenance={
                        "schema_version": ORIGIN,
                        "experiment_id": child.id,
                        "artifact_id": attached.id,
                        "parent_experiment_id": parent.id,
                    },
                )
                if record["id"] != ref_id:
                    raise ModelingError("development_refit_receipt_identity_changed")
                return attached

            journal = prepare_journal(
                directory, identity, ref_id, artifact, staged_files
            )
            artifact = uow.finalize_with_connection(
                runtime.experiments.transaction, commit
            )
            journal.unlink()
        except Exception:
            uow.rollback()
            if journal is not None:
                journal.unlink(missing_ok=True)
            raise
        child = runtime.experiments.get(child.id)
        load_receipt(runtime.settings.db_path, child, artifact, directory)
        return _result(runtime, child, artifact, payload, pre_refit_metrics)
