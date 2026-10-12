"""Independent original-source predictions, baseline drift and declared thresholds."""
from pathlib import Path
from dataclasses import asdict
import json
import tempfile

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from .runtime_archive_domain import FrozenDomainBinding, DomainArchive, HASH, ID, equal, numbers, one_step
from .runtime_archive_datasets import _dataset, _source_input
from .runtime_archive_modeling import FrozenModelingBinding, authenticated_model, revalidate_modeling_archive, native_hash
from .runtime_archive_reader import _Unsupported
from .runtime_monitoring_reference import monitoring_reference, validate_thresholds

DEFAULT_DECLARED_THRESHOLDS = {
    "score_psi": {"label": "Score PSI vs baseline", "metric": "score_psi", "direction": "max", "warn": .10, "fail": .25},
    "feature_csi_max": {"label": "Max feature CSI vs baseline", "metric": "feature_csi_max", "direction": "max", "warn": .10, "fail": .25},
    "ks_drop": {"label": "KS drop vs development", "metric": "ks_drop", "direction": "max", "warn": .05, "fail": .10},
    "auc_drop": {"label": "AUC drop vs development", "metric": "auc_drop", "direction": "max", "warn": .03, "fail": .05},
}


class FrozenMonitoringBinding(FrozenDomainBinding):
    dataset_id: str = Field(pattern=ID)
    dataset_sha256: str = Field(pattern=HASH)
    scored_dataset_id: str = Field(pattern=ID)
    scored_dataset_sha256: str = Field(pattern=HASH)
    experiment_id: str = Field(pattern=ID)
    artifact_id: str = Field(pattern=ID)
    monitoring_binding_sha256: str | None = Field(pattern=HASH)
    thresholds: dict[str, dict]
    modeling: FrozenModelingBinding

    @model_validator(mode="after")
    def monitoring_contract(self):
        if self.case.family != "monitoring" or not any(a.monitoring_request is not None for a in self.case.actions):
            raise ValueError("declared native monitoring case required")
        validate_thresholds(self.thresholds)
        if self.dataset_id == self.scored_dataset_id:
            raise ValueError("monitoring must retain separate source and scored datasets")
        for key in ("case", "run_id", "task_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"):
            if getattr(self, key) != getattr(self.modeling, key):
                raise ValueError("monitoring model evidence belongs to another frozen case/run/task/source")
        if self.modeling.selected_experiment_id != self.experiment_id or self.modeling.final_artifact_id != self.artifact_id:
            raise ValueError("monitoring requires the exact independently frozen final model")
        return self


def revalidate_monitoring_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenMonitoringBinding):
    if not isinstance(frozen_binding, FrozenMonitoringBinding):
        raise ValueError("explicit FrozenMonitoringBinding required")
    binding = FrozenMonitoringBinding.model_validate(frozen_binding.model_dump())
    model_result = revalidate_modeling_archive(archive, expected_manifest_sha256=expected_manifest_sha256,
        frozen_binding=binding.modeling)
    with DomainArchive(archive, expected_manifest_sha256, binding, family="monitoring",
                       checks=("model_fit_and_baseline", "registered_datasets", "source_material", "model_and_threshold_binding", "independent_predictions", "independent_monitoring")) as session:
        session.open()
        session.current = "model_fit_and_baseline"
        if not model_result.get("supported_checks_verified") or model_result.get("values", {}).get("scope", {}).get("baseline_verification") != "independently_recomputed":
            raise _Unsupported("independent final model fit, selection and baseline required")
        session.mark("model_fit_and_baseline", model_checks=model_result["checks"], fit_scope=model_result["values"]["scope"])
        session.current = "registered_datasets"
        source, frame = _dataset(session.conn, session.archive, binding.task_id, binding.dataset_id, binding.dataset_sha256)
        child, scored = _dataset(session.conn, session.archive, binding.task_id, binding.scored_dataset_id, binding.scored_dataset_sha256)
        session.mark("registered_datasets", rows=len(frame))
        session.current = "source_material"
        request = next(a.monitoring_request for a in binding.case.actions if a.monitoring_request is not None)
        material_index = next(i for i, m in enumerate(binding.case.materials) if m.path == request.material_path)
        scratch = Path(session.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        _source_input(session.archive, session.manifest, binding, source, frame, scratch, material_index=material_index)
        session.mark("source_material")
        session.current = "model_and_threshold_binding"
        experiment, artifact, metadata, config, metrics, receipt, _, predict = authenticated_model(session, binding.experiment_id, binding.artifact_id)
        if config["target_type"] != "binary":
            raise _Unsupported("nonbinary monitoring requires independently declared performance checks")
        score_step, monitor = one_step(session.steps, "modeling.score_dataset"), one_step(session.steps, "modeling.monitor_run")
        if (score_step["input"]["dataset_id"] != binding.dataset_id or score_step["input"]["experiment_id"] != binding.experiment_id
                or score_step["output"]["result_dataset_id"] != binding.scored_dataset_id
                or monitor["input"].get("scored_dataset_id") != binding.scored_dataset_id
                or monitor["input"]["experiment_id"] != binding.experiment_id):
            raise ValueError("monitoring source/model/result selection differs")
        frozen = score_step["input"].get("monitoring_binding")
        from .runtime_contracts import digest
        equal(None if frozen is None else digest(frozen), binding.monitoring_binding_sha256)
        equal(monitor["input"].get("monitoring_binding"), frozen)
        if frozen is not None:
            from marvis.repositories.modeling import _experiment_from_row, _model_artifact_from_row
            from marvis.repositories.datasets import _dataset_from_row
            # These installed codecs decode authenticated rows only. Prediction,
            # fit and metric calculations use the separate reference algorithms.
            equal(frozen["experiment_sha256"], native_hash(asdict(_experiment_from_row(experiment))))
            equal(frozen["artifact_sha256"], native_hash(asdict(_model_artifact_from_row(artifact))))
            equal(frozen["baseline_sha256"], native_hash(json.loads(artifact["baseline_distributions_json"])))
            equal(frozen["producer_receipt_id"], receipt["id"])
            equal(frozen["files"], json.loads(receipt["provenance_json"])["files"])
            equal(frozen["dataset"], {"id": source["id"], "task_id": binding.task_id,
                "content_hash": source["content_hash"], "identity_sha256": native_hash(asdict(_dataset_from_row(source)))})
            if "workspace_binding" in frozen:
                workspace = session.conn.execute("SELECT * FROM data_workspaces WHERE task_id=?", (binding.task_id,)).fetchone()
                if workspace is None:
                    raise ValueError("monitoring workspace binding missing")
                equal(frozen["workspace_binding"], {key: workspace[key] for key in (
                    "revision", "analysis_generation", "active_dataset_id", "active_dataset_content_hash")})
        declared = {k: dict(v) for k, v in DEFAULT_DECLARED_THRESHOLDS.items()}
        overrides = (monitor["input"].get("monitoring_policy") or {}).get("thresholds", {})
        for key, fields in overrides.items():
            if key not in declared or not isinstance(fields, dict) or set(fields) - set(declared[key]):
                raise ValueError("unsupported monitoring threshold override")
            declared[key].update(fields)
        equal(declared, binding.thresholds)
        baseline = json.loads(artifact["baseline_distributions_json"]) if artifact["baseline_distributions_json"] else None
        if not baseline:
            raise _Unsupported("missing native model baseline cannot be invented")
        session.mark("model_and_threshold_binding")
        session.current = "independent_predictions"
        scores = np.asarray(predict(frame), dtype=float).tolist()
        score_col = score_step["output"]["score_col"]
        if len(scored) != len(frame) or score_col not in scored:
            raise ValueError("monitoring scored row universe differs")
        pd.testing.assert_frame_equal(scored[list(frame.columns)], frame, check_exact=True)
        numbers(scored[score_col].tolist(), scores)
        session.mark("independent_predictions", rows=len(scores))
        session.current = "independent_monitoring"
        values = monitoring_reference(frame, scores, features=metadata["feature_list"], target=request.target_col,
            baseline=baseline, development_metrics=metrics, thresholds=binding.thresholds)
        numbers(monitor["output"], {**values, "experiment_id": binding.experiment_id,
            "artifact_id": binding.artifact_id, "dataset_id": binding.scored_dataset_id,
            "score_direction": artifact["score_direction"]})
        session.mark("independent_monitoring", scope="fixed_original_source_and_baseline_threshold_checks")
        session.result["values"] = {**values, "label_maturity_assurance": "unknown", "business_acceptance": "not_established"}
        session.result["unsupported"]["schedule_delivery_and_window_time_assurance"] = "source_date_windows_and_external_delivery_not_established_by_drift_arithmetic"
    return session.result
