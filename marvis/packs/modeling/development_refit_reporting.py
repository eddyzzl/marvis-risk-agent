"""Expose final-fit diagnostics separately from the parent's held-out evidence."""

from dataclasses import asdict

from marvis.packs.modeling.development_refit_evidence import (
    receipt_id,
    validate_development_refit,
)
from marvis.packs.modeling.errors import ModelingError


def load_report_scope(runtime, experiment, artifact, dataset_id):
    if artifact is None:
        return None
    proof = validate_development_refit(
        runtime.settings.db_path,
        experiment,
        artifact,
        runtime.settings.tasks_dir / experiment.task_id / "modeling_artifacts",
    )
    if proof is None:
        return None
    if dataset_id != proof["source"]["dataset_id"]:
        raise ModelingError("development_refit_report_source_mismatch")
    parent = runtime.experiments.get(proof["parent"]["experiment_id"])
    return {
        "proof": proof,
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "parent_metrics": asdict(parent.metrics),
        "metadata": {
            "final_fit_scope": "完整 development train+test；原 test 已参与拟合",
            "development_refit_receipt_id": receipt_id(artifact),
            "final_experiment_id": experiment.id,
            "final_artifact_id": artifact.id,
            "parent_selection_experiment_id": parent.id,
            "parent_selection_artifact_id": parent.artifact_id,
            "final_independent_test": "未评估；原 test 已用于最终 development 拟合",
            "final_oot_evaluation": "重训未执行 OOT 评估；报告内 OOT 分布仅为诊断",
            "historical_availability": "unknown",
            "label_maturity": "unknown",
            "monitoring_baseline_scope": "完整 development 拟合行；独立测试及 OOT 未用于建立基准",
            "monitoring_baseline_sha256": proof["baseline_sha256"],
        },
    }


def dataset_split_rows(scope, metrics):
    proof = scope["proof"]
    rows = [
        {
            "split": "development_train_plus_test",
            "sample_count": proof["memberships"]["development_fit"]["rows"],
            "ks": metrics.train_ks,
            "auc": metrics.train_auc,
            "experiment_id": scope["experiment_id"],
            "artifact_id": scope["artifact_id"],
            "purpose": "fit_diagnostic_not_independent_validation",
        }
    ]
    for role in ("train", "test"):
        rows.append(
            {
                "split": f"pre_refit_{role}",
                "sample_count": proof["memberships"][f"original_{role}"]["rows"],
                "ks": scope["parent_metrics"][f"{role}_ks"],
                "auc": scope["parent_metrics"][f"{role}_auc"],
                "experiment_id": proof["parent"]["experiment_id"],
                "artifact_id": proof["parent"]["artifact_id"],
                "purpose": "parent_heldout_selection_before_refit"
                if role == "test"
                else "parent_fit_diagnostic",
            }
        )
    rows.append(
        {
            "split": "final_independent_test",
            "sample_count": 0,
            "ks": None,
            "auc": None,
            "experiment_id": scope["experiment_id"],
            "artifact_id": scope["artifact_id"],
            "purpose": "unavailable_original_test_consumed_in_fit",
        }
    )
    return rows


def diagnostic_rows(rows, scope):
    return [
        {
            **row,
            "model_experiment_id": scope["experiment_id"],
            "model_artifact_id": scope["artifact_id"],
            "evaluation_purpose": "consumed_in_fit_diagnostic"
            if row.get("split") in ("train", "test")
            else "diagnostic_not_temporal_or_maturity_validation",
            "historical_availability": "unknown",
            "label_maturity": "unknown",
        }
        for row in rows
    ]


def model_narrative():
    return (
        "最终模型按冻结参数拟合完整 development train+test。其训练指标为拟合诊断，"
        "原 test 已参与最终拟合，不能作为最终模型的独立测试或过拟合通过证明。"
        "重训前候选的真实留出 test 选择指标以原实验和产物引用单独列示。"
        "分段、稳定性及 OOT 分布为报告诊断；历史可用性和标签成熟度未知。"
    )
