"""Expose final-fit diagnostics separately from the parent's held-out evidence."""

from dataclasses import asdict

from marvis.packs.modeling.development_refit_evidence import (
    receipt_id,
    validate_development_refit,
)
from marvis.packs.modeling.errors import ModelingError


# Business-Chinese labels for the codes this producer writes. The rows and the
# report metadata keep the codes (tests, archives and the delivery card key off
# them); the Excel report shows these labels in their place.
_REPORT_KEY_LABELS = {
    "purpose": "用途",
    "evaluation_purpose": "评估用途",
    "model_experiment_id": "模型实验编号",
    "model_artifact_id": "模型产物编号",
    "final_fit_scope": "最终拟合范围",
    "development_refit_receipt_id": "重训凭证编号",
    "final_experiment_id": "最终模型实验编号",
    "final_artifact_id": "最终模型产物编号",
    "parent_selection_experiment_id": "所选实验编号（重训前）",
    "parent_selection_artifact_id": "所选实验产物编号（重训前）",
    "final_independent_test": "最终模型独立测试",
    "final_oot_evaluation": "最终模型 OOT 评估",
    "historical_availability": "历史数据可得性",
    "label_maturity": "标签成熟度",
    "monitoring_baseline_scope": "监控基准范围",
    "monitoring_baseline_sha256": "监控基准指纹（SHA-256）",
}
_UNKNOWN_LABELS = {"unknown": "未知"}
_REPORT_VALUE_LABELS = {
    "split": {
        "development_train_plus_test": "最终模型拟合集（train+test）",
        "pre_refit_train": "所选实验训练集（重训前）",
        "pre_refit_test": "所选实验测试集（重训前留出）",
        "pre_refit_oot": "所选实验 OOT（重训前留出）",
        "final_independent_test": "最终模型独立测试集",
    },
    "purpose": {
        "fit_diagnostic_not_independent_validation": "拟合诊断，不是独立验证",
        "parent_fit_diagnostic": "所选实验的拟合诊断",
        "parent_heldout_selection_before_refit": "重训前的留出集选择依据",
        "parent_heldout_oot_before_refit": "重训前的 OOT 留出证据，重训后未重新评估",
        "unavailable_original_test_consumed_in_fit": "不可用：原 test 已参与最终拟合",
    },
    "evaluation_purpose": {
        "consumed_in_fit_diagnostic": "已参与拟合，仅作诊断",
        "diagnostic_not_temporal_or_maturity_validation": "仅作诊断，不是时间外推或标签成熟度验证",
    },
    "historical_availability": _UNKNOWN_LABELS,
    "label_maturity": _UNKNOWN_LABELS,
}


def report_display_labels() -> dict:
    """``ModelReportPayload.display_labels`` for a development-refit report."""
    return {
        "keys": dict(_REPORT_KEY_LABELS),
        "values": {key: dict(labels) for key, labels in _REPORT_VALUE_LABELS.items()},
    }


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
    # The parent's OOT reading is the held-out time-extrapolation evidence the
    # final fit never re-measured. Listed only when the parent measured OOT; the
    # OOT row count is not in the proof, so sample_count stays None (same columns).
    oot_ks = scope["parent_metrics"].get("oot_ks")
    oot_auc = scope["parent_metrics"].get("oot_auc")
    if any(_is_number(value) for value in (oot_ks, oot_auc)):
        rows.append(
            {
                "split": "pre_refit_oot",
                "sample_count": None,
                "ks": oot_ks,
                "auc": oot_auc,
                "experiment_id": proof["parent"]["experiment_id"],
                "artifact_id": proof["parent"]["artifact_id"],
                "purpose": "parent_heldout_oot_before_refit",
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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def model_narrative(scope=None):
    text = (
        "最终模型按冻结参数拟合完整 development train+test。其训练指标为拟合诊断，"
        "原 test 已参与最终拟合，不能作为最终模型的独立测试或过拟合通过证明。"
        "重训前候选的真实留出 test 选择指标以原实验和产物引用单独列示。"
        "分段、稳定性及 OOT 分布为报告诊断；历史可用性和标签成熟度未知。"
    )
    if scope is None:
        return text
    return f"{text}{_parent_heldout_sentence(scope)}"


def _parent_heldout_sentence(scope) -> str:
    """Name the parent's platform-computed held-out numbers verbatim."""
    metrics = scope.get("parent_metrics") or {}
    parts = [
        f"{label}={metrics[key]:.4f}"
        for key, label in (
            ("test_ks", "测试集 KS"), ("test_auc", "测试集 AUC"),
            ("oot_ks", "OOT KS"), ("oot_auc", "OOT AUC"),
        )
        if isinstance(metrics.get(key), (int, float)) and not isinstance(metrics.get(key), bool)
    ]
    if not parts:
        return ""
    # The experiment id stays in the split table and metadata; the sentence is
    # read by business users (review M4).
    sentence = f"评估指标来自所选实验（重新拟合前）的留出集：{'、'.join(parts)}"
    if metrics.get("overfit_flag") is True:
        sentence += "；该实验被平台标记了过拟合风险"
    return sentence + "。"
