from __future__ import annotations

import json
import numpy as np
from dataclasses import replace
from datetime import UTC, datetime
from marvis.artifacts import ArtifactUnitOfWork
from marvis.packs.modeling.artifact import export_pmml, load_model, persist_model_meta, validate_scorecard_pmml_payload
from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.handoff import create_challenger_backtest_task, handoff_to_validation
from pathlib import Path

from marvis.packs.modeling._common import PMML_SUPPORTED_ALGORITHMS, _format_number_token, _is_metric_key, _json_safe, _optional_float, _resolve_artifact_path, _unique_strings
from marvis.packs.modeling._runtime import _Runtime, _artifact, _artifact_base_dir, _runtime
from marvis.packs.modeling.report_tools import _scorecard_table_rows
from marvis.packs.modeling.scoring import _artifact_calibration_metadata


CHALLENGER_COMPARISON_VERSION = "champion_challenger_v1"


MODEL_CARD_VERSION = "model_card_v1"


MONITORING_POLICY_VERSION = "model_monitoring_v1"


DEFAULT_MONITORING_THRESHOLDS = {
    "psi_test_vs_train": {
        "label": "Test PSI vs Train",
        "metric": "psi_test_vs_train",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.25,
    },
    "psi_oot_vs_train": {
        "label": "OOT PSI vs Train",
        "metric": "psi_oot_vs_train",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.25,
    },
    "overfit_train_test_gap": {
        "label": "Train/Test KS gap",
        "metric": "overfit_train_test_gap",
        "direction": "max",
        "warn": 0.08,
        "fail": 0.12,
    },
    "overfit_train_oot_gap": {
        "label": "Train/OOT KS gap",
        "metric": "overfit_train_oot_gap",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.15,
    },
    "oot_ks": {
        "label": "OOT KS",
        "metric": "oot_ks",
        "direction": "min",
        "warn": 0.25,
        "fail": 0.20,
    },
    "oot_auc": {
        "label": "OOT AUC",
        "metric": "oot_auc",
        "direction": "min",
        "warn": 0.65,
        "fail": 0.60,
    },
    "oot_macro_auc": {
        "label": "OOT Macro AUC",
        "metric": "oot_macro_auc",
        "direction": "min",
        "warn": 0.65,
        "fail": 0.60,
    },
    "oot_rmse": {
        "label": "OOT RMSE",
        "metric": "oot_rmse",
        "direction": "max",
        "warn": None,
        "fail": None,
    },
}


DEFAULT_MONITORING_CHECKS_BY_TARGET = {
    "binary": (
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "overfit_train_test_gap",
        "overfit_train_oot_gap",
        "oot_ks",
        "oot_auc",
    ),
    "continuous": (
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "overfit_train_test_gap",
        "overfit_train_oot_gap",
        "oot_rmse",
    ),
    "multiclass": (
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "overfit_train_test_gap",
        "overfit_train_oot_gap",
        "oot_macro_auc",
    ),
}


def tool_export_pmml(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    experiment = runtime.experiments.get(artifact.experiment_id)
    _require_pmml_supported(
        artifact,
        base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
    )
    pmml_path = _pmml_path(runtime, artifact)
    return {"pmml_path": str(pmml_path)}


def tool_handoff_to_validation(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    _require_pmml_supported(
        artifact,
        operation="validation handoff",
        base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
    )
    validation_task_id = handoff_to_validation(
        runtime.experiments,
        artifact,
        sample_dataset_id=str(inputs["sample_dataset_id"]),
        settings=runtime.settings,
    )
    return {"validation_task_id": validation_task_id}


def tool_post_training_action(inputs: dict, ctx) -> dict:
    """Close the modeling workflow with safe delivery actions.

    PMML export and V1 validation handoff are compatibility deliverables, not a
    reason to fail native-only models. Unsupported actions are returned as
    ``skipped`` with a reason so the user can still use the native artifact/report.
    """
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    capabilities = _artifact_capabilities(artifact, base_dir=base_dir)
    selection_policy_decision = _approval_policy_decision(inputs.get("selection_policy_decision"))
    monitoring_policy = _monitoring_policy_payload(
        experiment=experiment,
        artifact=artifact,
        source=inputs.get("monitoring_policy"),
        selection_policy_decision=selection_policy_decision,
    )
    challenger_comparison = _challenger_comparison_payload(
        runtime=runtime,
        experiment=experiment,
        artifact=artifact,
        source=inputs.get("champion_reference"),
    )
    requested_actions = [
        str(item)
        for item in (
            inputs.get("actions")
            or ["export_pmml", "handoff_to_validation", "create_challenger_backtest"]
        )
        if str(item).strip()
    ]
    actions: list[dict] = []
    pmml_path = ""
    validation_task_id = ""
    challenger_task_id = ""
    challenger_package_path = ""
    challenger_package_markdown_path = ""
    reason = str(capabilities.get("reason") or "")

    if "export_pmml" in requested_actions:
        if capabilities.get("pmml_supported"):
            pmml_path = str(_pmml_path(runtime, artifact))
            action = {"action": "export_pmml", "status": "succeeded", "pmml_path": pmml_path}
            note = _pmml_delivery_note(capabilities)
            if note:
                action["reason"] = note
            actions.append(action)
        else:
            actions.append({"action": "export_pmml", "status": "skipped", "reason": reason})

    if "handoff_to_validation" in requested_actions:
        sample_dataset_id = str(inputs.get("sample_dataset_id") or "").strip()
        if capabilities.get("handoff_supported") and sample_dataset_id:
            validation_task_id = handoff_to_validation(
                runtime.experiments,
                artifact,
                sample_dataset_id=sample_dataset_id,
                settings=runtime.settings,
            )
            actions.append({
                "action": "handoff_to_validation",
                "status": "succeeded",
                "validation_task_id": validation_task_id,
                "reason": _handoff_delivery_note(capabilities),
            })
        else:
            actions.append({
                "action": "handoff_to_validation",
                "status": "skipped",
                "reason": reason or "sample_dataset_id is required for validation handoff",
            })

    if "create_challenger_backtest" in requested_actions:
        sample_dataset_id = str(inputs.get("sample_dataset_id") or "").strip()
        if capabilities.get("handoff_supported") and sample_dataset_id:
            challenger = create_challenger_backtest_task(
                runtime.experiments,
                artifact,
                sample_dataset_id=sample_dataset_id,
                settings=runtime.settings,
                selection_policy_decision=selection_policy_decision,
                monitoring_policy=monitoring_policy,
                challenger_comparison=challenger_comparison,
            )
            challenger_task_id = challenger["task_id"]
            challenger_package_path = challenger["package_path"]
            challenger_package_markdown_path = challenger["markdown_path"]
            actions.append({
                "action": "create_challenger_backtest",
                "status": "succeeded",
                "challenger_task_id": challenger_task_id,
                "package_path": challenger_package_path,
                "markdown_path": challenger_package_markdown_path,
            })
        else:
            actions.append({
                "action": "create_challenger_backtest",
                "status": "skipped",
                "reason": reason or "sample_dataset_id and PMML-capable model are required",
            })

    model_card = _model_card_payload(
        experiment=experiment,
        artifact=artifact,
        capabilities=capabilities,
        actions=actions,
        sample_dataset_id=str(inputs.get("sample_dataset_id") or ""),
        pmml_path=pmml_path,
        validation_task_id=validation_task_id,
        challenger_task_id=challenger_task_id,
        challenger_package_path=challenger_package_path,
        challenger_package_markdown_path=challenger_package_markdown_path,
        selection_policy_decision=selection_policy_decision,
        monitoring_policy=monitoring_policy,
        challenger_comparison=challenger_comparison,
    )
    approval_package = _write_approval_package(
        base_dir,
        experiment=experiment,
        artifact=artifact,
        capabilities=capabilities,
        actions=actions,
        sample_dataset_id=str(inputs.get("sample_dataset_id") or ""),
        pmml_path=pmml_path,
        validation_task_id=validation_task_id,
        challenger_task_id=challenger_task_id,
        challenger_package_path=challenger_package_path,
        challenger_package_markdown_path=challenger_package_markdown_path,
        selection_policy_decision=selection_policy_decision,
        monitoring_policy=monitoring_policy,
        challenger_comparison=challenger_comparison,
        model_card=model_card,
    )
    return {
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "native_model_path": str(artifact.model_path),
        "pmml_path": pmml_path,
        "validation_task_id": validation_task_id,
        "challenger_task_id": challenger_task_id,
        "challenger_package_path": challenger_package_path,
        "challenger_package_markdown_path": challenger_package_markdown_path,
        "approval_package_path": str(approval_package["json_path"]),
        "approval_package_markdown_path": str(approval_package["markdown_path"]),
        "monitoring_policy_path": str(approval_package["monitoring_policy_path"]),
        "monitoring_policy_markdown_path": str(approval_package["monitoring_policy_markdown_path"]),
        "monitoring_policy": monitoring_policy,
        "challenger_comparison_path": str(approval_package.get("challenger_comparison_path") or ""),
        "challenger_comparison_markdown_path": str(
            approval_package.get("challenger_comparison_markdown_path") or ""
        ),
        "challenger_comparison": challenger_comparison,
        "model_card_path": str(approval_package["model_card_path"]),
        "model_card_markdown_path": str(approval_package["model_card_markdown_path"]),
        "model_card": model_card,
        "capabilities": capabilities,
        "actions": actions,
    }


def _approval_policy_decision(value) -> dict:
    decision = value if isinstance(value, dict) else {}
    if not decision:
        return {}
    violations = []
    for item in decision.get("violations") or []:
        if not isinstance(item, dict):
            continue
        violations.append({
            "code": str(item.get("code") or ""),
            "message": str(item.get("message") or ""),
        })
    # SEL-7 quality warnings (overfit_warning / sanity_warning) are non-blocking
    # but must survive normalization so _model_card_limitations can surface them
    # on the human-facing model card; dropping them here re-hid the warning.
    warnings = []
    for item in decision.get("warnings") or []:
        if not isinstance(item, dict):
            continue
        warnings.append({
            "code": str(item.get("code") or ""),
            "message": str(item.get("message") or ""),
        })
    return {
        "status": str(decision.get("status") or ""),
        "explicit_selection": bool(decision.get("explicit_selection")),
        "selected_experiment_id": str(decision.get("selected_experiment_id") or ""),
        "policy": _json_safe(decision.get("policy") if isinstance(decision.get("policy"), dict) else {}),
        "profile": _json_safe(decision.get("profile") if isinstance(decision.get("profile"), dict) else {}),
        "violations": violations,
        "warnings": warnings,
        "override_reason": str(decision.get("override_reason") or ""),
    }


def _monitoring_policy_payload(
    *,
    experiment,
    artifact: ModelArtifact,
    source,
    selection_policy_decision: dict,
) -> dict:
    source_policy = source if isinstance(source, dict) else {}
    target_type = str(getattr(experiment.config, "target_type", "binary") or "binary")
    thresholds = _monitoring_thresholds(source_policy.get("thresholds"), target_type=target_type)
    baseline_metrics = _json_safe(experiment.metrics) or {}
    checks = [
        _monitoring_check_payload(check_id, spec, baseline_metrics)
        for check_id, spec in thresholds.items()
    ]
    overall_status = _monitoring_overall_status(checks)
    sample_weight_policy = _sample_weight_policy_payload(experiment=experiment, artifact=artifact)
    return {
        "schema_version": 1,
        "policy_version": str(source_policy.get("policy_version") or MONITORING_POLICY_VERSION),
        "created_at": datetime.now(UTC).isoformat(),
        "status": overall_status,
        "recommendation": _monitoring_recommendation(overall_status),
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "recipe": experiment.recipe_id,
        "algorithm": artifact.algorithm,
        "target_type": target_type,
        "dataset_id": getattr(experiment.config, "dataset_id", ""),
        "target_col": getattr(experiment.config, "target_col", ""),
        "split_col": getattr(experiment.config, "split_col", ""),
        "baseline_metrics": baseline_metrics,
        "checks": checks,
        "sample_weight_policy": sample_weight_policy,
        "selection_policy_status": str(selection_policy_decision.get("status") or ""),
        "review_cadence": str(source_policy.get("review_cadence") or "monthly"),
        "owner": str(source_policy.get("owner") or "model_risk"),
        "notes": str(source_policy.get("notes") or ""),
    }


def _monitoring_thresholds(source, *, target_type: str) -> dict:
    default_keys = DEFAULT_MONITORING_CHECKS_BY_TARGET.get(target_type, DEFAULT_MONITORING_CHECKS_BY_TARGET["binary"])
    thresholds = {
        key: dict(DEFAULT_MONITORING_THRESHOLDS[key])
        for key in default_keys
        if key in DEFAULT_MONITORING_THRESHOLDS
    }
    if not isinstance(source, dict):
        return thresholds
    for key, override in source.items():
        if not isinstance(override, dict):
            continue
        normalized_key = str(key)
        base = thresholds.get(normalized_key, {
            "label": normalized_key,
            "metric": normalized_key,
            "direction": "max",
            "warn": None,
            "fail": None,
        })
        merged = dict(base)
        for field in ("label", "metric", "direction", "warn", "fail"):
            if field in override:
                merged[field] = override[field]
        thresholds[normalized_key] = merged
    return thresholds


def _monitoring_check_payload(check_id: str, spec: dict, metrics: dict) -> dict:
    metric = str(spec.get("metric") or check_id)
    value = metrics.get(metric)
    direction = str(spec.get("direction") or "max")
    warn = _optional_float(spec.get("warn"))
    fail = _optional_float(spec.get("fail"))
    status, message = _monitoring_check_status(value, direction=direction, warn=warn, fail=fail)
    return {
        "id": str(check_id),
        "label": str(spec.get("label") or check_id),
        "metric": metric,
        "value": value,
        "direction": direction,
        "warn": warn,
        "fail": fail,
        "status": status,
        "message": message,
    }


def _monitoring_check_status(value, *, direction: str, warn: float | None, fail: float | None) -> tuple[str, str]:
    numeric = _optional_float(value)
    if numeric is None:
        return "missing", "指标缺失，需在监控任务中补充基线或跳过原因"
    if warn is None and fail is None:
        return "needs_policy", "缺少业务阈值，需配置 warn/fail 阈值后纳入自动判断"
    if direction == "min":
        if fail is not None and numeric < fail:
            return "fail", f"{_format_number_token(numeric)} 低于 fail 阈值 {_format_number_token(fail)}"
        if warn is not None and numeric < warn:
            return "warn", f"{_format_number_token(numeric)} 低于 warn 阈值 {_format_number_token(warn)}"
    else:
        if fail is not None and numeric > fail:
            return "fail", f"{_format_number_token(numeric)} 高于 fail 阈值 {_format_number_token(fail)}"
        if warn is not None and numeric > warn:
            return "warn", f"{_format_number_token(numeric)} 高于 warn 阈值 {_format_number_token(warn)}"
    return "pass", "基线指标在监控阈值内"


def _monitoring_overall_status(checks: list[dict]) -> str:
    statuses = {str(item.get("status") or "") for item in checks}
    if "fail" in statuses:
        return "fail"
    if statuses & {"warn", "missing", "needs_policy"}:
        return "warn"
    return "pass"


def _monitoring_recommendation(status: str) -> str:
    if status == "pass":
        return "可进入常规监控"
    if status == "fail":
        return "需模型风险复核后再交付"
    return "需补充监控阈值或业务说明"


def _sample_weight_policy_payload(*, experiment, artifact: ModelArtifact) -> dict:
    config_params = getattr(getattr(experiment, "config", None), "params", {})
    config_col = _sample_weight_col_from_params(config_params)
    artifact_col = _sample_weight_col_from_params(artifact.params)
    sample_weight_col = artifact_col or config_col
    used = bool(sample_weight_col)
    source = "artifact_params" if artifact_col else "train_config_params" if config_col else "none"
    approval_items = [
        "训练未使用样本权重；如后续引入拒绝推断、成本权重或抽样校正，需要重新执行筛选、调参、训练和审批。"
    ]
    monitoring_checks: list[dict] = []
    if used:
        approval_items = [
            "确认样本权重列的业务定义、生成逻辑、适用样本范围和取值边界。",
            "审批时同时查看加权与非加权验证指标，确认模型收益不是仅由权重口径驱动。",
            "上线监控需跟踪权重列可用率、非正值占比和分布漂移，权重口径变化时触发重新审批。",
        ]
        monitoring_checks = [
            {
                "id": "sample_weight_availability",
                "metric": f"{sample_weight_col}.missing_or_non_positive_rate",
                "status": "needs_baseline",
                "recommendation": "配置缺失、非正值和异常高权重占比阈值。",
            },
            {
                "id": "sample_weight_distribution",
                "metric": f"{sample_weight_col}.population_stability",
                "status": "needs_baseline",
                "recommendation": "配置训练基线分布并按月监控 PSI/分位数漂移。",
            },
        ]
    return _json_safe({
        "schema_version": 1,
        "used": used,
        "sample_weight_col": sample_weight_col,
        "source": source,
        "approval_policy": {
            "requires_manual_review": used,
            "review_items": approval_items,
        },
        "monitoring_defaults": {
            "requires_monitoring": used,
            "review_cadence": "monthly" if used else "standard",
            "checks": monitoring_checks,
        },
    })


def _sample_weight_col_from_params(params) -> str:
    if not isinstance(params, dict):
        return ""
    for key in ("sample_weight_col", "sample_weight_column", "weight_col"):
        value = str(params.get(key) or "").strip()
        if value:
            return value
    return ""


def _challenger_comparison_payload(
    *,
    runtime: _Runtime,
    experiment,
    artifact: ModelArtifact,
    source,
) -> dict:
    champion = _resolve_champion_reference(runtime, source)
    if not champion and source is None:
        champion = _previous_selected_champion_reference(
            runtime=runtime,
            experiment=experiment,
            artifact=artifact,
        )
    if not champion:
        return {}
    challenger_metrics = _json_safe(experiment.metrics) or {}
    rows = _challenger_metric_comparisons(
        champion.get("metrics") if isinstance(champion.get("metrics"), dict) else {},
        challenger_metrics if isinstance(challenger_metrics, dict) else {},
    )
    summary = _challenger_comparison_summary(rows)
    status = _challenger_comparison_status(summary)
    return _json_safe({
        "schema_version": 1,
        "comparison_version": CHALLENGER_COMPARISON_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "recommendation": _challenger_comparison_recommendation(status, summary),
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "target_type": getattr(experiment.config, "target_type", "binary"),
        "dataset_id": getattr(experiment.config, "dataset_id", ""),
        "challenger": {
            "label": "challenger",
            "experiment_id": experiment.id,
            "artifact_id": artifact.id,
            "recipe": experiment.recipe_id,
            "algorithm": artifact.algorithm,
            "metrics": challenger_metrics,
        },
        "champion": champion,
        "metric_comparisons": rows,
        "summary": summary,
    })


def _resolve_champion_reference(runtime: _Runtime, source) -> dict:
    if not isinstance(source, dict) or source.get("enabled") is False:
        return {}
    experiment_id = str(source.get("experiment_id") or "").strip()
    explicit_metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    label = str(source.get("label") or "prior_champion").strip() or "prior_champion"
    champion_experiment = None
    champion_artifact = None
    if experiment_id:
        try:
            champion_experiment = runtime.experiments.get(experiment_id)
        except KeyError as exc:
            raise ModelingError(f"champion_reference experiment_id not found: {experiment_id}") from exc
        if champion_experiment.artifact_id:
            champion_artifact = _artifact(runtime, champion_experiment.artifact_id)
    if not experiment_id and not explicit_metrics and not str(source.get("artifact_id") or "").strip():
        return {}
    metrics = explicit_metrics or (
        _json_safe(champion_experiment.metrics) if champion_experiment is not None else {}
    )
    return _json_safe({
        "label": label,
        "experiment_id": experiment_id,
        "artifact_id": str(
            source.get("artifact_id")
            or getattr(champion_experiment, "artifact_id", "")
            or ""
        ),
        "recipe": str(source.get("recipe") or getattr(champion_experiment, "recipe_id", "") or ""),
        "algorithm": str(source.get("algorithm") or getattr(champion_artifact, "algorithm", "") or ""),
        "metrics": metrics if isinstance(metrics, dict) else {},
        "notes": str(source.get("notes") or ""),
    })


def _previous_selected_champion_reference(
    *,
    runtime: _Runtime,
    experiment,
    artifact: ModelArtifact,
) -> dict:
    statuses = {"selected", "validated", "delivered", "approved", "champion"}
    current_created_at = str(getattr(experiment, "created_at", "") or "")
    candidates = []
    for candidate in runtime.experiments.list_for_task(experiment.task_id):
        if candidate.id == experiment.id or candidate.artifact_id == artifact.id:
            continue
        if candidate.metrics is None or str(candidate.status or "") not in statuses:
            continue
        candidate_created_at = str(getattr(candidate, "created_at", "") or "")
        if current_created_at and candidate_created_at and candidate_created_at >= current_created_at:
            continue
        candidates.append(candidate)
    if not candidates:
        return {}
    champion_experiment = max(
        candidates,
        key=lambda item: (str(getattr(item, "created_at", "") or ""), item.id),
    )
    champion_artifact = (
        _artifact(runtime, champion_experiment.artifact_id)
        if champion_experiment.artifact_id
        else None
    )
    return _json_safe({
        "label": "previous_selected_experiment",
        "experiment_id": champion_experiment.id,
        "artifact_id": str(champion_experiment.artifact_id or ""),
        "recipe": champion_experiment.recipe_id,
        "algorithm": str(getattr(champion_artifact, "algorithm", "") or ""),
        "metrics": _json_safe(champion_experiment.metrics) or {},
        "notes": "Auto-resolved from an earlier selected experiment in the same task.",
    })


def _challenger_metric_comparisons(champion_metrics: dict, challenger_metrics: dict) -> list[dict]:
    metric_keys = sorted(
        {
            key
            for key in set(champion_metrics) | set(challenger_metrics)
            if _is_metric_key(str(key))
        },
        key=_challenger_metric_sort_key,
    )
    rows: list[dict] = []
    for key in metric_keys:
        metric = str(key)
        champion_value = _numeric_metric(champion_metrics.get(metric))
        challenger_value = _numeric_metric(challenger_metrics.get(metric))
        direction = _metric_better_direction(metric)
        verdict = "missing"
        delta = None
        if champion_value is not None and challenger_value is not None:
            delta = challenger_value - champion_value
            verdict = _metric_verdict(delta, direction)
        rows.append({
            "metric": metric,
            "champion_value": champion_value,
            "challenger_value": challenger_value,
            "delta": delta,
            "direction": direction,
            "verdict": verdict,
        })
    return rows


def _challenger_metric_sort_key(metric: str) -> tuple[int, str]:
    preferred = [
        "oot_ks",
        "test_ks",
        "oot_auc",
        "test_auc",
        "oot_macro_auc",
        "test_macro_auc",
        "oot_accuracy",
        "test_accuracy",
        "oot_r2",
        "test_r2",
        "oot_rmse",
        "test_rmse",
        "oot_mae",
        "test_mae",
        "oot_logloss",
        "test_logloss",
        "psi_oot_vs_train",
        "psi_test_vs_train",
    ]
    try:
        return (preferred.index(metric), metric)
    except ValueError:
        return (len(preferred), metric)


def _numeric_metric(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
        return float(value)
    return None


def _metric_better_direction(metric: str) -> str:
    lowered = metric.lower()
    if (
        lowered.startswith("psi_")
        or "rmse" in lowered
        or "mae" in lowered
        or "logloss" in lowered
        or "loss" in lowered
        or "brier" in lowered
        or "ece" in lowered
        or "gap" in lowered
    ):
        return "lower"
    return "higher"


def _metric_verdict(delta: float, direction: str) -> str:
    if abs(delta) <= 1e-12:
        return "same"
    if direction == "lower":
        return "improved" if delta < 0 else "declined"
    return "improved" if delta > 0 else "declined"


def _challenger_comparison_summary(rows: list[dict]) -> dict:
    comparable = [item for item in rows if item.get("verdict") != "missing"]
    return {
        "metric_count": len(rows),
        "comparable_metric_count": len(comparable),
        "improved_count": sum(1 for item in comparable if item.get("verdict") == "improved"),
        "declined_count": sum(1 for item in comparable if item.get("verdict") == "declined"),
        "same_count": sum(1 for item in comparable if item.get("verdict") == "same"),
        "missing_count": sum(1 for item in rows if item.get("verdict") == "missing"),
    }


def _challenger_comparison_status(summary: dict) -> str:
    if int(summary.get("comparable_metric_count") or 0) <= 0:
        return "missing"
    if int(summary.get("declined_count") or 0) > 0:
        return "warn"
    return "pass"


def _challenger_comparison_recommendation(status: str, summary: dict) -> str:
    if status == "pass":
        return "Challenger 不弱于 Champion 的已配置指标"
    if status == "warn":
        return (
            "Challenger 有指标弱于 Champion, 需业务复核差异 "
            f"({summary.get('declined_count', 0)} 项下降)"
        )
    return "缺少可比较的 Champion/Challenger 指标, 需补充生产模型基线"


def _model_card_payload(
    *,
    experiment,
    artifact: ModelArtifact,
    capabilities: dict,
    actions: list[dict],
    sample_dataset_id: str,
    pmml_path: str,
    validation_task_id: str,
    challenger_task_id: str,
    challenger_package_path: str,
    challenger_package_markdown_path: str,
    selection_policy_decision: dict,
    monitoring_policy: dict,
    challenger_comparison: dict,
) -> dict:
    config = experiment.config
    metrics = _json_safe(experiment.metrics) or {}
    sample_weight_policy = _sample_weight_policy_payload(experiment=experiment, artifact=artifact)
    selection_policy = (
        selection_policy_decision.get("policy")
        if isinstance(selection_policy_decision.get("policy"), dict)
        else {}
    )
    selection_policy_requirements = [
        {"requirement": label, "configured": value}
        for label, value in _selection_policy_requirement_markdown_rows(selection_policy)
    ]
    limitations = _model_card_limitations(
        capabilities=capabilities,
        selection_policy_decision=selection_policy_decision,
        monitoring_policy=monitoring_policy,
        challenger_comparison=challenger_comparison,
    )
    return _json_safe({
        "schema_version": 1,
        "card_version": MODEL_CARD_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "title": f"{artifact.algorithm} model card",
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "recipe": experiment.recipe_id,
        "algorithm": artifact.algorithm,
        "target_type": getattr(config, "target_type", "binary"),
        "dataset_id": getattr(config, "dataset_id", ""),
        "sample_dataset_id": sample_dataset_id,
        "target_col": getattr(config, "target_col", ""),
        "split_col": getattr(config, "split_col", ""),
        "split_values": _json_safe(getattr(config, "split_values", {})),
        "seed": getattr(config, "seed", None),
        "feature_count": len(artifact.feature_list),
        "feature_preview": list(artifact.feature_list[:30]),
        "sample_weight_col": str(sample_weight_policy.get("sample_weight_col") or ""),
        "training": {
            "sample_weight": sample_weight_policy,
        },
        "key_metrics": _model_card_key_metrics(metrics),
        "governance": {
            "selection_policy_status": str(selection_policy_decision.get("status") or "not_requested"),
            "selection_policy": _json_safe(selection_policy),
            "selection_policy_requirements": selection_policy_requirements,
            "selection_policy_violations": _json_safe(selection_policy_decision.get("violations") or []),
            "selection_policy_override_reason": str(selection_policy_decision.get("override_reason") or ""),
            "monitoring_status": str(monitoring_policy.get("status") or "not_configured"),
            "monitoring_recommendation": str(monitoring_policy.get("recommendation") or ""),
            "champion_comparison_status": str(challenger_comparison.get("status") or "not_configured"),
            "champion_comparison_recommendation": str(challenger_comparison.get("recommendation") or ""),
        },
        "delivery": {
            "native_model_path": str(artifact.model_path or ""),
            "pmml_path": str(pmml_path or ""),
            "pmml_includes_calibration": capabilities.get("pmml_includes_calibration", True),
            "calibration": _json_safe(capabilities.get("calibration") or {}),
            "validation_task_id": str(validation_task_id or ""),
            "challenger_task_id": str(challenger_task_id or ""),
            "challenger_package_path": str(challenger_package_path or ""),
            "challenger_package_markdown_path": str(challenger_package_markdown_path or ""),
            "export_pmml_status": _model_card_action_status(actions, "export_pmml"),
            "validation_handoff_status": _model_card_action_status(actions, "handoff_to_validation"),
            "challenger_backtest_status": _model_card_action_status(actions, "create_challenger_backtest"),
        },
        "capabilities": _json_safe(capabilities),
        "limitations": limitations,
        "next_review_actions": _model_card_next_review_actions(
            limitations,
            monitoring_policy,
            challenger_comparison,
            sample_weight_policy,
        ),
    })


def _model_card_key_metrics(metrics: dict) -> list[dict]:
    rows = []
    for metric in [
        "oot_ks",
        "test_ks",
        "train_ks",
        "oot_auc",
        "test_auc",
        "oot_macro_auc",
        "test_macro_auc",
        "oot_accuracy",
        "test_accuracy",
        "oot_rmse",
        "test_rmse",
        "oot_mae",
        "test_mae",
        "oot_logloss",
        "test_logloss",
        "psi_oot_vs_train",
        "psi_test_vs_train",
        "overfit_flag",
        # TUNE-5: weighted KS/AUC, only present when the model trained with a
        # sample_weight_col -- the口径 that actually drove selection, alongside
        # the unweighted reading above.
        "weighted_oot_ks",
        "weighted_test_ks",
        "weighted_train_ks",
        "weighted_oot_auc",
        "weighted_test_auc",
    ]:
        if metric in metrics and metrics.get(metric) is not None:
            rows.append({"metric": metric, "value": metrics.get(metric)})
    return rows


def _model_card_action_status(actions: list[dict], action: str) -> str:
    for item in actions:
        if isinstance(item, dict) and item.get("action") == action:
            return str(item.get("status") or "")
    return "not_requested"


def _model_card_limitations(
    *,
    capabilities: dict,
    selection_policy_decision: dict,
    monitoring_policy: dict,
    challenger_comparison: dict,
) -> list[str]:
    limitations: list[str] = [
        str(item)
        for item in (capabilities.get("limitations") or [])
        if str(item)
    ]
    if not capabilities.get("pmml_supported"):
        reason = str(capabilities.get("reason") or "PMML export is not supported for this artifact.")
        limitations.append(reason)
    calibration = capabilities.get("calibration") if isinstance(capabilities.get("calibration"), dict) else {}
    if calibration and capabilities.get("pmml_includes_calibration") is False:
        method = str(calibration.get("method") or "unknown")
        limitations.append(
            f"模型已进行 {method} 概率校准，但 PMML 产物不包含校准器；"
            "验证移交 Notebook 会加载 calibration.joblib 应用校准。"
        )
    policy_status = str(selection_policy_decision.get("status") or "")
    if policy_status in {"blocked", "overridden"}:
        limitations.append(f"Selection policy status is {policy_status}.")
    violations = [
        str(item.get("message") or item.get("code") or "")
        for item in (selection_policy_decision.get("violations") or [])
        if isinstance(item, dict)
    ]
    limitations.extend(f"选型策略违规:{item}" for item in violations if item)
    # SEL-7 quality warnings (overfit_warning / sanity_warning) do not block
    # selection, but a reviewer still has to see them on the model card; they
    # were previously computed at selection time and then dropped on the floor.
    warnings = [
        str(item.get("message") or item.get("code") or "")
        for item in (selection_policy_decision.get("warnings") or [])
        if isinstance(item, dict)
    ]
    limitations.extend(f"选型策略警示:{item}" for item in warnings if item)
    monitoring_status = str(monitoring_policy.get("status") or "")
    if monitoring_status in {"warn", "fail", "missing", "needs_policy"}:
        limitations.append(str(monitoring_policy.get("recommendation") or "Monitoring policy needs review."))
    comparison_status = str(challenger_comparison.get("status") or "")
    if comparison_status in {"warn", "missing"}:
        limitations.append(
            str(challenger_comparison.get("recommendation") or "Champion comparison needs review.")
        )
    return _unique_strings([item for item in limitations if item])


def _model_card_next_review_actions(
    limitations: list[str],
    monitoring_policy: dict,
    challenger_comparison: dict,
    sample_weight_policy: dict | None = None,
) -> list[str]:
    actions = ["确认模型卡、审批包、监控策略与交付产物路径一致。"]
    if limitations:
        actions.append("逐项复核模型限制与放行说明。")
    if isinstance(sample_weight_policy, dict) and sample_weight_policy.get("used"):
        actions.append("复核样本权重业务口径、加权/非加权指标差异和上线权重列监控阈值。")
    if monitoring_policy:
        actions.append("按监控策略配置上线后的漂移/稳定性复核。")
    if challenger_comparison:
        actions.append("结合 Champion 对比结论确认是否接受 Challenger。")
    return actions


def _write_approval_package(
    base_dir: Path,
    *,
    experiment,
    artifact: ModelArtifact,
    capabilities: dict,
    actions: list[dict],
    sample_dataset_id: str,
    pmml_path: str,
    validation_task_id: str,
    challenger_task_id: str,
    challenger_package_path: str,
    challenger_package_markdown_path: str,
    selection_policy_decision: dict,
    monitoring_policy: dict,
    challenger_comparison: dict,
    model_card: dict,
) -> dict[str, Path | None]:
    uow = ArtifactUnitOfWork()
    json_artifact = uow.stage_file(base_dir, f"{artifact.id}.approval_package.json")
    markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.approval_package.md")
    monitoring_json_artifact = uow.stage_file(base_dir, f"{artifact.id}.monitoring_policy.json")
    monitoring_markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.monitoring_policy.md")
    model_card_json_artifact = uow.stage_file(base_dir, f"{artifact.id}.model_card.json")
    model_card_markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.model_card.md")
    comparison_json_artifact = None
    comparison_markdown_artifact = None
    if challenger_comparison:
        comparison_json_artifact = uow.stage_file(base_dir, f"{artifact.id}.champion_comparison.json")
        comparison_markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.champion_comparison.md")
    config = experiment.config
    sample_weight_policy = _sample_weight_policy_payload(experiment=experiment, artifact=artifact)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "recipe": experiment.recipe_id,
        "algorithm": artifact.algorithm,
        "target_type": getattr(config, "target_type", "binary"),
        "dataset_id": getattr(config, "dataset_id", ""),
        "sample_dataset_id": sample_dataset_id,
        "target_col": getattr(config, "target_col", ""),
        "split_col": getattr(config, "split_col", ""),
        "split_values": _json_safe(getattr(config, "split_values", {})),
        "seed": getattr(config, "seed", None),
        "sample_weight_col": str(sample_weight_policy.get("sample_weight_col") or ""),
        "training": {
            "sample_weight": sample_weight_policy,
        },
        "features": list(artifact.feature_list),
        "feature_count": len(artifact.feature_list),
        "metrics": _json_safe(experiment.metrics),
        "capabilities": _json_safe(capabilities),
        "selection_policy_decision": selection_policy_decision,
        "monitoring_policy": _json_safe(monitoring_policy),
        "challenger_comparison": _json_safe(challenger_comparison),
        "model_card": _json_safe(model_card),
        "delivery_actions": _json_safe(actions),
        "artifacts": {
            "native_model_path": str(artifact.model_path or ""),
            "pmml_path": str(pmml_path or ""),
            "validation_task_id": str(validation_task_id or ""),
            "challenger_task_id": str(challenger_task_id or ""),
            "challenger_package_path": str(challenger_package_path or ""),
            "challenger_package_markdown_path": str(challenger_package_markdown_path or ""),
            "model_card_path": str(model_card_json_artifact.final_path),
            "model_card_markdown_path": str(model_card_markdown_artifact.final_path),
            "challenger_comparison_path": (
                str(comparison_json_artifact.final_path) if comparison_json_artifact else ""
            ),
            "challenger_comparison_markdown_path": (
                str(comparison_markdown_artifact.final_path) if comparison_markdown_artifact else ""
            ),
        },
        "scorecard_table": _json_safe(_scorecard_table_rows(artifact)),
        "model_params": _json_safe(artifact.params),
    }
    safe_payload = _json_safe(payload)
    try:
        json_artifact.path.write_text(
            json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        markdown_artifact.path.write_text(
            _approval_package_markdown(safe_payload),
            encoding="utf-8",
        )
        monitoring_json_artifact.path.write_text(
            json.dumps(safe_payload["monitoring_policy"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        monitoring_markdown_artifact.path.write_text(
            _monitoring_policy_markdown(safe_payload["monitoring_policy"]),
            encoding="utf-8",
        )
        model_card_json_artifact.path.write_text(
            json.dumps(safe_payload["model_card"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        model_card_markdown_artifact.path.write_text(
            _model_card_markdown(safe_payload["model_card"]),
            encoding="utf-8",
        )
        if comparison_json_artifact and comparison_markdown_artifact:
            comparison_json_artifact.path.write_text(
                json.dumps(
                    safe_payload["challenger_comparison"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            comparison_markdown_artifact.path.write_text(
                _challenger_comparison_markdown(safe_payload["challenger_comparison"]),
                encoding="utf-8",
            )
        return uow.finalize(lambda: {
            "json_path": json_artifact.final_path,
            "markdown_path": markdown_artifact.final_path,
            "monitoring_policy_path": monitoring_json_artifact.final_path,
            "monitoring_policy_markdown_path": monitoring_markdown_artifact.final_path,
            "model_card_path": model_card_json_artifact.final_path,
            "model_card_markdown_path": model_card_markdown_artifact.final_path,
            "challenger_comparison_path": (
                comparison_json_artifact.final_path if comparison_json_artifact else None
            ),
            "challenger_comparison_markdown_path": (
                comparison_markdown_artifact.final_path if comparison_markdown_artifact else None
            ),
        })
    except Exception:
        uow.rollback()
        raise


def _approval_package_markdown(payload: dict) -> str:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    policy = (
        payload.get("selection_policy_decision")
        if isinstance(payload.get("selection_policy_decision"), dict)
        else {}
    )
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    actions = [item for item in (payload.get("delivery_actions") or []) if isinstance(item, dict)]
    capabilities = payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else {}
    limitations = [
        str(item)
        for item in (capabilities.get("limitations") or [])
        if str(item)
    ]
    monitoring = payload.get("monitoring_policy") if isinstance(payload.get("monitoring_policy"), dict) else {}
    comparison = (
        payload.get("challenger_comparison")
        if isinstance(payload.get("challenger_comparison"), dict)
        else {}
    )
    features = [str(item) for item in (payload.get("features") or []) if str(item)]
    violations = [item for item in (policy.get("violations") or []) if isinstance(item, dict)]
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    sample_weight = (
        training.get("sample_weight")
        if isinstance(training.get("sample_weight"), dict)
        else {}
    )
    lines = [
        "# 模型审批包",
        "",
        "## 基本信息",
        "",
        f"- 实验ID: `{_md_inline(payload.get('experiment_id'))}`",
        f"- 产物ID: `{_md_inline(payload.get('artifact_id'))}`",
        f"- 算法: `{_md_inline(payload.get('algorithm'))}`",
        f"- 目标类型: `{_md_inline(payload.get('target_type'))}`",
        f"- 目标列: `{_md_inline(payload.get('target_col'))}`",
        f"- 样本集: `{_md_inline(payload.get('sample_dataset_id') or payload.get('dataset_id'))}`",
        f"- 特征数: {_md_inline(payload.get('feature_count'))}",
        f"- 样本权重: `{_md_inline(payload.get('sample_weight_col') or '未使用')}`",
        "",
        "## 关键指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    metric_rows = [
        (key, value)
        for key, value in metrics.items()
        if key.startswith(("test_", "oot_", "psi_")) or key in {"overfit_flag"}
    ]
    if metric_rows:
        for key, value in sorted(metric_rows):
            lines.append(f"| {_md_cell(key)} | {_md_cell(_metric_display(value))} |")
    else:
        lines.append("| - | - |")
    lines.extend([
        "",
        "## 策略执行",
        "",
        f"- 状态: `{_md_inline(policy.get('status') or 'not_requested')}`",
        f"- Override原因: {_md_inline(policy.get('override_reason') or '-')}",
    ])
    policy_requirement_rows = _selection_policy_requirement_markdown_rows(policy.get("policy"))
    if policy_requirement_rows:
        lines.extend(["", "| 策略要求 | 配置 |", "| --- | --- |"])
        for label, value in policy_requirement_rows:
            lines.append(f"| {_md_cell(label)} | {_md_cell(value)} |")
    if sample_weight:
        lines.extend(_sample_weight_policy_markdown_section(sample_weight, heading="## 样本权重治理"))
    if violations:
        lines.extend(["", "| 违规项 | 说明 |", "| --- | --- |"])
        for item in violations:
            lines.append(
                f"| {_md_cell(item.get('code') or '-')} | {_md_cell(item.get('message') or '-')} |"
            )
    if limitations:
        lines.extend(["", "## 交付限制", ""])
        for item in limitations:
            lines.append(f"- {_md_inline(item)}")
    if monitoring:
        lines.extend([
            "",
            "## 监控策略",
            "",
            f"- 版本: `{_md_inline(monitoring.get('policy_version'))}`",
            f"- 状态: `{_md_inline(monitoring.get('status'))}`",
            f"- 建议: {_md_inline(monitoring.get('recommendation') or '-')}",
        ])
        monitor_checks = [
            item for item in (monitoring.get("checks") or [])
            if isinstance(item, dict)
        ][:8]
        if monitor_checks:
            lines.extend(["", "| 检查项 | 状态 | 当前值 | 阈值 |", "| --- | --- | ---: | --- |"])
            for item in monitor_checks:
                threshold = _monitoring_threshold_display(item)
                lines.append(
                    f"| {_md_cell(item.get('label') or item.get('id') or '-')} | "
                    f"{_md_cell(item.get('status') or '-')} | "
                    f"{_md_cell(_metric_display(item.get('value')))} | "
                    f"{_md_cell(threshold)} |"
                )
    if comparison:
        lines.extend([
            "",
            "## Champion对比",
            "",
            f"- 状态: `{_md_inline(comparison.get('status'))}`",
            f"- 建议: {_md_inline(comparison.get('recommendation') or '-')}",
        ])
        champion = comparison.get("champion") if isinstance(comparison.get("champion"), dict) else {}
        if champion:
            lines.extend([
                f"- Champion: `{_md_inline(champion.get('label') or 'prior_champion')}`",
                f"- Champion实验: `{_md_inline(champion.get('experiment_id') or '-')}`",
            ])
        rows = [
            item for item in (comparison.get("metric_comparisons") or [])
            if isinstance(item, dict)
        ][:12]
        if rows:
            lines.extend([
                "",
                "| 指标 | Champion | Challenger | 差异 | 方向 | 结论 |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ])
            for item in rows:
                lines.append(
                    f"| {_md_cell(item.get('metric') or '-')} | "
                    f"{_md_cell(_metric_display(item.get('champion_value')))} | "
                    f"{_md_cell(_metric_display(item.get('challenger_value')))} | "
                    f"{_md_cell(_metric_display(item.get('delta')))} | "
                    f"{_md_cell(item.get('direction') or '-')} | "
                    f"{_md_cell(item.get('verdict') or '-')} |"
                )
    lines.extend([
        "",
        "## 交付产物",
        "",
        "| 类型 | 路径/任务 |",
        "| --- | --- |",
        f"| 原生模型 | `{_md_cell(artifacts.get('native_model_path') or '-')}` |",
        f"| PMML | `{_md_cell(artifacts.get('pmml_path') or '-')}` |",
        f"| 验证任务 | `{_md_cell(artifacts.get('validation_task_id') or '-')}` |",
        f"| Challenger/Backtest任务 | `{_md_cell(artifacts.get('challenger_task_id') or '-')}` |",
        f"| Challenger/Backtest包 | `{_md_cell(artifacts.get('challenger_package_markdown_path') or artifacts.get('challenger_package_path') or '-')}` |",
        f"| 模型卡 | `{_md_cell(artifacts.get('model_card_markdown_path') or artifacts.get('model_card_path') or '-')}` |",
        f"| Champion对比 | `{_md_cell(artifacts.get('challenger_comparison_markdown_path') or artifacts.get('challenger_comparison_path') or '-')}` |",
    ])
    if actions:
        lines.extend(["", "## 交付动作", "", "| 动作 | 状态 | 说明 |", "| --- | --- | --- |"])
        for item in actions:
            lines.append(
                f"| {_md_cell(item.get('action') or '-')} | "
                f"{_md_cell(item.get('status') or '-')} | "
                f"{_md_cell(item.get('reason') or item.get('pmml_path') or item.get('validation_task_id') or item.get('challenger_task_id') or '-')} |"
            )
    lines.extend(["", "## 入模特征", ""])
    preview = features[:50]
    if preview:
        for feature in preview:
            lines.append(f"- `{_md_inline(feature)}`")
        if len(features) > len(preview):
            lines.append(f"- ... 另有 {len(features) - len(preview)} 个特征")
    else:
        lines.append("- -")
    return "\n".join(lines) + "\n"


def _selection_policy_requirement_markdown_rows(policy) -> list[tuple[str, str]]:
    if not isinstance(policy, dict) or not policy:
        return []
    rows: list[tuple[str, str]] = []
    boolean_labels = {
        "require_pmml": "要求 PMML",
        "require_handoff": "要求验证移交",
        "require_scorecard": "要求评分卡",
        "require_monotonicity": "要求单调性证据",
        "prefer_scorecard": "优先评分卡",
        "allow_policy_override": "允许策略 override",
    }
    for key, label in boolean_labels.items():
        if key in policy:
            rows.append((label, "是" if policy.get(key) else "否"))
    if policy.get("max_feature_count") is not None:
        rows.append(("最大特征数", _metric_display(policy.get("max_feature_count"))))
    if policy.get("max_oot_psi") is not None:
        rows.append(("最大 OOT PSI", _metric_display(policy.get("max_oot_psi"))))
    metric_thresholds = policy.get("metric_thresholds")
    if isinstance(metric_thresholds, dict):
        for metric in sorted(metric_thresholds):
            spec = metric_thresholds.get(metric)
            if not isinstance(spec, dict):
                continue
            parts = []
            if spec.get("min") is not None:
                parts.append(f">= {_metric_display(spec.get('min'))}")
            if spec.get("max") is not None:
                parts.append(f"<= {_metric_display(spec.get('max'))}")
            if parts:
                rows.append((f"指标 {metric}", " 且 ".join(parts)))
    return rows


def _sample_weight_policy_markdown_section(policy: dict, *, heading: str) -> list[str]:
    approval = policy.get("approval_policy") if isinstance(policy.get("approval_policy"), dict) else {}
    monitoring = (
        policy.get("monitoring_defaults")
        if isinstance(policy.get("monitoring_defaults"), dict)
        else {}
    )
    review_items = [str(item) for item in (approval.get("review_items") or []) if str(item)]
    monitor_checks = [
        item for item in (monitoring.get("checks") or [])
        if isinstance(item, dict)
    ]
    lines = [
        "",
        heading,
        "",
        f"- 是否使用: `{_md_inline('是' if policy.get('used') else '否')}`",
        f"- 权重列: `{_md_inline(policy.get('sample_weight_col') or '未使用')}`",
        f"- 来源: `{_md_inline(policy.get('source') or 'none')}`",
        f"- 需要人工复核: `{_md_inline('是' if approval.get('requires_manual_review') else '否')}`",
    ]
    if review_items:
        lines.extend(["", "### 审批复核项", ""])
        for item in review_items:
            lines.append(f"- {_md_inline(item)}")
    if monitor_checks:
        lines.extend([
            "",
            "### 监控默认项",
            "",
            "| 检查项 | 指标 | 状态 | 建议 |",
            "| --- | --- | --- | --- |",
        ])
        for item in monitor_checks:
            lines.append(
                f"| {_md_cell(item.get('id') or '-')} | "
                f"{_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(item.get('status') or '-')} | "
                f"{_md_cell(item.get('recommendation') or '-')} |"
            )
    return lines


def _monitoring_policy_markdown(policy: dict) -> str:
    checks = [item for item in (policy.get("checks") or []) if isinstance(item, dict)]
    sample_weight = (
        policy.get("sample_weight_policy")
        if isinstance(policy.get("sample_weight_policy"), dict)
        else {}
    )
    lines = [
        "# 模型监控策略",
        "",
        "## 基本信息",
        "",
        f"- 策略版本: `{_md_inline(policy.get('policy_version'))}`",
        f"- 状态: `{_md_inline(policy.get('status'))}`",
        f"- 建议: {_md_inline(policy.get('recommendation') or '-')}",
        f"- 实验ID: `{_md_inline(policy.get('experiment_id'))}`",
        f"- 产物ID: `{_md_inline(policy.get('artifact_id'))}`",
        f"- 目标类型: `{_md_inline(policy.get('target_type'))}`",
        f"- 复核频率: `{_md_inline(policy.get('review_cadence'))}`",
        f"- Owner: `{_md_inline(policy.get('owner'))}`",
        "",
        "## 阈值检查",
        "",
        "| 检查项 | 指标 | 状态 | 当前值 | 阈值 | 说明 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    if checks:
        for item in checks:
            lines.append(
                f"| {_md_cell(item.get('label') or item.get('id') or '-')} | "
                f"{_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(item.get('status') or '-')} | "
                f"{_md_cell(_metric_display(item.get('value')))} | "
                f"{_md_cell(_monitoring_threshold_display(item))} | "
                f"{_md_cell(item.get('message') or '-')} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")
    if sample_weight:
        lines.extend(_sample_weight_policy_markdown_section(sample_weight, heading="## 样本权重监控"))
    if policy.get("notes"):
        lines.extend(["", "## 备注", "", str(policy.get("notes"))])
    return "\n".join(lines) + "\n"


def _model_card_markdown(card: dict) -> str:
    key_metrics = [
        item for item in (card.get("key_metrics") or [])
        if isinstance(item, dict)
    ]
    governance = card.get("governance") if isinstance(card.get("governance"), dict) else {}
    selection_requirements = [
        item for item in (governance.get("selection_policy_requirements") or [])
        if isinstance(item, dict)
    ]
    selection_violations = [
        item for item in (governance.get("selection_policy_violations") or [])
        if isinstance(item, dict)
    ]
    delivery = card.get("delivery") if isinstance(card.get("delivery"), dict) else {}
    calibration = delivery.get("calibration") if isinstance(delivery.get("calibration"), dict) else {}
    training = card.get("training") if isinstance(card.get("training"), dict) else {}
    sample_weight = (
        training.get("sample_weight")
        if isinstance(training.get("sample_weight"), dict)
        else {}
    )
    limitations = [str(item) for item in (card.get("limitations") or []) if str(item)]
    review_actions = [str(item) for item in (card.get("next_review_actions") or []) if str(item)]
    feature_preview = [str(item) for item in (card.get("feature_preview") or []) if str(item)]
    lines = [
        "# 模型卡",
        "",
        "## 基本信息",
        "",
        f"- 模型卡版本: `{_md_inline(card.get('card_version'))}`",
        f"- 实验ID: `{_md_inline(card.get('experiment_id'))}`",
        f"- 产物ID: `{_md_inline(card.get('artifact_id'))}`",
        f"- 算法: `{_md_inline(card.get('algorithm'))}`",
        f"- 目标类型: `{_md_inline(card.get('target_type'))}`",
        f"- 目标列: `{_md_inline(card.get('target_col'))}`",
        f"- 样本集: `{_md_inline(card.get('sample_dataset_id') or card.get('dataset_id'))}`",
        f"- 特征数: {_md_inline(card.get('feature_count'))}",
        f"- 样本权重: `{_md_inline(card.get('sample_weight_col') or '未使用')}`",
        f"- 概率校准: `{_md_inline(calibration.get('method') if calibration else '未校准')}`",
        "",
        "## 关键指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    if key_metrics:
        for item in key_metrics:
            lines.append(
                f"| {_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(_metric_display(item.get('value')))} |"
            )
    else:
        lines.append("| - | - |")
    if sample_weight:
        lines.extend(_sample_weight_policy_markdown_section(sample_weight, heading="## 样本权重治理"))
    lines.extend([
        "",
        "## 治理状态",
        "",
        f"- 选择策略: `{_md_inline(governance.get('selection_policy_status') or 'not_requested')}`",
        f"- 监控策略: `{_md_inline(governance.get('monitoring_status') or 'not_configured')}`",
        f"- Champion对比: `{_md_inline(governance.get('champion_comparison_status') or 'not_configured')}`",
        f"- Override原因: {_md_inline(governance.get('selection_policy_override_reason') or '-')}",
        f"- 监控建议: {_md_inline(governance.get('monitoring_recommendation') or '-')}",
        f"- 对比建议: {_md_inline(governance.get('champion_comparison_recommendation') or '-')}",
    ])
    if selection_requirements:
        lines.extend([
            "",
            "### 选择策略要求",
            "",
            "| 策略要求 | 配置 |",
            "| --- | --- |",
        ])
        for item in selection_requirements:
            lines.append(
                f"| {_md_cell(item.get('requirement') or '-')} | "
                f"{_md_cell(item.get('configured') or '-')} |"
            )
    if selection_violations:
        lines.extend([
            "",
            "### 选择策略违规项",
            "",
            "| 违规项 | 说明 |",
            "| --- | --- |",
        ])
        for item in selection_violations:
            lines.append(
                f"| {_md_cell(item.get('code') or '-')} | "
                f"{_md_cell(item.get('message') or '-')} |"
            )
    lines.extend([
        "",
        "## 交付状态",
        "",
        "| 产物/动作 | 状态或路径 |",
        "| --- | --- |",
        f"| 原生模型 | `{_md_cell(delivery.get('native_model_path') or '-')}` |",
        f"| PMML | `{_md_cell(delivery.get('pmml_path') or delivery.get('export_pmml_status') or '-')}` |",
        f"| PMML包含校准 | `{_md_cell(delivery.get('pmml_includes_calibration'))}` |",
        f"| 验证移交 | `{_md_cell(delivery.get('validation_task_id') or delivery.get('validation_handoff_status') or '-')}` |",
        f"| Challenger/Backtest | `{_md_cell(delivery.get('challenger_task_id') or delivery.get('challenger_backtest_status') or '-')}` |",
    ])
    lines.extend(["", "## 限制与复核", ""])
    if limitations:
        for item in limitations:
            lines.append(f"- {_md_inline(item)}")
    else:
        lines.append("- 暂无已记录限制")
    lines.extend(["", "## 后续动作", ""])
    if review_actions:
        for item in review_actions:
            lines.append(f"- {_md_inline(item)}")
    else:
        lines.append("- -")
    lines.extend(["", "## 特征预览", ""])
    if feature_preview:
        for feature in feature_preview:
            lines.append(f"- `{_md_inline(feature)}`")
    else:
        lines.append("- -")
    return "\n".join(lines) + "\n"


def _challenger_comparison_markdown(comparison: dict) -> str:
    champion = comparison.get("champion") if isinstance(comparison.get("champion"), dict) else {}
    challenger = (
        comparison.get("challenger")
        if isinstance(comparison.get("challenger"), dict)
        else {}
    )
    summary = comparison.get("summary") if isinstance(comparison.get("summary"), dict) else {}
    rows = [
        item for item in (comparison.get("metric_comparisons") or [])
        if isinstance(item, dict)
    ]
    lines = [
        "# Champion / Challenger 对比",
        "",
        "## 基本信息",
        "",
        f"- 对比版本: `{_md_inline(comparison.get('comparison_version'))}`",
        f"- 状态: `{_md_inline(comparison.get('status'))}`",
        f"- 建议: {_md_inline(comparison.get('recommendation') or '-')}",
        f"- Champion: `{_md_inline(champion.get('label') or 'prior_champion')}`",
        f"- Champion实验: `{_md_inline(champion.get('experiment_id') or '-')}`",
        f"- Challenger实验: `{_md_inline(challenger.get('experiment_id') or comparison.get('experiment_id'))}`",
        f"- Challenger产物: `{_md_inline(challenger.get('artifact_id') or comparison.get('artifact_id'))}`",
        "",
        "## 汇总",
        "",
        f"- 可比指标: {_md_inline(summary.get('comparable_metric_count') or 0)}/{_md_inline(summary.get('metric_count') or 0)}",
        f"- 优于Champion: {_md_inline(summary.get('improved_count') or 0)}",
        f"- 弱于Champion: {_md_inline(summary.get('declined_count') or 0)}",
        f"- 持平: {_md_inline(summary.get('same_count') or 0)}",
        "",
        "## 指标差异",
        "",
        "| 指标 | Champion | Challenger | 差异 | 趋势 | 结论 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    if rows:
        for item in rows:
            lines.append(
                f"| {_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(_metric_display(item.get('champion_value')))} | "
                f"{_md_cell(_metric_display(item.get('challenger_value')))} | "
                f"{_md_cell(_metric_display(item.get('delta')))} | "
                f"{_md_cell(item.get('direction') or '-')} | "
                f"{_md_cell(item.get('verdict') or '-')} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")
    if champion.get("notes"):
        lines.extend(["", "## Champion备注", "", _md_inline(champion.get("notes"))])
    return "\n".join(lines) + "\n"


def _monitoring_threshold_display(item: dict) -> str:
    direction = str(item.get("direction") or "max")
    warn = item.get("warn")
    fail = item.get("fail")
    prefix = "<=" if direction != "min" else ">="
    if warn is None and fail is None:
        return "需配置"
    parts = []
    if warn is not None:
        parts.append(f"warn {prefix} {_metric_display(warn)}")
    if fail is not None:
        parts.append(f"fail {prefix} {_metric_display(fail)}")
    return "; ".join(parts)


def _metric_display(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return _format_number_token(float(value))
    return "-" if value is None else str(value)


def _md_inline(value) -> str:
    return str(value if value is not None else "-").replace("`", "'")


def _md_cell(value) -> str:
    return _md_inline(value).replace("|", "\\|").replace("\n", " ")


def _artifact_capabilities(artifact: ModelArtifact, *, base_dir: Path | None = None) -> dict:
    pmml_supported, payload_reason = _pmml_payload_support(artifact, base_dir=base_dir)
    calibration = _artifact_calibration_for_capabilities(artifact)
    reason = None if pmml_supported else _unsupported_pmml_reason(artifact, payload_reason)
    limitations = _artifact_delivery_limitations(
        artifact,
        pmml_supported=pmml_supported,
        unsupported_reason=reason,
        calibration=calibration,
    )
    return {
        "pmml_supported": pmml_supported,
        "handoff_supported": pmml_supported,
        "native_model_supported": True,
        "reason": reason,
        "calibrated": bool(calibration),
        "calibration": calibration,
        "pmml_includes_calibration": (
            bool(calibration.get("pmml_includes_calibration"))
            if calibration
            else True
        ),
        "limitations": limitations,
    }


def _unsupported_pmml_reason(artifact: ModelArtifact, payload_reason: str | None) -> str:
    if payload_reason:
        return payload_reason
    if artifact.algorithm == "catboost":
        return (
            "CatBoost 可保留原生 .pkl 模型和报告;当前 sklearn2pmml/JPMML "
            "不支持 CatBoostClassifier 直接导出 PMML,因此验证移交需使用 lr/lgb/xgb/scorecard。"
        )
    if artifact.algorithm == "ensemble":
        return (
            "模型融合(seed-bagging/blend)由多个异构成员模型概率加权平均得到,"
            "不存在单一 PMML 管线可表达该融合逻辑;PMML 导出/验证移交明确不支持,"
            "可保留原生成员模型列表(members/weights)与报告用于内部复现。"
        )
    return (
        f"当前 PMML 导出/验证移交支持 lr/lgb/xgb/scorecard;"
        f"{artifact.algorithm} 可保留原生模型文件和报告。"
    )


def _artifact_calibration_for_capabilities(artifact: ModelArtifact) -> dict:
    calibration = _artifact_calibration_metadata(artifact)
    if not calibration:
        return {}
    keys = (
        "method",
        "split",
        "split_value",
        "sample_count",
        "positive_count",
        "brier_raw",
        "brier_calibrated",
        "ece_raw",
        "ece_calibrated",
        "pmml_includes_calibration",
        "path",
    )
    return _json_safe({key: calibration.get(key) for key in keys if key in calibration}) or {}


def _artifact_delivery_limitations(
    artifact: ModelArtifact,
    *,
    pmml_supported: bool,
    unsupported_reason: str | None,
    calibration: dict,
) -> list[str]:
    limitations: list[str] = []
    if not pmml_supported:
        limitations.append(unsupported_reason or _unsupported_pmml_reason(artifact, None))
    if calibration and calibration.get("pmml_includes_calibration") is False:
        method = str(calibration.get("method") or "unknown")
        limitations.append(
            f"模型已进行 {method} 概率校准，但 PMML 产物不包含校准器；"
            "验证移交 Notebook 会加载 calibration.joblib 应用校准。"
        )
    if (artifact.params or {}).get("preprocessing_steps"):
        # PREP-2: PMML export intentionally does not embed the generic feature-pack
        # preprocessing layer (impute/cap/normalize/onehot) — out of scope for this
        # fix; only the platform scorer/handoff notebook replay it. Mirrors the
        # calibration limitation above so the model card/capabilities never imply
        # PMML alone reproduces this model's scores on new raw data.
        limitations.append(
            "模型训练前经过 impute/cap/normalize/onehot 等预处理；PMML 产物不包含该预处理层，"
            "对新数据打分须使用平台内 scorer 或验证移交 Notebook（会自动重放预处理链）。"
        )
    elif (artifact.params or {}).get("preprocessing_chain_traceable") is False:
        # PREP-2: the input dataset carried no lineage sidecar at all (predates this
        # mechanism, or was never derived through a chain-tracking FEATURE/
        # prepare_modeling_frame call) — flag explicitly rather than silently implying
        # this model has zero preprocessing.
        limitations.append("预处理链不可追溯：训练数据集无预处理血缘记录，无法确认打分重放是否完整。")
    return _unique_strings(limitations)


def _pmml_delivery_note(capabilities: dict) -> str:
    limitations = [
        str(item)
        for item in capabilities.get("limitations") or []
        if str(item)
    ]
    return " ".join(limitations)


def _handoff_delivery_note(capabilities: dict) -> str:
    calibration = capabilities.get("calibration") if isinstance(capabilities.get("calibration"), dict) else {}
    if calibration and capabilities.get("pmml_includes_calibration") is False:
        return "验证移交 Notebook 会加载 calibration.joblib 应用校准。"
    return ""


def _require_pmml_supported(
    artifact: ModelArtifact,
    *,
    operation: str = "PMML export",
    base_dir: Path | None = None,
) -> None:
    supported, reason = _pmml_payload_support(artifact, base_dir=base_dir)
    if not supported:
        raise ModelingError(
            f"{operation} currently supports lr/lgb/xgb/scorecard only; got: {artifact.algorithm}. "
            f"{reason or 'Use the native model artifact/report, or retrain/export a supported binary model for V1 validation handoff.'}"
        )


def _pmml_payload_support(artifact: ModelArtifact, *, base_dir: Path | None) -> tuple[bool, str | None]:
    if artifact.algorithm not in PMML_SUPPORTED_ALGORITHMS:
        return False, None
    if base_dir is None:
        return True, None
    try:
        model = load_model(artifact, base_dir=base_dir)
    except Exception as exc:
        return False, f"模型文件无法加载,不能导出 PMML:{exc}"
    if artifact.algorithm == "scorecard":
        try:
            validate_scorecard_pmml_payload(model, feature_list=list(artifact.feature_list))
        except ModelingError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"评分卡 PMML 预检失败:{exc}"
        if isinstance(model, dict) and "model" in model and "woe_maps" in model:
            return True, None
        return False, "评分卡 PMML 导出需要包含 model 与 woe_maps 的 scorecard payload。"
    if hasattr(model, "fit") and (hasattr(model, "predict_proba") or hasattr(model, "predict")):
        return True, None
    return False, (
        "当前 PMML 导出仅支持 sklearn 兼容模型对象；原生 LightGBM/XGBoost Booster "
        "请保留原生模型或使用专门 JPMML 导出链路。"
    )


def _pmml_path(runtime: _Runtime, artifact: ModelArtifact) -> Path:
    experiment = runtime.experiments.get(artifact.experiment_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    if artifact.pmml_path:
        existing = _resolve_artifact_path(artifact.pmml_path, base_dir=base_dir)
        if existing.exists():
            persist_model_meta(base_dir, artifact, config=experiment.config)
            return existing
    dataset = runtime.registry.get(experiment.config.dataset_id)
    out_path = base_dir / f"{artifact.id}.pmml"
    pmml_path = export_pmml(
        artifact,
        runtime.registry.resolve_path(dataset.id),
        out_path,
        base_dir=base_dir,
        target_col=experiment.config.target_col,
    )
    try:
        updated_artifact = replace(artifact, pmml_path=pmml_path.name)
        persist_model_meta(base_dir, updated_artifact, config=experiment.config)
        runtime.experiments.set_artifact_pmml_path(artifact.id, pmml_path.name)
    except Exception:
        pmml_path.unlink(missing_ok=True)
        try:
            persist_model_meta(base_dir, artifact, config=experiment.config)
        except Exception:
            pass
        raise
    return pmml_path
