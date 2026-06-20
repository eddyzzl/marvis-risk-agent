from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository
from marvis.feature.metrics import feature_metrics
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.output.model_report import ModelReportPayload, render_model_report
from marvis.packs.modeling.artifact import export_pmml
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.handoff import handoff_to_validation
from marvis.packs.modeling.report_compute import (
    BusinessColumns,
    build_feature_dictionary,
    compute_amount_bin_table,
    compute_sample_analysis,
    compute_vintage_report,
    resolve_report_sections,
    stress_low_pricing,
)
from marvis.packs.modeling.readiness import check_data_quality, modeling_readiness
from marvis.packs.modeling.prepare import prepare_modeling_frame
from marvis.packs.modeling.recipes.lgb import train_lgb
from marvis.packs.modeling.recipes.lgb_regressor import train_lgb_regressor
from marvis.packs.modeling.recipes.lr import train_lr
from marvis.packs.modeling.recipes.scorecard import train_scorecard
from marvis.packs.modeling.recipes.xgb import train_xgb
from marvis.packs.modeling.scenarios import apply_scenario
from marvis.packs.modeling.select import select_features
from marvis.packs.modeling.errors import ModelingError
from marvis.settings import build_settings


MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"


def tool_check_data_quality(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    issues = check_data_quality(
        runtime.backend,
        dataset,
        runtime.registry.resolve_path(dataset.id),
        target_col=_optional_str(inputs.get("target_col")),
    )
    return {"issues": [_jsonable(issue) for issue in issues]}


def tool_modeling_readiness(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    return _jsonable(
        modeling_readiness(
            runtime.backend,
            dataset,
            runtime.registry.resolve_path(dataset.id),
            target_col=str(inputs["target_col"]),
            split_col=_optional_str(inputs.get("split_col")),
        )
    )


def tool_prepare_modeling_frame(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        str(inputs["dataset_id"]),
        target_col=str(inputs["target_col"]),
        feature_cols=[str(item) for item in inputs["feature_cols"]],
        split_col=_optional_str(inputs.get("split_col")),
        split_config=inputs.get("split_config") or {},
        seed=int(inputs.get("seed") if inputs.get("seed") is not None else ctx.seed or 0),
    )
    split_col = _optional_str(inputs.get("split_col")) or "split"
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(result.id), columns=[split_col])
    counts = {
        str(key): int(value)
        for key, value in frame[split_col].value_counts().sort_index().items()
    }
    return {"result_dataset_id": result.id, "split_counts": counts}


def tool_select_features(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    result = select_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=[str(item) for item in inputs["features"]],
        target_col=str(inputs["target_col"]),
        iv_min=float(inputs.get("iv_min", 0.02)),
        corr_max=float(inputs.get("corr_max", 0.8)),
        vif_max=float(inputs.get("vif_max", 10.0)),
        top_k=_optional_int(inputs.get("top_k")),
        seed=int(ctx.seed or 0),
    )
    return {
        "selected": list(result.selected),
        "dropped": [[feature, reason] for feature, reason in result.dropped],
        "scores": _jsonable(result.scores),
    }


def tool_train_model(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    recipe = str(inputs["recipe"])
    config = TrainConfig(
        dataset_id=dataset.id,
        features=tuple(str(item) for item in inputs["features"]),
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]),
        split_values=dict(inputs["split_values"]),
        params=dict(inputs.get("params") or {}),
        seed=int(inputs["seed"]),
        early_stopping_rounds=_optional_int(inputs.get("early_stopping_rounds")),
        recipe_id=recipe,
    )
    if inputs.get("scenario"):
        config = apply_scenario(config, str(inputs["scenario"]))
        recipe = config.recipe_id or recipe

    experiment_id = runtime.experiments.create(ctx.task_id, recipe, config)
    try:
        result = _train_recipe(
            recipe,
            runtime.backend,
            runtime.registry.resolve_path(dataset.id),
            config,
            out_dir=_artifact_base_dir(runtime.settings, ctx.task_id),
        )
        runtime.experiments.attach_result(experiment_id, result)
    except Exception:
        runtime.experiments.set_status(experiment_id, "failed")
        raise

    experiment = runtime.experiments.get(experiment_id)
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact after training: {experiment_id}")
    artifact = runtime.modeling_repo.get_model_artifact(experiment.artifact_id)
    if artifact is None:
        raise ModelingError(f"model artifact not found: {experiment.artifact_id}")
    return {
        "experiment_id": experiment_id,
        "artifact_id": artifact.id,
        "metrics": _jsonable(experiment.metrics),
        "feature_importance": _jsonable(result.feature_importance),
    }


def tool_compare_experiments(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    return _jsonable(runtime.experiments.compare([str(item) for item in inputs["experiment_ids"]]))


def tool_export_pmml(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    pmml_path = _pmml_path(runtime, artifact)
    return {"pmml_path": str(pmml_path)}


def tool_handoff_to_validation(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    validation_task_id = handoff_to_validation(
        runtime.experiments,
        artifact,
        sample_dataset_id=str(inputs["sample_dataset_id"]),
        settings=runtime.settings,
    )
    return {"validation_task_id": validation_task_id}


def tool_generate_model_report(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    artifact = _artifact(runtime, experiment.artifact_id) if experiment.artifact_id else None
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    dataset_path = runtime.registry.resolve_path(dataset.id)
    business = _business_columns(inputs.get("business_columns") or {})
    statuses = resolve_report_sections(
        business,
        _optional_str(inputs.get("feature_dictionary_id")),
    )
    sample = None
    if _section_available(statuses, "sample_analysis") and business.loan_month_col:
        sample = compute_sample_analysis(
            runtime.backend,
            dataset_path,
            loan_month_col=business.loan_month_col,
            target_col=experiment.config.target_col,
            business=business,
            mob_cols=business.mob_observe_cols,
        )
    vintage = None
    if _section_available(statuses, "vintage") and business.loan_month_col:
        vintage = compute_vintage_report(
            runtime.backend,
            dataset_path,
            loan_month_col=business.loan_month_col,
            mob_observe_cols=business.mob_observe_cols,
            amount_col=business.loan_amount_col,
        )

    score_col = _report_score_col(runtime, dataset_path, artifact, experiment.config)
    low_pricing = None
    if _section_available(statuses, "low_pricing") and business.interest_rate_col:
        low_pricing = stress_low_pricing(
            runtime.backend,
            dataset_path,
            score_col=score_col,
            target_col=experiment.config.target_col,
            interest_rate_col=business.interest_rate_col,
            low_pricing_threshold=None,
        )
    oot_bin = _report_bin_table(
        runtime,
        dataset_path,
        score_col=score_col,
        target_col=experiment.config.target_col,
        business=business,
    )
    feature_dictionary_id = _optional_str(inputs.get("feature_dictionary_id"))
    feature_dictionary = (
        build_feature_dictionary(runtime.backend, feature_dictionary_id, runtime.registry)
        if feature_dictionary_id
        else {}
    )
    feature_importance = _feature_importance_rows(artifact, feature_dictionary=feature_dictionary)
    structured_summary = _report_structured_summary(
        project_meta=dict(inputs.get("project_meta") or {}),
        dataset_split=_dataset_split_rows(experiment.metrics),
        stability=_stability_rows(experiment.metrics),
        sample_analysis=sample,
        vintage=vintage,
        feature_importance=feature_importance,
        univariate=_univariate_rows(runtime, dataset_path, artifact, experiment.config),
        oot_bin_table=oot_bin,
        stress_product_removal={},
        stress_low_pricing=low_pricing,
        section_status=statuses,
    )
    narratives = _guard_no_invented_numbers(
        _draft_report_narratives(
            structured_summary,
            llm_factory=_report_llm_factory(runtime.settings.workspace, _optional_str(inputs.get("model_id"))),
        ),
        structured_summary,
    )
    report_path = Path(runtime.settings.tasks_dir) / ctx.task_id / "outputs" / "model_report.xlsx"
    render_model_report(
        ModelReportPayload(
            project_meta=structured_summary["project_meta"],
            dataset_split=structured_summary["dataset_split"],
            stability=structured_summary["stability"],
            sample_analysis=sample,
            vintage=vintage,
            feature_importance=structured_summary["feature_importance"],
            univariate=structured_summary["univariate"],
            oot_bin_table=oot_bin,
            stress_product_removal={},
            stress_low_pricing=low_pricing,
            narratives=narratives,
            section_status=statuses,
        ),
        report_path,
    )
    return {
        "report_path": str(report_path),
        "section_status": [_jsonable(status) for status in statuses],
    }


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


def _train_recipe(
    recipe: str,
    backend,
    dataset_path: Path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    if recipe == "lgb":
        return train_lgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "lgb_regressor":
        return train_lgb_regressor(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "xgb":
        return train_xgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "lr":
        return train_lr(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "scorecard":
        return train_scorecard(backend, dataset_path, config, out_dir=out_dir)
    raise ModelingError(f"unsupported modeling recipe: {recipe}")


def _artifact(runtime: _Runtime, artifact_id: str) -> ModelArtifact:
    artifact = runtime.modeling_repo.get_model_artifact(artifact_id)
    if artifact is None:
        raise ModelingError(f"model artifact not found: {artifact_id}")
    return artifact


def _pmml_path(runtime: _Runtime, artifact: ModelArtifact) -> Path:
    experiment = runtime.experiments.get(artifact.experiment_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    if artifact.pmml_path:
        existing = _resolve_artifact_path(artifact.pmml_path, base_dir=base_dir)
        if existing.exists():
            return existing
    dataset = runtime.registry.get(experiment.config.dataset_id)
    out_path = base_dir / f"{artifact.id}.pmml"
    pmml_path = export_pmml(
        artifact,
        runtime.registry.resolve_path(dataset.id),
        out_path,
        base_dir=base_dir,
    )
    runtime.experiments.set_artifact_pmml_path(artifact.id, pmml_path.name)
    return pmml_path


def _artifact_base_dir(settings, task_id: str) -> Path:
    return Path(settings.tasks_dir) / task_id / MODELING_ARTIFACTS_DIR_NAME


def _resolve_artifact_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _business_columns(payload: dict) -> BusinessColumns:
    return BusinessColumns(
        loan_month_col=_optional_str(payload.get("loan_month_col")),
        interest_rate_col=_optional_str(payload.get("interest_rate_col")),
        loan_amount_col=_optional_str(payload.get("loan_amount_col")),
        term_col=_optional_str(payload.get("term_col")),
        drawdown_amount_col=_optional_str(payload.get("drawdown_amount_col")),
        credit_limit_col=_optional_str(payload.get("credit_limit_col")),
        mob_observe_cols=tuple(str(item) for item in payload.get("mob_observe_cols") or ()),
    )


def _section_available(statuses, section: str) -> bool:
    return any(status.section == section and status.available for status in statuses)


def _dataset_split_rows(metrics) -> list[dict]:
    if metrics is None:
        return []
    if metrics.train_rmse is not None:
        return [
            {
                "split": "train",
                "rmse": metrics.train_rmse,
                "mae": metrics.train_mae,
                "r2": metrics.train_r2,
            },
            {
                "split": "test",
                "rmse": metrics.test_rmse,
                "mae": metrics.test_mae,
                "r2": metrics.test_r2,
            },
            {
                "split": "oot",
                "rmse": metrics.oot_rmse,
                "mae": metrics.oot_mae,
                "r2": metrics.oot_r2,
            },
        ]
    return [
        {"split": "train", "ks": metrics.train_ks, "auc": metrics.train_auc},
        {"split": "test", "ks": metrics.test_ks, "auc": metrics.test_auc},
        {"split": "oot", "ks": metrics.oot_ks, "auc": metrics.oot_auc},
    ]


def _stability_rows(metrics) -> list[dict]:
    if metrics is None:
        return []
    if metrics.train_rmse is not None:
        return [
            {"metric": "rmse_test_minus_train", "value": metrics.overfit_train_test_gap},
            {"metric": "rmse_oot_minus_train", "value": metrics.overfit_train_oot_gap},
            {"metric": "overfit_flag", "value": metrics.overfit_flag},
        ]
    return [
        {"metric": "psi_test_vs_train", "value": metrics.psi_test_vs_train},
        {"metric": "psi_oot_vs_train", "value": metrics.psi_oot_vs_train},
        {"metric": "overfit_flag", "value": metrics.overfit_flag},
    ]


def _feature_importance_rows(artifact: ModelArtifact | None, *, feature_dictionary: dict | None = None) -> list[dict]:
    if artifact is None:
        return []
    dictionary = feature_dictionary or {}
    metadata_keys = ("含义", "产品名称", "厂商名称")
    rows = []
    for feature in artifact.feature_list:
        row = {"feature": feature, "importance": 0.0}
        if dictionary:
            metadata = dictionary.get(str(feature))
            row.update({
                key: metadata.get(key) if isinstance(metadata, dict) and metadata.get(key) not in ("",) else None
                for key in metadata_keys
            })
        rows.append(row)
    return rows


def _univariate_rows(runtime: _Runtime, dataset_path: Path, artifact, config: TrainConfig) -> list[dict]:
    if artifact is None:
        return []
    frame = runtime.backend.read_frame(dataset_path, columns=[*artifact.feature_list, config.target_col])
    rows = []
    for feature in artifact.feature_list:
        metrics = feature_metrics(
            frame[feature].to_numpy(dtype=float),
            frame[config.target_col].to_numpy(dtype=int),
            feature=feature,
        )
        rows.append({"feature": feature, "iv": metrics.iv, "ks": metrics.ks})
    return rows


def _report_score_col(runtime: _Runtime, dataset_path: Path, artifact, config: TrainConfig) -> str:
    columns = runtime.backend.column_names(dataset_path)
    if "score" in columns:
        return "score"
    if artifact and artifact.feature_list:
        return artifact.feature_list[0]
    return config.features[0]


def _report_bin_table(
    runtime: _Runtime,
    dataset_path: Path,
    *,
    score_col: str,
    target_col: str,
    business: BusinessColumns,
) -> list[dict]:
    frame = runtime.backend.read_frame(dataset_path, columns=[score_col])
    from marvis.validation.binning import equal_frequency_bin_edges

    edges = equal_frequency_bin_edges(frame[score_col].to_numpy(dtype=float), 10)
    return compute_amount_bin_table(
        runtime.backend,
        dataset_path,
        score_col=score_col,
        target_col=target_col,
        edges=edges,
        business=business,
    )


def _report_structured_summary(**payload) -> dict:
    return _jsonable(payload)


REPORT_NARRATIVE_SYS = (
    "你为信贷风控建模报告起草章节文字。只能解释用户提供的结构化摘要，"
    "不得编造任何数字、百分比、阈值、金额或样本量。输出 JSON object。"
)
REPORT_NARRATIVE_KEYS = ("sample", "vintage", "model", "stress")
REPORT_NUMERIC_EVIDENCE_KEYS = (
    "dataset_split",
    "stability",
    "sample_analysis",
    "vintage",
    "feature_importance",
    "univariate",
    "oot_bin_table",
    "stress_product_removal",
    "stress_low_pricing",
)


def _draft_report_narratives(structured_summary: dict, *, llm_factory=None) -> dict:
    fallback = _fallback_report_narratives()
    if llm_factory is None:
        return fallback
    try:
        raw = llm_factory().complete(
            system_prompt=REPORT_NARRATIVE_SYS,
            user_prompt=_report_narrative_prompt(structured_summary),
            response_format={"type": "json_object"},
            stream=False,
        )
        payload = json.loads(str(raw))
    except (LLMClientError, LLMSettingsError, json.JSONDecodeError, TypeError, ValueError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return {
        key: str(payload.get(key) or fallback[key])
        for key in REPORT_NARRATIVE_KEYS
    }


def _fallback_report_narratives() -> dict:
    return {
        "sample": "样本分析基于平台聚合结果生成。",
        "vintage": "Vintage 结论基于平台计算曲线生成。",
        "model": "模型结论基于平台指标与特征重要性生成。",
        "stress": "压力测试结论基于平台压测结果生成。",
    }


def _report_narrative_prompt(structured_summary: dict) -> str:
    return (
        "请基于以下结构化摘要，输出 JSON："
        "{sample, vintage, model, stress}。\n"
        "要求：只写文字解释；所有数字必须来自摘要原文；缺少数据时说明缺业务数据。\n\n"
        f"结构化摘要：\n{json.dumps(structured_summary, ensure_ascii=False, sort_keys=True)}"
    )


def _report_llm_factory(workspace: Path, model_id: str | None):
    def factory():
        return OpenAICompatibleLLMClient(resolve_llm_model(workspace, model_id))

    return factory


_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")


def _guard_no_invented_numbers(narratives: dict, structured_summary: dict) -> dict:
    allowed = _allowed_number_tokens(_report_numeric_evidence(structured_summary))
    guarded: dict[str, str] = {}
    for key, value in narratives.items():
        text = str(value)
        guarded[str(key)] = _NUMBER_TOKEN_RE.sub(
            lambda match: match.group(0) if _number_token_allowed(match.group(0), allowed) else "[平台未提供该数字]",
            text,
        )
    return guarded


def _report_numeric_evidence(structured_summary: dict) -> dict:
    return {
        key: structured_summary.get(key)
        for key in REPORT_NUMERIC_EVIDENCE_KEYS
        if key in structured_summary
    }


def _allowed_number_tokens(value) -> set[str]:
    tokens: set[str] = set()

    def visit(item) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            numeric = float(item)
            tokens.add(_format_number_token(numeric))
            tokens.add(str(item))
            return
        if isinstance(item, str):
            for match in _NUMBER_TOKEN_RE.finditer(item):
                tokens.add(match.group(0))

    visit(value)
    return {token for token in tokens if token}


def _number_token_allowed(token: str, allowed: set[str]) -> bool:
    if token in allowed:
        return True
    if token.endswith("%"):
        return False
    try:
        numeric = float(token)
    except ValueError:
        return False
    return _format_number_token(numeric) in allowed


def _format_number_token(value: float) -> str:
    return f"{value:.12g}"


def _jsonable(value: Any):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "tool_check_data_quality",
    "tool_compare_experiments",
    "tool_export_pmml",
    "tool_handoff_to_validation",
    "tool_generate_model_report",
    "tool_modeling_readiness",
    "tool_prepare_modeling_frame",
    "tool_select_features",
    "tool_train_model",
]
