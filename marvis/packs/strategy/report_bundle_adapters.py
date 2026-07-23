"""Pure projections from authenticated V2 evidence into report inputs.

This module intentionally has no persistence or runtime dependency.  Every
input must already be a typed artifact binding produced by a governed loader.
The adapter revalidates the canonical evidence carried by those bindings and
copies only aggregate/structural facts into ``StrategyReportBundle`` fields and
tables.  It never reads raw rows, scores a model, evaluates a rule, or upgrades
development evidence to an independent-validation or production claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

from marvis.packs.modeling.evidence import (
    canonical_modeling_training_evidence_json,
    validate_modeling_training_evidence,
)
from marvis.packs.modeling.evidence_tools import (
    ModelingTrainingEvidenceArtifactBinding,
    build_training_evidence_ref,
)
from marvis.packs.modeling.score_evidence import (
    canonical_model_score_evidence_json,
    validate_model_score_evidence_envelope,
)
from marvis.packs.modeling.score_evidence_tools import (
    ModelScoreEvidenceArtifactBinding,
)
from marvis.packs.strategy.model_evidence import (
    canonical_strategy_model_evidence_bundle_json,
    validate_strategy_model_evidence_bundle,
)
from marvis.packs.strategy.model_evidence_tools import (
    StrategyModelEvidenceV2ArtifactBinding,
)
from marvis.packs.strategy.pool import (
    canonical_strategy_pool_json,
    compile_strategy_pool,
    validate_strategy_pool,
)
from marvis.packs.strategy.pool_impact import (
    canonical_strategy_pool_impact_json,
    validate_strategy_pool_impact_assessment,
)
from marvis.packs.strategy.pool_impact_tools import (
    StrategyPoolImpactArtifactBinding,
)
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
)
from marvis.packs.strategy.project_context import (
    build_report_field,
    build_source_ref,
    canonical_strategy_project_context_revision_json,
    validate_report_field,
    validate_strategy_project_context_revision,
)
from marvis.packs.strategy.project_context_tools import (
    StrategyProjectContextArtifactBinding,
)
from marvis.packs.strategy.report_bundle import (
    REPORT_SECTION_KEYS,
    StrategyReportBundleError,
    build_named_report_field,
    build_strategy_report_section,
    build_strategy_report_table,
)
from marvis.packs.strategy.sample_design_v2 import (
    canonical_strategy_sample_design_v2_bundle_json,
    validate_strategy_sample_design_v2_bundle,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
)


_SECTION_TITLES = {
    "current_project": "当前项目状况",
    "historical_versions": "历史版本策略效果",
    "sample_design": "本次样本设计",
    "univariate_and_models": "单变量与模型分析",
    "candidate_combinations": "候选组合与策略设计",
    "impact_assessment": "策略影响测算",
    "final_document": "最终策略结论",
}
_SAMPLE_STATUS_TO_AVAILABILITY = {
    "present": "present",
    "unavailable": "unavailable",
    "insufficient_data": "unavailable",
    "not_matured": "not_matured",
    "not_applicable": "not_applicable",
}
_MODEL_STATUS_TO_AVAILABILITY = {
    "present": "present",
    "unavailable": "unavailable",
    "not_matured": "not_matured",
    "not_applicable": "not_applicable",
}


def build_strategy_report_bundle_source_inputs(
    *,
    project_context: StrategyProjectContextArtifactBinding,
    sample_design: StrategySampleDesignV2ArtifactBinding,
    candidate_pool: StrategyCandidatePoolArtifactBinding,
    pool_impact: StrategyPoolImpactArtifactBinding,
    model_evidence: StrategyModelEvidenceV2ArtifactBinding | None = None,
    training_evidence: ModelingTrainingEvidenceArtifactBinding | None = None,
    score_evidence: ModelScoreEvidenceArtifactBinding | None = None,
) -> dict[str, Any]:
    """Build deterministic evidence-only kwargs for ``build_strategy_report_bundle``.

    The caller still owns title, report status/revision, generated time, and any
    independently governed strategy identity.  The returned mapping contains
    only the seven sections plus the four source inventories accepted by the
    report contract.
    """

    project = _authenticated_project_context(project_context)
    task_id = project_context.task_id
    _require_same_task(
        task_id,
        sample_design=sample_design,
        candidate_pool=candidate_pool,
        pool_impact=pool_impact,
        model_evidence=model_evidence,
        training_evidence=training_evidence,
        score_evidence=score_evidence,
    )
    sample = _authenticated_sample_design(sample_design)
    pool, design = _authenticated_candidate_pool(candidate_pool)
    impact = _authenticated_pool_impact(pool_impact)
    effective_training = _effective_training_binding(
        training_evidence=training_evidence,
        score_evidence=score_evidence,
    )
    training = (
        None
        if effective_training is None
        else _authenticated_training_evidence(effective_training, sample)
    )
    model = (
        None
        if model_evidence is None
        else _authenticated_model_evidence(model_evidence, sample)
    )
    score = (
        None
        if score_evidence is None
        else _authenticated_score_evidence(
            score_evidence,
            sample=sample,
            training_binding=effective_training,
            training=training,
        )
    )
    _require_same_task(
        task_id,
        training_evidence=effective_training,
    )
    _require_sample_identity(
        sample_design=sample_design,
        sample=sample,
        model_evidence=model_evidence,
        training_evidence=effective_training,
        score_evidence=score_evidence,
    )
    _require_pool_impact_identity(
        sample=sample,
        pool_binding=candidate_pool,
        pool=pool,
        compiled_design=design,
        impact_binding=pool_impact,
        impact=impact,
    )

    refs = _EvidenceRefs(
        project=_artifact_ref(
            "strategy_project_context",
            project_context.artifact_id,
            project_context.artifact_content_hash,
        ),
        sample=_artifact_ref(
            "sample_design",
            sample_design.bundle_artifact_id,
            sample_design.bundle_artifact_content_hash,
        ),
        pool=_artifact_ref(
            "strategy_candidate_pool",
            candidate_pool.artifact_id,
            candidate_pool.artifact_content_hash,
        ),
        impact=_artifact_ref(
            "pool_impact",
            pool_impact.artifact_id,
            pool_impact.artifact_content_hash,
        ),
        model=(
            None
            if model_evidence is None
            else _artifact_ref(
                "strategy_model_evidence",
                model_evidence.artifact_id,
                model_evidence.artifact_content_hash,
            )
        ),
        training=(
            None
            if effective_training is None
            else _artifact_ref(
                "modeling_training_evidence",
                _record_text(effective_training.evidence_record, "id"),
                _record_text(effective_training.evidence_record, "content_hash"),
            )
        ),
        score=(
            None
            if score_evidence is None
            else _artifact_ref(
                "model_score_evidence",
                _record_text(score_evidence.evidence_record, "id"),
                _record_text(score_evidence.evidence_record, "content_hash"),
            )
        ),
    )
    sections_by_key = {
        "current_project": _current_project_section(project, refs.project),
        "historical_versions": _history_section(project, refs.project),
        "sample_design": _sample_section(sample, refs.sample),
        "univariate_and_models": _model_section(
            model=model,
            training=training,
            score=score,
            refs=refs,
        ),
        "candidate_combinations": _candidate_section(
            pool=pool,
            compiled_design=design,
            source_ref=refs.pool,
        ),
        "impact_assessment": _impact_section(
            impact=impact,
            pool_ref=refs.pool,
            impact_ref=refs.impact,
        ),
        "final_document": _final_document_section(
            pool=pool,
            compiled_design=design,
            impact=impact,
            pool_ref=refs.pool,
            impact_ref=refs.impact,
        ),
    }
    state = project["state"]
    snapshot = state["current_project_snapshot"]
    dataset_refs = _dedupe_refs(
        [
            *snapshot["dataset_refs"],
            _dataset_ref_from_sample(sample),
            _dataset_ref_from_impact(impact),
        ]
    )
    tool_run_refs = _dedupe_refs(
        [
            *snapshot["tool_run_refs"],
            *(
                ref
                for history in state["historical_strategy_reviews"]
                for ref in history["tool_run_refs"]
            ),
        ]
    )
    return {
        "sections": [
            sections_by_key[key]
            for key in REPORT_SECTION_KEYS
        ],
        "dataset_refs": dataset_refs,
        "strategy_artifact_refs": _dedupe_refs([refs.pool, refs.impact]),
        "tool_run_refs": tool_run_refs,
        "missing_information": list(state["missing_information_records"]),
    }


class _EvidenceRefs:
    __slots__ = (
        "project",
        "sample",
        "pool",
        "impact",
        "model",
        "training",
        "score",
    )

    def __init__(
        self,
        *,
        project: dict[str, str],
        sample: dict[str, str],
        pool: dict[str, str],
        impact: dict[str, str],
        model: dict[str, str] | None,
        training: dict[str, str] | None,
        score: dict[str, str] | None,
    ) -> None:
        self.project = project
        self.sample = sample
        self.pool = pool
        self.impact = impact
        self.model = model
        self.training = training
        self.score = score


def _authenticated_project_context(
    binding: StrategyProjectContextArtifactBinding,
) -> dict[str, Any]:
    _require_binding_type(
        binding,
        StrategyProjectContextArtifactBinding,
        "project-context",
    )
    revision = validate_strategy_project_context_revision(binding.revision)
    if revision != binding.revision or revision["task_id"] != binding.task_id:
        raise StrategyReportBundleError(
            "project-context binding identity changed"
        )
    _require_canonical_artifact_hash(
        binding.artifact_content_hash,
        canonical_strategy_project_context_revision_json(revision),
        "project-context",
    )
    return revision


def _authenticated_sample_design(
    binding: StrategySampleDesignV2ArtifactBinding,
) -> dict[str, Any]:
    _require_binding_type(
        binding,
        StrategySampleDesignV2ArtifactBinding,
        "sample-design V2",
    )
    bundle = validate_strategy_sample_design_v2_bundle(binding.bundle)
    if bundle != binding.bundle:
        raise StrategyReportBundleError(
            "sample-design V2 binding content changed"
        )
    identity = bundle["sample_design"]["identity"]
    if identity["task_id"] != binding.task_id:
        raise StrategyReportBundleError(
            "sample-design V2 binding belongs to another task"
        )
    _require_canonical_artifact_hash(
        binding.bundle_artifact_content_hash,
        canonical_strategy_sample_design_v2_bundle_json(bundle),
        "sample-design V2",
    )
    return bundle


def _authenticated_candidate_pool(
    binding: StrategyCandidatePoolArtifactBinding,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_binding_type(
        binding,
        StrategyCandidatePoolArtifactBinding,
        "candidate-pool",
    )
    pool = validate_strategy_pool(binding.pool)
    if pool != binding.pool or pool["task_id"] != binding.task_id:
        raise StrategyReportBundleError(
            "candidate-pool binding identity changed"
        )
    compiled = compile_strategy_pool(pool)
    if compiled != binding.compiled_design:
        raise StrategyReportBundleError(
            "candidate-pool compiled design changed"
        )
    if (
        pool["strategy_type"] != binding.strategy_type
        or pool["status"] != "draft"
        or pool["validation_status"] != "unvalidated"
    ):
        raise StrategyReportBundleError(
            "candidate-pool must remain draft and unvalidated"
        )
    _require_canonical_artifact_hash(
        binding.artifact_content_hash,
        canonical_strategy_pool_json(pool),
        "candidate-pool",
    )
    return pool, compiled


def _authenticated_pool_impact(
    binding: StrategyPoolImpactArtifactBinding,
) -> dict[str, Any]:
    _require_binding_type(
        binding,
        StrategyPoolImpactArtifactBinding,
        "pool-impact",
    )
    impact = validate_strategy_pool_impact_assessment(binding.assessment)
    lifecycle = impact["lifecycle"]
    if (
        impact != binding.assessment
        or impact["identity"]["task_id"] != binding.task_id
        or binding.stage != "development_backtest"
        or binding.validation_status != "unvalidated"
        or lifecycle
        != {
            "candidate_stage": "development",
            "observation_stage": "backtested",
            "validation_status": "unvalidated",
            "creates_strategy": False,
            "adopted": False,
            "deployed": False,
        }
    ):
        raise StrategyReportBundleError(
            "pool-impact must remain development, backtested, and unvalidated"
        )
    _require_canonical_artifact_hash(
        binding.artifact_content_hash,
        canonical_strategy_pool_impact_json(impact),
        "pool-impact",
    )
    return impact


def _authenticated_model_evidence(
    binding: StrategyModelEvidenceV2ArtifactBinding,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    _require_binding_type(
        binding,
        StrategyModelEvidenceV2ArtifactBinding,
        "model-evidence V2",
    )
    bundle = validate_strategy_model_evidence_bundle(
        binding.bundle,
        sample_design_bundle=sample,
    )
    if bundle != binding.bundle or bundle["sample_design_binding"]["task_id"] != binding.task_id:
        raise StrategyReportBundleError(
            "model-evidence V2 binding identity changed"
        )
    _require_canonical_artifact_hash(
        binding.artifact_content_hash,
        canonical_strategy_model_evidence_bundle_json(
            bundle,
            sample_design_bundle=sample,
        ),
        "model-evidence V2",
    )
    return bundle


def _authenticated_training_evidence(
    binding: ModelingTrainingEvidenceArtifactBinding,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    _require_binding_type(
        binding,
        ModelingTrainingEvidenceArtifactBinding,
        "training-evidence",
    )
    evidence = validate_modeling_training_evidence(
        binding.evidence,
        sample_design_bundle=sample,
    )
    if evidence != binding.evidence or evidence["task_id"] != binding.task_id:
        raise StrategyReportBundleError(
            "training-evidence binding identity changed"
        )
    canonical = canonical_modeling_training_evidence_json(
        evidence,
        sample_design_bundle=sample,
    )
    _require_canonical_artifact_hash(
        _record_text(binding.evidence_record, "content_hash"),
        canonical,
        "training-evidence",
    )
    if (
        _record_text(binding.model_binary_record, "id")
        != evidence["model_artifact"]["model_binary_ref"]["artifact_id"]
        or _record_text(binding.model_binary_record, "content_hash")
        != evidence["model_artifact"]["model_binary_ref"]["content_hash"]
    ):
        raise StrategyReportBundleError(
            "training-evidence model artifact identity changed"
        )
    return evidence


def _authenticated_score_evidence(
    binding: ModelScoreEvidenceArtifactBinding,
    *,
    sample: Mapping[str, Any],
    training_binding: ModelingTrainingEvidenceArtifactBinding | None,
    training: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_binding_type(
        binding,
        ModelScoreEvidenceArtifactBinding,
        "model-score-evidence",
    )
    if training_binding is None or training is None:
        raise StrategyReportBundleError(
            "model-score-evidence requires authenticated training evidence"
        )
    expected_ref = build_training_evidence_ref(training_binding)
    envelope = validate_model_score_evidence_envelope(
        binding.envelope,
        sample_design_bundle=sample,
        training_evidence=training,
        expected_training_evidence_ref=expected_ref,
        score_vector=binding.vector,
    )
    if envelope != binding.envelope or envelope["task_id"] != binding.task_id:
        raise StrategyReportBundleError(
            "model-score-evidence binding identity changed"
        )
    canonical = canonical_model_score_evidence_json(
        envelope,
        sample_design_bundle=sample,
        training_evidence=training,
        expected_training_evidence_ref=expected_ref,
        score_vector=binding.vector,
    )
    _require_canonical_artifact_hash(
        _record_text(binding.evidence_record, "content_hash"),
        canonical,
        "model-score-evidence",
    )
    if (
        _record_text(binding.vector_record, "id")
        != envelope["score_vector_ref"]["ref_id"]
        or _record_text(binding.vector_record, "content_hash")
        != envelope["score_vector_ref"]["content_hash"]
    ):
        raise StrategyReportBundleError(
            "model-score-evidence vector artifact identity changed"
        )
    return envelope


def _effective_training_binding(
    *,
    training_evidence: ModelingTrainingEvidenceArtifactBinding | None,
    score_evidence: ModelScoreEvidenceArtifactBinding | None,
) -> ModelingTrainingEvidenceArtifactBinding | None:
    if score_evidence is None:
        if training_evidence is not None:
            _require_binding_type(
                training_evidence,
                ModelingTrainingEvidenceArtifactBinding,
                "training-evidence",
            )
        return training_evidence
    _require_binding_type(
        score_evidence,
        ModelScoreEvidenceArtifactBinding,
        "model-score-evidence",
    )
    score_training = score_evidence.training
    _require_binding_type(
        score_training,
        ModelingTrainingEvidenceArtifactBinding,
        "model-score-evidence training",
    )
    if training_evidence is not None and _training_identity(
        training_evidence
    ) != _training_identity(score_training):
        raise StrategyReportBundleError(
            "model-score-evidence uses different training evidence"
        )
    return score_training


def _require_same_task(task_id: str, **bindings: object | None) -> None:
    for name, binding in bindings.items():
        if binding is not None and getattr(binding, "task_id", None) != task_id:
            raise StrategyReportBundleError(
                f"{name.replace('_', '-')} binding belongs to another task"
            )


def _require_sample_identity(
    *,
    sample_design: StrategySampleDesignV2ArtifactBinding,
    sample: Mapping[str, Any],
    model_evidence: StrategyModelEvidenceV2ArtifactBinding | None,
    training_evidence: ModelingTrainingEvidenceArtifactBinding | None,
    score_evidence: ModelScoreEvidenceArtifactBinding | None,
) -> None:
    expected = _sample_identity(sample_design)
    if (
        model_evidence is not None
        and _sample_identity(model_evidence.sample_design_binding) != expected
    ):
        raise StrategyReportBundleError(
            "model-evidence V2 references another sample design"
        )
    if (
        training_evidence is not None
        and _sample_identity(training_evidence.sample) != expected
    ):
        raise StrategyReportBundleError(
            "training-evidence references another sample design"
        )
    if score_evidence is not None and _sample_identity(
        score_evidence.training.sample
    ) != expected:
        raise StrategyReportBundleError(
            "model-score-evidence references another sample design"
        )
    design = sample["sample_design"]
    if (
        design["sample_design_id"] != expected["sample_design_id"]
        or design["content_hash"] != expected["sample_design_content_hash"]
    ):
        raise StrategyReportBundleError(
            "sample-design binding semantic identity changed"
        )


def _require_pool_impact_identity(
    *,
    sample: Mapping[str, Any],
    pool_binding: StrategyCandidatePoolArtifactBinding,
    pool: Mapping[str, Any],
    compiled_design: Mapping[str, Any],
    impact_binding: StrategyPoolImpactArtifactBinding,
    impact: Mapping[str, Any],
) -> None:
    if impact_binding.pool is not pool_binding and (
        impact_binding.pool.artifact_id != pool_binding.artifact_id
        or impact_binding.pool.artifact_content_hash
        != pool_binding.artifact_content_hash
    ):
        raise StrategyReportBundleError(
            "pool-impact references another candidate-pool artifact"
        )
    identity = impact["identity"]
    expected_pool = {
        "pool_id": pool["pool_id"],
        "task_id": pool["task_id"],
        "strategy_type": pool["strategy_type"],
        "revision": pool["revision"],
        "revision_id": pool["revision_id"],
        "snapshot_hash": pool["snapshot_hash"],
        "design_hash": compiled_design["design_hash"],
    }
    if any(identity[key] != value for key, value in expected_pool.items()):
        raise StrategyReportBundleError(
            "pool-impact candidate-pool identity changed"
        )
    if (
        impact["bindings"]["sample_design_ref"]
        != sample["sample_design"]["compatibility"]["legacy_development_ref"]
    ):
        raise StrategyReportBundleError(
            "pool-impact is not bound to the V2 risk/development compatibility sample"
        )


def _current_project_section(
    revision: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    state = revision["state"]
    snapshot = state["current_project_snapshot"]
    fields = [
        _named("scope", "项目范围", snapshot["scope"]),
        *(
            _named(f"current_{key}", label, snapshot["status_fields"][key])
            for key, label in (
                ("volume", "当前规模"),
                ("approval", "当前通过表现"),
                ("risk", "当前风险表现"),
                ("economics", "当前收益表现"),
            )
        ),
        _named("maturity_summary", "样本成熟度", snapshot["maturity_summary"]),
    ]
    return build_strategy_report_section(
        key="current_project",
        title=_SECTION_TITLES["current_project"],
        availability="present",
        summary_fields=fields,
        red_flags=[*snapshot["red_flags"], *state["red_flags"]],
        source_refs=[source_ref],
    )


def _history_section(
    revision: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    histories = revision["state"]["historical_strategy_reviews"]
    if not histories:
        return _absent_section(
            "historical_versions",
            "unavailable",
        )
    columns = [
        _column("version", "版本"),
        _column("effective_period", "生效区间"),
        _column("asset_status", "资产状态"),
        _column("scope", "策略范围"),
        _column("traffic_allocation", "流量分配"),
        _column("evidence_status", "证据状态"),
    ]
    rows = []
    for history in histories:
        version_field = (
            _present_field(history["version"], source_ref)
            if history["version"] is not None
            else _absent_field("unavailable", note="历史材料未提供版本号。")
        )
        rows.append(
            {
                "row_id": history["review_id"],
                "cells": {
                    "version": version_field,
                    "effective_period": history["effective_period"],
                    "asset_status": history["asset_status"],
                    "scope": history["scope"],
                    "traffic_allocation": history["traffic_allocation"],
                    "evidence_status": _present_field(
                        history["availability"],
                        source_ref,
                    ),
                },
            }
        )
    table = build_strategy_report_table(
        table_id="historical_strategy_versions",
        title="历史策略版本",
        sheet_key="02_history",
        granularity="aggregate",
        content_class="lineage",
        columns=columns,
        rows=rows,
        source_refs=[source_ref],
    )
    return build_strategy_report_section(
        key="historical_versions",
        title=_SECTION_TITLES["historical_versions"],
        availability="present",
        summary_fields=[
            _named(
                "historical_version_count",
                "历史版本数",
                _present_field(len(histories), source_ref),
            )
        ],
        tables=[table],
        red_flags=[
            flag
            for history in histories
            for flag in history["red_flags"]
        ],
        source_refs=[source_ref],
    )


def _sample_section(
    bundle: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    design = bundle["sample_design"]
    populations = {
        item["role"]: item
        for item in bundle["populations"]
    }
    risk_maturity = populations["risk"]["maturity_evidence"]
    maturity_field = _maturity_field(risk_maturity, source_ref)
    observations_by_definition = {
        item["metric_definition_id"]: item
        for item in bundle["metric_definitions"]
    }
    metric_rows = []
    for observation in bundle["metric_observations"]:
        definition = observations_by_definition[
            observation["metric_definition_ref"]["metric_definition_id"]
        ]
        metric_rows.append(
            {
                "row_id": observation["observation_id"],
                "cells": {
                    "population": _present_field(
                        observation["population"],
                        source_ref,
                    ),
                    "partition": _present_field(
                        observation["partition"],
                        source_ref,
                    ),
                    "metric": _present_field(
                        definition["display_name"],
                        source_ref,
                    ),
                    "value": _status_field(
                        observation["status"],
                        observation["value"],
                        source_ref,
                        status_map=_SAMPLE_STATUS_TO_AVAILABILITY,
                    ),
                    "unit": _present_field(
                        observation["unit"],
                        source_ref,
                    ),
                },
            }
        )
    table = build_strategy_report_table(
        table_id="sample_population_metrics",
        title="样本分区指标",
        sheet_key="03_sample",
        granularity="aggregate",
        content_class="metric_summary",
        columns=[
            _column("population", "样本口径"),
            _column("partition", "样本分区"),
            _column("metric", "指标"),
            _column("value", "指标值"),
            _column("unit", "单位"),
        ],
        rows=metric_rows,
        source_refs=[source_ref],
    )
    historical = bundle["historical_score"]
    historical_field = (
        _present_field("available", source_ref)
        if historical["status"] == "available"
        else _absent_field(
            "unavailable",
            note=historical["reason"],
        )
    )
    return build_strategy_report_section(
        key="sample_design",
        title=_SECTION_TITLES["sample_design"],
        availability="present",
        summary_fields=[
            _named(
                "sample_design_id",
                "样本设计ID",
                _present_field(design["sample_design_id"], source_ref),
            ),
            _named(
                "sample_relationship",
                "双样本关系",
                _present_field(design["relationship"], source_ref),
            ),
            _named(
                "analysis_universe_count",
                "分析总体样本数",
                _present_field(bundle["membership"]["row_count"], source_ref),
            ),
            _named(
                "approval_population_count",
                "审批口径样本数",
                _present_field(populations["approval"]["total_count"], source_ref),
            ),
            _named(
                "risk_population_count",
                "风险口径样本数",
                _present_field(populations["risk"]["total_count"], source_ref),
            ),
            _named("risk_maturity", "风险样本成熟度", maturity_field),
            _named("historical_score", "历史分可用性", historical_field),
        ],
        tables=[table],
        red_flags=[
            {
                "code": item["code"],
                "level": "amber" if item["status"] == "warn" else "red",
                "message": item["message"],
                "source_refs": [source_ref],
            }
            for item in bundle["diagnostics"]
            if item["status"] in {"warn", "fail"}
        ],
        source_refs=[source_ref],
    )


def _model_section(
    *,
    model: Mapping[str, Any] | None,
    training: Mapping[str, Any] | None,
    score: Mapping[str, Any] | None,
    refs: _EvidenceRefs,
) -> dict[str, Any]:
    source_refs = [
        ref
        for ref in (refs.model, refs.training, refs.score)
        if ref is not None
    ]
    if not source_refs:
        return _absent_section(
            "univariate_and_models",
            "unavailable",
        )
    fields: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    if model is not None and refs.model is not None:
        fields.extend(
            [
                _named(
                    "univariate_evidence_count",
                    "单变量分析数",
                    _present_field(len(model["univariate_evidence"]), refs.model),
                ),
                _named(
                    "model_evidence_count",
                    "模型证据数",
                    _present_field(len(model["model_evidence"]), refs.model),
                ),
                _named(
                    "model_comparison_count",
                    "模型比较数",
                    _present_field(len(model["comparison_evidence"]), refs.model),
                ),
            ]
        )
        observation_rows = _model_bundle_observation_rows(model, refs.model)
        if observation_rows:
            tables.append(
                _observation_table(
                    table_id="univariate_model_observations",
                    title="单变量与模型指标",
                    rows=observation_rows,
                    source_ref=refs.model,
                )
            )
        tables.extend(_model_comparison_tables(model, refs.model))
    if training is not None and refs.training is not None:
        training_model_ref = _training_model_ref(training)
        fields.extend(
            [
                _named(
                    "training_algorithm",
                    "训练算法",
                    _present_field(
                        training["model_artifact"]["algorithm"],
                        refs.training,
                    ),
                ),
                _named(
                    "training_feature_count",
                    "训练特征数",
                    _present_field(
                        len(training["training_contract"]["features"]),
                        refs.training,
                    ),
                ),
                _named(
                    "training_selection_status",
                    "模型选择状态",
                    _model_selection_field(
                        model=model,
                        model_ref=training_model_ref,
                        identity_ref=refs.training,
                        selection_ref=refs.model,
                    ),
                ),
            ]
        )
        metric_rows = [
            {
                "row_id": f"training-metric-{metric}",
                "cells": {
                    "metric": _present_field(metric, refs.training),
                    "value": (
                        _present_field(value, refs.training)
                        if value is not None
                        else _absent_field("unavailable")
                    ),
                },
            }
            for metric, value in sorted(
                training["metrics_snapshot"]["values"].items()
            )
        ]
        tables.append(
            build_strategy_report_table(
                table_id="training_metrics",
                title="模型训练指标",
                sheet_key="04_univariate_model",
                granularity="aggregate",
                content_class="metric_summary",
                columns=[
                    _column("metric", "指标"),
                    _column("value", "指标值"),
                ],
                rows=metric_rows,
                source_refs=[refs.training],
            )
        )
        importance_rows = [
            {
                "row_id": f"feature-importance-{index:05d}",
                "cells": {
                    "feature": _present_field(item["feature"], refs.training),
                    "importance": _present_field(
                        item["importance"],
                        refs.training,
                    ),
                },
            }
            for index, item in enumerate(training["feature_importance"])
        ]
        if importance_rows:
            tables.append(
                build_strategy_report_table(
                    table_id="training_feature_importance",
                    title="特征重要性",
                    sheet_key="04_univariate_model",
                    granularity="aggregate",
                    content_class="metric_summary",
                    columns=[
                        _column("feature", "特征"),
                        _column("importance", "重要性"),
                    ],
                    rows=importance_rows,
                    source_refs=[refs.training],
                )
            )
    if score is not None and refs.score is not None:
        score_model_ref = score["single_model_evidence"]["model_ref"]
        fields.extend(
            [
                _named(
                    "score_product",
                    "评分产品",
                    _present_field(score["score_product"], refs.score),
                ),
                _named(
                    "score_row_count",
                    "评分行数",
                    _present_field(
                        score["score_vector_contract"]["row_count"],
                        refs.score,
                    ),
                ),
                _named(
                    "score_selection_status",
                    "评分选择状态",
                    _model_selection_field(
                        model=model,
                        model_ref=score_model_ref,
                        identity_ref=refs.score,
                        selection_ref=refs.model,
                    ),
                ),
            ]
        )
        score_rows = _single_model_observation_rows(
            score["single_model_evidence"],
            refs.score,
            prefix="score",
        )
        if score_rows:
            tables.append(
                _observation_table(
                    table_id="governed_score_observations",
                    title="模型评分指标",
                    rows=score_rows,
                    source_ref=refs.score,
                )
            )
    return build_strategy_report_section(
        key="univariate_and_models",
        title=_SECTION_TITLES["univariate_and_models"],
        availability="present",
        summary_fields=fields,
        tables=tables,
        source_refs=source_refs,
    )


def _candidate_section(
    *,
    pool: Mapping[str, Any],
    compiled_design: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    candidate_rows = [
        {
            "row_id": entry["entry_id"],
            "cells": {
                "position": _present_field(entry["position"] + 1, source_ref),
                "rule_id": _present_field(entry["rule_id"], source_ref),
                "asset_type": _present_field(
                    entry["source"]["asset_type"],
                    source_ref,
                ),
                "action": _present_field(entry["action"]["type"], source_ref),
                "candidate_stage": _present_field(
                    entry["source"]["candidate_stage"],
                    source_ref,
                ),
                "validation_status": _present_field(
                    entry["source"]["validation_status"],
                    source_ref,
                ),
            },
        }
        for entry in pool["entries"]
    ]
    candidates = build_strategy_report_table(
        table_id="candidate_pool_entries",
        title="候选策略池",
        sheet_key="05_candidates",
        granularity="aggregate",
        content_class="rule_summary",
        columns=[
            _column("position", "顺序"),
            _column("rule_id", "规则ID"),
            _column("asset_type", "候选类型"),
            _column("action", "动作"),
            _column("candidate_stage", "候选阶段"),
            _column("validation_status", "验证状态"),
        ],
        rows=candidate_rows,
        source_refs=[source_ref],
    )
    strategy_rows = [
        {
            "row_id": f"compiled-rule-{item['rule_id']}",
            "cells": {
                "priority": _present_field(item["priority"], source_ref),
                "rule_id": _present_field(item["rule_id"], source_ref),
                "action": _present_field(item["action"]["type"], source_ref),
                "adoption_status": _present_field("not_adopted", source_ref),
                "deployment_status": _present_field("not_deployed", source_ref),
            },
        }
        for item in compiled_design["strategy_spec"]["rules"]
    ]
    strategy = build_strategy_report_table(
        table_id="compiled_candidate_design",
        title="编译后的候选策略设计",
        sheet_key="06_strategy",
        granularity="aggregate",
        content_class="rule_summary",
        columns=[
            _column("priority", "优先级"),
            _column("rule_id", "规则ID"),
            _column("action", "动作"),
            _column("adoption_status", "采纳状态"),
            _column("deployment_status", "部署状态"),
        ],
        rows=strategy_rows,
        source_refs=[source_ref],
    )
    return build_strategy_report_section(
        key="candidate_combinations",
        title=_SECTION_TITLES["candidate_combinations"],
        availability="present",
        summary_fields=[
            _named(
                "candidate_count",
                "候选规则数",
                _present_field(len(pool["entries"]), source_ref),
            ),
            _named(
                "pool_status",
                "策略池状态",
                _present_field(pool["status"], source_ref),
            ),
            _named(
                "pool_validation_status",
                "策略池验证状态",
                _present_field(pool["validation_status"], source_ref),
            ),
            _named(
                "compiled_design_hash",
                "候选设计哈希",
                _present_field(compiled_design["design_hash"], source_ref),
            ),
            _named(
                "adoption_status",
                "采纳状态",
                _present_field("not_adopted", source_ref),
            ),
            _named(
                "deployment_status",
                "部署状态",
                _present_field("not_deployed", source_ref),
            ),
        ],
        tables=[candidates, strategy],
        source_refs=[source_ref],
    )


def _impact_section(
    *,
    impact: Mapping[str, Any],
    pool_ref: Mapping[str, str],
    impact_ref: Mapping[str, str],
) -> dict[str, Any]:
    dataset_ref = _dataset_ref_from_impact(impact)
    population = impact["population"]
    lifecycle = impact["lifecycle"]
    summary_fields = [
        _named(
            "candidate_stage",
            "候选阶段",
            _present_field(lifecycle["candidate_stage"], impact_ref),
        ),
        _named(
            "observation_stage",
            "效果阶段",
            _present_field(lifecycle["observation_stage"], impact_ref),
        ),
        _named(
            "validation_status",
            "验证状态",
            _present_field(lifecycle["validation_status"], impact_ref),
        ),
        _named(
            "impact_population_count",
            "测算样本数",
            _present_field(population["population_count"], impact_ref),
        ),
        _named(
            "impact_label_coverage",
            "标签覆盖率",
            _present_field(population["label_coverage"], impact_ref),
        ),
        _named(
            "monthly_status",
            "逐月测算状态",
            (
                _present_field("available", impact_ref)
                if impact["monthly"]["status"] == "available"
                else _absent_field(
                    "unavailable",
                    note=impact["monthly"]["reason"],
                )
            ),
        ),
    ]
    tables = [
        _waterfall_table(impact, impact_ref),
        _action_impact_table(impact, impact_ref),
        _lifecycle_table(impact, impact_ref),
    ]
    monthly = _monthly_impact_table(impact, impact_ref)
    if monthly is not None:
        tables.append(monthly)
    economics, economics_fields = _economics_table(impact, impact_ref)
    summary_fields.extend(economics_fields)
    if economics is not None:
        tables.append(economics)
    baseline = _baseline_table(impact, impact_ref)
    if baseline is not None:
        tables.append(baseline)
    return build_strategy_report_section(
        key="impact_assessment",
        title=_SECTION_TITLES["impact_assessment"],
        availability="present",
        summary_fields=summary_fields,
        tables=tables,
        stage_evidence=[
            {
                "effect_stage": "backtested",
                "population": "risk",
                "partition": "development",
                "binding": {
                    "kind": "development_backtest",
                    "dataset_ref": dataset_ref,
                    "frozen_artifact_ref": pool_ref,
                    "result_ref": impact_ref,
                },
            }
        ],
        red_flags=[
            {
                "code": item["code"],
                "level": item["level"],
                "message": item["message"],
                "source_refs": [impact_ref],
            }
            for item in impact["red_flags"]
        ],
        source_refs=[pool_ref, impact_ref],
    )


def _final_document_section(
    *,
    pool: Mapping[str, Any],
    compiled_design: Mapping[str, Any],
    impact: Mapping[str, Any],
    pool_ref: Mapping[str, str],
    impact_ref: Mapping[str, str],
) -> dict[str, Any]:
    return build_strategy_report_section(
        key="final_document",
        title=_SECTION_TITLES["final_document"],
        availability="present",
        summary_fields=[
            _named(
                "strategy_type",
                "策略类型",
                _present_field(pool["strategy_type"], pool_ref),
            ),
            _named(
                "design_hash",
                "候选策略设计哈希",
                _present_field(compiled_design["design_hash"], pool_ref),
            ),
            _named(
                "evidence_stage",
                "当前证据阶段",
                _present_field(
                    "development_backtested_unvalidated",
                    impact_ref,
                ),
            ),
            _named(
                "adoption_status",
                "采纳状态",
                _present_field("not_adopted", impact_ref),
            ),
            _named(
                "deployment_status",
                "部署状态",
                _present_field("not_deployed", impact_ref),
            ),
            _named(
                "creates_strategy",
                "是否已创建生产策略",
                _present_field(False, impact_ref),
            ),
        ],
        source_refs=[pool_ref, impact_ref],
    )


def _model_bundle_observation_rows(
    bundle: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence in bundle["univariate_evidence"]:
        for observation in evidence["observations"]:
            rows.append(
                _observation_row(
                    observation,
                    source_ref,
                    row_id=f"univariate-{observation['observation_id']}",
                    subject=evidence["feature"],
                    status_map=_MODEL_STATUS_TO_AVAILABILITY,
                )
            )
    for evidence in bundle["model_evidence"]:
        rows.extend(
            _single_model_observation_rows(
                evidence,
                source_ref,
                prefix=evidence["evidence_id"],
            )
        )
    return rows


def _model_comparison_tables(
    bundle: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> list[dict[str, Any]]:
    comparisons = bundle["comparison_evidence"]
    if not comparisons:
        return []
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        sample_ref = comparison["evaluation_sample_ref"]
        selection = comparison["selection"]
        selected_ref = selection["selected_model_evidence_ref"]
        for metric in comparison["metrics"]:
            values_by_model = {
                (
                    item["model_evidence_ref"]["evidence_id"],
                    item["model_evidence_ref"]["content_hash"],
                ): item["value"]
                for item in (metric["model_values"] or [])
            }
            for model_ref in comparison["model_evidence_refs"]:
                model_value = values_by_model.get(
                    (
                        model_ref["evidence_id"],
                        model_ref["content_hash"],
                    )
                )
                metric_rows.append(
                    {
                        "row_id": (
                            f"{comparison['comparison_id']}-"
                            f"{metric['comparison_metric_id']}-"
                            f"{model_ref['evidence_id']}"
                        ),
                        "cells": {
                            "comparison_id": _present_field(
                                comparison["comparison_id"],
                                source_ref,
                            ),
                            "population": _present_field(
                                sample_ref["population"],
                                source_ref,
                            ),
                            "partition": _present_field(
                                sample_ref["partition"],
                                source_ref,
                            ),
                            "metric": _present_field(
                                metric["metric_key"],
                                source_ref,
                            ),
                            "period": (
                                _present_field(metric["period"], source_ref)
                                if metric["period"] is not None
                                else _absent_field("not_applicable")
                            ),
                            "model_evidence_id": _present_field(
                                model_ref["evidence_id"],
                                source_ref,
                            ),
                            "value": _status_field(
                                metric["status"],
                                model_value,
                                source_ref,
                                status_map=_MODEL_STATUS_TO_AVAILABILITY,
                                note=metric["reason"],
                            ),
                            "delta": _status_field(
                                metric["status"],
                                metric["delta"],
                                source_ref,
                                status_map=_MODEL_STATUS_TO_AVAILABILITY,
                                note=metric["reason"],
                            ),
                            "selected": _present_field(
                                selection["status"] == "selected"
                                and model_ref == selected_ref,
                                source_ref,
                            ),
                        },
                    }
                )
        selection_rows.append(
            {
                "row_id": f"{comparison['comparison_id']}-selection",
                "cells": {
                    "comparison_id": _present_field(
                        comparison["comparison_id"],
                        source_ref,
                    ),
                    "status": _present_field(selection["status"], source_ref),
                    "selected_model_evidence_id": (
                        _present_field(
                            selected_ref["evidence_id"],
                            source_ref,
                        )
                        if selected_ref is not None
                        else _absent_field("not_applicable")
                    ),
                    "metric": (
                        _present_field(selection["metric_key"], source_ref)
                        if selection["metric_key"] is not None
                        else _absent_field("not_applicable")
                    ),
                    "period": (
                        _present_field(selection["period"], source_ref)
                        if selection["period"] is not None
                        else _absent_field("not_applicable")
                    ),
                    "direction": (
                        _present_field(selection["direction"], source_ref)
                        if selection["direction"] is not None
                        else _absent_field("not_applicable")
                    ),
                    "reason": (
                        _present_field(selection["reason"], source_ref)
                        if selection["reason"] is not None
                        else _absent_field("not_applicable")
                    ),
                },
            }
        )
    return [
        build_strategy_report_table(
            table_id="model_comparison_metrics",
            title="模型比较指标",
            sheet_key="04_univariate_model",
            granularity="aggregate",
            content_class="metric_summary",
            columns=[
                _column("comparison_id", "比较ID"),
                _column("population", "样本口径"),
                _column("partition", "样本分区"),
                _column("metric", "指标"),
                _column("period", "月份"),
                _column("model_evidence_id", "模型证据ID"),
                _column("value", "指标值"),
                _column("delta", "比较差异"),
                _column("selected", "是否选中"),
            ],
            rows=metric_rows,
            source_refs=[source_ref],
        ),
        build_strategy_report_table(
            table_id="model_selection_results",
            title="模型选择结果",
            sheet_key="04_univariate_model",
            granularity="aggregate",
            content_class="lineage",
            columns=[
                _column("comparison_id", "比较ID"),
                _column("status", "选择状态"),
                _column("selected_model_evidence_id", "选中模型证据ID"),
                _column("metric", "选择指标"),
                _column("period", "月份"),
                _column("direction", "优化方向"),
                _column("reason", "未选择原因"),
            ],
            rows=selection_rows,
            source_refs=[source_ref],
        ),
    ]


def _training_model_ref(evidence: Mapping[str, Any]) -> dict[str, str]:
    model_binary = evidence["model_artifact"]["model_binary_ref"]
    return {
        "kind": model_binary["kind"],
        "ref_id": model_binary["artifact_id"],
        "content_hash": model_binary["content_hash"],
    }


def _model_selection_field(
    *,
    model: Mapping[str, Any] | None,
    model_ref: Mapping[str, Any],
    identity_ref: Mapping[str, str],
    selection_ref: Mapping[str, str] | None,
) -> dict[str, Any]:
    if model is None or selection_ref is None:
        return _absent_field(
            "unavailable",
            note="未提供可与该模型精确关联的模型比较选择证据。",
        )
    matching_models = [
        evidence
        for evidence in model["model_evidence"]
        if evidence["model_ref"] == model_ref
    ]
    if len(matching_models) != 1:
        return _absent_field(
            "unavailable",
            note="模型比较证据无法与该模型建立唯一精确引用。",
        )
    evidence_ref = {
        "evidence_id": matching_models[0]["evidence_id"],
        "content_hash": matching_models[0]["content_hash"],
    }
    statuses: set[str] = set()
    for comparison in model["comparison_evidence"]:
        if evidence_ref not in comparison["model_evidence_refs"]:
            continue
        selection = comparison["selection"]
        if selection["status"] == "no_selection":
            statuses.add("no_selection")
        elif selection["selected_model_evidence_ref"] == evidence_ref:
            statuses.add("selected")
        else:
            statuses.add("not_selected")
    if len(statuses) != 1:
        return _absent_field(
            "unavailable",
            note="该模型没有唯一一致的模型选择结论。",
        )
    return _present_field_many(
        statuses.pop(),
        [identity_ref, selection_ref],
    )


def _single_model_observation_rows(
    evidence: Mapping[str, Any],
    source_ref: Mapping[str, str],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    return [
        _observation_row(
            observation,
            source_ref,
            row_id=f"{prefix}-{observation['observation_id']}",
            subject=evidence["evidence_id"],
            status_map=_MODEL_STATUS_TO_AVAILABILITY,
        )
        for observation in evidence["observations"]
    ]


def _observation_row(
    observation: Mapping[str, Any],
    source_ref: Mapping[str, str],
    *,
    row_id: str,
    subject: str,
    status_map: Mapping[str, str],
) -> dict[str, Any]:
    sample_ref = observation["sample_ref"]
    return {
        "row_id": row_id,
        "cells": {
            "subject": _present_field(subject, source_ref),
            "population": _present_field(sample_ref["population"], source_ref),
            "partition": _present_field(sample_ref["partition"], source_ref),
            "metric": _present_field(observation["metric_key"], source_ref),
            "bin": (
                _present_field(observation["bin_id"], source_ref)
                if observation["bin_id"] is not None
                else _absent_field("not_applicable")
            ),
            "period": (
                _present_field(observation["period"], source_ref)
                if observation["period"] is not None
                else _absent_field("not_applicable")
            ),
            "value": _status_field(
                observation["status"],
                observation["value"],
                source_ref,
                status_map=status_map,
                note=observation["reason"],
            ),
            "unit": _present_field(observation["unit"], source_ref),
        },
    }


def _observation_table(
    *,
    table_id: str,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    return build_strategy_report_table(
        table_id=table_id,
        title=title,
        sheet_key="04_univariate_model",
        granularity="aggregate",
        content_class="metric_summary",
        columns=[
            _column("subject", "分析对象"),
            _column("population", "样本口径"),
            _column("partition", "样本分区"),
            _column("metric", "指标"),
            _column("bin", "分箱"),
            _column("period", "月份"),
            _column("value", "指标值"),
            _column("unit", "单位"),
        ],
        rows=rows,
        source_refs=[source_ref],
    )


def _waterfall_table(
    impact: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    rows = [
        {
            "row_id": row["entry_id"],
            "cells": {
                "position": _present_field(row["position"], source_ref),
                "rule_id": _present_field(row["rule_id"], source_ref),
                "action": _present_field(row["action"]["type"], source_ref),
                "standalone_count": _present_field(
                    row["standalone"]["population_count"],
                    source_ref,
                ),
                "incremental_count": _present_field(
                    row["incremental"]["population_count"],
                    source_ref,
                ),
                "shadowed_count": _present_field(
                    row["shadowed"]["population_count"],
                    source_ref,
                ),
                "remaining_count": _present_field(
                    row["remaining_after"]["population_count"],
                    source_ref,
                ),
                "incremental_bad_rate": _nullable_metric_field(
                    row["incremental"]["bad_rate"],
                    source_ref,
                ),
            },
        }
        for row in impact["waterfall"]
    ]
    return build_strategy_report_table(
        table_id="strategy_waterfall",
        title="规则瀑布与覆盖重叠",
        sheet_key="07_waterfall_swap",
        granularity="aggregate",
        content_class="rule_summary",
        effect_stage="backtested",
        columns=[
            _column("position", "顺序"),
            _column("rule_id", "规则ID"),
            _column("action", "动作"),
            _column("standalone_count", "独立命中数", unit="count", precision=0),
            _column("incremental_count", "增量命中数", unit="count", precision=0),
            _column("shadowed_count", "被遮蔽数", unit="count", precision=0),
            _column("remaining_count", "剩余数", unit="count", precision=0),
            _column("incremental_bad_rate", "增量坏账率", unit="%", precision=4),
        ],
        rows=rows,
        source_refs=[source_ref],
    )


def _action_impact_table(
    impact: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    rows = [
        {
            "row_id": f"overall-action-{row['action']}",
            "cells": {
                "action": _present_field(row["action"], source_ref),
                "count": _present_field(row["count"], source_ref),
                "rate": _present_field(row["rate"], source_ref),
                "labelled_count": _present_field(
                    row["labelled_count"],
                    source_ref,
                ),
                "bad_count": _present_field(row["bad_count"], source_ref),
                "bad_rate": _nullable_metric_field(
                    row["bad_rate"],
                    source_ref,
                ),
            },
        }
        for row in impact["overall"]["actions"]["breakdown"]
    ]
    return build_strategy_report_table(
        table_id="overall_action_impact",
        title="总体动作影响",
        sheet_key="08_impact",
        granularity="aggregate",
        content_class="metric_summary",
        effect_stage="backtested",
        columns=[
            _column("action", "动作"),
            _column("count", "样本数", unit="count", precision=0),
            _column("rate", "样本占比", unit="%", precision=4),
            _column("labelled_count", "有标签数", unit="count", precision=0),
            _column("bad_count", "坏样本数", unit="count", precision=0),
            _column("bad_rate", "坏账率", unit="%", precision=4),
        ],
        rows=rows,
        source_refs=[source_ref],
    )


def _monthly_impact_table(
    impact: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any] | None:
    monthly = impact["monthly"]
    if monthly["status"] != "available":
        return None
    rows = []
    for item in monthly["periods"]:
        metrics = item["actions"]["metrics"]
        rows.append(
            {
                "row_id": f"monthly-impact-{item['period']}",
                "cells": {
                    "period": _present_field(item["period"], source_ref),
                    "population_count": _present_field(
                        item["effect"]["population_count"],
                        source_ref,
                    ),
                    "approve_rate": _present_field(
                        metrics["approve_rate"],
                        source_ref,
                    ),
                    "reject_rate": _present_field(
                        metrics["reject_rate"],
                        source_ref,
                    ),
                    "review_rate": _present_field(
                        metrics["review_rate"],
                        source_ref,
                    ),
                    "overall_bad_rate": _nullable_metric_field(
                        metrics["overall_bad_rate"],
                        source_ref,
                    ),
                    "bad_capture_rate": _nullable_metric_field(
                        metrics["bad_capture_rate"],
                        source_ref,
                    ),
                    "good_reject_rate": _nullable_metric_field(
                        metrics["good_reject_rate"],
                        source_ref,
                    ),
                },
            }
        )
    return build_strategy_report_table(
        table_id="monthly_strategy_impact",
        title="逐月策略影响",
        sheet_key="08_impact",
        granularity="aggregate",
        content_class="monthly_summary",
        effect_stage="backtested",
        columns=[
            _column("period", "月份"),
            _column("population_count", "样本数", unit="count", precision=0),
            _column("approve_rate", "通过率", unit="%", precision=4),
            _column("reject_rate", "拒绝率", unit="%", precision=4),
            _column("review_rate", "复核率", unit="%", precision=4),
            _column("overall_bad_rate", "整体坏账率", unit="%", precision=4),
            _column("bad_capture_rate", "坏样本捕获率", unit="%", precision=4),
            _column("good_reject_rate", "好样本误拒率", unit="%", precision=4),
        ],
        rows=rows,
        source_refs=[source_ref],
    )


def _economics_table(
    impact: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    paired = impact["overall"]["effect"]["amounts"]["paired"]
    fields = [
        _named(
            "loan_amount_sum",
            "贷款金额合计",
            (
                _present_field(paired["loan_amount_sum"], source_ref)
                if paired["status"] == "available"
                else _absent_field("unavailable")
            ),
        ),
        _named(
            "overdue_amount_sum",
            "逾期金额合计",
            (
                _present_field(paired["overdue_amount_sum"], source_ref)
                if paired["status"] == "available"
                else _absent_field("unavailable")
            ),
        ),
        _named(
            "overdue_rate",
            "逾期金额率",
            (
                _nullable_metric_field(paired["overdue_rate"], source_ref)
                if paired["status"] == "available"
                else _absent_field("unavailable")
            ),
        ),
    ]
    if paired["status"] != "available":
        return None, fields
    table = build_strategy_report_table(
        table_id="overall_economics_impact",
        title="总体金额影响",
        sheet_key="09_economics",
        granularity="aggregate",
        content_class="metric_summary",
        effect_stage="backtested",
        columns=[
            _column("coverage_count", "金额覆盖样本数", unit="count", precision=0),
            _column("coverage_rate", "金额覆盖率", unit="%", precision=4),
            _column("loan_amount_sum", "贷款金额合计"),
            _column("overdue_amount_sum", "逾期金额合计"),
            _column("overdue_rate", "逾期金额率", unit="%", precision=4),
        ],
        rows=[
            {
                "row_id": "overall-economics",
                "cells": {
                    "coverage_count": _present_field(
                        paired["coverage_count"],
                        source_ref,
                    ),
                    "coverage_rate": _present_field(
                        paired["coverage_rate"],
                        source_ref,
                    ),
                    "loan_amount_sum": _present_field(
                        paired["loan_amount_sum"],
                        source_ref,
                    ),
                    "overdue_amount_sum": _present_field(
                        paired["overdue_amount_sum"],
                        source_ref,
                    ),
                    "overdue_rate": _nullable_metric_field(
                        paired["overdue_rate"],
                        source_ref,
                    ),
                },
            }
        ],
        source_refs=[source_ref],
    )
    return table, fields


def _baseline_table(
    impact: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any] | None:
    baseline = impact["baseline"]
    if baseline["status"] != "available":
        return None
    rows = [
        {
            "row_id": f"baseline-delta-{key}",
            "cells": {
                "metric": _present_field(key, source_ref),
                "delta": _present_field(value, source_ref),
            },
        }
        for key, value in sorted(
            baseline["overall"]["metric_deltas"].items()
        )
    ]
    return build_strategy_report_table(
        table_id="baseline_metric_deltas",
        title="相对基准策略变化",
        sheet_key="08_impact",
        granularity="aggregate",
        content_class="metric_summary",
        effect_stage="backtested",
        columns=[
            _column("metric", "指标"),
            _column("delta", "变化值"),
        ],
        rows=rows,
        source_refs=[source_ref],
    )


def _lifecycle_table(
    impact: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    lifecycle = impact["lifecycle"]
    return build_strategy_report_table(
        table_id="strategy_validation_status",
        title="策略验证与生命周期状态",
        sheet_key="10_validation",
        granularity="aggregate",
        content_class="lineage",
        effect_stage="backtested",
        columns=[
            _column("candidate_stage", "候选阶段"),
            _column("observation_stage", "效果阶段"),
            _column("validation_status", "验证状态"),
            _column("creates_strategy", "创建生产策略"),
            _column("adopted", "已采纳"),
            _column("deployed", "已部署"),
        ],
        rows=[
            {
                "row_id": "development-backtest-lifecycle",
                "cells": {
                    key: _present_field(lifecycle[key], source_ref)
                    for key in (
                        "candidate_stage",
                        "observation_stage",
                        "validation_status",
                        "creates_strategy",
                        "adopted",
                        "deployed",
                    )
                },
            }
        ],
        source_refs=[source_ref],
    )


def _maturity_field(
    maturity: Mapping[str, Any],
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    status = maturity["status"]
    if status == "confirmed_matured":
        return _present_field(status, source_ref)
    if status == "not_matured":
        return _absent_field(
            "not_matured",
            note=maturity["reason"],
            blocking="validation",
        )
    if status == "not_applicable":
        return _absent_field("not_applicable", note=maturity["reason"])
    return _absent_field(
        "unavailable",
        note=maturity["reason"],
        blocking="validation",
    )


def _status_field(
    status: str,
    value: Any,
    source_ref: Mapping[str, str],
    *,
    status_map: Mapping[str, str],
    note: str | None = None,
) -> dict[str, Any]:
    availability = status_map.get(status)
    if availability is None:
        raise StrategyReportBundleError(
            f"unsupported evidence availability status: {status}"
        )
    if availability == "present":
        return _present_field(value, source_ref)
    return _absent_field(availability, note=note or status)


def _nullable_metric_field(
    value: Any,
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    if value is None:
        return _absent_field("unavailable")
    return _present_field(value, source_ref)


def _present_field(
    value: Any,
    source_ref: Mapping[str, str],
) -> dict[str, Any]:
    return build_report_field(
        value=value,
        availability="present",
        origin="tool_output",
        source_refs=[source_ref],
    )


def _present_field_many(
    value: Any,
    source_refs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return build_report_field(
        value=value,
        availability="present",
        origin="tool_output",
        source_refs=source_refs,
    )


def _absent_field(
    availability: str,
    *,
    note: str | None = None,
    blocking: str = "none",
) -> dict[str, Any]:
    return build_report_field(
        value=None,
        availability=availability,
        origin="repository",
        source_refs=[],
        blocking=blocking,
        note=note,
    )


def _named(
    field_id: str,
    label: str,
    field: Mapping[str, Any],
) -> dict[str, Any]:
    return build_named_report_field(
        field_id=field_id,
        label=label,
        field=validate_report_field(field),
    )


def _column(
    key: str,
    label: str,
    *,
    unit: str | None = None,
    precision: int | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "unit": unit,
        "precision": precision,
    }


def _absent_section(key: str, availability: str) -> dict[str, Any]:
    return build_strategy_report_section(
        key=key,
        title=_SECTION_TITLES[key],
        availability=availability,
    )


def _dataset_ref_from_sample(
    bundle: Mapping[str, Any],
) -> dict[str, str]:
    dataset = bundle["sample_design"]["identity"]["dataset_ref"]
    return _artifact_ref(
        "dataset",
        dataset["dataset_id"],
        dataset["content_hash"],
    )


def _dataset_ref_from_impact(
    impact: Mapping[str, Any],
) -> dict[str, str]:
    sample = impact["bindings"]["sample"]
    return _artifact_ref(
        "dataset",
        sample["dataset_id"],
        sample["dataset_content_hash"],
    )


def _artifact_ref(
    kind: str,
    ref_id: str,
    content_hash: str,
) -> dict[str, str]:
    try:
        return build_source_ref(
            kind=kind,
            ref_id=ref_id,
            content_hash=content_hash,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyReportBundleError(
            f"authenticated {kind} source identity is invalid"
        ) from exc


def _dedupe_refs(
    refs: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized = [
        _artifact_ref(
            str(item["kind"]),
            str(item["ref_id"]),
            str(item["content_hash"]),
        )
        for item in refs
    ]
    by_identity = {
        (item["kind"], item["ref_id"], item["content_hash"]): item
        for item in normalized
    }
    return [
        by_identity[key]
        for key in sorted(by_identity)
    ]


def _sample_identity(
    binding: StrategySampleDesignV2ArtifactBinding,
) -> dict[str, str]:
    _require_binding_type(
        binding,
        StrategySampleDesignV2ArtifactBinding,
        "sample-design V2",
    )
    design = binding.bundle["sample_design"]
    return {
        "membership_artifact_id": binding.membership_artifact_id,
        "membership_artifact_content_hash": (
            binding.membership_artifact_content_hash
        ),
        "bundle_artifact_id": binding.bundle_artifact_id,
        "bundle_artifact_content_hash": binding.bundle_artifact_content_hash,
        "bundle_id": binding.bundle["bundle_id"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
    }


def _training_identity(
    binding: ModelingTrainingEvidenceArtifactBinding,
) -> tuple[str, str, str]:
    _require_binding_type(
        binding,
        ModelingTrainingEvidenceArtifactBinding,
        "training-evidence",
    )
    return (
        _record_text(binding.evidence_record, "id"),
        _record_text(binding.evidence_record, "content_hash"),
        str(binding.evidence["content_hash"]),
    )


def _record_text(record: Mapping[str, Any], field: str) -> str:
    if not isinstance(record, Mapping):
        raise StrategyReportBundleError(
            "authenticated TaskArtifact record is invalid"
        )
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise StrategyReportBundleError(
            f"authenticated TaskArtifact {field} is invalid"
        )
    return value


def _require_binding_type(
    value: object,
    expected: type,
    name: str,
) -> None:
    if not isinstance(value, expected):
        raise StrategyReportBundleError(
            f"{name} must be an authenticated {expected.__name__} binding"
        )


def _require_canonical_artifact_hash(
    supplied: object,
    canonical: str,
    name: str,
) -> None:
    if not isinstance(supplied, str):
        raise StrategyReportBundleError(
            f"{name} artifact content hash is invalid"
        )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if supplied != expected:
        raise StrategyReportBundleError(
            f"{name} artifact content hash does not match canonical evidence"
        )


__all__ = ["build_strategy_report_bundle_source_inputs"]
