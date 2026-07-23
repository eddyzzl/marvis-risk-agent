from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marvis.artifacts.model_score_vector import write_model_score_vector
from marvis.packs.modeling.evidence import (
    canonical_modeling_training_evidence_json,
)
from marvis.packs.modeling.evidence_tools import (
    ModelingTrainingEvidenceArtifactBinding,
    build_training_evidence_ref,
)
from marvis.packs.modeling.score_evidence import (
    build_model_score_evidence_envelope,
    build_single_model_score_evidence,
    canonical_model_score_evidence_json,
)
from marvis.packs.modeling.score_evidence_tools import (
    ModelScoreEvidenceArtifactBinding,
)
from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.model_evidence import (
    build_model_comparison_evidence,
    build_model_comparison_metric,
    build_model_selection,
    build_strategy_model_evidence_bundle,
    canonical_strategy_model_evidence_bundle_json,
)
from marvis.packs.strategy.model_evidence_tools import (
    StrategyModelEvidenceV2ArtifactBinding,
)
from marvis.packs.strategy.pool import (
    add_verified_candidate_fragment,
    canonical_strategy_pool_json,
    compile_strategy_pool,
)
from marvis.packs.strategy.pool_impact import (
    build_strategy_pool_impact_assessment,
    canonical_strategy_pool_impact_json,
)
from marvis.packs.strategy.pool_impact_tools import (
    StrategyPoolImpactArtifactBinding,
)
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
)
from marvis.packs.strategy.project_context import (
    build_context_field,
    build_current_project_snapshot,
    build_missing_information_record,
    build_report_field,
    build_source_ref,
    build_strategy_project_context_revision,
    build_strategy_project_context_state,
    canonical_strategy_project_context_revision_json,
)
from marvis.packs.strategy.project_context_tools import (
    StrategyProjectContextArtifactBinding,
)
from marvis.packs.strategy.report_bundle import (
    StrategyReportBundleError,
    build_strategy_report_bundle,
)
from marvis.packs.strategy.report_bundle_adapters import (
    build_strategy_report_bundle_source_inputs,
)
from marvis.packs.strategy.sample_design_v2 import (
    build_strategy_sample_design_v2,
    build_strategy_sample_design_v2_bundle,
    canonical_strategy_sample_design_v2_bundle_json,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    StrategySampleDesignV2ArtifactBinding,
)
from tests.test_modeling_evidence_contract import (
    _artifact as _model_artifact,
    _evidence as _training_evidence,
    _experiment as _model_experiment,
)
from tests.test_strategy_model_evidence import (
    _comparison as _model_comparison,
    _model as _single_model_evidence,
    _univariate as _univariate_evidence,
)
from tests.test_strategy_sample_design_v2 import (
    _components as _sample_components,
    _design_kwargs as _sample_design_kwargs,
    _diagnostic_statistics as _sample_diagnostic_statistics,
    _decoded_membership,
    _metric_observations as _sample_metric_observations,
    _policy as _sample_policy,
    _source_ref as _sample_source_ref,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_hash(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source(
    label: str,
    *,
    kind: str = "tool_output",
) -> dict[str, str]:
    return build_source_ref(
        kind=kind,
        ref_id=label,
        content_hash=_hash(label),
    )


def _present(value, source: dict[str, str]) -> dict:
    return build_report_field(
        value=value,
        availability="present",
        origin="tool_output",
        source_refs=[source],
    )


def _unavailable(note: str = "No governed evidence.") -> dict:
    return build_report_field(
        value=None,
        availability="unavailable",
        origin="repository",
        source_refs=[],
        note=note,
    )


def _project_binding(
    tmp_path: Path,
    *,
    task_id: str,
    dataset_ref: dict[str, str],
) -> StrategyProjectContextArtifactBinding:
    source = _source("current-project-metrics", kind="metric_observation")
    snapshot = build_current_project_snapshot(
        task_id=task_id,
        as_of="2026-07-23",
        scope=_present("存量经营复借策略", source),
        dataset_refs=[dataset_ref],
        workspace_ref=None,
        champion_strategy_ref=None,
        status_fields={
            "volume": _present(8, source),
            "approval": _present(0, source),
            "risk": _unavailable("风险表现暂缺。"),
            "economics": _unavailable("收益口径暂缺。"),
        },
        metric_definition_refs=[],
        metric_observation_refs=[source],
        monthly_observation_refs=[],
        segment_observation_refs=[],
        maturity_summary=_present("confirmed_matured", source),
        user_context_fields=[
            build_context_field(
                field_path="sensitive.customer_id",
                field=_present("PII-CUSTOMER-0001", source),
            )
        ],
        red_flags=[],
        tool_run_refs=[_source("project-tool-run", kind="tool_run")],
    )
    answer = _source("missing-answer", kind="agent_message")
    missing = build_missing_information_record(
        task_id=task_id,
        field_path="historical_strategy_reviews",
        reason="No historical review is currently available.",
        blocking="report_optional",
        question="请提供历史版本策略评审材料。",
        status="unavailable",
        asked_count=1,
        asked_at="2026-07-23T08:00:00+00:00",
        answered_at="2026-07-23T08:05:00+00:00",
        answer_source_ref=answer,
        dependency_hash=_hash("history-missing-dependency"),
    )
    state = build_strategy_project_context_state(
        task_id=task_id,
        as_of="2026-07-23",
        current_project_snapshot=snapshot,
        historical_strategy_reviews=[],
        missing_information_records=[missing],
        red_flags=[],
    )
    revision = build_strategy_project_context_revision(
        state=state,
        revision=1,
        parent_revision_id=None,
        parent_state_hash=None,
        operation_kind="test_report_projection",
    )
    canonical = canonical_strategy_project_context_revision_json(revision)
    return StrategyProjectContextArtifactBinding(
        task_id=task_id,
        artifact_id=_hash("project-context-artifact"),
        artifact_path=tmp_path / "project-context.json",
        artifact_content_hash=_file_hash(canonical),
        provenance={},
        revision=revision,
        tasks_root=tmp_path,
        datasets_root=tmp_path,
        db_path=tmp_path / "marvis.sqlite",
    )


def _sample_binding(
    tmp_path: Path,
    *,
    maturity_status: str = "confirmed_matured",
) -> StrategySampleDesignV2ArtifactBinding:
    membership = _decoded_membership()
    approval, risk, target, historical = _sample_components(
        membership,
        maturity_status=maturity_status,
    )
    design_kwargs = _sample_design_kwargs(maturity_status=maturity_status)
    design_kwargs["legacy_development_ref"] = {
        "artifact_id": _hash("legacy-artifact-id"),
        "artifact_content_hash": _hash("legacy-artifact"),
        "sample_design_id": (
            "strategy-sample-design-" + _hash("legacy-design-id")[:24]
        ),
        "sample_design_content_hash": _hash("legacy-design"),
        "partition": "development",
    }
    source_refs = [
        _sample_source_ref("design-a"),
        _sample_source_ref("design-b"),
    ]
    design = build_strategy_sample_design_v2(
        task_id="task-v2",
        membership_header=membership["header"],
        relationship="nested_same_cohort",
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_sample_policy(),
        source_refs=source_refs,
        **design_kwargs,
    )
    bundle = build_strategy_sample_design_v2_bundle(
        task_id="task-v2",
        membership_header=membership["header"],
        membership_masks=membership["masks"],
        relationship="nested_same_cohort",
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_sample_policy(),
        diagnostic_statistics=_sample_diagnostic_statistics(membership),
        metric_observations=_sample_metric_observations(
            membership,
            design,
            maturity_status=maturity_status,
        ),
        source_refs=source_refs,
        **design_kwargs,
    )
    canonical = canonical_strategy_sample_design_v2_bundle_json(bundle)
    return StrategySampleDesignV2ArtifactBinding(
        task_id="task-v2",
        membership_artifact_id=_hash("membership-artifact-id"),
        membership_path=tmp_path / "membership.bin",
        membership_artifact_content_hash=_hash("membership-artifact-bytes"),
        bundle_artifact_id=_hash("bundle-artifact-id"),
        bundle_path=tmp_path / "sample-bundle.json",
        bundle_artifact_content_hash=_file_hash(canonical),
        provenance={},
        membership_provenance={},
        membership=_decoded_membership(),
        bundle=bundle,
        source_binding=object(),
    )


def _action(action_type: str) -> dict:
    values = {
        "approval": "approve",
        "reject": "reject",
        "review": "review",
    }
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": None if action_type == "approval" else action_type.upper(),
        "stop": True,
    }


def _pool_and_impact_bindings(
    tmp_path: Path,
    sample: StrategySampleDesignV2ArtifactBinding,
) -> tuple[
    StrategyCandidatePoolArtifactBinding,
    StrategyPoolImpactArtifactBinding,
]:
    design = sample.bundle["sample_design"]
    dataset = design["identity"]["dataset_ref"]
    workspace = design["identity"]["workspace_ref"]
    evidence_identity = {
        "dataset_id": dataset["dataset_id"],
        "dataset_content_hash": dataset["content_hash"],
        "workspace_revision": workspace["revision"],
        "workspace_generation": workspace["generation"],
        "semantic_mapping_hash": workspace["semantic_mapping_hash"],
        "sample_context_hash": _hash("sample-context"),
    }
    fragment = build_verified_candidate_fragment(
        artifact={
            "artifact_id": "candidate-artifact-1",
            "artifact_kind": "test_candidate_json",
            "artifact_schema_version": "test.candidate-artifact.v1",
            "artifact_content_hash": _hash("candidate-artifact"),
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "test.candidate.v1",
            "asset_id": "candidate-asset-1",
            "asset_hash": _hash("candidate-asset"),
            "asset_type": "univariate_refinement",
        },
        fragment_type="strategy_rule",
        rule_id="rule-risk-1",
        condition={
            "op": "compare",
            "field": "customer_id",
            "operator": "==",
            "value": "PII-CUSTOMER-0001",
            "missing": "no_match",
        },
        requirements=[],
        effect_id="candidate-effect-1",
        evidence_id="candidate-evidence-1",
        evidence_hash=_hash("candidate-evidence"),
        evidence_identity=evidence_identity,
    )
    pool = add_verified_candidate_fragment(
        None,
        task_id="task-v2",
        strategy_type="approval",
        default_action=_action("approval"),
        verified_candidate_fragment=fragment,
        action=_action("reject"),
    )
    compiled = compile_strategy_pool(pool)
    pool_canonical = canonical_strategy_pool_json(pool)
    pool_binding = StrategyCandidatePoolArtifactBinding(
        task_id="task-v2",
        strategy_type="approval",
        pool=pool,
        compiled_design=compiled,
        artifact_id=_hash("candidate-pool-artifact"),
        artifact_path=tmp_path / "candidate-pool.json",
        artifact_content_hash=_file_hash(pool_canonical),
        artifact_origin_tool="strategy.add_candidate_to_pool",
        artifact_provenance={},
        artifact_provenance_json="{}",
        lineages=(),
        tasks_root=tmp_path,
        datasets_root=tmp_path,
        db_path=tmp_path / "marvis.sqlite",
    )
    frame = pd.DataFrame(
        {
            "customer_id": [
                "PII-CUSTOMER-0001",
                "PII-CUSTOMER-0002",
                "PII-CUSTOMER-0003",
                "PII-CUSTOMER-0004",
                "PII-CUSTOMER-0005",
                "PII-CUSTOMER-0006",
                "PII-CUSTOMER-0007",
                "PII-CUSTOMER-0008",
            ],
            "target": [1, 0, 1, 0, 1, 0, 1, 0],
            "apply_month": [
                "202601",
                "202601",
                "202601",
                "202601",
                "202602",
                "202602",
                "202602",
                "202602",
            ],
            "loan_amount": [100.0] * 8,
            "overdue_amount": [10.0, 0.0, 5.0, 0.0, 8.0, 0.0, 4.0, 0.0],
        }
    )
    sample_ref = design["compatibility"]["legacy_development_ref"]
    assessment = build_strategy_pool_impact_assessment(
        pool=pool,
        frame=frame,
        sample_binding={"task_id": "task-v2", **evidence_identity},
        sample_design_ref=sample_ref,
        target_col="target",
        target_bad_value=1,
        month_col="apply_month",
        loan_amount_col="loan_amount",
        overdue_amount_col="overdue_amount",
    )
    impact_binding = StrategyPoolImpactArtifactBinding(
        task_id="task-v2",
        artifact_id=_hash("pool-impact-artifact"),
        artifact_path=tmp_path / "pool-impact.json",
        artifact_content_hash=_file_hash(
            canonical_strategy_pool_impact_json(assessment)
        ),
        artifact_provenance={},
        artifact_provenance_json="{}",
        assessment=assessment,
        request={},
        pool=pool_binding,
        dataset=object(),
        sample_design=SimpleNamespace(to_ref_dict=lambda: sample_ref),
        baseline=None,
        stage="development_backtest",
        validation_status="unvalidated",
        tasks_root=tmp_path,
        db_path=tmp_path / "marvis.sqlite",
    )
    return pool_binding, impact_binding


def _bindings(
    tmp_path: Path,
    *,
    maturity_status: str = "confirmed_matured",
) -> tuple[
    StrategyProjectContextArtifactBinding,
    StrategySampleDesignV2ArtifactBinding,
    StrategyCandidatePoolArtifactBinding,
    StrategyPoolImpactArtifactBinding,
]:
    sample = _sample_binding(tmp_path, maturity_status=maturity_status)
    dataset = sample.bundle["sample_design"]["identity"]["dataset_ref"]
    project = _project_binding(
        tmp_path,
        task_id=sample.task_id,
        dataset_ref=build_source_ref(
            kind="dataset",
            ref_id=dataset["dataset_id"],
            content_hash=dataset["content_hash"],
        ),
    )
    pool, impact = _pool_and_impact_bindings(tmp_path, sample)
    return project, sample, pool, impact


def _training_binding(
    tmp_path: Path,
    sample: StrategySampleDesignV2ArtifactBinding,
) -> ModelingTrainingEvidenceArtifactBinding:
    experiment = _model_experiment()
    artifact = _model_artifact()
    evidence = _training_evidence(
        bundle=sample.bundle,
        experiment=experiment,
        artifact=artifact,
    )
    canonical = canonical_modeling_training_evidence_json(
        evidence,
        sample_design_bundle=sample.bundle,
    )
    model_binary = evidence["model_artifact"]["model_binary_ref"]
    return ModelingTrainingEvidenceArtifactBinding(
        task_id=sample.task_id,
        sample=sample,
        experiment=experiment,
        model_artifact=artifact,
        model_binary_record={
            "id": model_binary["artifact_id"],
            "content_hash": model_binary["content_hash"],
        },
        evidence_record={
            "id": _hash("training-evidence-record"),
            "content_hash": _file_hash(canonical),
        },
        evidence=evidence,
        model_binary_path=tmp_path / "model.bin",
        evidence_path=tmp_path / "training-evidence.json",
    )


def _model_binding(
    tmp_path: Path,
    sample: StrategySampleDesignV2ArtifactBinding,
    *,
    selection_status: str = "selected",
    comparison_metric_status: str = "present",
) -> StrategyModelEvidenceV2ArtifactBinding:
    univariate = _univariate_evidence(sample.bundle)
    first_model = _single_model_evidence(
        sample.bundle,
        "first",
        validation_ks=0.38,
    )
    second_model = _single_model_evidence(
        sample.bundle,
        "second",
        auc=0.71,
        validation_ks=0.35,
    )
    comparison = _model_comparison(sample.bundle, first_model, second_model)
    if comparison_metric_status != "present":
        evaluation = comparison["evaluation_sample_ref"]
        source_metric = comparison["metrics"][0]
        unavailable_metric = build_model_comparison_metric(
            sample_design_bundle=sample.bundle,
            population=evaluation["population"],
            partition=evaluation["partition"],
            metric_key=source_metric["metric_key"],
            status=comparison_metric_status,
            unit=source_metric["unit"],
            source_ref=source_metric["source_ref"],
            model_values=None,
            delta=None,
            period=source_metric["period"],
            reason="验证样本尚未形成可用比较指标。",
        )
        comparison = build_model_comparison_evidence(
            sample_design_bundle=sample.bundle,
            population=evaluation["population"],
            partition=evaluation["partition"],
            comparison_ref=comparison["comparison_ref"],
            model_evidence_refs=comparison["model_evidence_refs"],
            metrics=[unavailable_metric],
            selection=build_model_selection(
                status="no_selection",
                reason="比较指标不可用，未选择模型。",
            ),
        )
    if selection_status == "no_selection":
        evaluation = comparison["evaluation_sample_ref"]
        comparison = build_model_comparison_evidence(
            sample_design_bundle=sample.bundle,
            population=evaluation["population"],
            partition=evaluation["partition"],
            comparison_ref=comparison["comparison_ref"],
            model_evidence_refs=comparison["model_evidence_refs"],
            metrics=comparison["metrics"],
            selection=build_model_selection(
                status="no_selection",
                reason="业务尚未确认最终模型。",
            ),
        )
    bundle = build_strategy_model_evidence_bundle(
        sample_design_bundle=sample.bundle,
        univariate_evidence=[univariate],
        model_evidence=[first_model, second_model],
        comparison_evidence=[comparison],
    )
    canonical = canonical_strategy_model_evidence_bundle_json(
        bundle,
        sample_design_bundle=sample.bundle,
    )
    return StrategyModelEvidenceV2ArtifactBinding(
        task_id=sample.task_id,
        artifact_id=_hash("model-evidence-artifact"),
        path=tmp_path / "model-evidence.json",
        artifact_content_hash=_file_hash(canonical),
        provenance={},
        bundle=bundle,
        sample_design_binding=sample,
        sources=(),
        warnings=(),
        tasks_root=tmp_path,
        db_path=tmp_path / "marvis.sqlite",
    )


def _score_binding(
    tmp_path: Path,
    training: ModelingTrainingEvidenceArtifactBinding,
) -> ModelScoreEvidenceArtifactBinding:
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 1, 0],
            "apply_month": [
                "202601",
                "202601",
                "202601",
                "202602",
                "202603",
                "202602",
                "202603",
                "202604",
            ],
            "income": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "age": [21, 22, 23, 24, 25, 26, 27, 28],
        }
    )
    scores = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.6, 0.4])
    vector = write_model_score_vector(tmp_path / "scores.parquet", scores)
    training_ref = build_training_evidence_ref(training)
    model_binary = training.evidence["model_artifact"]["model_binary_ref"]
    model_ref = {
        "kind": model_binary["kind"],
        "ref_id": model_binary["artifact_id"],
        "content_hash": model_binary["content_hash"],
    }
    vector_record_id = _hash("score-vector-record")
    score_ref = {
        "kind": "model_score_vector_parquet",
        "ref_id": vector_record_id,
        "content_hash": vector.content_hash,
    }
    single = build_single_model_score_evidence(
        sample_design_bundle=training.sample.bundle,
        membership_masks=_decoded_membership()["masks"],
        frame=frame,
        scores=scores,
        training_evidence_ref={
            "kind": "modeling_training_evidence_json",
            "ref_id": training_ref["evidence_artifact_id"],
            "content_hash": training_ref[
                "expected_evidence_artifact_content_hash"
            ],
        },
        model_ref=model_ref,
        score_ref=score_ref,
        features=training.evidence["training_contract"]["features"],
    )
    envelope = build_model_score_evidence_envelope(
        task_id=training.task_id,
        training_evidence_ref=training_ref,
        training_evidence=training.evidence,
        sample_design_bundle=training.sample.bundle,
        model_ref=model_ref,
        score_ref=score_ref,
        score_vector=vector,
        single_model_evidence=single,
    )
    canonical = canonical_model_score_evidence_json(
        envelope,
        sample_design_bundle=training.sample.bundle,
        training_evidence=training.evidence,
        expected_training_evidence_ref=training_ref,
        score_vector=vector,
    )
    return ModelScoreEvidenceArtifactBinding(
        task_id=training.task_id,
        training=training,
        vector_record={
            "id": vector_record_id,
            "content_hash": vector.content_hash,
        },
        evidence_record={
            "id": _hash("score-evidence-record"),
            "content_hash": _file_hash(canonical),
        },
        vector=vector,
        envelope=envelope,
        vector_path=vector.path,
        evidence_path=tmp_path / "score-evidence.json",
    )


def _project(
    bindings: tuple[
        StrategyProjectContextArtifactBinding,
        StrategySampleDesignV2ArtifactBinding,
        StrategyCandidatePoolArtifactBinding,
        StrategyPoolImpactArtifactBinding,
    ],
    **optional,
) -> dict:
    project, sample, pool, impact = bindings
    return build_strategy_report_bundle_source_inputs(
        project_context=project,
        sample_design=sample,
        candidate_pool=pool,
        pool_impact=impact,
        **optional,
    )


def test_adapter_is_deterministic_bundle_ready_and_uses_exact_source_identities(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)
    project, sample, pool, impact = bindings

    first = _project(bindings)
    second = _project(bindings)

    assert first == second
    assert [item["key"] for item in first["sections"]] == [
        "current_project",
        "historical_versions",
        "sample_design",
        "univariate_and_models",
        "candidate_combinations",
        "impact_assessment",
        "final_document",
    ]
    assert first["strategy_artifact_refs"] == [
        {
            "kind": "pool_impact",
            "ref_id": impact.artifact_id,
            "content_hash": impact.artifact_content_hash,
        },
        {
            "kind": "strategy_candidate_pool",
            "ref_id": pool.artifact_id,
            "content_hash": pool.artifact_content_hash,
        },
    ]
    bundle = build_strategy_report_bundle(
        task_id=project.task_id,
        report_revision=1,
        strategy_id=None,
        strategy_version=None,
        strategy_type=None,
        title=_present("V2策略开发评审", first["strategy_artifact_refs"][1]),
        status="partial",
        generated_at="2026-07-23T16:00:00+08:00",
        **first,
    )
    assert bundle["effect_stages"] == ["backtested"]
    assert bundle["missing_information"] == project.revision["state"][
        "missing_information_records"
    ]
    sample_refs = first["sections"][2]["source_refs"]
    assert sample_refs == [
        {
            "kind": "sample_design",
            "ref_id": sample.bundle_artifact_id,
            "content_hash": sample.bundle_artifact_content_hash,
        }
    ]


def test_adapter_never_leaks_raw_rows_pii_or_overclaims_lifecycle(
    tmp_path: Path,
) -> None:
    result = _project(_bindings(tmp_path))
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
    impact = result["sections"][5]
    candidates = result["sections"][4]
    final = result["sections"][6]

    assert "PII-CUSTOMER" not in serialized
    assert "customer_id" not in serialized
    assert "row_ordinal" not in serialized
    assert "oot_validated" not in serialized
    assert "post_launch_observed" not in serialized
    assert impact["stage_evidence"] == [
        {
            "effect_stage": "backtested",
            "population": "risk",
            "partition": "development",
            "binding": {
                "kind": "development_backtest",
                "dataset_ref": impact["stage_evidence"][0]["binding"][
                    "dataset_ref"
                ],
                "frozen_artifact_ref": impact["stage_evidence"][0]["binding"][
                    "frozen_artifact_ref"
                ],
                "result_ref": impact["stage_evidence"][0]["binding"][
                    "result_ref"
                ],
            },
        }
    ]
    assert {
        item["field_id"]: item["field"]["value"]
        for item in candidates["summary_fields"]
    }["adoption_status"] == "not_adopted"
    final_fields = {
        item["field_id"]: item["field"]["value"]
        for item in final["summary_fields"]
    }
    assert final_fields["adoption_status"] == "not_adopted"
    assert final_fields["deployment_status"] == "not_deployed"
    assert final_fields["creates_strategy"] is False


def test_adapter_preserves_zero_unavailable_and_not_matured_without_filling(
    tmp_path: Path,
) -> None:
    result = _project(
        _bindings(tmp_path, maturity_status="not_matured")
    )
    current = {
        item["field_id"]: item["field"]
        for item in result["sections"][0]["summary_fields"]
    }
    sample = {
        item["field_id"]: item["field"]
        for item in result["sections"][2]["summary_fields"]
    }
    sample_rows = result["sections"][2]["tables"][0]["rows"]

    assert current["current_approval"] == {
        "value": 0,
        "availability": "present",
        "origin": "tool_output",
        "source_refs": current["current_approval"]["source_refs"],
        "as_of": None,
        "blocking": "none",
        "note": None,
    }
    assert current["current_risk"]["value"] is None
    assert current["current_risk"]["availability"] == "unavailable"
    assert sample["risk_maturity"]["value"] is None
    assert sample["risk_maturity"]["availability"] == "not_matured"
    assert sample["risk_maturity"]["blocking"] == "validation"
    assert any(
        row["cells"]["value"]["value"] is None
        and row["cells"]["value"]["availability"] == "not_matured"
        for row in sample_rows
    )


def test_optional_model_training_and_score_evidence_project_only_aggregates(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)
    sample = bindings[1]
    model = _model_binding(tmp_path, sample)
    training = _training_binding(tmp_path, sample)
    score = _score_binding(tmp_path, training)

    result = _project(
        bindings,
        model_evidence=model,
        training_evidence=training,
        score_evidence=score,
    )
    section = result["sections"][3]
    table_ids = {item["table_id"] for item in section["tables"]}
    serialized = json.dumps(section, ensure_ascii=False, sort_keys=True)

    assert section["availability"] == "present"
    assert {
        "univariate_model_observations",
        "model_comparison_metrics",
        "model_selection_results",
        "training_metrics",
        "training_feature_importance",
        "governed_score_observations",
    } <= table_ids
    selection_table = next(
        item
        for item in section["tables"]
        if item["table_id"] == "model_selection_results"
    )
    assert selection_table["rows"][0]["cells"]["status"]["value"] == "selected"
    assert (
        selection_table["rows"][0]["cells"]["selected_model_evidence_id"][
            "value"
        ]
        is not None
    )
    fields = {
        item["field_id"]: item["field"]
        for item in section["summary_fields"]
    }
    assert fields["training_selection_status"]["availability"] == "unavailable"
    assert fields["score_selection_status"]["availability"] == "unavailable"
    assert fields["training_selection_status"]["value"] is None
    assert fields["score_selection_status"]["value"] is None
    assert score.evidence_record["id"] in serialized
    assert score.vector_record["id"] not in serialized
    assert "row_ordinal" not in serialized
    assert "score_min" not in serialized
    assert "score_max" not in serialized
    assert "PII-CUSTOMER" not in serialized


def test_model_no_selection_is_projected_without_inventing_a_selected_model(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)
    model = _model_binding(
        tmp_path,
        bindings[1],
        selection_status="no_selection",
    )

    result = _project(bindings, model_evidence=model)
    section = result["sections"][3]
    selection_table = next(
        item
        for item in section["tables"]
        if item["table_id"] == "model_selection_results"
    )
    row = selection_table["rows"][0]["cells"]

    assert row["status"]["value"] == "no_selection"
    assert row["selected_model_evidence_id"]["value"] is None
    assert row["selected_model_evidence_id"]["availability"] == "not_applicable"
    assert row["reason"]["value"] == "业务尚未确认最终模型。"
    comparison_table = next(
        item
        for item in section["tables"]
        if item["table_id"] == "model_comparison_metrics"
    )
    assert all(
        item["cells"]["selected"]["value"] is False
        for item in comparison_table["rows"]
    )


def test_non_present_model_comparison_projects_one_empty_row_per_model(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path)
    model = _model_binding(
        tmp_path,
        bindings[1],
        comparison_metric_status="unavailable",
    )

    result = _project(bindings, model_evidence=model)
    table = next(
        item
        for item in result["sections"][3]["tables"]
        if item["table_id"] == "model_comparison_metrics"
    )

    assert len(table["rows"]) == 2
    assert all(
        row["cells"]["value"]["value"] is None
        and row["cells"]["value"]["availability"] == "unavailable"
        and row["cells"]["delta"]["value"] is None
        and row["cells"]["delta"]["availability"] == "unavailable"
        for row in table["rows"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("untyped_project", "authenticated"),
        ("sample_hash", "canonical evidence"),
        ("compiled_design", "compiled design"),
        ("impact_stage", "development"),
        ("cross_task", "another task"),
    ],
)
def test_adapter_rejects_untyped_or_forged_bindings(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    project, sample, pool, impact = _bindings(tmp_path)
    if mutation == "untyped_project":
        project = {"task_id": project.task_id}  # type: ignore[assignment]
    elif mutation == "sample_hash":
        sample = replace(
            sample,
            bundle_artifact_content_hash="f" * 64,
        )
    elif mutation == "compiled_design":
        pool = replace(
            pool,
            compiled_design={
                **pool.compiled_design,
                "requirements": [{"forged": True}],
            },
        )
    elif mutation == "impact_stage":
        impact = replace(impact, stage="independent_oot")
    else:
        impact = replace(impact, task_id="foreign-task")

    with pytest.raises(StrategyReportBundleError, match=message):
        build_strategy_report_bundle_source_inputs(
            project_context=project,  # type: ignore[arg-type]
            sample_design=sample,
            candidate_pool=pool,
            pool_impact=impact,
        )
