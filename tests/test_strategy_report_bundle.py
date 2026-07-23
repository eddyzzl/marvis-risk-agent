from __future__ import annotations

from copy import deepcopy
import json

import pytest

from marvis.packs.strategy import report_bundle as report_contract
from marvis.packs.strategy.project_context import (
    build_missing_information_record,
    build_report_field,
    build_source_ref,
)
from marvis.packs.strategy.report_bundle import (
    REPORT_SECTION_KEYS,
    StrategyReportBundleError,
    build_named_report_field,
    build_strategy_report_bundle,
    build_strategy_report_section,
    build_strategy_report_table,
    canonical_strategy_report_bundle_json,
    strategy_report_bundle_from_json,
    validate_strategy_report_bundle,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _source(
    kind: str = "task_artifact",
    ref_id: str = "artifact-1",
    content_hash: str = _HASH_A,
) -> dict[str, str]:
    return build_source_ref(
        kind=kind,
        ref_id=ref_id,
        content_hash=content_hash,
    )


def _present(value, *, source=None, origin: str = "tool_output"):
    return build_report_field(
        value=value,
        availability="present",
        origin=origin,
        source_refs=[source or _source()],
    )


def _absent(availability: str = "unavailable", *, note: str | None = None):
    return build_report_field(
        value=None,
        availability=availability,
        origin="repository",
        source_refs=[],
        note=note,
    )


def _missing(
    *,
    task_id: str = "task-1",
    blocking: str = "report_optional",
    status: str = "unavailable",
):
    answer = _source("agent_message", "message-1", _HASH_C)
    resolved = status != "pending"
    return build_missing_information_record(
        task_id=task_id,
        field_path="current_project.business_background",
        reason="No governed business background was available.",
        blocking=blocking,
        question="Can you provide the business background?",
        status=status,
        asked_count=1,
        asked_at="2026-07-23T08:00:00+00:00",
        answered_at="2026-07-23T08:05:00+00:00" if resolved else None,
        answer_source_ref=answer if resolved else None,
        dependency_hash=_HASH_B,
    )


def _sections() -> list[dict]:
    current_ref = _source("metric_observation", "approval-rate", _HASH_A)
    dataset_ref = _source("dataset", "dataset-1", _HASH_A)
    sample_ref = _source("sample_design", "sample-design-1", _HASH_B)
    candidate_ref = _source("candidate", "candidate-1", _HASH_C)
    strategy_ref = _source("strategy", "strategy-1", _HASH_B)
    impact_ref = _source("strategy_impact", "impact-1", _HASH_B)
    report_ref = _source("report_context", "context-1", _HASH_C)
    current = build_strategy_report_section(
        key="current_project",
        title="当前项目状况",
        availability="present",
        summary_fields=[
            build_named_report_field(
                field_id="approval_rate",
                label="通过率",
                field=_present(0, source=current_ref),
            ),
            build_named_report_field(
                field_id="business_background",
                label="业务背景",
                field=_absent(note="用户明确表示暂时没有"),
            ),
        ],
        source_refs=[current_ref],
    )
    impact_table = build_strategy_report_table(
        table_id="monthly_impact",
        title="逐月影响",
        sheet_key="08_impact",
        granularity="aggregate",
        content_class="monthly_summary",
        effect_stage="backtested",
        columns=[
            {
                "key": "month",
                "label": "月份",
                "unit": None,
                "precision": None,
            },
            {
                "key": "approval_rate",
                "label": "通过率",
                "unit": "%",
                "precision": 4,
            },
            {
                "key": "risk_rate",
                "label": "风险率",
                "unit": "%",
                "precision": 4,
            },
        ],
        rows=[
            {
                "row_id": "2026-06",
                "cells": {
                    "month": _present("2026-06", source=impact_ref),
                    "approval_rate": _present(0, source=impact_ref),
                    "risk_rate": _absent(
                        "not_matured",
                        note="MOB3 performance window is not mature.",
                    ),
                },
            }
        ],
        source_refs=[impact_ref],
    )
    impact = build_strategy_report_section(
        key="impact_assessment",
        title="策略影响测算",
        availability="present",
        tables=[impact_table],
        stage_evidence=[
            {
                "effect_stage": "backtested",
                "population": "risk",
                "partition": "development",
                "binding": {
                    "kind": "development_backtest",
                    "dataset_ref": dataset_ref,
                    "frozen_artifact_ref": strategy_ref,
                    "result_ref": impact_ref,
                },
            }
        ],
        source_refs=[impact_ref],
    )
    sample = build_strategy_report_section(
        key="sample_design",
        title="本次样本设计",
        availability="present",
        source_refs=[sample_ref],
    )
    candidates = build_strategy_report_section(
        key="candidate_combinations",
        title="候选组合与最终策略",
        availability="present",
        source_refs=[candidate_ref],
    )
    final_document = build_strategy_report_section(
        key="final_document",
        title="最终文档",
        availability="present",
        summary_fields=[
            build_named_report_field(
                field_id="adoption_status",
                label="采纳状态",
                field=_present(
                    "not_adopted",
                    source=report_ref,
                    origin="repository",
                ),
            )
        ],
        source_refs=[report_ref],
    )
    by_key = {
        current["key"]: current,
        sample["key"]: sample,
        candidates["key"]: candidates,
        impact["key"]: impact,
        final_document["key"]: final_document,
    }
    return [
        by_key.get(
            key,
            build_strategy_report_section(
                key=key,
                title=key,
                availability="unavailable",
            ),
        )
        for key in REPORT_SECTION_KEYS
    ]


def _bundle(
    *,
    report_revision: int = 1,
    previous_report_id: str | None = None,
    status: str = "partial",
    sections=None,
    missing_information=None,
    title: str = "风险策略迭代评审",
    strategy_id: str | None = "strategy-1",
    strategy_version: str | None = "1",
    strategy_type: str | None = "approval",
):
    dataset = _source("dataset", "dataset-1", _HASH_A)
    strategy = _source("strategy", "strategy-1", _HASH_B)
    tool_run = _source("tool_run", "tool-run-1", _HASH_C)
    return build_strategy_report_bundle(
        task_id="task-1",
        report_revision=report_revision,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        strategy_type=strategy_type,
        title=_present(
            title,
            source=_source("report_context", "title-1", _HASH_A),
            origin="repository",
        ),
        status=status,
        sections=_sections() if sections is None else sections,
        dataset_refs=[dataset],
        strategy_artifact_refs=[strategy],
        tool_run_refs=[tool_run],
        missing_information=(
            [_missing()] if missing_information is None else missing_information
        ),
        generated_at="2026-07-23T16:00:00+08:00",
        previous_report_id=previous_report_id,
    )


def test_report_bundle_is_canonical_content_addressed_and_roundtrips():
    first = _bundle()
    rebuilt = _bundle()

    assert first == rebuilt
    assert validate_strategy_report_bundle(first) == first
    raw = canonical_strategy_report_bundle_json(first)
    assert strategy_report_bundle_from_json(raw) == first
    assert json.loads(raw)["report_id"] == first["report_id"]
    assert first["effect_stages"] == ["backtested"]
    assert first["completeness_summary"] == {
        "field_counts": {
            "present": 4,
            "unavailable": 1,
            "not_applicable": 0,
            "not_matured": 1,
            "total": 6,
        },
        "section_counts": {
            "present": 5,
            "unavailable": 2,
            "not_applicable": 0,
            "not_matured": 0,
        },
        "missing_information_counts": {
            "pending": 0,
            "provided": 0,
            "unavailable": 1,
        },
        "blocking_counts": {
            "strategy": 0,
            "impact": 0,
            "validation": 0,
            "report_optional": 1,
        },
        "has_strategy_blocker": False,
        "has_impact_blocker": False,
        "has_validation_blocker": False,
    }


def test_report_rejects_non_utf8_surrogate_text_with_typed_error():
    with pytest.raises(StrategyReportBundleError, match="valid UTF-8"):
        _bundle(title="\ud800")

    with pytest.raises(StrategyReportBundleError, match="valid UTF-8"):
        strategy_report_bundle_from_json('{"title":"\ud800"}')


def test_new_information_creates_new_immutable_report_revision():
    first = _bundle()
    second = _bundle(
        report_revision=2,
        previous_report_id=first["report_id"],
        title="风险策略迭代评审（补充版）",
    )

    assert second["previous_report_id"] == first["report_id"]
    assert second["report_id"] != first["report_id"]
    assert second["content_sha256"] != first["content_sha256"]
    with pytest.raises(
        StrategyReportBundleError,
        match="requires valid previous_report_id",
    ):
        _bundle(report_revision=2)


def test_present_zero_stays_zero_while_unavailable_and_not_matured_stay_null():
    bundle = _bundle()
    current = bundle["sections"][0]
    current_fields = {
        item["field_id"]: item["field"] for item in current["summary_fields"]
    }
    monthly = bundle["sections"][5]["tables"][0]["rows"][0]["cells"]

    assert current_fields["approval_rate"]["value"] == 0
    assert current_fields["approval_rate"]["availability"] == "present"
    assert current_fields["business_background"]["value"] is None
    assert current_fields["business_background"]["availability"] == "unavailable"
    assert monthly["approval_rate"]["value"] == 0
    assert monthly["risk_rate"]["value"] is None
    assert monthly["risk_rate"]["availability"] == "not_matured"


def test_final_report_rejects_strategy_blocker_but_allows_optional_blank():
    final = _bundle(status="final")
    assert final["status"] == "final"
    with pytest.raises(StrategyReportBundleError, match="strategy blocker"):
        _bundle(
            status="final",
            missing_information=[_missing(blocking="strategy")],
        )


def test_final_proposed_strategy_report_does_not_claim_adoption_identity():
    bundle = _bundle(
        status="final",
        strategy_id=None,
        strategy_version=None,
        strategy_type="approval",
        missing_information=[],
    )

    assert bundle["status"] == "final"
    assert bundle["strategy_id"] is None
    assert bundle["strategy_version"] is None
    assert bundle["strategy_type"] == "approval"


def test_final_report_counts_field_blockers_and_requires_core_strategy_sections():
    sections = _sections()
    sections[0]["summary_fields"][1]["field"] = build_report_field(
        value=None,
        availability="unavailable",
        origin="repository",
        source_refs=[],
        blocking="strategy",
        note="Strategy scope is unresolved.",
    )
    with pytest.raises(StrategyReportBundleError, match="strategy blocker"):
        _bundle(status="final", sections=sections)

    sections = _sections()
    sections[2] = build_strategy_report_section(
        key="sample_design",
        title="本次样本设计",
        availability="unavailable",
    )
    with pytest.raises(StrategyReportBundleError, match="sample_design"):
        _bundle(status="final", sections=sections)


def test_validation_blocker_cannot_coexist_with_oot_claim():
    sections = _sections()
    impact = sections[5]
    validation_ref = _source(
        "strategy_validation",
        "validation-1",
        _HASH_B,
    )
    impact["tables"][0]["effect_stage"] = "oot_validated"
    impact["tables"][0]["source_refs"] = [validation_ref]
    impact["stage_evidence"] = [
        {
            "effect_stage": "oot_validated",
            "population": "risk",
            "partition": "oot",
            "binding": {
                "kind": "independent_validation",
                "dataset_ref": _source("dataset", "oot-dataset", _HASH_C),
                "frozen_artifact_ref": _source(
                    "strategy",
                    "strategy-1",
                    _HASH_B,
                ),
                "result_ref": validation_ref,
            },
        }
    ]
    with pytest.raises(
        StrategyReportBundleError,
        match="validation blocker cannot claim OOT",
    ):
        _bundle(
            sections=sections,
            missing_information=[_missing(blocking="validation")],
        )


def test_stage_evidence_rejects_role_masquerading_source_kinds():
    sections = _sections()
    binding = sections[5]["stage_evidence"][0]["binding"]
    binding["dataset_ref"] = _source(
        "agent_message",
        "not-a-dataset",
        _HASH_A,
    )
    binding["frozen_artifact_ref"] = _source(
        "agent_message",
        "not-a-strategy",
        _HASH_B,
    )
    binding["result_ref"] = _source(
        "agent_message",
        "not-an-impact",
        _HASH_C,
    )

    with pytest.raises(StrategyReportBundleError, match=r"dataset_ref\.kind"):
        _bundle(sections=sections)


def test_report_rejects_global_source_identity_hash_drift():
    sections = _sections()
    sections[5]["stage_evidence"][0]["binding"]["dataset_ref"] = _source(
        "dataset",
        "dataset-1",
        _HASH_B,
    )

    with pytest.raises(
        StrategyReportBundleError,
        match="global source_ref identity drift",
    ):
        _bundle(sections=sections)


def test_section_order_table_cells_and_effect_stage_are_fail_closed():
    sections = _sections()
    with pytest.raises(StrategyReportBundleError, match="canonical sections in order"):
        _bundle(sections=list(reversed(sections)))

    with pytest.raises(StrategyReportBundleError, match="match declared"):
        build_strategy_report_table(
            table_id="broken",
            title="Broken",
            sheet_key="08_impact",
            granularity="aggregate",
            content_class="metric_summary",
            columns=[
                {
                    "key": "declared",
                    "label": "Declared",
                    "unit": None,
                    "precision": None,
                }
            ],
            rows=[{"row_id": "row-1", "cells": {"other": _present(1)}}],
            source_refs=[_source()],
        )

    with pytest.raises(
        StrategyReportBundleError,
        match="backtested evidence requires a development population",
    ):
        build_strategy_report_section(
            key="impact_assessment",
            title="Impact",
            availability="present",
            stage_evidence=[
                {
                    "effect_stage": "backtested",
                    "population": "risk",
                    "partition": "oot",
                    "binding": {
                        "kind": "development_backtest",
                        "dataset_ref": _source("dataset", "dataset-1"),
                        "frozen_artifact_ref": _source(
                            "strategy",
                            "strategy-1",
                        ),
                        "result_ref": _source(),
                    },
                }
            ],
            source_refs=[_source()],
        )

    with pytest.raises(StrategyReportBundleError, match="granularity"):
        build_strategy_report_table(
            table_id="customer_rows",
            title="Customer rows",
            sheet_key="08_impact",
            granularity="customer",
            content_class="metric_summary",
            columns=[
                {
                    "key": "customer_id",
                    "label": "Customer ID",
                    "unit": None,
                    "precision": None,
                }
            ],
            rows=[],
            source_refs=[],
        )

    with pytest.raises(StrategyReportBundleError, match="finite ratio"):
        build_strategy_report_table(
            table_id="bad_percentage",
            title="Bad percentage",
            sheet_key="08_impact",
            granularity="aggregate",
            content_class="metric_summary",
            columns=[
                {
                    "key": "rate",
                    "label": "Rate",
                    "unit": "%",
                    "precision": 2,
                }
            ],
            rows=[
                {
                    "row_id": "row-1",
                    "cells": {"rate": _present(1.2)},
                }
            ],
            source_refs=[_source()],
        )
    wrong_sheet_table = build_strategy_report_table(
        table_id="wrong_sheet",
        title="Wrong sheet",
        sheet_key="09_economics",
        granularity="aggregate",
        content_class="metric_summary",
        columns=[
            {
                "key": "value",
                "label": "Value",
                "unit": None,
                "precision": None,
            }
        ],
        rows=[],
        source_refs=[],
    )
    with pytest.raises(StrategyReportBundleError, match="not valid for section"):
        build_strategy_report_section(
            key="current_project",
            title="Current",
            availability="present",
            tables=[wrong_sheet_table],
            source_refs=[_source()],
        )
    impact_source = _source("strategy_impact", "impact-1", _HASH_A)
    unrelated_source = _source("strategy_impact", "impact-2", _HASH_B)
    unbound_stage_table = build_strategy_report_table(
        table_id="unbound_stage",
        title="Unbound stage",
        sheet_key="08_impact",
        granularity="aggregate",
        content_class="metric_summary",
        effect_stage="backtested",
        columns=[
            {
                "key": "value",
                "label": "Value",
                "unit": None,
                "precision": None,
            }
        ],
        rows=[],
        source_refs=[unrelated_source],
    )
    with pytest.raises(StrategyReportBundleError, match="not bound"):
        build_strategy_report_section(
            key="impact_assessment",
            title="Impact",
            availability="present",
            tables=[unbound_stage_table],
            stage_evidence=[
                {
                    "effect_stage": "backtested",
                    "population": "risk",
                    "partition": "development",
                    "binding": {
                        "kind": "development_backtest",
                        "dataset_ref": _source("dataset", "dataset-1"),
                        "frozen_artifact_ref": _source(
                            "strategy",
                            "strategy-1",
                        ),
                        "result_ref": impact_source,
                    },
                }
            ],
            source_refs=[impact_source],
        )
    with pytest.raises(
        StrategyReportBundleError,
        match="requires a validation or oot population",
    ):
        build_strategy_report_section(
            key="impact_assessment",
            title="Impact",
            availability="present",
            stage_evidence=[
                {
                    "effect_stage": "oot_validated",
                    "population": "risk",
                    "partition": "development",
                    "binding": {
                        "kind": "independent_validation",
                        "dataset_ref": _source("dataset", "dataset-1"),
                        "frozen_artifact_ref": _source(
                            "strategy",
                            "strategy-1",
                        ),
                        "result_ref": _source(),
                    },
                }
            ],
            source_refs=[_source()],
        )


def test_hash_id_source_task_and_json_parser_tampering_are_rejected():
    bundle = _bundle()
    drifted = deepcopy(bundle)
    drifted["sections"][0]["summary_fields"][0]["field"]["value"] = 0.1
    with pytest.raises(StrategyReportBundleError, match="report_id"):
        validate_strategy_report_bundle(drifted)

    bad_hash = {**bundle, "content_sha256": _HASH_A}
    with pytest.raises(StrategyReportBundleError, match="content_sha256"):
        validate_strategy_report_bundle(bad_hash)

    with pytest.raises(StrategyReportBundleError, match="another task"):
        _bundle(missing_information=[_missing(task_id="task-2")])

    with pytest.raises(StrategyReportBundleError, match="duplicate key"):
        strategy_report_bundle_from_json('{"schema_version":1,"schema_version":2}')
    with pytest.raises(StrategyReportBundleError, match="non-finite constant"):
        strategy_report_bundle_from_json('{"value":NaN}')


@pytest.mark.parametrize(
    ("constant", "limit", "message"),
    [
        ("MAX_REPORT_FIELDS", 1, "fields exceed budget"),
        ("MAX_REPORT_REFS", 1, "references exceed budget"),
    ],
)
def test_report_resource_budgets_are_global(
    monkeypatch,
    constant: str,
    limit: int,
    message: str,
):
    bundle = _bundle()
    monkeypatch.setattr(report_contract, constant, limit)

    with pytest.raises(StrategyReportBundleError, match=message):
        validate_strategy_report_bundle(bundle)


@pytest.mark.parametrize(
    ("constant", "message"),
    [
        ("MAX_REPORT_TABLES", "strategy report tables exceed budget"),
        ("MAX_REPORT_ROWS", "strategy report rows exceed budget"),
    ],
)
def test_table_and_row_budgets_apply_across_sections(
    monkeypatch,
    constant: str,
    message: str,
):
    sections = _sections()
    second_table = deepcopy(sections[5]["tables"][0])
    second_table["table_id"] = "current_monthly"
    second_table["sheet_key"] = "01_current_state"
    second_table["effect_stage"] = None
    second_table["rows"][0]["row_id"] = "current-2026-06"
    sections[0]["tables"] = [second_table]
    bundle = _bundle(sections=sections)
    monkeypatch.setattr(report_contract, constant, 1)

    with pytest.raises(StrategyReportBundleError, match=message):
        validate_strategy_report_bundle(bundle)


def test_json_depth_node_and_byte_budgets_are_enforced(monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(report_contract, "MAX_REPORT_BUNDLE_JSON_DEPTH", 1)
    with pytest.raises(StrategyReportBundleError, match="depth budget"):
        canonical_strategy_report_bundle_json(bundle)

    monkeypatch.setattr(report_contract, "MAX_REPORT_BUNDLE_JSON_DEPTH", 40)
    monkeypatch.setattr(report_contract, "MAX_REPORT_BUNDLE_JSON_NODES", 1)
    with pytest.raises(StrategyReportBundleError, match="node budget"):
        canonical_strategy_report_bundle_json(bundle)

    monkeypatch.setattr(report_contract, "MAX_REPORT_BUNDLE_JSON_BYTES", 1)
    with pytest.raises(StrategyReportBundleError, match="byte budget"):
        strategy_report_bundle_from_json("{}")


def test_cycles_and_parser_recursion_fail_with_typed_errors():
    cyclic = deepcopy(_bundle())
    cyclic["sections"][0]["summary_fields"][0]["field"]["value"] = cyclic
    with pytest.raises(StrategyReportBundleError, match="cycle"):
        validate_strategy_report_bundle(cyclic)

    raw = ("[" * 2_000) + "0" + ("]" * 2_000)
    with pytest.raises(
        StrategyReportBundleError,
        match="not valid JSON|must be an object",
    ):
        strategy_report_bundle_from_json(raw)
