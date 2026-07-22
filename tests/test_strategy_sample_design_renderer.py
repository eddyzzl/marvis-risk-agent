from __future__ import annotations

import hashlib

import pandas as pd

from marvis.agent.renderers import render_tool_output
from marvis.packs.strategy.sample_design import (
    build_strategy_sample_design_bundle,
    canonical_strategy_sample_design_bundle_json,
)


def _bundle(
    *,
    exploration: bool = False,
    split: bool = True,
    empty_downstream_splits: bool = False,
    maturity: str | None = None,
) -> dict:
    frame = pd.DataFrame(
        {
            "bad": [0, 1, 0, 1, 0, 1],
            "sample_role": ["dev", "dev", "validation", "validation", "oot", "oot"],
        }
    )
    if empty_downstream_splits:
        frame["sample_role"] = "dev"
    return build_strategy_sample_design_bundle(
        frame=frame,
        task_id="task-1",
        dataset_id="dataset-1",
        dataset_content_hash="a" * 64,
        workspace_revision=3,
        workspace_generation=2,
        semantic_mapping_hash="b" * 64,
        target_col="bad",
        target_bad_value=1,
        drop_nan_labels=False,
        performance_window=(
            {"status": "unavailable", "days": None}
            if exploration
            else {"status": "provided", "days": 90}
        ),
        observation_window=(
            {"status": "unavailable", "start": None, "end": None}
            if exploration
            else {
                "status": "provided",
                "start": "2025-01-01",
                "end": "2025-12-31",
            }
        ),
        split_definition=(
            {
                "status": "available",
                "column": "sample_role",
                "development_values": ["dev"],
                "validation_values": []
                if empty_downstream_splits
                else ["validation"],
                "oot_values": [] if empty_downstream_splits else ["oot"],
            }
            if split
            else {
                "status": "unavailable",
                "column": None,
                "development_values": [],
                "validation_values": [],
                "oot_values": [],
            }
        ),
        maturity=maturity or ("unknown" if exploration else "confirmed_matured"),
    )


def _output(bundle: dict) -> dict:
    design = bundle["sample_design"]
    artifact_hash = hashlib.sha256(
        canonical_strategy_sample_design_bundle_json(bundle).encode("utf-8")
    ).hexdigest()
    artifact_id = "c" * 64
    return {
        "schema_version": "strategy.materialize-sample-design-tool.v1",
        "sample_design_id": design["sample_design_id"],
        "content_hash": design["content_hash"],
        "bundle": bundle,
        "warnings": [flag["message"] for flag in design["red_flags"]],
        "artifact": {
            "artifact_id": artifact_id,
            "kind": "strategy_sample_design_json",
            "format": "json",
            "filename": f"{design['sample_design_id']}.json",
            "content_hash": artifact_hash,
            "download_url": (
                f"/api/tasks/task-1/task-artifacts/{artifact_id}/download"
                f"?expected_content_hash={artifact_hash}"
            ),
        },
        "development": True,
        "unvalidated": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def test_sample_design_renderer_uses_strict_bundle_metrics_and_no_lifecycle_claims() -> None:
    text, tables = render_tool_output(
        "materialize_sample_design",
        _output(_bundle()),
    )

    assert "策略样本设计已固化" in text
    assert "活动样本 **6** 行" in text
    assert "provided / 90 天" in text
    assert "development / unvalidated" in text
    assert "坏样本值 `1`，好样本值 `0`" in text
    assert "未创建或修改策略" in text
    assert "未建模、未建树、未入池、未采纳、未部署" in text
    assert f"{'c' * 64}/download?expected_content_hash=" in text
    overall = next(
        table for table in tables if table["title"] == "策略样本设计 Overall 指标"
    )
    population = next(row for row in overall["rows"] if row[1] == "population_count")
    bad_rate = next(row for row in overall["rows"] if row[1] == "bad_rate")
    loan_amount = next(
        row for row in overall["rows"] if row[1] == "loan_amount_sum"
    )
    assert population[3:8] == ["present", "6", "n/a", "n/a", "6"]
    assert bad_rate[3:8] == ["present", "50.0%", "3", "6", "6"]
    assert loan_amount[3:8] == ["unavailable", "n/a", "n/a", "n/a", "6"]
    split = next(
        table
        for table in tables
        if table["title"] == "策略样本设计开发/验证/OOT 指标"
    )
    assert {row[0] for row in split["rows"]} == {
        "development",
        "validation",
        "oot",
    }


def test_sample_design_renderer_surfaces_exploration_unavailable_and_red_flags() -> None:
    text, tables = render_tool_output(
        "materialize_sample_design",
        _output(_bundle(exploration=True, split=False)),
    )

    assert "表现窗 `unavailable`" in text
    assert "观察窗 `unavailable`" in text
    assert "`exploration-only`" in text
    assert "切分 unavailable" in text
    flags = next(table for table in tables if table["title"] == "策略样本设计红旗")
    codes = {row[1] for row in flags["rows"]}
    assert "performance_window_unavailable" in codes
    assert "observation_window_unavailable" in codes
    assert "split_unavailable" in codes
    assert "sample_maturity_unknown" in codes
    assert not any(
        table["title"] == "策略样本设计开发/验证/OOT 指标" for table in tables
    )


def test_sample_design_renderer_does_not_show_fake_zero_for_undefined_splits() -> None:
    _, tables = render_tool_output(
        "materialize_sample_design",
        _output(_bundle(empty_downstream_splits=True)),
    )
    split = next(
        table
        for table in tables
        if table["title"] == "策略样本设计开发/验证/OOT 指标"
    )
    assert {row[0] for row in split["rows"]} == {"development"}


def test_sample_design_renderer_marks_not_matured_scope_as_exploration_only() -> None:
    text, tables = render_tool_output(
        "materialize_sample_design",
        _output(_bundle(maturity="not_matured")),
    )

    assert "成熟度 `not_matured`" in text
    assert "`exploration-only`" in text
    flags = next(table for table in tables if table["title"] == "策略样本设计红旗")
    assert "sample_not_matured" in {row[1] for row in flags["rows"]}


def test_sample_design_renderer_never_uses_generic_scalar_fallback_on_drift() -> None:
    text, tables = render_tool_output(
        "materialize_sample_design",
        {"sample_design_id": "forged", "population_count": 999999},
    )

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert tables == []


def test_sample_design_renderer_keeps_dedicated_failure_on_unexpected_validator_error(
    monkeypatch,
) -> None:
    def _unexpected(_value):
        raise RuntimeError("unexpected validator failure")

    monkeypatch.setattr(
        "marvis.packs.strategy.sample_design_tools."
        "validate_materialize_sample_design_tool_output",
        _unexpected,
    )

    text, tables = render_tool_output(
        "materialize_sample_design",
        {"population_count": 999999},
    )

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert tables == []
