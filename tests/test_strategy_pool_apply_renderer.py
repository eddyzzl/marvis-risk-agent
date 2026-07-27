"""Renderer contract for governed current-Pool application output."""

from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output() -> dict:
    return {
        "schema_version": "strategy.apply-strategy-pool-tool.v1",
        "run_id": "spar_" + "1" * 32,
        "input_hash": "2" * 64,
        "cached": False,
        "activated": False,
        "adopted": False,
        "deployed": False,
        "source": {
            "pool_id": "strategy-pool-" + "3" * 32,
            "revision": 7,
            "revision_id": "strategy-pool-revision-" + "4" * 32,
            "snapshot_hash": "5" * 64,
            "pool_artifact_id": "artifact-pool-1",
            "pool_artifact_content_hash": "6" * 64,
            "design_hash": "7" * 64,
            "strategy_spec_hash": "8" * 64,
            "dataset_id": "dataset-source",
            "dataset_content_hash": "9" * 64,
            "row_count": 3,
            "sample_design_ref": {
                "artifact_id": "artifact-sample-1",
                "artifact_content_hash": "a" * 64,
                "sample_design_id": "strategy-sample-design-1",
                "sample_design_content_hash": "b" * 64,
                "partition": "development",
            },
        },
        "result": {
            "dataset_id": "dataset-derived",
            "dataset_content_hash": "c" * 64,
            "row_count": 3,
            "result_hash": "d" * 64,
        },
        "columns": {
            "action": "decision_action",
            "value": "decision_value",
            "value_type": "decision_value_type",
            "rule_id": "decision_rule_id",
            "entry_id": "decision_entry_id",
            "reason_code": "decision_reason_code",
        },
        "action_counts": {"approval": 2, "reject": 1},
        "rule_counts": {"candidate-rule-1": 1},
        "entry_counts": {"pool-entry-1": 1},
        "default_count": 2,
        "requirements": {
            "requirements_hash": "e" * 64,
            "virtual_fields": [],
        },
        "workspace": {
            "source_revision": 4,
            "source_analysis_generation": 9,
            "source_semantic_mapping_hash": "f" * 64,
            "active_dataset_id": "dataset-source",
            "result_revision": None,
            "result_analysis_generation": None,
        },
        "evidence": {
            "artifact_id": "artifact-evidence-1",
            "content_hash": "0" * 64,
            "download_url": (
                "/api/tasks/task-1/task-artifacts/artifact-evidence-1/download"
            ),
        },
    }


def test_pool_apply_renderer_surfaces_derived_dataset_and_inactive_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "marvis.packs.strategy.pool_apply_tools."
        "validate_apply_strategy_pool_tool_output",
        lambda value: value,
    )

    text, tables = render_tool_output("apply_strategy_pool", _output())

    assert "Strategy Pool 应用完成" in text
    assert "dataset-source" in text
    assert "dataset-derived" in text
    assert "保留 **3** 行" in text
    assert "当前 workspace 未切换" in text
    assert "未激活、未采纳、未部署" in text
    assert "不会修改当前 Strategy Pool" in text
    assert "/api/tasks/task-1/task-artifacts/artifact-evidence-1/download" in text
    identity = next(
        table for table in tables if table["title"] == "Strategy Pool 应用身份"
    )
    assert ["Pool Revision", "7"] in identity["rows"]
    assert ["Result Dataset", "dataset-derived"] in identity["rows"]
    columns = next(
        table for table in tables if table["title"] == "派生数据集输出列"
    )
    assert ["action", "decision_action"] in columns["rows"]
    counts = next(
        table for table in tables if table["title"] == "Strategy Pool 应用分布"
    )
    assert ["action", "approval", "2"] in counts["rows"]
    assert ["default", "unmatched", "2"] in counts["rows"]


def test_pool_apply_renderer_fails_closed_on_validator_error(
    monkeypatch,
) -> None:
    def _reject(_value):
        raise ValueError("tampered")

    monkeypatch.setattr(
        "marvis.packs.strategy.pool_apply_tools."
        "validate_apply_strategy_pool_tool_output",
        _reject,
    )
    payload = _output()
    payload["result"]["row_count"] = 999999

    text, tables = render_tool_output("apply_strategy_pool", payload)

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert tables == []
