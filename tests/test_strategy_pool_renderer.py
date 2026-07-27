"""Strategy Pool mutation and compile renderer contracts."""

from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _entry(index: int) -> dict:
    return {
        "entry_id": f"pool-entry-{index}",
        "rule_id": f"candidate-rule-{index}",
        "candidate_asset_id": f"candidate-asset-{index}",
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "action": {
            "type": "reject" if index == 1 else "review",
            "value": "reject" if index == 1 else "review",
            "reason_code": None,
            "stop": True,
        },
        "effect": {
            "selected_count": 10 * index,
            "selected_share": 0.1 * index,
            "bad_rate": 0.2 * index,
            "lift": 1.0 + index,
        },
    }


def _mutation_output() -> dict:
    return {
        "schema_version": "strategy.candidate-pool-tool.v1",
        "operation": "reorder",
        "pool_id": "strategy-pool-1",
        "revision": 4,
        "snapshot_hash": "a" * 64,
        "status": "draft",
        "validation_status": "unvalidated",
        "entry_count": 2,
        "entries": [_entry(2), _entry(1)],
        "pool": {"effect_stage": "development"},
        "artifacts": [
            {
                "artifact_id": "artifact-pool-4",
                "kind": "strategy_candidate_pool_json",
                "filename": "pool-v4.json",
                "content_hash": "b" * 64,
                "download_url": "/api/tasks/t/task-artifacts/a/download",
            }
        ],
    }


def test_mutation_renderer_shows_complete_order_and_draft_governance_state() -> None:
    text, tables = render_tool_output("reorder_strategy_pool", _mutation_output())

    assert "revision 4" in text
    assert "a" * 64 in text
    assert "development / unvalidated" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert "/api/tasks/t/task-artifacts/a/download" in text
    ordered = next(table for table in tables if table["title"] == "Strategy Pool 完整顺序")
    assert [row[1] for row in ordered["rows"]] == [
        "candidate-rule-2",
        "candidate-rule-1",
    ]
    assert [row[2] for row in ordered["rows"]] == ["pool-entry-2", "pool-entry-1"]


def test_all_pool_mutations_share_the_governed_renderer() -> None:
    for tool in (
        "add_candidate_to_pool",
        "remove_pool_entry",
        "set_pool_entry_action",
        "reorder_strategy_pool",
    ):
        text, tables = render_tool_output(tool, _mutation_output())
        assert "Strategy Pool" in text
        assert tables


def test_voting_pool_placement_operation_is_explained() -> None:
    before = _mutation_output()
    before["operation"] = "insert_candidate_before_entries"
    text, _ = render_tool_output("add_candidate_to_pool", before)
    assert "最早位置之前" in text
    assert "原成员保留" in text

    replaced = _mutation_output()
    replaced["operation"] = "replace_entries_with_candidate"
    text, _ = render_tool_output("add_candidate_to_pool", replaced)
    assert "原子 revision" in text
    assert "替代所选成员" in text


def test_mutation_renderer_reads_governance_fields_from_canonical_source() -> None:
    output = _mutation_output()
    output["entries"] = [
        {
            "entry_id": "pool-entry-canonical",
            "rule_id": "candidate-rule-canonical",
            "position": 0,
            "source": {
                "asset_id": "candidate-asset-canonical",
                "effect_stage": "development",
                "validation_status": "unvalidated",
            },
            "execution": {"condition": {"op": "is_null", "field": "income"}},
            "action": {"type": "review", "value": "review", "stop": True},
            "enabled": True,
        }
    ]

    _, tables = render_tool_output("add_candidate_to_pool", output)

    row = tables[0]["rows"][0]
    assert row[3] == "candidate-asset-canonical"
    assert row[-1] == "development / unvalidated"


def test_compile_renderer_labels_spec_as_unadopted_design_only() -> None:
    output = {
        "schema_version": "strategy.compile-candidate-pool-tool.v2",
        "pool_id": "strategy-pool-1",
        "revision": 4,
        "snapshot_hash": "a" * 64,
        "requirements": [],
        "strategy_spec": {
            "schema_version": "strategy.dsl.v1",
            "strategy_type": "approval",
            "match_policy": "first_match",
            "default_action": {"type": "approval", "value": "approve"},
            "rules": [
                {
                    "rule_id": "candidate-rule-1",
                    "priority": 10,
                    "condition": {"op": "is_null", "field": "income"},
                    "action": {"type": "reject", "value": "reject"},
                }
            ],
            "metadata": {"lineage": {"pool_id": "strategy-pool-1"}},
        },
        "source_entry_refs": [
            {"entry_id": "pool-entry-1", "rule_id": "candidate-rule-1"}
        ],
        "design_hash": "c" * 64,
        "selected_strategy_design": {"status": "draft"},
        "artifacts": [
            {
                "artifact_id": "artifact-pool-4",
                "kind": "strategy_candidate_pool_json",
                "filename": "pool-v4.json",
                "content_hash": "b" * 64,
                "download_url": "/api/tasks/t/task-artifacts/a/download",
            }
        ],
    }

    text, tables = render_tool_output("compile_strategy_pool", output)

    assert "StrategySpec" in text
    assert "c" * 64 in text
    assert "只读草案" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert any(table["title"] == "编译后的 StrategySpec 规则" for table in tables)
