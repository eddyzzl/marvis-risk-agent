from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output() -> dict:
    return {
        "schema_version": "strategy.build-voting-candidate-tool.v1",
        "asset_id": "candidate-asset-" + "a" * 32,
        "asset_hash": "b" * 64,
        "candidate_id": "candidate-" + "c" * 32,
        "evidence_hash": "d" * 64,
        "rule_id": "candidate-rule-" + "e" * 32,
        "rule_hash": "f" * 64,
        "fragment_id": "candidate-fragment-" + "1" * 32,
        "fragment_hash": "2" * 64,
        "effect_id": "candidate-effect-" + "3" * 32,
        "effect_hash": "4" * 64,
        "pool_id": "strategy-pool-task-approval",
        "revision": 4,
        "snapshot_hash": "5" * 64,
        "selected_entries": [
            {
                "entry_id": "pool-entry-1",
                "rule_id": "candidate-rule-source-1",
                "pool_position": 0,
            },
            {
                "entry_id": "pool-entry-2",
                "rule_id": "candidate-rule-source-2",
                "pool_position": 1,
            },
            {
                "entry_id": "pool-entry-3",
                "rule_id": "candidate-rule-source-3",
                "pool_position": 2,
            },
        ],
        "n": 2,
        "k": 3,
        "dataset_id": "dataset-voting-source",
        "target_col": "bad",
        "population_count": 100,
        "labeled_count": 100,
        "drop_nan_labels": False,
        "nan_labels_dropped": 0,
        "effect": {
            "population_count": 100,
            "labeled_count": 100,
            "matched_count": 20,
            "matched_rate": 0.2,
            "matched_bad_count": 8,
            "matched_bad_rate": 0.4,
            "unmatched_count": 80,
            "unmatched_bad_count": 12,
            "unmatched_bad_rate": 0.15,
            "bad_capture_rate": 0.4,
            "lift": 2.0,
        },
        "metrics": {"metrics_hash": "6" * 64},
        "hit_distribution": [
            {
                "hit_count": 0,
                "count": 50,
                "share": 0.5,
                "bad_count": 5,
                "bad_rate": 0.1,
                "lift": 0.5,
            },
            {
                "hit_count": 1,
                "count": 30,
                "share": 0.3,
                "bad_count": 7,
                "bad_rate": 7 / 30,
                "lift": 7 / 6,
            },
            {
                "hit_count": 2,
                "count": 15,
                "share": 0.15,
                "bad_count": 6,
                "bad_rate": 0.4,
                "lift": 2.0,
            },
            {
                "hit_count": 3,
                "count": 5,
                "share": 0.05,
                "bad_count": 2,
                "bad_rate": 0.4,
                "lift": 2.0,
            },
        ],
        "metric_observations": [
            {
                "metric_name": "voting.hit_share",
                "dimension": "loan_amount",
                "status": "observed",
                "value": 0.25,
            },
            {
                "metric_name": "voting.bad_capture_rate",
                "dimension": "loan_amount",
                "status": "insufficient_data",
                "value": None,
            },
            {
                "metric_name": "voting.hit_share",
                "dimension": "overdue_amount",
                "status": "unavailable",
                "value": None,
            },
            {
                "metric_name": "voting.bad_capture_rate",
                "dimension": "overdue_amount",
                "status": "observed",
                "value": 0.42,
            },
            {
                "metric_name": "voting.hit_count.0.share",
                "dimension": "loan_amount",
                "status": "observed",
                "value": 0.55,
            },
        ],
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
        "artifacts": [
            {
                "artifact_id": "artifact-voting-json",
                "kind": "strategy_voting_candidate_json",
                "format": "json",
                "filename": "voting.json",
                "content_hash": "7" * 64,
                "download_url": "/api/tasks/t/task-artifacts/a/download",
            }
        ],
    }


def test_voting_renderer_labels_candidate_boundary_and_surfaces_lineage() -> None:
    output = _output()

    text, tables = render_tool_output("build_voting_candidate", output)

    assert output["asset_id"] in text
    assert output["asset_hash"] in text
    assert "2-of-3" in text
    assert "revision 4" in text
    assert output["snapshot_hash"] in text
    assert "仅生成候选" in text
    assert "尚未入池" in text
    assert "未应用写回" in text
    assert "未采纳" in text
    assert "未部署" in text
    assert "绑定样本口径" in text
    assert "dataset-voting-source" in text
    assert "target `bad`" in text
    assert "population 100" in text
    assert "labeled 100" in text
    assert "drop_nan_labels `false`" in text
    assert "nan_labels_dropped 0" in text
    assert "绑定样本观测（未独立验证）" in text
    assert "确定性效果" not in text
    assert "20.0%" in text
    assert "40.0%" in text
    assert "voting.json" in text
    assert "/api/tasks/t/task-artifacts/a/download" in text
    assert [table["title"] for table in tables] == [
        "Voting 成员规则（按 Pool position）",
        "Voting 命中数分布",
        "Voting 金额维度关键观测",
    ]
    assert [row[1] for row in tables[0]["rows"]] == [
        "candidate-rule-source-1",
        "candidate-rule-source-2",
        "candidate-rule-source-3",
    ]
    assert [row[0] for row in tables[1]["rows"]] == ["0", "1", "2", "3"]


def test_voting_renderer_preserves_amount_observation_statuses_and_values() -> None:
    text, tables = render_tool_output("build_voting_candidate", _output())

    amount_table = next(
        table for table in tables if table["title"] == "Voting 金额维度关键观测"
    )
    assert amount_table["columns"] == ["金额维度", "指标", "状态", "观测值"]
    assert amount_table["rows"] == [
        ["放款金额", "Voting 命中金额占比", "observed", "25.0%"],
        ["放款金额", "坏样本捕获金额占比", "insufficient_data", "n/a"],
        ["逾期金额", "Voting 命中金额占比", "unavailable", "n/a"],
        ["逾期金额", "坏样本捕获金额占比", "observed", "42.0%"],
    ]
    assert "金额维度观测状态和值见下表" in text
    assert all("hit_count.0" not in str(row) for row in amount_table["rows"])


def test_voting_renderer_reports_dropped_label_sample_scope() -> None:
    output = _output()
    output.update(
        {
            "population_count": 100,
            "labeled_count": 94,
            "drop_nan_labels": True,
            "nan_labels_dropped": 6,
        }
    )

    text, _ = render_tool_output("build_voting_candidate", output)

    assert "population 100" in text
    assert "labeled 94" in text
    assert "drop_nan_labels `true`" in text
    assert "nan_labels_dropped 6" in text


def test_voting_renderer_keeps_null_rates_explicitly_unavailable() -> None:
    output = _output()
    output["effect"] = {
        **output["effect"],
        "matched_count": 0,
        "matched_rate": 0.0,
        "matched_bad_count": 0,
        "matched_bad_rate": None,
        "bad_capture_rate": None,
        "lift": None,
    }

    text, _ = render_tool_output("build_voting_candidate", output)

    assert "命中坏率 n/a" in text
    assert "坏样本捕获率 n/a" in text
    assert "Lift n/a" in text
