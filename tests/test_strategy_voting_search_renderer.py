"""Safe fixed projection for aggregate Voting search evidence."""

from __future__ import annotations

import hashlib

from marvis.agent.renderers import render_tool_output
from marvis.packs.strategy.voting_candidate_search import (
    VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
    canonical_voting_candidate_search_result_json,
    search_voting_candidate_combinations,
)


def _output(*, truncated: bool = True) -> dict:
    candidate_ids = [f"candidate-rule-{index:032x}" for index in range(1, 7)]
    result = search_voting_candidate_combinations(
        {
            "schema_version": VOTING_CANDIDATE_SEARCH_REQUEST_SCHEMA_VERSION,
            "candidate_ids": candidate_ids,
            "hit_matrix": [
                [((row + index) % (index + 2)) == 0 for row in range(10)]
                for index in range(1, 7)
            ],
            "target": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "weights": None,
            "amounts": [100.0 * (index + 1) for index in range(10)],
            "member_count": 2,
            "n": 1,
            "objective": {
                "metric": "bad_rate",
                "direction": "minimize",
            },
            "constraints": [],
            "include": [],
            "exclude": [],
            "max_combinations": 13 if truncated else 15,
        }
    )
    canonical = canonical_voting_candidate_search_result_json(result).encode(
        "utf-8"
    )
    artifact_hash = hashlib.sha256(canonical).hexdigest()
    return {
        "schema_version": "strategy.search-voting-candidates-tool.v1",
        "search_id": result["search_id"],
        "request_hash": result["request_hash"],
        "content_hash": result["content_hash"],
        "pool_id": "strategy-pool-task-approval",
        "pool_revision": 7,
        "pool_snapshot_hash": "d" * 64,
        "search_space": result["search_space"],
        "evaluated": result["evaluated"],
        "truncated": result["truncated"],
        "eligible": result["eligible"],
        "excluded_unsupported_rule_ids": [
            "candidate-rule-" + "f" * 32,
        ],
        "search_result": result,
        "artifacts": [
            {
                "artifact_id": "e" * 64,
                "kind": "strategy_voting_candidate_search_json",
                "format": "json",
                "filename": "voting-search.json",
                "content_hash": artifact_hash,
                "download_url": (
                    "/api/tasks/t/task-artifacts/"
                    + "e" * 64
                    + f"/download?expected_content_hash={artifact_hash}"
                ),
            }
        ],
        "not_mutated_pool": True,
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def test_voting_search_renderer_shows_only_first_ten_eligible_aggregate_rows() -> None:
    output = _output()
    expected = [
        item
        for item in output["search_result"]["combinations"]
        if item["eligible"] is True
    ][:10]

    text, tables = render_tool_output("search_voting_candidates", output)

    assert output["search_id"] in text
    assert "search_space 15" in text
    assert "evaluated 13" in text
    assert "eligible 13" in text
    assert "只读搜索" in text
    assert "未修改 Pool" in text
    assert "未选择" in text
    assert "未入池" in text
    assert "未采纳、未部署" in text
    assert output["artifacts"][0]["content_hash"] in text
    assert "统一产物栏" in text
    assert "voting-search.json" not in text
    assert "/api/tasks/" not in text
    assert output["pool_id"] not in text
    assert output["pool_snapshot_hash"] not in text
    assert output["excluded_unsupported_rule_ids"][0] not in text
    assert [table["title"] for table in tables] == [
        "Voting 搜索已评估范围内符合约束的 Top-10"
    ]
    rows = tables[0]["rows"]
    assert len(rows) == 10
    assert [row[0] for row in rows] == [str(item["rank"]) for item in expected]
    assert [row[1] for row in rows] == [item["combo_id"] for item in expected]
    assert all("candidate-rule-" in row[2] for row in rows)
    assert all(row[3] == "1" for row in rows)
    assert "weighted_bad_rate" not in tables[0]["columns"]
    assert "hit_matrix" not in text
    assert "forged-winner" not in text
    assert "target" not in text


def test_voting_search_renderer_discloses_canonical_budget_prefix_without_superlative() -> (
    None
):
    text, _ = render_tool_output("search_voting_candidates", _output(truncated=True))

    assert ("按 canonical 组合顺序评估预算前缀，以下只是已评估范围内 Top-N") in text
    for forbidden in ("winner", "champion", "全局最佳", "冠军", "获胜"):
        assert forbidden not in text


def test_voting_search_renderer_does_not_claim_truncation_for_full_search() -> None:
    text, _ = render_tool_output("search_voting_candidates", _output(truncated=False))

    assert "按 canonical 组合顺序评估预算前缀" not in text
    assert "已完整评估当前搜索空间" in text


def test_voting_search_renderer_rejects_tampered_inner_metrics() -> None:
    output = _output()
    output["search_result"]["combinations"][1]["metrics"]["bad_rate"] = 0.99

    text, tables = render_tool_output("search_voting_candidates", output)

    assert "完整性校验失败" in text
    assert "搜索完成" not in text
    assert "已完整评估" not in text
    assert tables == []


def test_voting_search_renderer_rejects_outer_summary_or_artifact_drift() -> None:
    outputs = [_output(), _output(), _output()]
    outputs[0]["evaluated"] += 1
    outputs[1]["truncated"] = not outputs[1]["truncated"]
    outputs[2]["artifacts"][0]["content_hash"] = "0" * 64

    for output in outputs:
        text, tables = render_tool_output("search_voting_candidates", output)

        assert "完整性校验失败" in text
        assert "搜索完成" not in text
        assert tables == []


def test_voting_search_renderer_never_projects_unbound_outer_provenance() -> None:
    output = _output()
    output["pool_id"] = "forged-pool"
    output["excluded_unsupported_rule_ids"] = [
        "candidate-rule-" + "9" * 32,
    ]
    output["artifacts"][0]["artifact_id"] = "0" * 64
    output["artifacts"][0]["download_url"] = (
        "/api/tasks/other/task-artifacts/"
        + "0" * 64
        + "/download?expected_content_hash="
        + output["artifacts"][0]["content_hash"]
    )

    text, tables = render_tool_output("search_voting_candidates", output)

    assert tables
    assert "forged-pool" not in text
    assert output["excluded_unsupported_rule_ids"][0] not in text
    assert "/api/tasks/other/" not in text
    assert "0" * 64 not in text
