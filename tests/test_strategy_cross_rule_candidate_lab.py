from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.packs.strategy.cross_rule_search_tools import (
    run_build_cross_rule_candidate_from_search,
    run_search_cross_threshold_rules,
)
from tests.test_strategy_cross_matrix_candidate_tool import _setup
from tests.test_strategy_cross_rule_search_tools import _inputs


def test_candidate_lab_projects_cross_rule_search_candidate_and_pool_source(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    searched = run_search_cross_threshold_rules(
        _inputs(fixture, max_trials=4),
        fixture["ctx"],
        fixture["runtime"],
    )
    selected = searched["search_result"]["rules"][0]
    built = run_build_cross_rule_candidate_from_search(
        {
            "search_id": searched["search_id"],
            "rule_id": selected["rule_id"],
            "selection_reason": "人工评审后用于候选实验。",
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    response = TestClient(create_app(fixture["settings"])).get(
        f"/api/tasks/{fixture['task'].id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "strategy.candidate-lab-projection.v9"
    [search] = body["candidates"]["cross_rule_search"]["all"]
    assert search["search_id"] == searched["search_id"]
    assert search["dimension"] == 2
    assert search["evaluated"] == 4
    assert search["rules"][0]["rule_id"]
    assert "winner" not in json.dumps(search, ensure_ascii=False)

    [candidate] = body["candidates"]["cross_rule_candidate"]["all"]
    assert candidate["candidate_id"] == built["candidate"]["asset_id"]
    assert candidate["detail"]["rule_id"] == selected["rule_id"]
    assert candidate["detail"]["selection_reason"] == (
        "人工评审后用于候选实验。"
    )
    pool_source = next(
        item
        for item in body["pool_add_sources"]["all"]
        if item["source_kind"] == "cross_threshold_rule"
    )
    assert pool_source["candidate_asset_id"] == (
        built["candidate"]["asset_id"]
    )
