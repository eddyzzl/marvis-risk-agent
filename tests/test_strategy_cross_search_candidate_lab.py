from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.packs.strategy.cross_candidate_search_tools import (
    run_search_cross_matrix_candidates,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_cross_candidate_search_tools import _search_inputs
from tests.test_strategy_cross_matrix_candidate_tool import _setup


def test_candidate_lab_projects_authenticated_cross_search_pairs_without_hashes(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    searched = run_search_cross_matrix_candidates(
        _search_inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )
    [expected_pair] = searched["search_result"]["pairs"]

    response = TestClient(create_app(fixture["settings"])).get(
        f"/api/tasks/{fixture['task'].id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "strategy.candidate-lab-projection.v9"
    collection = body["candidates"]["cross_search"]
    assert collection["total"] == 1
    assert collection["truncated"] is False
    [projected] = collection["all"]
    assert projected == {
        "search_id": searched["search_id"],
        "features": [
            {
                "feature": item["feature"],
                "method": item["method"],
                "axis_iv": item["axis_iv"],
                "bin_count": item["bin_count"],
            }
            for item in searched["search_result"]["configuration"]["features"]
        ],
        "max_pairs": searched["search_result"]["configuration"]["max_pairs"],
        "search_space": 1,
        "evaluated": 1,
        "eligible": searched["search_result"]["eligible"],
        "truncated": False,
        "pairs": [
            {
                key: expected_pair[key]
                for key in (
                    "pair_id",
                    "x_feature",
                    "x_method",
                    "y_feature",
                    "y_method",
                    "x_axis_iv",
                    "y_axis_iv",
                    "cross_total_iv",
                    "interaction_gain_iv",
                    "cell_count",
                    "empty_cell_count",
                    "empty_cell_share",
                    "min_nonempty_cell_count",
                    "eligible",
                    "rank",
                )
            }
        ],
        "artifact": {
            "artifact_id": searched["artifacts"][0]["artifact_id"],
            "created_at": projected["artifact"]["created_at"],
            "download_url": searched["artifacts"][0]["download_url"],
        },
    }
    rendered = json.dumps(projected, ensure_ascii=False, sort_keys=True)
    for platform_binding in (
        '"asset_fingerprint"',
        '"content_hash"',
        '"request_hash"',
        '"source"',
        '"population"',
        '"trial_accounting"',
    ):
        assert platform_binding not in rendered


def test_candidate_lab_fails_closed_when_cross_search_bytes_drift(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    searched = run_search_cross_matrix_candidates(
        _search_inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )
    repository = TaskArtifactRepository(fixture["settings"].db_path)
    record = repository.get_for_task(
        fixture["task"].id,
        searched["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    Path(record["path"]).write_text("{}", encoding="utf-8")

    response = TestClient(create_app(fixture["settings"])).get(
        f"/api/tasks/{fixture['task'].id}/strategy-candidate-lab"
    )

    assert response.status_code == 409
    assert searched["search_id"] not in response.text
