"""Authenticated Candidate Lab projection for Pool-admissible materialized sources."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.packs.strategy import (
    automatic_tree_leaf_tools,
    candidate_lab_projection,
    cross_matrix_cell_selection_tools,
)
from marvis.packs.strategy.voting_candidate_search_tools import (
    run_build_voting_candidate_from_search,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_automatic_tree_leaf_tool import (
    _fixture as _automatic_tree_fixture,
)
from tests.test_strategy_cross_matrix_cell_selection_tool import (
    _fixture as _cross_matrix_fixture,
)
from tests.test_strategy_interactive_tree_frontier_group_tool import (
    _materialized_group,
)
from tests.test_strategy_interactive_tree_frontier_tool import (
    _materialized_selection,
)
from tests.test_strategy_candidate_lab_api import (
    _install_fast_scorecard_live_revalidation,
    _register_refined_asset,
    _register_scorecard_candidate,
    _register_scorecard_selection,
    _searched_candidate_lab_fixture,
    _strategy_task,
)

pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


@pytest.fixture(autouse=True)
def _fast_scorecard_live_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fast_scorecard_live_revalidation(monkeypatch)


def test_candidate_lab_projects_one_verified_univariate_asset_for_pool_add(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _record, _path, fragment, _evidence = _register_refined_asset(
        app,
        task_id,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    sources = response.json()["pool_add_sources"]
    assert sources == {
        "latest": {
            "source_kind": "univariate_asset",
            "candidate_asset_id": fragment["asset"]["asset_id"],
            "strategy_type": None,
            "candidate_stage": "development",
            "validation_status": "unvalidated",
        },
        "all": [
            {
                "source_kind": "univariate_asset",
                "candidate_asset_id": fragment["asset"]["asset_id"],
                "strategy_type": None,
                "candidate_stage": "development",
                "validation_status": "unvalidated",
            }
        ],
        "total": 1,
        "truncated": False,
    }


def test_candidate_lab_never_fakes_unmaterialized_leaf_cell_or_voting_search(
    tmp_path: Path,
) -> None:
    automatic_root = tmp_path / "automatic"
    automatic_root.mkdir()
    automatic = _automatic_tree_fixture(automatic_root)
    automatic_response = TestClient(create_app(automatic.settings)).get(
        f"/api/tasks/{automatic.task.id}/strategy-candidate-lab"
    )
    assert automatic_response.status_code == 200, automatic_response.text
    assert not any(
        source["source_kind"] == "automatic_tree_leaf_selection"
        for source in automatic_response.json()["pool_add_sources"]["all"]
    )

    cross_root = tmp_path / "cross"
    cross_root.mkdir()
    cross = _cross_matrix_fixture(cross_root)
    cross_response = TestClient(create_app(cross.settings)).get(
        f"/api/tasks/{cross.task.id}/strategy-candidate-lab"
    )
    assert cross_response.status_code == 200, cross_response.text
    assert not any(
        source["source_kind"] == "cross_matrix_cell_selection"
        for source in cross_response.json()["pool_add_sources"]["all"]
    )

    voting_root = tmp_path / "voting"
    voting_root.mkdir()
    searched = _searched_candidate_lab_fixture(voting_root)
    voting_response = searched["client"].get(
        f"/api/tasks/{searched['task'].id}/strategy-candidate-lab"
    )
    assert voting_response.status_code == 200, voting_response.text
    assert not any(
        source["source_kind"] == "voting_candidate"
        for source in voting_response.json()["pool_add_sources"]["all"]
    )


def test_candidate_lab_projects_one_verified_automatic_tree_leaf_selection(
    tmp_path: Path,
) -> None:
    fixture = _automatic_tree_fixture(tmp_path)
    materialized = (
        automatic_tree_leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fixture.inputs,
            fixture.ctx,
            fixture.runtime,
        )
    )
    client = TestClient(create_app(fixture.settings))

    response = client.get(
        f"/api/tasks/{fixture.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    assert response.json()["pool_add_sources"] == {
        "latest": {
            "source_kind": "automatic_tree_leaf_selection",
            "selection_id": materialized["selection_id"],
            "strategy_type": None,
            "candidate_stage": "development",
            "validation_status": "unvalidated",
        },
        "all": [
            {
                "source_kind": "automatic_tree_leaf_selection",
                "selection_id": materialized["selection_id"],
                "strategy_type": None,
                "candidate_stage": "development",
                "validation_status": "unvalidated",
            }
        ],
        "total": 1,
        "truncated": False,
    }


def test_candidate_lab_projects_one_verified_cross_matrix_cell_selection(
    tmp_path: Path,
) -> None:
    fixture = _cross_matrix_fixture(tmp_path)
    materialized = (
        cross_matrix_cell_selection_tools
        .run_materialize_cross_matrix_cell_selection(
            fixture.inputs,
            fixture.ctx,
            fixture.runtime,
        )
    )
    client = TestClient(create_app(fixture.settings))

    response = client.get(
        f"/api/tasks/{fixture.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    [source] = response.json()["pool_add_sources"]["all"]
    assert source == {
        "source_kind": "cross_matrix_cell_selection",
        "selection_id": materialized["selection_id"],
        "strategy_type": None,
        "candidate_stage": "development",
        "validation_status": "unvalidated",
    }


def test_candidate_lab_projects_one_verified_interactive_tree_frontier_selection(
    scenario,
) -> None:
    materialized, _selection, _record = _materialized_selection(scenario)
    client = TestClient(create_app(scenario.settings))

    response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    [source] = response.json()["pool_add_sources"]["all"]
    assert source == {
        "source_kind": "interactive_tree_frontier_selection",
        "selection_id": materialized["selection_id"],
        "strategy_type": None,
        "candidate_stage": "development",
        "validation_status": "unvalidated",
    }


def test_candidate_lab_projects_one_verified_interactive_tree_frontier_group(
    scenario,
) -> None:
    materialized, _selection, _record, _revision = _materialized_group(
        scenario
    )
    client = TestClient(create_app(scenario.settings))

    response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    [source] = response.json()["pool_add_sources"]["all"]
    assert source == {
        "source_kind": "interactive_tree_frontier_group_selection",
        "selection_id": materialized["selection_id"],
        "strategy_type": None,
        "candidate_stage": "development",
        "validation_status": "unvalidated",
    }


def test_candidate_lab_projects_one_verified_scorecard_cutoff_selection(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        _band_record,
        _band_path,
        _asset,
        _selection_record,
        _selection_path,
        fragment,
        _sources,
    ) = _register_scorecard_candidate(app, task_id)

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    [source] = response.json()["pool_add_sources"]["all"]
    assert source == {
        "source_kind": "scorecard_cutoff_selection",
        "selection_id": response.json()["candidates"][
            "scorecard_cutoff_selection"
        ]["latest"]["detail"]["selection_id"],
        "strategy_type": None,
        "candidate_stage": fragment["candidate_stage"],
        "validation_status": fragment["validation_status"],
    }


def test_candidate_lab_projects_one_verified_built_voting_candidate(
    tmp_path: Path,
) -> None:
    fixture = _searched_candidate_lab_fixture(tmp_path)
    search = fixture["search"]
    combo = search["search_result"]["combinations"][0]
    built = run_build_voting_candidate_from_search(
        {
            "search_id": search["search_id"],
            "combo_id": combo["combo_id"],
            "strategy_type": "approval",
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    response = fixture["client"].get(
        f"/api/tasks/{fixture['task'].id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    voting_sources = [
        source
        for source in response.json()["pool_add_sources"]["all"]
        if source["source_kind"] == "voting_candidate"
    ]
    [source] = voting_sources
    assert source == {
        "source_kind": "voting_candidate",
        "candidate_asset_id": built["voting_candidate"]["asset_id"],
        "strategy_type": "approval",
        "candidate_stage": "development",
        "validation_status": "unvalidated",
    }


def test_candidate_lab_pool_add_sources_fail_closed_when_bytes_drift(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _record, path, _fragment, _evidence = _register_refined_asset(
        app,
        task_id,
    )
    path.write_text('{"forged":true}', encoding="utf-8")

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_pool_add_sources_are_task_scoped(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    foreign_task_id = _strategy_task(app)
    _register_refined_asset(app, foreign_task_id)

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    assert response.json()["pool_add_sources"] == {
        "latest": None,
        "all": [],
        "total": 0,
        "truncated": False,
    }


def test_candidate_lab_pool_add_sources_reject_duplicate_projected_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    record, _path, _fragment, _evidence = _register_refined_asset(
        app,
        task_id,
    )
    original = TaskArtifactRepository.list_recent_for_task_kind_with_count

    def duplicate_identity(self, query_task_id, kind, *, limit):
        if (
            query_task_id == task_id
            and kind == "strategy_candidate_asset_json"
        ):
            return [record, dict(record)], 2
        return original(self, query_task_id, kind, limit=limit)

    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_recent_for_task_kind_with_count",
        duplicate_identity,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_pool_add_source_window_reports_true_total_and_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        band_record,
        _band_path,
        asset,
        _first_record,
        _first_path,
        _first_fragment,
        _sources,
    ) = _register_scorecard_candidate(app, task_id)
    _register_scorecard_selection(
        app,
        task_id,
        asset=asset,
        band_record=band_record,
        cutoff_ordinal=1,
        selection_reason="第二个独立 Cutoff",
    )
    monkeypatch.setattr(
        candidate_lab_projection,
        "_MAX_POOL_ADD_SOURCES_PER_KIND",
        1,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    sources = response.json()["pool_add_sources"]
    assert sources["total"] == 2
    assert len(sources["all"]) == 1
    assert sources["latest"] == sources["all"][0]
    assert sources["truncated"] is True
