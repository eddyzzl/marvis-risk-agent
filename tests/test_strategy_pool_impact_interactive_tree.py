from __future__ import annotations

from pathlib import Path

import pytest

from marvis.data.workspace import data_semantic_mapping_hash
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
import marvis.packs.strategy.pool_impact_tools as impact_tools
from marvis.packs.strategy.pool_impact_tools import (
    POOL_IMPACT_ARTIFACT_KIND,
    load_strategy_pool_impact_artifact,
    run_measure_pool_impact,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_pool_interactive_tree_frontier import (
    _add_inputs,
    _materialize_frontier,
)


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


def _impact_request(scenario, added: dict) -> dict:
    return {
        "strategy_type": "approval",
        "expected_pool_revision": added["revision"],
        "expected_pool_snapshot_hash": added["snapshot_hash"],
        "dataset_id": scenario.dataset.id,
        "expected_dataset_content_hash": scenario.dataset.content_hash,
        "workspace_revision": scenario.workspace.revision,
        "workspace_generation": scenario.workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(scenario.mapping),
        "target_col": "bad",
        "sample_design_ref": scenario.sample_design_ref,
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "comparison_mode": "absolute",
        "drop_nan_labels": False,
    }


def _materialize_add_and_measure(scenario) -> tuple[dict, dict, dict]:
    selection, revision = _materialize_frontier(scenario)
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(selection),
        scenario.ctx,
    )
    runtime = strategy_tools._runtime(scenario.ctx)
    impact = run_measure_pool_impact(
        _impact_request(scenario, added),
        scenario.ctx,
        runtime,
    )
    return selection, added, impact


def test_interactive_tree_frontier_pool_measures_and_reloads_impact(
    scenario,
) -> None:
    selection, added, impact = _materialize_add_and_measure(scenario)
    runtime = strategy_tools._runtime(scenario.ctx)

    assert impact["pool_id"] == added["pool_id"]
    assert impact["revision"] == added["revision"]
    assert impact["population_count"] == len(scenario.development_frame)
    assert impact["monthly_status"] == "unavailable"
    assert impact["assessment"]["waterfall"][0]["source_ref"]["artifact_id"] == (
        selection["artifacts"][0]["artifact_id"]
    )

    artifact = impact["artifacts"][0]
    binding = load_strategy_pool_impact_artifact(
        runtime,
        task_id=scenario.task.id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
        expected_assessment_id=impact["assessment_id"],
        expected_assessment_content_hash=impact["content_hash"],
    )
    assert binding.assessment == impact["assessment"]


def test_interactive_tree_frontier_drift_before_impact_fails_closed(
    scenario,
) -> None:
    selection, _revision = _materialize_frontier(scenario)
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(selection),
        scenario.ctx,
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        selection["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    path = Path(record["path"])
    path.write_bytes(path.read_bytes() + b"\n")

    runtime = strategy_tools._runtime(scenario.ctx)
    with pytest.raises(StrategyError, match="hash|changed|canonical"):
        run_measure_pool_impact(
            _impact_request(scenario, added),
            scenario.ctx,
            runtime,
        )

    assert not [
        artifact
        for artifact in TaskArtifactRepository(
            scenario.settings.db_path
        ).list_for_task(scenario.task.id)
        if artifact["kind"] == POOL_IMPACT_ARTIFACT_KIND
    ]


def test_interactive_tree_frontier_drift_before_registration_fails_closed(
    scenario,
    monkeypatch,
) -> None:
    selection, _revision = _materialize_frontier(scenario)
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(selection),
        scenario.ctx,
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        selection["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    path = Path(record["path"])
    original_persist = impact_tools._persist_assessment

    def drift_then_persist(*args, **kwargs):
        path.write_bytes(path.read_bytes() + b"\n")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(impact_tools, "_persist_assessment", drift_then_persist)
    runtime = strategy_tools._runtime(scenario.ctx)

    with pytest.raises(StrategyError, match="hash|changed|canonical"):
        run_measure_pool_impact(
            _impact_request(scenario, added),
            scenario.ctx,
            runtime,
        )

    assert not [
        artifact
        for artifact in TaskArtifactRepository(
            scenario.settings.db_path
        ).list_for_task(scenario.task.id)
        if artifact["kind"] == POOL_IMPACT_ARTIFACT_KIND
    ]
    assert not list(
        (
            Path(scenario.settings.tasks_dir)
            / scenario.task.id
            / "strategy_pool_impacts"
        ).glob("*.json")
    )
