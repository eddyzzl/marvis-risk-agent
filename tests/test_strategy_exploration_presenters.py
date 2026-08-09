from __future__ import annotations

from copy import deepcopy

import pytest

from marvis.agent.presenters import modeling_evidence as modeling_presenters
from marvis.agent.presenters import strategy_exploration as presenters
from marvis.canonical_results import (
    CanonicalResultAuthenticationError,
    authenticate_canonical_result,
)
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.cross_candidate_search_tools import (
    run_build_cross_matrix_candidate_from_search,
    run_search_cross_matrix_candidates,
)
from marvis.packs.strategy.impact_cube_tools import (
    run_measure_strategy_impact_cube,
)
from tests.test_strategy_cross_candidate_search_tools import _search_inputs
from tests.test_strategy_cross_matrix_candidate_tool import _setup as cross_setup
from tests.test_strategy_impact_cube_tools import (
    _artifacts as impact_artifacts,
    _setup as impact_setup,
)
from tests.test_strategy_interactive_tree_continuation_tool import (
    _inputs as auto_continue_inputs,
    _search_frontier,
)
from tests.test_strategy_interactive_tree_tool import _Scenario


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)

EXPECTED_TOOLS = {
    "search_cross_matrix_candidates",
    "build_cross_matrix_candidate_from_search",
    "search_interactive_tree_split_candidates",
    "auto_continue_interactive_tree",
    "revise_interactive_tree",
    "measure_strategy_impact_cube",
}


def _drifted_inputs(tool: str, value: dict) -> dict:
    changed = deepcopy(value)
    if tool == "search_cross_matrix_candidates":
        changed["max_pairs"] = 9
    elif tool == "build_cross_matrix_candidate_from_search":
        changed["pair_id"] = "cross-pair-" + "f" * 32
    elif tool == "search_interactive_tree_split_candidates":
        changed["max_thresholds_per_feature"] = 3
    elif tool == "auto_continue_interactive_tree":
        changed["min_gini_gain"] = 0.01
    elif tool == "revise_interactive_tree":
        changed["reason"] = "A different reviewed edit."
    else:
        changed["partitions"] = ["development"]
    return changed


def _outputs(tmp_path, scenario: _Scenario):
    cross = cross_setup(tmp_path / "cross", with_split=True)
    cross_search_inputs = _search_inputs(cross)
    cross_search = run_search_cross_matrix_candidates(
        cross_search_inputs,
        cross["ctx"],
        cross["runtime"],
    )
    [pair] = cross_search["search_result"]["pairs"]
    cross_selection_inputs = {
        "search_id": cross_search["search_id"],
        "pair_id": pair["pair_id"],
    }
    cross_selection = run_build_cross_matrix_candidate_from_search(
        cross_selection_inputs,
        cross["ctx"],
        cross["runtime"],
    )

    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    revision_inputs = {
        "source_tree_id": scenario.source_asset["asset_id"],
        "node_id": root_id,
        "operation": "adjust_split_threshold",
        "threshold": 1.5,
        "reason": "Exercise the authenticated presenter boundary.",
    }
    revision = strategy_tools.tool_revise_interactive_tree(
        revision_inputs,
        scenario.ctx,
    )
    split_search = _search_frontier(scenario)
    split_search_inputs = {
        "source_tree_id": split_search["source_tree_id"],
        "node_id": split_search["node_id"],
        "mode": "all_features",
        "max_thresholds_per_feature": 4,
        "max_row_evaluations": 10_000,
    }
    continuation_inputs = auto_continue_inputs(split_search)
    auto_continue = strategy_tools.tool_auto_continue_interactive_tree(
        continuation_inputs,
        scenario.ctx,
    )

    impact = impact_setup(tmp_path / "impact")
    impact_output = run_measure_strategy_impact_cube(
        impact["impact_request"],
        impact["ctx"],
        impact["runtime"],
    )
    [impact_record] = impact_artifacts(impact)

    return {
        "search_cross_matrix_candidates": (
            cross_search,
            cross["runtime"],
            cross["task"].id,
            cross_search_inputs,
            None,
        ),
        "build_cross_matrix_candidate_from_search": (
            cross_selection,
            cross["runtime"],
            cross["task"].id,
            cross_selection_inputs,
            None,
        ),
        "search_interactive_tree_split_candidates": (
            split_search,
            strategy_tools._runtime(scenario.ctx),
            scenario.task.id,
            split_search_inputs,
            None,
        ),
        "auto_continue_interactive_tree": (
            auto_continue,
            strategy_tools._runtime(scenario.ctx),
            scenario.task.id,
            continuation_inputs,
            None,
        ),
        "revise_interactive_tree": (
            revision,
            strategy_tools._runtime(scenario.ctx),
            scenario.task.id,
            revision_inputs,
            None,
        ),
        "measure_strategy_impact_cube": (
            impact_output,
            impact["runtime"],
            impact["task"].id,
            impact["impact_request"],
            {"impact_cube": {"record": impact_record}},
        ),
    }


def test_strategy_exploration_registry_authenticates_real_outputs_and_fails_closed(
    tmp_path,
    scenario: _Scenario,
) -> None:
    cases = _outputs(tmp_path, scenario)

    assert set(presenters.STRATEGY_EXPLORATION_PRESENTERS) == EXPECTED_TOOLS
    for tool, (output, runtime, task_id, trusted_inputs, trusted_artifacts) in cases.items():
        assert authenticate_canonical_result(
            tool,
            output,
            trusted_inputs=trusted_inputs,
            task_id=task_id,
            workspace=runtime.settings.workspace,
        ) is True
        with pytest.raises(CanonicalResultAuthenticationError):
            authenticate_canonical_result(
                tool,
                output,
                trusted_inputs=_drifted_inputs(tool, trusted_inputs),
                task_id=task_id,
                workspace=runtime.settings.workspace,
            )
        text, tables = presenters.STRATEGY_EXPLORATION_PRESENTERS[tool](
            output,
            runtime=runtime,
            task_id=task_id,
            trusted_inputs=trusted_inputs,
            trusted_artifacts=trusted_artifacts,
        )
        assert all(status in text for status in ("未入池", "未采纳", "未部署"))
        assert tables

        with pytest.raises(modeling_presenters.CanonicalPresenterIntegrityError):
            presenters.STRATEGY_EXPLORATION_PRESENTERS[tool](
                output,
                runtime=runtime,
                task_id=task_id,
                trusted_inputs=_drifted_inputs(tool, trusted_inputs),
                trusted_artifacts=trusted_artifacts,
            )

        forged = deepcopy(output)
        forged["unexpected_presenter_field"] = "must-not-render"
        with pytest.raises(modeling_presenters.CanonicalPresenterIntegrityError):
            presenters.STRATEGY_EXPLORATION_PRESENTERS[tool](
                forged,
                runtime=runtime,
                task_id=task_id,
                trusted_inputs=trusted_inputs,
                trusted_artifacts=trusted_artifacts,
            )

        partial = deepcopy(output)
        partial.pop("schema_version")
        with pytest.raises(modeling_presenters.CanonicalPresenterIntegrityError):
            presenters.STRATEGY_EXPLORATION_PRESENTERS[tool](
                partial,
                runtime=runtime,
                task_id=task_id,
                trusted_inputs=trusted_inputs,
                trusted_artifacts=trusted_artifacts,
            )


def test_strategy_exploration_validator_internal_errors_are_uniformly_fail_closed(
    tmp_path,
    scenario: _Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _outputs(tmp_path, scenario)
    validator_names = {
        "search_cross_matrix_candidates": "load_cross_candidate_search_artifact",
        "build_cross_matrix_candidate_from_search": (
            "resolve_cross_candidate_search_pair"
        ),
        "search_interactive_tree_split_candidates": (
            "load_verified_interactive_tree_split_search"
        ),
        "auto_continue_interactive_tree": (
            "load_verified_interactive_tree_revision"
        ),
        "revise_interactive_tree": "load_verified_interactive_tree_revision",
        "measure_strategy_impact_cube": (
            "validate_measure_strategy_impact_cube_tool_output"
        ),
    }

    for tool, validator_name in validator_names.items():
        output, runtime, task_id, trusted_inputs, trusted_artifacts = cases[tool]
        with monkeypatch.context() as patch:
            patch.setattr(
                presenters,
                validator_name,
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("injected validator failure")
                ),
            )
            with pytest.raises(
                modeling_presenters.CanonicalPresenterIntegrityError,
                match="canonical Tool output",
            ):
                presenters.STRATEGY_EXPLORATION_PRESENTERS[tool](
                    output,
                    runtime=runtime,
                    task_id=task_id,
                    trusted_inputs=trusted_inputs,
                    trusted_artifacts=trusted_artifacts,
                )


def test_strategy_exploration_presenters_require_live_trusted_context() -> None:
    with pytest.raises(modeling_presenters.CanonicalPresenterIntegrityError):
        presenters.STRATEGY_EXPLORATION_PRESENTERS[
            "search_cross_matrix_candidates"
        ]({}, runtime=None, task_id="task-1", trusted_inputs={})
    with pytest.raises(modeling_presenters.CanonicalPresenterIntegrityError):
        presenters.STRATEGY_EXPLORATION_PRESENTERS[
            "measure_strategy_impact_cube"
        ](
            {},
            runtime=object(),
            task_id="task-1",
            trusted_inputs={},
            trusted_artifacts=None,
        )
