"""Plan-time binding and templates for Cross automatic search."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _strategy_cross_candidate_build_from_search_plan_slots,
    _strategy_cross_candidate_search_plan_slots,
    _strategy_request_requires_dataset,
)
from marvis.orchestrator.templates.strategy import (
    STRATEGY_CROSS_MATRIX_CANDIDATE_BUILD_FROM_SEARCH,
    STRATEGY_CROSS_MATRIX_CANDIDATE_SEARCH,
)
from marvis.packs.strategy.errors import StrategyError


SEARCH_ID = "cross-search-" + "a" * 32
PAIR_ID = "cross-pair-" + "b" * 32
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _search_draft() -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="cross_matrix_candidate_search",
        workflow_inputs={
            "features": ["age", "score", "income"],
            "max_pairs": 3,
        },
    )


def _build_draft() -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="cross_matrix_candidate_build_from_search",
        workflow_inputs={"search_id": SEARCH_ID, "pair_id": PAIR_ID},
    )


def test_cross_search_plan_binds_latest_source_but_not_axis_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    task = SimpleNamespace(id="task-1")
    resolved = {
        "source_artifact_id": HASH_A,
        "expected_artifact_content_hash": HASH_B,
        "expected_candidate_id": "candidate-" + "c" * 32,
        "expected_evidence_hash": HASH_C,
    }
    calls: list[tuple[object, str]] = []

    def bind(actual_runtime, *, task_id):
        calls.append((actual_runtime, task_id))
        return resolved

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._latest_cross_candidate_search_source_slots",
        bind,
    )

    slots = _strategy_cross_candidate_search_plan_slots(
        runtime,
        task,
        _search_draft(),
    )

    assert slots == {
        **resolved,
        "features": ["age", "score", "income"],
        "max_pairs": 3,
    }
    assert calls == [(runtime, "task-1")]
    assert {
        "x_method",
        "y_method",
        "axis_methods",
        "rank",
        "winner",
        "champion",
    }.isdisjoint(slots)
    assert _strategy_request_requires_dataset(_search_draft()) is False


def test_cross_search_build_preflights_but_plan_keeps_only_user_pointers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    read_runtime = SimpleNamespace(kind="governed-read-runtime")
    task = SimpleNamespace(id="task-1")
    calls: list[dict] = []

    def resolve(actual_runtime, *, task_id, search_id, pair_id):
        calls.append(
            {
                "runtime": actual_runtime,
                "task_id": task_id,
                "search_id": search_id,
                "pair_id": pair_id,
            }
        )
        return (SimpleNamespace(), {"rank": 7, "x_method": "tree"})

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_cross_candidate_search_pair",
        resolve,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_read_runtime",
        lambda actual_runtime: (
            read_runtime if actual_runtime is runtime else None
        ),
    )

    slots = _strategy_cross_candidate_build_from_search_plan_slots(
        runtime,
        task,
        _build_draft(),
    )

    assert slots == {"search_id": SEARCH_ID, "pair_id": PAIR_ID}
    assert calls == [
        {
            "runtime": read_runtime,
            "task_id": "task-1",
            "search_id": SEARCH_ID,
            "pair_id": PAIR_ID,
        }
    ]
    assert _strategy_request_requires_dataset(_build_draft()) is False


def test_cross_search_build_reports_preflight_failure_as_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_cross_candidate_search_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StrategyError("Cross search pair no longer authenticates")
        ),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_read_runtime",
        lambda _runtime: SimpleNamespace(),
    )

    with pytest.raises(StrategySetupError, match="no longer authenticates"):
        _strategy_cross_candidate_build_from_search_plan_slots(
            SimpleNamespace(),
            SimpleNamespace(id="task-1"),
            _build_draft(),
        )


def test_cross_search_templates_call_only_the_governed_tools() -> None:
    search = STRATEGY_CROSS_MATRIX_CANDIDATE_SEARCH
    build = STRATEGY_CROSS_MATRIX_CANDIDATE_BUILD_FROM_SEARCH

    assert search.id == "strategy_cross_matrix_candidate_search"
    assert search.steps[0].tool_ref.plugin == "strategy"
    assert search.steps[0].tool_ref.tool == "search_cross_matrix_candidates"
    assert search.steps[0].inputs_template == {
        "source_artifact_id": "{slot:source_artifact_id}",
        "expected_artifact_content_hash": (
            "{slot:expected_artifact_content_hash}"
        ),
        "expected_candidate_id": "{slot:expected_candidate_id}",
        "expected_evidence_hash": "{slot:expected_evidence_hash}",
        "features": "{slot:features}",
        "max_pairs": "{slot:max_pairs}",
    }
    assert build.id == "strategy_cross_matrix_candidate_build_from_search"
    assert build.steps[0].tool_ref.plugin == "strategy"
    assert (
        build.steps[0].tool_ref.tool
        == "build_cross_matrix_candidate_from_search"
    )
    assert build.steps[0].inputs_template == {
        "search_id": "{slot:search_id}",
        "pair_id": "{slot:pair_id}",
    }
