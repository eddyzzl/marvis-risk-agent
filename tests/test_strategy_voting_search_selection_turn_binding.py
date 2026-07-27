"""Plan-time binding for exact Voting search-result materialization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _strategy_request_requires_dataset,
    _strategy_voting_candidate_build_from_search_plan_slots,
)
from marvis.packs.strategy.errors import StrategyError


SEARCH_ID = "voting-search-" + "a" * 32
COMBO_ID = "voting-combo-" + "b" * 32


def _draft(*, strategy_type: str | None = None) -> StandardWorkflowRequestDraft:
    inputs = {"search_id": SEARCH_ID, "combo_id": COMBO_ID}
    if strategy_type is not None:
        inputs["strategy_type"] = strategy_type
    return StandardWorkflowRequestDraft(
        workflow="voting_candidate_build_from_search",
        workflow_inputs=inputs,
    )


def test_voting_search_selection_preflights_but_plan_keeps_only_user_pointers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    read_runtime = SimpleNamespace(kind="governed-read-runtime")
    task = SimpleNamespace(id="task-1")
    calls: list[dict] = []

    def resolve(
        actual_runtime,
        *,
        task_id,
        search_id,
        combo_id,
        strategy_type=None,
    ):
        calls.append(
            {
                "runtime": actual_runtime,
                "task_id": task_id,
                "search_id": search_id,
                "combo_id": combo_id,
                "strategy_type": strategy_type,
            }
        )
        return SimpleNamespace(
            member_rule_ids=("candidate-rule-" + "c" * 32,),
            selected_entry_ids=("pool-entry-" + "d" * 32,),
            n=1,
            rank=3,
            eligible=False,
            constraint_failures=(
                {
                    "metric": "hit_share",
                    "operator": "gte",
                    "threshold": 0.2,
                    "actual": 0.1,
                },
            ),
        )

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_voting_candidate_search_selection",
        resolve,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_read_runtime",
        lambda actual_runtime: read_runtime if actual_runtime is runtime else None,
    )

    slots = _strategy_voting_candidate_build_from_search_plan_slots(
        runtime,
        task,
        _draft(),
    )

    assert slots == {"search_id": SEARCH_ID, "combo_id": COMBO_ID}
    assert calls == [
        {
            "runtime": read_runtime,
            "task_id": "task-1",
            "search_id": SEARCH_ID,
            "combo_id": COMBO_ID,
            "strategy_type": None,
        }
    ]
    assert {
        "member_rule_ids",
        "selected_entry_ids",
        "n",
        "rank",
        "eligible",
        "constraint_failures",
        "pool_ref",
    }.isdisjoint(slots)
    assert _strategy_request_requires_dataset(_draft()) is False


def test_voting_search_selection_preserves_only_explicit_optional_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_voting_candidate_search_selection",
        lambda *_args, **_kwargs: SimpleNamespace(eligible=True),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_read_runtime",
        lambda _runtime: SimpleNamespace(kind="governed-read-runtime"),
    )

    slots = _strategy_voting_candidate_build_from_search_plan_slots(
        SimpleNamespace(),
        SimpleNamespace(id="task-1"),
        _draft(strategy_type="approval"),
    )

    assert slots == {
        "search_id": SEARCH_ID,
        "combo_id": COMBO_ID,
        "strategy_type": "approval",
    }


def test_voting_search_selection_turn_reports_resolver_failure_as_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args, **_kwargs):
        raise StrategyError("Voting search selection no longer matches current Pool")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_voting_candidate_search_selection",
        reject,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_read_runtime",
        lambda _runtime: SimpleNamespace(kind="governed-read-runtime"),
    )

    with pytest.raises(
        StrategySetupError,
        match="no longer matches current Pool",
    ):
        _strategy_voting_candidate_build_from_search_plan_slots(
            SimpleNamespace(),
            SimpleNamespace(id="task-1"),
            _draft(),
        )
