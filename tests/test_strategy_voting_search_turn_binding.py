"""Plan-time server binding for Voting combination search."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _strategy_request_requires_dataset,
    _strategy_voting_candidate_search_plan_slots,
)
from marvis.packs.strategy.errors import StrategyError


RULE_A = "candidate-rule-" + "a" * 32
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _draft() -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="voting_candidate_search",
        workflow_inputs={
            "strategy_type": "approval",
            "member_count": 3,
            "n": 2,
            "objective": {
                "metric": "bad_rate",
                "direction": "minimize",
            },
            "constraints": [
                {
                    "metric": "hit_share",
                    "operator": "gte",
                    "value": 0.1,
                }
            ],
            "include_rule_ids": [RULE_A],
            "exclude_rule_ids": [],
            "max_combinations": 500,
        },
    )


def _resolved() -> dict[str, object]:
    return {
        **_draft().to_dict()["workflow_inputs"],
        "pool_ref": {
            "artifact_id": HASH_A,
            "expected_artifact_content_hash": HASH_B,
            "expected_pool_id": "strategy-pool-task-1-approval",
            "expected_revision": 7,
            "expected_revision_id": "strategy-pool-revision-7",
            "expected_snapshot_hash": HASH_C,
        },
    }


def test_voting_search_turn_uses_governed_resolver_without_dataset_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    read_runtime = SimpleNamespace(kind="governed-read-runtime")
    task = SimpleNamespace(id="task-1")
    calls: list[dict] = []

    def resolve(actual_runtime, *, task_id, user_controls):
        calls.append(
            {
                "runtime": actual_runtime,
                "task_id": task_id,
                "user_controls": user_controls,
            }
        )
        return _resolved()

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_voting_candidate_search_inputs",
        resolve,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_read_runtime",
        lambda actual_runtime: read_runtime if actual_runtime is runtime else None,
    )

    slots = _strategy_voting_candidate_search_plan_slots(runtime, task, _draft())

    assert slots == _resolved()
    assert calls == [
        {
            "runtime": read_runtime,
            "task_id": "task-1",
            "user_controls": _draft().to_dict()["workflow_inputs"],
        }
    ]
    assert "dataset_id" not in slots
    assert "target_col" not in slots
    assert _strategy_request_requires_dataset(_draft()) is False


def test_voting_search_turn_reports_resolver_failure_as_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args, **_kwargs):
        raise StrategyError("Voting search current Pool changed")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_voting_candidate_search_inputs",
        reject,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_read_runtime",
        lambda _runtime: SimpleNamespace(kind="governed-read-runtime"),
    )

    with pytest.raises(
        StrategySetupError,
        match="Voting search current Pool changed",
    ):
        _strategy_voting_candidate_search_plan_slots(
            SimpleNamespace(),
            SimpleNamespace(id="task-1"),
            _draft(),
        )
