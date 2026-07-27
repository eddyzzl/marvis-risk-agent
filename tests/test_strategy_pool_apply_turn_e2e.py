"""Turn-boundary binding for current Strategy Pool application."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _is_strategy_request_intent,
    _run_validated_strategy_request,
    _standard_workflow_request_preflight,
    _strategy_pool_apply_plan_slots,
    _strategy_request_requires_dataset,
    _strategy_request_requires_target,
)
from marvis.api_schemas import ManualStrategyRequest


POOL_HASH = "a" * 64


class _PoolRepository:
    def __init__(self, current) -> None:
        self.current = current
        self.calls: list[tuple[str, str]] = []

    def get_current(self, task_id: str, strategy_type: str):
        self.calls.append((task_id, strategy_type))
        return self.current


def _runtime(tmp_path: Path):
    return SimpleNamespace(settings=SimpleNamespace(db_path=tmp_path / "marvis.sqlite"))


def _draft(
    strategy_type: str = "approval",
    *,
    output_prefix: str | None = None,
) -> StandardWorkflowRequestDraft:
    inputs = {"strategy_type": strategy_type}
    if output_prefix is not None:
        inputs["output_prefix"] = output_prefix
    return StandardWorkflowRequestDraft(
        workflow="strategy_pool_apply",
        workflow_inputs=inputs,
    )


def test_turn_binds_only_current_nonempty_pool_cas_and_user_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pool = {
        "pool_id": "strategy-pool-1",
        "revision": 7,
        "entries": [{"rule_id": "candidate-rule-1"}],
    }
    repository = _PoolRepository(pool)
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: repository,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda current: POOL_HASH,
    )

    slots = _strategy_pool_apply_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        _draft(output_prefix="decision_"),
    )

    assert repository.calls == [("task-1", "approval")]
    assert slots == {
        "strategy_type": "approval",
        "output_prefix": "decision_",
        "expected_pool_revision": 7,
        "expected_pool_snapshot_hash": POOL_HASH,
    }
    assert not {
        "pool_id",
        "artifact_id",
        "dataset_id",
        "sample_design_ref",
        "requirements",
        "strategy_spec",
    } & set(slots)


@pytest.mark.parametrize(
    "utterance",
    [
        "把当前审批策略池应用到当前样本",
        "把当前拒绝策略池写回当前样本",
        "apply current limit pool to the current sample",
    ],
)
def test_turn_router_recognizes_every_pool_apply_verb(
    utterance: str,
) -> None:
    assert _is_strategy_request_intent(utterance) is True


@pytest.mark.parametrize(
    ("current", "message"),
    [
        (None, "没有 approval Strategy Pool"),
        ({"revision": 1, "entries": []}, "为空"),
        ({"revision": "bad", "entries": [{"rule_id": "r"}]}, "revision/hash"),
    ],
)
def test_turn_fails_closed_for_missing_empty_or_invalid_pool(
    tmp_path: Path,
    monkeypatch,
    current,
    message: str,
) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: _PoolRepository(current),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda pool: POOL_HASH,
    )

    with pytest.raises(StrategySetupError, match=message):
        _strategy_pool_apply_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            _draft(),
        )


def test_turn_routes_without_dataset_or_target_and_binds_pool_once_at_plan_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    draft = _draft("pricing")
    captured: dict = {}
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_pool_apply_plan_slots",
        lambda runtime, task, candidate: {
            "strategy_type": "pricing",
            "expected_pool_revision": 3,
            "expected_pool_snapshot_hash": POOL_HASH,
        },
    )

    def _start(runtime, repo, task, **kwargs):
        captured.update(kwargs)
        return {"status": "started"}

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._start_confirmed_strategy_plan",
        _start,
    )

    assert _strategy_request_requires_dataset(draft) is False
    assert _strategy_request_requires_target(draft) is False
    response = _run_validated_strategy_request(
        _runtime(tmp_path),
        SimpleNamespace(),
        SimpleNamespace(id="task-1"),
        draft,
        context=None,
        auto_start=True,
        drop_nan_labels=False,
    )

    assert response == {"status": "started"}
    assert captured == {
        "template_id": "strategy_pool_apply",
        "slots": {
            "strategy_type": "pricing",
            "expected_pool_revision": 3,
            "expected_pool_snapshot_hash": POOL_HASH,
        },
        "auto_start": True,
    }


def test_preflight_does_not_open_a_second_pool_selection_window(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)

    assert (
        _standard_workflow_request_preflight(
            runtime,
            SimpleNamespace(id="task-1"),
            _draft(),
        )
        is None
    )


def test_manual_request_accepts_only_type_and_optional_prefix() -> None:
    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_apply",
            "workflow_inputs": {
                "strategy_type": "segmentation",
                "output_prefix": "segment_",
            },
        }
    )

    assert request.workflow_inputs == {
        "strategy_type": "segmentation",
        "output_prefix": "segment_",
    }
    with pytest.raises(ValidationError, match="expected_pool_revision"):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_apply",
                "workflow_inputs": {
                    "strategy_type": "approval",
                    "expected_pool_revision": 7,
                },
            }
        )
