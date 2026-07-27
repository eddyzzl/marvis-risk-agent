"""Turn binding for exact current-Pool draft materialization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _is_strategy_request_intent,
    _run_validated_strategy_request,
    _standard_workflow_request_preflight,
    _strategy_pool_materialize_plan_slots,
    _strategy_request_requires_dataset,
    _strategy_request_requires_target,
)
from marvis.packs.strategy.errors import StrategyError


def _runtime(tmp_path: Path):
    settings = SimpleNamespace(
        db_path=tmp_path / "marvis.sqlite",
        tasks_dir=tmp_path / "tasks",
        datasets_dir=tmp_path / "datasets",
    )
    return SimpleNamespace(settings=settings)


def _draft(strategy_type: str = "approval") -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_pool_materialize",
        workflow_inputs={"strategy_type": strategy_type},
    )


def _binding() -> SimpleNamespace:
    return SimpleNamespace(
        task_id="task-1",
        strategy_type="approval",
        pool={
            "pool_id": "strategy-pool-1",
            "revision_id": "strategy-pool-revision-" + "1" * 32,
            "revision": 7,
            "snapshot_hash": "a" * 64,
            "entries": [{"rule_id": "candidate-rule-" + "2" * 32}],
        },
        artifact_id="b" * 64,
        artifact_content_hash="c" * 64,
        compiled_design={"design_hash": "d" * 64},
    )


def test_turn_deeply_authenticates_and_freezes_all_six_tool_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict] = []

    def _load(runtime, **kwargs):
        calls.append(kwargs)
        return _binding()

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_current_strategy_candidate_pool_artifact",
        _load,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_v2_read_runtime",
        lambda runtime: SimpleNamespace(kind="real-local-read-runtime"),
    )

    slots = _strategy_pool_materialize_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        _draft(),
    )

    assert calls == [{"task_id": "task-1", "strategy_type": "approval"}]
    assert slots == {
        "strategy_type": "approval",
        "expected_pool_revision": 7,
        "expected_pool_snapshot_hash": "a" * 64,
        "expected_pool_artifact_id": "b" * 64,
        "expected_pool_artifact_content_hash": "c" * 64,
        "expected_design_hash": "d" * 64,
    }


@pytest.mark.parametrize(
    "binding",
    [
        SimpleNamespace(
            **{
                **_binding().__dict__,
                "pool": {**_binding().pool, "entries": []},
            }
        ),
        SimpleNamespace(
            **{
                **_binding().__dict__,
                "compiled_design": {"design_hash": ""},
            }
        ),
    ],
)
def test_turn_rejects_empty_or_incomplete_authenticated_binding(
    tmp_path: Path,
    monkeypatch,
    binding,
) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_current_strategy_candidate_pool_artifact",
        lambda runtime, **kwargs: binding,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_v2_read_runtime",
        lambda runtime: SimpleNamespace(),
    )

    with pytest.raises(StrategySetupError, match="完整认证|非空"):
        _strategy_pool_materialize_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            _draft(),
        )


def test_turn_converts_deep_binding_failure_to_setup_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fail(runtime, **kwargs):
        raise StrategyError("artifact lineage changed")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_current_strategy_candidate_pool_artifact",
        _fail,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_v2_read_runtime",
        lambda runtime: SimpleNamespace(),
    )

    with pytest.raises(StrategySetupError, match="完整认证"):
        _strategy_pool_materialize_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            _draft(),
        )


def test_turn_routes_without_dataset_target_or_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    draft = _draft("pricing")
    captured: dict = {}
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_pool_materialize_plan_slots",
        lambda runtime, task, candidate: {
            "strategy_type": "pricing",
            "expected_pool_revision": 3,
            "expected_pool_snapshot_hash": "a" * 64,
            "expected_pool_artifact_id": "b" * 64,
            "expected_pool_artifact_content_hash": "c" * 64,
            "expected_design_hash": "d" * 64,
        },
    )

    def _start(runtime, repo, task, **kwargs):
        captured.update(kwargs)
        return {"status": "started"}

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._start_confirmed_strategy_plan",
        _start,
    )

    assert _is_strategy_request_intent(
        "把当前定价策略池物化为 draft Strategy"
    ) is True
    assert _strategy_request_requires_dataset(draft) is False
    assert _strategy_request_requires_target(draft) is False
    assert (
        _standard_workflow_request_preflight(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            draft,
        )
        is None
    )
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
    assert captured["template_id"] == "strategy_pool_materialize"
    assert captured["auto_start"] is True
