"""Turn-boundary binding for independent Strategy Pool replay validation."""

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
    _strategy_pool_validation_plan_slots,
    _strategy_request_requires_dataset,
    _strategy_request_requires_target,
)
from marvis.api_schemas import ManualStrategyRequest


def _draft(
    strategy_type: str = "approval",
    partition: str = "validation",
) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_pool_validation",
        workflow_inputs={
            "strategy_type": strategy_type,
            "partition": partition,
        },
    )


def _runtime(tmp_path: Path):
    return SimpleNamespace(settings=SimpleNamespace(db_path=tmp_path / "marvis.sqlite"))


def _pool_binding() -> SimpleNamespace:
    return SimpleNamespace(
        task_id="task-1",
        strategy_type="approval",
        artifact_id="1" * 64,
        artifact_content_hash="2" * 64,
        pool={
            "pool_id": "strategy-pool-" + "3" * 32,
            "revision": 7,
            "revision_id": "strategy-pool-revision-" + "4" * 32,
            "snapshot_hash": "5" * 64,
            "entries": [{"entry_id": "pool-entry-" + "6" * 32}],
        },
        compiled_design={"requirements": [{"kind": "model_score"}]},
    )


def _sample_binding(
    *,
    maturity: str = "confirmed_matured",
    validation_count: int = 4,
    oot_count: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        membership_artifact_id="7" * 64,
        membership_artifact_content_hash="8" * 64,
        bundle_artifact_id="9" * 64,
        bundle_artifact_content_hash="a" * 64,
        bundle={
            "bundle_id": "sample-bundle-1",
            "sample_design": {
                "sample_design_id": "sample-design-1",
                "content_hash": "b" * 64,
                "target_selector": {"status": "resolved", "column": "bad"},
                "sample_semantics": {"scope": "strategy_development"},
            },
            "populations": [
                {
                    "role": "approval",
                    "maturity_evidence": {"status": "not_applicable"},
                },
                {
                    "role": "risk",
                    "maturity_evidence": {"status": maturity},
                },
            ],
        },
        membership={
            "header": {
                "counts": {
                    "risk": {
                        "development": 10,
                        "validation": validation_count,
                        "oot": oot_count,
                    }
                }
            }
        },
        source_binding=SimpleNamespace(
            dataset_id="dataset-1",
            dataset_content_hash="c" * 64,
            workspace_revision=3,
            workspace_generation=2,
            semantic_mapping_hash="d" * 64,
        ),
    )


def test_turn_binds_exact_pool_sample_and_fixed_replay_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool_binding()
    sample = _sample_binding()
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_v2_read_runtime",
        lambda runtime: "read-runtime",
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_current_pool_binding",
        lambda read_runtime, *, task_id, requested_type: (
            calls.update(
                {
                    "pool": (read_runtime, task_id, requested_type),
                }
            )
            or pool
        ),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_latest_sample_binding",
        lambda read_runtime, *, task_id: (
            calls.update({"sample": (read_runtime, task_id)}) or sample
        ),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_pool_requirements",
        lambda runtime, *, task_id, compiled_design, sample_design: (
            calls.update(
                {
                    "requirements": (
                        runtime,
                        task_id,
                        compiled_design,
                        sample_design,
                    )
                }
            )
            or SimpleNamespace()
        ),
    )

    slots = _strategy_pool_validation_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        _draft(),
    )

    assert calls["pool"] == ("read-runtime", "task-1", "approval")
    assert calls["sample"] == ("read-runtime", "task-1")
    assert calls["requirements"] == (
        "read-runtime",
        "task-1",
        pool.compiled_design,
        sample,
    )
    assert slots == {
        "strategy_type": "approval",
        "partition": "validation",
        "pool_ref": {
            "artifact_id": "1" * 64,
            "expected_artifact_content_hash": "2" * 64,
            "expected_pool_id": "strategy-pool-" + "3" * 32,
            "expected_revision": 7,
            "expected_revision_id": "strategy-pool-revision-" + "4" * 32,
            "expected_snapshot_hash": "5" * 64,
        },
        "sample_design_ref": {
            "membership_artifact_id": "7" * 64,
            "expected_membership_artifact_content_hash": "8" * 64,
            "bundle_artifact_id": "9" * 64,
            "expected_bundle_artifact_content_hash": "a" * 64,
            "expected_bundle_id": "sample-bundle-1",
            "expected_sample_design_id": "sample-design-1",
            "expected_sample_design_content_hash": "b" * 64,
        },
        "population": "risk",
        "comparison_mode": "absolute",
    }


@pytest.mark.parametrize(
    ("sample", "partition", "message"),
    [
        (_sample_binding(maturity="not_matured"), "validation", "成熟"),
        (_sample_binding(validation_count=0), "validation", "validation"),
        (_sample_binding(oot_count=0), "oot", "oot"),
    ],
)
def test_turn_rejects_immature_or_empty_independent_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample: SimpleNamespace,
    partition: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_v2_read_runtime",
        lambda runtime: "read-runtime",
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_current_pool_binding",
        lambda *args, **kwargs: _pool_binding(),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_latest_sample_binding",
        lambda *args, **kwargs: sample,
    )

    with pytest.raises(StrategySetupError, match=message):
        _strategy_pool_validation_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            _draft(partition=partition),
        )


def test_turn_routes_without_dataset_or_target_and_starts_typed_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = _draft("reject", "oot")
    captured: dict = {}
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_pool_validation_plan_slots",
        lambda runtime, task, candidate: {
            "strategy_type": "reject",
            "partition": "oot",
            "pool_ref": {"exact": "pool"},
            "sample_design_ref": {"exact": "sample"},
            "population": "risk",
            "comparison_mode": "absolute",
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
    assert captured["template_id"] == "strategy_pool_validation"
    assert captured["slots"]["strategy_type"] == "reject"
    assert captured["slots"]["partition"] == "oot"
    assert captured["auto_start"] is True


def test_preflight_does_not_open_a_second_evidence_selection_window(
    tmp_path: Path,
) -> None:
    assert (
        _standard_workflow_request_preflight(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            _draft(),
        )
        is None
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "对当前审批策略池执行 validation 独立样本回放验证",
        "run independent replay validation for the current reject pool on OOT",
    ],
)
def test_turn_router_recognizes_independent_replay_commands(
    utterance: str,
) -> None:
    assert _is_strategy_request_intent(utterance) is True


def test_manual_request_accepts_exactly_type_and_partition() -> None:
    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_validation",
            "workflow_inputs": {
                "strategy_type": "reject",
                "partition": "oot",
            },
        },
        strict=True,
    )

    assert request.workflow_inputs == {
        "strategy_type": "reject",
        "partition": "oot",
    }
    for forged in (
        {"strategy_type": "limit", "partition": "validation"},
        {"strategy_type": "approval", "partition": "development"},
        {
            "strategy_type": "approval",
            "partition": "validation",
            "pool_ref": {"forged": True},
        },
    ):
        with pytest.raises(ValidationError):
            ManualStrategyRequest.model_validate(
                {
                    "request_kind": "standard_workflow",
                    "workflow": "strategy_pool_validation",
                    "workflow_inputs": forged,
                },
                strict=True,
            )
