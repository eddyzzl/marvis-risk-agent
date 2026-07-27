"""Turn-boundary binding for natural-language Strategy Pool impact plans."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _STRATEGY_DROP_NAN_CONFIRM_RE,
    _STRATEGY_POOL_IMPACT_REQUEST_RE,
    _resume_strategy_after_nan_label_confirmation,
    _strategy_nan_label_clarification_response,
    _strategy_pool_impact_plan_slots,
)
from marvis.data.workspace import DataSemanticMapping


POOL_HASH = "a" * 64
DATASET_HASH = "b" * 64
SEMANTIC_HASH = "c" * 64
SAMPLE_HASH = "d" * 64
SAMPLE_DESIGN_REF = {
    "artifact_id": "e" * 64,
    "artifact_content_hash": "f" * 64,
    "sample_design_id": "strategy-sample-design-1",
    "sample_design_content_hash": "0" * 64,
    "partition": "development",
}


class _PoolRepository:
    def get_current(self, task_id: str, strategy_type: str):
        assert task_id == "task-1"
        assert strategy_type == "approval"
        return {
            "revision": 7,
            "entries": [
                {
                    "rule_id": "candidate-rule-1",
                    "source": {
                        "evidence_identity": {
                            "dataset_id": "dataset-1",
                            "dataset_content_hash": DATASET_HASH,
                            "workspace_revision": 4,
                            "workspace_generation": 9,
                            "semantic_mapping_hash": SEMANTIC_HASH,
                            "sample_context_hash": SAMPLE_HASH,
                        }
                    },
                }
            ],
        }


class _WorkspaceRepository:
    def __init__(self, mapping: DataSemanticMapping) -> None:
        self.mapping = mapping

    def get_or_default(self, task_id: str):
        assert task_id == "task-1"
        return SimpleNamespace(
            active_dataset_id="dataset-1",
            active_dataset_content_hash=DATASET_HASH,
            revision=4,
            analysis_generation=9,
            semantic_mapping=self.mapping,
        )


def _runtime(tmp_path: Path):
    return SimpleNamespace(settings=SimpleNamespace(db_path=tmp_path / "marvis.sqlite"))


def _context(*columns: str):
    return SimpleNamespace(
        dataset_id="dataset-1",
        dataset_content_hash=DATASET_HASH,
        target_col="bad",
        columns=columns,
    )


def _install_state(monkeypatch, mapping: DataSemanticMapping) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: _PoolRepository(),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.DataWorkspaceRepository",
        lambda db_path: _WorkspaceRepository(mapping),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda pool: POOL_HASH,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.data_semantic_mapping_hash",
        lambda mapping: SEMANTIC_HASH,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._latest_matching_strategy_sample_design_ref",
        lambda *args, **kwargs: dict(SAMPLE_DESIGN_REF),
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "计算审批策略池的通过率和坏账率",
        "测算拒绝策略池逐月风险率",
        "calculate approval pool impact",
    ],
)
def test_turn_uses_governed_workspace_preview_for_every_impact_phrase(
    utterance: str,
) -> None:
    assert _STRATEGY_POOL_IMPACT_REQUEST_RE.search(utterance) is not None


def test_turn_binds_pool_workspace_target_and_unique_semantic_roles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_state(
        monkeypatch,
        DataSemanticMapping(
            target_col="bad",
            field_roles={
                "bad": "target",
                "month": "month",
                "loan": "loan_amount",
            },
        ),
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_impact",
        workflow_inputs={
            "strategy_type": "approval",
            "comparison_mode": "absolute",
            "drop_nan_labels": False,
        },
    )

    slots = _strategy_pool_impact_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        draft,
        context=_context("bad", "month", "loan", "overdue"),
        drop_nan_labels=False,
    )

    assert slots == {
        "strategy_type": "approval",
        "expected_pool_revision": 7,
        "expected_pool_snapshot_hash": POOL_HASH,
        "dataset_id": "dataset-1",
        "expected_dataset_content_hash": DATASET_HASH,
        "workspace_revision": 4,
        "workspace_generation": 9,
        "semantic_mapping_hash": SEMANTIC_HASH,
        "target_col": "bad",
        "sample_design_ref": SAMPLE_DESIGN_REF,
        "comparison_mode": "absolute",
        "drop_nan_labels": False,
        "month_col": "month",
        "loan_amount_col": "loan",
    }
    assert "overdue_amount_col" not in slots
    assert "baseline_strategy_id" not in slots


def test_turn_final_slot_binding_rejects_pool_changed_after_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_state(
        monkeypatch,
        DataSemanticMapping(target_col="bad", field_roles={"bad": "target"}),
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_impact",
        workflow_inputs={"strategy_type": "approval", "comparison_mode": "absolute"},
    )

    with pytest.raises(StrategySetupError, match="用户确认期间已变化"):
        _strategy_pool_impact_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            draft,
            context=_context("bad"),
            drop_nan_labels=True,
            expected_pool_binding={
                "strategy_type": "approval",
                "expected_pool_revision": 6,
                "expected_pool_snapshot_hash": "f" * 64,
            },
        )


def test_nan_confirmation_binds_pool_revision_and_refuses_new_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _MessageRepository:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def add_agent_message(self, task_id: str, **message) -> None:
            assert task_id == "task-1"
            self.messages.append(dict(message))

        def list_agent_messages(self, task_id: str) -> list[dict]:
            assert task_id == "task-1"
            return list(self.messages)

    revision = {"value": 7}
    preview = SimpleNamespace(
        dataset_id="dataset-1",
        target_col="bad",
        identity={"dataset_content_hash": DATASET_HASH},
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_pool_impact_dataset_preview",
        lambda runtime, task: preview,
    )

    def _pool_binding(runtime, task, strategy_type):
        assert strategy_type == "approval"
        return (
            {"entries": [{"rule_id": "candidate-rule-1"}]},
            {
                "strategy_type": "approval",
                "expected_pool_revision": revision["value"],
                "expected_pool_snapshot_hash": str(revision["value"]) * 64,
            },
        )

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_pool_impact_pool_binding",
        _pool_binding,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._prepare_and_run_validated_strategy_request",
        lambda *args, **kwargs: pytest.fail("changed Pool must not create a plan"),
    )
    runtime = SimpleNamespace(
        settings=SimpleNamespace(db_path=tmp_path / "marvis.sqlite"),
        plan_repo=SimpleNamespace(list_plans_for_task=lambda task_id: []),
    )
    repo = _MessageRepository()
    task = SimpleNamespace(id="task-1")
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_impact",
        workflow_inputs={"strategy_type": "approval", "comparison_mode": "absolute"},
    )

    opened = _strategy_nan_label_clarification_response(
        runtime,
        repo,
        task,
        draft=draft,
        context=SimpleNamespace(dataset_id="dataset-1", target_col="bad"),
        n_total=10,
        n_nan=2,
    )
    state = repo.messages[-1]["metadata"]["strategy_nan_label_confirmation"]
    assert opened["code"] == "strategy_drop_nan_labels_confirmation_required"
    assert state["pool_binding"]["expected_pool_revision"] == 7
    assert "仍会保留在总体、动作和金额统计中" in repo.messages[-1]["content"]
    assert _STRATEGY_DROP_NAN_CONFIRM_RE.search(
        "确认将空标签仅从风险分母排除并继续"
    )

    revision["value"] = 8
    resumed = _resume_strategy_after_nan_label_confirmation(
        runtime,
        repo,
        task,
        state,
    )

    assert resumed["code"] == "strategy_pool_context_changed"
    assert "已变化" in repo.messages[-1]["content"]
    assert runtime.plan_repo.list_plans_for_task(task.id) == []


def test_turn_clarifies_multiple_semantic_role_candidates_but_explicit_column_wins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_state(
        monkeypatch,
        DataSemanticMapping(
            target_col="bad",
            field_roles={
                "bad": "target",
                "month_a": "month",
                "month_b": "month",
            },
        ),
    )
    implicit = StandardWorkflowRequestDraft(
        workflow="strategy_pool_impact",
        workflow_inputs={"strategy_type": "approval", "comparison_mode": "absolute"},
    )
    with pytest.raises(StrategySetupError, match="多个 `month`.*明确指定 month_col"):
        _strategy_pool_impact_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            implicit,
            context=_context("bad", "month_a", "month_b"),
            drop_nan_labels=False,
        )

    explicit = StandardWorkflowRequestDraft(
        workflow="strategy_pool_impact",
        workflow_inputs={
            "strategy_type": "approval",
            "comparison_mode": "absolute",
            "month_col": "month_b",
        },
    )
    slots = _strategy_pool_impact_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        explicit,
        context=_context("bad", "month_a", "month_b"),
        drop_nan_labels=False,
    )
    assert slots["month_col"] == "month_b"


def test_turn_verifies_same_task_same_type_canonical_baseline_and_passes_only_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_state(
        monkeypatch,
        DataSemanticMapping(target_col="bad", field_roles={"bad": "target"}),
    )

    class _StrategyRepository:
        def get_strategy_meta(self, strategy_id: str):
            return {
                "id": strategy_id,
                "task_id": "task-1",
                "strategy_type": "approval",
            }

        def get_strategy(self, strategy_id: str):
            return SimpleNamespace(strategy_type="approval", spec=object())

        def get_strategy_spec_hash(self, strategy_id: str):
            return "d" * 64

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyRepository",
        lambda db_path: _StrategyRepository(),
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_impact",
        workflow_inputs={
            "strategy_type": "approval",
            "comparison_mode": "vs_baseline",
            "baseline_strategy_id": "strategy-baseline-1",
        },
    )

    slots = _strategy_pool_impact_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        draft,
        context=_context("bad"),
        drop_nan_labels=False,
    )

    assert slots["baseline_strategy_id"] == "strategy-baseline-1"
    assert not {"baseline_strategy_spec", "baseline_strategy_hash"} & set(slots)


def test_turn_rejects_active_workspace_that_differs_from_pool_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_state(
        monkeypatch,
        DataSemanticMapping(target_col="bad", field_roles={"bad": "target"}),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.data_semantic_mapping_hash",
        lambda mapping: "e" * 64,
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_impact",
        workflow_inputs={"strategy_type": "approval", "comparison_mode": "absolute"},
    )

    with pytest.raises(StrategySetupError, match="Pool 创建时绑定的样本或语义版本不同"):
        _strategy_pool_impact_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            draft,
            context=_context("bad"),
            drop_nan_labels=False,
        )
