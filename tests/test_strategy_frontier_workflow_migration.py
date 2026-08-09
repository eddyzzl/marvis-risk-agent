from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from marvis.agent import strategy_workflows
from marvis.agent import turn_handlers
from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_workflows import migrated_workflow_requirements
from marvis.agent.strategy_workflows._catalog import build_catalog
from marvis.agent.strategy_workflows.contracts import (
    PreparedStrategyPlan,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowSpec,
)


REVISION_ID = "interactive-tree-revision-" + "a" * 32
NODE_ID = "node-" + "1" * 20
LEAF_ID = "leaf-" + "2" * 20


def test_workflow_spec_rejects_partially_migrated_adapters() -> None:
    with pytest.raises(ValueError, match="all three adapters"):
        StrategyWorkflowSpec(
            workflow_id="partial",
            fresh=True,
            replayable=True,
            manual=False,
            requirements=StrategyWorkflowRequirements(
                dataset=None,
                target=None,
                complete_labels=None,
            ),
            validator=lambda inputs, _context: dict(inputs),
        )


def test_tree_requirements_are_resolved_by_the_canonical_spec() -> None:
    tree_spec = next(
        spec
        for spec in build_catalog()
        if spec.workflow_id == "automatic_tree_candidate_build"
    )

    assert tree_spec.requirements == StrategyWorkflowRequirements(
        dataset=True,
        target=True,
        complete_labels=True,
    )
    assert migrated_workflow_requirements(tree_spec.workflow_id) == (
        True,
        True,
        True,
    )


def test_canonical_contract_and_adapter_queries_are_explicitly_exported() -> None:
    assert {
        "StrategyWorkflowRequirements",
        "StrategyWorkflowSpec",
        "is_migrated_workflow",
        "migrated_workflow_confirmation",
        "migrated_workflow_requirements",
    } <= set(strategy_workflows.__all__)


def test_plan_preparation_rejects_a_mismatched_workflow_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = StrategyWorkflowSpec(
        workflow_id="declared",
        fresh=True,
        replayable=True,
        manual=False,
        requirements=StrategyWorkflowRequirements(False, False, False),
        template_ids=("declared-template",),
        validator=lambda inputs, _context: dict(inputs),
        confirmation=lambda _inputs: "confirm",
        preparer=lambda _inputs, _context: PreparedStrategyPlan(
            workflow_id="different",
            template_id="declared-template",
            slots=MappingProxyType({}),
        ),
    )
    monkeypatch.setitem(strategy_workflows._BY_ID, "declared", spec)

    with pytest.raises(RuntimeError, match="prepared mismatched workflow"):
        strategy_workflows.prepare_strategy_plan(
            "declared",
            {},
            context=StrategyWorkflowPreparationContext(dataset_id=None),
        )


def test_singleton_frontier_resolves_and_prepares_through_canonical_spec() -> None:
    workflow = "interactive_tree_frontier_materialization"
    resolved = strategy_workflows.resolve_strategy_request(
        workflow,
        {
            "revision_id": REVISION_ID,
            "source_node_id": NODE_ID,
            "selection_reason": "  人工\t确认  e\u0301 节点  ",
        },
        context=StrategyWorkflowResolutionContext(
            allowed_columns=(),
            target_col=None,
        ),
    )

    assert resolved.workflow_inputs == {
        "revision_id": REVISION_ID,
        "source_node_id": NODE_ID,
        "selection_reason": "人工 确认 é 节点",
    }
    assert "交互树前沿精确物化" in resolved.confirmation
    prepared = strategy_workflows.prepare_strategy_plan(
        resolved.workflow_id,
        resolved.workflow_inputs,
        context=StrategyWorkflowPreparationContext(dataset_id=None),
    )
    assert prepared.template_id == (
        "strategy_interactive_tree_frontier_materialization"
    )
    assert prepared.to_runtime_slots() == dict(resolved.workflow_inputs)


def test_group_frontier_resolves_and_prepares_through_canonical_spec() -> None:
    workflow = "interactive_tree_frontier_group_materialization"
    resolved = strategy_workflows.resolve_strategy_request(
        workflow,
        {
            "revision_id": REVISION_ID,
            "source_node_ids": [LEAF_ID, NODE_ID],
            "selection_reason": "  人工\t确认 OR 组合  ",
        },
        context=StrategyWorkflowResolutionContext(
            allowed_columns=(),
            target_col=None,
        ),
    )

    assert resolved.workflow_inputs == {
        "revision_id": REVISION_ID,
        "source_node_ids": (LEAF_ID, NODE_ID),
        "selection_reason": "人工 确认 OR 组合",
    }
    assert "交互树前沿显式 OR 分组物化" in resolved.confirmation
    prepared = strategy_workflows.prepare_strategy_plan(
        resolved.workflow_id,
        resolved.workflow_inputs,
        context=StrategyWorkflowPreparationContext(dataset_id=None),
    )
    assert prepared.template_id == (
        "strategy_interactive_tree_frontier_group_materialization"
    )
    assert prepared.to_runtime_slots() == {
        "revision_id": REVISION_ID,
        "source_node_ids": [LEAF_ID, NODE_ID],
        "selection_reason": "人工 确认 OR 组合",
    }


def test_frontier_migration_preserves_id_whitespace_normalization() -> None:
    resolved = strategy_workflows.resolve_strategy_request(
        "interactive_tree_frontier_materialization",
        {
            "revision_id": f"  {REVISION_ID}\n",
            "source_node_id": f"\t{NODE_ID}  ",
        },
        context=StrategyWorkflowResolutionContext(
            allowed_columns=(),
            target_col=None,
        ),
    )

    assert resolved.workflow_inputs == {
        "revision_id": REVISION_ID,
        "source_node_id": NODE_ID,
    }


def test_migrated_preparation_precedes_legacy_routes_and_passes_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_calls: list[str] = []
    started: dict[str, object] = {}

    def prepare(workflow_id, _workflow_inputs, *, context):
        prepared_calls.append(workflow_id)
        assert context.dataset_id is None
        return PreparedStrategyPlan(
            workflow_id=workflow_id,
            template_id="canonical-template",
            slots=MappingProxyType({"canonical": True}),
            success_criteria=(
                MappingProxyType({"metric": "published_pointer_count", "min": 1}),
            ),
        )

    def start(_runtime, _repo, _task, **kwargs):
        started.update(kwargs)
        return {"status": "started"}

    monkeypatch.setattr(turn_handlers, "prepare_strategy_plan", prepare)
    monkeypatch.setattr(
        turn_handlers,
        "_start_confirmed_strategy_plan",
        start,
    )

    result = turn_handlers._run_validated_strategy_request(
        SimpleNamespace(),
        object(),
        SimpleNamespace(id="task-1"),
        StandardWorkflowRequestDraft(
            workflow="interactive_tree_frontier_materialization",
            workflow_inputs={
                "revision_id": REVISION_ID,
                "source_node_id": NODE_ID,
            },
        ),
        context=None,
        auto_start=True,
        drop_nan_labels=False,
    )

    assert result == {"status": "started"}
    assert prepared_calls == ["interactive_tree_frontier_materialization"]
    assert started == {
        "template_id": "canonical-template",
        "slots": {"canonical": True},
        "success_criteria": [
            {"metric": "published_pointer_count", "min": 1}
        ],
        "auto_start": True,
    }
