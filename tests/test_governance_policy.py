from pathlib import Path

import pytest

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import Plan, PlanStep, plan_from_dict, plan_to_dict
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.errors import ManifestError
from marvis.plugins.manifest import (
    EffectTargetPolicy,
    GovernancePolicy,
    ToolRef,
    manifest_to_dict,
    parse_manifest,
)
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _manifest(*, policy: dict | None = None):
    tool = {
        "name": "protected_write",
        "summary": "Change a governed strategy",
        "input_schema": {
            "type": "object",
            "properties": {"strategy_id": {"type": "string"}},
            "required": ["strategy_id"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
        "determinism": "deterministic",
        "timeout_seconds": 10,
        "failure_policy": "fail",
        "entrypoint": "tool_protected_write",
        "side_effects": ["write:strategy"],
    }
    if policy is not None:
        tool["policy"] = policy
    return {
        "name": "governed",
        "version": "1.0.0",
        "display_name": "Governed",
        "description": "Governance policy fixture",
        "module": "governed.tools",
        "tools": [tool],
        "hooks": [],
        "permissions": ["write:strategy"],
    }


def _required_policy() -> dict:
    return {
        "schema_version": "tool-policy.v1",
        "human_decision_gate": "required",
        "effect_authorization": "required",
        "effect_target": {
            "kind": "strategy",
            "id_input": "strategy_id",
            "expected_statuses": ["draft"],
            "result_status": "adopted",
        },
    }


def _registry(tmp_path: Path, *, policy: dict | None = None) -> ToolRegistry:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    plugins.register(
        parse_manifest(
            _manifest(policy=policy or _required_policy()),
            builtin=True,
        )
    )
    return ToolRegistry(plugins)


def _plan(step: PlanStep) -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="governed write",
        source="generated",
        template_id=None,
        steps=[step],
        autonomy_level=1,
    )


def _step(*, policy: GovernancePolicy | None = None, needs_confirmation: bool = False):
    return PlanStep(
        id="step-1",
        plan_id="plan-1",
        index=0,
        title="Protected write",
        tool_ref=ToolRef("governed", "protected_write"),
        inputs={"strategy_id": "strategy-1"},
        depends_on=[],
        post_checks=[],
        needs_confirmation=needs_confirmation,
        policy=policy or GovernancePolicy(),
    )


def test_manifest_policy_round_trips_and_defaults_fail_open_only_for_unprotected_tools():
    protected = parse_manifest(_manifest(policy=_required_policy()), builtin=True)
    policy = protected.tools[0].policy

    assert policy.human_decision_gate == "required"
    assert policy.effect_authorization == "required"
    assert policy.effect_target == EffectTargetPolicy(
        kind="strategy",
        id_input="strategy_id",
        expected_statuses=("draft",),
        result_status="adopted",
    )
    assert parse_manifest(manifest_to_dict(protected), builtin=True) == protected

    unprotected = parse_manifest(_manifest(), builtin=True)
    assert unprotected.tools[0].policy == GovernancePolicy()


@pytest.mark.parametrize(
    "policy",
    [
        {
            "human_decision_gate": "optional",
            "effect_authorization": "none",
        },
        {
            "human_decision_gate": "none",
            "effect_authorization": "required",
            "effect_target": {
                "kind": "strategy",
                "id_input": "strategy_id",
                "expected_statuses": ["draft"],
                "result_status": "adopted",
            },
        },
        {
            "human_decision_gate": "required",
            "effect_authorization": "required",
        },
    ],
)
def test_manifest_rejects_invalid_governance_policy(policy):
    with pytest.raises(ManifestError, match="policy"):
        parse_manifest(_manifest(policy=policy), builtin=True)


def test_plan_policy_snapshot_round_trips():
    policy = GovernancePolicy.from_dict(_required_policy())
    restored = plan_from_dict(plan_to_dict(_plan(_step(policy=policy, needs_confirmation=True))))

    assert restored.steps[0].policy == policy
    assert restored.steps[0].needs_confirmation is True


def test_validator_rejects_policy_downgrade_and_missing_human_gate(tmp_path):
    validator = PlanValidator(_registry(tmp_path))

    problems = validator.validate(_plan(_step()))

    assert any("human_decision_gate" in problem for problem in problems)
    assert any("effect_authorization" in problem for problem in problems)


def test_validator_accepts_required_policy_snapshot_with_confirmation(tmp_path):
    validator = PlanValidator(_registry(tmp_path))
    policy = GovernancePolicy.from_dict(_required_policy())

    assert validator.validate(_plan(_step(policy=policy, needs_confirmation=True))) == []


def test_validator_forbids_human_decision_tools_in_subagent_grants(tmp_path):
    policy_payload = {
        "schema_version": "tool-policy.v1",
        "human_decision_gate": "required",
        "effect_authorization": "none",
    }
    validator = PlanValidator(_registry(tmp_path, policy=policy_payload))
    step = _step(
        policy=GovernancePolicy.from_dict(policy_payload),
        needs_confirmation=True,
    )
    step.sub_agent_scope = "delegate governed decision"
    step.granted_tools = [ToolRef("governed", "protected_write")]

    problems = validator.validate(_plan(step))

    assert any("human-decision-gated" in problem for problem in problems)
