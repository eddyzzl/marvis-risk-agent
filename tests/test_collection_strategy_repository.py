"""Canonical collection actions never travel through the V1 legacy adapter."""

from dataclasses import asdict, replace
import json

import pytest

from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.packs.strategy.dsl import canonical_strategy_json, StrategySpec
from marvis.packs.strategy.strategy import build_strategy_from_spec
from test_collection_planning import action, policy, spec
from test_modeling_pack import _runtime


@pytest.fixture
def stored(tmp_path):
    *_, settings, task = _runtime(tmp_path)
    p = policy()
    contract = spec(
        p,
        rules=[
            {
                "rule_id": "priority",
                "priority": 1,
                "condition": {
                    "op": "compare",
                    "field": "dpd",
                    "operator": ">",
                    "value": 10,
                },
                "action": {**action(p), "reason_code": "DELINQUENCY"},
            }
        ],
    )
    strategy = build_strategy_from_spec(contract)
    repo = StrategyRepository(settings.db_path)
    repo.create_strategy(task.id, strategy)
    return repo, strategy, settings, task


def test_collection_storage_read_and_version_preserve_typed_actions(stored):
    repo, strategy, _, _ = stored
    read = repo.get_strategy(strategy.id)
    assert canonical_strategy_json(read.spec) == canonical_strategy_json(strategy.spec)
    assert read.rules[0].decision == "collection"
    assert read.rules[0].value == strategy.spec.rules[0].action.value
    child = repo.new_version_from(strategy.id)
    assert child.spec.schema_version == "strategy.dsl.v2"
    assert child.rules == strategy.rules
    assert child.spec.default_action == strategy.spec.default_action
    payload = child.spec.to_dict()
    payload["rules"][0]["action"]["value"]["priority"] = 200
    next_version = repo.new_version_from(
        child.id, strategy_spec=StrategySpec.from_dict(payload)
    )
    assert repo.get_strategy(next_version.id).rules[0].value["priority"] == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "hold"),
        ("queue_id", "other"),
        ("policy_hash", "0" * 64),
        ("channel", "phone"),
        ("priority", 500),
        ("estimated_cost_minor", 21),
        ("estimated_cost_minor", 20.0),
        ("unexpected", "field"),
    ],
)
@pytest.mark.parametrize("target", ["default", "rule"])
def test_collection_action_projection_tampering_is_rejected(
    stored, field, value, target
):
    repo, strategy, settings, task = stored
    if target == "default":
        changed = replace(
            strategy,
            id="tampered",
            default_decision={**strategy.default_decision, field: value},
        )
        column = "default_decision_json"
        projection = changed.default_decision
    else:
        rule = replace(
            strategy.rules[0], value={**strategy.rules[0].value, field: value}
        )
        changed = replace(strategy, id="tampered", rules=(rule,))
        column = "rules_json"
        projection = [asdict(rule)]
    with pytest.raises(ValueError, match="compatibility fields"):
        repo.create_strategy(task.id, changed)
    # Read-time consistency is checked too, even if the legacy display columns drift.
    with connect(settings.db_path) as conn:
        conn.execute(
            f"UPDATE strategies SET {column}=? WHERE id=?",
            (json.dumps(projection), strategy.id),
        )
    with pytest.raises(ValueError, match="compatibility fields"):
        repo.get_strategy(strategy.id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision", "segment"),
        ("rule_id", "other"),
        ("priority", 2),
        ("reason_code", "OTHER"),
        ("condition", '{"op":"literal","value":true}'),
    ],
)
def test_collection_rule_identity_and_condition_cannot_drift(stored, field, value):
    repo, strategy, _, task = stored
    changed = replace(
        strategy, id="other", rules=(replace(strategy.rules[0], **{field: value}),)
    )
    with pytest.raises(ValueError, match="compatibility fields"):
        repo.create_strategy(task.id, changed)
