"""Field-based intake computes a policy hash and uses original proposal/effect gates."""

from copy import deepcopy

import pytest

from marvis.collection.actions import CollectionPolicy
from tests import test_collection_runtime as runtime_fixture
from tests.test_collection_runtime import plan, approve


@pytest.fixture
def runtime(tmp_path):
    return runtime_fixture.runtime.__wrapped__(tmp_path)


def payload(rt, **changes):
    frozen = rt.proposal["request"]
    action = dict(frozen["strategy"]["default_action"]["value"])
    action.pop("policy_hash")
    action.pop("schema_version", None)
    return {
        "batch_id": "business-intake",
        "policy": frozen["policy"],
        "action": action,
        "cases": frozen["cases"],
        "histories": frozen["histories"],
        "as_of": frozen["as_of"],
        "knowledge_cutoff": frozen["knowledge_cutoff"],
        **changes,
    }


def test_business_proposal_freezes_only_declared_action_and_server_policy_hash(runtime):
    rt = runtime
    response = rt.maker.post(rt.base + "/batch-proposals", json=payload(rt))
    assert response.status_code == 201, response.text
    proposal = response.json()
    assert (
        proposal["status"] == "proposed" and proposal["execution_authorized"] is False
    )
    policy = CollectionPolicy.model_validate(proposal["request"]["policy"])
    strategy = proposal["request"]["strategy"]
    assert strategy["rules"] == [] and strategy["strategy_type"] == "collection"
    assert strategy["default_action"]["value"]["policy_hash"] == policy.content_hash
    assert proposal["preview"]["policy_hash"] == policy.content_hash
    assert rt.maker.get(rt.base + "/batches/business-intake").json()["actions"] == []
    rt.proposal = proposal
    approved = approve(rt, plan(rt, "queue_batch", "催收参考排队"))
    assert approved["status"] == "done"
    queued = rt.maker.get(rt.base + "/batches/business-intake").json()
    assert queued["actions"][0]["state"] == "queued"
    assert queued["effects"][0]["external_action_executed"] is False


@pytest.mark.parametrize("field", ["policy_hash", "strategy", "execution_mode"])
def test_business_proposal_rejects_internal_contract_injection(runtime, field):
    data = payload(runtime)
    if field == "policy_hash":
        data["action"][field] = "a" * 64
    else:
        data[field] = (
            "external" if field == "execution_mode" else {"anything": "unreviewed"}
        )
    assert (
        runtime.maker.post(runtime.base + "/batch-proposals", json=data).status_code
        == 422
    )


def test_proposal_is_role_scoped_and_missing_history_stays_insufficient(runtime):
    data = payload(runtime, histories=[])
    assert (
        runtime.checker.post(runtime.base + "/batch-proposals", json=data).status_code
        == 403
    )
    assert (
        runtime.other.post(runtime.base + "/batch-proposals", json=data).status_code
        == 409
    )
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == 201, response.text
    assert response.json()["preview"]["results"][0]["status"] == "insufficient_evidence"
    assert response.json()["preview"]["estimated_contact_cost_minor"] == 0


@pytest.mark.parametrize("kind", ["review", "hold"])
def test_noncontact_actions_preserve_inapplicable_cost_and_channel(runtime, kind):
    data = payload(runtime)
    data["action"].update(kind=kind, channel=None, estimated_cost_minor=None)
    if kind == "hold":
        data["action"].update(queue_id=None, priority=0)
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == 201, response.text
    value = response.json()["request"]["strategy"]["default_action"]["value"]
    assert value["channel"] is None and value["estimated_cost_minor"] is None
    invalid = deepcopy(data)
    invalid["batch_id"] = "invalid"
    invalid["action"]["channel"] = "sms"
    assert (
        runtime.maker.post(runtime.base + "/batch-proposals", json=invalid).status_code
        == 409
    )


def rule(rule_id, action, **condition):
    return {
        "rule_id": rule_id,
        "combination": "all",
        "action": action,
        "conditions": [
            {
                "field": "dpd",
                "value_type": "number",
                "operator": ">=",
                "value": 20,
                "missing": "no_match",
                **condition,
            }
        ],
    }


def test_ordered_rules_compile_into_original_first_match_and_human_effect_gates(
    runtime,
):
    data = payload(runtime)
    contact = deepcopy(data["action"])
    review = {
        **contact,
        "kind": "review",
        "channel": None,
        "estimated_cost_minor": None,
    }
    data["rules"] = [rule("review-first", review), rule("contact-second", contact)]
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["preview"]["results"][0]["matched_rule_id"] == "review-first"
    strategy = result["request"]["strategy"]
    assert [r["priority"] for r in strategy["rules"]] == [0, 1]
    assert all(
        r["action"]["value"]["policy_hash"] == result["preview"]["policy_hash"]
        for r in strategy["rules"]
    )
    runtime.proposal = result
    approved = approve(runtime, plan(runtime, "queue_batch", "催收参考排队"))
    assert approved["status"] == "done"
    actual = runtime.maker.get(runtime.base + "/batches/business-intake").json()
    assert actual["actions"][0]["state"] == "queued"
    assert actual["effects"][0]["external_action_executed"] is False
    data["batch_id"] = "reordered"
    data["rules"].reverse()
    reordered = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert reordered.status_code == 201, reordered.text
    assert (
        reordered.json()["preview"]["results"][0]["matched_rule_id"] == "contact-second"
    )
    assert (
        runtime.maker.get(runtime.base + "/batches/reordered").json()["actions"] == []
    )


@pytest.mark.parametrize(
    "missing,expected",
    [("no_match", None), ("match", "explicit-missing"), ("error", "blocked")],
)
def test_missing_feature_policy_is_executed_only_by_original_dsl(
    runtime, missing, expected
):
    data = payload(runtime)
    data["cases"][0]["features"]["dpd"] = None
    data["rules"] = [
        rule("explicit-missing", deepcopy(data["action"]), missing=missing)
    ]
    data["action"] = {
        "kind": "hold",
        "queue_id": None,
        "priority": 0,
        "channel": None,
        "estimated_cost_minor": None,
    }
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    if expected == "blocked":
        assert response.status_code == 409
    else:
        assert response.status_code == 201, response.text
        assert response.json()["preview"]["results"][0]["matched_rule_id"] == expected


@pytest.mark.parametrize(
    "variant,status",
    [
        ("literal_type", 422),
        ("feature_type", 409),
        ("undeclared_field", 409),
        ("no_cost", 409),
        ("internal_hash", 422),
    ],
)
def test_typed_rules_do_not_coerce_or_bypass_action_contracts(runtime, variant, status):
    data = payload(runtime)
    data["rules"] = [rule("typed", deepcopy(data["action"]))]
    if variant == "literal_type":
        data["rules"][0]["conditions"][0]["value"] = "20"
    elif variant == "feature_type":
        data["cases"][0]["features"]["dpd"] = "30"
    elif variant == "undeclared_field":
        data["cases"][0]["features"] = {}
    elif variant == "no_cost":
        data["rules"][0]["action"]["estimated_cost_minor"] = None
    else:
        data["rules"][0]["action"]["policy_hash"] = "f" * 64
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == status, response.text


def test_absent_or_empty_rules_keep_existing_batch_identity(runtime):
    from marvis.collection.proposals import CollectionBusinessProposal

    data = payload(runtime)
    before = CollectionBusinessProposal.model_validate(data).batch().content_hash
    after = (
        CollectionBusinessProposal.model_validate({**data, "rules": []})
        .batch()
        .content_hash
    )
    assert before == after


@pytest.mark.parametrize(
    "combination,expected",
    [("all", None), ("any", "mixed-types")],
)
def test_boolean_and_text_groups_use_existing_dsl_without_numeric_coercion(
    runtime, combination, expected
):
    data = payload(runtime)
    data["cases"][0]["features"] = {"segment": "01", "vip": False}
    entry = rule(
        "mixed-types", deepcopy(data["action"]),
        field="segment", value_type="text", operator="==", value="1",
    )
    entry["combination"] = combination
    entry["conditions"].append({
        "field": "vip", "value_type": "boolean", "operator": "==",
        "value": False, "missing": "error",
    })
    data["rules"] = [entry]
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == 201, response.text
    assert response.json()["preview"]["results"][0]["matched_rule_id"] == expected


@pytest.mark.parametrize("operator,matched", [("is_null", True), ("is_not_null", False)])
def test_null_predicate_has_no_ignored_missing_policy(runtime, operator, matched):
    data = payload(runtime)
    data["cases"][0]["features"]["dpd"] = None
    data["rules"] = [rule(
        "null-check", deepcopy(data["action"]), operator=operator,
        value=None, missing=None,
    )]
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == 201, response.text
    assert response.json()["preview"]["results"][0]["matched_rule_id"] == (
        "null-check" if matched else None
    )
    data["batch_id"] = "contradictory-null"
    data["rules"][0]["conditions"][0]["missing"] = "error"
    assert runtime.maker.post(runtime.base + "/batch-proposals", json=data).status_code == 422



def test_unknown_field_still_has_one_declared_type_across_rules(runtime):
    data = payload(runtime)
    data["cases"][0]["features"]["dpd"] = None
    data["rules"] = [
        rule("numeric", deepcopy(data["action"])),
        rule("text", deepcopy(data["action"]), value_type="text", operator="==", value="30"),
    ]
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "collection_material_invalid"
    from marvis.collection.proposals import CollectionBusinessProposal

    with pytest.raises(ValueError, match="collection_rule_field_type_conflict"):
        CollectionBusinessProposal.model_validate(data).batch()



@pytest.mark.parametrize(
    "scenario,expected",
    [("and_false", None), ("or_true", "short-circuit"), ("first_match", "first")],
)
def test_missing_error_only_blocks_conditions_reached_by_existing_dsl(
    runtime, scenario, expected
):
    data = payload(runtime)
    data["cases"][0]["features"] = {"dpd": 30, "unknown": None}
    entry = rule("short-circuit", deepcopy(data["action"]), value=40 if scenario == "and_false" else 20)
    entry["combination"] = "all" if scenario == "and_false" else "any"
    entry["conditions"].append({
        "field": "unknown", "value_type": "number", "operator": ">",
        "value": 0, "missing": "error",
    })
    if scenario == "first_match":
        entry["conditions"] = entry["conditions"][1:]
        data["rules"] = [rule("first", deepcopy(data["action"])), entry]
    else:
        data["rules"] = [entry]
    response = runtime.maker.post(runtime.base + "/batch-proposals", json=data)
    assert response.status_code == 201, response.text
    assert response.json()["preview"]["results"][0]["matched_rule_id"] == expected
