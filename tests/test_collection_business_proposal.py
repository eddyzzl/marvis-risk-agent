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
