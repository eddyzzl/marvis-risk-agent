"""Hard local request admission and no returned-usage capacity refund in formal mode."""
from decimal import Decimal

import pytest

from marvis.orchestrator.eval.runtime_budget import AdmissionBudgetExceeded, AggregateBudget
from marvis.orchestrator.eval.runtime_formal_budget import ProviderLimits, validate_formal_payload
from marvis.orchestrator.eval.runtime_scoring import PriceBook


def _limits():
    return ProviderLimits(protocol="single_text_utf8_bound_v1", context_window=4096, max_output_tokens=32,
                          framing_token_allowance=32, price_book_sha256="a" * 64, retain_full_reservations=True,
                          approval={"path": "approval.json", "sha256": "b" * 64})


def _payload():
    return {"model": "fixture", "messages": [{"role": "system", "content": "Follow user constraints"},
            {"role": "user", "content": "请分析样本"}], "max_tokens": 16, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.}


def test_forwarded_utf8_bytes_and_framing_are_bounded_before_provider_call():
    body, metadata = validate_formal_payload(_payload(), model_name="fixture", limits=_limits())
    assert metadata["input_utf8_bytes"] == len(body)
    assert metadata["input_token_upper_bound"] == len(body) + 32
    assert metadata["input_token_upper_bound"] + 16 <= 4096
    assert metadata["retain_full_reservations"] is True
    assert b"\\u" not in body  # actual UTF-8 bytes, not a character-count estimate


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["messages"][1].update(content="界" * 1500), "input_ceiling"),
    (lambda p: p["messages"][1].update(content=[{"type": "image_url", "image_url": "http://example.com/x"}]), "plain_text"),
    (lambda p: p.update(tools=[{"type": "web_search"}]), "outside_provider"),
    (lambda p: p.update(max_completion_tokens=100), "outside_provider"),
    (lambda p: p.update(max_tokens=33), "outside_provider"),
    (lambda p: p.update(n=2), "one_completion"),
    (lambda p: p.update(best_of=True), "one_completion"),
    (lambda p: p.update(thinking={"type": "enabled"}), "reasoning_budget"),
    (lambda p: p.update(stream_options={"include_usage": True, "other": 1}), "stream_options"),
    (lambda p: p.update(stream_options={"include_usage": 1}), "stream_options"),
    (lambda p: p.update(response_format={"type": "json_schema", "json_schema": {"$ref": "http://example.com/schema"}}), "external_reference"),
    (lambda p: p.update(response_format={"type": "json_schema", "json_schema": {"$dynamicRef": "http://example.com/schema"}}), "external_reference"),
])
def test_unknown_unbounded_or_multiplied_payloads_fail_before_forwarding(mutation, reason):
    payload = _payload()
    mutation(payload)
    with pytest.raises(ValueError, match=reason):
        validate_formal_payload(payload, model_name="fixture", limits=_limits())


def _budget(*, formal):
    price = PriceBook(model_name="fixture", provider="test", currency="TEST", source="test", effective_date="2026-11-04",
                      prompt_per_million=1., completion_per_million=2.)
    return AggregateBudget(context_window=4096, max_total_tokens=4136, max_cost=.0042, currency="TEST", price=price,
                           retain_full_reservations=formal)


def test_formal_retains_full_reservation_even_when_provider_reports_zero_usage():
    budget = _budget(formal=True)
    original = budget.reserve("first", 32)
    result = budget.settle("first", {"prompt_tokens": 0, "completion_tokens": 0, "usage_final": True})
    assert result["reservation_status"] == "retained_formal_full_reservation"
    assert budget.tokens == 4128 and budget.cost == Decimal(original["reserved_cost"])
    with pytest.raises(AdmissionBudgetExceeded):
        budget.reserve("second", 32)
    assert budget.policy()["provider_billing_guarantee"] is False


def test_nonformal_legacy_settlement_and_policy_remain_unchanged():
    budget = _budget(formal=False)
    budget.reserve("first", 32)
    assert budget.settle("first", {"prompt_tokens": 0, "completion_tokens": 0, "usage_final": True})["reservation_status"] == "settled_reported_usage"
    assert budget.tokens == 0 and "retain_full_reservations" not in budget.policy()
    budget.reserve("second", 32)
