"""Formal gateway admission before any local fixture call."""
import time

from test_runtime_agent_benchmark import fixture_model
from test_runtime_aggregate_budget import events, price, profile_for, request

from marvis.orchestrator.eval.runtime_contracts import RuntimeBudget, digest
from marvis.orchestrator.eval.runtime_runner import _model_gateway


def limits(raw):
    return {"protocol": "single_text_utf8_bound_v1", "context_window": 4096, "max_output_tokens": 64,
            "framing_token_allowance": 32, "price_book_sha256": digest(raw), "retain_full_reservations": True,
            "approval": {"path": "local-fixture-only.json", "sha256": "a" * 64}}


def test_formal_gateway_rejects_oversized_input_before_provider_and_retains_full_reservation(tmp_path):
    prices = price()
    path = tmp_path / "events.jsonl"
    with fixture_model() as (model, calls):
        with _model_gateway(profile_for(model), {}, path, max_attempts=10, deadline=time.monotonic() + 15,
                            budget=RuntimeBudget(max_total_tokens=4160, max_cost=1., currency="CNY"),
                            price_bytes=prices, formal_provider_limits=limits(prices)) as profile:
            assert request(profile, messages=[{"role": "user", "content": "x" * 5000}]).status_code == 400
            assert calls == []
            assert request(profile, messages=[{"role": "user", "content": "local fixture"}]).status_code == 200
            assert request(profile, messages=[{"role": "user", "content": "local fixture"}]).status_code == 429
    assert len(calls) == 1
    rows = events(path)
    admitted = [row for row in rows if row.get("event") == "started"]
    assert admitted and admitted[0]["retain_full_reservations"] is True
    assert admitted[0]["input_token_upper_bound"] > 0


def test_formal_gateway_rejects_profile_or_budget_bypass_before_provider(tmp_path):
    import pytest
    from marvis.orchestrator.eval.runtime_runner import RuntimeJourneyError
    prices = price()
    with fixture_model() as (model, calls):
        for modified in ({"context_window": 8192}, {"enable_thinking": True}, {"max_output_tokens": 128}):
            with pytest.raises(RuntimeJourneyError, match="formal_gateway_budget_binding_invalid"):
                with _model_gateway(profile_for(model) | modified, {}, tmp_path / "events.jsonl", max_attempts=2,
                                    deadline=time.monotonic() + 3, price_bytes=prices,
                                    budget=RuntimeBudget(max_cost=1., currency="CNY"), formal_provider_limits=limits(prices)):
                    pytest.fail("invalid frozen profile must not be admitted")
        with pytest.raises(RuntimeJourneyError, match="formal_gateway_budget_binding_invalid"):
            with _model_gateway(profile_for(model), {}, tmp_path / "events.jsonl", max_attempts=2,
                                deadline=time.monotonic() + 3, price_bytes=prices, formal_provider_limits=limits(prices)):
                pytest.fail("missing per-case cost cap must not be admitted")
    assert not calls
