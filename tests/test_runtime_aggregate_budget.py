"""Shared admission limits tested through the actual loopback model gateway."""

from concurrent.futures import ThreadPoolExecutor
import json
import time

import httpx
import pytest
from pydantic import ValidationError

from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.orchestrator.eval.runtime_contracts import RuntimeBudget, RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import _model_gateway, RuntimeJourneyError, run_runtime_suite
from marvis.orchestrator.eval.runtime_scoring import usage_summary
from test_runtime_agent_benchmark import fixture_model


def request(profile, **overrides):
    return httpx.post(
        profile["api_base_url"] + "/chat/completions",
        headers={"Authorization": "Bearer " + profile["api_key"]},
        json={"model": profile["model_name"], "max_tokens": 64,
              "messages": [{"role": "user", "content": "synthetic"}], **overrides}, timeout=5,
    )


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def profile_for(model):
    return {**model.profile("fixture_model"), "context_window": 4096, "max_output_tokens": 64}


def price(**changes):
    return json.dumps({"model_name": "fixture", "provider": "local_fixture", "currency": "CNY",
                       "source": "synthetic_test_rates", "effective_date": "2026-09-28",
                       "prompt_per_million": 10.0, "completion_per_million": 20.0, **changes}).encode()


@pytest.mark.parametrize("fields", [
    {"max_total_tokens": True}, {"max_total_tokens": -1}, {"max_cost": 1.0},
    {"currency": "CNY"}, {"max_cost": float("nan"), "currency": "CNY"},
    {"max_cost": float("inf"), "currency": "CNY"}, {"max_cost": -1.0, "currency": "CNY"},
])
def test_invalid_budget_rejected_before_runtime(fields):
    with pytest.raises(ValidationError):
        RuntimeBudget(**fields)


def test_legacy_case_identity_keeps_original_budget_keys():
    legacy = {"wall_seconds": 180, "max_llm_attempts": 30,
              "max_http_requests": 150, "max_output_tokens_per_attempt": 2048}
    assert RuntimeBudget().model_dump() == legacy
    case = RuntimeCase(id="old", revision="1", family="feature_analysis",
                       task={"task_type": "feature_analysis"}, business_constraints_source="synthetic")
    dumped = case.model_dump()
    assert dumped["budget"] == legacy
    assert digest(dumped) == digest({**dumped, "budget": legacy})
    assert json.loads(case.model_dump_json())["budget"] == legacy
    assert RuntimeBudget(max_total_tokens=0).model_dump()["max_total_tokens"] == 0


def test_concurrent_calls_reserve_before_outbound(tmp_path):
    path = tmp_path / "attempts.jsonl"
    with fixture_model(delay=0.4) as (model, calls):
        with _model_gateway(profile_for(model), {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=4160)) as profile:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(request, profile)
                deadline = time.monotonic() + 3
                while not calls and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert calls, "first reservation must be in flight"
                assert request(profile).status_code == 429
                assert first.result().status_code == 200
    assert len(calls) == 1
    summary = usage_summary(events(path), model_name="fixture", price_bytes=None)
    assert summary["token_budget_status"] == "admission_blocked"
    assert summary["aggregate_budget"]["accounted_tokens_including_reservations"] == 120
    assert summary["aggregate_budget"]["provider_billing_guarantee"] is False


@pytest.mark.parametrize("currency_budget", [False, True])
def test_known_usage_refunds_and_later_call_is_blocked(tmp_path, currency_budget):
    path = tmp_path / "attempts.jsonl"
    budget = RuntimeBudget(max_cost=0.044, currency="CNY") if currency_budget else RuntimeBudget(max_total_tokens=4280)
    prices = price() if currency_budget else None
    with fixture_model() as (model, calls):
        with _model_gateway(profile_for(model), {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10, budget=budget,
                            price_bytes=prices) as profile:
            assert request(profile).status_code == 200
            assert request(profile).status_code == 200
            assert request(profile).status_code == 429
    assert len(calls) == 2
    summary = usage_summary(events(path), model_name="fixture", price_bytes=prices)
    assert summary["aggregate_budget"]["accounted_tokens_including_reservations"] == 240
    assert summary["aggregate_budget"]["pending_attempts"] == 0
    if currency_budget:
        assert summary["cost_budget_status"] == "admission_blocked"
        assert float(summary["aggregate_budget"]["accounted_cost_including_reservations"]) == pytest.approx(0.0028)
        assert summary["aggregate_budget"]["price_book_sha256"] == digest(prices)


def test_unknown_transport_usage_cannot_free_retry_capacity(tmp_path):
    path = tmp_path / "attempts.jsonl"
    with fixture_model(missing_usage=True, fail_first=True) as (model, calls):
        with _model_gateway(profile_for(model), {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=4160)) as profile:
            with pytest.raises(LLMClientError):
                OpenAICompatibleLLMClient(profile).complete(system_prompt="test", user_prompt="synthetic", stream=False)
    assert len(calls) == 1
    summary = usage_summary(events(path), model_name="fixture", price_bytes=None)
    assert summary["usage_status"] == "unknown"
    assert summary["prompt_tokens"] is None
    assert summary["aggregate_budget"]["unknown_usage_attempts"] == 1
    assert summary["aggregate_budget"]["accounted_tokens_including_reservations"] == 4160


@pytest.mark.parametrize("prices", [None, price(model_name="other"), price(currency="USD")])
def test_cost_limit_requires_matching_frozen_price_before_any_call(tmp_path, prices):
    with fixture_model() as (model, calls):
        with pytest.raises(RuntimeJourneyError, match="cost_budget"):
            with _model_gateway(profile_for(model), {}, tmp_path / "attempts.jsonl", max_attempts=3,
                                deadline=time.monotonic() + 10,
                                budget=RuntimeBudget(max_cost=1.0, currency="CNY"), price_bytes=prices):
                pytest.fail("must reject before yielding")
    assert not calls


def test_provider_ceiling_violation_is_visible_and_blocks_next_attempt(tmp_path):
    # The provider fixture reports 100 prompt tokens, larger than this declared
    # ceiling. We cannot undo that charge, but must report and stop further calls.
    path = tmp_path / "attempts.jsonl"
    with fixture_model() as (model, calls):
        with _model_gateway({**profile_for(model), "context_window": 50}, {}, path,
                            max_attempts=10, deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=500)) as profile:
            assert request(profile).status_code == 200
            assert request(profile).status_code == 429
    assert len(calls) == 1
    summary = usage_summary(events(path), model_name="fixture", price_bytes=None)
    assert summary["token_budget_status"] == "provider_ceiling_violated"
    assert summary["aggregate_budget"]["accounted_tokens_including_reservations"] == 164


@pytest.mark.parametrize("overrides", [{"n": 3}, {"best_of": 2}, {"max_completion_tokens": 9999}])
def test_output_multiplier_or_alternate_limit_never_reaches_provider(tmp_path, overrides):
    path = tmp_path / "attempts.jsonl"
    with fixture_model() as (model, calls):
        with _model_gateway(profile_for(model), {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=4160)) as profile:
            assert request(profile, **overrides).status_code == 400
    assert not calls
    assert any(e.get("reason") == "unsupported_output_budget_parameters" for e in events(path))


@pytest.mark.parametrize("done,final_usage", [(False, True), (True, False)])
def test_stream_partial_usage_does_not_refund_reservation(tmp_path, done, final_usage):
    from test_runtime_response_observation import provider

    path = tmp_path / "attempts.jsonl"
    usage = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 0}}
    finish = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
    chunks = [finish, usage] if final_usage else [usage, finish]
    body = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
    if done:
        body += b"data: [DONE]\n\n"
    with provider([(200, "text/event-stream", body)]) as (upstream, calls):
        with _model_gateway({**upstream, "context_window": 4096}, {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=4280)) as profile:
            assert request(profile).status_code == 200
            assert request(profile).status_code == 429
    assert len(calls) == 1
    summary = usage_summary(events(path), model_name="fixture", price_bytes=None)
    assert summary["usage_status"] == "unknown"
    assert summary["aggregate_budget"]["accounted_tokens_including_reservations"] == 4160


def test_observed_reasoning_conflict_is_not_hidden_by_small_completion(tmp_path):
    from test_runtime_response_observation import provider, json_response

    path = tmp_path / "attempts.jsonl"
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                          "completion_tokens_details": {"reasoning_tokens": 10000}}}
    with provider([json_response(response)]) as (upstream, calls):
        with _model_gateway({**upstream, "context_window": 4096}, {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=4280)) as profile:
            assert request(profile).status_code == 200
            assert request(profile).status_code == 429
    assert len(calls) == 1
    summary = usage_summary(events(path), model_name="fixture", price_bytes=None)
    assert summary["token_budget_status"] == "provider_ceiling_violated"
    assert summary["usage_status"] == "unknown"
    assert summary["aggregate_budget"]["accounted_tokens_including_reservations"] == 14096


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 5000}, {"completion_tokens": 100},
    {"completion_tokens_details": {"reasoning_tokens": 100}},
])
def test_known_partial_usage_violation_blocks_even_when_other_fields_missing(tmp_path, usage):
    from test_runtime_response_observation import provider, json_response

    path = tmp_path / "attempts.jsonl"
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": usage}
    with provider([json_response(response)]) as (upstream, calls):
        with _model_gateway({**upstream, "context_window": 4096}, {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=100000)) as profile:
            assert request(profile).status_code == 200
            assert request(profile).status_code == 429
    assert len(calls) == 1
    summary = usage_summary(events(path), model_name="fixture", price_bytes=None)
    assert summary["token_budget_status"] == "provider_ceiling_violated"
    assert summary["usage_status"] == "unknown"


@pytest.mark.parametrize("later_usage", [{"completion_tokens": 0}, {"prompt_tokens": 100, "completion_tokens": 0}])
def test_partial_or_regressing_usage_after_complete_chunk_cannot_refund(tmp_path, later_usage):
    from test_runtime_response_observation import provider

    path = tmp_path / "attempts.jsonl"
    chunks = [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        {"choices": [], "usage": later_usage},
    ]
    body = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"
    with provider([(200, "text/event-stream", body)]) as (upstream, calls):
        with _model_gateway({**upstream, "context_window": 4096}, {}, path, max_attempts=10,
                            deadline=time.monotonic() + 10,
                            budget=RuntimeBudget(max_total_tokens=4280)) as profile:
            assert request(profile).status_code == 200
            assert request(profile).status_code == 429
    assert len(calls) == 1
    summary = usage_summary(events(path), model_name="fixture", price_bytes=None)
    assert summary["usage_status"] == "unknown"
    assert summary["aggregate_budget"]["accounted_tokens_including_reservations"] == 4160


def test_suite_enforces_public_case_budget_without_expected_in_runtime(tmp_path):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite

    paths = write_synthetic_suite(tmp_path / "synthetic")
    public = json.loads(paths["cases"].read_text())
    public["cases"] = [public["cases"][1]]
    public["cases"][0]["budget"]["max_total_tokens"] = 0
    public["cases"][0]["budget"]["wall_seconds"] = 15
    paths["cases"].write_text(json.dumps(public))
    with fixture_model() as (model, calls):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
                                   dataset_root=paths["dataset_root"], model=model,
                                   model_source="fixture_model", output_dir=tmp_path / "evidence")
    assert not calls
    case = report["cases"][0]
    assert case["runtime_status"] == "budget_exceeded"
    assert case["score"]["passed"] is False
    assert case["score"]["usage"]["token_budget_status"] == "admission_blocked"
    assert report["denominator"] == 1
