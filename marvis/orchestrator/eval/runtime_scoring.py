"""Post-execution, allowlisted predicates over real, authenticated results.

Expected content is never an Agent input. A successful fixture run is runtime
regression evidence only; acceptance also requires real model calls, frozen
independent cases, thresholds and business approval outside this local runner.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from .runtime_contracts import StrictModel, digest


class Assertion(StrictModel):
    kind: Literal[
        "tool_succeeded",
        "output_equals",
        "output_close",
        "output_length",
        "http_status",
        "message_metadata",
        "dataset_rows",
        "artifact_exists",
    ]
    tool: str = ""
    path: list[str | int] = Field(default_factory=list)
    stage: str = ""
    value: str | int | float | bool | None = None
    tolerance: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class ExpectedCase(StrictModel):
    result: Literal["done", "clarification", "rejected", "failed", "cancelled"]
    assertions: list[Assertion] = Field(min_length=1)


class ExpectedSuite(StrictModel):
    schema_version: Literal[1] = 1
    cases: dict[str, ExpectedCase]


class PriceBook(StrictModel):
    model_name: str
    provider: str
    currency: str = Field(min_length=1)
    source: str = Field(min_length=1)
    effective_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    prompt_per_million: float = Field(ge=0, allow_inf_nan=False)
    completion_per_million: float = Field(ge=0, allow_inf_nan=False)


def _at(value, path):
    for part in path:
        value = value[part]
    return value


def _assertion(assertion, record, private):
    if assertion.kind == "http_status":
        values = [
            event.get("status_code")
            for event in record["http_events"]
            if event["stage"] == assertion.stage
        ]
        return bool(values) and all(value == assertion.value for value in values)
    if assertion.kind == "message_metadata":
        for message in private["messages"]:
            try:
                if _at(message.get("metadata", {}), assertion.path) == assertion.value:
                    return True
            except (KeyError, IndexError, TypeError):
                pass
        return False
    steps = [
        step
        for step in record.get("execution", {}).get("steps", [])
        if step["tool"] == assertion.tool
        and step.get("binding_verified") is True
        and step["status"] == "done"
    ]
    if assertion.kind == "tool_succeeded":
        return bool(steps) and all(
            any(
                run.get("invocation_id") == step.get("producer_invocation_id")
                and run["status"] == "succeeded"
                for run in step["runs"]
            )
            for step in steps
        )
    if len(steps) != 1:
        return False
    if assertion.kind == "dataset_rows":
        datasets = steps[0].get("result_datasets", [])
        return bool(datasets) and all(
            item["file_verified"] and item["row_count"] == assertion.value
            for item in datasets
        )
    if assertion.kind == "artifact_exists":
        return any(
            item["size_bytes"] > 0 and [item["field"]] == assertion.path
            for item in steps[0].get("output_files", [])
        )
    try:
        actual = _at(private["outputs"][steps[0]["id"]], assertion.path)
    except (KeyError, IndexError, TypeError):
        return False
    if assertion.kind == "output_equals":
        return type(actual) is type(assertion.value) and actual == assertion.value
    if assertion.kind == "output_length":
        return isinstance(actual, (dict, list)) and len(actual) == assertion.value
    if assertion.kind == "output_close":
        return (
            type(actual) in {int, float}
            and type(assertion.value) in {int, float}
            and math.isfinite(actual)
            and abs(actual - assertion.value) <= assertion.tolerance
        )
    return False


def usage_summary(
    events: list[dict], *, model_name: str, price_bytes: bytes | None
) -> dict:
    started = {
        event["attempt_id"]: event for event in events if event["event"] == "started"
    }
    finished = {
        event["attempt_id"]: event for event in events if event["event"] == "finished"
    }
    attempts = list(started)

    def known(field):
        return sum(
            event[field]
            for key, event in finished.items()
            if key in started and type(event.get(field)) is int
        )

    complete = (
        bool(attempts)
        and set(started) == set(finished)
        and all(
            type(finished[key].get(field)) is int
            for key in attempts
            for field in ("prompt_tokens", "completion_tokens")
        )
    )
    prompt = known("prompt_tokens")
    completion = known("completion_tokens")
    result = {
        "transport_attempts": len(attempts),
        "logical_calls": len({event["logical_call_id"] for event in started.values()}),
        "retry_attempts": sum(event["attempt"] > 1 for event in started.values()),
        "trace_complete": bool(attempts)
        and all(event.get("trace_complete") is True for event in started.values()),
        "incomplete_attempts": len(set(started) - set(finished)),
        "usage_status": "known" if complete else "unknown",
        "prompt_tokens": prompt if complete else None,
        "completion_tokens": completion if complete else None,
        "known_prompt_tokens_subtotal": prompt,
        "known_completion_tokens_subtotal": completion,
        "cost": None,
        "cost_status": "unknown",
        "cost_budget_status": "not_enforced",
        "token_budget_status": "not_enforced",
    }
    if price_bytes is not None:
        price = PriceBook.model_validate_json(price_bytes)
        result["price_book_sha256"] = digest(price_bytes)
        result["price_source"] = price.source
        result["price_effective_date"] = price.effective_date
        result["currency"] = price.currency
        if price.model_name == model_name and complete:
            result["cost"] = (
                prompt * price.prompt_per_million
                + completion * price.completion_per_million
            ) / 1_000_000
            result["cost_status"] = "estimate_from_usage_and_supplied_uncached_rates"
            result["cost_assumptions"] = [
                "all prompt tokens charged at supplied uncached rate",
                "completion usage includes reasoning only if the provider includes it",
            ]
    return result


def score_case(
    case, record, private, expected_bytes, *, model_source, price_bytes=None
):
    expected = ExpectedSuite.model_validate_json(expected_bytes).cases[case.id]
    statuses = [plan["status"] for plan in record.get("execution", {}).get("plans", [])]
    if expected.result == "done":
        terminal_ok = bool(statuses) and all(status == "done" for status in statuses)
    elif expected.result in {"failed", "cancelled"}:
        terminal_ok = expected.result in statuses
    else:
        # Clarification/refusal requires an explicit structured metadata/status
        # assertion. Absence of a plan alone never establishes a correct refusal.
        terminal_ok = not any(
            status in {"running", "confirmed", "done"} for status in statuses
        )
        terminal_ok &= any(
            a.kind in {"message_metadata", "http_status"} for a in expected.assertions
        )
    checks = [_assertion(a, record, private) for a in expected.assertions]
    job = record.get("execution", {}).get("latest_job") or {}
    job_settled = job.get("status") not in {"queued", "running"}
    passed = (
        record["runtime_status"] == "completed"
        and terminal_ok
        and all(checks)
        and job_settled
    )
    events = record["llm_events"]
    names = {event["model_name"] for event in events if event["event"] == "started"}
    usage = usage_summary(
        events,
        model_name=next(iter(names)) if len(names) == 1 else "",
        price_bytes=price_bytes,
    )
    observed_tools = any(
        step.get("binding_verified") is True
        for step in record.get("execution", {}).get("steps", [])
    )
    return {
        "passed": passed,
        "expected_result": expected.result,
        "terminal_ok": bool(terminal_ok),
        "assertions": [
            {"index": i, "kind": assertion.kind, "passed": checks[i]}
            for i, assertion in enumerate(expected.assertions)
        ],
        "usage": usage,
        "a_evidence_eligible": passed
        and model_source == "real_model"
        and usage["transport_attempts"] > 0
        and usage["trace_complete"]
        and observed_tools,
        "acceptance_claim": "not_established",
        "evidence_scope": "runtime_regression"
        if model_source == "fixture_model"
        else "real_model_runtime_candidate",
    }


def _rate(records):
    n = len(records)
    passed = sum(record["score"]["passed"] for record in records)
    if not n:
        return {"denominator": 0, "passed": 0, "pass_rate": None, "wilson_95": None}
    p, z = passed / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    delta = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / (1 + z * z / n)
    return {
        "denominator": n,
        "passed": passed,
        "pass_rate": p,
        "wilson_95": [max(0.0, center - delta), min(1.0, center + delta)],
    }


def summarize(records):
    groups = {}
    for field in ("family", "scenario", "case_set"):
        groups[field] = {
            key: _rate([record for record in records if record[field] == key])
            for key in sorted({record[field] for record in records})
        }
    return {
        "summary": _rate(records),
        "groups": groups,
        "all_passed": all(record["score"]["passed"] for record in records),
        "a_evidence_case_count": sum(
            record["score"].get("a_evidence_eligible", False) for record in records
        ),
    }


def compare_baseline(baseline_bytes: bytes, report: dict) -> dict:
    """Read-only comparison; a new corpus/answer key cannot erase old failures."""
    import json

    problems = []
    try:
        previous = json.loads(baseline_bytes)
        for field in (
            "schema_version",
            "execution_mode",
            "model_source",
            "cases_sha256",
            "expected_sha256",
            "case_ids",
            "denominator",
        ):
            if previous.get(field) != report.get(field):
                problems.append(f"incomparable_{field}")
        old_cases = {item["case_id"]: item for item in previous["cases"]}
        for current in report["cases"]:
            old = old_cases.get(current["case_id"], {})
            if old.get("case_sha256") != current.get("case_sha256"):
                problems.append(f"incomparable_case_contract:{current['case_id']}")
            if (
                old.get("score", {}).get("passed") is True
                and current["score"]["passed"] is not True
            ):
                problems.append(f"case_regressed:{current['case_id']}")
    except (ValueError, KeyError, TypeError, AttributeError):
        problems.append("invalid_baseline_report")
    return {"regression_ok": not problems, "regression_problems": problems}
