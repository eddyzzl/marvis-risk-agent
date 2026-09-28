"""Real HTTP/Agent/PlanDriver/Executor/ToolRunner journeys, synthetic data only.

Only the model is a local protocol fixture. Application routes, repositories,
plugins, subprocess tools, dataset uploads and result bindings are production.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import subprocess
import sys

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_contracts import ModelConnection, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite


@contextmanager
def fixture_model(*, missing_usage=False, fail_first=False, delay=0, sse=False,
                  answer_factory=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if delay:
                time.sleep(delay)
            if fail_first and len(requests) == 1:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":{"code":"unavailable"}}')
                return
            answer = {
                "action": "confirm",
                "reason": "合成数据检查完成，按既定业务要求继续。",
                "passed": True,
                "reasons": [],
            }
            user = payload["messages"][-1]["content"]
            try:
                request = json.loads(user)
            except ValueError:
                request = {}
            if "instruction" in request and "allowed_directions" not in request:
                answer.update(
                    intent="current_workflow",
                    evidence_quote=request["instruction"],
                    confidence="high",
                    is_question=False,
                    is_conditional=False,
                    requests_change=False,
                    withholds_action=False,
                )
            if answer_factory is not None:
                answer = answer_factory(request, answer, payload)
            content = json.dumps(answer, ensure_ascii=False)
            body = {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
            }
            if not missing_usage:
                body["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
            encoded = json.dumps(body).encode()
            if sse:
                chunks = [
                    {"choices": [{"delta": {"content": content[:10]}}]},
                    {
                        "choices": [
                            {
                                "delta": {"content": content[10:]},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                    {"choices": [], "usage": body.get("usage", {})},
                ]
                encoded = (
                    b"".join(
                        b"data: " + json.dumps(chunk).encode() + b"\n\n"
                        for chunk in chunks
                    )
                    + b"data: [DONE]\n\n"
                )
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream" if sse else "application/json"
            )
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = ModelConnection(
        model_id="fixture",
        model_name="fixture",
        api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
    )
    try:
        yield model, requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def materials(root: Path):
    root.mkdir()
    phones = [f"138{i:08d}" for i in range(80)]
    frames = {
        "sample.parquet": pd.DataFrame(
            {"mobile": phones, "y": [i % 2 for i in range(80)]}
        ),
        "features.parquet": pd.DataFrame(
            {
                "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones],
                "balance": list(range(80)),
            }
        ),
        "analysis.parquet": pd.DataFrame(
            {
                "y": [i % 2 for i in range(80)],
                "balance": [i + (i % 2) * 20 for i in range(80)],
                "age": [20 + i % 30 for i in range(80)],
            }
        ),
    }
    for name, frame in frames.items():
        frame.to_parquet(root / name)
    return {
        name: {
            "path": name,
            "sha256": digest((root / name).read_bytes()),
            "role": "sample" if name != "features.parquet" else "feature",
        }
        for name in frames
    }


def write_suite(tmp_path, *, cases, expected):
    public = tmp_path / "cases.json"
    private = tmp_path / "private-expected.json"
    public.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    private.write_text(json.dumps({"schema_version": 1, "cases": expected}))
    return public, private


def base_case(case_id, task_type, material_list):
    return {
        "id": case_id,
        "revision": "1",
        "family": task_type,
        "task": {"task_type": task_type, "target_col": "y"},
        "materials": material_list,
        "business_constraints_source": "synthetic regression fixture v1",
    }


def test_join_and_feature_cross_real_http_agent_and_tools(tmp_path):
    data = tmp_path / "data"
    mat = materials(data)
    join = base_case(
        "join", "data_join", [mat["sample.parquet"], mat["features.parquet"]]
    )
    join["actions"] = [
        {
            "kind": "approve_step",
            "tool": "data_ops.execute_join",
            "content": "我已检查合成数据，批准最终拼接。",
        },
        {"kind": "replay_approval"},
    ]
    feature = base_case("feature", "feature_analysis", [mat["analysis.parquet"]])
    feature["task"]["feature_columns"] = ["balance", "age"]
    feature["actions"] = [
        {
            "kind": "approve_step",
            "tool": "feature.analyze_feature_bins",
            "content": "保留预设分箱选择，生成合成数据的特征分析报告。",
        }
    ]
    expected = {
        "join": {
            "result": "done",
            "assertions": [
                {"kind": "tool_succeeded", "tool": "data_ops.execute_join"},
                {"kind": "dataset_rows", "tool": "data_ops.execute_join", "value": 80},
                {"kind": "http_status", "stage": "stale_approval_probe", "value": 409},
            ],
        },
        "feature": {
            "result": "done",
            "assertions": [
                {"kind": "tool_succeeded", "tool": "feature.compute_feature_metrics"},
                {
                    "kind": "output_length",
                    "tool": "feature.compute_feature_metrics",
                    "path": ["metrics"],
                    "value": 2,
                },
                {"kind": "tool_succeeded", "tool": "feature.generate_feature_report"},
                {
                    "kind": "artifact_exists",
                    "tool": "feature.generate_feature_report",
                    "path": ["report_path"],
                },
            ],
        },
    }
    cases_path, expected_path = write_suite(
        tmp_path, cases=[join, feature], expected=expected
    )
    with fixture_model() as (model, calls):
        report = run_runtime_suite(
            cases_path=cases_path,
            expected_path=expected_path,
            dataset_root=data,
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert report["denominator"] == 2
    assert report["a_evidence_case_count"] == 0
    assert len(calls) == sum(
        case["score"]["usage"]["transport_attempts"] for case in report["cases"]
    )
    assert all(case["isolated_workspace_removed"] for case in report["cases"])
    assert all(case["score"]["usage"]["logical_calls"] > 0 for case in report["cases"])
    assert {c["family"] for c in report["cases"]} == {"data_join", "feature_analysis"}


def test_all_transport_retries_measured_and_missing_usage_unknown(tmp_path):
    from marvis.llm_client import OpenAICompatibleLLMClient
    from marvis.orchestrator.eval.runtime_runner import _model_gateway
    from marvis.orchestrator.eval.runtime_scoring import usage_summary

    path = tmp_path / "attempts.jsonl"
    with fixture_model(missing_usage=True, fail_first=True) as (model, calls):
        with _model_gateway(
            model.profile("fixture_model"),
            {},
            path,
            max_attempts=4,
            deadline=time.monotonic() + 20,
        ) as profile:
            client = OpenAICompatibleLLMClient(profile)
            client.complete(system_prompt="test", user_prompt="synthetic", stream=False)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    summary = usage_summary(events, model_name="fixture", price_bytes=None)
    assert len(calls) == summary["transport_attempts"] == 2
    assert summary["logical_calls"] == summary["retry_attempts"] == 1
    assert [e["http_status"] for e in events if e["event"] == "finished"] == [503, 200]
    assert summary["usage_status"] == "unknown"
    assert (
        summary["prompt_tokens"]
        is summary["completion_tokens"]
        is summary["cost"]
        is None
    )
    assert "synthetic" not in path.read_text()


def test_failures_budgets_and_private_expected_stay_in_denominator(tmp_path):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite

    paths = write_synthetic_suite(tmp_path / "synthetic")
    public = json.loads(paths["cases"].read_text())
    feature = public["cases"][1]
    feature["id"] = "sentinel"
    bad_material = {
        **feature,
        "id": "bad_material",
        "materials": [{**feature["materials"][0], "sha256": "0" * 64}],
    }
    budget = {**feature, "id": "budget", "budget": {"max_llm_attempts": 0}}
    frame = pd.read_parquet(paths["dataset_root"] / "analysis.parquet")
    frame.loc[0, "y"] = float("nan")
    frame.to_parquet(paths["dataset_root"] / "missing_labels.parquet", index=False)
    bad_labels = {
        **feature,
        "id": "tool_failure",
        "actions": [],
        "materials": [
            {
                "path": "missing_labels.parquet",
                "role": "sample",
                "sha256": digest(
                    (paths["dataset_root"] / "missing_labels.parquet").read_bytes()
                ),
            }
        ],
    }
    sentinel = "EXPECTED_ONLY_SENTINEL_547142_DO_NOT_EXPOSE"
    expected = {
        case_id: {
            "result": "done",
            "assertions": [
                {
                    "kind": "output_equals",
                    "tool": "feature.generate_feature_report",
                    "path": ["never_present"],
                    "value": sentinel,
                }
            ],
        }
        for case_id in ("sentinel", "bad_material", "budget", "tool_failure")
    }
    paths["cases"].write_text(
        json.dumps(
            {"schema_version": 1, "cases": [feature, bad_material, budget, bad_labels]}
        )
    )
    paths["expected"].write_text(json.dumps({"schema_version": 1, "cases": expected}))
    with fixture_model(missing_usage=True) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["summary"]["denominator"] == 4
    assert report["summary"]["passed"] == 0
    records = {record["case_id"]: record for record in report["cases"]}
    assert records["sentinel"]["runtime_status"] == "completed"
    assert records["sentinel"]["score"]["terminal_ok"]
    assert records["bad_material"]["error_code"] == "material_identity_mismatch"
    assert records["budget"]["runtime_status"] == "budget_exceeded"
    assert records["budget"]["score"]["usage"]["transport_attempts"] == 0
    assert any(
        run["status"] == "failed"
        for step in records["tool_failure"]["execution"]["steps"]
        for run in step["runs"]
    )
    assert all(record["score"]["usage"]["cost"] is None for record in records.values())
    assert sentinel not in json.dumps(calls)
    assert sentinel not in json.dumps(report)
    assert all(record["isolated_workspace_removed"] for record in records.values())
    # The execution artifact was frozen before scoring, so changing the private
    # answer can only produce another score/run; it cannot erase these failures.
    run_dir = Path(report["report_path"]).parent
    original = (run_dir / "sentinel" / "execution.json").read_bytes()
    assert digest(original) == records["sentinel"]["execution_sha256"]
    assert "score" not in json.loads(original)
    assert (run_dir / "manifest.json").exists()


def test_wall_budget_stops_actual_active_agent_request(tmp_path):
    data = tmp_path / "data"
    mat = materials(data)
    case = base_case("wall", "feature_analysis", [mat["analysis.parquet"]])
    case["budget"] = {"wall_seconds": 8}
    paths = write_suite(
        tmp_path,
        cases=[case],
        expected={
            "wall": {
                "result": "done",
                "assertions": [
                    {
                        "kind": "tool_succeeded",
                        "tool": "feature.compute_feature_metrics",
                    }
                ],
            }
        },
    )
    started = time.monotonic()
    with fixture_model(delay=15) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths[0],
            expected_path=paths[1],
            dataset_root=data,
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    record = report["cases"][0]
    assert calls, "test must exercise active LLM cancellation, not just startup timeout"
    assert record["runtime_status"] == "budget_exceeded"
    assert any(
        event["stage"] == "budget_or_failure_stop" for event in record["http_events"]
    )
    assert record["isolated_workspace_removed"]
    assert not record["score"]["passed"]
    assert time.monotonic() - started < 23


def test_unknown_tokens_and_no_calls_never_establish_a():
    from marvis.orchestrator.eval.runtime_contracts import RuntimeCase
    from marvis.orchestrator.eval.runtime_scoring import score_case, usage_summary

    case = RuntimeCase.model_validate(base_case("no_call", "feature_analysis", []))
    record = {
        "runtime_status": "completed",
        "llm_events": [],
        "http_events": [],
        "execution": {
            "plans": [{"status": "done"}],
            "steps": [
                {
                    "id": "step",
                    "tool": "feature.compute_feature_metrics",
                    "status": "done",
                    "binding_verified": True,
                    "producer_invocation_id": "run",
                    "runs": [{"status": "succeeded", "invocation_id": "run"}],
                }
            ],
        },
    }
    expected = json.dumps(
        {
            "schema_version": 1,
            "cases": {
                "no_call": {
                    "result": "done",
                    "assertions": [
                        {
                            "kind": "tool_succeeded",
                            "tool": "feature.compute_feature_metrics",
                        }
                    ],
                }
            },
        }
    ).encode()
    result = score_case(
        case,
        record,
        {"outputs": {}, "messages": []},
        expected,
        model_source="real_model",
    )
    assert result["passed"] and not result["a_evidence_eligible"]
    assert result["acceptance_claim"] == "not_established"
    unknown = usage_summary(
        [{"event": "started", "attempt_id": "a", "logical_call_id": "l", "attempt": 1}],
        model_name="fixture",
        price_bytes=None,
    )
    assert unknown["incomplete_attempts"] == 1
    assert unknown["prompt_tokens"] is None


def test_contract_rejects_expected_or_paths_in_public_runtime_case(tmp_path):
    from pydantic import ValidationError
    from marvis.orchestrator.eval.runtime_contracts import RuntimeCase

    base = base_case("contract", "feature_analysis", [])
    with pytest.raises(ValidationError):
        RuntimeCase.model_validate({**base, "expected": {"status": "done"}})
    with pytest.raises(ValidationError):
        RuntimeCase.model_validate(
            {**base, "task": {**base["task"], "source_dir": "/user/workspace"}}
        )
    with pytest.raises(ValidationError):
        ModelConnection.model_validate(
            {
                "model_id": "x",
                "model_name": "x",
                "api_base_url": "http://127.0.0.1",
                "api_key": "SECRET_SENTINEL",
            }
        )
    with pytest.raises(ValueError):
        ModelConnection(
            model_id="fixture", model_name="fixture", api_base_url="https://example.org"
        ).profile("fixture_model")


def test_gateway_shares_budget_with_real_report_tool_factory_in_child(tmp_path):
    from marvis.llm_client import OpenAICompatibleLLMClient
    from marvis.orchestrator.eval.runtime_runner import _model_gateway

    attempts = tmp_path / "attempts.jsonl"
    workspace = tmp_path / "child-workspace"
    (workspace / "settings").mkdir(parents=True)
    with fixture_model(sse=True) as (model, calls):
        actual_profile = {
            **model.profile("fixture_model"),
            "api_key": "PROVIDER_SECRET_PARENT_ONLY_SENTINEL",
        }
        with _model_gateway(
            actual_profile, {}, attempts, max_attempts=2, deadline=time.monotonic() + 25
        ) as child_profile:
            settings = {"default_model_id": model.model_id, "models": [child_profile]}
            settings_path = workspace / "settings" / "llm.json"
            settings_path.write_text(json.dumps(settings))
            assert actual_profile["api_key"] not in settings_path.read_text()
            assert actual_profile["api_base_url"] not in settings_path.read_text()
            deltas = []
            OpenAICompatibleLLMClient(child_profile).complete(
                system_prompt="test", user_prompt="stream", on_delta=deltas.append
            )
            assert len(deltas) == 2  # proxy preserves actual SSE streaming chunks
            code = (
                "from pathlib import Path\n"
                "from marvis.packs.modeling.report_tools import _draft_report_narratives, _report_llm_factory\n"
                f"factory = _report_llm_factory(Path({str(workspace)!r}), None)\n"
                "_draft_report_narratives({}, llm_factory=factory)\n"
                "_draft_report_narratives({}, llm_factory=factory)\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                timeout=20,
            )
            assert completed.returncode == 0, completed.stderr.decode()
            assert len(calls) == 2, (
                "third transport must be stopped before reaching provider"
            )
        records = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert len([item for item in records if item["event"] == "started"]) == 2
    assert len([item for item in records if item["event"] == "budget_blocked"]) == 1
    assert all(item["trace_complete"] for item in records if item["event"] == "started")
    assert actual_profile["api_key"] not in attempts.read_text()
    assert child_profile["api_key"] not in attempts.read_text()
    assert all(
        item["prompt_tokens"] == 100 for item in records if item["event"] == "finished"
    )


def test_runtime_cli_executes_public_synthetic_suite_and_exports_only_metadata(
    tmp_path,
):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite

    paths = write_synthetic_suite(tmp_path / "public-suite")
    output = tmp_path / "runs"
    with fixture_model() as (model, calls):
        connection = tmp_path / "connection.json"
        connection.write_text(model.model_dump_json())
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marvis",
                "eval-agent",
                "--cases",
                str(paths["cases"]),
                "--expected",
                str(paths["expected"]),
                "--dataset-root",
                str(paths["dataset_root"]),
                "--output-dir",
                str(output),
                "--model-config",
                str(connection),
                "--model-source",
                "fixture_model",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=65,
        )
    assert result.returncode == 0, result.stderr + result.stdout
    assert calls
    report_path = next(output.rglob("report.json"))
    report = json.loads(report_path.read_text())
    assert report["summary"]["passed"] == 2
    assert report["regression_gate_passed"]
    assert report["a_evidence_case_count"] == 0
    assert report["acceptance_claim"] == "not_established"
    assert report["declared_max_total_llm_attempts"] == 40
    assert model.api_base_url not in report_path.read_text()
    assert "messages" not in report_path.read_text()


def test_saved_profile_resolved_explicitly_and_key_stays_only_in_parent(
    tmp_path, monkeypatch
):
    from marvis import __main__
    from marvis.llm_settings import save_llm_settings
    from marvis.orchestrator.eval import cli, runtime_runner

    workspace = tmp_path / "explicit-profile"
    save_llm_settings(
        workspace,
        {
            "models": [
                {
                    "model_id": "owned-model",
                    "model_name": "synthetic",
                    "api_base_url": "http://127.0.0.1:1/v1",
                    "api_key": "PARENT_ONLY_CONFIG_TEST_SENTINEL",
                    "enable_thinking": True,
                    "thinking_style": "openai_reasoning",
                    "reasoning_effort": "high",
                }
            ]
        },
    )
    before = (workspace / "settings" / "llm.json").read_bytes()
    captured = {}

    def inspect_arguments(**kwargs):
        captured.update(kwargs)
        return {"test_only": True}

    monkeypatch.setattr(runtime_runner, "run_runtime_suite", inspect_arguments)
    args = __main__._parse_args(
        [
            "eval-agent",
            "--cases",
            str(tmp_path / "cases.json"),
            "--expected",
            str(tmp_path / "expected.json"),
            "--dataset-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "runs"),
            "--profile-workspace",
            str(workspace),
            "--model-id",
            "owned-model",
        ]
    )
    cli.run_eval_agent_cli(args)
    assert captured["model"].enable_thinking
    assert captured["secret_env"] == {
        "MARVIS_RUNTIME_MODEL_KEY": "PARENT_ONLY_CONFIG_TEST_SENTINEL"
    }
    assert "PARENT_ONLY_CONFIG_TEST_SENTINEL" not in captured["model"].model_dump_json()
    assert (workspace / "settings" / "llm.json").read_bytes() == before


def test_missing_model_credential_is_recorded_as_case_failure(tmp_path, monkeypatch):
    data = tmp_path / "data"
    mat = materials(data)
    case = base_case("credential", "feature_analysis", [mat["analysis.parquet"]])
    paths = write_suite(
        tmp_path,
        cases=[case],
        expected={
            "credential": {
                "result": "done",
                "assertions": [
                    {
                        "kind": "tool_succeeded",
                        "tool": "feature.compute_feature_metrics",
                    }
                ],
            }
        },
    )
    monkeypatch.delenv("MARVIS_TEST_UNSET_CREDENTIAL", raising=False)
    model = ModelConnection(
        model_id="no-credential",
        model_name="synthetic",
        api_base_url="http://127.0.0.1:1/v1",
        api_key_env="MARVIS_TEST_UNSET_CREDENTIAL",
    )
    report = run_runtime_suite(
        cases_path=paths[0],
        expected_path=paths[1],
        dataset_root=data,
        output_dir=tmp_path / "runs",
        model=model,
        model_source="real_model",
    )
    assert report["denominator"] == 1
    assert not report["all_passed"]
    assert report["cases"][0]["error_code"] == "missing_explicit_model_credential"
    assert report["a_evidence_case_count"] == 0


def test_cost_source_and_comparable_baseline_are_explicit():
    from marvis.orchestrator.eval.runtime_scoring import compare_baseline, usage_summary

    events = [
        {
            "event": "started",
            "attempt_id": "a",
            "logical_call_id": "l",
            "attempt": 1,
            "trace_complete": True,
        },
        {
            "event": "finished",
            "attempt_id": "a",
            "prompt_tokens": 1000,
            "completion_tokens": 200,
        },
    ]
    prices = json.dumps(
        {
            "model_name": "synthetic",
            "provider": "test-only",
            "currency": "TEST",
            "source": "synthetic fixture rate table; not a provider price",
            "effective_date": "2026-09-21",
            "prompt_per_million": 2.0,
            "completion_per_million": 5.0,
        }
    ).encode()
    usage = usage_summary(events, model_name="synthetic", price_bytes=prices)
    assert usage["cost"] == pytest.approx(0.003)
    assert usage["price_book_sha256"] == digest(prices)
    assert usage["cost_budget_status"] == "not_enforced"
    assert (
        usage_summary(events, model_name="different", price_bytes=prices)["cost"]
        is None
    )
    base = {
        "schema_version": 1,
        "execution_mode": "real_http_agent_tools",
        "model_source": "fixture_model",
        "cases_sha256": "fixed",
        "expected_sha256": "fixed",
        "case_ids": ["c"],
        "denominator": 1,
        "cases": [{"case_id": "c", "score": {"passed": True}}],
    }
    frozen = json.dumps(base).encode()
    current = {**base, "cases": [{"case_id": "c", "score": {"passed": False}}]}
    assert compare_baseline(frozen, current)["regression_problems"] == [
        "case_regressed:c"
    ]
    assert json.dumps(base).encode() == frozen
    assert (
        "incomparable_expected_sha256"
        in compare_baseline(frozen, {**base, "expected_sha256": "changed"})[
            "regression_problems"
        ]
    )
    assert not compare_baseline(b"broken", base)["regression_ok"]


@pytest.mark.parametrize("mode", ["json_body", "sse_line", "headers"])
def test_gateway_wall_deadline_breaks_real_slow_trickle_and_seals_receipts(
    tmp_path, mode
):
    from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
    from marvis.orchestrator.eval.runtime_runner import _model_gateway

    arrived = threading.Event()
    stopped = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            arrived.set()
            try:
                if mode == "headers":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Unfinished: ")
                else:
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "text/event-stream"
                        if mode == "sse_line"
                        else "application/json",
                    )
                    self.send_header("Content-Length", "999999")
                    self.end_headers()
                    self.wfile.write(
                        b'data: {"x": "' if mode == "sse_line" else b'{"x": "'
                    )
                self.wfile.flush()
                # Continuous data defeats a per-socket inactivity timeout. The
                # parent deadline must terminate this before the stream ends.
                while not stopped.wait(0.015):
                    self.wfile.write(b"x")
                    self.wfile.flush()
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = ModelConnection(
        model_id="slow",
        model_name="slow",
        api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
        timeout_seconds=20,
        transport_max_retries=0,
    ).profile("fixture_model")
    path = tmp_path / "attempts.jsonl"
    try:
        started = time.monotonic()
        with _model_gateway(
            profile, {}, path, max_attempts=1, deadline=started + 0.6
        ) as child_profile:
            with pytest.raises(LLMClientError):
                OpenAICompatibleLLMClient(child_profile).complete(
                    system_prompt="test", user_prompt="test", stream=True
                )
        elapsed = time.monotonic() - started
        assert arrived.is_set()
        assert elapsed < 2.5, f"hard deadline did not stop trickle: {elapsed}"
        frozen = path.read_bytes()
        time.sleep(0.15)
        assert path.read_bytes() == frozen, "late handler rewrote the sealed receipt"
        events = [json.loads(line) for line in frozen.splitlines()]
        assert any(
            event["event"] == "measurement_closed"
            and event["reason"] == "wall_deadline"
            for event in events
        )
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_code_copy_rejects_expected_and_public_case_aliases(tmp_path):
    from marvis.orchestrator.eval.runtime_runner import (
        _package_identity,
        _validate_package_isolation,
    )

    package = tmp_path / "marvis"
    package.mkdir()
    expected = package / "private-expected.json"
    expected.write_text('{"private":"EXPECTED_IN_PACKAGE_SENTINEL"}')
    cases = tmp_path / "cases.json"
    cases.write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        _validate_package_isolation(package, (cases, expected))
    external = tmp_path / "private-expected.json"
    expected.rename(external)
    alias = package / "answer-link.json"
    alias.symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"):
        _validate_package_isolation(package, (cases, external))
    with pytest.raises(ValueError, match="symlinks"):
        _package_identity(tmp_path)
    alias.unlink()
    alias.hardlink_to(external)
    with pytest.raises(ValueError, match="hardlink"):
        _validate_package_isolation(package, (cases, external))
    alias.unlink()
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        _validate_package_isolation(package, (cases, external))


def test_gateway_caps_explicit_output_requests_and_rejects_model_switch(tmp_path):
    from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
    from marvis.orchestrator.eval.runtime_runner import _model_gateway

    path = tmp_path / "attempts.jsonl"
    with fixture_model() as (model, calls):
        with _model_gateway(
            model.profile("fixture_model"),
            {},
            path,
            max_attempts=3,
            deadline=time.monotonic() + 10,
            max_output_tokens=31,
        ) as profile:
            OpenAICompatibleLLMClient(profile).complete(
                system_prompt="test", user_prompt="test", stream=False, max_tokens=999
            )
            assert calls[0]["max_tokens"] == 31
            with pytest.raises(LLMClientError):
                OpenAICompatibleLLMClient(
                    {**profile, "model_name": "not-selected"}
                ).complete(system_prompt="test", user_prompt="test", stream=False)
            assert len(calls) == 1
    events = [json.loads(line) for line in path.read_text().splitlines()]
    attempt = next(item for item in events if item["event"] == "started")
    assert attempt["requested_max_output_tokens"] == 999
    assert attempt["forwarded_max_output_tokens"] == 31
    assert attempt["request_sha256"] != attempt["provider_request_sha256"]
    assert any(
        item["event"] == "transport_rejected" and item["reason"] == "model_not_found"
        for item in events
    )


@pytest.mark.parametrize("settle_after", [40.0, None])
def test_slow_workflow_polling_preserves_http_budget_and_wall_deadline(monkeypatch, settle_after):
    import httpx
    from types import SimpleNamespace
    from marvis.orchestrator.eval import runtime_runner

    clock = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    def serve(request):
        done = settle_after is not None and clock[0] >= settle_after
        if request.url.path.endswith("/plans"):
            return httpx.Response(200, json={"plans": [{
                "id": "p", "status": "done" if done else "running",
                "steps": [{"id": "s", "status": "done" if done else "running"}],
            }]})
        return httpx.Response(200, json={"job": {"status": "succeeded" if done else "running"}})

    monkeypatch.setattr(runtime_runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime_runner.time, "sleep", sleep)
    case = SimpleNamespace(budget=SimpleNamespace(max_http_requests=120))
    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(serve)) as client:
        journey = runtime_runner.Journey(client, case, 60.0)
        journey.task_id = "t"
        if settle_after is None:
            with pytest.raises(runtime_runner.RuntimeBudgetExceeded):
                journey.wait_idle()
            assert clock[0] == 60.0
        else:
            assert journey.wait_idle()[0]["status"] == "done"
            assert settle_after <= clock[0] <= settle_after + 2
        assert journey.count < 45
        assert waits and max(waits) <= 2.0
