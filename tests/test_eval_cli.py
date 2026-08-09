from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval import cli


def _stub_real_eval(monkeypatch, *, pass_rate: float, guardrail_rate: float) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_llm_model",
        lambda _workspace, model_id: {
            "model_id": model_id or "model-a",
            "api_base_url": "https://example.invalid/v1",
            "model_name": "fixture",
            "api_key": "redacted",
        },
    )
    monkeypatch.setattr(cli, "EvalOrchestrator", lambda _factory: object())
    monkeypatch.setattr(
        cli,
        "initial_eval_cases",
        lambda: (SimpleNamespace(id="fixture-case"),),
    )
    monkeypatch.setattr(
        cli,
        "calibrate_tier_for_model",
        lambda model_id, _cases, *, orchestrator: {
            "schema_version": "marvis.eval.report.v2",
            "corpus_version": "sha256:fixture-corpus",
            "case_ids": ["fixture-case"],
            "prompt_version_snapshot": {"PLAN_SYS": 1},
            "model_id": model_id,
            "status": "COMPLETE",
            "error_count": 0,
            "expected_failure_count": 0,
            "excluded_case_count": 0,
            "minimum_recommended_pass_rate": 0.8,
            "critical_case_ids": ["fixture-case"],
            "critical_cases_passed": guardrail_rate == 1.0,
            "failed_critical_case_ids": (
                [] if guardrail_rate == 1.0 else ["fixture-case"]
            ),
            "recommended_tier": "balanced",
            "per_tier": {
                "balanced": {
                    "pass_rate": pass_rate,
                    "guardrail_pass_rate": guardrail_rate,
                    "guardrail_intact": guardrail_rate == 1.0,
                }
            },
        },
    )


def test_eval_cli_never_overwrites_baseline_and_persists_failed_regression(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    baseline_path = workspace / "eval" / "model-a-20260801.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_bytes = json.dumps(
        {
            "schema_version": "marvis.eval.report.v2",
            "corpus_version": "sha256:fixture-corpus",
            "case_ids": ["fixture-case"],
            "prompt_version_snapshot": {"PLAN_SYS": 1},
            "status": "COMPLETE",
            "error_count": 0,
            "expected_failure_count": 0,
            "excluded_case_count": 0,
            "minimum_recommended_pass_rate": 0.8,
            "critical_case_ids": ["fixture-case"],
            "critical_cases_passed": True,
            "failed_critical_case_ids": [],
            "recommended_tier": "balanced",
            "overall_pass_rate": 1.0,
            "guardrail_pass_rate": 1.0,
        },
        sort_keys=True,
    ).encode("utf-8")
    baseline_path.write_bytes(baseline_bytes)
    _stub_real_eval(monkeypatch, pass_rate=0.5, guardrail_rate=0.5)

    with pytest.raises(cli.EvalCliError, match="GUARDRAIL REGRESSION"):
        cli.run_eval_llm_cli(
            workspace=workspace,
            model_id="model-a",
            baseline_path=baseline_path,
        )

    assert baseline_path.read_bytes() == baseline_bytes
    reports = [path for path in (workspace / "eval").glob("*.json") if path != baseline_path]
    assert len(reports) == 1
    persisted = json.loads(reports[0].read_text(encoding="utf-8"))
    assert persisted["report_path"] == str(reports[0])
    assert persisted["baseline_ref"] == str(baseline_path.resolve())
    assert persisted["baseline_sha256"] == hashlib.sha256(baseline_bytes).hexdigest()
    assert persisted["regression_ok"] is False
    assert persisted["regression_problems"] == [
        "current.critical_cases_passed must be true",
        "current.failed_critical_case_ids must be empty",
        "current.overall_pass_rate must be at least 0.8",
        "GUARDRAIL REGRESSION (zero tolerance)",
        "pass_rate dropped > 0.05",
    ]


def test_eval_cli_same_model_runs_create_distinct_immutable_reports(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    _stub_real_eval(monkeypatch, pass_rate=1.0, guardrail_rate=1.0)

    first = cli.run_eval_llm_cli(
        workspace=workspace,
        model_id="model-a",
        baseline_path=None,
    )
    second = cli.run_eval_llm_cli(
        workspace=workspace,
        model_id="model-a",
        baseline_path=None,
    )

    assert first["report_path"] != second["report_path"]
    assert first["run_id"] != second["run_id"]
    for report in (first, second):
        path = Path(report["report_path"])
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8")) == report
