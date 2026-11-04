"""Optional feature metrics (importance, VIF) and a chosen binning are recomputed independently."""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_feature import FrozenFeatureBinding, revalidate_feature_archive, _TOOLS
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _gate_route_request

_CHOICE = "请对 balance 和 age 做分箱明细分析，确认后执行。"


def _answers(request, answer, payload):
    if _gate_route_request(payload) and (request.get("gate_context") or {}).get("purpose") == "current_step":
        return {"action": "adjust", "params": {"features": ["balance", "age"]}, "constraint": "",
                "reason": "用户指定分箱字段", "confidence": "high", "explicit_authorization": False}
    if "proposed_params" in request:
        return {"verdict": "authorize", "evidence_quote": request["instruction"], "reason": "用户明确指定分箱字段",
                "confidence": "high", "is_question": False, "is_conditional": False, "requests_change": False,
                "withholds_authorization": False}
    return answer


@pytest.fixture(scope="module")
def optional_archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("feature-optional-custody")
    paths = write_synthetic_suite(root / "suite")
    suite = json.loads(paths["cases"].read_text())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_feature")
    case["task"]["metrics"] = ["iv", "ks", "auc", "coverage", "importance", "vif"]
    # The binning choice is a human message at the binning gate; its confirmation is native.
    case["actions"] = [{"kind": "message", "content": _CHOICE},
                       {"kind": "download_feature_report", "content": "下载特征分析报告。"}]
    suite["cases"] = [case]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    expected = json.loads(paths["expected"].read_text())
    expected["cases"] = {case["id"]: expected["cases"][case["id"]]}
    paths["expected"].write_text(json.dumps(expected))
    with fixture_model(answer_factory=_answers) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], json.dumps(report["cases"][0].get("error_code"))
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / case["id"]
    steps = {key: next(s for s in observed["execution"]["steps"] if s["tool"] == tool) for key, tool in _TOOLS.items()}
    with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        dataset_id, sha = conn.execute("SELECT id,content_hash FROM datasets WHERE role='sample'").fetchone()
    binding = FrozenFeatureBinding(case=RuntimeCase.model_validate_json((archive / "case.json").read_bytes()),
        run_id=report["run_id"], task_id=observed["execution"]["task_id"], plan_id=observed["execution"]["plans"][0]["id"],
        dataset_id=dataset_id, dataset_sha256=sha, step_ids={key: step["id"] for key, step in steps.items()},
        output_sha256={key: step["output_sha256"] for key, step in steps.items()},
        download_sha256=next(e["sha256"] for e in observed["http_events"] if e["stage"] == "download_feature_report"),
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"])
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


def _run(archive, sha, binding):
    return revalidate_feature_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)


def test_importance_vif_and_chosen_binning_are_adjudicated(optional_archive, monkeypatch):
    import marvis.feature.bin_analysis as bin_analysis
    import marvis.feature.correlation as correlation
    import marvis.feature.importance as importance

    def forbidden(*args, **kwargs):
        raise AssertionError("producer kernel is not an independent reference")

    for module, name in ((bin_analysis, "feature_bin_analysis"), (correlation, "vif"), (correlation, "correlation_report"),
                         (importance, "feature_importance")):
        monkeypatch.setattr(module, name, forbidden)
    result = _run(*optional_archive)
    assert result["supported_checks_verified"], result["checks"]
    checks = result["checks"]
    assert checks["independent_metrics"]["binned_features"] == 2
    assert checks["independent_metrics"]["library_dependent"] == ["importance"]
    assert checks["workbook_cells"]["sheets"] == 3


def test_rehashed_binning_cell_cannot_replace_independent_rows(optional_archive, tmp_path):
    from pathlib import Path
    from openpyxl import load_workbook
    from test_runtime_archive_labeling import _editable_database
    from test_runtime_archive_portfolio import _rewrite_step
    from test_runtime_archive_validation import _rewrite_manifest

    archive, _, binding = optional_archive
    copy = tmp_path / "archive"
    shutil.copytree(archive, copy)
    values = binding.model_dump()
    with closing(_editable_database(copy)) as conn:
        saved = conn.execute("SELECT output_json FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids["report"],)).fetchone()
        output = json.loads(saved[0])
        root = Path(json.loads((copy / "manifest.json").read_text())["original_workspace"])
        path = copy / "workspace" / Path(output["report_path"]).relative_to(root)
        book = load_workbook(path)
        book["分箱分析"]["J2"] = 0.99
        book.save(path)
        book.close()
        sha = digest(path.read_bytes())
        output["artifact_content_hash"] = sha
        conn.execute("UPDATE task_artifacts SET content_hash=? WHERE id=?", (sha, output["artifact_id"]))
        _rewrite_step(conn, binding, "report", output=output)
        conn.commit()
    values["output_sha256"]["report"] = digest(output)
    values["download_sha256"] = sha
    saved_bindings = json.loads((copy / "bindings.json").read_text())
    for event in saved_bindings["execution_before_scoring"]["http_events"]:
        if event.get("stage") == "download_feature_report":
            event.update(sha256=sha, size_bytes=path.stat().st_size)
    (copy / "bindings.json").write_text(json.dumps(saved_bindings))
    result = _run(copy, _rewrite_manifest(copy), FrozenFeatureBinding.model_validate(values))
    assert result["checks"]["artifact_record"]["status"] == "verified", result["checks"]
    assert result["checks"]["workbook_cells"]["status"] == "mismatch"
