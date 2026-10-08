"""Feature reports travel through actual native execution and protected download."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import digest, RuntimeAction
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_scoring import Assertion, _assertion
from test_runtime_agent_benchmark import fixture_model


@pytest.fixture(scope="module")
def feature_download(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("feature-original-custody"))


def _build(root):
    paths = write_synthetic_suite(root / "suite")
    suite = json.loads(paths["cases"].read_text())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_feature")
    case["actions"].append({"kind": "download_feature_report", "content": "下载特征分析报告。"})
    suite["cases"] = [case]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    expected = json.loads(paths["expected"].read_text())
    expected["cases"][case["id"]]["assertions"].append({
        "kind": "feature_report_download", "tool": "feature.generate_feature_report"})
    paths["expected"].write_text(json.dumps(expected))
    with fixture_model() as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    archive = root / "custody" / report["run_id"] / case["id"]
    return archive, report, paths


def test_native_feature_download_matches_registered_original_file(feature_download):
    archive, report, _ = feature_download
    observed = report["cases"][0]
    event = next(e for e in observed["http_events"] if e["stage"] == "download_feature_report")
    with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM task_artifacts WHERE id=?", (event["artifact_id"],)).fetchone()
        assert row["kind"] == "feature_report_xlsx" and row["task_id"] == observed["execution"]["task_id"]
        original = Path(json.loads((archive / "manifest.json").read_text())["original_workspace"])
        path = archive / "workspace" / Path(row["path"]).relative_to(original)
        assert event["sha256"] == row["content_hash"] == digest(path.read_bytes())
        assert event["size_bytes"] == path.stat().st_size
    assert observed["isolated_workspace_removed"] and not original.exists()
    assert report["a_evidence_case_count"] == 0 and report["acceptance_claim"] == "not_established"


@pytest.mark.parametrize("field,value", [("artifact_id", "foreign"), ("step_id", "foreign"),
                                         ("sha256", "0" * 64), ("status_code", 403), ("size_bytes", 0)])
def test_download_score_requires_exact_native_identity_and_bytes(field, value):
    output = {"artifact_id": "artifact", "artifact_content_hash": "a" * 64}
    record = {"execution": {"steps": [{"id": "step", "tool": "feature.generate_feature_report", "status": "done", "binding_verified": True,
        "output_files": [{"field": "report_path", "sha256": "a" * 64, "size_bytes": 100, "suffix": ".xlsx"}]}]},
        "http_events": [{"stage": "download_feature_report", "status_code": 200, "sha256": "a" * 64,
                         "artifact_id": "artifact", "step_id": "step", "size_bytes": 100}]}
    assertion = Assertion(kind="feature_report_download", tool="feature.generate_feature_report")
    private = {"outputs": {"step": output}}
    assert _assertion(assertion, record, private)
    record["http_events"][0][field] = value
    assert not _assertion(assertion, record, private)


def test_download_action_cannot_inject_tool():
    with pytest.raises(ValueError):
        RuntimeAction(kind="download_feature_report", content="下载", tool="other.tool")
