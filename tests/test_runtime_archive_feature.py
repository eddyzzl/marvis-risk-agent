"""Feature report originals are recomputed independently after workspace removal."""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_feature import FrozenFeatureBinding, revalidate_feature_archive, _TOOLS
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from test_runtime_feature_download import feature_download as feature_download


@pytest.fixture(scope="module")
def feature_archive(feature_download):
    archive, report, _ = feature_download
    observed = report["cases"][0]
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


def _copy(original, tmp_path):
    archive, sha, binding = original
    destination = tmp_path / "archive"
    shutil.copytree(archive, destination)
    return destination, sha, binding


def _manifest(archive):
    from test_runtime_archive_validation import _rewrite_manifest
    return _rewrite_manifest(archive)


def _database(archive):
    from test_runtime_archive_labeling import _editable_database
    return _editable_database(archive)


def test_independent_feature_originals_do_not_call_producer_metrics_or_report(feature_archive, monkeypatch):
    import marvis.feature.metrics as metrics
    import marvis.feature.iv as iv
    import marvis.feature.binning as binning
    import marvis.output.feature_report as report
    import marvis.packs.feature.tools as tools

    def forbidden(*args, **kwargs):
        raise AssertionError("producer kernel is not an independent reference")
    for module, names in ((metrics, ("selected_feature_metrics", "feature_ks", "feature_auc")),
                          (iv, ("compute_woe_iv",)), (binning, ("equal_frequency_edges", "assign_bins")),
                          (report, ("render_feature_report", "_cell")), (tools, ("_feature_recommendation",))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    archive, sha, binding = feature_archive
    before = {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}
    result = _run(archive, sha, binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["values"]["feature_count"] == 2
    assert result["values"]["metrics"][0]["iv"] == 1.047541
    assert result["checks"]["workbook_cells"]["cells_compared"] == 60
    assert result["acceptance_claim"] == result["source_authentication"] == "not_established"
    assert before == {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field", ["run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"])
def test_feature_archive_cannot_replace_external_identity(feature_archive, tmp_path, field):
    archive, _, binding = _copy(feature_archive, tmp_path)
    path = archive / "bindings.json"
    value = json.loads(path.read_text())
    value[field] = {"source_sha256": "b" * 64} if field == "source" else "b" * 64
    path.write_text(json.dumps(value))
    assert _run(archive, _manifest(archive), binding)["checks"]["external_bindings"]["status"] == "mismatch"


@pytest.mark.parametrize("field,value", [("parent_output_refs", []), ("parent_output_bindings", []),
                                         ("source_dataset_refs", []), ("result_dataset_bindings", [{"dataset_id": "foreign"}])])
def test_feature_native_dependencies_and_dataset_refs_are_required(feature_archive, tmp_path, field, value):
    archive, _, binding = _copy(feature_archive, tmp_path)
    with closing(_database(archive)) as conn:
        step_id = binding.step_ids["bins"]
        row = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=?", (step_id,)).fetchone()
        evidence = json.loads(row["evidence_json"])
        evidence[field] = value
        conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=?", (json.dumps(evidence), step_id))
        conn.commit()
    assert _run(archive, _manifest(archive), binding)["checks"]["native_steps"]["status"] == "mismatch"


@pytest.mark.parametrize("metric", ["iv", "ks", "auc", "mean", "valid_count"])
def test_rehashed_native_output_cannot_replace_feature_arithmetic(feature_archive, tmp_path, metric):
    from test_runtime_archive_portfolio import _rewrite_step
    from marvis.orchestrator.evidence import payload_hash

    archive, _, binding = _copy(feature_archive, tmp_path)
    values = binding.model_dump()
    with closing(_database(archive)) as conn:
        output = json.loads(conn.execute("SELECT output_json FROM plan_step_output_versions WHERE step_id=?",
                                        (binding.step_ids["metrics"],)).fetchone()[0])
        output["metrics"][0][metric] += 1
        _rewrite_step(conn, binding, "metrics", output=output)
        values["output_sha256"]["metrics"] = digest(output)
        for key in ("bins", "report"):
            row = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids[key],)).fetchone()
            evidence = json.loads(row["evidence_json"])
            evidence["parent_output_bindings"][0]["output_hash"] = payload_hash(output)
            conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=?", (json.dumps(evidence), binding.step_ids[key]))
        conn.commit()
    result = _run(archive, _manifest(archive), FrozenFeatureBinding.model_validate(values))
    assert result["checks"]["native_steps"]["status"] == "verified", result["checks"]
    assert result["checks"]["independent_metrics"]["status"] == "mismatch"


def test_rehashed_workbook_cannot_replace_independent_metric_cell(feature_archive, tmp_path):
    from openpyxl import load_workbook
    from test_runtime_archive_portfolio import _rewrite_step
    from pathlib import Path

    archive, _, binding = _copy(feature_archive, tmp_path)
    values = binding.model_dump()
    with closing(_database(archive)) as conn:
        saved = conn.execute("SELECT output_json FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids["report"],)).fetchone()
        output = json.loads(saved[0])
        root = Path(json.loads((archive / "manifest.json").read_text())["original_workspace"])
        path = archive / "workspace" / Path(output["report_path"]).relative_to(root)
        book = load_workbook(path)
        book["特征指标"]["D2"] = 9.0
        book.save(path)
        book.close()
        sha = digest(path.read_bytes())
        output["artifact_content_hash"] = sha
        conn.execute("UPDATE task_artifacts SET content_hash=? WHERE id=?", (sha, output["artifact_id"]))
        _rewrite_step(conn, binding, "report", output=output)
        conn.commit()
    values["output_sha256"]["report"] = digest(output)
    values["download_sha256"] = sha
    saved_bindings = json.loads((archive / "bindings.json").read_text())
    for event in saved_bindings["execution_before_scoring"]["http_events"]:
        if event.get("stage") == "download_feature_report":
            event.update(sha256=sha, size_bytes=path.stat().st_size)
    (archive / "bindings.json").write_text(json.dumps(saved_bindings))
    result = _run(archive, _manifest(archive), FrozenFeatureBinding.model_validate(values))
    assert result["checks"]["artifact_record"]["status"] == "verified", result["checks"]
    assert result["checks"]["workbook_cells"]["status"] == "mismatch"
