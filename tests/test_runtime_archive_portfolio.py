"""Independent portfolio revalidation over actual native HTTP run originals."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_portfolio import (
    FrozenPortfolioBinding, revalidate_portfolio_archive, _TOOLS,
)
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


@pytest.fixture(scope="module")
def portfolio_archive(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("portfolio-original-custody"))


def _build(root, variant=None):
    paths = write_synthetic_suite(root / "suite", include_workflow_families=True)
    suite = json.loads(paths["cases"].read_text())
    definition = next(case for case in suite["cases"] if case["id"] == "synthetic_portfolio")
    if variant:
        import pandas as pd
        material = definition["materials"][0]
        source = paths["dataset_root"] / material["path"]
        frame = pd.read_parquet(source)
        if variant == "csv":
            source = source.with_suffix(".csv")
            frame.to_csv(source, index=False)
        elif variant == "xlsx":
            source = source.with_suffix(".xlsx")
            frame.to_excel(source, index=False, sheet_name="Portfolio")
        elif variant == "literal_labels":
            labels = dict(zip(frame["segment"].unique(), ["=1+1", "#N/A"]))
            frame["segment"] = frame["segment"].map(labels)
            frame.to_parquet(source, index=False)
        else:
            raise AssertionError(variant)
        material.update(path=source.name, sha256=digest(source.read_bytes()))
    definition["family"] = "risk_portfolio"
    definition["actions"].append({"kind": "download_portfolio_report", "content": "下载组合报告。"})
    suite["cases"] = [definition]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    expected = json.loads(paths["expected"].read_text())
    expected["cases"][definition["id"]]["assertions"].append({
        "kind": "portfolio_report_download", "tool": "analysis.portfolio_report",
    })
    paths["expected"].write_text(json.dumps(expected))
    case = RuntimeCase.model_validate(definition)
    with fixture_model(answer_factory=_business_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    observed = report["cases"][0]
    steps = {key: next(s for s in observed["execution"]["steps"] if s["tool"] == tool) for key, tool in _TOOLS.items()}
    archive = root / "custody" / report["run_id"] / case.id
    with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        source = json.loads(conn.execute("SELECT input_json FROM plan_step_runs WHERE step_id=? AND status='succeeded'",
                                        (steps["flow"]["id"],)).fetchone()[0])
    binding = FrozenPortfolioBinding(case=case, run_id=report["run_id"], task_id=observed["execution"]["task_id"],
        plan_id=observed["execution"]["plans"][0]["id"], dataset_id=source["dataset_id"],
        dataset_sha256=source["expected_content_hash"], state_action_index=1, states=["current", "M1", "charged_off"],
        step_ids={key: step["id"] for key, step in steps.items()},
        output_sha256={key: step["output_sha256"] for key, step in steps.items()},
        download_sha256=next(e["sha256"] for e in observed["http_events"] if e["stage"] == "download_portfolio_report"),
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"])
    assert observed["isolated_workspace_removed"]
    assert not Path(json.loads((archive / "manifest.json").read_text())["original_workspace"]).exists()
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


def _run(archive, sha, binding):
    return revalidate_portfolio_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)


@pytest.mark.parametrize("variant", ["csv", "xlsx", "literal_labels"])
def test_source_formats_and_literal_labels_survive_native_download_and_recomputation(tmp_path, variant):
    archive, sha, binding = _build(tmp_path, variant)
    result = _run(archive, sha, binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["checks"]["workbook_cells"]["sheets"] == 7
    assert result["checks"]["download_binding"]["status"] == "verified"


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


def test_actual_portfolio_originals_independently_recompute_all_metrics_and_workbook(portfolio_archive, monkeypatch):
    import marvis.packs.analysis.flow as flow
    import marvis.packs.analysis.loss as loss
    import marvis.packs.analysis.segment as segment
    import marvis.output.portfolio_report as report

    def forbidden(*args, **kwargs):
        raise AssertionError("producer kernel or report writer is not an independent reference")
    for module, names in ((flow, ("flow_rate", "bucket_migration")), (loss, ("expected_loss_estimate",)),
                          (segment, ("segment_profile",)), (report, ("render_portfolio_report", "_write_overview"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    archive, sha, binding = portfolio_archive
    before = {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}
    result = _run(archive, sha, binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["checks"]["workbook_cells"]["sheets"] == 7
    assert result["checks"]["workbook_cells"]["cells_compared"] > 100
    assert result["values"]["segment_count"] == 2
    assert result["archive_unchanged_during_revalidation"]
    assert result["acceptance_claim"] == result["source_authentication"] == "not_established"
    assert before == {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field", ["run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"])
def test_archive_cannot_replace_external_portfolio_identity(portfolio_archive, tmp_path, field):
    archive, _, binding = _copy(portfolio_archive, tmp_path)
    path = archive / "bindings.json"
    values = json.loads(path.read_text())
    values[field] = {"source_sha256": "b" * 64} if field == "source" else "b" * 64
    path.write_text(json.dumps(values))
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["external_bindings"]["status"] == "mismatch"


@pytest.mark.parametrize("field,value", [("parent_output_refs", []), ("parent_output_bindings", []),
                                         ("source_dataset_refs", []), ("result_dataset_bindings", [{"dataset_id": "foreign"}])])
def test_portfolio_native_dependencies_and_dataset_refs_are_required(portfolio_archive, tmp_path, field, value):
    archive, _, binding = _copy(portfolio_archive, tmp_path)
    with closing(_database(archive)) as conn:
        step_id = binding.step_ids["expected_loss"]
        row = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=?", (step_id,)).fetchone()
        evidence = json.loads(row["evidence_json"])
        evidence[field] = value
        conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=?", (json.dumps(evidence), step_id))
        conn.commit()
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["native_steps"]["status"] == "mismatch"


def test_modified_bytes_and_rewritten_manifest_do_not_replace_original_digest(portfolio_archive, tmp_path):
    archive, sha, binding = _copy(portfolio_archive, tmp_path)
    path = next((archive / "inputs").glob("*.parquet"))
    path.write_bytes(path.read_bytes() + b"tampered")
    assert _manifest(archive) != sha
    assert _run(archive, sha, binding)["checks"]["archive_integrity"]["status"] == "mismatch"


def _rewrite_step(conn, binding, key, output=None, run_input=None):
    """Adversarial test copy only: synchronize hashes to challenge domain math."""
    from marvis.orchestrator.evidence import payload_hash
    row = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids[key],)).fetchone()
    evidence = json.loads(row["evidence_json"])
    if output is not None:
        evidence["output_hash"] = payload_hash(output)
        conn.execute("UPDATE plan_step_output_versions SET output_json=? WHERE step_id=?", (json.dumps(output), binding.step_ids[key]))
        conn.execute("UPDATE plan_step_runs SET output_hash=? WHERE id=?", (payload_hash(output), evidence["step_run_id"]))
    if run_input is not None:
        evidence["input_hash"] = payload_hash(run_input)
        conn.execute("UPDATE plan_step_runs SET input_json=? WHERE id=?", (json.dumps(run_input), evidence["step_run_id"]))
    conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=?", (json.dumps(evidence), binding.step_ids[key]))


def test_forged_matrix_still_fails_after_rewriting_all_native_parent_and_output_hashes(portfolio_archive, tmp_path):
    from marvis.orchestrator.evidence import payload_hash
    archive, _, binding = _copy(portfolio_archive, tmp_path)
    with closing(_database(archive)) as conn:
        row = conn.execute("SELECT output_json FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids["migration"],)).fetchone()
        output = json.loads(row[0])
        output["avg_matrix"][0][0] += 0.125
        _rewrite_step(conn, binding, "migration", output)
        for key in ("expected_loss", "gate", "report"):
            row = conn.execute("SELECT evidence_json FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids[key],)).fetchone()
            evidence = json.loads(row[0])
            for parent in evidence["parent_output_bindings"]:
                if parent["step_id"] == binding.step_ids["migration"]:
                    parent["output_hash"] = payload_hash(output)
            conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=?", (json.dumps(evidence), binding.step_ids[key]))
        conn.commit()
    changed = binding.model_copy(update={"output_sha256": {**binding.output_sha256, "migration": digest(output)}})
    result = _run(archive, _manifest(archive), changed)
    assert result["checks"]["native_steps"]["status"] == "verified", result["checks"]
    assert result["checks"]["independent_metrics"]["status"] == "mismatch"


@pytest.mark.parametrize("formula", [False, True])
def test_forged_workbook_cells_fail_even_when_file_output_download_and_provenance_hashes_match(portfolio_archive, tmp_path, formula):
    from openpyxl import load_workbook
    archive, _, binding = _copy(portfolio_archive, tmp_path)
    with closing(_database(archive)) as conn:
        output = json.loads(conn.execute("SELECT output_json FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids["report"],)).fetchone()[0])
        original = Path(json.loads((archive / "manifest.json").read_text())["original_workspace"])
        path = archive / "workspace" / Path(output["report_path"]).relative_to(original)
        book = load_workbook(path)
        book["预期损失"]["C2"] = "=1+1" if formula else 999999
        book.save(path)
        book.close()
        sha = digest(path.read_bytes())
        output["artifact_content_hash"] = sha
        conn.execute("UPDATE task_artifacts SET content_hash=? WHERE id=?", (sha, output["artifact_id"]))
        _rewrite_step(conn, binding, "report", output)
        conn.commit()
    path = archive / "bindings.json"
    values = json.loads(path.read_text())
    for event in values["execution_before_scoring"]["http_events"]:
        if event["stage"] == "download_portfolio_report":
            event.update(sha256=sha, size_bytes=(archive / "workspace" / Path(output["report_path"]).relative_to(original)).stat().st_size)
    path.write_text(json.dumps(values))
    changed = binding.model_copy(update={"download_sha256": sha, "output_sha256": {**binding.output_sha256, "report": digest(output)}})
    result = _run(archive, _manifest(archive), changed)
    assert result["checks"]["artifact_record"]["status"] == "verified", result["checks"]
    assert result["checks"]["workbook_cells"]["status"] == "mismatch"


def test_wrong_download_step_does_not_certify_report(portfolio_archive, tmp_path):
    archive, _, binding = _copy(portfolio_archive, tmp_path)
    path = archive / "bindings.json"
    values = json.loads(path.read_text())
    for event in values["execution_before_scoring"]["http_events"]:
        if event["stage"] == "download_portfolio_report":
            event["step_id"] = binding.step_ids["gate"]
    path.write_text(json.dumps(values))
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["download_binding"]["status"] == "mismatch", result["checks"]


def test_mutated_binding_cannot_replace_the_frozen_human_state_order(portfolio_archive):
    archive, sha, binding = portfolio_archive
    changed = binding.model_copy(update={"states": list(reversed(binding.states))})
    with pytest.raises(ValueError, match="ordered states"):
        _run(archive, sha, changed)


def test_changed_loss_assumption_is_rejected_after_native_input_hash_is_rewritten(portfolio_archive, tmp_path):
    archive, _, binding = _copy(portfolio_archive, tmp_path)
    with closing(_database(archive)) as conn:
        row = conn.execute("SELECT input_json FROM plan_step_runs WHERE step_id=? AND status='succeeded'",
                           (binding.step_ids["expected_loss"],)).fetchone()
        value = json.loads(row[0])
        value["lgd"] = 0.9
        _rewrite_step(conn, binding, "expected_loss", run_input=value)
        conn.commit()
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["native_steps"]["status"] == "verified", result["checks"]
    assert result["checks"]["business_contract"]["status"] == "mismatch"
