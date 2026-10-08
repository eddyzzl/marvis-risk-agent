"""Actual retained label originals; the network model is a local protocol fixture."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_labeling import FrozenLabelingBinding, revalidate_labeling_archive
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


@pytest.fixture(scope="module")
def label_archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("label-original-custody")
    return _build_label_archive(root)


def _build_label_archive(root, variant=None):
    paths = write_synthetic_suite(root / "suite", normal_labeling_only=True)
    suite = json.loads(paths["cases"].read_text())
    definition = suite["cases"][0]
    # Use the formal family's name while retaining explicit public synthetic
    # development provenance. The formal gate must still reject this evidence.
    definition["family"] = "labeling"
    if variant:
        material = definition["materials"][0]
        source = paths["dataset_root"] / material["path"]
        frame = pd.read_parquet(source)
        request = definition["actions"][0]["labeling_request"]
        if variant == "csv_status":
            frame["status"] = frame.pop("dpd").map(lambda value: "loss" if value >= 90 else "clean")
            request.pop("dpd_col")
            request.pop("threshold_dpd")
            request.update(rule_kind="status", status_col="status", states=["clean", "watch", "loss"], threshold_status="loss")
            destination = source.with_suffix(".csv")
            frame.to_csv(destination, index=False)
        elif variant == "xlsx_fractional":
            frame["dpd"] = frame["dpd"].astype(float)
            frame.loc[frame["dpd"] == 90, "dpd"] = 90.75
            frame.loc[(frame["loan_id"] == "0002") & (frame["mob"] == 2), "dpd"] = 90.25
            request["threshold_dpd"] = 90.5
            destination = source.with_suffix(".xlsx")
            frame.to_excel(destination, index=False, sheet_name="Repayment")
        else:
            raise AssertionError(variant)
        material.update(path=destination.name, sha256=digest(destination.read_bytes()))
        definition["actions"][3]["content"] = "确认按照已审核的标签口径写入标签，未成熟空标签保留。"
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    case = RuntimeCase.model_validate(definition)
    with fixture_model(answer_factory=_business_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], [(r["runtime_status"], r.get("evidence_custody"), r.get("error_code")) for r in report["cases"]]
    observed = report["cases"][0]
    assert observed["isolated_workspace_removed"]
    steps = {"labels" if step["tool"] == "labeling.define_label" else "maturity": step
             for step in observed["execution"]["steps"] if step["tool"] in {"labeling.define_label", "labeling.check_cohort_maturity"}}
    binding = FrozenLabelingBinding(case=case, run_id=report["run_id"], task_id=observed["execution"]["task_id"],
        plan_id=observed["execution"]["plans"][0]["id"], step_ids={key: step["id"] for key, step in steps.items()},
        output_sha256={key: step["output_sha256"] for key, step in steps.items()},
        download_sha256={kind: next(event["sha256"] for event in observed["http_events"] if event["stage"] == f"download_labeling_{kind}")
                         for kind in ("dataset", "evidence")},
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"])
    archive = root / "custody" / report["run_id"] / case.id
    assert not Path(json.loads((archive / "manifest.json").read_text())["original_workspace"]).exists()
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


@pytest.mark.parametrize("variant", ["csv_status", "xlsx_fractional"])
def test_archive_reference_matches_actual_uploaded_formats_and_declared_rule(tmp_path, variant):
    archive, sha, binding = _build_label_archive(tmp_path, variant)
    result = _run(archive, sha, binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["values"]["quality"]["n_bad"] == 1
    assert result["values"]["quality"]["n_good"] == 2
    assert result["values"]["quality"]["n_unmatured"] == 1


def _copy(label_archive, tmp_path):
    archive, sha, binding = label_archive
    destination = tmp_path / "archive"
    shutil.copytree(archive, destination)
    return destination, sha, binding


def _manifest(archive):
    from test_runtime_archive_validation import _rewrite_manifest
    return _rewrite_manifest(archive)


def _run(archive, sha, binding):
    return revalidate_labeling_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)


def test_label_originals_are_independently_recomputed_without_old_workspace_or_producer_kernel(label_archive, monkeypatch):
    import marvis.data.label_construction as production
    import marvis.packs.labeling.evidence as receipts

    def forbidden(*args, **kwargs):
        raise AssertionError("a producer kernel is not an independent label reference")
    monkeypatch.setattr(production, "construct_label", forbidden)
    monkeypatch.setattr(production, "check_cohort_maturity", forbidden)
    monkeypatch.setattr(receipts, "_replay_labeling_evidence", forbidden)
    archive, sha, binding = label_archive
    before = {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}
    result = _run(archive, sha, binding)
    assert result["supported_checks_verified"], json.dumps(result["checks"], indent=2)
    assert result["values"]["quality"] == {"n_loans": 4, "n_bad": 1, "n_good": 2, "n_unmatured": 1,
                                          "label_coverage": 0.75, "bad_rate": 1 / 3}
    assert result["checks"]["label_recomputation"]["rows_excluded_after_as_of"] == 2
    assert not result["values"]["all_matured"] and not result["values"]["active_dataset_changed"]
    assert result["archive_unchanged_during_revalidation"] and result["acceptance_claim"] == "not_established"
    assert result["source_authentication"] == "not_established"
    assert before == {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["bytes", "delete", "manifest"])
def test_label_archive_changes_cannot_replace_a_caller_held_digest(label_archive, tmp_path, damage):
    archive, sha, binding = _copy(label_archive, tmp_path)
    path = next((archive / "inputs").glob("*.parquet"))
    if damage == "delete":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"modified")
    if damage == "manifest":
        assert _manifest(archive) != sha
    result = _run(archive, sha, binding)
    assert result["checks"]["archive_integrity"]["status"] == "mismatch"
    assert not result["supported_checks_verified"]


@pytest.mark.parametrize("field", ["run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"])
def test_label_archive_does_not_supply_its_own_frozen_identity(label_archive, tmp_path, field):
    archive, _, binding = _copy(label_archive, tmp_path)
    path = archive / "bindings.json"
    payload = json.loads(path.read_bytes())
    payload[field] = {"source_sha256": "b" * 64} if field == "source" else "b" * 64
    path.write_text(json.dumps(payload))
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["external_bindings"]["status"] == "mismatch"
    assert not result["supported_checks_verified"]


def _editable_database(archive):
    conn = sqlite3.connect(archive / "workspace/marvis.sqlite")
    conn.row_factory = sqlite3.Row
    # Attack only a private test copy. Production immutability triggers remain
    # intact. A newly authenticated inventory must still not certify bad math.
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute('DROP TRIGGER "' + row[0].replace('"', '""') + '"')
    return conn


def _rebuilt_wrong_producer(archive, binding, mutation):
    from marvis.orchestrator.evidence import payload_hash

    manifest = json.loads((archive / "manifest.json").read_bytes())
    original_root = Path(manifest["original_workspace"])
    with closing(_editable_database(archive)) as conn:
        saved = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=? ORDER BY version DESC LIMIT 1",
                             (binding.step_ids["labels"],)).fetchone()
        output = json.loads(saved["output_json"])
        native = json.loads(saved["evidence_json"])
        records = {kind: conn.execute("SELECT * FROM task_artifacts WHERE id=?", (output[f"{kind}_artifact_id"],)).fetchone()
                   for kind in ("dataset", "evidence")}
        paths = {kind: archive / "workspace" / Path(row["path"]).relative_to(original_root) for kind, row in records.items()}
        evidence = json.loads(paths["evidence"].read_bytes())
        if mutation == "labels":
            dataset = conn.execute("SELECT * FROM datasets WHERE id=?", (output["result_dataset_id"],)).fetchone()
            result_path = archive / "workspace/datasets" / dataset["source_path"]
            labels = pd.read_parquet(result_path)
            labels[output["target_col"]] = labels[output["target_col"]].fillna(0.0)
            labels.to_parquet(result_path, index=False)
            new_hash = digest(result_path.read_bytes())
            conn.execute("UPDATE datasets SET content_hash=? WHERE id=?", (new_hash, dataset["id"]))
            output["result_content_hash"] = evidence["result"]["content_hash"] = new_hash
            output["n_good"], output["n_unmatured"], output["bad_rate"] = 3, 0, 0.25
            output["quality"].update(n_good=3, n_unmatured=0, bad_rate=0.25, label_coverage=1.0)
            evidence["quality"] = deepcopy(output["quality"])
            for item in native.get("result_dataset_bindings", []):
                if item["dataset_id"] == dataset["id"]:
                    item["content_hash"] = new_hash
            raw = paths["dataset"].read_text(encoding="utf-8-sig")
            # The native fixture's third entity is the only null target.
            assert "'0003,'2026-03," in raw
            paths["dataset"].write_text(raw.replace("'0003,'2026-03,\n", "'0003,'2026-03,0.0\n"), encoding="utf-8-sig")
        elif mutation == "quality":
            output["bad_rate"] = output["quality"]["bad_rate"] = evidence["quality"]["bad_rate"] = 0.75
        elif mutation == "csv":
            raw = paths["dataset"].read_text(encoding="utf-8-sig")
            assert "1.0" in raw
            paths["dataset"].write_text(raw.replace("1.0", "0.0", 1), encoding="utf-8-sig")
        else:
            raise AssertionError(mutation)
        paths["evidence"].write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        for kind, row in records.items():
            output[f"{kind}_content_hash"] = digest(paths[kind].read_bytes())
            output[f"{kind}_download_url"] = output[f"{kind}_download_url"].split("?", 1)[0] + "?expected_content_hash=" + output[f"{kind}_content_hash"]
            provenance = json.loads(row["provenance_json"])
            provenance["result_content_hash"] = output["result_content_hash"]
            conn.execute("UPDATE task_artifacts SET content_hash=?, provenance_json=? WHERE id=?",
                         (output[f"{kind}_content_hash"], json.dumps(provenance), row["id"]))
        native["output_hash"] = native["raw_output_hash"] = payload_hash(output)
        conn.execute("UPDATE plan_step_output_versions SET output_json=?, evidence_json=? WHERE step_id=? AND version=?",
                     (json.dumps(output), json.dumps(native), binding.step_ids["labels"], saved["version"]))
        conn.execute("UPDATE plan_step_runs SET output_hash=?,raw_output_hash=? WHERE id=?",
                     (payload_hash(output), payload_hash(output), native["step_run_id"]))
        conn.commit()
    changed = binding.model_copy(deep=True, update={
        "output_sha256": {**binding.output_sha256, "labels": digest(output)},
        "download_sha256": {kind: output[f"{kind}_content_hash"] for kind in ("dataset", "evidence")},
    })
    path = archive / "bindings.json"
    payload = json.loads(path.read_bytes())
    for event in payload["execution_before_scoring"]["http_events"]:
        for kind in ("dataset", "evidence"):
            if event["stage"] == f"download_labeling_{kind}":
                event["sha256"] = changed.download_sha256[kind]
    path.write_text(json.dumps(payload))
    return changed


@pytest.mark.parametrize("mutation,check", [("labels", "label_recomputation"), ("quality", "quality_and_maturity"), ("csv", "csv_values")])
def test_rebuilt_hashes_and_native_success_cannot_certify_incorrect_labels_or_carrier_values(label_archive, tmp_path, mutation, check):
    archive, _, binding = _copy(label_archive, tmp_path)
    changed = _rebuilt_wrong_producer(archive, binding, mutation)
    result = _run(archive, _manifest(archive), changed)
    assert result["checks"]["native_steps"]["status"] == "verified", result["checks"]
    assert result["checks"]["source_material"]["status"] == "verified", result["checks"]
    assert result["checks"][check]["status"] == "mismatch", result["checks"]
    assert not result["supported_checks_verified"]


@pytest.mark.parametrize("mutation", ["task_owner", "run_status", "download_binding"])
def test_label_native_ownership_run_and_external_download_identity_are_required(label_archive, tmp_path, mutation):
    archive, _, binding = _copy(label_archive, tmp_path)
    check = "native_steps"
    if mutation == "download_binding":
        binding = binding.model_copy(deep=True, update={"download_sha256": {**binding.download_sha256, "dataset": "f" * 64}})
        check = "download_binding"
    else:
        with closing(_editable_database(archive)) as conn:
            if mutation == "task_owner":
                conn.execute("UPDATE plans SET task_id='foreign-task' WHERE id=?", (binding.plan_id,))
            else:
                conn.execute("UPDATE plan_step_runs SET status='failed' WHERE step_id=?", (binding.step_ids["labels"],))
            conn.commit()
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"][check]["status"] == "mismatch", result["checks"]
    assert not result["supported_checks_verified"]


@pytest.mark.parametrize("field,value", [
    ("parent_output_bindings", []), ("parent_output_refs", []),
    ("source_dataset_refs", ["dataset:foreign"]), ("result_dataset_bindings", []),
    ("output_ref", "metrics:foreign:v1"),
])
def test_label_native_parent_and_dataset_bindings_cannot_be_replaced(label_archive, tmp_path, field, value):
    archive, _, binding = _copy(label_archive, tmp_path)
    with closing(_editable_database(archive)) as conn:
        saved = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=? ORDER BY version DESC LIMIT 1",
                             (binding.step_ids["labels"],)).fetchone()
        native = json.loads(saved["evidence_json"])
        native[field] = value
        conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=? AND version=?",
                     (json.dumps(native), binding.step_ids["labels"], saved["version"]))
        conn.commit()
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["native_steps"]["status"] == "mismatch", result["checks"]
    assert not result["supported_checks_verified"]


def test_label_recomputation_uses_copied_bytes_and_detects_changed_original_archive(label_archive, tmp_path, monkeypatch):
    import marvis.orchestrator.eval.runtime_archive_labeling as module

    archive, sha, binding = _copy(label_archive, tmp_path)
    original = module._authenticated_snapshot
    def snapshot_then_modify(source, target, expected):
        original(source, target, expected)
        next((source / "inputs").glob("*.parquet")).write_bytes(b"altered after snapshot")
    monkeypatch.setattr(module, "_authenticated_snapshot", snapshot_then_modify)
    result = _run(archive, sha, binding)
    assert result["checks"]["label_recomputation"]["status"] == "verified", result["checks"]
    assert result["checks"]["csv_values"]["status"] == "verified"
    assert result["checks"]["archive_integrity"]["status"] == "mismatch"
    assert not result["supported_checks_verified"] and not result["archive_unchanged_during_revalidation"]


def test_label_recomputation_does_not_echo_arbitrary_external_source_extensions(label_archive):
    archive, sha, binding = label_archive
    extended = binding.model_copy(deep=True, update={"source": {**binding.source, "private_extension": "do-not-echo-this-marker"}})
    result = _run(archive, sha, extended)
    assert not result["supported_checks_verified"]
    assert "do-not-echo-this-marker" not in json.dumps(result)
