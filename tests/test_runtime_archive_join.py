"""Read-only independent JOIN verification over retained native HTTP originals."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_join import FrozenJoinBinding, revalidate_join_archive, _TOOLS
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model


@pytest.fixture(scope="module")
def join_archive(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("join-original-custody"))


def _build(root, variant=None):
    paths = write_synthetic_suite(root / "suite")
    suite = json.loads(paths["cases"].read_text())
    definition = next(case for case in suite["cases"] if case["id"] == "synthetic_join")
    if variant:
        import pandas as pd
        if variant == "collision":
            material = definition["materials"][0]
            source = paths["dataset_root"] / material["path"]
            frame = pd.read_parquet(source)
            frame["balance"] = range(1000, 1000 + len(frame))
            frame.to_parquet(source, index=False)
            material["sha256"] = digest(source.read_bytes())
        elif variant == "unmatched":
            material = definition["materials"][1]
            source = paths["dataset_root"] / material["path"]
            pd.read_parquet(source).iloc[:-1].to_parquet(source, index=False)
            material["sha256"] = digest(source.read_bytes())
            definition["actions"][-1]["content"] = "样本80行，特征79行；批准保留全部样本的左连接。"
        else:
            raise AssertionError(variant)
    suite["cases"] = [definition]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    case = RuntimeCase.model_validate(definition)
    with fixture_model() as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / case.id
    binding = _binding(archive, report, observed, case)
    assert observed["isolated_workspace_removed"]
    assert not Path(json.loads((archive / "manifest.json").read_text())["original_workspace"]).exists()
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


@pytest.mark.parametrize("variant", ["collision", "unmatched"])
def test_actual_http_join_collision_and_unmatched_rows_recompute(tmp_path, variant):
    archive, sha, binding = _build(tmp_path, variant)
    result = _run(archive, sha, binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["values"]["output_rows"] == 80
    assert result["values"]["output_columns"] == (4 if variant == "collision" else 3)


def _binding(archive, report, observed, case):
    steps = {key: next(s for s in observed["execution"]["steps"] if s["tool"] == tool) for key, tool in _TOOLS.items()}
    with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        join = conn.execute("SELECT * FROM joins").fetchone()
        feature_ids = [s["feature_dataset_id"] for s in json.loads(join["joins_json"])]
        source_ids = [join["anchor_dataset_id"], *feature_ids]
        hashes = [conn.execute("SELECT content_hash FROM datasets WHERE id=?", (identity,)).fetchone()[0]
                  for identity in [*source_ids, join["result_dataset_id"]]]
    return FrozenJoinBinding(case=case, run_id=report["run_id"], task_id=observed["execution"]["task_id"],
        plan_id=observed["execution"]["plans"][0]["id"], join_plan_id=join["id"],
        source_dataset_ids=source_ids, source_sha256=hashes[:-1], result_dataset_id=join["result_dataset_id"],
        result_sha256=hashes[-1], join_specs=[{"keys": [{"anchor_col": "id", "feature_col": "id",
            "match_method": "exact", "transform_side": "both"}], "dedup_strategy": None}],
        step_ids={key: step["id"] for key, step in steps.items()},
        output_sha256={key: step["output_sha256"] for key, step in steps.items()},
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"])


def _run(archive, sha, binding):
    return revalidate_join_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)


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


def test_native_join_originals_recompute_without_producer_kernel(join_archive, monkeypatch):
    from marvis.data.backend import DataBackend
    import marvis.data.join_evidence as receipt

    def forbidden(*args, **kwargs):
        raise AssertionError("producer implementation is not an independent reference")
    for name in ("left_join", "_left_join_membership", "feature_column_mapping"):
        monkeypatch.setattr(DataBackend, name, forbidden)
    monkeypatch.setattr(receipt, "_verify_proof", forbidden)
    monkeypatch.setattr(receipt, "_verify_copied_rows", forbidden)
    archive, sha, binding = join_archive
    before = {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}
    result = _run(archive, sha, binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["values"] == {"source_rows": [80, 80], "output_rows": 80, "output_columns": 3, "join_stages": 1}
    assert result["archive_unchanged_during_revalidation"]
    assert result["acceptance_claim"] == result["source_authentication"] == "not_established"
    assert before == {p.relative_to(archive): digest(p.read_bytes()) for p in archive.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field", ["run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"])
def test_join_archive_cannot_replace_external_bindings(join_archive, tmp_path, field):
    archive, _, binding = _copy(join_archive, tmp_path)
    path = archive / "bindings.json"
    value = json.loads(path.read_text())
    value[field] = {"source_sha256": "b" * 64} if field == "source" else "b" * 64
    path.write_text(json.dumps(value))
    assert _run(archive, _manifest(archive), binding)["checks"]["external_bindings"]["status"] == "mismatch"


@pytest.mark.parametrize("field,value", [("parent_output_refs", []), ("parent_output_bindings", []),
    ("source_dataset_refs", ["dataset:unbound"]), ("result_dataset_bindings", [])])
def test_join_native_dependencies_and_result_binding_required(join_archive, tmp_path, field, value):
    archive, _, binding = _copy(join_archive, tmp_path)
    with closing(_database(archive)) as conn:
        step_id = binding.step_ids["execute"]
        row = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=?", (step_id,)).fetchone()
        evidence = json.loads(row["evidence_json"])
        evidence[field] = value
        conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=?", (json.dumps(evidence), step_id))
        conn.commit()
    assert _run(archive, _manifest(archive), binding)["checks"]["native_steps"]["status"] == "mismatch"


def test_external_keys_cannot_be_replaced_by_saved_auto_detection(join_archive):
    archive, sha, binding = join_archive
    values = binding.model_dump()
    values["join_specs"][0]["keys"][0]["match_method"] = "exact_lower"
    result = _run(archive, sha, FrozenJoinBinding.model_validate(values))
    assert result["checks"]["join_contract"]["status"] == "mismatch"


def test_each_source_is_bound_to_its_original_material(join_archive, tmp_path):
    archive, _, binding = _copy(join_archive, tmp_path)
    path = archive / "workspace/datasets" / binding.task_id / ".source-identities" / f"{binding.source_dataset_ids[1]}.json"
    value = json.loads(path.read_text())
    value["sha256"] = binding.case.materials[0].sha256
    path.write_text(json.dumps(value))
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["source_materials"]["status"] == "mismatch"


def _rewrite_receipt(archive, conn, proof_row, proof):
    """Synchronize only an adversarial archive copy to test independent semantics."""
    path = archive / "workspace" / proof_row["path"]
    path.write_text(json.dumps(proof))
    sha = digest(path.read_bytes())
    conn.execute("UPDATE task_artifacts SET content_hash=? WHERE id=?", (sha, proof_row["id"]))
    for row in conn.execute("SELECT * FROM task_artifacts WHERE kind='dataset_join_membership_v1'").fetchall():
        provenance = json.loads(row["provenance_json"])
        provenance["evidence_hash"] = sha
        conn.execute("UPDATE task_artifacts SET provenance_json=? WHERE id=?", (json.dumps(provenance), row["id"]))


@pytest.mark.parametrize("mutation", ["wrong_right_row", "wrong_left_row", "wrong_output_row", "duplicate_member"])
def test_rehashed_receipt_cannot_replace_independent_physical_membership(join_archive, tmp_path, mutation):
    import pandas as pd

    archive, _, binding = _copy(join_archive, tmp_path)
    with closing(_database(archive)) as conn:
        row = conn.execute("SELECT * FROM task_artifacts WHERE kind='dataset_join_membership_v1'").fetchone()
        path = archive / "workspace" / row["path"]
        frame = pd.read_parquet(path)
        if mutation == "wrong_right_row":
            frame.at[0, "right_rows"] = [1]
        elif mutation == "wrong_left_row":
            frame.at[0, "left_row"] = 1
        elif mutation == "wrong_output_row":
            frame.at[0, "output_row"] = 1
        else:
            frame.at[0, "right_rows"] = [0, 0]
        frame.to_parquet(path, index=False)
        sha = digest(path.read_bytes())
        conn.execute("UPDATE task_artifacts SET content_hash=? WHERE id=?", (sha, row["id"]))
        proof_row = conn.execute("SELECT * FROM task_artifacts WHERE kind='dataset_join_evidence_v1'").fetchone()
        proof = json.loads((archive / "workspace" / proof_row["path"]).read_text())
        proof["steps"][0]["member_hash"] = sha
        _rewrite_receipt(archive, conn, proof_row, proof)
        conn.commit()
    result = _run(archive, _manifest(archive), binding)
    assert result["checks"]["receipt_artifacts"]["status"] == "verified", result["checks"]
    assert result["checks"]["independent_join_values"]["status"] == "verified"
    assert result["checks"]["physical_membership"]["status"] == "mismatch"


def test_rehashed_output_same_row_count_cannot_hide_wrong_join_values(join_archive, tmp_path):
    import pandas as pd

    archive, _, binding = _copy(join_archive, tmp_path)
    with closing(_database(archive)) as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=?", (binding.result_dataset_id,)).fetchone()
        path = archive / "workspace/datasets" / row["source_path"]
        frame = pd.read_parquet(path)
        frame.loc[0, "balance"] = 9000
        frame.to_parquet(path, index=False)
        sha = digest(path.read_bytes())
        conn.execute("UPDATE datasets SET content_hash=? WHERE id=?", (sha, row["id"]))
        saved = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=?", (binding.step_ids["execute"],)).fetchone()
        evidence = json.loads(saved["evidence_json"])
        evidence["result_dataset_bindings"][0]["content_hash"] = sha
        conn.execute("UPDATE plan_step_output_versions SET evidence_json=? WHERE step_id=?",
                     (json.dumps(evidence), binding.step_ids["execute"]))
        proof_row = conn.execute("SELECT * FROM task_artifacts WHERE kind='dataset_join_evidence_v1'").fetchone()
        proof = json.loads((archive / "workspace" / proof_row["path"]).read_text())
        proof["output_hash"] = proof["steps"][-1]["output_hash"] = sha
        provenance = json.loads(proof_row["provenance_json"])
        provenance["output_hash"] = sha
        conn.execute("UPDATE task_artifacts SET provenance_json=? WHERE id=?", (json.dumps(provenance), proof_row["id"]))
        _rewrite_receipt(archive, conn, proof_row, proof)
        conn.commit()
    values = binding.model_dump()
    values["result_sha256"] = sha
    result = _run(archive, _manifest(archive), FrozenJoinBinding.model_validate(values))
    assert result["checks"]["receipt_artifacts"]["status"] == "verified", result["checks"]
    assert result["checks"]["independent_join_values"]["status"] == "mismatch"
