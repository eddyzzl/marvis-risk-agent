from __future__ import annotations

from dataclasses import replace
import json
import sqlite3

import pytest

from marvis.db import TaskRepository, connect, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.repositories.validation_contracts import (
    ValidationContractDataError,
    ValidationContractMaterialMismatch,
    ValidationContractRepository,
    ValidationContractRevisionConflict,
)
from tests.validation_builders import (
    make_candidate_contract,
    make_ready_contract,
    make_validation_confirmation,
)


def _create_material_task(tmp_path):
    paths = {
        "notebook": tmp_path / "model.ipynb",
        "sample": tmp_path / "sample.parquet",
        "pmml": tmp_path / "model.pmml",
        "dictionary": tmp_path / "metadata.xlsx",
    }
    for role, path in paths.items():
        path.write_bytes(f"fixture-{role}".encode())
    task = TaskRepository(tmp_path / "app.sqlite").create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(tmp_path),
            notebook_path=paths["notebook"].name,
            sample_path=paths["sample"].name,
            pmml_path=paths["pmml"].name,
            dictionary_path=paths["dictionary"].name,
        )
    )
    hashes = {role: sha256_file(path) for role, path in paths.items()}
    return task, paths, hashes


def test_validation_workflow_versions_are_server_assigned_and_immutable(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    validation = repo.create_task(
        TaskCreate(
            model_name="A", model_version="v1", validator="qa", source_dir=str(tmp_path)
        )
    )
    unrelated = repo.create_task(
        TaskCreate(
            task_type="modeling",
            model_name="M",
            model_version="v1",
            validator="qa",
            source_dir=str(tmp_path),
        )
    )

    assert validation.validation_workflow_version == 2
    assert unrelated.validation_workflow_version == 0
    init_db(db_path)
    assert repo.get_task(validation.id).validation_workflow_version == 2
    assert repo.get_task(unrelated.id).validation_workflow_version == 0


def test_migration_versions_only_historical_validation_rows(tmp_path):
    db_path = tmp_path / "schema-v2.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, task_type TEXT NOT NULL, "
            "validation_workflow_version INTEGER NOT NULL DEFAULT 0)"
        )
        conn.executemany(
            "INSERT INTO tasks (id, task_type, validation_workflow_version) VALUES (?, ?, ?)",
            [
                ("old-validation", "validation", 0),
                ("old-modeling", "modeling", 0),
                ("existing-v2", "validation", 2),
            ],
        )
        conn.execute("PRAGMA user_version = 2")

    init_db(db_path)
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        versions = dict(
            conn.execute(
                "SELECT id, validation_workflow_version FROM tasks ORDER BY id"
            )
        )
    assert versions == {"existing-v2": 2, "old-modeling": 0, "old-validation": 1}


def test_migration_adds_missing_workflow_version_column(tmp_path):
    db_path = tmp_path / "schema-v2-without-version-column.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, task_type TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO tasks (id, task_type) VALUES (?, ?)",
            [("old-validation", "validation"), ("old-modeling", "modeling")],
        )
        conn.execute("PRAGMA user_version = 2")

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        versions = dict(
            conn.execute(
                "SELECT id, validation_workflow_version FROM tasks ORDER BY id"
            )
        )
    assert versions == {"old-modeling": 0, "old-validation": 1}


def test_replace_candidates_round_trips_canonical_json_and_invalidates_confirmation(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    candidate = make_candidate_contract(material_hashes=hashes)

    first = repo.replace_candidates(task.id, candidate)
    confirmed = repo.confirm(
        task.id,
        make_validation_confirmation(),
        expected_revision=first.revision,
    )
    changed = replace(candidate, material_hashes={**hashes, "sample": "0" * 64})
    replaced = repo.replace_candidates(task.id, changed)

    assert first.revision == 1
    assert confirmed.revision == 2
    assert confirmed.status == "ready"
    assert replaced.revision == 3
    assert replaced.status == "pending_confirmation"
    assert replaced.contract.confirmed == {}
    assert repo.get(task.id) == replaced
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT candidate_json, confirmed_json, material_hashes_json FROM validation_input_contracts"
        ).fetchone()
    for value in row:
        assert (
            json.dumps(
                json.loads(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            == value
        )


def test_replace_candidates_never_trusts_incoming_ready_status(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    incoming_ready = replace(make_ready_contract(), material_hashes=hashes)

    pending = repo.replace_candidates(task.id, incoming_ready)

    assert pending.status == "pending_confirmation"
    assert pending.contract.status == "pending_confirmation"
    assert pending.contract.confirmed == {}
    assert repo.get(task.id) == pending

    incoming_blocked = replace(
        make_candidate_contract(material_hashes=hashes),
        status="blocked",
        conflicts=("unsupported derivation",),
    )
    blocked = repo.replace_candidates(task.id, incoming_blocked)

    assert blocked.status == "blocked"
    assert blocked.contract.status == "blocked"
    assert blocked.contract.confirmed == {}
    assert repo.get(task.id) == blocked


def test_confirmation_rejects_blocked_contract_even_without_conflict_details(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    blocked = repo.replace_candidates(
        task.id,
        replace(
            make_candidate_contract(material_hashes=hashes),
            status="blocked",
            conflicts=(),
        ),
    )

    with pytest.raises(ValueError, match="blocked.*cannot be confirmed"):
        repo.confirm(
            task.id,
            make_validation_confirmation(),
            expected_revision=blocked.revision,
        )

    assert repo.get(task.id) == blocked
    refreshed = TaskRepository(db_path).get_task(task.id)
    assert refreshed.algorithm == ""
    assert TaskRepository(db_path).list_audit(target_ref=task.id) == []


def test_confirmation_uses_optimistic_revision_syncs_task_and_writes_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    first = repo.replace_candidates(
        task.id, make_candidate_contract(material_hashes=hashes)
    )

    confirmed = repo.confirm(
        task.id,
        make_validation_confirmation(),
        expected_revision=first.revision,
    )

    assert confirmed.revision == 2
    assert confirmed.contract.require_output_field() == "probability_1"
    refreshed = TaskRepository(db_path).get_task(task.id)
    assert (
        refreshed.target_col,
        refreshed.split_col,
        refreshed.time_col,
        refreshed.algorithm,
    ) == (
        "y",
        "split",
        "apply_month",
        "xgb",
    )
    audits = TaskRepository(db_path).list_audit(target_ref=task.id)
    assert [(row["kind"], row["outcome"]) for row in audits] == [
        ("validation.input_contract.confirm", "succeeded")
    ]
    with pytest.raises(ValidationContractRevisionConflict, match="revision"):
        repo.confirm(
            task.id,
            make_validation_confirmation(),
            expected_revision=first.revision,
        )


def test_confirmation_rehashes_all_materials_and_rejects_changed_file(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    first = repo.replace_candidates(
        task.id, make_candidate_contract(material_hashes=hashes)
    )
    paths["pmml"].write_bytes(b"changed")

    with pytest.raises(ValidationContractMaterialMismatch, match="pmml"):
        repo.confirm(
            task.id, make_validation_confirmation(), expected_revision=first.revision
        )

    assert repo.get(task.id) == first
    assert TaskRepository(db_path).get_task(task.id).algorithm == ""


def test_confirmation_rejects_material_path_escape(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    outside = tmp_path.parent / "outside.pmml"
    outside.write_bytes(b"outside")
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET pmml_path = ? WHERE id = ?", (str(outside), task.id)
        )
    repo = ValidationContractRepository(db_path)
    first = repo.replace_candidates(
        task.id, make_candidate_contract(material_hashes=hashes)
    )

    with pytest.raises(ValueError, match="inside source_dir"):
        repo.confirm(
            task.id, make_validation_confirmation(), expected_revision=first.revision
        )


def test_repository_raises_data_error_for_corrupt_persisted_json(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    repo.replace_candidates(task.id, make_candidate_contract(material_hashes=hashes))
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE validation_input_contracts SET pmml_manifest_json = '{bad json' WHERE task_id = ?",
            (task.id,),
        )

    with pytest.raises(ValidationContractDataError, match="corrupt"):
        repo.get(task.id)


def test_replace_requires_existing_task_and_rejects_invalid_contract(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ValidationContractRepository(db_path)
    with pytest.raises(KeyError, match="Task not found"):
        repo.replace_candidates("missing", make_candidate_contract())
    contract = make_candidate_contract()
    with pytest.raises(ValueError, match="material hash roles"):
        repo.replace_candidates(
            "missing", replace(contract, material_hashes={"sample": "x"})
        )


def test_confirm_can_atomically_replace_resolved_schema_and_metadata(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    candidate = make_candidate_contract(material_hashes=hashes)
    first = repo.replace_candidates(task.id, candidate)
    resolved_schema = replace(candidate.require_sample_schema(), row_count=99)
    resolved_metadata = replace(
        candidate.require_feature_metadata(),
        extra_features=("unused",),
    )

    confirmed = repo.confirm(
        task.id,
        make_validation_confirmation(),
        expected_revision=first.revision,
        resolved_sample_schema=resolved_schema,
        resolved_feature_metadata=resolved_metadata,
    )

    assert confirmed.contract.require_sample_schema().row_count == 99
    assert confirmed.contract.require_feature_metadata().extra_features == ("unused",)


def test_confirmation_rolls_back_contract_and_task_when_audit_write_fails(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _paths, hashes = _create_material_task(tmp_path)
    repo = ValidationContractRepository(db_path)
    first = repo.replace_candidates(
        task.id, make_candidate_contract(material_hashes=hashes)
    )

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(
        "marvis.repositories.validation_contracts._write_audit_row", fail_audit
    )
    with pytest.raises(RuntimeError, match="audit unavailable"):
        repo.confirm(
            task.id, make_validation_confirmation(), expected_revision=first.revision
        )

    assert repo.get(task.id) == first
    refreshed = TaskRepository(db_path).get_task(task.id)
    assert (
        refreshed.target_col,
        refreshed.split_col,
        refreshed.time_col,
        refreshed.algorithm,
    ) == (
        "y",
        "split",
        "apply_month",
        "",
    )
