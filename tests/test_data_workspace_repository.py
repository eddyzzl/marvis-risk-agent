from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from threading import Barrier

import pytest

from marvis.data.workspace import (
    DATA_FIELD_ROLES,
    DATA_WORKSPACE_PAGES,
    DATA_WORKSPACE_SCHEMA_VERSION,
    DataSemanticMapping,
    DataWorkspaceDraft,
)
from marvis.db_schema import connect, init_db
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceDatasetMismatch,
    DataWorkspaceDatasetNotFound,
    DataWorkspaceRepository,
    DataWorkspaceResetRequired,
    DataWorkspaceRevisionConflict,
)


_CREATED_AT = "2026-07-19T01:02:03+00:00"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_task(db_path, task_id: str = "task-1") -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, 'data task', 'v1', 'tester', '/tmp/source',
                      'created', 'created', ?, ?)
            """,
            (task_id, _CREATED_AT, _CREATED_AT),
        )


def _seed_dataset(
    db_path,
    *,
    dataset_id: str,
    task_id: str = "task-1",
    content_hash: str | None = None,
    columns: tuple[str, ...] = ("customer_id", "bad", "score"),
) -> str | None:
    profiles = [
        {
            "name": column,
            "dtype": "string",
            "semantic_role": "unknown",
            "fingerprint": {
                "value_kind": "text",
                "length_mode": None,
                "regex_pattern": None,
                "is_hashed": False,
                "hash_type": None,
                "hex_case": None,
                "date_format": None,
            },
            "null_rate": 0.0,
            "cardinality": 10,
            "sample_values": [],
        }
        for column in columns
    ]
    resolved_hash = _sha(dataset_id) if content_hash is None else content_hash
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO datasets(
                id, task_id, role, source_path, format, row_count,
                columns_json, has_target, target_col, created_at, content_hash
            ) VALUES (?, ?, 'analysis', ?, 'parquet', 10, ?, 1, 'bad', ?, ?)
            """,
            (
                dataset_id,
                task_id,
                f"/tmp/{dataset_id}.parquet",
                json.dumps(profiles, separators=(",", ":")),
                _CREATED_AT,
                resolved_hash,
            ),
        )
    return resolved_hash


def _activation(dataset_id: str, content_hash: str) -> DataWorkspaceDraft:
    return DataWorkspaceDraft(
        active_dataset_id=dataset_id,
        active_dataset_content_hash=content_hash,
    )


def test_workspace_constants_and_mapping_are_strict_and_canonical():
    assert DATA_WORKSPACE_SCHEMA_VERSION == "data-workspace.v1"
    assert DATA_WORKSPACE_PAGES == (
        "overview",
        "fields",
        "semantics",
        "history",
        "statistics",
    )
    assert set(DATA_FIELD_ROLES) >= {
        "id",
        "target",
        "score",
        "feature",
        "rule_node",
    }

    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={"score": "score", "bad": "target"},
        business_names={"score": "模型分", "bad": "风险标签"},
    )
    assert list(mapping.field_roles) == ["bad", "score"]
    with pytest.raises(TypeError):
        mapping.field_roles["score"] = "ignore"  # type: ignore[index]
    with pytest.raises(TypeError):
        mapping.business_names["score"] = "篡改"  # type: ignore[index]

    with pytest.raises(ValueError, match="unsupported role"):
        DataSemanticMapping(field_roles={"score": "prediction"})
    with pytest.raises(ValueError, match="must match target_col"):
        DataSemanticMapping(target_col="bad", field_roles={"score": "target"})
    with pytest.raises(ValueError, match="canonical non-empty text"):
        DataSemanticMapping(business_names={"score": " 模型分"})
    with pytest.raises(ValueError, match="canonical non-empty text"):
        DataWorkspaceDraft(selected_field="score\x00unsafe")


def test_get_or_default_is_stable_and_does_not_persist_or_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    repo = DataWorkspaceRepository(db_path)

    first = repo.get_or_default("task-1")
    replay = repo.save("task-1", DataWorkspaceDraft(), expected_revision=0)
    second = repo.get_or_default("task-1")

    assert first == replay == second
    assert first.schema_version == DATA_WORKSPACE_SCHEMA_VERSION
    assert first.revision == 0
    assert first.analysis_generation == 0
    assert first.active_dataset_id is None
    assert first.active_dataset_content_hash is None
    assert first.page == "overview"
    assert first.updated_at == _CREATED_AT
    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM data_workspaces").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 0


def test_real_change_uses_cas_and_writes_one_audit_while_replay_is_noop(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    repo = DataWorkspaceRepository(db_path)
    draft = DataWorkspaceDraft(page="history")

    saved = repo.save("task-1", draft, expected_revision=0)
    replay = repo.save("task-1", draft, expected_revision=1)

    assert saved == replay
    assert saved.revision == 1
    assert saved.analysis_generation == 0
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT kind, target_ref, outcome, detail_json FROM audit"
        ).fetchone()
        stored = conn.execute(
            "SELECT revision, page, semantic_mapping_json FROM data_workspaces"
        ).fetchone()
    assert row["kind"] == "data.workspace.update"
    assert row["target_ref"] == "task-1"
    assert row["outcome"] == "succeeded"
    audit_detail = json.loads(row["detail_json"])
    assert audit_detail["revision"] == 1
    assert audit_detail["analysis_generation"] == 0
    assert audit_detail["active_dataset_id"] is None
    assert audit_detail["inputs_hash"]
    assert tuple(stored) == (
        1,
        "history",
        '{"business_names":{},"field_roles":{},"target_col":null}',
    )

    with pytest.raises(DataWorkspaceRevisionConflict, match="expected 0, found 1"):
        repo.save("task-1", draft, expected_revision=0)


def test_concurrent_saves_with_same_revision_have_one_winner(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    barrier = Barrier(2)

    def save_page(page: str):
        barrier.wait()
        try:
            snapshot = DataWorkspaceRepository(db_path).save(
                "task-1",
                DataWorkspaceDraft(page=page),
                expected_revision=0,
            )
            return "saved", snapshot.page
        except DataWorkspaceRevisionConflict:
            return "conflict", page

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save_page, ("history", "statistics")))

    assert sorted(status for status, _page in results) == ["conflict", "saved"]
    winner_page = next(page for status, page in results if status == "saved")
    snapshot = DataWorkspaceRepository(db_path).get_or_default("task-1")
    assert snapshot.revision == 1
    assert snapshot.page == winner_page
    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_caller_audit_can_add_context_but_cannot_override_canonical_evidence(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    repo = DataWorkspaceRepository(db_path)

    saved = repo.save(
        "task-1",
        DataWorkspaceDraft(page="history"),
        expected_revision=0,
        audit={
            "actor": "user:analyst",
            "detail": {"revision": 999, "note": "natural-language request"},
        },
    )

    with connect(db_path) as conn:
        audit = conn.execute(
            "SELECT kind, actor, inputs_hash, detail_json FROM audit"
        ).fetchone()
    detail = json.loads(audit["detail_json"])
    assert audit["kind"] == "data.workspace.update"
    assert audit["actor"] == "user:analyst"
    assert detail["revision"] == saved.revision == 1
    assert detail["inputs_hash"] == audit["inputs_hash"]
    assert detail["note"] == "natural-language request"

    with pytest.raises(DataWorkspaceDataError, match="only customize"):
        repo.save(
            "task-1",
            DataWorkspaceDraft(page="statistics"),
            expected_revision=1,
            audit={"kind": "caller.override"},
        )
    with connect(db_path) as conn:
        assert conn.execute(
            "SELECT revision FROM data_workspaces WHERE task_id = 'task-1'"
        ).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_initial_activation_requires_reset_and_increments_generation(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    dataset_hash = _seed_dataset(db_path, dataset_id="dataset-1")
    assert dataset_hash is not None
    repo = DataWorkspaceRepository(db_path)

    with pytest.raises(DataWorkspaceResetRequired, match="requires reset payload"):
        repo.save(
            "task-1",
            DataWorkspaceDraft(
                active_dataset_id="dataset-1",
                active_dataset_content_hash=dataset_hash,
                page="fields",
            ),
            expected_revision=0,
        )

    activated = repo.save(
        "task-1",
        _activation("dataset-1", dataset_hash),
        expected_revision=0,
    )

    assert activated.revision == 1
    assert activated.analysis_generation == 1
    assert repo.get_or_default("task-1") == activated


def test_same_dataset_allows_valid_semantics_without_generation_change(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    dataset_hash = _seed_dataset(db_path, dataset_id="dataset-1")
    assert dataset_hash is not None
    repo = DataWorkspaceRepository(db_path)
    activated = repo.save(
        "task-1",
        _activation("dataset-1", dataset_hash),
        expected_revision=0,
    )

    configured = repo.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id="dataset-1",
            active_dataset_content_hash=dataset_hash,
            page="semantics",
            selected_field="score",
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={
                    "customer_id": "id",
                    "bad": "target",
                    "score": "score",
                },
                business_names={"score": "模型分"},
            ),
        ),
        expected_revision=activated.revision,
    )

    assert configured.revision == 2
    assert configured.analysis_generation == 1
    assert configured.semantic_mapping.target_col == "bad"


@pytest.mark.parametrize(
    "draft",
    [
        DataWorkspaceDraft(
            active_dataset_id="dataset-1",
            active_dataset_content_hash=_sha("dataset-1"),
            selected_field="missing",
        ),
        DataWorkspaceDraft(
            active_dataset_id="dataset-1",
            active_dataset_content_hash=_sha("dataset-1"),
            semantic_mapping=DataSemanticMapping(target_col="missing"),
        ),
        DataWorkspaceDraft(
            active_dataset_id="dataset-1",
            active_dataset_content_hash=_sha("dataset-1"),
            semantic_mapping=DataSemanticMapping(field_roles={"missing": "score"}),
        ),
        DataWorkspaceDraft(
            active_dataset_id="dataset-1",
            active_dataset_content_hash=_sha("dataset-1"),
            semantic_mapping=DataSemanticMapping(
                business_names={"missing": "缺失字段"}
            ),
        ),
    ],
)
def test_save_rejects_unknown_dataset_column_references(tmp_path, draft):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    _seed_dataset(db_path, dataset_id="dataset-1")
    repo = DataWorkspaceRepository(db_path)
    repo.save(
        "task-1",
        _activation("dataset-1", _sha("dataset-1")),
        expected_revision=0,
    )

    with pytest.raises(DataWorkspaceDataError, match="unknown dataset column.*missing"):
        repo.save("task-1", draft, expected_revision=1)


def test_dataset_switch_and_clear_require_reset_and_increment_generation(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    first_hash = _seed_dataset(db_path, dataset_id="dataset-1")
    second_hash = _seed_dataset(
        db_path,
        dataset_id="dataset-2",
        columns=("replacement_field",),
    )
    assert first_hash is not None and second_hash is not None
    repo = DataWorkspaceRepository(db_path)
    first = repo.save(
        "task-1", _activation("dataset-1", first_hash), expected_revision=0
    )
    configured = repo.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id="dataset-1",
            active_dataset_content_hash=first_hash,
            page="fields",
            selected_field="score",
        ),
        expected_revision=first.revision,
    )

    with pytest.raises(DataWorkspaceResetRequired):
        repo.save(
            "task-1",
            DataWorkspaceDraft(
                active_dataset_id="dataset-2",
                active_dataset_content_hash=second_hash,
                page="fields",
                selected_field="score",
            ),
            expected_revision=configured.revision,
        )

    switched = repo.save(
        "task-1",
        _activation("dataset-2", second_hash),
        expected_revision=configured.revision,
    )
    cleared = repo.save(
        "task-1",
        DataWorkspaceDraft(),
        expected_revision=switched.revision,
    )

    assert switched.analysis_generation == 2
    assert cleared.analysis_generation == 3
    assert cleared.active_dataset_id is None


def test_dataset_binding_requires_existence_same_task_and_exact_verified_hash(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path, "task-1")
    _seed_task(db_path, "task-2")
    owned_hash = _seed_dataset(db_path, dataset_id="owned", task_id="task-1")
    foreign_hash = _seed_dataset(db_path, dataset_id="foreign", task_id="task-2")
    unverified_hash = _seed_dataset(
        db_path,
        dataset_id="unverified",
        task_id="task-1",
    )
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = NULL WHERE id = 'unverified'"
        )
    assert owned_hash is not None and foreign_hash is not None
    assert unverified_hash is not None
    repo = DataWorkspaceRepository(db_path)

    with pytest.raises(DataWorkspaceDatasetNotFound, match="dataset not found"):
        repo.save(
            "task-1",
            _activation("missing", _sha("missing")),
            expected_revision=0,
        )
    with pytest.raises(DataWorkspaceDatasetMismatch, match="belongs to task task-2"):
        repo.save(
            "task-1",
            _activation("foreign", foreign_hash),
            expected_revision=0,
        )
    with pytest.raises(DataWorkspaceDatasetMismatch, match="does not match"):
        repo.save(
            "task-1",
            _activation("owned", _sha("wrong")),
            expected_revision=0,
        )
    with pytest.raises(DataWorkspaceDatasetMismatch, match="no verified content_hash"):
        repo.save(
            "task-1",
            _activation("unverified", unverified_hash),
            expected_revision=0,
        )


def test_get_fails_closed_for_noncanonical_or_drifted_persisted_data(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path)
    dataset_hash = _seed_dataset(db_path, dataset_id="dataset-1")
    assert dataset_hash is not None
    repo = DataWorkspaceRepository(db_path)
    saved = repo.save(
        "task-1",
        _activation("dataset-1", dataset_hash),
        expected_revision=0,
    )

    with connect(db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = ? WHERE id = 'dataset-1'",
            (_sha("tampered"),),
        )
    with pytest.raises(DataWorkspaceDatasetMismatch, match="does not match"):
        repo.get_or_default("task-1")

    with connect(db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = ? WHERE id = 'dataset-1'",
            (dataset_hash,),
        )
        conn.execute(
            "UPDATE data_workspaces SET semantic_mapping_json = ? WHERE task_id = ?",
            (
                '{"target_col": null, "field_roles": {}, "business_names": {}}',
                "task-1",
            ),
        )
    with pytest.raises(DataWorkspaceDataError, match="corrupt data workspace"):
        repo.get_or_default("task-1")

    assert saved.revision == 1


def test_missing_task_and_noncanonical_direct_inputs_fail_closed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DataWorkspaceRepository(db_path)

    with pytest.raises(KeyError, match="Task not found"):
        repo.get_or_default("missing")
    with pytest.raises(DataWorkspaceDataError, match="canonical non-empty text"):
        repo.get_or_default(" missing")
    with pytest.raises(DataWorkspaceDataError, match="non-negative integer"):
        repo.save("missing", DataWorkspaceDraft(), expected_revision=True)
