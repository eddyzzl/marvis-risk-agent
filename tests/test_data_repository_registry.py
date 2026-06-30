import pandas as pd
import pytest

from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import (
    ColumnFingerprint,
    ColumnProfile,
    Dataset,
    JoinDiagnostics,
    JoinPlan,
    JoinSpec,
    KeyPair,
)
from marvis.data.join_engine import JoinEngine
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, init_db
import marvis.db as db_module
from marvis.state_machine import ConflictError


def _profile(name: str, role: str = "id") -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        semantic_role=role,
        fingerprint=ColumnFingerprint("categorical", None, None, False, None, None, None),
        null_rate=0.0,
        cardinality=2,
        sample_values=("a***", "b***"),
    )


def _dataset(dataset_id: str, *, task_id: str = "task-1", role: str = "sample") -> Dataset:
    return Dataset(
        id=dataset_id,
        task_id=task_id,
        role=role,
        source_path=f"{task_id}/{dataset_id}.parquet",
        format="parquet",
        sheet=None,
        row_count=2,
        columns=(_profile("customer_id"),),
        has_target=False,
        target_col=None,
        created_at="2026-06-19T00:00:00Z",
    )


def _join_spec(*, confirmed: bool = False) -> JoinSpec:
    return JoinSpec(
        feature_dataset_id="feature-1",
        key_pairs=[
            KeyPair(
                anchor_col="customer_id",
                feature_col="customer_id",
                match_method="exact",
                transform_side="both",
                match_rate=1.0,
                resolved_by="empirical",
            ),
        ],
        diagnostics=JoinDiagnostics(
            anchor_rows=2,
            feature_rows=2,
            feature_key_unique=True,
            matched_rows=2,
            match_rate=1.0,
            joined_rows_preview=2,
            fan_out_detected=False,
            shrink_detected=False,
            new_columns=1,
            new_columns_null_rate=0.0,
        ),
        dedup_strategy=None,
        confirmed=confirmed,
    )


def test_dataset_repository_round_trips_datasets_and_roles(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    dataset = _dataset("dataset-1")

    repo.create_dataset(dataset)
    repo.create_dataset(_dataset("dataset-2", task_id="task-2", role="feature"))
    repo.set_dataset_role(dataset.id, "feature")

    loaded = repo.get_dataset(dataset.id)
    task_datasets = repo.list_datasets("task-1")

    assert loaded is not None
    assert loaded.id == dataset.id
    assert loaded.role == "feature"
    assert loaded.columns[0].fingerprint.value_kind == "categorical"
    assert task_datasets == [loaded]
    assert repo.list_datasets("task-2")[0].id == "dataset-2"


def test_dataset_repository_round_trips_join_plans_and_updates_specs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec()],
        status="draft",
    )

    repo.create_join_plan(plan)
    loaded = repo.load_join_plan(plan.id)
    spec = loaded.joins[0]
    spec.confirmed = True
    spec.dedup_strategy = "first"
    repo.update_join_spec(plan.id, spec)
    repo.set_join_plan_executed(plan.id, "derived-1")
    executed = repo.load_join_plan(plan.id)

    assert loaded.id == plan.id
    assert loaded.joins[0].key_pairs[0].match_method == "exact"
    assert executed.status == "executed"
    assert executed.result_dataset_id == "derived-1"
    assert executed.joins[0].confirmed is True
    assert executed.joins[0].dedup_strategy == "first"


def test_dataset_repository_rejects_reexecuting_join_plan(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec(confirmed=True)],
        status="draft",
    )
    repo.create_join_plan(plan)

    repo.set_join_plan_executed(plan.id, "derived-1")
    with pytest.raises(ConflictError, match="cannot execute again"):
        repo.set_join_plan_executed(plan.id, "derived-2")

    loaded = repo.load_join_plan(plan.id)
    assert loaded.status == "executed"
    assert loaded.result_dataset_id == "derived-1"


def test_dataset_repository_rolls_back_join_spec_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec()],
        status="draft",
    )
    repo.create_join_plan(plan)
    spec = repo.load_join_plan(plan.id).joins[0]
    spec.confirmed = True

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.update_join_spec_with_audit(
            plan.id,
            spec,
            audit={
                "kind": "join.confirmed",
                "target_ref": plan.id,
                "actor": "system",
                "outcome": "confirmed",
                "detail": {"task_id": plan.task_id},
            },
        )

    loaded = repo.load_join_plan(plan.id)
    assert loaded.joins[0].confirmed is False


def test_dataset_repository_rolls_back_join_executed_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec(confirmed=True)],
        status="draft",
    )
    repo.create_join_plan(plan)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.set_join_plan_executed_with_audit(
            plan.id,
            "derived-1",
            audit={
                "kind": "join.executed",
                "target_ref": plan.id,
                "actor": "system",
                "outcome": "succeeded",
                "detail": {"task_id": plan.task_id},
            },
        )

    loaded = repo.load_join_plan(plan.id)
    assert loaded.status == "draft"
    assert loaded.result_dataset_id is None


def test_dataset_repository_records_join_result_dataset_and_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec(confirmed=True)],
        status="draft",
    )
    result = _dataset("derived-1", role="derived")
    repo.create_join_plan(plan)

    repo.record_join_result_with_audit(
        plan.id,
        result,
        audit={
            "kind": "join.executed",
            "target_ref": plan.id,
            "actor": "system",
            "outcome": "succeeded",
            "detail": {"task_id": plan.task_id, "result_dataset_id": result.id},
        },
    )

    loaded = repo.load_join_plan(plan.id)
    assert loaded.status == "executed"
    assert loaded.result_dataset_id == result.id
    assert repo.get_dataset(result.id) == result
    audit = db_module.PluginRepository(db_path).list_audit(kind="join.executed")[0]
    assert audit["target_ref"] == plan.id
    assert audit["detail"]["result_dataset_id"] == result.id


def test_dataset_repository_rejects_second_join_result_dataset(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec(confirmed=True)],
        status="draft",
    )
    repo.create_join_plan(plan)
    repo.record_join_result_with_audit(
        plan.id,
        _dataset("derived-1", role="derived"),
        audit={
            "kind": "join.executed",
            "target_ref": plan.id,
            "actor": "system",
            "outcome": "succeeded",
            "detail": {"task_id": plan.task_id, "result_dataset_id": "derived-1"},
        },
    )

    with pytest.raises(ConflictError, match="cannot execute again"):
        repo.record_join_result_with_audit(
            plan.id,
            _dataset("derived-2", role="derived"),
            audit={
                "kind": "join.executed",
                "target_ref": plan.id,
                "actor": "system",
                "outcome": "succeeded",
                "detail": {"task_id": plan.task_id, "result_dataset_id": "derived-2"},
            },
        )

    loaded = repo.load_join_plan(plan.id)
    assert loaded.result_dataset_id == "derived-1"
    assert repo.get_dataset("derived-2") is None


def test_dataset_repository_rolls_back_join_result_dataset_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec(confirmed=True)],
        status="draft",
    )
    result = _dataset("derived-1", role="derived")
    repo.create_join_plan(plan)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.record_join_result_with_audit(
            plan.id,
            result,
            audit={
                "kind": "join.executed",
                "target_ref": plan.id,
                "actor": "system",
                "outcome": "succeeded",
                "detail": {"task_id": plan.task_id, "result_dataset_id": result.id},
            },
        )

    loaded = repo.load_join_plan(plan.id)
    assert loaded.status == "draft"
    assert loaded.result_dataset_id is None
    assert repo.get_dataset(result.id) is None


def test_dataset_repository_connection_scoped_join_result_rolls_back_with_transaction(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[_join_spec(confirmed=True)],
        status="draft",
    )
    result = _dataset("derived-1", role="derived")
    repo.create_join_plan(plan)

    with pytest.raises(RuntimeError, match="later write failed"):
        with repo.transaction() as conn:
            repo.record_join_result_with_audit_on_connection(
                conn,
                plan.id,
                result,
                audit={
                    "kind": "join.executed",
                    "target_ref": plan.id,
                    "actor": "system",
                    "outcome": "succeeded",
                    "detail": {"task_id": plan.task_id, "result_dataset_id": result.id},
                },
            )
            raise RuntimeError("later write failed")

    loaded = repo.load_join_plan(plan.id)
    assert loaded.status == "draft"
    assert loaded.result_dataset_id is None
    assert repo.get_dataset(result.id) is None


def test_dataset_registry_registers_csv_and_feather_as_profiled_parquet(tmp_path):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    csv_path = tmp_path / "sample.csv"
    feather_path = tmp_path / "feature.feather"
    pd.DataFrame({
        "mobile": ["13800138000", "13900139000"],
        "bad_flag": [0, 1],
    }).to_csv(csv_path, index=False)
    pd.DataFrame({"acct_num": ["A1", "B2"], "balance": [10, 20]}).to_feather(
        feather_path,
    )

    sample = registry.register_from_upload("task-1", csv_path, role="sample")
    feature = registry.register_from_upload("task-1", feather_path, role="feature")

    assert sample.format == "parquet"
    assert not sample.source_path.startswith("/")
    assert registry.resolve_path(sample.id).exists()
    assert sample.has_target is True
    assert sample.target_col == "bad_flag"
    assert {column.name: column.semantic_role for column in sample.columns} == {
        "mobile": "phone",
        "bad_flag": "target",
    }
    assert feature.format == "parquet"
    assert registry.list_for_task("task-1") == [sample, feature]


def test_join_engine_rolls_back_result_dataset_and_file_when_executed_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    engine = JoinEngine(backend, ColumnAligner(backend), registry, repo)
    anchor_csv = tmp_path / "anchor.csv"
    feature_csv = tmp_path / "feature.csv"
    anchor_csv.write_text("customer_id,bad_flag\nA,0\nB,1\n", encoding="utf-8")
    feature_csv.write_text("customer_id,score\nA,10\nB,20\n", encoding="utf-8")
    anchor = registry.register_from_upload("task-1", anchor_csv, role="sample")
    feature = registry.register_from_upload("task-1", feature_csv, role="feature")
    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    engine.confirm_join_spec(plan.id, feature.id, dedup_strategy=None)

    def fail_join_executed_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "join.executed":
            raise RuntimeError("audit down")
        return original_write_audit(conn, *args, **kwargs)

    original_write_audit = db_module._write_audit_row
    monkeypatch.setattr(db_module, "_write_audit_row", fail_join_executed_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        engine.execute_join_plan(plan.id, out_dir=datasets_root / "task-1" / "joins")

    loaded = repo.load_join_plan(plan.id)
    assert loaded.status == "draft"
    assert loaded.result_dataset_id is None
    assert [dataset.id for dataset in registry.list_for_task("task-1")] == [anchor.id, feature.id]
    assert not list((datasets_root / "task-1" / "joins").glob("*.parquet"))
    assert not (datasets_root / "task-1" / "joins" / ".staging").exists()


def test_join_engine_uses_connection_scoped_artifact_unit_of_work(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    engine = JoinEngine(backend, ColumnAligner(backend), registry, repo)
    anchor_csv = tmp_path / "anchor.csv"
    feature_csv = tmp_path / "feature.csv"
    anchor_csv.write_text("customer_id,bad_flag\nA,0\nB,1\n", encoding="utf-8")
    feature_csv.write_text("customer_id,score\nA,10\nB,20\n", encoding="utf-8")
    anchor = registry.register_from_upload("task-1", anchor_csv, role="sample")
    feature = registry.register_from_upload("task-1", feature_csv, role="feature")
    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    engine.confirm_join_spec(plan.id, feature.id, dedup_strategy=None)

    def fail_old_registration_path(*args, **kwargs):
        raise AssertionError("old join registration path used")

    monkeypatch.setattr(repo, "record_join_result_with_audit", fail_old_registration_path)

    result = engine.execute_join_plan(plan.id, out_dir=datasets_root / "task-1" / "joins")

    loaded = repo.load_join_plan(plan.id)
    assert loaded.status == "executed"
    assert loaded.result_dataset_id == result.id
    assert repo.get_dataset(result.id) == result
    assert registry.resolve_path(result.id).exists()


def test_dataset_registry_register_existing_copies_and_inherits_anchor_target(tmp_path):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    sample_path = tmp_path / "sample.csv"
    external_join = tmp_path / "joined.parquet"
    pd.DataFrame({"mobile": ["13800138000", "13900139000"], "bad_flag": [0, 1]}).to_csv(
        sample_path,
        index=False,
    )
    pd.DataFrame({
        "mobile": ["13800138000", "13900139000"],
        "bad_flag": [0, 1],
        "balance": [10, 20],
    }).to_parquet(external_join, index=False)

    sample = registry.register_from_upload("task-1", sample_path, role="sample")
    derived = registry.register_existing(
        external_join,
        task_id="task-1",
        role="derived",
        anchor_target=sample.id,
    )

    assert derived.role == "derived"
    assert derived.has_target is True
    assert derived.target_col == "bad_flag"
    assert registry.resolve_path(derived.id).exists()
    assert registry.resolve_path(derived.id).is_relative_to(datasets_root)


def test_dataset_registry_register_existing_with_audit_records_lineage(tmp_path):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    source = tmp_path / "source.parquet"
    derived_path = tmp_path / "derived.parquet"
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20]}).to_parquet(source, index=False)
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20], "split": ["train", "test"]}).to_parquet(
        derived_path,
        index=False,
    )
    sample = registry.register_existing(source, task_id="task-1", role="sample")

    derived = registry.register_existing_with_audit(
        derived_path,
        task_id="task-1",
        role="derived",
        anchor_target=sample.id,
        audit_factory=lambda dataset: {
            "kind": "modeling.dataset.derived",
            "target_ref": dataset.id,
            "outcome": "succeeded",
            "detail": {"source_dataset_id": sample.id},
        },
    )

    audits = db_module.PluginRepository(db_path).list_audit(kind="modeling.dataset.derived")
    assert audits[0]["target_ref"] == derived.id
    assert audits[0]["detail"]["source_dataset_id"] == sample.id


def test_dataset_registry_register_existing_on_connection(tmp_path):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    source = tmp_path / "source.parquet"
    derived_path = tmp_path / "derived.parquet"
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20]}).to_parquet(source, index=False)
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20], "split": ["train", "test"]}).to_parquet(
        derived_path,
        index=False,
    )
    sample = registry.register_existing(source, task_id="task-1", role="sample")

    with repo.transaction() as conn:
        derived = registry.register_existing_on_connection(
            conn,
            derived_path,
            task_id="task-1",
            role="derived",
            anchor_target=sample.id,
        )

    assert repo.get_dataset(derived.id) is not None
    assert derived.target_col == "bad_flag"


def test_dataset_registry_register_existing_with_audit_on_connection(tmp_path):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    source = tmp_path / "source.parquet"
    derived_path = tmp_path / "derived.parquet"
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20]}).to_parquet(source, index=False)
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20], "split": ["train", "test"]}).to_parquet(
        derived_path,
        index=False,
    )
    sample = registry.register_existing(source, task_id="task-1", role="sample")

    with repo.transaction() as conn:
        derived = registry.register_existing_with_audit_on_connection(
            conn,
            derived_path,
            task_id="task-1",
            role="derived",
            anchor_target=sample.id,
            audit_factory=lambda dataset: {
                "kind": "modeling.dataset.derived",
                "target_ref": dataset.id,
                "outcome": "succeeded",
                "detail": {"source_dataset_id": sample.id},
            },
        )

    audits = db_module.PluginRepository(db_path).list_audit(kind="modeling.dataset.derived")
    assert repo.get_dataset(derived.id) is not None
    assert audits[0]["target_ref"] == derived.id


def test_dataset_registry_register_existing_with_audit_failure_removes_copied_dataset(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    source = tmp_path / "source.parquet"
    derived_path = tmp_path / "derived.parquet"
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20]}).to_parquet(source, index=False)
    pd.DataFrame({"bad_flag": [0, 1], "x": [10, 20], "split": ["train", "test"]}).to_parquet(
        derived_path,
        index=False,
    )
    sample = registry.register_existing(source, task_id="task-1", role="sample")
    original_write_audit = db_module._write_audit_row

    def fail_dataset_audit(conn, *args, **kwargs):
        if kwargs.get("kind") == "modeling.dataset.derived":
            raise RuntimeError("audit down")
        return original_write_audit(conn, *args, **kwargs)

    monkeypatch.setattr(db_module, "_write_audit_row", fail_dataset_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        registry.register_existing_with_audit(
            derived_path,
            task_id="task-1",
            role="derived",
            anchor_target=sample.id,
            audit_factory=lambda dataset: {
                "kind": "modeling.dataset.derived",
                "target_ref": dataset.id,
                "outcome": "succeeded",
            },
        )

    assert [dataset.id for dataset in registry.list_for_task("task-1")] == [sample.id]
    assert not list((datasets_root / "task-1").glob("derived_*.parquet"))
