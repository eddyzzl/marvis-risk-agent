import pandas as pd

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
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, init_db


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
