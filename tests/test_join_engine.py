import hashlib

import pandas as pd
import pytest

from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset
from marvis.data.errors import DedupRequiredError, FanOutError, JoinNotConfirmedError
from marvis.data.join_engine import JoinEngine
from marvis.data.schema_infer import infer_dataset_schema


class FakeRegistry:
    def __init__(self, root):
        self.root = root
        self.datasets = {}
        self.paths = {}

    def add(self, dataset: Dataset, path):
        self.datasets[dataset.id] = dataset
        self.paths[dataset.id] = path

    def get(self, dataset_id: str) -> Dataset:
        return self.datasets[dataset_id]

    def resolve_path(self, dataset_id: str):
        return self.paths[dataset_id]

    def register_existing(self, path, *, task_id: str, role: str, anchor_target: str | None):
        frame = pd.read_parquet(path)
        dataset = Dataset(
            id=f"derived-{len(self.datasets)}",
            task_id=task_id,
            role=role,
            source_path=path.name,
            format="parquet",
            sheet=None,
            row_count=len(frame),
            columns=tuple(infer_dataset_schema(frame)),
            has_target=False,
            target_col=None,
            created_at="2026-06-19T00:00:00Z",
        )
        self.add(dataset, path)
        return dataset


class FakeJoinRepo:
    def __init__(self):
        self.plans = {}

    def create_join_plan(self, plan):
        self.plans[plan.id] = plan

    def load_join_plan(self, plan_id):
        return self.plans[plan_id]

    def update_join_spec(self, plan_id, spec):
        plan = self.plans[plan_id]
        plan.joins = [
            spec if item.feature_dataset_id == spec.feature_dataset_id else item
            for item in plan.joins
        ]

    def set_join_plan_executed(self, plan_id, result_dataset_id):
        plan = self.plans[plan_id]
        plan.status = "executed"
        plan.result_dataset_id = result_dataset_id


def _dataset(dataset_id: str, frame: pd.DataFrame, source_path: str) -> Dataset:
    return Dataset(
        id=dataset_id,
        task_id="task-1",
        role="sample",
        source_path=source_path,
        format="csv",
        sheet=None,
        row_count=len(frame),
        columns=tuple(infer_dataset_schema(frame)),
        has_target=False,
        target_col=None,
        created_at="2026-06-19T00:00:00Z",
    )


def _write_dataset(registry: FakeRegistry, tmp_path, dataset_id: str, frame: pd.DataFrame):
    path = tmp_path / f"{dataset_id}.csv"
    frame.to_csv(path, index=False)
    dataset = _dataset(dataset_id, frame, path.name)
    registry.add(dataset, path)
    return dataset


def _engine(tmp_path):
    backend = DataBackend(tmp_path)
    registry = FakeRegistry(tmp_path)
    repo = FakeJoinRepo()
    engine = JoinEngine(backend, ColumnAligner(backend), registry, repo)
    return engine, registry, repo


def test_join_engine_proposes_confirms_and_executes_unique_hash_join(tmp_path):
    engine, registry, repo = _engine(tmp_path)
    anchor = _write_dataset(
        registry,
        tmp_path,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"], "score": [0.1, 0.2]}),
    )
    feature = _write_dataset(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "credit_limit": [1000, 2000],
        }),
    )

    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    spec = plan.joins[0]

    assert plan.status == "draft"
    assert spec.confirmed is False
    assert spec.key_pairs[0].match_method == "hash:md5"
    assert spec.diagnostics.feature_key_unique is True
    assert spec.diagnostics.fan_out_detected is False
    assert spec.diagnostics.joined_rows_preview == anchor.row_count
    assert repo.load_join_plan(plan.id) is plan

    engine.confirm_join_spec(plan.id, feature.id, dedup_strategy=None)
    result = engine.execute_join_plan(plan.id, out_dir=tmp_path / "joined")

    assert result.row_count == anchor.row_count
    assert repo.load_join_plan(plan.id).status == "executed"
    joined = pd.read_parquet(registry.resolve_path(result.id))
    assert joined["credit_limit"].tolist() == [1000, 2000]


def test_join_engine_blocks_unconfirmed_and_requires_dedup_for_duplicate_keys(tmp_path):
    engine, registry, _repo = _engine(tmp_path)
    anchor = _write_dataset(
        registry,
        tmp_path,
        "anchor",
        pd.DataFrame({"acct_num": ["A1", "B2"]}),
    )
    feature = _write_dataset(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({"acct_no": ["A1", "A1", "B2"], "balance": [10, 11, 20]}),
    )

    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    spec = plan.joins[0]

    assert spec.diagnostics.feature_key_unique is False
    assert spec.diagnostics.fan_out_detected is True
    with pytest.raises(JoinNotConfirmedError):
        engine.execute_join_plan(plan.id, out_dir=tmp_path / "joined")
    with pytest.raises(DedupRequiredError):
        engine.confirm_join_spec(plan.id, feature.id, dedup_strategy=None)

    engine.confirm_join_spec(plan.id, feature.id, dedup_strategy="first")
    result = engine.execute_join_plan(plan.id, out_dir=tmp_path / "joined")

    joined = pd.read_parquet(registry.resolve_path(result.id))
    assert result.row_count == anchor.row_count
    assert joined["balance"].tolist() == [10, 20]


def test_join_engine_marks_shrink_when_no_key_pair_matches(tmp_path):
    engine, registry, _repo = _engine(tmp_path)
    anchor = _write_dataset(
        registry,
        tmp_path,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000", "13700137000"]}),
    )
    feature = _write_dataset(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13600136000", "13500135000"]
            ],
        }),
    )

    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    diagnostics = plan.joins[0].diagnostics

    assert plan.joins[0].key_pairs == []
    assert diagnostics.shrink_detected is True
    assert diagnostics.match_rate == 0.0
    assert diagnostics.new_columns_null_rate == 1.0


def test_join_engine_has_final_fanout_defense_when_diagnostics_are_wrong(tmp_path):
    engine, registry, _repo = _engine(tmp_path)
    anchor = _write_dataset(
        registry,
        tmp_path,
        "anchor",
        pd.DataFrame({"acct_num": ["A1", "B2"]}),
    )
    feature = _write_dataset(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({"acct_no": ["A1", "A1", "B2"], "balance": [10, 11, 20]}),
    )
    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    spec = plan.joins[0]
    spec.confirmed = True
    spec.diagnostics.feature_key_unique = True
    spec.dedup_strategy = None

    with pytest.raises(FanOutError):
        engine.execute_join_plan(plan.id, out_dir=tmp_path / "joined")
