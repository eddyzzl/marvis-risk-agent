import hashlib

import pandas as pd
import pytest

from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset, KeyPair
from marvis.data.errors import DataBackendError, DedupRequiredError, FanOutError, JoinNotConfirmedError
from marvis.data.join_engine import JoinEngine
from marvis.data.schema_infer import infer_dataset_schema


class FakeRegistry:
    def __init__(self, root, repo=None):
        self.root = root
        self.datasets = {}
        self.paths = {}
        self._repo = repo

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

    def register_join_result_with_audit(
        self,
        path,
        *,
        join_plan_id: str,
        audit_factory,
        task_id: str,
        role: str,
        anchor_target: str | None = None,
    ) -> Dataset:
        # Mirrors DatasetRegistry.register_join_result_with_audit: register the dataset,
        # then write the join-executed audit through the shared repo (matching how the
        # real registry writes the audit via self._repo.record_join_result_with_audit,
        # not via the registry's own state).
        dataset = self.register_existing(
            path, task_id=task_id, role=role, anchor_target=anchor_target
        )
        audit = audit_factory(dataset)
        self._repo.set_join_plan_executed(join_plan_id, dataset.id)
        self._repo.audits.append(audit)
        return dataset


class FakeJoinRepo:
    def __init__(self):
        self.plans = {}
        self.audits = []

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

    def update_join_spec_with_audit(self, plan_id, spec, *, audit):
        self.update_join_spec(plan_id, spec)
        self.audits.append(audit)

    def set_join_plan_executed(self, plan_id, result_dataset_id):
        plan = self.plans[plan_id]
        plan.status = "executed"
        plan.result_dataset_id = result_dataset_id

    def set_join_plan_executed_with_audit(self, plan_id, result_dataset_id, *, audit):
        self.set_join_plan_executed(plan_id, result_dataset_id)
        self.audits.append(audit)

    def write_audit(self, **kwargs):
        self.audits.append(kwargs)


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
    repo = FakeJoinRepo()
    registry = FakeRegistry(tmp_path, repo)
    engine = JoinEngine(backend, ColumnAligner(backend), registry, repo)
    return engine, registry, repo


def test_join_engine_requires_audit_writer(tmp_path):
    class RepoWithoutAudit:
        pass

    with pytest.raises(TypeError, match="write_audit"):
        JoinEngine(
            DataBackend(tmp_path),
            ColumnAligner(DataBackend(tmp_path)),
            FakeRegistry(tmp_path, FakeJoinRepo()),
            RepoWithoutAudit(),
        )


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
    assert list((tmp_path / "joined").glob("*.parquet")) == [registry.resolve_path(result.id)]
    assert not (tmp_path / "joined" / ".staging").exists()
    assert [audit["kind"] for audit in repo.audits] == ["join.confirmed", "join.executed"]
    assert repo.audits[0]["target_ref"] == plan.id
    assert repo.audits[1]["detail"]["result_dataset_id"] == result.id
    # G4 provenance: the executed-join audit records which feature each column came from
    # (the key column is excluded; only contributed columns are traced).
    assert repo.audits[1]["detail"]["provenance"] == [
        {"feature_dataset_id": feature.id, "columns": ["credit_limit"]}
    ]


def test_join_engine_diagnoses_composite_keys_with_per_pair_match_methods(tmp_path):
    engine, registry, _repo = _engine(tmp_path)
    anchor = _write_dataset(
        registry,
        tmp_path,
        "anchor",
        pd.DataFrame({
            "mobile": ["13800138000", "13900139000"],
            "apply_date": ["20260101", "20260102"],
        }),
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
            "biz_date": ["2026-01-01", "2026-01-02"],
            "limit": [1000, 2000],
        }),
    )
    key_pairs = [
        KeyPair(
            anchor_col="mobile",
            feature_col="phone_md5",
            match_method="hash:md5",
            transform_side="anchor",
            match_rate=1.0,
            resolved_by="test",
        ),
        KeyPair(
            anchor_col="apply_date",
            feature_col="biz_date",
            match_method="date",
            transform_side="both",
            match_rate=1.0,
            resolved_by="test",
        ),
    ]

    diagnostics = engine.diagnose_join(
        anchor,
        registry.resolve_path(anchor.id),
        feature,
        registry.resolve_path(feature.id),
        key_pairs,
        seed=0,
    )

    assert diagnostics.matched_rows == 2
    assert diagnostics.match_rate == 1.0
    assert diagnostics.shrink_detected is False


def test_diagnose_join_proposes_key_relaxation_when_full_key_matches_poorly(tmp_path):
    """Spec §4/§5 动态择键: a composite phone+name key that matches poorly (names differ)
    yields a 'drop 姓名 → phone-only' alternative with the reduced key's re-checked
    match/uniqueness/fan-out. Proposal only — diagnose never swaps the key."""
    engine, registry, _repo = _engine(tmp_path)
    phones = [f"138{i:08d}" for i in range(30)]
    anchor = _write_dataset(
        registry, tmp_path, "anchor",
        pd.DataFrame({"mobile": phones, "姓名": [f"anchor{i}" for i in range(30)]}),
    )
    feature = _write_dataset(
        registry, tmp_path, "feature",
        # same phones (phone-only matches 100%) but DIFFERENT names → composite matches 0%
        pd.DataFrame({"mobile": phones, "姓名": [f"other{i}" for i in range(30)], "val": list(range(30))}),
    )
    key_pairs = [
        KeyPair("mobile", "mobile", "exact", "both", match_rate=1.0, resolved_by="test"),
        KeyPair("姓名", "姓名", "exact", "both", match_rate=0.0, resolved_by="test"),
    ]

    diagnostics = engine.diagnose_join(
        anchor, registry.resolve_path(anchor.id),
        feature, registry.resolve_path(feature.id),
        key_pairs, seed=0,
    )

    assert diagnostics.match_rate < 0.5          # full phone+name key matches poorly
    assert diagnostics.key_alternatives          # a relaxation was proposed
    best = diagnostics.key_alternatives[0]
    assert best.dropped == "姓名"
    assert best.key_pairs == (("mobile", "mobile"),)
    assert best.match_rate > diagnostics.match_rate   # relaxation improves the match
    assert best.feature_key_unique is True
    assert best.fan_out_detected is False
    # the full key is NOT silently changed — diagnose still reports the original key
    assert [p.feature_col for p in key_pairs] == ["mobile", "姓名"]


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
    # G1b: diagnose attaches a two-level dedup breakdown — A1 has disagreeing balance
    # (10 vs 11), a same-key value CONFLICT (not a safe duplicate).
    report = spec.diagnostics.conflict_report
    assert report is not None
    assert report.n_conflict_keys == 1
    assert "balance" in report.conflict_columns
    assert report.safe_dropped == 0
    with pytest.raises(JoinNotConfirmedError):
        engine.execute_join_plan(plan.id, out_dir=tmp_path / "joined")
    with pytest.raises(DedupRequiredError):
        engine.confirm_join_spec(plan.id, feature.id, dedup_strategy=None)
    with pytest.raises(DataBackendError, match="unsupported dedup_strategy"):
        engine.confirm_join_spec(plan.id, feature.id, dedup_strategy="drop_all")
    assert spec.confirmed is False

    engine.confirm_join_spec(plan.id, feature.id, dedup_strategy="first")
    result = engine.execute_join_plan(plan.id, out_dir=tmp_path / "joined")

    joined = pd.read_parquet(registry.resolve_path(result.id))
    assert result.row_count == anchor.row_count
    assert joined["balance"].tolist() == [10, 20]


def test_join_engine_large_feature_table_uses_bounded_conflict_report(tmp_path, monkeypatch):
    import marvis.data.join_engine as join_engine_module

    monkeypatch.setattr(join_engine_module, "LARGE_ROW_THRESHOLD", 2)
    engine, registry, _repo = _engine(tmp_path)
    anchor = _write_dataset(
        registry,
        tmp_path,
        "anchor",
        pd.DataFrame({"acct_num": ["A1", "B2", "C3"]}),
    )
    feature = _write_dataset(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({
            "acct_no": ["A1", "A1", "B2", "B2", "C3"],
            "balance": [10, 11, 20, 20, 30],
        }),
    )

    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    report = plan.joins[0].diagnostics.conflict_report

    assert report is not None
    assert report.n_conflict_keys == 2
    assert report.n_conflict_rows == 4
    assert "balance" in report.conflict_columns
    assert ("A1",) in report.sample_keys


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
