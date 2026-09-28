"""Native ordinary JOIN membership and conservative temporal inheritance."""
from dataclasses import replace
import json
import uuid

import pandas as pd
import pytest

from marvis.data.align import ColumnAligner
from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.contracts import JoinPlan, JoinSpec, KeyPair
from marvis.data.errors import DataBackendError, DatasetContentDriftError, DedupRequiredError, FanOutError
from marvis.data.feature_time import feature_time_evidence
from marvis.data.join_engine import JoinEngine
from marvis.data.join_evidence import KIND
from marvis.feature.errors import FeatureError
from marvis.plugins.manifest import ToolRef
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_feature_time_evidence import scenario as _pit_scenario


@pytest.fixture
def scenario(tmp_path):
    return _pit_scenario.__wrapped__(tmp_path)


def register(registry, frame, name):
    path = registry.datasets_root / f"{name}-{uuid.uuid4().hex}.parquet"
    frame.to_parquet(path, index=False)
    with registry.transaction() as conn:
        return registry.register_existing_on_connection(conn, path, task_id="task-feature", role="feature")


def plan_for(registry, backend, anchor, features, dedup=None):
    repo = registry._repo
    engine = JoinEngine(backend, ColumnAligner(backend), registry, repo)
    specs = []
    for feature in features:
        pairs = [KeyPair("subject", "subject", "exact", "none", 1.0, "user")]
        diagnostics = engine.diagnose_join(
            anchor, registry.resolve_path(anchor.id), feature,
            registry.resolve_path(feature.id), pairs, seed=42,
        )
        specs.append(JoinSpec(feature.id, pairs, diagnostics, dedup, True))
    plan = JoinPlan("join-" + uuid.uuid4().hex, anchor.task_id, anchor.id, specs, "draft")
    repo.create_join_plan(plan)
    return engine, plan


def run_join(registry, backend, anchor, features, dedup=None):
    engine, plan = plan_for(registry, backend, anchor, features, dedup)
    result = engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / anchor.task_id / "joins")
    records = TaskArtifactRepository(registry._repo.db_path).list_for_task(anchor.task_id)
    proof = next(r for r in records if r["kind"] == KIND and r["provenance"]["output_dataset_id"] == result.id)
    payload = json.loads((registry.datasets_root.parent / proof["path"]).read_text())
    return result, payload, proof


def test_native_pit_left_survives_two_actual_joins_collision_and_unmatched(scenario):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({
        "subject": ["s2", "s0"], "asof__amount": [999.0, 777.0],
    }), "right")
    before = [registry.get(native.dataset.id), registry.get(feature.id)]
    result, payload, record = run_join(registry, backend, native.dataset, [feature, feature])
    evidence = feature_time_evidence(
        registry, result.id, ["asof__amount", "feature_asof__amount", "feature_asof__amount_2"],
    )
    assert evidence["fields"]["asof__amount"]["assurance"] == "verified"
    assert evidence["fields"]["feature_asof__amount"]["assurance"] == "unknown"
    assert evidence["fields"]["feature_asof__amount_2"]["assurance"] == "unknown"
    assert evidence["assurance"] == "unknown"
    assert record["id"] in evidence["artifact_ids"]
    assert registry.get(native.dataset.id) == before[0]
    assert registry.get(feature.id) == before[1]
    rows = pd.read_parquet(registry.datasets_root.parent / payload["steps"][0]["member_path"])
    assert rows.output_row.tolist() == list(range(40))
    assert rows.left_row.tolist() == list(range(40))
    assert rows.right_rows.iloc[0].tolist() == [1]
    assert rows.right_rows.iloc[2].tolist() == [0]
    assert rows.right_rows.iloc[1] is None
    engine = AsOfJoinEngine(registry, TaskArtifactRepository(registry._repo.db_path), workspace_root=registry.datasets_root.parent)
    assert engine.dataset_time_status(result.id).assurance == "unknown"
    assert "s0" not in json.dumps(payload)


@pytest.mark.parametrize(("dedup", "expected", "members"), [
    ("first", 10.0, [0]), ("last", 30.0, [1]),
    ("agg_mean", 20.0, [0, 1]), ("agg_max", 30.0, [0, 1]),
])
def test_native_duplicate_policy_captures_actual_members(scenario, dedup, expected, members):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0", "s0"], "amount": [10.0, 30.0]}), "duplicates")
    result, proof, _ = run_join(registry, backend, native.dataset, [feature], dedup)
    assert pd.read_parquet(registry.resolve_path(result.id)).amount.iloc[0] == expected
    rows = pd.read_parquet(registry.datasets_root.parent / proof["steps"][0]["member_path"])
    assert rows.right_rows.iloc[0].tolist() == members
    assert feature_time_evidence(registry, result.id, ["asof__amount"])["assurance"] == "verified"


def test_unreviewed_fanout_rejected_without_artifact(scenario):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0", "s0"], "amount": [1, 2]}), "duplicate")
    engine, plan = plan_for(registry, backend, native.dataset, [feature])
    with pytest.raises(DedupRequiredError):
        engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / "task-feature" / "joins")
    plan.joins[0].diagnostics = replace(plan.joins[0].diagnostics, feature_key_unique=True)
    registry._repo.update_join_spec(plan.id, plan.joins[0])
    with pytest.raises(FanOutError):
        engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / "task-feature" / "joins")
    assert not [r for r in TaskArtifactRepository(registry._repo.db_path).list_for_task("task-feature") if r["kind"] == KIND]


def test_native_proof_and_membership_tamper_fail_closed(scenario):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0"], "amount": [1]}), "right")
    result, proof, artifact = run_join(registry, backend, native.dataset, [feature])
    member = registry.datasets_root.parent / proof["steps"][0]["member_path"]
    original = member.read_bytes()
    member.write_bytes(original + b"changed")
    with pytest.raises(FeatureError, match="artifact bytes changed"):
        feature_time_evidence(registry, result.id, ["asof__amount"])
    member.write_bytes(original)
    path = registry.datasets_root.parent / artifact["path"]
    path.write_text("{}")
    with pytest.raises(FeatureError, match="artifact bytes changed"):
        feature_time_evidence(registry, result.id, ["asof__amount"])


def test_legacy_join_no_native_artifact_stays_unknown(scenario):
    _, registry, _, native = scenario
    copied = register(registry, pd.read_parquet(registry.resolve_path(native.dataset.id)), "legacy")
    assert feature_time_evidence(registry, copied.id, ["asof__amount"])["assurance"] == "unknown"


def test_real_toolrunner_produces_receipt(scenario):
    runner, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0"], "amount": [1]}), "runner")
    _, plan = plan_for(registry, backend, native.dataset, [feature])
    result = runner.invoke(ToolRef("data_ops", "execute_join"), {"join_plan_id": plan.id}, task_id="task-feature")
    assert result.ok is True, result.error
    assert feature_time_evidence(registry, result.output["result_dataset_id"], ["asof__amount"])["assurance"] == "verified"


def test_large_integer_identity_and_real_file_row_number_are_preserved(scenario):
    _, registry, backend, _ = scenario
    ids = [2**53 + 3, 2**53 + 1, 2**53 + 2]
    anchor = register(registry, pd.DataFrame({
        "subject": ids, "file_row_number": [20, 10, 30],
        "x": [103, 101, 102], "feature_x": [1, 2, 3],
        "__marvis_source_row": [90, 80, 70],
    }), "large-anchor")
    feature = register(registry, pd.DataFrame({
        "subject": [ids[1], ids[0]], "file_row_number": [99, 88], "x": [201, 203],
    }), "large-right")
    result, proof, _ = run_join(registry, backend, anchor, [feature])
    frame = pd.read_parquet(registry.resolve_path(result.id))
    assert frame.subject.tolist() == ids
    assert frame.x.tolist() == [103, 101, 102]
    assert frame.file_row_number.tolist() == [20, 10, 30]
    assert frame.feature_x_2.iloc[:2].tolist() == [203, 201]
    assert pd.isna(frame.feature_x_2.iloc[2])
    members = pd.read_parquet(registry.datasets_root.parent / proof["steps"][0]["member_path"])
    assert members.right_rows.iloc[0].tolist() == [1]
    assert members.right_rows.iloc[1].tolist() == [0]
    assert feature_time_evidence(registry, result.id, ["x", "feature_x_2"])["assurance"] == "unknown"


@pytest.mark.parametrize(("dedup", "expected_row"), [("first", 1), ("last", 0)])
def test_real_row_number_column_retains_existing_dedup_order(scenario, dedup, expected_row):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({
        "subject": ["s0", "s0"], "file_row_number": [1, 2], "amount": [30, 10],
    }), "colliding-physical")
    result, proof, _ = run_join(registry, backend, native.dataset, [feature], dedup)
    # Existing collision fallback sorts full original content, beginning amount.
    expected = 10 if dedup == "first" else 30
    assert pd.read_parquet(registry.resolve_path(result.id)).amount.iloc[0] == expected
    members = pd.read_parquet(registry.datasets_root.parent / proof["steps"][0]["member_path"])
    assert members.right_rows.iloc[0].tolist() == [expected_row]


def test_even_certified_right_fields_cannot_borrow_other_decision_timing(scenario):
    _, registry, backend, native = scenario
    result, _, _ = run_join(registry, backend, native.dataset, [native.dataset])
    evidence = feature_time_evidence(registry, result.id, ["asof__amount", "feature_asof__amount"])
    assert evidence["fields"]["asof__amount"]["assurance"] == "verified"
    assert evidence["fields"]["feature_asof__amount"]["assurance"] == "unknown"


def test_changed_left_values_are_rejected_before_publication(scenario, monkeypatch):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0"], "amount": [1]}), "corrupting")
    original = backend.left_join

    def corrupt(*args, **kwargs):
        count = original(*args, **kwargs)
        path = kwargs["out_path"]
        frame = pd.read_parquet(path)
        frame.loc[[0, 1], "asof__amount"] = frame.loc[[1, 0], "asof__amount"].to_numpy()
        frame.to_parquet(path, index=False)
        return count

    monkeypatch.setattr(backend, "left_join", corrupt)
    before = registry.list_for_task("task-feature")
    engine, plan = plan_for(registry, backend, native.dataset, [feature])
    with pytest.raises(FeatureError, match="copied left values"):
        engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / "task-feature" / "joins")
    assert registry.list_for_task("task-feature") == before
    assert registry._repo.load_join_plan(plan.id).status == "draft"
    assert not list((registry.datasets_root / "task-feature" / "joins").rglob("*.parquet"))
    assert not list((registry.datasets_root / "task-feature" / ".join").rglob("*.parquet"))
    assert not list((registry.datasets_root / "task-feature" / ".join").rglob("*.json"))


def test_dataset_audit_and_artifacts_roll_back_together(scenario, monkeypatch):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0"], "amount": [1]}), "rollback")
    before = registry.list_for_task("task-feature")
    repo = TaskArtifactRepository(registry._repo.db_path)
    records_before = repo.list_for_task("task-feature")
    original = TaskArtifactRepository.register_on_connection

    def fail(self, conn, **kwargs):
        if kwargs["kind"] == KIND:
            raise RuntimeError("synthetic artifact commit failure")
        return original(self, conn, **kwargs)

    monkeypatch.setattr(TaskArtifactRepository, "register_on_connection", fail)
    engine, plan = plan_for(registry, backend, native.dataset, [feature])
    with pytest.raises(RuntimeError, match="synthetic artifact"):
        engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / "task-feature" / "joins")
    assert registry.list_for_task("task-feature") == before
    assert repo.list_for_task("task-feature") == records_before
    assert registry._repo.load_join_plan(plan.id).status == "draft"
    assert not list((registry.datasets_root / "task-feature" / "joins").rglob("*.parquet"))
    assert not list((registry.datasets_root / "task-feature" / ".join").rglob("*.parquet"))
    assert not list((registry.datasets_root / "task-feature" / ".join").rglob("*.json"))


def test_plan_changed_during_join_cannot_commit_previous_review(scenario, monkeypatch):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0"], "amount": [1]}), "plan-race")
    engine, plan = plan_for(registry, backend, native.dataset, [feature])
    original = backend.left_join

    def change(*args, **kwargs):
        count = original(*args, **kwargs)
        changed = registry._repo.load_join_plan(plan.id).joins[0]
        changed.confirmed = False
        registry._repo.update_join_spec(plan.id, changed)
        return count

    monkeypatch.setattr(backend, "left_join", change)
    with pytest.raises(DataBackendError, match="reviewed contract changed"):
        engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / "task-feature" / "joins")
    assert registry._repo.load_join_plan(plan.id).result_dataset_id is None


def test_source_changed_after_execution_does_not_receive_native_receipt(scenario, monkeypatch):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0"], "amount": [1]}), "source-race")
    engine, plan = plan_for(registry, backend, native.dataset, [feature])
    original = backend.left_join

    def change(*args, **kwargs):
        count = original(*args, **kwargs)
        pd.DataFrame({"subject": ["s0"], "amount": [9]}).to_parquet(registry.resolve_path(feature.id), index=False)
        return count

    monkeypatch.setattr(backend, "left_join", change)
    with pytest.raises(DatasetContentDriftError, match="content|bytes"):
        engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / "task-feature" / "joins")
    assert registry._repo.load_join_plan(plan.id).result_dataset_id is None


def test_empty_anchor_and_feature_keep_empty_membership_and_unknown(scenario):
    _, registry, backend, _ = scenario
    anchor = register(registry, pd.DataFrame({"subject": pd.Series(dtype="int64"), "value": pd.Series(dtype="float64")}), "empty-left")
    feature = register(registry, pd.DataFrame({"subject": pd.Series(dtype="int64"), "amount": pd.Series(dtype="float64")}), "empty-right")
    result, proof, _ = run_join(registry, backend, anchor, [feature])
    assert result.row_count == 0
    assert pd.read_parquet(registry.datasets_root.parent / proof["steps"][0]["member_path"]).empty
    assert feature_time_evidence(registry, result.id, ["value", "amount"])["assurance"] == "unknown"


def test_false_row_ordinal_types_are_rejected_before_publication(scenario, monkeypatch):
    _, registry, backend, native = scenario
    feature = register(registry, pd.DataFrame({"subject": ["s0"], "amount": [1]}), "fractional")
    engine, plan = plan_for(registry, backend, native.dataset, [feature])
    original = backend.left_join

    def corrupt(*args, **kwargs):
        count = original(*args, **kwargs)
        path = kwargs["membership_path"]
        members = pd.read_parquet(path)
        members["left_row"] = members.left_row.astype(float) + 0.5
        members.to_parquet(path, index=False)
        return count

    monkeypatch.setattr(backend, "left_join", corrupt)
    with pytest.raises(FeatureError, match="membership schema changed"):
        engine.execute_join_plan(plan.id, out_dir=registry.datasets_root / "task-feature" / "joins")
    assert registry._repo.load_join_plan(plan.id).result_dataset_id is None


@pytest.mark.parametrize(("left_keys", "right_keys", "method", "dedup"), [
    (["007", "7", "x", "42"], ["7", "007", "42", "x"], "exact", None),
    ([7, 42, 2**53 + 3], ["42", "9007199254740995", "7"], "exact", None),
    ([7.0, 42.0, None], ["42", "7", "unused"], "exact", None),
    (["ABC", "def", "miss"], ["abc", "ABC", "DEF"], "exact_lower", "first"),
    (["ABC", "def", "miss"], ["abc", "ABC", "DEF"], "exact_lower", "agg_mean"),
])
def test_native_parquet_reader_and_transformed_dedup_match_existing_backend(
    scenario, left_keys, right_keys, method, dedup,
):
    _, registry, backend, _ = scenario
    left = register(registry, pd.DataFrame({"key": left_keys, "row": list(range(len(left_keys)))}), "reader-left")
    right = register(registry, pd.DataFrame({"key": right_keys, "payload": list(range(len(right_keys)))}), "reader-right")
    pairs = [KeyPair("key", "key", method, "both" if method != "exact" else "none", 1.0, "user")]
    directory = registry.datasets_root / "reader-comparison"
    directory.mkdir()
    legacy, native, member = [directory / name for name in ("legacy.parquet", "native.parquet", "members.parquet")]
    for path, membership in ((legacy, None), (native, member)):
        backend.left_join(
            registry.resolve_path(left.id), registry.resolve_path(right.id), pairs,
            dedup_strategy=dedup, out_path=path, membership_path=membership,
        )
    pd.testing.assert_frame_equal(
        pd.read_parquet(legacy).sort_values("row").reset_index(drop=True),
        pd.read_parquet(native).sort_values("row").reset_index(drop=True),
    )
    assert pd.read_parquet(member).left_row.tolist() == list(range(len(left_keys)))
