from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from filelock import FileLock
import pandas as pd
import pytest

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import DatasetRepository, ModelingRepository, TaskRepository, init_db
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.packs.modeling import evidence_tools
from marvis.packs.modeling import tools as modeling_tools
from marvis.packs.modeling.evidence import (
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
    MODEL_BINARY_REF_KIND,
    NON_FINITE_BOUNDARY_TAG,
    decode_modeling_scoring_woe_maps_boundaries,
)
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.sample_design_tools import run_materialize_sample_design
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    run_materialize_sample_design_v2,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _fixture(
    tmp_path: Path,
    *,
    target_bad_value: int = 1,
    missing_train_label: bool = False,
    drop_nan_labels: bool = False,
    approval_model_splits: tuple[object, object, object] = (
        "approval_only",
        "approval_only",
        "approval_only",
    ),
    training_split_col: str = "model_split",
    bad_weight_split: str | None = None,
    bad_weight_value: object = None,
) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="governed-training-v2",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    rows: list[dict] = []
    ordinal = 0
    partitions = (
        ("development", "train", "202601"),
        ("validation", "test", "202602"),
        ("oot", "oot", "202603"),
    )
    for (partition, model_split, month), approval_model_split in zip(
        partitions,
        approval_model_splits,
        strict=True,
    ):
        for index in range(6):
            bad = index % 2
            rows.append(
                {
                    "sample_partition": partition,
                    "model_split": model_split,
                    "evidence_split": model_split,
                    "risk_flag": 1,
                    "apply_date": f"2026-{int(month[-2:]):02d}-{index + 1:02d}",
                    "apply_month": month,
                    "customer_id": f"risk-{ordinal}",
                    "channel": "app" if index % 2 == 0 else "web",
                    "legacy_score": 100.0 + ordinal * 3,
                    "x1": float(index + ordinal / 10),
                    "x2": float((index * 7 + ordinal) % 13),
                    "weight": 1.0 + (index % 2) * 0.1,
                    "loan_amount": 1000.0 + ordinal * 10,
                    "overdue_amount": 50.0 if bad else 0.0,
                    "bad": bad,
                }
            )
            ordinal += 1
        # Most tests use a non-modeling value; selector regressions can give
        # approval-only rows a real split value or NULL to prove the selector
        # is always intersected with the governed risk population.
        rows.append(
            {
                "sample_partition": partition,
                "model_split": "approval_only",
                "evidence_split": approval_model_split,
                "risk_flag": 0,
                "apply_date": f"2026-{int(month[-2:]):02d}-20",
                "apply_month": month,
                "customer_id": f"approval-{partition}",
                "channel": "branch",
                "legacy_score": 300.0 + ordinal,
                "x1": 999.0,
                "x2": 999.0,
                "weight": 1.0,
                "loan_amount": 500.0,
                "overdue_amount": 0.0,
                "bad": 0,
            }
        )
        ordinal += 1
    frame = pd.DataFrame(rows)
    if bad_weight_split is not None:
        split_value = {"train": "train", "test": "test", "oot": "oot"}[
            bad_weight_split
        ]
        row_index = frame.index[
            (frame["risk_flag"] == 1) & (frame["model_split"] == split_value)
        ][0]
        frame.loc[row_index, "weight"] = bad_weight_value
    if any(value is None or value is pd.NA for value in approval_model_splits):
        frame["evidence_split"] = frame["evidence_split"].astype("string")
    if missing_train_label:
        first_train = frame.index[
            (frame["risk_flag"] == 1) & (frame["model_split"] == "train")
        ][0]
        frame.loc[first_train, "bad"] = None
    if target_bad_value == 0:
        frame["bad"] = frame["bad"].map({0: 1, 1: 0})
    source = tmp_path / "governed_training.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    workspaces = DataWorkspaceRepository(settings.db_path)
    activated = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "sample_partition": "segment",
            "model_split": "segment",
            "evidence_split": "segment",
            "risk_flag": "categorical",
            "apply_date": "date",
            "apply_month": "month",
            "customer_id": "id",
            "channel": "categorical",
            "bad": "target",
        },
    )
    workspace = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    strategy_runtime = strategy_tools._runtime(ctx)
    legacy = run_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": strategy_tools.data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": target_bad_value,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-04-30",
            "maturity_status": "confirmed_matured",
            "split_col": "model_split",
            "development_values": ["train"],
            "validation_values": ["test", "approval_only"],
            "oot_values": ["oot"],
            "month_col": "apply_month",
            "weight_col": "weight",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": drop_nan_labels,
        },
        ctx,
        strategy_runtime,
    )
    legacy_ref = {
        "artifact_id": legacy["artifact"]["artifact_id"],
        "artifact_content_hash": legacy["artifact"]["content_hash"],
        "sample_design_id": legacy["sample_design_id"],
        "sample_design_content_hash": legacy["content_hash"],
        "partition": "development",
    }
    sample_request = {
        "legacy_sample_design_ref": legacy_ref,
        "relationship": "nested_same_cohort",
        "scope": "strategy_development",
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {
            "inclusion": _eq("risk_flag", 1),
            "exclusion": None,
        },
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": _eq("sample_partition", "development"),
                "validation": _eq("sample_partition", "validation"),
                "oot": _eq("sample_partition", "oot"),
            },
        },
        "maturity": {
            "status": "confirmed_matured",
            "performance_window_days": 30,
            "cutoff_date": "2026-04-30",
            "reason": None,
        },
        "performance_window": {"status": "provided", "days": 30},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-04-30",
        },
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": "apply_date",
            "group_field": "channel",
            "month_field": "apply_month",
            "weight_field": "weight",
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "historical_score": {
            "status": "available",
            "column": "legacy_score",
            "direction": "higher_is_riskier",
            "reason": None,
        },
        "policy": {
            "minimum_partition_count": 1,
            "minimum_bad_count": 1,
            "minimum_label_coverage": 1.0,
            "minimum_historical_score_coverage": 1.0,
            "maximum_group_coverage_gap": 1.0,
            "diagnostic_severities": {
                "entity_overlap": "fail",
                "temporal_oot": "fail",
                "risk_outside_approval": "fail",
                "maturity": "fail",
                "label_coverage": "fail",
                "historical_score_coverage": "warn",
                "group_coverage_gap": "warn",
                "sufficiency": "fail",
            },
        },
    }
    sample = run_materialize_sample_design_v2(
        sample_request,
        ctx,
        strategy_runtime,
    )
    sample_records = TaskArtifactRepository(settings.db_path).list_for_task(task.id)
    membership_record = next(
        record
        for record in sample_records
        if record["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle_record = next(
        record
        for record in sample_records
        if record["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    )
    sample_ref = {
        "membership_artifact_id": membership_record["id"],
        "expected_membership_artifact_content_hash": membership_record["content_hash"],
        "bundle_artifact_id": bundle_record["id"],
        "expected_bundle_artifact_content_hash": bundle_record["content_hash"],
        "expected_bundle_id": sample["bundle_id"],
        "expected_sample_design_id": sample["sample_design_id"],
        "expected_sample_design_content_hash": sample["sample_design_content_hash"],
    }
    inputs = {
        "sample_design_ref": sample_ref,
        "recipe": "lr",
        "features": ["x1", "x2"],
        "split_col": training_split_col,
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "params": {"max_iter": 200, "sample_weight_col": "weight"},
        "seed": 23,
        "early_stopping_rounds": None,
    }
    return {
        "settings": settings,
        "task": task,
        "dataset": dataset,
        "workspace": workspace,
        "ctx": ctx,
        "runtime": modeling_tools._runtime(ctx),
        "sample": sample,
        "sample_ref": sample_ref,
        "inputs": inputs,
    }


def _run(fx: dict) -> dict:
    return evidence_tools.run_train_model_with_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )


def _governed_records(fx: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"]
        in {MODEL_BINARY_REF_KIND, MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND}
    ]


def test_lr_success_publishes_exact_pair_and_risk_subset(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)

    output = _run(fx)

    assert output["governance"] == {
        "not_selected": True,
        "not_calibrated": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    assert len(_governed_records(fx)) == 2
    assert evidence_tools.validate_train_model_with_evidence_v2_tool_output(
        output,
        runtime=fx["runtime"],
        task_id=fx["task"].id,
    ) == output
    binding = evidence_tools.load_modeling_training_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        sample_design_ref=fx["sample_ref"],
        model_binary_artifact_id=output["artifacts"]["model_binary"]["artifact_id"],
        expected_model_binary_artifact_content_hash=output["artifacts"][
            "model_binary"
        ]["content_hash"],
        evidence_artifact_id=output["artifacts"]["training_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"][
            "training_evidence"
        ]["content_hash"],
        expected_experiment_id=output["experiment_id"],
        expected_model_artifact_id=output["model_artifact_id"],
        expected_evidence_id=output["evidence_id"],
        expected_evidence_content_hash=output["evidence_content_hash"],
    )
    assert binding.model_binary_record["content_hash"] == sha256_file(
        binding.model_binary_path
    )
    assert binding.evidence_record["content_hash"] == sha256_file(
        binding.evidence_path
    )
    assert binding.evidence["content_hash"] == output["evidence_content_hash"]
    assert binding.evidence["model_artifact"]["model_binary_ref"]["content_hash"] == (
        binding.model_binary_record["content_hash"]
    )
    snapshot_hashes = {
        "scoring_metadata_hash": binding.evidence["model_artifact"][
            "scoring_metadata_hash"
        ],
        "train_config_hash": binding.evidence["training_contract"][
            "train_config_hash"
        ],
        "metrics_snapshot_content_hash": binding.evidence["metrics_snapshot"][
            "content_hash"
        ],
    }
    assert {
        key: binding.model_binary_record["provenance"][key]
        for key in snapshot_hashes
    } == snapshot_hashes
    assert {
        key: binding.evidence_record["provenance"][key]
        for key in snapshot_hashes
    } == snapshot_hashes
    latest_meta = (
        Path(fx["settings"].tasks_dir)
        / fx["task"].id
        / "modeling_artifacts"
        / "model_meta.json"
    )
    assert json.loads(latest_meta.read_text(encoding="utf-8"))["artifact_id"] == (
        output["model_artifact_id"]
    )
    # 21 approval rows exist, but only the 18 governed risk rows (six per split)
    # reached training and baseline scoring.
    baseline = binding.model_artifact.baseline_distributions
    assert baseline is not None
    assert {
        name: row["sample_count"]
        for name, row in baseline["score_distribution"].items()
    } == {"train": 6, "test": 6, "oot": 6}


def test_selector_ignores_approval_split_values_and_nullable_split(
    tmp_path: Path,
) -> None:
    fx = _fixture(
        tmp_path,
        approval_model_splits=("train", None, "oot"),
        training_split_col="evidence_split",
    )

    output = _run(fx)
    binding = evidence_tools.load_modeling_training_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        sample_design_ref=fx["sample_ref"],
        model_binary_artifact_id=output["artifacts"]["model_binary"]["artifact_id"],
        expected_model_binary_artifact_content_hash=output["artifacts"][
            "model_binary"
        ]["content_hash"],
        evidence_artifact_id=output["artifacts"]["training_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"][
            "training_evidence"
        ]["content_hash"],
        expected_experiment_id=output["experiment_id"],
        expected_model_artifact_id=output["model_artifact_id"],
        expected_evidence_id=output["evidence_id"],
        expected_evidence_content_hash=output["evidence_content_hash"],
    )

    assert {
        proof["name"]: proof["row_count"]
        for proof in binding.evidence["training_contract"]["split_proof"]["splits"]
    } == {"train": 6, "test": 6, "oot": 6}


def test_real_scorecard_tags_infinite_woe_boundaries_and_rejects_live_drift(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["recipe"] = "scorecard"
    fx["inputs"]["params"].update(
        {
            "max_iter": 200,
            "scorecard_max_bins": 3,
        }
    )

    output = _run(fx)
    evidence_record = next(
        record
        for record in _governed_records(fx)
        if record["kind"] == MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND
    )
    evidence = json.loads(Path(evidence_record["path"]).read_text(encoding="utf-8"))
    canonical_woe = evidence["model_artifact"]["scoring_metadata"]["woe_maps"]
    tagged_boundaries = [
        edge[NON_FINITE_BOUNDARY_TAG]
        for mapping in canonical_woe.values()
        for edge in mapping["edges"]
        if isinstance(edge, dict) and NON_FINITE_BOUNDARY_TAG in edge
    ]
    assert "negative_infinity" in tagged_boundaries
    assert "positive_infinity" in tagged_boundaries
    decoded = decode_modeling_scoring_woe_maps_boundaries(canonical_woe)
    assert all(
        decoded_mapping["edges"][0] == float("-inf")
        and decoded_mapping["edges"][-1] == float("inf")
        for decoded_mapping in decoded.values()
    )

    with connect(fx["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT woe_maps_json FROM model_artifacts WHERE id = ?",
            (output["model_artifact_id"],),
        ).fetchone()
        assert row is not None
        live_woe = json.loads(row["woe_maps_json"])
        first_feature = sorted(live_woe)[0]
        live_woe[first_feature]["edges"][0] = -999999.0
        conn.execute(
            "UPDATE model_artifacts SET woe_maps_json = ? WHERE id = ?",
            (
                json.dumps(
                    live_woe,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                output["model_artifact_id"],
            ),
        )

    with pytest.raises(ModelingError, match="scoring_metadata_hash.*drifted"):
        evidence_tools.validate_train_model_with_evidence_v2_tool_output(
            output,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )


def test_bad_target_direction_is_rejected_before_training(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, target_bad_value=0)

    with pytest.raises(ModelingError, match="good=0, bad=1"):
        _run(fx)

    assert _governed_records(fx) == []
    assert ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    ) == []


def test_missing_label_drop_policy_is_bound_from_sample_design(tmp_path: Path) -> None:
    confirmed = _fixture(
        tmp_path / "confirmed",
        missing_train_label=True,
        drop_nan_labels=True,
    )
    output = _run(confirmed)
    binding = evidence_tools.validate_train_model_with_evidence_v2_tool_output(
        output,
        runtime=confirmed["runtime"],
        task_id=confirmed["task"].id,
    )
    assert binding == output
    loaded = evidence_tools.load_modeling_training_evidence_artifacts(
        confirmed["runtime"],
        task_id=confirmed["task"].id,
        sample_design_ref=confirmed["sample_ref"],
        model_binary_artifact_id=output["artifacts"]["model_binary"]["artifact_id"],
        expected_model_binary_artifact_content_hash=output["artifacts"][
            "model_binary"
        ]["content_hash"],
        evidence_artifact_id=output["artifacts"]["training_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"][
            "training_evidence"
        ]["content_hash"],
        expected_experiment_id=output["experiment_id"],
        expected_model_artifact_id=output["model_artifact_id"],
        expected_evidence_id=output["evidence_id"],
        expected_evidence_content_hash=output["evidence_content_hash"],
    )
    assert loaded.evidence["training_contract"]["label_handling"] == {
        "drop_nan_labels": True,
        "nan_labels_dropped": 1,
    }

    caller_override = _fixture(tmp_path / "caller-override")
    caller_override["inputs"]["drop_nan_labels"] = True
    with pytest.raises(ModelingError, match="unknown=drop_nan_labels"):
        _run(caller_override)


@pytest.mark.parametrize("recipe", ["ensemble", "lgb_regressor", "unknown"])
def test_unsupported_recipe_is_rejected(tmp_path: Path, recipe: str) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["recipe"] = recipe

    with pytest.raises(ModelingError, match="authoritative binary"):
        _run(fx)


@pytest.mark.parametrize(
    "params",
    [
        {"calibration": {"method": "isotonic"}},
        {"method": "platt"},
        {"refit_on_train_plus_test": True},
        {"preprocessing_steps": []},
    ],
)
def test_calibration_and_platform_owned_params_are_rejected(
    tmp_path: Path, params: dict
) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["params"] = params

    with pytest.raises(ModelingError):
        _run(fx)


def test_split_alias_overlap_and_feature_leak_are_rejected(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["split_values"]["test"] = "train"
    with pytest.raises(ModelingError, match="distinct"):
        _run(fx)

    fx = _fixture(tmp_path / "feature-leak")
    fx["inputs"]["features"].append("model_split")
    with pytest.raises(ModelingError, match="leak"):
        _run(fx)

    fx = _fixture(tmp_path / "target-leak")
    fx["inputs"]["features"].append("bad")
    with pytest.raises(ModelingError, match="target column.*leak"):
        _run(fx)


def test_requested_weight_must_equal_governed_field_before_recipe_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["params"]["sample_weight_col"] = "loan_amount"
    recipe_calls = 0

    def forbidden_recipe(*args, **kwargs):
        nonlocal recipe_calls
        recipe_calls += 1
        raise AssertionError("recipe must not be called")

    monkeypatch.setattr(evidence_tools, "_train_recipe", forbidden_recipe)

    with pytest.raises(ModelingError, match="governed.*weight_field"):
        _run(fx)

    assert recipe_calls == 0
    assert _governed_records(fx) == []


@pytest.mark.parametrize(
    ("split_name", "bad_weight"),
    [
        ("test", None),
        ("oot", 0.0),
    ],
)
def test_bad_governed_weight_fails_before_recipe_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split_name: str,
    bad_weight: object,
) -> None:
    fx = _fixture(
        tmp_path,
        bad_weight_split=split_name,
        bad_weight_value=bad_weight,
    )
    recipe_calls = 0

    def forbidden_recipe(*args, **kwargs):
        nonlocal recipe_calls
        recipe_calls += 1
        raise AssertionError("recipe must not be called")

    monkeypatch.setattr(evidence_tools, "_train_recipe", forbidden_recipe)

    with pytest.raises(ModelingError, match=f"risk/{split_name}"):
        _run(fx)

    assert recipe_calls == 0
    assert _governed_records(fx) == []


def test_governed_weight_field_can_be_explicitly_unused(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["params"].pop("sample_weight_col")

    output = _run(fx)
    evidence_record = next(
        record
        for record in _governed_records(fx)
        if record["kind"] == MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND
    )
    evidence = json.loads(Path(evidence_record["path"]).read_text(encoding="utf-8"))

    assert output["experiment_id"].startswith("experiment_")
    assert evidence["training_contract"]["weighting"] == {
        "used": False,
        "column": None,
    }


def test_live_selector_must_equal_membership_row_for_row(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["split_values"]["oot"] = "approval_only"

    with pytest.raises(ModelingError, match="selector mask"):
        _run(fx)

    assert _governed_records(fx) == []


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("membership_artifact_id", "artifact|invalid"),
        ("expected_membership_artifact_content_hash", "artifact|hash|invalid"),
        ("bundle_artifact_id", "artifact|invalid"),
        ("expected_bundle_artifact_content_hash", "artifact|hash|invalid"),
    ],
)
def test_wrong_sample_pair_identity_fails_closed(
    tmp_path: Path, field: str, message: str
) -> None:
    fx = _fixture(tmp_path)
    fx["inputs"]["sample_design_ref"][field] = hashlib.sha256(
        f"wrong-{field}".encode()
    ).hexdigest()

    with pytest.raises(ModelingError, match=message):
        _run(fx)


def test_sample_artifact_kind_swap_fails_closed(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    ref = fx["inputs"]["sample_design_ref"]
    ref["membership_artifact_id"] = ref["bundle_artifact_id"]
    ref["expected_membership_artifact_content_hash"] = ref[
        "expected_bundle_artifact_content_hash"
    ]

    with pytest.raises(ModelingError, match="kind|invalid|artifact"):
        _run(fx)

    assert _governed_records(fx) == []


def test_dataset_byte_tamper_fails_before_training(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    path = fx["runtime"].registry.resolve_path(fx["dataset"].id)
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ModelingError, match="drift|changed|hash"):
        _run(fx)

    assert _governed_records(fx) == []


def test_workspace_revision_drift_fails_before_training(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    workspaces = DataWorkspaceRepository(fx["settings"].db_path)
    current = workspaces.get_or_default(fx["task"].id)
    changed_mapping = DataSemanticMapping(
        target_col=current.semantic_mapping.target_col,
        field_roles={**current.semantic_mapping.field_roles, "x1": "numeric"},
    )
    workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=current.active_dataset_id,
            active_dataset_content_hash=current.active_dataset_content_hash,
            semantic_mapping=changed_mapping,
        ),
        expected_revision=current.revision,
    )

    with pytest.raises(ModelingError, match="workspace|binding|revision"):
        _run(fx)

    assert _governed_records(fx) == []


@pytest.mark.parametrize("failure_index", [1, 2])
def test_task_artifact_register_failure_rolls_back_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    fx = _fixture(tmp_path)
    original = fx["runtime"].task_artifacts.register_on_connection
    calls = 0

    def fail_register(conn, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_index:
            raise RuntimeError(f"register {failure_index} down")
        return original(conn, **kwargs)

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        fail_register,
    )

    with pytest.raises(RuntimeError, match=f"register {failure_index} down"):
        _run(fx)

    assert _governed_records(fx) == []
    assert ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    ) == []
    artifact_dir = Path(fx["settings"].tasks_dir) / fx["task"].id / "modeling_artifacts"
    assert not list(artifact_dir.glob("artifact_*"))
    assert not list(artifact_dir.glob("*.training_evidence.json"))


def test_recipe_write_failure_isolated_to_private_training_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    artifact_dir = (
        Path(fx["settings"].tasks_dir)
        / fx["task"].id
        / "modeling_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    legacy_model = artifact_dir / "artifact_legacy.joblib"
    legacy_latest = artifact_dir / "model_meta.json"
    legacy_model.write_bytes(b"legacy-model")
    legacy_latest.write_bytes(b'{"artifact_id":"artifact_legacy"}\n')

    def fail_after_private_binary_write(*args, out_dir: Path, **kwargs):
        private = Path(out_dir)
        assert private != artifact_dir
        assert private.parent.name == ".train_model_with_evidence_v2.staging"
        (private / "artifact_ghost.joblib").write_bytes(b"partial")
        raise RuntimeError("recipe metadata write down")

    monkeypatch.setattr(
        evidence_tools,
        "_train_recipe",
        fail_after_private_binary_write,
    )

    with pytest.raises(RuntimeError, match="recipe metadata write down"):
        _run(fx)

    assert legacy_model.read_bytes() == b"legacy-model"
    assert legacy_latest.read_bytes() == b'{"artifact_id":"artifact_legacy"}\n'
    assert not (artifact_dir / "artifact_ghost.joblib").exists()
    staging_parent = (
        Path(fx["settings"].tasks_dir)
        / fx["task"].id
        / ".train_model_with_evidence_v2.staging"
    )
    assert not staging_parent.exists()
    assert _governed_records(fx) == []


def test_uow_rollback_failure_is_attempted_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    rollback_calls = 0

    def fail_register(conn, **kwargs):
        raise RuntimeError("registration down")

    def fail_rollback(self):
        nonlocal rollback_calls
        rollback_calls += 1
        raise RuntimeError("uow rollback down")

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        fail_register,
    )
    monkeypatch.setattr(ArtifactUnitOfWork, "rollback", fail_rollback)

    with pytest.raises(RuntimeError, match="uow rollback down"):
        _run(fx)

    assert rollback_calls == 1
    assert _governed_records(fx) == []
    assert ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    ) == []


def test_failed_training_cleanup_preserves_concurrent_latest_model_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    artifact_dir = (
        Path(fx["settings"].tasks_dir)
        / fx["task"].id
        / "modeling_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    latest = artifact_dir / "model_meta.json"
    previous = b'{"artifact_id":"artifact_previous"}\n'
    concurrent = b'{"artifact_id":"artifact_concurrent"}\n'
    latest.write_bytes(previous)

    def concurrent_publish_then_fail(conn, **kwargs):
        # V2 keeps its shared latest pointer staged until after DB commit, so
        # transaction rollback never reads/deletes/restores this legacy-owned
        # path and cannot race with the publication below.
        assert latest.read_bytes() == previous
        latest.write_bytes(concurrent)
        raise RuntimeError("registration lost race")

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        concurrent_publish_then_fail,
    )

    with pytest.raises(RuntimeError, match="registration lost race"):
        _run(fx)

    assert latest.read_bytes() == concurrent
    assert not list(artifact_dir.glob("artifact_*"))
    assert _governed_records(fx) == []
    assert ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    ) == []


def test_same_task_training_lock_fails_fast_before_recipe_or_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    lock_path = evidence_tools._training_task_lock_path(
        fx["settings"].tasks_dir,
        task_id=fx["task"].id,
    )
    recipe_calls = 0
    cleanup_calls = 0

    def forbidden_recipe(*args, **kwargs):
        nonlocal recipe_calls
        recipe_calls += 1
        raise AssertionError("busy run must not enter recipe")

    def forbidden_cleanup(*args, **kwargs):
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise AssertionError("busy run must not enter cleanup")

    monkeypatch.setattr(evidence_tools, "_train_recipe", forbidden_recipe)
    monkeypatch.setattr(
        evidence_tools,
        "_cleanup_training_stage_dir",
        forbidden_cleanup,
    )

    with FileLock(str(lock_path), timeout=0):
        with pytest.raises(ModelingError, match="already running"):
            _run(fx)

    assert recipe_calls == 0
    assert cleanup_calls == 0
    assert _governed_records(fx) == []


def test_task_scoped_training_lock_does_not_block_another_task(
    tmp_path: Path,
) -> None:
    first = _fixture(tmp_path)
    second = _fixture(tmp_path)
    assert first["task"].id != second["task"].id
    first_lock = evidence_tools._training_task_lock_path(
        first["settings"].tasks_dir,
        task_id=first["task"].id,
    )
    second_lock = evidence_tools._training_task_lock_path(
        second["settings"].tasks_dir,
        task_id=second["task"].id,
    )
    assert first_lock != second_lock

    with FileLock(str(first_lock), timeout=0):
        output = _run(second)

    assert output["experiment_id"].startswith("experiment_")
    assert len(_governed_records(second)) == 2
    assert _governed_records(first) == []


def test_evidence_write_failure_rolls_back_model_and_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    original = Path.write_bytes

    def fail_evidence_write(path: Path, data: bytes):
        if "training_evidence" in path.name:
            raise OSError("evidence disk down")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_evidence_write)

    with pytest.raises(OSError, match="evidence disk down"):
        _run(fx)

    assert _governed_records(fx) == []
    assert ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    ) == []


def test_audit_failure_rolls_back_pair_model_and_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)

    def fail_audit(conn, **kwargs):
        raise RuntimeError("evidence audit down")

    monkeypatch.setattr(fx["runtime"].repo, "write_audit_on_connection", fail_audit)

    with pytest.raises(RuntimeError, match="evidence audit down"):
        _run(fx)

    assert _governed_records(fx) == []
    assert ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    ) == []


def test_database_commit_failure_rolls_back_files_rows_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)

    class FailingCommitConnection:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def commit(self):
            raise RuntimeError("database commit down")

    @contextmanager
    def failing_transaction():
        with connect(fx["settings"].db_path) as conn:
            yield FailingCommitConnection(conn)

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "transaction",
        failing_transaction,
    )

    with pytest.raises(RuntimeError, match="database commit down"):
        _run(fx)

    assert _governed_records(fx) == []
    assert ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    ) == []
    artifact_dir = Path(fx["settings"].tasks_dir) / fx["task"].id / "modeling_artifacts"
    assert not list(artifact_dir.glob("artifact_*"))
    assert not list(artifact_dir.glob("*.training_evidence.json"))


def test_post_database_uow_cleanup_failure_keeps_durable_files_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)

    def fail_commit(self):
        raise RuntimeError("post-db cleanup down")

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", fail_commit)

    output = _run(fx)

    records = _governed_records(fx)
    assert output["experiment_id"].startswith("experiment_")
    assert len(records) == 2
    assert all(Path(record["path"]).is_file() for record in records)
    assert len(ModelingRepository(fx["settings"].db_path).list_experiments(
        fx["task"].id
    )) == 1
    with connect(fx["settings"].db_path) as conn:
        warning = conn.execute(
            """
            SELECT outcome, detail_json FROM audit
             WHERE kind = ?
             ORDER BY at DESC
             LIMIT 1
            """,
            (
                evidence_tools.TRAIN_MODEL_WITH_EVIDENCE_V2_HOUSEKEEPING_WARNING_AUDIT_KIND,
            ),
        ).fetchone()
    assert warning is not None
    assert warning["outcome"] == "warning"
    assert json.loads(warning["detail_json"])["publication_committed"] is True


def test_post_database_housekeeping_warning_failure_cannot_mask_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)

    def fail_commit(self):
        raise RuntimeError("post-db cleanup down")

    def fail_warning_audit(**kwargs):
        raise RuntimeError("warning audit down")

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", fail_commit)
    monkeypatch.setattr(fx["runtime"].repo, "write_audit", fail_warning_audit)

    output = _run(fx)

    records = _governed_records(fx)
    assert output["experiment_id"].startswith("experiment_")
    assert len(records) == 2
    assert all(Path(record["path"]).is_file() for record in records)
    assert len(
        ModelingRepository(fx["settings"].db_path).list_experiments(
            fx["task"].id
        )
    ) == 1


@pytest.mark.parametrize("drift", ["scoring_metadata", "train_config", "metrics"])
def test_live_loader_rejects_current_training_snapshot_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    fx = _fixture(tmp_path)
    output = _run(fx)
    repository = ModelingRepository(fx["settings"].db_path)

    if drift == "scoring_metadata":
        artifact = repository.get_model_artifact(output["model_artifact_id"])
        assert artifact is not None
        repository.set_model_artifact_params(
            artifact.id,
            {
                **artifact.params,
                "calibration": {
                    "method": "isotonic",
                    "path": "forged.calibration.joblib",
                },
            },
        )
    else:
        column = "config_json" if drift == "train_config" else "metrics_json"
        with connect(fx["settings"].db_path) as conn:
            row = conn.execute(
                f"SELECT {column} FROM experiments WHERE id = ?",
                (output["experiment_id"],),
            ).fetchone()
            assert row is not None
            payload = json.loads(row[column])
            if drift == "train_config":
                payload["seed"] = int(payload["seed"]) + 1
            else:
                payload["test_auc"] = (
                    0.123456
                    if payload.get("test_auc") != 0.123456
                    else 0.654321
                )
            conn.execute(
                f"UPDATE experiments SET {column} = ? WHERE id = ?",
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    output["experiment_id"],
                ),
            )

    with pytest.raises(ModelingError, match="drifted from immutable"):
        evidence_tools.validate_train_model_with_evidence_v2_tool_output(
            output,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )


def test_live_loader_rejects_tamper_wrong_task_and_forged_output(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    output = _run(fx)
    missing_experiment = deepcopy(output)
    missing_experiment["experiment_id"] = "experiment_missing"
    with pytest.raises(ModelingError, match="experiment was not found"):
        evidence_tools.validate_train_model_with_evidence_v2_tool_output(
            missing_experiment,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )

    forged = deepcopy(output)
    original_id = forged["artifacts"]["model_binary"]["artifact_id"]
    forged["artifacts"]["model_binary"]["artifact_id"] = "0" * 64
    forged["artifacts"]["model_binary"]["download_url"] = forged["artifacts"][
        "model_binary"
    ]["download_url"].replace(original_id, "0" * 64)
    with pytest.raises(ModelingError, match="not found|invalid|artifact"):
        evidence_tools.validate_train_model_with_evidence_v2_tool_output(
            forged,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )

    evidence_record = next(
        record
        for record in _governed_records(fx)
        if record["kind"] == MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND
    )
    Path(evidence_record["path"]).write_bytes(
        Path(evidence_record["path"]).read_bytes() + b" "
    )
    with pytest.raises(ModelingError, match="bytes changed"):
        evidence_tools.validate_train_model_with_evidence_v2_tool_output(
            output,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )

    other = _fixture(tmp_path / "other")
    with pytest.raises(ModelingError, match="not found|invalid|artifact"):
        evidence_tools.validate_train_model_with_evidence_v2_tool_output(
            output,
            runtime=other["runtime"],
            task_id=other["task"].id,
        )


def test_legacy_train_model_path_still_smokes(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    output = modeling_tools.tool_train_model(
        {
            "dataset_id": fx["dataset"].id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "bad",
            "split_col": "model_split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"max_iter": 200},
            "seed": 23,
        },
        fx["ctx"],
    )

    assert output["artifact_id"].startswith("artifact_")
    multiple = modeling_tools.tool_train_models(
        {
            "dataset_id": fx["dataset"].id,
            "recipes": ["lr"],
            "features": ["x1", "x2"],
            "target_col": "bad",
            "split_col": "model_split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"max_iter": 200},
            "seed": 23,
        },
        fx["ctx"],
    )
    assert multiple["best_experiment_id"].startswith("experiment_")
    assert _governed_records(fx) == []
