from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import sqlite3

import pytest

from marvis.data.workspace import data_semantic_mapping_hash
from marvis.files import sha256_file
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.pool_validation import (
    canonical_strategy_pool_validation_json,
)
from marvis.packs.strategy.pool_validation_tools import (
    POOL_VALIDATION_ARTIFACT_KIND,
    POOL_VALIDATION_TOOL_SCHEMA_VERSION,
    run_measure_strategy_pool_validation,
    validate_measure_strategy_pool_validation_tool_output,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    run_materialize_sample_design_v2,
)
import marvis.packs.strategy.pool_validation_tools as validation_tools
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_sample_design_v2_tool import _setup as _sample_v2_setup


def _action(action_type: str) -> dict:
    return {
        "type": action_type,
        "value": "approve" if action_type == "approval" else action_type,
        "reason_code": None if action_type == "approval" else "RISK",
        "stop": True,
    }


def _setup(
    tmp_path: Path,
    *,
    target_bad_value: int = 1,
    empty_validation: bool = False,
) -> dict:
    fx = _sample_v2_setup(tmp_path, target_bad_value=target_bad_value)
    if empty_validation:
        fx["request"]["partitioning"]["selectors"]["validation"] = {
            "op": "eq",
            "left": {"column": "sample_split"},
            "right": {"literal": "missing"},
        }
        fx["request"]["partitioning"]["selectors"]["oot"] = {
            "op": "or",
            "args": [
                {
                    "op": "eq",
                    "left": {"column": "sample_split"},
                    "right": {"literal": "valid"},
                },
                {
                    "op": "eq",
                    "left": {"column": "sample_split"},
                    "right": {"literal": "oot"},
                },
            ],
        }
    v2_output = run_materialize_sample_design_v2(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    legacy_ref = fx["request"]["legacy_sample_design_ref"]
    workspace = fx["workspace"]
    mapping_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    analysis = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": mapping_hash,
            "target_col": "bad",
            "sample_design_ref": legacy_ref,
            "features": ["legacy_score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "drop_nan_labels": True,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        fx["ctx"],
    )
    candidate_report = next(
        item
        for item in analysis["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    method = analysis["candidate_evidence"]["analysis"]["features"][0][
        "methods"
    ][0]
    candidate = strategy_tools.tool_refine_univariate_candidate(
        {
            "source_artifact_id": candidate_report["artifact_id"],
            "expected_artifact_content_hash": candidate_report["content_hash"],
            "expected_candidate_id": analysis["candidate_id"],
            "expected_evidence_hash": analysis["evidence_hash"],
            "feature": "legacy_score",
            "method": "equal_width",
            "merge_groups": [],
            "selection": {"source_bin_ids": [method["bins"][0]["id"]]},
        },
        fx["ctx"],
    )
    candidate_artifact = candidate["artifacts"][0]
    added = run_add_candidate_to_pool(
        {
            "source_artifact_id": candidate_artifact["artifact_id"],
            "expected_artifact_content_hash": candidate_artifact[
                "content_hash"
            ],
            "expected_asset_id": candidate["asset_id"],
            "expected_asset_hash": candidate["asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject"),
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
        },
        fx["ctx"],
        fx["runtime"],
    )
    pool_artifact = added["artifacts"][0]
    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    membership = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    )
    sample_ref = {
        "membership_artifact_id": membership["id"],
        "expected_membership_artifact_content_hash": membership[
            "content_hash"
        ],
        "bundle_artifact_id": bundle["id"],
        "expected_bundle_artifact_content_hash": bundle["content_hash"],
        "expected_bundle_id": v2_output["bundle_id"],
        "expected_sample_design_id": v2_output["sample_design_id"],
        "expected_sample_design_content_hash": v2_output[
            "sample_design_content_hash"
        ],
    }
    request = {
        "strategy_type": "approval",
        "pool_ref": {
            "artifact_id": pool_artifact["artifact_id"],
            "expected_artifact_content_hash": pool_artifact["content_hash"],
            "expected_pool_id": added["pool_id"],
            "expected_revision": added["revision"],
            "expected_revision_id": added["pool"]["revision_id"],
            "expected_snapshot_hash": added["snapshot_hash"],
        },
        "sample_design_ref": sample_ref,
        "partition": "validation",
        "population": "risk",
        "comparison_mode": "absolute",
    }
    return {
        **fx,
        "v2_output": v2_output,
        "pool": added,
        "pool_artifact": pool_artifact,
        "sample_ref": sample_ref,
        "validation_request": request,
    }


def _validation_artifacts(fx: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["kind"] == POOL_VALIDATION_ARTIFACT_KIND
    ]


@pytest.mark.parametrize(
    ("partition", "population_count", "labelled_count"),
    [("validation", 2, 2), ("oot", 2, 1)],
)
def test_measure_pool_validation_publishes_exact_independent_evidence(
    tmp_path: Path,
    partition: str,
    population_count: int,
    labelled_count: int,
) -> None:
    fx = _setup(tmp_path)
    request = {**fx["validation_request"], "partition": partition}

    output = run_measure_strategy_pool_validation(
        request,
        fx["ctx"],
        fx["runtime"],
    )

    assert output["schema_version"] == POOL_VALIDATION_TOOL_SCHEMA_VERSION
    assert output["partition"] == partition
    assert output["population"] == "risk"
    assert output["lifecycle_stage"] == partition
    assert output["validation_status"] == "independent_evidence"
    assert output["population_count"] == population_count
    assert output["labeled_count"] == labelled_count
    assert output["unlabeled_count"] == population_count - labelled_count
    assert output["evidence"]["population_metrics"]["population_count"] == (
        population_count
    )
    assert output["evidence"]["source_bindings"]["target"]["bad_value"] == 1
    assert validate_measure_strategy_pool_validation_tool_output(output) == output
    assert output["not_mutated_pool"] is True
    assert output["not_created_strategy"] is True
    assert output["not_adopted"] is True
    assert output["not_promoted"] is True
    assert output["not_deployed"] is True

    artifacts = _validation_artifacts(fx)
    assert len(artifacts) == 1
    record = artifacts[0]
    path = Path(record["path"])
    assert path.parent.name == "strategy_pool_validations"
    assert path.read_text("utf-8") == canonical_strategy_pool_validation_json(
        output["evidence"]
    )
    assert sha256_file(path) == output["artifact"]["content_hash"]
    assert record["origin_tool"] == "strategy.measure_strategy_pool_validation"
    assert fx["runtime"].strategies.list_for_task(fx["task"].id) == []


def test_measure_pool_validation_is_idempotent_and_cache_scalars_fail_closed(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_measure_strategy_pool_validation(
        fx["validation_request"],
        fx["ctx"],
        fx["runtime"],
    )
    second = run_measure_strategy_pool_validation(
        fx["validation_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert first == second
    assert len(_validation_artifacts(fx)) == 1
    tampered = copy.deepcopy(first)
    tampered["population_count"] = 999
    with pytest.raises(StrategyError, match="population_count drifted"):
        validate_measure_strategy_pool_validation_tool_output(tampered)


def test_measure_pool_validation_uses_v2_bad_zero_polarity(
    tmp_path: Path,
) -> None:
    bad_zero = _setup(tmp_path / "bad-zero", target_bad_value=0)
    bad_one = _setup(tmp_path / "bad-one", target_bad_value=1)

    zero = run_measure_strategy_pool_validation(
        bad_zero["validation_request"],
        bad_zero["ctx"],
        bad_zero["runtime"],
    )
    one = run_measure_strategy_pool_validation(
        bad_one["validation_request"],
        bad_one["ctx"],
        bad_one["runtime"],
    )

    assert zero["evidence"]["source_bindings"]["target"]["bad_value"] == 0
    assert one["evidence"]["source_bindings"]["target"]["bad_value"] == 1
    assert zero["evidence"]["population_metrics"] == one["evidence"][
        "population_metrics"
    ]
    assert zero["evidence"]["overall"] == one["evidence"]["overall"]
    assert [
        row["incremental"] for row in zero["evidence"]["waterfall"]
    ] == [
        row["incremental"] for row in one["evidence"]["waterfall"]
    ]


def test_measure_pool_validation_empty_partition_fails_without_artifact(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path, empty_validation=True)

    with pytest.raises(StrategyError, match="validation partition is empty"):
        run_measure_strategy_pool_validation(
            fx["validation_request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert _validation_artifacts(fx) == []


def test_measure_pool_validation_rejects_pool_sample_mismatch(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path / "primary")
    other = _setup(tmp_path / "other")
    request = {
        **fx["validation_request"],
        "sample_design_ref": other["sample_ref"],
    }

    with pytest.raises(StrategyError, match="artifact|task"):
        run_measure_strategy_pool_validation(
            request,
            fx["ctx"],
            fx["runtime"],
        )
    assert _validation_artifacts(fx) == []


def test_measure_pool_validation_rejects_v2_membership_and_dataset_drift(
    tmp_path: Path,
) -> None:
    membership_fx = _setup(tmp_path / "membership")
    membership_record = next(
        item
        for item in TaskArtifactRepository(
            membership_fx["settings"].db_path
        ).list_for_task(membership_fx["task"].id)
        if item["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    membership_path = Path(membership_record["path"])
    membership_path.write_bytes(membership_path.read_bytes() + b"drift")
    with pytest.raises(StrategyError, match="content hash|changed"):
        run_measure_strategy_pool_validation(
            membership_fx["validation_request"],
            membership_fx["ctx"],
            membership_fx["runtime"],
        )
    assert _validation_artifacts(membership_fx) == []

    dataset_fx = _setup(tmp_path / "dataset")
    dataset_path = Path(
        dataset_fx["runtime"].registry.resolve_path(
            dataset_fx["dataset"].id
        )
    )
    dataset_path.write_bytes(dataset_path.read_bytes() + b"drift")
    with pytest.raises(StrategyError, match="drift|changed|hash"):
        run_measure_strategy_pool_validation(
            dataset_fx["validation_request"],
            dataset_fx["ctx"],
            dataset_fx["runtime"],
        )
    assert _validation_artifacts(dataset_fx) == []


def test_measure_pool_validation_rejects_current_pool_drift_and_injected_metrics(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    stale = copy.deepcopy(fx["validation_request"])
    stale["pool_ref"]["expected_revision"] += 1
    with pytest.raises(StrategyError, match="stale"):
        run_measure_strategy_pool_validation(
            stale,
            fx["ctx"],
            fx["runtime"],
        )

    with pytest.raises(StrategyError, match="unsupported fields"):
        run_measure_strategy_pool_validation(
            {
                **fx["validation_request"],
                "metrics": {"bad_rate": 0.0},
            },
            fx["ctx"],
            fx["runtime"],
        )
    assert _validation_artifacts(fx) == []


def test_measure_pool_validation_revalidates_registry_source_under_write_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fx = _setup(tmp_path)
    original = validation_tools._persist_evidence

    def drift_then_persist(*args, **kwargs):
        with sqlite3.connect(fx["settings"].db_path) as conn:
            conn.execute(
                "UPDATE datasets SET source_path = ? WHERE id = ?",
                ("drifted/source.parquet", fx["dataset"].id),
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        validation_tools,
        "_persist_evidence",
        drift_then_persist,
    )
    with pytest.raises(StrategyError, match="registry path changed"):
        run_measure_strategy_pool_validation(
            fx["validation_request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert _validation_artifacts(fx) == []
    assert not list(
        (
            fx["settings"].tasks_dir
            / fx["task"].id
            / "strategy_pool_validations"
        ).glob("*.json")
    )


def test_measure_pool_validation_registration_failure_rolls_back_file_and_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fx = _setup(tmp_path)

    def fail_registration(*args, **kwargs):
        raise RuntimeError("simulated validation registry failure")

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        fail_registration,
    )
    with pytest.raises(RuntimeError, match="simulated validation registry failure"):
        run_measure_strategy_pool_validation(
            fx["validation_request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert _validation_artifacts(fx) == []
    assert not list(
        (
            fx["settings"].tasks_dir
            / fx["task"].id
            / "strategy_pool_validations"
        ).glob("*.json")
    )


def test_measure_pool_validation_concurrent_replay_registers_one_artifact(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    def run_once() -> dict:
        return run_measure_strategy_pool_validation(
            fx["validation_request"],
            fx["ctx"],
            fx["runtime"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda _item: run_once(), range(2)))

    assert outputs[0] == outputs[1]
    assert len(_validation_artifacts(fx)) == 1


def test_measure_pool_validation_rejects_symlink_output_directory(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    task_dir = fx["settings"].tasks_dir / fx["task"].id
    external = tmp_path / "external"
    external.mkdir()
    output_dir = task_dir / "strategy_pool_validations"
    output_dir.symlink_to(external, target_is_directory=True)

    with pytest.raises(StrategyError, match="regular directory"):
        run_measure_strategy_pool_validation(
            fx["validation_request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert list(external.iterdir()) == []


def test_measure_pool_validation_pack_entrypoint_forwards_runtime(monkeypatch) -> None:
    inputs = {"partition": "validation"}
    ctx = object()
    runtime = object()

    monkeypatch.setattr(strategy_tools, "_runtime", lambda received_ctx: runtime)

    def fake_run(received_inputs, received_ctx, received_runtime):
        assert received_inputs is inputs
        assert received_ctx is ctx
        assert received_runtime is runtime
        return {"forwarded": True}

    monkeypatch.setattr(
        strategy_tools,
        "run_measure_strategy_pool_validation",
        fake_run,
    )

    assert strategy_tools.tool_measure_strategy_pool_validation(inputs, ctx) == {
        "forwarded": True
    }
