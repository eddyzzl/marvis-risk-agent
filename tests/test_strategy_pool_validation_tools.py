from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.data.workspace import data_semantic_mapping_hash
from marvis.files import sha256_file
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import (
    ABSENT_POOL_SNAPSHOT_HASH,
    add_verified_candidate_fragment,
    compile_strategy_pool,
)
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
)
from marvis.packs.strategy.pool_validation import (
    canonical_strategy_pool_validation_json,
)
from marvis.packs.strategy.pool_validation_tools import (
    POOL_VALIDATION_ARTIFACT_KIND,
    POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION,
    POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION,
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


def _controlled_score_requirement(
    *,
    pool_binding,
    virtual_field: str = "__marvis_model_pd_0123456789abcdef",
) -> tuple[object, ResolvedPoolRequirements]:
    requirement = {
        "type": "model_score_vector.v1",
        "virtual_field": virtual_field,
        "score_product": "raw_native_uncalibrated_bad_probability",
        "score_evidence_artifact_id": "1" * 64,
        "score_evidence_artifact_content_hash": "2" * 64,
        "score_vector_artifact_id": "0123456789abcdef" + "3" * 48,
        "score_vector_artifact_content_hash": "4" * 64,
    }
    source = pool_binding.pool["entries"][0]["source"]
    fragment = build_verified_candidate_fragment(
        artifact={
            "artifact_id": source["artifact_id"],
            "artifact_kind": source["artifact_kind"],
            "artifact_schema_version": source["artifact_schema_version"],
            "artifact_content_hash": source["artifact_content_hash"],
            "origin_tool": source["origin_tool"],
        },
        asset={
            "schema_version": source["asset_schema_version"],
            "asset_id": source["asset_id"],
            "asset_hash": source["asset_hash"],
            "asset_type": source["asset_type"],
        },
        fragment_type=source["fragment_type"],
        rule_id="scorecard-cutoff-rule",
        condition={
            "op": "compare",
            "field": virtual_field,
            "operator": ">=",
            "value": 0.5,
            "missing": "no_match",
        },
        requirements=[requirement],
        effect_id=source["effect_id"],
        evidence_id=source["evidence_id"],
        evidence_hash=source["evidence_hash"],
        evidence_identity=source["evidence_identity"],
    )
    score_pool = add_verified_candidate_fragment(
        None,
        task_id=pool_binding.task_id,
        strategy_type=pool_binding.strategy_type,
        default_action=pool_binding.pool["default_action"],
        verified_candidate_fragment=fragment,
        action=pool_binding.pool["entries"][0]["action"],
    )
    compiled = compile_strategy_pool(score_pool)
    outer = compiled["requirements"][0]
    controlled_binding = replace(
        pool_binding,
        pool=score_pool,
        compiled_design=compiled,
    )
    evidence_binding = SimpleNamespace(
        evidence_record={
            "id": requirement["score_evidence_artifact_id"],
            "content_hash": requirement[
                "score_evidence_artifact_content_hash"
            ],
        },
        vector_record={
            "id": requirement["score_vector_artifact_id"],
            "content_hash": requirement[
                "score_vector_artifact_content_hash"
            ],
        },
    )
    resolved = ResolvedPoolRequirements(
        task_id=pool_binding.task_id,
        requirements=(outer,),
        requirements_hash=hashlib.sha256(
            json.dumps(
                [outer],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        field_bindings=((virtual_field, evidence_binding),),
    )
    return controlled_binding, resolved


def test_measure_pool_validation_hydrates_score_requirement_before_partition_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    original_load = validation_tools._load_pool_binding
    base_binding = original_load(
        fx["runtime"],
        task_id=fx["task"].id,
        request=fx["validation_request"],
    )
    base_development = (
        validation_tools.bind_strategy_pool_development_execution(
            fx["runtime"],
            base_binding,
        )
    )
    controlled_binding, resolved = _controlled_score_requirement(
        pool_binding=base_binding,
    )
    controlled_development = replace(
        base_development,
        pool=controlled_binding,
    )
    outer = controlled_binding.compiled_design["requirements"][0]
    original_build = validation_tools.build_strategy_pool_validation_evidence
    original_read = validation_tools.pd.read_parquet
    calls = {"hydrate": 0, "cas": 0}

    def load_with_requirement(*args, **kwargs):
        return controlled_binding

    def bind_development(runtime, pool):
        assert runtime is fx["runtime"]
        assert pool is controlled_binding
        return controlled_development

    def resolve_requirements(
        runtime,
        *,
        task_id,
        compiled_design,
        sample_design,
    ):
        assert runtime is fx["runtime"]
        assert task_id == fx["task"].id
        assert compiled_design["requirements"] == [outer]
        assert sample_design.task_id == fx["task"].id
        return resolved

    def hydrate(frame: pd.DataFrame, *, resolved: object) -> pd.DataFrame:
        assert resolved is globals_resolved
        assert len(frame) == fx["dataset"].row_count
        assert isinstance(frame.index, pd.RangeIndex)
        assert globals_resolved.virtual_fields[0] not in frame.columns
        calls["hydrate"] += 1
        hydrated = frame.copy(deep=True)
        hydrated[globals_resolved.virtual_fields[0]] = [
            index / max(1, len(frame) - 1) for index in range(len(frame))
        ]
        return hydrated

    def read_snapshot(source, *args, **kwargs):
        assert resolved.virtual_fields[0] not in kwargs["columns"]
        frame = original_read(source, *args, **kwargs)
        frame.index = pd.Index(
            range(1000, 1000 + len(frame)),
            name="persisted_source_index",
        )
        return frame

    def build(**kwargs):
        frame = kwargs["frame"]
        assert resolved.virtual_fields[0] in frame.columns
        assert len(frame) < fx["dataset"].row_count
        return original_build(**kwargs)

    def require_on_connection(conn, received):
        assert received is resolved
        assert conn.in_transaction
        calls["cas"] += 1

    def requirement_provenance(received):
        assert received is resolved
        return {
            "requirements_hash": resolved.requirements_hash,
            "requirements": list(resolved.requirements),
            "virtual_fields": list(resolved.virtual_fields),
        }

    globals_resolved = resolved
    monkeypatch.setattr(validation_tools, "_load_pool_binding", load_with_requirement)
    monkeypatch.setattr(
        validation_tools,
        "bind_strategy_pool_development_execution",
        bind_development,
    )
    monkeypatch.setattr(
        validation_tools,
        "resolve_pool_requirements",
        resolve_requirements,
        raising=False,
    )
    monkeypatch.setattr(
        validation_tools,
        "hydrate_requirement_fields",
        hydrate,
        raising=False,
    )
    monkeypatch.setattr(validation_tools.pd, "read_parquet", read_snapshot)
    monkeypatch.setattr(
        validation_tools,
        "build_strategy_pool_validation_evidence",
        build,
    )
    monkeypatch.setattr(
        validation_tools,
        "require_resolved_pool_requirements_on_connection",
        require_on_connection,
        raising=False,
    )
    monkeypatch.setattr(
        validation_tools,
        "pool_requirement_bindings_provenance",
        requirement_provenance,
    )
    monkeypatch.setattr(
        validation_tools,
        "require_strategy_candidate_pool_artifact_binding_on_connection",
        lambda conn, binding: None,
    )

    output = run_measure_strategy_pool_validation(
        fx["validation_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert output["population_count"] == 2
    assert output["evidence"]["waterfall"][0]["incremental"][
        "population_count"
    ] == 1
    assert calls["hydrate"] == 1
    assert calls["cas"] >= 4
    provenance = _validation_artifacts(fx)[0]["provenance"]
    assert (
        provenance["schema_version"]
        == POOL_VALIDATION_REQUIREMENTS_ARTIFACT_SCHEMA_VERSION
    )
    bindings = provenance["field_bindings"]["requirements"]
    assert bindings["requirements_hash"] == resolved.requirements_hash
    assert bindings["virtual_fields"] == list(resolved.virtual_fields)
    assert bindings["requirements"] == list(resolved.requirements)
    forged_v1 = copy.deepcopy(provenance)
    forged_v1["schema_version"] = POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION
    with pytest.raises(StrategyError, match="field bindings|schema"):
        validation_tools._validate_provenance(forged_v1)
    forged_v2 = copy.deepcopy(provenance)
    del forged_v2["field_bindings"]["requirements"]
    with pytest.raises(StrategyError, match="field bindings|schema"):
        validation_tools._validate_provenance(forged_v2)


@pytest.mark.parametrize(
    "message",
    [
        "virtual score field conflicts with physical dataset column",
        "model score evidence and SampleDesign V2 differ",
    ],
)
def test_measure_pool_validation_fails_closed_on_requirement_resolution_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    fx = _setup(tmp_path)

    def reject_requirement(*args, **kwargs):
        raise StrategyError(message)

    monkeypatch.setattr(
        validation_tools,
        "resolve_pool_requirements",
        reject_requirement,
    )

    with pytest.raises(StrategyError, match=message):
        run_measure_strategy_pool_validation(
            fx["validation_request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert _validation_artifacts(fx) == []


def test_measure_pool_validation_requirement_cas_drift_rolls_back_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    real_require = (
        validation_tools.require_resolved_pool_requirements_on_connection
    )
    calls = 0

    def drift_on_final_write(conn, resolved):
        nonlocal calls
        calls += 1
        real_require(conn, resolved)
        if calls == 2:
            raise StrategyError(
                "model score vector disappeared before validation commit"
            )

    monkeypatch.setattr(
        validation_tools,
        "require_resolved_pool_requirements_on_connection",
        drift_on_final_write,
    )

    with pytest.raises(StrategyError, match="disappeared"):
        run_measure_strategy_pool_validation(
            fx["validation_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert calls == 2
    assert _validation_artifacts(fx) == []
    assert not list(
        (
            fx["settings"].tasks_dir
            / fx["task"].id
            / "strategy_pool_validations"
        ).glob("*.json")
    )


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
    assert (
        validate_measure_strategy_pool_validation_tool_output(
            output,
            expected_task_id=fx["task"].id,
            expected_artifact_id=output["artifact"]["artifact_id"],
        )
        == output
    )
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
    assert (
        record["provenance"]["schema_version"]
        == POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION
    )
    assert "requirements" not in record["provenance"]["field_bindings"]
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
        validate_measure_strategy_pool_validation_tool_output(
            tampered,
            expected_task_id=fx["task"].id,
            expected_artifact_id=first["artifact"]["artifact_id"],
        )

    forged_binding = copy.deepcopy(first)
    forged_binding["artifact"]["artifact_id"] = "0" * 64
    forged_binding["artifact"]["download_url"] = (
        f"/api/tasks/{fx['task'].id}/task-artifacts/{'0' * 64}/download"
    )
    with pytest.raises(StrategyError, match="artifact_id drifted"):
        validate_measure_strategy_pool_validation_tool_output(
            forged_binding,
            expected_task_id=fx["task"].id,
            expected_artifact_id=first["artifact"]["artifact_id"],
        )


def test_measure_pool_validation_recovers_an_exact_promoted_orphan(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_measure_strategy_pool_validation(
        fx["validation_request"],
        fx["ctx"],
        fx["runtime"],
    )
    record = _validation_artifacts(fx)[0]
    path = Path(record["path"])
    original_bytes = path.read_bytes()
    with sqlite3.connect(fx["settings"].db_path) as conn:
        conn.execute(
            "DELETE FROM task_artifacts WHERE id = ?",
            (record["id"],),
        )

    replay = run_measure_strategy_pool_validation(
        fx["validation_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert replay == first
    assert path.read_bytes() == original_bytes
    recovered = _validation_artifacts(fx)
    assert len(recovered) == 1
    assert recovered[0]["id"] == record["id"]


def test_measure_pool_validation_reads_an_authenticated_private_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fx = _setup(tmp_path)
    reads: list[object] = []
    live_reads = 0
    original_live_read = fx["runtime"].backend.read_frame
    original_read_parquet = validation_tools.pd.read_parquet

    def allow_sample_load_only(*args, **kwargs):
        nonlocal live_reads
        live_reads += 1
        if live_reads > 1:
            raise AssertionError(
                "Pool replay must not reopen the live dataset path"
            )
        return original_live_read(*args, **kwargs)

    def record_snapshot_read(source, *args, **kwargs):
        if not isinstance(source, (str, Path)):
            reads.append(source)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(
        fx["runtime"].backend,
        "read_frame",
        allow_sample_load_only,
    )
    monkeypatch.setattr(
        validation_tools.pd,
        "read_parquet",
        record_snapshot_read,
    )

    output = run_measure_strategy_pool_validation(
        fx["validation_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert output["population_count"] == 2
    assert live_reads == 1
    assert len(reads) == 1


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
