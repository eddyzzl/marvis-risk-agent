from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from marvis.packs.strategy import pool_stability_tools
from marvis.packs.strategy.pool_stability import (
    build_strategy_pool_stability,
    canonical_strategy_pool_stability_json,
)
from marvis.packs.strategy.pool_stability_tools import (
    POOL_STABILITY_ARTIFACT_KIND,
    StrategyPoolStabilityArtifactBinding,
)
from marvis.packs.strategy.report_bundle import (
    StrategyReportBundleError,
    build_strategy_report_bundle,
)
from marvis.packs.strategy.report_bundle_adapters import (
    build_strategy_report_bundle_source_inputs,
)
from marvis.repositories.task_artifacts import stable_task_artifact_id
from tests.test_strategy_report_bundle_adapters import (
    _bindings,
    _impact_cube_binding,
    _pool_binding_for_strategy_type,
    _present,
)


def _stability_binding(
    tmp_path: Path,
    impact_cube,
) -> StrategyPoolStabilityArtifactBinding:
    source_ref = {
        "artifact_id": impact_cube.artifact_id,
        "expected_artifact_content_hash": (
            impact_cube.artifact_content_hash
        ),
        "expected_cube_id": impact_cube.cube["cube_id"],
        "expected_cube_content_hash": impact_cube.cube["content_hash"],
    }
    stability = build_strategy_pool_stability(
        impact_cube=impact_cube.cube,
        impact_cube_ref=source_ref,
    )
    canonical = canonical_strategy_pool_stability_json(stability)
    artifact_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    artifact_path = (
        tmp_path
        / impact_cube.task_id
        / "strategy_pool_stabilities"
        / f"{stability['stability_id']}.json"
    )
    artifact_id = stable_task_artifact_id(
        task_id=impact_cube.task_id,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        path=str(artifact_path),
    )
    producer_run = pool_stability_tools._build_producer_run(
        task_id=impact_cube.task_id,
        request=source_ref,
        stability=stability,
        artifact_id=artifact_id,
        artifact_filename=artifact_path.name,
        artifact_content_hash=artifact_hash,
    )
    provenance = pool_stability_tools._artifact_provenance(
        task_id=impact_cube.task_id,
        stability=stability,
        producer_run=producer_run,
    )
    provenance_json = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return StrategyPoolStabilityArtifactBinding(
        task_id=impact_cube.task_id,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
        artifact_content_hash=artifact_hash,
        artifact_provenance=provenance,
        artifact_provenance_json=provenance_json,
        stability=stability,
        impact_cube=impact_cube,
        tasks_root=tmp_path,
        db_path=tmp_path / "marvis.sqlite",
    )


def _with_artifact_id(
    binding: StrategyPoolStabilityArtifactBinding,
    artifact_id: str,
) -> StrategyPoolStabilityArtifactBinding:
    source_ref = binding.stability["source_bindings"]["impact_cube"]
    producer_run = pool_stability_tools._build_producer_run(
        task_id=binding.task_id,
        request=source_ref,
        stability=binding.stability,
        artifact_id=artifact_id,
        artifact_filename=binding.artifact_path.name,
        artifact_content_hash=binding.artifact_content_hash,
    )
    provenance = pool_stability_tools._artifact_provenance(
        task_id=binding.task_id,
        stability=binding.stability,
        producer_run=producer_run,
    )
    return replace(
        binding,
        artifact_id=artifact_id,
        artifact_provenance=provenance,
        artifact_provenance_json=json.dumps(
            provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def _source_inputs(
    tmp_path: Path,
    *,
    with_stability: bool,
) -> tuple[dict, StrategyPoolStabilityArtifactBinding | None]:
    project, sample, pool, _legacy_impact = _bindings(tmp_path)
    impact_cube = _impact_cube_binding(tmp_path, sample, pool)
    stability = (
        _stability_binding(tmp_path, impact_cube)
        if with_stability
        else None
    )
    return (
        build_strategy_report_bundle_source_inputs(
            project_context=project,
            sample_design=sample,
            candidate_pool=pool,
            impact_cube=impact_cube,
            pool_stability=stability,
        ),
        stability,
    )


def _bundle(task_id: str, source_inputs: dict) -> dict:
    return build_strategy_report_bundle(
        task_id=task_id,
        report_revision=1,
        strategy_id=None,
        strategy_version=None,
        strategy_type="approval",
        title=_present(
            "Pool stability report",
            source_inputs["strategy_artifact_refs"][0],
        ),
        status="partial",
        generated_at="2026-07-26T16:00:00+08:00",
        **source_inputs,
    )


def test_pool_stability_adds_bounded_distribution_summary_without_effect_claims(
    tmp_path: Path,
) -> None:
    without, _ = _source_inputs(tmp_path, with_stability=False)
    with_stability, stability_binding = _source_inputs(
        tmp_path,
        with_stability=True,
    )
    assert stability_binding is not None
    task_id = stability_binding.task_id

    without_bundle = _bundle(task_id, without)
    stability_bundle = _bundle(task_id, with_stability)
    assert stability_bundle["effect_stages"] == without_bundle["effect_stages"]

    base_impact = without["sections"][5]
    impact = with_stability["sections"][5]
    assert impact["stage_evidence"] == base_impact["stage_evidence"]
    table = next(
        item
        for item in impact["tables"]
        if item["table_id"] == "strategy_pool_stability_summary"
    )
    assert impact["tables"].index(table) <= 1
    assert table["sheet_key"] == "10_validation"
    assert table["effect_stage"] is None
    assert len(table["rows"]) == 8
    assert {
        (
            row["cells"]["population"]["value"],
            row["cells"]["comparison_partition"]["value"],
            row["cells"]["basis"]["value"],
        )
        for row in table["rows"]
    } == {
        (population, partition, basis)
        for population in ("approval", "risk")
        for partition in ("validation", "oot")
        for basis in ("waterfall_incremental", "new_action")
    }

    stability_ref = {
        "kind": "pool_stability",
        "ref_id": stability_binding.artifact_id,
        "content_hash": stability_binding.artifact_content_hash,
    }
    assert table["source_refs"] == [stability_ref]
    assert stability_ref in impact["source_refs"]
    assert stability_ref in with_stability["strategy_artifact_refs"]

    final = with_stability["sections"][6]
    assert stability_ref in final["source_refs"]
    final_fields = {
        item["field_id"]: item["field"] for item in final["summary_fields"]
    }
    summary = final_fields["pool_stability_distribution_drift_summary"]
    assert summary["source_refs"] == [stability_ref]
    assert summary["value"]["baseline_partition"] == "development"
    assert summary["value"]["comparison_partitions"] == [
        "validation",
        "oot",
    ]
    assert summary["value"]["scope"] == "distribution_drift_only"

    drift_flags = [
        item
        for item in impact["red_flags"]
        if item["code"].startswith("pool_stability_distribution_drift_")
    ]
    assert drift_flags
    assert all("分布漂移" in item["message"] for item in drift_flags)


def test_pool_stability_binding_or_exact_impact_source_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    project, sample, pool, _legacy_impact = _bindings(tmp_path)
    impact_cube = _impact_cube_binding(tmp_path, sample, pool)
    stability = _stability_binding(tmp_path, impact_cube)
    other_pool = _pool_binding_for_strategy_type(
        tmp_path,
        sample,
        "reject",
    )
    other_impact_cube = _impact_cube_binding(
        tmp_path,
        sample,
        other_pool,
    )
    other_stability = _stability_binding(tmp_path, other_impact_cube)

    for tampered in (
        replace(stability, stability=object()),
        replace(stability, tasks_root=object()),
        _with_artifact_id(stability, "0" * 64),
        other_stability,
    ):
        with pytest.raises(
            StrategyReportBundleError,
            match="Pool stability",
        ):
            build_strategy_report_bundle_source_inputs(
                project_context=project,
                sample_design=sample,
                candidate_pool=pool,
                impact_cube=impact_cube,
                pool_stability=tampered,
            )
