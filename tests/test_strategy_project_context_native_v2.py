from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from marvis.data.workspace import data_semantic_mapping_hash
from marvis.db import TaskRepository
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_tools import (
    IMPACT_CUBE_ARTIFACT_KIND,
    run_measure_strategy_impact_cube,
)
from marvis.packs.strategy.pool_tools import (
    ABSENT_POOL_SNAPSHOT_HASH,
    run_add_candidate_to_pool,
)
from marvis.packs.strategy.project_context_tools import (
    PROJECT_CONTEXT_ARTIFACT_KIND,
    load_current_strategy_project_context,
    run_materialize_project_context,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    run_materialize_sample_design_v2_native,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
)
from marvis.packs.strategy.pool_impact_tools import (
    POOL_IMPACT_ARTIFACT_KIND,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_sample_design_v2_native_tool import (
    _setup_native,
)


def _action(action_type: str) -> dict:
    return {
        "type": action_type,
        "value": "approve" if action_type == "approval" else action_type,
        "reason_code": None if action_type == "approval" else "RISK",
        "stop": True,
    }


def _materialize_native_v2_context_sources(
    tmp_path: Path,
    *,
    include_older_evidence: bool = False,
    older_impact_partitions: tuple[str, ...] = ("development",),
    latest_impact_partitions: tuple[str, ...] = (
        "development",
        "validation",
        "oot",
    ),
) -> dict:
    fx = _setup_native(tmp_path)
    if include_older_evidence:
        older_request = deepcopy(fx["request"])
        older_request["target_bad_value"] = 0
        run_materialize_sample_design_v2_native(
            older_request,
            fx["ctx"],
            fx["runtime"],
        )
    sample_output = run_materialize_sample_design_v2_native(
        deepcopy(fx["request"]),
        fx["ctx"],
        fx["runtime"],
    )
    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    bundle_record = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
        and item["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
        and item["provenance"]["bundle_id"] == sample_output["bundle_id"]
    )
    membership_record = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
        and item["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
        and item["id"]
        == bundle_record["provenance"]["membership_artifact_id"]
    )
    design = sample_output["bundle"]["sample_design"]
    risk_development_ref = {
        "artifact_id": bundle_record["id"],
        "artifact_content_hash": bundle_record["content_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "partition": "risk/development",
    }
    analysis = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "analysis_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                fx["workspace"].semantic_mapping
            ),
            "target_col": "bad",
            "sample_design_ref": risk_development_ref,
            "features": ["unused_feature"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "drop_nan_labels": True,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        fx["ctx"],
    )
    report_artifact = next(
        item
        for item in analysis["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    method = analysis["candidate_evidence"]["analysis"]["features"][0][
        "methods"
    ][0]
    selected_bin = next(
        item for item in method["bins"] if item["kind"] == "numeric_interval"
    )
    candidate = strategy_tools.tool_refine_univariate_candidate(
        {
            "source_artifact_id": report_artifact["artifact_id"],
            "expected_artifact_content_hash": report_artifact["content_hash"],
            "expected_candidate_id": analysis["candidate_id"],
            "expected_evidence_hash": analysis["evidence_hash"],
            "feature": "unused_feature",
            "method": "equal_width",
            "merge_groups": [],
            "selection": {"source_bin_ids": [selected_bin["id"]]},
        },
        fx["ctx"],
    )
    candidate_artifact = candidate["artifacts"][0]
    pool = run_add_candidate_to_pool(
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
    pool_artifact = pool["artifacts"][0]
    impact_request = {
        "strategy_type": "approval",
        "pool_ref": {
            "artifact_id": pool_artifact["artifact_id"],
            "expected_artifact_content_hash": pool_artifact[
                "content_hash"
            ],
            "expected_pool_id": pool["pool_id"],
            "expected_revision": pool["revision"],
            "expected_revision_id": pool["pool"]["revision_id"],
            "expected_snapshot_hash": pool["snapshot_hash"],
        },
        "sample_design_ref": {
            "membership_artifact_id": membership_record["id"],
            "expected_membership_artifact_content_hash": membership_record[
                "content_hash"
            ],
            "bundle_artifact_id": bundle_record["id"],
            "expected_bundle_artifact_content_hash": bundle_record[
                "content_hash"
            ],
            "expected_bundle_id": sample_output["bundle_id"],
            "expected_sample_design_id": sample_output["sample_design_id"],
            "expected_sample_design_content_hash": sample_output[
                "sample_design_content_hash"
            ],
        },
        "partitions": ["development", "validation", "oot"],
        "population": "risk",
        "dimension_bindings": {
            "month_col": None,
            "group_col": None,
            "segment_col": None,
        },
        "current_strategy_ref": None,
        "economics_inputs": None,
    }
    if include_older_evidence:
        run_measure_strategy_impact_cube(
            {**impact_request, "partitions": list(older_impact_partitions)},
            fx["ctx"],
            fx["runtime"],
        )
    impact = run_measure_strategy_impact_cube(
        {
            **impact_request,
            "partitions": list(latest_impact_partitions),
        },
        fx["ctx"],
        fx["runtime"],
    )
    message = TaskRepository(fx["settings"].db_path).add_agent_message(
        fx["task"].id,
        role="user",
        stage="chat",
        content="请基于原生 V2 样本与 ImpactCube 整理当前项目现状。",
    )
    request = {
        "expected_revision": 0,
        "expected_revision_id": None,
        "expected_state_hash": None,
        "user_message_ref": {
            "message_id": message["id"],
            "content_hash": hashlib.sha256(
                message["content"].encode("utf-8")
            ).hexdigest(),
        },
        "as_of": "2026-07-27",
        "scope": "原生 V2 策略开发",
        "business_context": {},
        "explicit_unavailable": ["historical_strategy_reviews"],
        "external_report_filenames": [],
    }
    return {
        **fx,
        "sample_output": sample_output,
        "bundle_record": bundle_record,
        "membership_record": membership_record,
        "impact": impact,
        "request": request,
    }


def test_project_context_discovers_native_v2_sample_and_full_impact_cube(
    tmp_path: Path,
) -> None:
    fx = _materialize_native_v2_context_sources(tmp_path)
    source_records = TaskArtifactRepository(
        fx["settings"].db_path
    ).list_for_task(fx["task"].id)
    assert all(
        item["kind"] not in {
            SAMPLE_DESIGN_ARTIFACT_KIND,
            POOL_IMPACT_ARTIFACT_KIND,
        }
        for item in source_records
    )
    assert any(item["kind"] == IMPACT_CUBE_ARTIFACT_KIND for item in source_records)

    output = run_materialize_project_context(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )

    state = output["revision"]["state"]
    snapshot = state["current_project_snapshot"]
    assert snapshot["status_fields"]["volume"]["value"] == [
        {
            "metric_key": "population_count",
            "partition": "overall",
            "population": "approval",
            "unit": "count",
            "value": 6,
        }
    ]
    approval = snapshot["status_fields"]["approval"]
    assert approval["availability"] == "present"
    assert [
        (item["partition"], item["effect_stage"], item["validation_status"])
        for item in approval["value"]["partitions"]
    ] == [
        ("development", "backtested", "unvalidated"),
        ("validation", "oot_validated", "independent_evidence"),
        ("oot", "oot_validated", "independent_evidence"),
    ]
    assert all(
        item["new"]["population_count"] == 2
        for item in approval["value"]["partitions"]
    )
    assert approval["value"]["partitions"][0]["new"]["metrics"][
        "approve_bad_rate"
    ] is None
    assert snapshot["status_fields"]["risk"]["value"] == [
        {
            "metric_key": "bad_count",
            "partition": "overall",
            "population": "risk",
            "unit": "count",
            "value": 3,
        },
        {
            "metric_key": "bad_rate",
            "partition": "overall",
            "population": "risk",
            "unit": "ratio",
            "value": 0.6,
        },
    ]
    assert snapshot["maturity_summary"]["value"] == {
        "cutoff_date": "2026-04-30",
        "eligible_count": 6,
        "labeled_count": 5,
        "performance_window_days": 30,
        "population": "risk",
        "reason": None,
        "status": "confirmed_matured",
    }
    missing_paths = {
        item["field_path"] for item in output["missing_information_records"]
    }
    assert {
        "current.status_fields.volume",
        "current.status_fields.approval",
        "current.status_fields.risk",
        "current.maturity_summary",
    }.isdisjoint(missing_paths)
    assert (
        load_current_strategy_project_context(
            fx["runtime"],
            task_id=fx["task"].id,
        )
        == output["revision"]
    )


@pytest.mark.parametrize(
    ("artifact_kind", "origin_tool"),
    [
        (
            SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        ),
        (
            SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        ),
        (IMPACT_CUBE_ARTIFACT_KIND, "strategy.measure_strategy_impact_cube"),
    ],
)
def test_project_context_fails_closed_when_latest_native_evidence_is_corrupt(
    tmp_path: Path,
    artifact_kind: str,
    origin_tool: str,
) -> None:
    fx = _materialize_native_v2_context_sources(
        tmp_path,
        include_older_evidence=True,
    )
    candidates = [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["kind"] == artifact_kind
        and item["origin_tool"] == origin_tool
    ]
    assert len(candidates) >= 2
    latest = max(candidates, key=lambda item: (item["created_at"], item["id"]))
    Path(latest["path"]).write_bytes(b"{}")

    with pytest.raises(StrategyError, match="artifact|bytes|hash|canonical"):
        run_materialize_project_context(
            fx["request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert all(
        item["kind"] != PROJECT_CONTEXT_ARTIFACT_KIND
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
    )


def test_project_context_fails_closed_when_latest_native_origin_is_changed(
    tmp_path: Path,
) -> None:
    fx = _materialize_native_v2_context_sources(
        tmp_path,
        include_older_evidence=True,
    )
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET origin_tool = ? WHERE id = ?",
            ("strategy.materialize_sample_design_v2", fx["bundle_record"]["id"]),
        )
        conn.commit()

    with pytest.raises(StrategyError, match="artifact origin changed"):
        run_materialize_project_context(
            fx["request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert all(
        item["kind"] != PROJECT_CONTEXT_ARTIFACT_KIND
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
    )


def test_project_context_does_not_fallback_to_older_validated_impact_cube(
    tmp_path: Path,
) -> None:
    fx = _materialize_native_v2_context_sources(
        tmp_path,
        include_older_evidence=True,
        older_impact_partitions=("development", "validation", "oot"),
        latest_impact_partitions=("development",),
    )

    output = run_materialize_project_context(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )

    approval = output["revision"]["state"]["current_project_snapshot"][
        "status_fields"
    ]["approval"]
    assert approval["availability"] == "unavailable"
    assert approval["value"] is None
