from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from pathlib import Path
import sqlite3

import pytest

from marvis.db import TaskRepository
from marvis.output.strategy_report_bundle import render_strategy_report_bundle
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_impact_tools import run_measure_pool_impact
from marvis.packs.strategy.pool_tools import (
    load_current_strategy_candidate_pool_artifact,
)
from marvis.packs.strategy.project_context_tools import (
    load_current_strategy_project_context_artifact,
    run_materialize_project_context,
)
from marvis.packs.strategy.report_bundle_tools import (
    BUILD_STRATEGY_REPORT_BUNDLE_V2_AUDIT_KIND,
    BUILD_STRATEGY_REPORT_BUNDLE_V2_TOOL_SCHEMA_VERSION,
    run_build_strategy_report_bundle_v2,
    validate_build_strategy_report_bundle_v2_tool_output,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    run_materialize_sample_design_v2,
)
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy_reports import (
    STRATEGY_REPORT_ORIGIN_TOOL,
    STRATEGY_REPORT_OUTPUT_KINDS,
    StrategyReportRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
import marvis.packs.strategy.report_bundle_tools as report_tools
from test_strategy_pool_impact_tools import _setup as _impact_setup


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _setup(tmp_path: Path) -> dict:
    fixture = _impact_setup(tmp_path, partitioned=True)
    impact_output = run_measure_pool_impact(
        fixture["request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    v2_output = run_materialize_sample_design_v2(
        {
            "legacy_sample_design_ref": fixture["sample_design_ref"],
            "relationship": "nested_same_cohort",
            "scope": "exploration_only",
            "approval_population": {"inclusion": None, "exclusion": None},
            "risk_population": {"inclusion": None, "exclusion": None},
            "partitioning": {
                "method": "predicate_ast",
                "selectors": {
                    "development": _eq("sample_split", "development"),
                    "validation": _eq("sample_split", "validation"),
                    "oot": _eq("sample_split", "oot"),
                },
            },
            "maturity": {
                "status": "unavailable",
                "performance_window_days": None,
                "cutoff_date": None,
                "reason": "No row-level application date field was supplied.",
            },
            "performance_window": {"status": "provided", "days": 90},
            "observation_window": {
                "status": "unavailable",
                "start": None,
                "end": None,
            },
            "field_bindings": {
                "entity_field": None,
                "time_field": None,
                "group_field": None,
                "month_field": "apply_month",
                "weight_field": None,
                "loan_amount_field": "loan_amount",
                "overdue_amount_field": "overdue_amount",
            },
            "historical_score": {
                "status": "unavailable",
                "column": None,
                "direction": None,
                "reason": "No historical score field was supplied.",
            },
            "policy": {
                "minimum_partition_count": 1,
                "minimum_bad_count": 1,
                "minimum_label_coverage": 0.5,
                "minimum_historical_score_coverage": 0.0,
                "maximum_group_coverage_gap": 1.0,
                "diagnostic_severities": {
                    "entity_overlap": "warn",
                    "temporal_oot": "warn",
                    "risk_outside_approval": "fail",
                    "maturity": "fail",
                    "label_coverage": "fail",
                    "historical_score_coverage": "warn",
                    "group_coverage_gap": "warn",
                    "sufficiency": "warn",
                },
            },
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    message = TaskRepository(fixture["settings"].db_path).add_agent_message(
        fixture["task"].id,
        role="user",
        stage="chat",
        content="生成当前准入策略的受治理报告，历史补充材料暂时没有。",
    )
    run_materialize_project_context(
        {
            "expected_revision": 0,
            "expected_revision_id": None,
            "expected_state_hash": None,
            "user_message_ref": {
                "message_id": message["id"],
                "content_hash": hashlib.sha256(
                    message["content"].encode("utf-8")
                ).hexdigest(),
            },
            "as_of": "2026-07-23",
            "scope": "贷前准入策略",
            "business_context": {"project.goal": "准入策略迭代"},
            "explicit_unavailable": ["historical_strategy_reviews"],
            "external_report_filenames": [],
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    records = TaskArtifactRepository(
        fixture["settings"].db_path
    ).list_for_task(fixture["task"].id)
    membership_record = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle_record = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    )
    pool = load_current_strategy_candidate_pool_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        strategy_type="approval",
    )
    impact_artifact = impact_output["artifacts"][0]
    design = v2_output["bundle"]["sample_design"]
    project_context = load_current_strategy_project_context_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
    )
    assert project_context is not None
    request = {
        "title": "准入策略迭代报告",
        "status": "partial",
        "report_revision": 1,
        "previous_report_id": None,
        "previous_report_content_hash": None,
        "generated_at": "2026-07-23T08:00:00+00:00",
        "project_context_ref": {
            "artifact_id": project_context.artifact_id,
            "expected_artifact_content_hash": (
                project_context.artifact_content_hash
            ),
            "expected_revision": project_context.revision["revision"],
            "expected_revision_id": project_context.revision["revision_id"],
            "expected_state_hash": project_context.revision["state_hash"],
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
            "expected_bundle_id": v2_output["bundle"]["bundle_id"],
            "expected_sample_design_id": design["sample_design_id"],
            "expected_sample_design_content_hash": design["content_hash"],
        },
        "candidate_pool_ref": {
            "strategy_type": pool.strategy_type,
            "expected_pool_revision": pool.pool["revision"],
            "expected_pool_snapshot_hash": pool.pool["snapshot_hash"],
            "expected_artifact_id": pool.artifact_id,
            "expected_artifact_content_hash": pool.artifact_content_hash,
        },
        "pool_impact_ref": {
            "artifact_id": impact_artifact["artifact_id"],
            "expected_artifact_content_hash": impact_artifact["content_hash"],
            "expected_assessment_id": impact_output["assessment_id"],
            "expected_assessment_content_hash": impact_output["content_hash"],
        },
        "strategy_identity": None,
        "model_evidence_ref": None,
        "training_evidence_ref": None,
        "score_evidence_ref": None,
    }
    return {
        **fixture,
        "impact_output": impact_output,
        "v2_output": v2_output,
        "request": request,
    }


def _run(fixture: dict) -> dict:
    return run_build_strategy_report_bundle_v2(
        fixture["request"],
        fixture["ctx"],
        fixture["runtime"],
    )


def _report_rows(fixture: dict) -> list[dict]:
    return [
        item
        for item in fixture["runtime"].task_artifacts.list_for_task(
            fixture["task"].id
        )
        if item["kind"] in set(STRATEGY_REPORT_OUTPUT_KINDS.values())
    ]


def _audit_rows(fixture: dict) -> list[sqlite3.Row]:
    with fixture["runtime"].task_artifacts.transaction() as conn:
        return conn.execute(
            "SELECT * FROM audit WHERE kind = ? ORDER BY at, id",
            (BUILD_STRATEGY_REPORT_BUNDLE_V2_AUDIT_KIND,),
        ).fetchall()


def test_build_report_bundle_publishes_three_exact_governed_outputs(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    output = _run(fixture)

    assert output["schema_version"] == (
        BUILD_STRATEGY_REPORT_BUNDLE_V2_TOOL_SCHEMA_VERSION
    )
    assert validate_build_strategy_report_bundle_v2_tool_output(output) == output
    rendered = render_strategy_report_bundle(output["bundle"])
    rows = _report_rows(fixture)
    assert {row["kind"] for row in rows} == set(
        STRATEGY_REPORT_OUTPUT_KINDS.values()
    )
    assert all(row["origin_tool"] == STRATEGY_REPORT_ORIGIN_TOOL for row in rows)
    for artifact in output["artifacts"]:
        row = next(item for item in rows if item["id"] == artifact["artifact_id"])
        assert Path(row["path"]).read_bytes() == rendered[artifact["format"]]
        assert row["provenance"]["report_id"] == output["report_id"]
        assert row["provenance"]["bundle_content_sha256"] == output["content_hash"]
        assert artifact["download_url"].endswith(
            f"?expected_content_hash={artifact['content_hash']}"
        )
    current = StrategyReportRepository(
        fixture["settings"].db_path
    ).get_current(task_id=fixture["task"].id, strategy_id=None)
    assert current is not None
    assert current["bundle"] == output["bundle"]
    audits = _audit_rows(fixture)
    assert len(audits) == 1
    assert audits[0]["target_ref"] == output["report_id"]

    tool = next(
        item
        for item in load_manifest(
            Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
            builtin=True,
        ).tools
        if item.name == "build_report_bundle_v2"
    )
    validate_against_schema(
        fixture["request"],
        tool.input_schema,
        label="report bundle input",
    )
    validate_against_schema(output, tool.output_schema, label="report bundle output")


def test_first_report_revision_defaults_omitted_previous_head_to_null(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    fixture["request"].pop("previous_report_id")
    fixture["request"].pop("previous_report_content_hash")

    output = _run(fixture)

    assert output["report_revision"] == 1
    assert output["bundle"]["previous_report_id"] is None
    assert validate_build_strategy_report_bundle_v2_tool_output(output) == output
    tool = next(
        item
        for item in load_manifest(
            Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
            builtin=True,
        ).tools
        if item.name == "build_report_bundle_v2"
    )
    validate_against_schema(
        fixture["request"],
        tool.input_schema,
        label="report bundle input without previous head",
    )


def test_build_report_bundle_exact_retry_is_idempotent_and_validator_is_strict(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    first = _run(fixture)
    replay = _run(fixture)

    assert replay == first
    assert len(_report_rows(fixture)) == 3
    assert len(_audit_rows(fixture)) == 1
    with fixture["runtime"].task_artifacts.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_report_revisions"
        ).fetchone()[0] == 1

    extra = deepcopy(first)
    extra["caller_metric"] = 0.99
    with pytest.raises(StrategyError, match="output envelope"):
        validate_build_strategy_report_bundle_v2_tool_output(extra)
    forged = deepcopy(first)
    forged["artifacts"][0]["content_hash"] = "f" * 64
    with pytest.raises(StrategyError, match="artifact"):
        validate_build_strategy_report_bundle_v2_tool_output(forged)
    forged_url = deepcopy(first)
    forged_url["artifacts"][0]["download_url"] = forged_url["artifacts"][0][
        "download_url"
    ].replace(
        forged_url["artifacts"][0]["content_hash"],
        "0" * 64,
    )
    with pytest.raises(StrategyError, match="artifact"):
        validate_build_strategy_report_bundle_v2_tool_output(forged_url)


def test_build_report_bundle_rolls_back_files_rows_publication_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(
        fixture["runtime"].repo,
        "write_audit_on_connection",
        fail_audit,
    )
    with pytest.raises(RuntimeError, match="audit unavailable"):
        _run(fixture)

    assert _report_rows(fixture) == []
    assert _audit_rows(fixture) == []
    with fixture["runtime"].task_artifacts.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_report_revisions"
        ).fetchone()[0] == 0
    report_root = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_reports"
    )
    assert not any(report_root.rglob("report.*")) if report_root.exists() else True


def test_build_report_bundle_identical_concurrent_writers_share_one_revision(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    def execute() -> dict:
        runtime = strategy_tools._runtime(fixture["ctx"])
        return run_build_strategy_report_bundle_v2(
            fixture["request"],
            fixture["ctx"],
            runtime,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: execute(), range(2)))

    assert results[0] == results[1]
    assert len(_report_rows(fixture)) == 3
    assert len(_audit_rows(fixture)) == 1
    with fixture["runtime"].task_artifacts.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_report_revisions"
        ).fetchone()[0] == 1


def test_build_report_bundle_revalidates_source_tamper_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    original_render = report_tools.render_strategy_report_bundle
    impact_path = Path(
        next(
            item
            for item in fixture["runtime"].task_artifacts.list_for_task(
                fixture["task"].id
            )
            if item["id"]
            == fixture["impact_output"]["artifacts"][0]["artifact_id"]
        )["path"]
    )

    def tampering_render(bundle):
        rendered = original_render(bundle)
        impact_path.write_bytes(b"{}")
        return rendered

    monkeypatch.setattr(
        report_tools,
        "render_strategy_report_bundle",
        tampering_render,
    )
    with pytest.raises(StrategyError, match="impact|artifact|hash|bytes"):
        _run(fixture)

    assert _report_rows(fixture) == []
    assert _audit_rows(fixture) == []


def test_build_report_bundle_never_rebinds_a_planned_project_context(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    previous = load_current_strategy_project_context_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
    )
    assert previous is not None
    message = TaskRepository(fixture["settings"].db_path).add_agent_message(
        fixture["task"].id,
        role="user",
        stage="chat",
        content="补充项目目标：控制准入风险并保持通过率稳定。",
    )
    run_materialize_project_context(
        {
            "expected_revision": previous.revision["revision"],
            "expected_revision_id": previous.revision["revision_id"],
            "expected_state_hash": previous.revision["state_hash"],
            "user_message_ref": {
                "message_id": message["id"],
                "content_hash": hashlib.sha256(
                    message["content"].encode("utf-8")
                ).hexdigest(),
            },
            "as_of": "2026-07-23",
            "scope": "贷前准入策略",
            "business_context": {
                "project.goal": "控制准入风险并保持通过率稳定"
            },
            "explicit_unavailable": [],
            "external_report_filenames": [],
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    with pytest.raises(
        StrategyError,
        match="project context.*exact planned revision",
    ):
        _run(fixture)

    assert _report_rows(fixture) == []
    assert _audit_rows(fixture) == []


def test_build_report_bundle_detects_output_tamper_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    original_register = fixture["runtime"].task_artifacts.register_on_connection
    calls = 0

    def tampering_register(*args, **kwargs):
        nonlocal calls
        record = original_register(*args, **kwargs)
        calls += 1
        if calls == 3:
            Path(record["path"]).write_bytes(b"tampered")
        return record

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "register_on_connection",
        tampering_register,
    )
    with pytest.raises(
        StrategyError,
        match="report output|artifact file is invalid|bytes|hash",
    ):
        _run(fixture)

    assert _report_rows(fixture) == []
    assert _audit_rows(fixture) == []


def test_build_report_bundle_never_follows_existing_output_symlink(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = _run(fixture)
    json_artifact = next(
        item for item in output["artifacts"] if item["format"] == "json"
    )
    row = next(
        item
        for item in _report_rows(fixture)
        if item["id"] == json_artifact["artifact_id"]
    )
    path = Path(row["path"])
    target = tmp_path / "outside.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(
        StrategyError,
        match="must stay under|symlink|regular|unavailable|invalid",
    ):
        _run(fixture)

    assert len(_audit_rows(fixture)) == 1
