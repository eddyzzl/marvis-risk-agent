from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository, connect, init_db
from marvis.domain import StrategyTaskInput, TaskCreate
from marvis.files import sha256_file
from marvis.packs.strategy import monitor_tools as strategy_monitor_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    canonical_economics_bindings_hash,
)
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository
from marvis.settings import build_settings


_DISPOSITION_AUDIT_KIND = "strategy.monitoring.disposition"


def _spec() -> dict:
    return {
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": "reject-high-risk",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "score",
                    "operator": ">=",
                    "value": 700,
                },
                "action": {"type": "reject", "reason_code": "HIGH_RISK"},
            }
        ],
    }


def _fixture(tmp_path: Path, *, level: str = "red") -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task_repo = TaskRepository(settings.db_path)
    task = task_repo.create_task(
        TaskCreate(
            model_name="approval monitoring disposition",
            model_version="v1",
            validator="risk-owner",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
            target_col="bad",
            score_col="score",
            strategy_input=StrategyTaskInput(strategy_type="approval"),
        )
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    source_path = tmp_path / "monitoring.parquet"
    pd.DataFrame(
        {"score": [500.0, 650.0, 800.0], "bad": [0, 1, 1]}
    ).to_parquet(source_path, index=False)
    dataset = registry.register_existing(
        source_path,
        task_id=task.id,
        role="strategy.monitoring",
    )
    dataset_hash = sha256_file(registry.resolve_path(dataset.id))
    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = ? WHERE id = ?",
            (dataset_hash, dataset.id),
        )
    dataset = DatasetRepository(settings.db_path).get_dataset(dataset.id)
    assert dataset is not None
    strategy = build_strategy_from_spec(_spec(), description="adopted champion")
    strategies = StrategyRepository(settings.db_path)
    strategies.create_strategy(task.id, strategy)
    strategies.adopt_strategy_with_audit(
        strategy.id,
        reason="committee adopted champion",
        audit={
            "kind": "strategy.adopt.fixture",
            "target_ref": strategy.id,
            "outcome": "succeeded",
            "detail": {"task_id": task.id},
        },
    )
    effect_hash = strategies.get_strategy_spec_hash(strategy.id)
    assert effect_hash is not None
    monitoring = StrategyMonitoringRepository(settings.db_path)
    plan = monitoring.create_plan(
        MonitoringPlan(
            strategy_id=strategy.id,
            version=1,
            thresholds={
                "approval_rate": {
                    "label": "审批率",
                    "metric": "approval_rate",
                    "direction": "min",
                    "warn": 0.65,
                    "fail": 0.55,
                },
                "approved_bad_rate": {
                    "label": "通过客群坏率",
                    "metric": "approved_bad_rate",
                    "direction": "max",
                    "warn": 0.08,
                    "fail": 0.12,
                },
            },
            expectation_baseline={"strategy_effect_hash": effect_hash},
        ),
        expected_revision=0,
    )
    assert dataset.content_hash == dataset_hash
    run_result = {
        "strategy_id": strategy.id,
        "dataset_id": dataset.id,
        "overall_level": level,
        "checks": [
            {
                "id": "approved_bad_rate",
                "label": "通过客群坏率",
                "metric": "approved_bad_rate",
                "direction": "max",
                "warn": 0.08,
                "fail": 0.12,
                "value": 0.20 if level == "red" else 0.05,
                "level": level,
            }
        ],
        "metrics": {"approved_bad_rate": 0.20 if level == "red" else 0.05},
        "economics": {},
    }
    run = monitoring.create_run(
        strategy_id=strategy.id,
        monitoring_plan_id=plan.id,
        expected_plan_revision=plan.revision,
        expected_plan_payload_hash=plan.payload_hash,
        dataset_id=dataset.id,
        dataset_content_hash=dataset.content_hash,
        strategy_effect_hash=effect_hash,
        economics_binding_hash=canonical_economics_bindings_hash({}),
        result=run_result,
        overall_level=level,
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    return {
        "settings": settings,
        "task": task,
        "task_repo": task_repo,
        "registry": registry,
        "dataset": dataset,
        "strategy": strategy,
        "strategies": strategies,
        "monitoring": monitoring,
        "plan": plan,
        "run": run,
        "ctx": ctx,
    }


def _inputs(fx: dict, *, disposition="observe", **overrides) -> dict:
    payload = {
        "strategy_id": fx["strategy"].id,
        "monitoring_run_id": fx["run"].id,
        "expected_plan_id": fx["plan"].id,
        "expected_plan_revision": fx["plan"].revision,
        "expected_plan_hash": fx["plan"].payload_hash,
        "disposition": disposition,
        "reason": "risk owner reviewed the persisted red evidence",
    }
    payload.update(overrides)
    return payload


def _disposition_audits(fx: dict) -> list[dict]:
    with connect(fx["settings"].db_path) as conn:
        rows = conn.execute(
            "SELECT target_ref, outcome, detail_json FROM audit WHERE kind = ? ORDER BY at, id",
            (_DISPOSITION_AUDIT_KIND,),
        ).fetchall()
    return [
        {
            "target_ref": str(row["target_ref"]),
            "outcome": str(row["outcome"]),
            "detail": json.loads(str(row["detail_json"])),
        }
        for row in rows
    ]


def _output_schema() -> dict:
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    return next(
        tool.output_schema
        for tool in manifest.tools
        if tool.name == "apply_monitoring_disposition"
    )


def _input_schema() -> dict:
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    return next(
        tool.input_schema
        for tool in manifest.tools
        if tool.name == "apply_monitoring_disposition"
    )


def _report_tool():
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    return next(
        tool for tool in manifest.tools if tool.name == "render_monitoring_report"
    )


def test_disposition_manifest_requires_human_gate_without_effect_authorization() -> None:
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tool = next(
        tool
        for tool in manifest.tools
        if tool.name == "apply_monitoring_disposition"
    )

    assert tool.entrypoint == "tool_apply_monitoring_disposition"
    assert tool.policy.human_decision_gate == "required"
    assert tool.policy.effect_authorization == "none"
    assert {
        "read:task",
        "read:dataset",
        "read:strategy",
        "write:artifact",
        "write:task",
        "write:strategy",
    }.issubset(tool.side_effects)


def test_observe_records_one_immutable_disposition_and_rejects_replay(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    payload = _inputs(fx, threshold_patch=None)
    validate_against_schema(payload, _input_schema(), label="observe disposition input")

    output = strategy_tools.tool_apply_monitoring_disposition(
        payload, fx["ctx"]
    )

    assert output["status"] == "observed"
    assert output["disposition"] == "observe"
    assert output["source_monitoring_run_id"] == fx["run"].id
    assert output["resolved_monitoring_run_id"] == fx["run"].id
    assert output["overall_level"] == "red"
    assert output["checks"] == fx["run"].result["checks"]
    assert output["monitoring_plan_id"] == fx["plan"].id
    assert output["monitoring_plan_revision"] == fx["plan"].revision
    assert output["monitoring_plan_hash"] == fx["plan"].payload_hash
    validate_against_schema(output, _output_schema(), label="observe disposition")
    audits = _disposition_audits(fx)
    assert len(audits) == 1
    assert audits[0]["target_ref"] == fx["run"].id
    assert audits[0]["detail"]["reason"] == _inputs(fx)["reason"]
    assert "metrics" not in audits[0]["detail"]

    with pytest.raises(StrategyError, match="already has a disposition"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx), fx["ctx"]
        )
    assert len(_disposition_audits(fx)) == 1


def test_monitoring_report_uses_only_verified_receipt_ids_and_is_idempotent(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    strategy_tools.tool_apply_monitoring_disposition(
        _inputs(fx, threshold_patch=None), fx["ctx"]
    )
    report_input = {
        "strategy_id": fx["strategy"].id,
        "source_monitoring_run_id": fx["run"].id,
    }
    report_tool = _report_tool()
    assert "write:artifact" in report_tool.side_effects
    validate_against_schema(
        report_input,
        report_tool.input_schema,
        label="monitoring report receipt input",
    )

    first = strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])
    second = strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])

    validate_against_schema(
        first,
        report_tool.output_schema,
        label="verified monitoring report",
    )
    assert second["artifact_id"] == first["artifact_id"]
    assert second["report_path"] == first["report_path"]
    assert first["overall_level"] == fx["run"].overall_level
    assert first["next_action"]["action"] == "observe"
    markdown = Path(first["report_path"]).read_text(encoding="utf-8")
    assert "策略监控报告" in markdown
    assert "维持并观察" in markdown
    artifacts = [
        artifact
        for artifact in fx["strategies"].list_strategy_artifacts(
            fx["strategy"].id
        )
        if artifact["kind"] == "monitoring_report_md"
    ]
    assert len(artifacts) == 1


def test_monitoring_report_markdown_escapes_dynamic_evidence_content() -> None:
    markdown = strategy_monitor_tools._render_report_markdown(
        strategy_id="strategy-safe",
        version=1,
        overall_level="red",
        checks=[
            {
                "id": "risk_check",
                "label": "风险|检查\n## 伪标题",
                "level": "red",
                "value": "high|risk",
                "message": "第一行\n| --- | 注入",
            }
        ],
        timeline=[
            {
                "at": "2026-07-19\n## 伪时间线",
                "overall_level": "red",
                "row_count": 10,
            }
        ],
        next_action={
            "prompt": "处置已记录。",
            "action": "observe",
            "reason": "包含 `反引号`\n## 伪处置标题 | 值",
        },
    )

    assert "风险\\|检查<br>## 伪标题" in markdown
    assert "high\\|risk" in markdown
    assert "第一行<br>\\| --- \\| 注入" in markdown
    assert "2026-07-19<br>## 伪时间线" in markdown
    assert "\n## 伪标题" not in markdown
    assert "\n## 伪时间线" not in markdown
    assert "\n## 伪处置标题" not in markdown
    assert "包含 `反引号` ## 伪处置标题 | 值" in markdown


def test_monitoring_report_rejects_caller_verdicts_and_tampered_receipts(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    strategy_tools.tool_apply_monitoring_disposition(
        _inputs(fx, threshold_patch=None), fx["ctx"]
    )
    report_input = {
        "strategy_id": fx["strategy"].id,
        "source_monitoring_run_id": fx["run"].id,
    }
    with pytest.raises(StrategyError, match="unexpected fields"):
        strategy_tools.tool_render_monitoring_report(
            {
                **report_input,
                "overall_level": "green",
                "checks": [],
            },
            fx["ctx"],
        )

    with connect(fx["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT id, detail_json FROM audit WHERE kind = ? AND target_ref = ?",
            (_DISPOSITION_AUDIT_KIND, fx["run"].id),
        ).fetchone()
        assert row is not None
        detail = json.loads(str(row["detail_json"]))
        detail["reason"] = "tampered after disposition"
        conn.execute(
            "UPDATE audit SET detail_json = ? WHERE id = ?",
            (
                json.dumps(detail, ensure_ascii=False, sort_keys=True),
                str(row["id"]),
            ),
        )

    with pytest.raises(StrategyError, match="receipt hash does not match"):
        strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])


def test_monitoring_report_rejects_hash_valid_but_semantically_invalid_receipt(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    strategy_tools.tool_apply_monitoring_disposition(
        _inputs(fx, threshold_patch=None), fx["ctx"]
    )
    with connect(fx["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT id, detail_json FROM audit WHERE kind = ? AND target_ref = ?",
            (_DISPOSITION_AUDIT_KIND, fx["run"].id),
        ).fetchone()
        assert row is not None
        detail = json.loads(str(row["detail_json"]))
        detail["status"] = "acknowledged"
        conn.execute(
            "UPDATE audit SET detail_json = ?, inputs_hash = ? WHERE id = ?",
            (
                json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
                strategy_tools._monitoring_disposition_inputs_hash(detail),
                str(row["id"]),
            ),
        )

    with pytest.raises(StrategyError, match="status is inconsistent"):
        strategy_tools.tool_render_monitoring_report(
            {
                "strategy_id": fx["strategy"].id,
                "source_monitoring_run_id": fx["run"].id,
            },
            fx["ctx"],
        )


def test_monitoring_report_stays_stable_after_future_monitoring_runs(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    strategy_tools.tool_apply_monitoring_disposition(
        _inputs(fx, threshold_patch=None), fx["ctx"]
    )
    report_input = {
        "strategy_id": fx["strategy"].id,
        "source_monitoring_run_id": fx["run"].id,
    }
    first = strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])

    future_path = tmp_path / "future-monitoring.parquet"
    pd.DataFrame({"score": [450.0, 550.0], "bad": [0, 1]}).to_parquet(
        future_path,
        index=False,
    )
    future_dataset = fx["registry"].register_existing(
        future_path,
        task_id=fx["task"].id,
        role="strategy.monitoring",
    )
    future_hash = sha256_file(fx["registry"].resolve_path(future_dataset.id))
    with connect(fx["settings"].db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = ? WHERE id = ?",
            (future_hash, future_dataset.id),
        )
    fx["monitoring"].create_run(
        strategy_id=fx["strategy"].id,
        monitoring_plan_id=fx["plan"].id,
        expected_plan_revision=fx["plan"].revision,
        expected_plan_payload_hash=fx["plan"].payload_hash,
        dataset_id=future_dataset.id,
        dataset_content_hash=future_hash,
        strategy_effect_hash=fx["run"].strategy_effect_hash,
        economics_binding_hash=fx["run"].economics_binding_hash,
        result={
            "overall_level": "green",
            "checks": [{"id": "approval_rate", "level": "green"}],
        },
        overall_level="green",
        created_at="9999-01-01T00:00:00+00:00",
    )

    second = strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])
    assert second["artifact_id"] == first["artifact_id"]
    assert second["report_path"] == first["report_path"]
    assert second["timeline"] == first["timeline"]


def test_new_version_report_survives_valid_child_lifecycle_progress(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    disposition = strategy_tools.tool_apply_monitoring_disposition(
        _inputs(fx, disposition="new_version"), fx["ctx"]
    )
    report_input = {
        "strategy_id": fx["strategy"].id,
        "source_monitoring_run_id": fx["run"].id,
    }
    first = strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])

    fx["strategies"].adopt_strategy_with_audit(
        disposition["new_strategy_id"],
        reason="child strategy completed its own review",
        audit={
            "kind": "strategy.adopt.fixture",
            "target_ref": disposition["new_strategy_id"],
            "outcome": "succeeded",
            "detail": {"task_id": disposition["new_task_id"]},
        },
    )

    second = strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])
    assert second["artifact_id"] == first["artifact_id"]
    assert second["report_path"] == first["report_path"]


def test_monitoring_report_rejects_cross_task_receipt_access(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    strategy_tools.tool_apply_monitoring_disposition(
        _inputs(fx, threshold_patch=None), fx["ctx"]
    )
    foreign_task = fx["task_repo"].create_task(
        TaskCreate(
            model_name="foreign report reader",
            model_version="v1",
            validator="qa",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    foreign_ctx = ToolContext(
        task_id=foreign_task.id,
        seed=0,
        datasets_root=fx["settings"].datasets_dir,
        workspace=fx["settings"].workspace,
    )

    with pytest.raises(StrategyError, match="strategy not found"):
        strategy_tools.tool_render_monitoring_report(
            {
                "strategy_id": fx["strategy"].id,
                "source_monitoring_run_id": fx["run"].id,
            },
            foreign_ctx,
        )


def test_new_version_creates_real_task_strategy_and_dataset_and_rejects_replay(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)

    output = strategy_tools.tool_apply_monitoring_disposition(
        _inputs(fx, disposition="new_version"), fx["ctx"]
    )

    assert output["status"] == "new_version_created"
    assert output["resolved_monitoring_run_id"] == fx["run"].id
    assert output["new_task_id"]
    assert output["new_strategy_id"]
    assert output["new_dataset_id"]
    child_task = fx["task_repo"].get_task(output["new_task_id"])
    assert child_task.strategy_input is not None
    assert child_task.strategy_input.baseline_strategy_id == fx["strategy"].id
    child_meta = fx["strategies"].get_strategy_meta(output["new_strategy_id"])
    assert child_meta["task_id"] == output["new_task_id"]
    assert child_meta["status"] == "draft"
    child_dataset = DatasetRepository(fx["settings"].db_path).get_dataset(
        output["new_dataset_id"]
    )
    assert child_dataset is not None
    assert child_dataset.task_id == output["new_task_id"]
    assert child_dataset.content_hash == fx["dataset"].content_hash
    validate_against_schema(output, _output_schema(), label="new-version disposition")
    assert len(_disposition_audits(fx)) == 1

    with pytest.raises(StrategyError, match="already has a disposition"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx, disposition="new_version"), fx["ctx"]
        )
    with connect(fx["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE parent_strategy_id = ?",
            (fx["strategy"].id,),
        ).fetchone()[0] == 1


@pytest.mark.parametrize(
    "override",
    [
        {"expected_plan_id": "stale-plan"},
        {"expected_plan_revision": 99},
        {"expected_plan_hash": "0" * 64},
    ],
)
def test_disposition_rejects_stale_plan_cas_without_writes(
    tmp_path: Path,
    override: dict,
) -> None:
    fx = _fixture(tmp_path)

    with pytest.raises(StrategyError, match="monitoring plan CAS"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx, **override), fx["ctx"]
        )

    assert _disposition_audits(fx) == []


def test_disposition_rejects_a_non_latest_run(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    latest_path = tmp_path / "latest-monitoring.parquet"
    pd.DataFrame({"score": [400.0, 900.0], "bad": [0, 1]}).to_parquet(
        latest_path, index=False
    )
    latest_dataset = fx["registry"].register_existing(
        latest_path,
        task_id=fx["task"].id,
        role="strategy.monitoring",
    )
    latest_hash = sha256_file(fx["registry"].resolve_path(latest_dataset.id))
    with connect(fx["settings"].db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = ? WHERE id = ?",
            (latest_hash, latest_dataset.id),
        )
    fx["monitoring"].create_run(
        strategy_id=fx["strategy"].id,
        monitoring_plan_id=fx["plan"].id,
        expected_plan_revision=fx["plan"].revision,
        expected_plan_payload_hash=fx["plan"].payload_hash,
        dataset_id=latest_dataset.id,
        dataset_content_hash=latest_hash,
        strategy_effect_hash=fx["run"].strategy_effect_hash,
        economics_binding_hash=fx["run"].economics_binding_hash,
        result={
            "overall_level": "red",
            "checks": [{"id": "approval_rate", "level": "red"}],
        },
        overall_level="red",
        created_at="9999-01-01T00:00:00+00:00",
    )

    with pytest.raises(StrategyError, match="latest monitoring run"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx), fx["ctx"]
        )
    assert _disposition_audits(fx) == []


def test_non_red_requires_null_acknowledgement_and_red_rejects_null(
    tmp_path: Path,
) -> None:
    green = _fixture(tmp_path / "green", level="green")
    output = strategy_tools.tool_apply_monitoring_disposition(
        _inputs(green, disposition=None), green["ctx"]
    )
    assert output["status"] == "acknowledged"
    assert output["disposition"] is None
    assert output["overall_level"] == "green"

    red = _fixture(tmp_path / "red", level="red")
    with pytest.raises(StrategyError, match="red monitoring run requires"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(red, disposition=None), red["ctx"]
        )
    with pytest.raises(StrategyError, match="only red monitoring runs"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(green, disposition="observe"), green["ctx"]
        )


@pytest.mark.parametrize(
    ("threshold_patch", "message"),
    [
        ({"unknown_check": {"warn": 0.1}}, "unknown monitoring check"),
        ({"approval_rate": {"metric": "other"}}, "only change warn/fail"),
        ({"approval_rate": {"direction": "max"}}, "only change warn/fail"),
        ({"approval_rate": {"warn": True}}, "finite number"),
        ({"approval_rate": {"warn": float("nan")}}, "finite number"),
        (
            {"approval_rate": {"warn": 0.65}},
            "must change at least one warn/fail value",
        ),
        ({"approval_rate": {"warn": 0.50}}, "min direction requires warn >= fail"),
        ({"approved_bad_rate": {"fail": 0.05}}, "max direction requires warn <= fail"),
    ],
)
def test_adjust_threshold_rejects_unsafe_patch_without_new_plan(
    tmp_path: Path,
    threshold_patch: dict,
    message: str,
) -> None:
    fx = _fixture(tmp_path)

    with pytest.raises(StrategyError, match=message):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(
                fx,
                disposition="adjust_threshold",
                threshold_patch=threshold_patch,
            ),
            fx["ctx"],
        )

    assert len(fx["monitoring"].list_plans(fx["strategy"].id)) == 1
    assert len(fx["monitoring"].list_runs(fx["strategy"].id)) == 1
    assert _disposition_audits(fx) == []


def test_valid_threshold_patch_builds_candidate_then_stops_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    captured: dict = {}

    def stop_at_rerun_boundary(*, candidate_plan, **_kwargs):
        captured["candidate"] = candidate_plan
        raise StrategyError("atomic candidate monitoring rerun is not implemented")

    monkeypatch.setattr(
        strategy_tools,
        "_rerun_monitoring_candidate_atomically",
        stop_at_rerun_boundary,
    )

    with pytest.raises(StrategyError, match="atomic candidate monitoring rerun"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(
                fx,
                disposition="adjust_threshold",
                threshold_patch={"approval_rate": {"warn": 0.60}},
            ),
            fx["ctx"],
        )

    candidate = captured["candidate"]
    assert candidate.revision == 2
    assert candidate.supersedes_plan_id == fx["plan"].id
    assert candidate.monitoring_plan_id != fx["plan"].id
    assert candidate.thresholds["approval_rate"] == {
        **fx["plan"].plan.thresholds["approval_rate"],
        "warn": 0.60,
    }
    assert candidate.expectation_baseline == fx["plan"].plan.expectation_baseline
    assert candidate.economics_bindings == fx["plan"].plan.economics_bindings
    assert len(fx["monitoring"].list_plans(fx["strategy"].id)) == 1
    assert len(fx["monitoring"].list_runs(fx["strategy"].id)) == 1
    assert _disposition_audits(fx) == []


def test_adjust_threshold_atomically_creates_resolved_plan_run_and_artifact(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)

    output = strategy_tools.tool_apply_monitoring_disposition(
        _inputs(
            fx,
            disposition="adjust_threshold",
            threshold_patch={
                "approved_bad_rate": {"warn": 0.50, "fail": 0.60}
            },
        ),
        fx["ctx"],
    )

    assert output["status"] == "threshold_adjusted"
    assert output["disposition"] == "adjust_threshold"
    assert output["source_monitoring_run_id"] == fx["run"].id
    assert output["resolved_monitoring_run_id"] != fx["run"].id
    assert output["monitoring_plan_revision"] == 2
    assert output["monitoring_plan_id"] != fx["plan"].id
    assert output["overall_level"] == "green"
    assert Path(output["plan_artifact_path"]).is_file()
    validate_against_schema(output, _output_schema(), label="adjust disposition")
    assert [
        plan.revision
        for plan in fx["monitoring"].list_plans(fx["strategy"].id)
    ] == [1, 2]
    assert len(fx["monitoring"].list_runs(fx["strategy"].id)) == 2
    audits = _disposition_audits(fx)
    assert len(audits) == 1
    assert audits[0]["detail"]["disposition"] == "adjust_threshold"
    assert audits[0]["detail"]["new_monitoring_run_id"] == output[
        "resolved_monitoring_run_id"
    ]
    assert "metrics" not in audits[0]["detail"]

    report_input = {
        "strategy_id": fx["strategy"].id,
        "source_monitoring_run_id": fx["run"].id,
    }
    report = strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])
    assert report["next_action"]["action"] == "adjust_threshold"
    assert report["next_action"]["monitoring_run_id"] == output[
        "resolved_monitoring_run_id"
    ]
    markdown = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "risk owner reviewed the persisted red evidence" in markdown
    assert "approved_bad_rate" in markdown

    with connect(fx["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT id, detail_json FROM audit WHERE kind = ? AND target_ref = ?",
            (_DISPOSITION_AUDIT_KIND, fx["run"].id),
        ).fetchone()
        assert row is not None
        detail = json.loads(str(row["detail_json"]))
        detail["threshold_patch"] = {
            "approved_bad_rate": {"warn": 0.49, "fail": 0.60}
        }
        conn.execute(
            "UPDATE audit SET detail_json = ?, inputs_hash = ? WHERE id = ?",
            (
                json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
                strategy_tools._monitoring_disposition_inputs_hash(detail),
                str(row["id"]),
            ),
        )
    with pytest.raises(StrategyError, match="patch does not match plan diff"):
        strategy_tools.tool_render_monitoring_report(report_input, fx["ctx"])


def test_disposition_rejects_blank_reason_metrics_and_foreign_task(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(StrategyError, match="reason must be non-empty"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx, reason="   "), fx["ctx"]
        )
    with pytest.raises(StrategyError, match="metrics must not be supplied"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx, metrics={"bad_rate": 0.99}), fx["ctx"]
        )
    foreign_task = fx["task_repo"].create_task(
        TaskCreate(
            model_name="foreign",
            model_version="v1",
            validator="qa",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    foreign_ctx = ToolContext(
        task_id=foreign_task.id,
        seed=0,
        datasets_root=fx["settings"].datasets_dir,
        workspace=fx["settings"].workspace,
    )
    with pytest.raises(StrategyError, match="strategy not found"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx), foreign_ctx
        )
    with connect(fx["settings"].db_path) as conn:
        conn.execute(
            """
            UPDATE strategies
               SET status = 'retired', asset_status = 'retired'
             WHERE id = ?
            """,
            (fx["strategy"].id,),
        )
    with pytest.raises(StrategyError, match="requires an adopted strategy"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx), fx["ctx"]
        )
    assert _disposition_audits(fx) == []


def test_disposition_rejects_hash_valid_semantically_inconsistent_run(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    invalid_path = tmp_path / "semantic-invalid.parquet"
    pd.DataFrame({"score": [450.0, 900.0], "bad": [0, 1]}).to_parquet(
        invalid_path,
        index=False,
    )
    invalid_dataset = fx["registry"].register_existing(
        invalid_path,
        task_id=fx["task"].id,
        role="strategy.monitoring",
    )
    invalid_dataset_hash = sha256_file(
        fx["registry"].resolve_path(invalid_dataset.id)
    )
    result = {
        "overall_level": "red",
        "checks": [{"id": "approved_bad_rate", "level": "green"}],
    }
    raw = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    invalid_run_id = "run-semantic-invalid"
    with connect(fx["settings"].db_path) as conn:
        conn.execute(
            "UPDATE datasets SET content_hash = ? WHERE id = ?",
            (invalid_dataset_hash, invalid_dataset.id),
        )
        conn.execute(
            """
            INSERT INTO strategy_monitoring_runs(
                id, strategy_id, monitoring_plan_id, dataset_id,
                dataset_content_hash, strategy_effect_hash,
                economics_binding_hash, result_json, result_hash,
                overall_level, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'red', ?)
            """,
            (
                invalid_run_id,
                fx["strategy"].id,
                fx["plan"].id,
                invalid_dataset.id,
                invalid_dataset_hash,
                fx["run"].strategy_effect_hash,
                fx["run"].economics_binding_hash,
                raw,
                hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "9999-01-01T00:00:00+00:00",
            ),
        )

    with pytest.raises(StrategyError, match="semantic contract"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx, monitoring_run_id=invalid_run_id),
            fx["ctx"],
        )
    assert _disposition_audits(fx) == []


def test_new_version_and_disposition_audit_are_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)

    def fail_disposition_audit(*_args, **_kwargs):
        raise RuntimeError("injected disposition audit failure")

    monkeypatch.setattr(strategy_tools, "_write_audit_row", fail_disposition_audit)
    with pytest.raises(RuntimeError, match="injected disposition audit failure"):
        strategy_tools.tool_apply_monitoring_disposition(
            _inputs(fx, disposition="new_version"), fx["ctx"]
        )

    with connect(fx["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE parent_strategy_id = ?",
            (fx["strategy"].id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            ("strategy.monitoring.new_version_handoff",),
        ).fetchone()[0] == 0
    assert _disposition_audits(fx) == []
