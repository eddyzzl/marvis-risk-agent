from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.monitoring_plan import MonitoringPlan
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.project_context_tools import (
    PROJECT_CONTEXT_ARTIFACT_KIND,
    PROJECT_CONTEXT_TOOL_SCHEMA_VERSION,
    load_current_strategy_project_context,
    load_strategy_project_context_revision_for_audit,
    run_materialize_project_context,
    validate_materialize_project_context_tool_output,
)
import marvis.packs.strategy.project_context_tools as project_context_tools
from marvis.packs.strategy.sample_design_tools import run_materialize_sample_design
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.typed_backtest import run_typed_backtest
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository
from marvis.repositories.strategy_project_context import (
    StrategyProjectContextRepository,
)
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _setup(tmp_path: Path) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="project-context",
            model_version="dev",
            validator="qa",
            source_dir=str(source_dir),
            task_type="strategy",
        )
    )
    message = tasks.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content="本次先完成准入策略，历史材料暂时没有。",
    )
    content_hash = hashlib.sha256(message["content"].encode("utf-8")).hexdigest()
    runtime = SimpleNamespace(
        settings=settings,
        task_artifacts=TaskArtifactRepository(settings.db_path),
        project_contexts=StrategyProjectContextRepository(settings.db_path),
    )
    request = {
        "expected_revision": 0,
        "expected_revision_id": None,
        "expected_state_hash": None,
        "user_message_ref": {
            "message_id": message["id"],
            "content_hash": content_hash,
        },
        "as_of": "2026-07-22",
        "scope": "贷前准入策略",
        "business_context": {"project.channel": "自营"},
        "explicit_unavailable": ["historical_strategy_reviews"],
        "external_report_filenames": [],
    }
    return {
        "settings": settings,
        "task": task,
        "runtime": runtime,
        "ctx": SimpleNamespace(task_id=task.id),
        "request": request,
    }


def _request_bound_to_message(fx: dict, content: str, **updates) -> dict:
    message = TaskRepository(fx["settings"].db_path).add_agent_message(
        fx["task"].id,
        role="user",
        stage="chat",
        content=content,
    )
    return {
        **fx["request"],
        "user_message_ref": {
            "message_id": message["id"],
            "content_hash": hashlib.sha256(
                message["content"].encode("utf-8")
            ).hexdigest(),
        },
        **updates,
    }


def test_materialize_minimal_context_is_canonical_idempotent_and_loadable(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    first = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])
    replay = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])

    assert first["schema_version"] == PROJECT_CONTEXT_TOOL_SCHEMA_VERSION
    assert first["created"] is True
    assert replay["created"] is False
    assert replay["revision"] == first["revision"]
    assert first["context_artifact"]["kind"] == PROJECT_CONTEXT_ARTIFACT_KIND
    state = first["revision"]["state"]
    assert state["current_project_snapshot"]["scope"]["value"] == "贷前准入策略"
    assert (
        state["current_project_snapshot"]["status_fields"]["volume"]["availability"]
        == "unavailable"
    )
    assert state["current_project_snapshot"]["status_fields"]["volume"]["value"] is None
    assert {
        row["field_path"]: row["status"] for row in state["missing_information_records"]
    }["historical_strategy_reviews"] == "unavailable"
    assert (
        load_strategy_project_context_revision_for_audit(
            fx["runtime"],
            task_id=fx["task"].id,
            revision_id=first["revision"]["revision_id"],
        )
        == first["revision"]
    )
    assert (
        load_current_strategy_project_context(fx["runtime"], task_id=fx["task"].id)
        == first["revision"]
    )

    records = [
        item
        for item in fx["runtime"].task_artifacts.list_for_task(fx["task"].id)
        if item["kind"] == PROJECT_CONTEXT_ARTIFACT_KIND
    ]
    assert len(records) == 1


@pytest.mark.parametrize(
    "forbidden",
    [
        {"dataset_id": "dataset-user-picked"},
        {"strategy_id": "strategy-user-picked"},
        {"artifact_id": "artifact-user-picked"},
        {"approval_rate": 0.82},
        {"asset_status": "adopted_local"},
    ],
)
def test_materialize_rejects_user_supplied_platform_facts(
    tmp_path: Path, forbidden: dict
) -> None:
    fx = _setup(tmp_path)

    with pytest.raises(StrategyError, match="unsupported"):
        run_materialize_project_context(
            {**fx["request"], **forbidden}, fx["ctx"], fx["runtime"]
        )


def test_external_report_is_opaque_content_addressed_evidence(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    source = Path(fx["task"].source_dir) / "历史策略.xlsx"
    source.write_bytes(b"opaque-xlsx-evidence\x00\x01")
    request = _request_bound_to_message(
        fx,
        "请使用历史策略.xlsx 作为历史材料。",
        explicit_unavailable=[],
        external_report_filenames=[source.name],
    )

    output = run_materialize_project_context(request, fx["ctx"], fx["runtime"])
    replay = run_materialize_project_context(request, fx["ctx"], fx["runtime"])

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert output["external_artifacts"][0]["filename"] == f"{digest}.xlsx"
    assert replay["created"] is False
    assert replay["revision"] == output["revision"]
    copied = (
        fx["settings"].tasks_dir
        / fx["task"].id
        / "strategy_project_context_sources"
        / f"{digest}.xlsx"
    )
    assert copied.read_bytes() == source.read_bytes()
    histories = output["revision"]["state"]["historical_strategy_reviews"]
    external = next(item for item in histories if item["strategy_ref"] is None)
    assert external["external_source_refs"] == [
        {"kind": "external_report", "ref_id": digest, "content_hash": digest}
    ]
    assert "historical_strategy_reviews" not in {
        item["field_path"] for item in output["missing_information_records"]
    }


def test_same_external_content_is_reusable_across_user_messages(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    source = Path(fx["task"].source_dir) / "历史策略.pdf"
    source.write_bytes(b"same-opaque-evidence")
    first = run_materialize_project_context(
        _request_bound_to_message(
            fx,
            "请继续整理，历史材料是历史策略.pdf。",
            external_report_filenames=[source.name],
        ),
        fx["ctx"],
        fx["runtime"],
    )
    tasks = TaskRepository(fx["settings"].db_path)
    followup = tasks.add_agent_message(
        fx["task"].id,
        role="user",
        stage="chat",
        content="继续使用同一份历史策略报告 历史策略.pdf。",
    )
    followup_hash = hashlib.sha256(followup["content"].encode("utf-8")).hexdigest()

    second = run_materialize_project_context(
        {
            **fx["request"],
            "expected_revision": first["revision"]["revision"],
            "expected_revision_id": first["revision"]["revision_id"],
            "expected_state_hash": first["revision"]["state_hash"],
            "user_message_ref": {
                "message_id": followup["id"],
                "content_hash": followup_hash,
            },
            "business_context": {},
            "external_report_filenames": [source.name],
        },
        fx["ctx"],
        fx["runtime"],
    )

    assert second["created"] is True
    assert second["revision"]["revision"] == 2
    assert second["external_artifacts"] == first["external_artifacts"]


def test_external_report_must_be_grounded_in_bound_message_and_basename_unique(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    source_dir = Path(fx["task"].source_dir)
    (source_dir / "one").mkdir()
    (source_dir / "two").mkdir()
    (source_dir / "one" / "report.pdf").write_bytes(b"one")
    (source_dir / "two" / "report.pdf").write_bytes(b"two")

    with pytest.raises(StrategyError, match="not grounded"):
        run_materialize_project_context(
            {
                **fx["request"],
                "external_report_filenames": ["one/report.pdf"],
            },
            fx["ctx"],
            fx["runtime"],
        )

    ambiguous = _request_bound_to_message(
        fx,
        "请使用 report.pdf。",
        external_report_filenames=["one/report.pdf"],
    )
    with pytest.raises(StrategyError, match="exactly one"):
        run_materialize_project_context(ambiguous, fx["ctx"], fx["runtime"])

    exact = _request_bound_to_message(
        fx,
        "请使用 one/report.pdf。",
        external_report_filenames=["one/report.pdf"],
    )
    assert (
        run_materialize_project_context(exact, fx["ctx"], fx["runtime"])["created"]
        is True
    )


def test_external_reports_have_bounded_aggregate_size(
    tmp_path: Path, monkeypatch
) -> None:
    fx = _setup(tmp_path)
    source_dir = Path(fx["task"].source_dir)
    (source_dir / "one.pdf").write_bytes(b"12345")
    (source_dir / "two.pdf").write_bytes(b"67890")
    request = _request_bound_to_message(
        fx,
        "使用 one.pdf 和 two.pdf。",
        external_report_filenames=["one.pdf", "two.pdf"],
    )
    monkeypatch.setattr(project_context_tools, "MAX_EXTERNAL_REPORT_TOTAL_BYTES", 8)

    with pytest.raises(StrategyError, match="total byte limit"):
        run_materialize_project_context(request, fx["ctx"], fx["runtime"])


def test_tool_output_rejects_context_and_external_artifact_binding_tamper(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    source = Path(fx["task"].source_dir) / "evidence.pdf"
    source.write_bytes(b"opaque")
    output = run_materialize_project_context(
        _request_bound_to_message(
            fx,
            "使用 evidence.pdf。",
            external_report_filenames=[source.name],
        ),
        fx["ctx"],
        fx["runtime"],
    )
    tampered_outputs = []
    for key, replacement in (
        ("content_hash", "0" * 64),
        ("artifact_id", "1" * 64),
        ("download_url", "/api/tasks/wrong/task-artifacts/wrong/download"),
        ("filename", "wrong.json"),
    ):
        tampered = copy.deepcopy(output)
        tampered["context_artifact"][key] = replacement
        tampered_outputs.append(tampered)
    for key, replacement in (
        ("content_hash", "2" * 64),
        ("artifact_id", "3" * 64),
        ("download_url", "/api/tasks/wrong/task-artifacts/wrong/download"),
        ("filename", "wrong.pdf"),
    ):
        tampered = copy.deepcopy(output)
        tampered["external_artifacts"][0][key] = replacement
        tampered_outputs.append(tampered)

    for tampered in tampered_outputs:
        with pytest.raises(StrategyError, match="artifact|revision|source|URL"):
            validate_materialize_project_context_tool_output(tampered)


@pytest.mark.parametrize("filename", ["../escape.xlsx", "/tmp/escape.xlsx"])
def test_external_report_rejects_absolute_and_traversal_paths(
    tmp_path: Path, filename: str
) -> None:
    fx = _setup(tmp_path)

    with pytest.raises(StrategyError, match="safe relative path"):
        run_materialize_project_context(
            {**fx["request"], "external_report_filenames": [filename]},
            fx["ctx"],
            fx["runtime"],
        )


def test_external_report_rejects_symlink_legacy_xls_and_oversize(
    tmp_path: Path, monkeypatch
) -> None:
    fx = _setup(tmp_path)
    source_dir = Path(fx["task"].source_dir)
    real = source_dir / "real.xlsx"
    real.write_bytes(b"real")
    (source_dir / "linked.xlsx").symlink_to(real)
    (source_dir / "legacy.xls").write_bytes(b"legacy")
    (source_dir / "large.pdf").write_bytes(b"123456789")
    request = _request_bound_to_message(
        fx,
        "检查 linked.xlsx、legacy.xls 和 large.pdf。",
    )

    with pytest.raises(StrategyError, match="symlink"):
        run_materialize_project_context(
            {
                **request,
                "external_report_filenames": ["linked.xlsx"],
            },
            fx["ctx"],
            fx["runtime"],
        )
    with pytest.raises(StrategyError, match=r"\.xls"):
        run_materialize_project_context(
            {**request, "external_report_filenames": ["legacy.xls"]},
            fx["ctx"],
            fx["runtime"],
        )
    monkeypatch.setattr(project_context_tools, "MAX_EXTERNAL_REPORT_BYTES", 8)
    with pytest.raises(StrategyError, match="byte limit"):
        run_materialize_project_context(
            {**request, "external_report_filenames": ["large.pdf"]},
            fx["ctx"],
            fx["runtime"],
        )


def test_current_cas_no_change_does_not_create_revision_or_repeat_questions(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])
    request = {
        **fx["request"],
        "expected_revision": first["revision"]["revision"],
        "expected_revision_id": first["revision"]["revision_id"],
        "expected_state_hash": first["revision"]["state_hash"],
    }

    unchanged = run_materialize_project_context(request, fx["ctx"], fx["runtime"])

    assert unchanged["created"] is False
    assert unchanged["revision"] == first["revision"]
    first_missing = {
        item["field_path"]: (item["asked_count"], item["asked_at"])
        for item in first["missing_information_records"]
    }
    unchanged_missing = {
        item["field_path"]: (item["asked_count"], item["asked_at"])
        for item in unchanged["missing_information_records"]
    }
    assert unchanged_missing == first_missing
    with sqlite3.connect(fx["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_project_context_revisions WHERE task_id = ?",
                (fx["task"].id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind = 'strategy.project_context.materialize'"
            ).fetchone()[0]
            == 1
        )


def test_artifact_and_database_roll_back_when_audit_fails(
    tmp_path: Path, monkeypatch
) -> None:
    fx = _setup(tmp_path)
    source = Path(fx["task"].source_dir) / "rollback.pdf"
    source.write_bytes(b"rollback-external")
    request = _request_bound_to_message(
        fx,
        "使用 rollback.pdf。",
        external_report_filenames=[source.name],
    )

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(project_context_tools, "_write_audit_row", fail_audit)
    with pytest.raises(StrategyError, match="audit unavailable"):
        run_materialize_project_context(request, fx["ctx"], fx["runtime"])

    assert fx["runtime"].project_contexts.get_current(fx["task"].id) is None
    assert fx["runtime"].task_artifacts.list_for_task(fx["task"].id) == []
    task_dir = fx["settings"].tasks_dir / fx["task"].id
    assert not list(task_dir.rglob("*.json")) if task_dir.exists() else True
    assert not list(task_dir.rglob("*.pdf")) if task_dir.exists() else True


def test_staged_external_is_promoted_when_context_final_already_exists(
    tmp_path: Path, monkeypatch
) -> None:
    fx = _setup(tmp_path)
    source = Path(fx["task"].source_dir) / "existing-context.pdf"
    source.write_bytes(b"external-stage-must-promote")
    request = _request_bound_to_message(
        fx,
        "使用 existing-context.pdf。",
        external_report_filenames=[source.name],
    )
    original_stage_new_file = project_context_tools._stage_new_file

    def simulate_existing_context(uow, **kwargs):
        final_path = kwargs["final_path"]
        if final_path.suffix == ".json":
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_bytes(kwargs["data"])
            return None
        return original_stage_new_file(uow, **kwargs)

    monkeypatch.setattr(
        project_context_tools, "_stage_new_file", simulate_existing_context
    )

    output = run_materialize_project_context(request, fx["ctx"], fx["runtime"])

    external_path = (
        fx["settings"].tasks_dir
        / fx["task"].id
        / "strategy_project_context_sources"
        / output["external_artifacts"][0]["filename"]
    )
    assert output["created"] is True
    assert external_path.read_bytes() == source.read_bytes()


def test_identical_concurrent_writers_share_one_revision(tmp_path: Path) -> None:
    fx = _setup(tmp_path)

    def materialize() -> dict:
        return run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: materialize(), range(2)))

    assert {item["revision"]["revision_id"] for item in outputs} == {
        outputs[0]["revision"]["revision_id"]
    }
    assert sorted(item["created"] for item in outputs) == [False, True]
    with sqlite3.connect(fx["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_project_context_revisions WHERE task_id = ?",
                (fx["task"].id,),
            ).fetchone()[0]
            == 1
        )


def test_audit_loader_preserves_history_but_current_loader_rejects_live_ref_tamper(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    source = Path(fx["task"].source_dir) / "history.pdf"
    source.write_bytes(b"historical-report")
    output = run_materialize_project_context(
        _request_bound_to_message(
            fx,
            "使用 history.pdf 作为历史证据。",
            external_report_filenames=[source.name],
        ),
        fx["ctx"],
        fx["runtime"],
    )
    copied = Path(output["external_artifacts"][0]["filename"])
    copied = (
        fx["settings"].tasks_dir
        / fx["task"].id
        / "strategy_project_context_sources"
        / copied
    )
    copied.write_bytes(b"tampered-report")

    assert (
        load_strategy_project_context_revision_for_audit(
            fx["runtime"],
            task_id=fx["task"].id,
            revision_id=output["revision"]["revision_id"],
        )
        == output["revision"]
    )
    with pytest.raises(StrategyError, match="bytes|source changed"):
        load_current_strategy_project_context(fx["runtime"], task_id=fx["task"].id)


def test_both_loaders_reject_context_artifact_tamper(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])
    context_path = (
        fx["settings"].tasks_dir
        / fx["task"].id
        / "strategy_project_contexts"
        / output["context_artifact"]["filename"]
    )
    context_path.write_bytes(context_path.read_bytes() + b"\n")

    with pytest.raises(StrategyError, match="bytes"):
        load_strategy_project_context_revision_for_audit(
            fx["runtime"],
            task_id=fx["task"].id,
            revision_id=output["revision"]["revision_id"],
        )
    with pytest.raises(StrategyError, match="bytes"):
        load_current_strategy_project_context(fx["runtime"], task_id=fx["task"].id)


def test_discovers_strategy_lineage_rule_diff_typed_backtest_and_monitoring_plan(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    frame = pd.DataFrame({"score": [400, 650, 720, 810], "bad": [1, 1, 0, 0]})
    source = tmp_path / "strategy-sample.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(fx["settings"].db_path),
        DataBackend(fx["settings"].datasets_dir),
        fx["settings"].datasets_dir,
    )
    dataset = registry.register_existing(
        source, task_id=fx["task"].id, role="strategy_sample"
    )
    baseline_spec = {
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": "reject-low",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "score",
                    "operator": "<",
                    "value": 600,
                },
                "action": {"type": "reject"},
            }
        ],
    }
    baseline = build_strategy_from_spec(baseline_spec, description="baseline approval")
    strategies = StrategyRepository(fx["settings"].db_path)
    strategies.create_strategy(
        fx["task"].id, baseline, created_at="2026-06-01T00:00:00+00:00"
    )
    result = run_typed_backtest(
        frame,
        baseline.spec,
        target_col="bad",
        strategy_id=baseline.id,
    )
    strategies.save_backtest(
        "backtest-context-1",
        baseline.id,
        dataset.id,
        result,
        created_at="2026-06-02T00:00:00+00:00",
    )
    strategies.adopt_strategy_with_audit(
        baseline.id,
        reason="committee approved",
        audit={
            "kind": "strategy.adopt",
            "target_ref": baseline.id,
            "outcome": "succeeded",
            "detail": {},
        },
        adopted_at="2026-06-03T00:00:00+00:00",
    )
    revised_spec = baseline.spec.to_dict()
    revised_spec["rules"][0]["condition"]["value"] = 620
    child = strategies.new_version_from(
        baseline.id,
        strategy_spec=revised_spec,
        description="tightened cutoff",
        created_at="2026-06-04T00:00:00+00:00",
    )
    monitoring = StrategyMonitoringRepository(fx["settings"].db_path)
    monitoring.create_plan(
        MonitoringPlan(
            strategy_id=baseline.id,
            version=1,
            monitoring_plan_id="monitor-plan-context-1",
        ),
        expected_revision=0,
        plan_id="monitor-plan-context-1",
    )

    output = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])

    snapshot = output["revision"]["state"]["current_project_snapshot"]
    assert snapshot["champion_strategy_ref"]["ref_id"] == baseline.id
    assert snapshot["status_fields"]["approval"]["availability"] == "present"
    histories = {
        item["strategy_ref"]["ref_id"]: item
        for item in output["revision"]["state"]["historical_strategy_reviews"]
        if item["strategy_ref"] is not None
    }
    assert set(histories) == {baseline.id, child.id}
    assert (
        histories[child.id]["change_set"]["modified_rule_refs"][0]["rule_id"]
        == "reject-low"
    )
    assert [
        item["observation_ref"]["ref_id"]
        for item in histories[baseline.id]["observation_refs_by_effect_stage"][
            "backtested"
        ]
    ] == ["backtest-context-1"]
    assert (
        histories[baseline.id]["observation_refs_by_effect_stage"][
            "post_launch_observed"
        ]
        == []
    )
    assert {item["kind"] for item in histories[baseline.id]["tool_run_refs"]} == {
        "backtest",
        "monitoring_plan",
    }
    assert (
        load_current_strategy_project_context(fx["runtime"], task_id=fx["task"].id)
        == output["revision"]
    )


def test_discovers_sample_design_before_missing_and_preserves_observed_zero(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    frame = pd.DataFrame(
        {
            "apply_month": ["202601", "202601", "202602"],
            "bad": [0, 0, 0],
        }
    )
    source = tmp_path / "zero-risk.parquet"
    frame.to_parquet(source, index=False)
    backend = DataBackend(fx["settings"].datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(fx["settings"].db_path),
        backend,
        fx["settings"].datasets_dir,
    )
    dataset = registry.register_existing(
        source, task_id=fx["task"].id, role="strategy_sample"
    )
    workspaces = DataWorkspaceRepository(fx["settings"].db_path)
    activated = workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={"apply_month": "month", "bad": "target"},
    )
    workspace = workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    fx["runtime"].backend = backend
    fx["runtime"].registry = registry
    run_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-02-28",
            "maturity_status": "confirmed_matured",
            "month_col": "apply_month",
            "drop_nan_labels": False,
        },
        fx["ctx"],
        fx["runtime"],
    )

    output = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])

    snapshot = output["revision"]["state"]["current_project_snapshot"]
    assert snapshot["status_fields"]["volume"]["availability"] == "present"
    assert snapshot["status_fields"]["risk"]["availability"] == "present"
    risk_values = {
        item["metric_key"]: item["value"]
        for item in snapshot["status_fields"]["risk"]["value"]
    }
    assert risk_values["bad_rate"] == 0.0
    assert risk_values["bad_count"] == 0
    assert snapshot["maturity_summary"]["availability"] == "present"
    missing_paths = {
        item["field_path"] for item in output["missing_information_records"]
    }
    assert "current.status_fields.volume" not in missing_paths
    assert "current.status_fields.risk" not in missing_paths
    assert "current.maturity_summary" not in missing_paths
    assert (
        load_current_strategy_project_context(fx["runtime"], task_id=fx["task"].id)
        == output["revision"]
    )


def test_as_of_excludes_future_strategy_backtest_adoption_and_monitoring(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    frame = pd.DataFrame({"score": [400, 650, 720], "bad": [1, 0, 0]})
    source = tmp_path / "future-evidence.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(fx["settings"].db_path),
        DataBackend(fx["settings"].datasets_dir),
        fx["settings"].datasets_dir,
    )
    dataset = registry.register_existing(
        source, task_id=fx["task"].id, role="strategy_sample"
    )
    spec = {
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": "reject-low",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "score",
                    "operator": "<",
                    "value": 600,
                },
                "action": {"type": "reject"},
            }
        ],
    }
    baseline = build_strategy_from_spec(spec, description="past candidate")
    strategies = StrategyRepository(fx["settings"].db_path)
    strategies.create_strategy(
        fx["task"].id,
        baseline,
        created_at="2026-07-01T00:00:00+00:00",
    )
    result = run_typed_backtest(
        frame,
        baseline.spec,
        target_col="bad",
        strategy_id=baseline.id,
    )
    strategies.save_backtest(
        "future-backtest",
        baseline.id,
        dataset.id,
        result,
        created_at="2026-08-01T00:00:00+00:00",
    )
    strategies.adopt_strategy_with_audit(
        baseline.id,
        reason="future committee decision",
        audit={
            "kind": "strategy.adopt",
            "target_ref": baseline.id,
            "outcome": "succeeded",
            "detail": {},
        },
        adopted_at="2026-08-02T00:00:00+00:00",
    )
    future_child = strategies.new_version_from(
        baseline.id,
        strategy_spec=baseline.spec.to_dict(),
        description="future version",
        created_at="2026-08-03T00:00:00+00:00",
    )
    StrategyMonitoringRepository(fx["settings"].db_path).create_plan(
        MonitoringPlan(
            strategy_id=baseline.id,
            version=1,
            monitoring_plan_id="future-monitor-plan",
        ),
        expected_revision=0,
        plan_id="future-monitor-plan",
        created_at="2026-08-04T00:00:00+00:00",
    )

    output = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])

    snapshot = output["revision"]["state"]["current_project_snapshot"]
    assert snapshot["champion_strategy_ref"] is None
    assert snapshot["status_fields"]["approval"]["availability"] == "unavailable"
    histories = output["revision"]["state"]["historical_strategy_reviews"]
    assert [item["strategy_ref"]["ref_id"] for item in histories] == [baseline.id]
    assert histories[0]["asset_status"]["availability"] == "unavailable"
    assert histories[0]["observation_refs_by_effect_stage"]["backtested"] == []
    assert histories[0]["tool_run_refs"] == []
    source_ids = {item["ref_id"] for item in output["revision"]["state"]["source_refs"]}
    assert "future-backtest" not in source_ids
    assert "future-monitor-plan" not in source_ids
    assert future_child.id not in source_ids
    assert "current.status_fields.approval" in {
        item["field_path"] for item in output["missing_information_records"]
    }


def test_sample_observation_window_after_as_of_cannot_close_status_fields(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    frame = pd.DataFrame({"apply_month": ["202608", "202608"], "bad": [0, 1]})
    source = tmp_path / "future-window.parquet"
    frame.to_parquet(source, index=False)
    backend = DataBackend(fx["settings"].datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(fx["settings"].db_path),
        backend,
        fx["settings"].datasets_dir,
    )
    dataset = registry.register_existing(
        source, task_id=fx["task"].id, role="strategy_sample"
    )
    workspaces = DataWorkspaceRepository(fx["settings"].db_path)
    activated = workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={"apply_month": "month", "bad": "target"},
    )
    workspace = workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    fx["runtime"].backend = backend
    fx["runtime"].registry = registry
    run_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-08-01",
            "observation_window_end": "2026-08-31",
            "maturity_status": "confirmed_matured",
            "month_col": "apply_month",
            "drop_nan_labels": False,
        },
        fx["ctx"],
        fx["runtime"],
    )

    output = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])

    snapshot = output["revision"]["state"]["current_project_snapshot"]
    assert snapshot["metric_definition_refs"] == []
    assert snapshot["metric_observation_refs"] == []
    assert snapshot["status_fields"]["volume"]["availability"] == "unavailable"
    assert snapshot["status_fields"]["risk"]["availability"] == "unavailable"
    assert snapshot["maturity_summary"]["availability"] == "unavailable"
    missing_paths = {
        item["field_path"] for item in output["missing_information_records"]
    }
    assert {
        "current.status_fields.volume",
        "current.status_fields.risk",
        "current.maturity_summary",
    } <= missing_paths


def test_limit_backtest_does_not_close_approval_information(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    frame = pd.DataFrame({"x": [0, 1, 2], "bad": [1, 0, 0]})
    source = tmp_path / "limit-backtest.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(fx["settings"].db_path),
        DataBackend(fx["settings"].datasets_dir),
        fx["settings"].datasets_dir,
    )
    dataset = registry.register_existing(
        source, task_id=fx["task"].id, role="strategy_sample"
    )
    strategy = build_strategy_from_spec(
        {
            "strategy_type": "limit",
            "default_action": {"type": "limit", "value": 1000},
            "rules": [
                {
                    "rule_id": "limit-high",
                    "priority": 1,
                    "condition": {
                        "op": "compare",
                        "field": "x",
                        "operator": ">=",
                        "value": 1,
                    },
                    "action": {"type": "limit", "value": 2000},
                }
            ],
        },
        description="limit strategy",
    )
    strategies = StrategyRepository(fx["settings"].db_path)
    strategies.create_strategy(
        fx["task"].id,
        strategy,
        created_at="2026-07-01T00:00:00+00:00",
    )
    result = run_typed_backtest(
        frame,
        strategy.spec,
        target_col="bad",
        strategy_id=strategy.id,
    )
    strategies.save_backtest(
        "limit-backtest",
        strategy.id,
        dataset.id,
        result,
        created_at="2026-07-02T00:00:00+00:00",
    )

    output = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])

    approval = output["revision"]["state"]["current_project_snapshot"]["status_fields"][
        "approval"
    ]
    assert approval["availability"] == "unavailable"
    assert approval["value"] is None
    assert "current.status_fields.approval" in {
        item["field_path"] for item in output["missing_information_records"]
    }


def test_user_claims_resolve_questions_without_becoming_governed_evidence(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    claims = {
        "current.status_fields.volume": "月均申请量约 10 万，尚未核验",
        "current.status_fields.approval": "当前通过率约 80%，尚未核验",
        "current.status_fields.risk": "当前 M3+ 约 2%，尚未核验",
        "current.status_fields.economics": "收益口径暂估为正，尚未核验",
        "current.maturity_summary": "用户说明 MOB3 已成熟",
        "historical_strategy_reviews": "用户说明 2025 年曾调整过策略",
    }

    output = run_materialize_project_context(
        {
            **fx["request"],
            "business_context": claims,
            "explicit_unavailable": [],
        },
        fx["ctx"],
        fx["runtime"],
    )

    snapshot = output["revision"]["state"]["current_project_snapshot"]
    for field_name in ("volume", "approval", "risk", "economics"):
        field_path = f"current.status_fields.{field_name}"
        field = snapshot["status_fields"][field_name]
        assert field["value"] == claims[field_path]
        assert field["availability"] == "present"
        assert field["origin"] == "user"
        assert field["note"] == (
            "user-provided/unverified; not deterministic metric evidence"
        )
    assert snapshot["metric_definition_refs"] == []
    assert snapshot["metric_observation_refs"] == []
    assert snapshot["maturity_summary"]["value"] is None
    assert snapshot["maturity_summary"]["availability"] == "unavailable"
    assert output["revision"]["state"]["historical_strategy_reviews"] == []

    user_context = {
        item["field_path"]: item["field"] for item in snapshot["user_context_fields"]
    }
    assert set(claims) <= set(user_context)
    assert (
        user_context["current.maturity_summary"]["value"]
        == claims["current.maturity_summary"]
    )
    assert (
        user_context["historical_strategy_reviews"]["value"]
        == claims["historical_strategy_reviews"]
    )

    records = {
        item["field_path"]: item for item in output["missing_information_records"]
    }
    for field_path in claims:
        assert records[field_path]["status"] == "provided"
        assert records[field_path]["asked_count"] == 0
        assert (
            records[field_path]["answer_source_ref"]
            == user_context[field_path]["source_refs"][0]
        )


def test_user_unavailable_answers_remain_null_and_are_never_imputed_as_zero(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    unavailable_paths = {
        "current.status_fields.approval",
        "current.status_fields.risk",
        "current.status_fields.economics",
        "current.maturity_summary",
        "historical_strategy_reviews",
    }

    output = run_materialize_project_context(
        {
            **fx["request"],
            "business_context": {
                "current.status_fields.risk": None,
                "current.maturity_summary": None,
            },
            "explicit_unavailable": [
                "current.status_fields.approval",
                "current.status_fields.economics",
                "historical_strategy_reviews",
            ],
        },
        fx["ctx"],
        fx["runtime"],
    )

    snapshot = output["revision"]["state"]["current_project_snapshot"]
    for field_name in ("approval", "risk", "economics"):
        field = snapshot["status_fields"][field_name]
        assert field["availability"] == "unavailable"
        assert field["value"] is None
    assert snapshot["maturity_summary"]["availability"] == "unavailable"
    assert snapshot["maturity_summary"]["value"] is None
    assert output["revision"]["state"]["historical_strategy_reviews"] == []

    user_context = {
        item["field_path"]: item["field"] for item in snapshot["user_context_fields"]
    }
    records = {
        item["field_path"]: item for item in output["missing_information_records"]
    }
    for field_path in unavailable_paths:
        assert user_context[field_path]["availability"] == "unavailable"
        assert user_context[field_path]["value"] is None
        assert user_context[field_path]["note"] == (
            "user reported unavailable; no zero imputation"
        )
        assert records[field_path]["status"] == "unavailable"
        assert records[field_path]["asked_count"] == 0


def test_followup_user_answer_closes_existing_question_without_reasking(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])
    first_risk = next(
        item
        for item in first["missing_information_records"]
        if item["field_path"] == "current.status_fields.risk"
    )
    assert first_risk["status"] == "pending"
    assert first_risk["asked_count"] == 1

    tasks = TaskRepository(fx["settings"].db_path)
    answer = tasks.add_agent_message(
        fx["task"].id,
        role="user",
        stage="chat",
        content="当前 M3+ 约 2%，这是业务提供的暂未核验口径。",
    )
    answer_hash = hashlib.sha256(answer["content"].encode("utf-8")).hexdigest()
    request = {
        **fx["request"],
        "expected_revision": first["revision"]["revision"],
        "expected_revision_id": first["revision"]["revision_id"],
        "expected_state_hash": first["revision"]["state_hash"],
        "user_message_ref": {
            "message_id": answer["id"],
            "content_hash": answer_hash,
        },
        "business_context": {
            "current.status_fields.risk": "当前 M3+ 约 2%，暂未核验",
        },
    }

    updated = run_materialize_project_context(request, fx["ctx"], fx["runtime"])

    updated_risk = next(
        item
        for item in updated["missing_information_records"]
        if item["field_path"] == "current.status_fields.risk"
    )
    assert (
        updated_risk["missing_information_id"] == first_risk["missing_information_id"]
    )
    assert updated_risk["dependency_hash"] == first_risk["dependency_hash"]
    assert updated_risk["status"] == "provided"
    assert updated_risk["asked_count"] == 1
    assert updated_risk["asked_at"] == first_risk["asked_at"]
    assert updated_risk["answered_at"] == answer["created_at"]
    assert (
        updated["revision"]["state"]["current_project_snapshot"]["status_fields"][
            "risk"
        ]["origin"]
        == "user"
    )


def test_delta_refresh_preserves_untouched_scope_user_evidence_and_answer_state(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_materialize_project_context(fx["request"], fx["ctx"], fx["runtime"])
    first_snapshot = first["revision"]["state"]["current_project_snapshot"]
    first_history_missing = next(
        item
        for item in first["missing_information_records"]
        if item["field_path"] == "historical_strategy_reviews"
    )
    tasks = TaskRepository(fx["settings"].db_path)
    message = tasks.add_agent_message(
        fx["task"].id,
        role="user",
        stage="chat",
        content="截止 2026-07-23，仅补充项目目标：降低人工审核量。",
    )
    message_hash = hashlib.sha256(message["content"].encode("utf-8")).hexdigest()

    second = run_materialize_project_context(
        {
            "expected_revision": first["revision"]["revision"],
            "expected_revision_id": first["revision"]["revision_id"],
            "expected_state_hash": first["revision"]["state_hash"],
            "user_message_ref": {
                "message_id": message["id"],
                "content_hash": message_hash,
            },
            "as_of": "2026-07-23",
            "business_context": {"project.goal": "降低人工审核量"},
            "explicit_unavailable": [],
            "external_report_filenames": [],
        },
        fx["ctx"],
        fx["runtime"],
    )

    second_snapshot = second["revision"]["state"]["current_project_snapshot"]
    assert second_snapshot["scope"] == first_snapshot["scope"]
    first_context = {
        item["field_path"]: item["field"]
        for item in first_snapshot["user_context_fields"]
    }
    second_context = {
        item["field_path"]: item["field"]
        for item in second_snapshot["user_context_fields"]
    }
    assert second_context["project.channel"] == first_context["project.channel"]
    assert (
        second_context["historical_strategy_reviews"]
        == first_context["historical_strategy_reviews"]
    )
    second_history_missing = next(
        item
        for item in second["missing_information_records"]
        if item["field_path"] == "historical_strategy_reviews"
    )
    assert second_history_missing == first_history_missing
