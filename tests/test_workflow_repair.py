"""B-line extension: diagnose / repair / verify for V2 workflows.

Spec: docs/superpowers/specs/2026-10-23-self-repair-extension-spec.md

The failures are reproduced with the real PlanDriver, PlanExecutor,
subprocess ToolRunner and builtin tools; only the tuning timeout uses a stub
step error because a real timeout takes the tool's 36-hour ceiling.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marvis.agent import workflow_repair as repair
from marvis.agent.plan_driver import PlanDriver
from marvis.agent.turn_handlers import shared as shared_lane
from marvis.agent.turn_handlers import workflow_repair_turn as repair_turn
from marvis.agent.validation_repair import UNDO_COMMAND
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PlanRepository, PluginRepository, TaskRepository, init_db
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.governance.repository import GovernanceRepository
from marvis.governance.service import GovernanceService
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.reviewer import Reviewer
from marvis.orchestrator.templates import load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.packs.modeling.split_preflight import builtin_preflight_checks
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.strategy import StrategyRepository
from marvis.settings import build_settings
from marvis.task_environment import load_task_environment, task_python_resolver


TASK_ID = "task-1"


class FakeLLM:
    def complete(self, **kwargs):
        return '{"summary": "done", "open_items": [], "goal_doubt": false, "goal_met": true}'


class FakeHooks:
    def dispatch(self, event, payload, *, task_id):
        return []


def _harness(tmp_path, *, task_type="feature_analysis", run_mode="agent", tier="balanced",
             python_executable=None):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(TaskCreate(
        model_name="repair", model_version="v1", validator="test",
        source_dir=str(tmp_path), task_type=task_type, run_mode=run_mode, capability_tier=tier,
    ))
    with connect(settings.db_path) as conn:
        conn.execute("UPDATE tasks SET id=? WHERE id=?", (TASK_ID, task.id))
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    tool_registry = ToolRegistry(plugin_registry)
    plan_repo = PlanRepository(settings.db_path)
    governance_repo = GovernanceRepository(settings.db_path)
    principal = governance_repo.create_local_principal(display_name="修复测试操作员")
    governance = GovernanceService(
        plan_repo=plan_repo, tool_registry=tool_registry,
        strategy_repo=StrategyRepository(settings.db_path), governance_repo=governance_repo,
    )
    runner = ToolRunner(
        tool_registry, plugin_repo,
        python_executable=python_executable or sys.executable,
        datasets_root=settings.datasets_dir, workspace=settings.workspace,
        governance=governance_repo, binding_resolver=governance,
        task_python_resolver=task_python_resolver(settings.tasks_dir),
    )
    executor = PlanExecutor(
        plan_repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(plan_repo),
        authorizer=governance,
        preflight_checks=builtin_preflight_checks(db_path=settings.db_path, datasets_root=settings.datasets_dir),
    )
    validator = PlanValidator(tool_registry)
    planner = Planner(tool_registry, lambda: FakeLLM(), validator)
    load_builtin_templates()
    runtime = SimpleNamespace(
        settings=settings, plan_repo=plan_repo, plan_executor=executor, planner=planner,
        plan_validator=validator, llm_client=None, tier=tier, allow_manual_gate_adapters=True,
        require_semantic_text_authorization=False, governance_service=governance,
        local_principal=principal, cancellation_check=None, hook_dispatcher=None,
        recovery_responder=None,
    )
    driver = PlanDriver(plan_repo, executor, planner=planner, validator=validator,
                        governance_service=governance, local_principal=principal)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir),
                               settings.datasets_dir)
    return SimpleNamespace(settings=settings, tasks=tasks, runtime=runtime, driver=driver,
                           registry=registry, plan_repo=plan_repo)


def _register(h, tmp_path, name, frame, *, role="sample"):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return h.registry.register_from_upload(TASK_ID, path, role=role)


def _run(h, template_id, slots):
    task = h.tasks.get_task(TASK_ID)
    turn = h.driver.start(task_id=TASK_ID, template_id=template_id, slots=slots)
    shared_lane.append_driver_messages(h.tasks, task, turn, runtime=h.runtime)
    turn = h.driver.resume(plan_id=turn.plan_id, user_text="开始", run_seq=1)
    shared_lane.append_driver_messages(h.tasks, task, turn, runtime=h.runtime)
    return turn.plan_id


def _messages(h):
    return h.tasks.list_agent_messages(TASK_ID)


def _card(h):
    return next(m for m in reversed(_messages(h))
                if (m.get("metadata") or {}).get("error_diagnostic"))["metadata"]["error_diagnostic"]


def _send(h, text):
    return repair_turn.maybe_handle_command(h.runtime, h.tasks, h.tasks.get_task(TASK_ID), user_text=text,
                                            append=shared_lane.append_driver_messages)


def _step(h, plan_id, tool):
    return next(s for s in h.plan_repo.load_plan(plan_id).steps if s.tool_ref.tool == tool)


RNG = np.random.RandomState(0)
N = 200
BASE = pd.DataFrame({"x1": RNG.randn(N), "x2": RNG.randn(N), "y": RNG.randint(0, 2, N)})


# ---------- 1. data processing: key type mismatch (one_click) ----------


def test_join_key_text_vs_float_is_confirmed_with_one_click(tmp_path):
    h = _harness(tmp_path, task_type="data_join")
    anchor = _register(h, tmp_path, "anchor", pd.DataFrame(
        {"cust_id": ["1001", "1002", "1003", "1004", "1005"], "y": [0, 1, 0, 1, 0]}))
    feature = _register(h, tmp_path, "feature", pd.DataFrame(
        {"cust_id": [1001.0, 1002.0, 1003.0, np.nan, 1005.0], "bal": [1, 2, 3, 4, 5]}), role="feature")
    plan_id = _run(h, "data_join", {"anchor_id": anchor.id, "feature_ids": [feature.id]})

    card = _card(h)
    assert card["code"] == "join_key_dtype_mismatch"
    facts = {item["label"]: item["value"] for item in card["evidence"]}
    assert facts["文本侧"] == "全部是纯数字、没有前导零"
    assert facts["数值侧"].startswith("4 个非空值都是整数")
    [option] = card["repair_options"]
    assert option["tier"] == "one_click" and option["params"]["feature_ids"] == [feature.id]
    # one_click never runs by itself
    assert not any((m.get("metadata") or {}).get("repair_report") for m in _messages(h))

    _send(h, option["command"])
    report = next(m for m in reversed(_messages(h)) if (m.get("metadata") or {}).get("repair_report"))
    assert report["metadata"]["repair_report"]["verified"] is True
    plan = h.plan_repo.load_plan(plan_id)
    assert plan.status.value == "awaiting_confirm"
    assert _step(h, plan_id, "confirm_join").status.value == "done"
    assert _step(h, plan_id, "execute_join").status.value == "awaiting_confirm"


def test_join_key_probe_refuses_leading_zeros_and_fractions(tmp_path):
    h = _harness(tmp_path, task_type="data_join")
    text = _register(h, tmp_path, "anchor", pd.DataFrame({"cust_id": ["01001", "1002"]}))
    numeric = _register(h, tmp_path, "feature", pd.DataFrame({"cust_id": [1001.0, np.nan]}), role="feature")
    fraction = _register(h, tmp_path, "fraction", pd.DataFrame({"cust_id": [1001.5, 1002.0]}), role="feature")
    zeros = repair._key_probe(h.registry, text.id, "cust_id", "str", numeric.id, "cust_id", "float64")
    assert zeros["safe"] is False and zeros["text_note"] == "1 个编号有前导零"
    halves = repair._key_probe(h.registry, text.id, "cust_id", "str", fraction.id, "cust_id", "float64")
    assert halves["safe"] is False and halves["numeric_note"] == "1 个值带小数"


# ---------- 2. feature analysis: text columns (auto) + undo ----------


def test_text_feature_is_excluded_automatically_then_undone(tmp_path):
    h = _harness(tmp_path)
    dataset = _register(h, tmp_path, "sample", BASE.assign(channel=RNG.choice(["A", "B", "C"], N)))
    plan_id = _run(h, "feature_analysis", {
        "dataset_id": dataset.id, "target_col": "y", "features": ["x1", "channel"], "metrics": []})

    messages = _messages(h)
    reports = [m["metadata"]["repair_report"] for m in messages if (m.get("metadata") or {}).get("repair_report")]
    assert len(reports) == 1 and reports[0]["verified"] is True and reports[0]["trigger"] == "auto"
    assert "channel" in reports[0]["action"]
    # no failure card: the report carries the finding
    assert not any((m.get("metadata") or {}).get("error_diagnostic") for m in messages)
    assert _step(h, plan_id, "compute_feature_metrics").status.value == "done"
    assert _step(h, plan_id, "compute_feature_metrics").inputs["features"] == ["x1"]

    annotated = [dict(m) for m in messages]
    repair.annotate_workflow_repair_reports(h.settings, h.plan_repo, TASK_ID, annotated)
    assert next(m for m in annotated if (m.get("metadata") or {}).get("repair_report"))[
        "metadata"]["repair_report"]["undo_available"] is True

    _send(h, UNDO_COMMAND)
    step = _step(h, plan_id, "compute_feature_metrics")
    assert step.status.value == "failed" and step.inputs["features"] == ["x1", "channel"]
    assert h.plan_repo.load_plan(plan_id).status.value == "failed"
    card = _card(h)
    assert card["code"] == "feature_text_columns"
    state = repair.load_repair_state(h.settings.tasks_dir, TASK_ID)
    assert reports[0]["fingerprint"] in state["rejected_fingerprints"]
    # undone once: nothing left to undo, and never auto-applied again
    assert _send(h, UNDO_COMMAND)["status"] == "message_saved"
    assert _messages(h)[-1]["content"] == repair.UNDO_UNAVAILABLE_REPLY
    assert repair.plan_auto_repair(h.settings, h.tasks.get_task(TASK_ID), card, _messages(h)) is None


def test_manual_mode_gets_the_diagnosis_without_repair(tmp_path):
    h = _harness(tmp_path, run_mode="manual")
    dataset = _register(h, tmp_path, "sample", BASE.assign(channel=RNG.choice(["A", "B"], N)))
    _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                  "features": ["x1", "channel"], "metrics": []})
    card = _card(h)
    assert card["code"] == "feature_text_columns"
    assert card["repair_options"] == []
    assert [a["command"] for a in card["recovery_actions"]] == [repair.RETRY_COMMAND]
    assert not any((m.get("metadata") or {}).get("repair_report") for m in _messages(h))
    assert _send(h, "这次先不算文本列 channel，其余特征照常计算") is None


def test_no_candidate_features_is_explained(tmp_path):
    h = _harness(tmp_path)
    dataset = _register(h, tmp_path, "sample", pd.DataFrame({
        "channel": RNG.choice(["A", "B"], N), "city": RNG.choice(["sh", "bj"], N), "y": BASE.y}))
    _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y", "features": [], "metrics": []})
    card = _card(h)
    assert card["code"] == "feature_no_candidates"
    facts = {item["label"]: item["value"] for item in card["evidence"]}
    assert "channel" in facts["文本列"] and "city" in facts["文本列"]


# ---------- 7. drop NaN-label rows (one_click data action with audit) ----------


def test_nan_labels_are_dropped_by_the_tool_with_an_audit_record(tmp_path):
    h = _harness(tmp_path)
    frame = BASE.assign(y=BASE.y.astype(float))
    frame.loc[:17, "y"] = np.nan
    dataset = _register(h, tmp_path, "sample", frame)
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": ["iv", "ks"]})
    card = _card(h)
    assert card["code"] == "feature_nan_labels"
    [option] = card["repair_options"]
    assert option["params"]["rows"] == 18
    _send(h, option["command"])
    report = next(m for m in reversed(_messages(h)) if (m.get("metadata") or {}).get("repair_report"))[
        "metadata"]["repair_report"]
    assert report["verified"] is True and "18" in report["verification"]
    assert report["audit"]["kind"] == "drop_rows"
    assert report["audit"]["actor"]["label"] == "修复测试操作员"
    assert report["audit"]["at"] and "撤销" in report["audit"]["undo"]
    bins = _step(h, plan_id, "analyze_feature_bins")
    assert bins.inputs["drop_nan_labels"] is True
    audit_kinds = {row["kind"] for row in h.plan_repo.list_audit(limit=200)}
    assert "workflow.repair.applied" in audit_kinds


# ---------- 8. redefine the target (one_click, derived dataset) ----------


def test_overdue_days_target_is_redefined_into_a_derived_dataset(tmp_path):
    h = _harness(tmp_path)
    dpd = RNG.choice([0, 0, 0, 5, 35, 95], N)
    dataset = _register(h, tmp_path, "sample", BASE.assign(y=dpd))
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": ["iv", "ks"]})
    card = _card(h)
    assert card["code"] == "feature_target_not_binary"
    option = next(item for item in card["repair_options"] if item["params"]["rule"]["threshold"] == 30)
    _send(h, option["command"])
    report = next(m for m in reversed(_messages(h)) if (m.get("metadata") or {}).get("repair_report"))[
        "metadata"]["repair_report"]
    assert report["verified"] is True and report["audit"]["kind"] == "redefine_target"
    metrics = _step(h, plan_id, "compute_feature_metrics")
    derived_id = metrics.inputs["dataset_id"]
    assert derived_id != dataset.id
    assert _step(h, plan_id, "analyze_feature_bins").inputs["dataset_id"] == derived_id
    derived = h.registry.read_authenticated_parquet_snapshot(derived_id)
    assert set(derived["y"].unique()) == {0, 1}
    assert int(derived["y"].sum()) == int((dpd >= 30).sum())
    original = h.registry.read_authenticated_parquet_snapshot(dataset.id)
    assert sorted(original["y"].unique().tolist()) == [0, 5, 35, 95]


# ---------- 4. modeling split (preflight + one_click) ----------


def _modeling_slots(dataset_id, split_config):
    return {"dataset_id": dataset_id, "target_col": "y", "feature_cols": ["x1", "x2"], "split_col": "",
            "split_values": {}, "recipe": "lgb", "recipes": ["lgb"], "seed": 7, "split_config": split_config}


def test_uneven_time_split_is_rejected_before_dispatch_and_repaired(tmp_path):
    h = _harness(tmp_path, task_type="modeling")
    frame = pd.DataFrame({"x1": RNG.randn(300), "x2": RNG.randn(300), "y": RNG.randint(0, 2, 300),
                          "apply_month": ["202601"] * 290 + ["202602"] * 10})
    dataset = _register(h, tmp_path, "sample", frame)
    plan_id = _run(h, "modeling", _modeling_slots(dataset.id, {"oot_by_time": "apply_month", "oot_size": 0.2}))

    split = _step(h, plan_id, "make_split")
    assert split.status.value == "failed"
    assert "切分后这些集合为空" in split.error
    assert h.plan_repo.latest_failed_step_run_error_kind(split.id) == "preflight_rejected"
    assert h.plan_repo.unreconciled_step_ids([split.id]) == []
    card = _card(h)
    assert card["code"] == "modeling_split_empty"
    [option] = card["repair_options"]
    assert option["id"] == "split_latest_oot"
    assert option["params"]["split_config"]["rules"][-1]["when"][0] == {
        "col": "apply_month", "op": "ge", "val": "202602"}
    facts = {item["label"]: item["value"] for item in card["evidence"]}
    assert "OOT 10 行" in facts["按新口径预演"]

    _send(h, option["command"])
    report = next(m for m in reversed(_messages(h)) if (m.get("metadata") or {}).get("repair_report"))[
        "metadata"]["repair_report"]
    assert report["verified"] is True
    assert _step(h, plan_id, "make_split").status.value == "done"


# ---------- 6. task-level execution environment (one_click + undo) ----------


def _old_python_shim(tmp_path) -> str:
    shim = tmp_path / "old-python"
    shim.write_text(
        "#!/bin/sh\n"
        'case "$*" in *version_info*) echo 3.7.12; exit 0;; esac\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return str(shim)


def test_incompatible_worker_python_is_fixed_for_this_task_only(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("shell shim needs POSIX")
    global_python = _old_python_shim(tmp_path)
    h = _harness(tmp_path, python_executable=global_python)
    monkeypatch.setattr("marvis.task_environment._detected_interpreters",
                        lambda: [("开发环境 py_313", sys.executable)])
    dataset = _register(h, tmp_path, "sample", BASE)
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": []})
    card = _card(h)
    assert card["code"] == "workflow_environment_incompatible"
    options = {item["id"]: item for item in card["repair_options"]}
    assert options["change_global_environment"]["command"] == ""
    option = options["use_task_environment"]
    global_settings = h.settings.workspace / "execution_environment.json"
    before = global_settings.read_text() if global_settings.exists() else None

    _send(h, option["command"])
    report = next(m for m in reversed(_messages(h)) if (m.get("metadata") or {}).get("repair_report"))[
        "metadata"]["repair_report"]
    assert report["verified"] is True and report["audit"]["kind"] == "environment_override"
    assert _step(h, plan_id, "compute_feature_metrics").status.value == "done"
    assert load_task_environment(h.settings.tasks_dir, TASK_ID)["python_executable"] == sys.executable
    assert (global_settings.read_text() if global_settings.exists() else None) == before

    _send(h, UNDO_COMMAND)
    assert load_task_environment(h.settings.tasks_dir, TASK_ID) is None
    assert (m := _messages(h)[-1])["metadata"].get("repair_undo") or m["metadata"].get("intent")


def test_unusable_task_environment_is_rolled_back(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("shell shim needs POSIX")
    h = _harness(tmp_path, python_executable=_old_python_shim(tmp_path))
    dataset = _register(h, tmp_path, "sample", BASE)
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": []})
    step = _step(h, plan_id, "compute_feature_metrics")
    context = repair.failure_context(h.settings, h.tasks.get_task(TASK_ID), h.plan_repo, {
        "plan_id": plan_id, "failure_envelope": {"failed_step_id": step.id}})
    (tmp_path / "again").mkdir()
    broken = _old_python_shim(tmp_path / "again")
    option = repair._option("use_task_environment", tier="one_click", label="l", command="c", consequence="x",
                            params={"python_executable": broken, "label": "坏环境", "python_version": "3.7.12"},
                            reversible=True)
    # Not a probed candidate at write time: refused before anything is written.
    with pytest.raises(ValueError, match="不是可用的执行环境"):
        repair.apply_workflow_repair(h.runtime, repair_turn._driver(h.runtime), context, option,
                                     finding="f", trigger="user")
    assert load_task_environment(h.settings.tasks_dir, TASK_ID) is None
    # Passes the probe but the tool still fails: verified false, override rolled back.
    monkeypatch.setattr("marvis.task_environment.probe_python_candidates", lambda **_kw: [
        {"label": "坏环境", "python_executable": broken, "python_version": "3.11.0"}])
    outcome = repair.apply_workflow_repair(h.runtime, repair_turn._driver(h.runtime), context, option,
                                           finding="f", trigger="user")
    assert outcome.report["verified"] is False
    assert outcome.report["verification"].startswith("核对没通过")
    assert load_task_environment(h.settings.tasks_dir, TASK_ID) is None
    assert _step(h, plan_id, "compute_feature_metrics").status.value == "failed"


def _raise_on_retry(monkeypatch):
    from marvis.agent.plan_driver import DriverError

    def boom(self, *args, **kwargs):
        raise DriverError("注入的重试失败")

    monkeypatch.setattr(PlanDriver, "retry_failed_step", boom)


def _rollback_reply(h):
    reply = _messages(h)[-1]["content"]
    assert "已全部还原" in reply and "注入的重试失败" in reply
    kinds = [row["kind"] for row in h.plan_repo.list_audit(limit=200)]
    assert "workflow.repair.rolled_back" in kinds and "workflow.repair.applied" not in kinds
    [entry] = repair.load_repair_state(h.settings.tasks_dir, TASK_ID)["audit"]
    assert entry["outcome"] == "rolled_back" and "注入的重试失败" in entry["error"]
    assert entry["remaining"] == []
    assert repair.load_repair_state(h.settings.tasks_dir, TASK_ID)["applied"] == []
    return reply


def test_environment_repair_is_rolled_back_when_retry_raises(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("shell shim needs POSIX")
    h = _harness(tmp_path, python_executable=_old_python_shim(tmp_path))
    monkeypatch.setattr("marvis.task_environment._detected_interpreters",
                        lambda: [("开发环境 py_313", sys.executable)])
    dataset = _register(h, tmp_path, "sample", BASE)
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": []})
    option = next(item for item in _card(h)["repair_options"] if item["id"] == "use_task_environment")
    _raise_on_retry(monkeypatch)
    _send(h, option["command"])
    reply = _rollback_reply(h)
    assert "平台没有做任何改动" in reply
    assert load_task_environment(h.settings.tasks_dir, TASK_ID) is None
    assert _step(h, plan_id, "compute_feature_metrics").status.value == "failed"


def test_environment_rollback_and_undo_keep_the_previous_override(tmp_path, monkeypatch):
    from marvis.task_environment import save_task_environment

    if os.name != "posix":
        pytest.skip("shell shim needs POSIX")
    h = _harness(tmp_path, python_executable=_old_python_shim(tmp_path))
    monkeypatch.setattr("marvis.task_environment._detected_interpreters",
                        lambda: [("开发环境 py_313", sys.executable)])
    dataset = _register(h, tmp_path, "sample", BASE)
    _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                 "features": ["x1", "x2"], "metrics": []})
    option = next(item for item in _card(h)["repair_options"] if item["id"] == "use_task_environment")
    (tmp_path / "prev").mkdir()
    previous = _old_python_shim(tmp_path / "prev")
    save_task_environment(h.settings.tasks_dir, TASK_ID, python_executable=previous, label="之前的环境",
                          python_version="3.7.12", repair_id="earlier")
    _send(h, option["command"])
    report = next(m for m in reversed(_messages(h)) if (m.get("metadata") or {}).get("repair_report"))[
        "metadata"]["repair_report"]
    assert report["verified"] is True
    _send(h, UNDO_COMMAND)
    restored = load_task_environment(h.settings.tasks_dir, TASK_ID)
    assert restored["python_executable"] == previous and restored["repair_id"] == "earlier"


def test_nan_label_repair_is_rolled_back_when_retry_raises(tmp_path, monkeypatch):
    h = _harness(tmp_path)
    frame = BASE.assign(y=BASE.y.astype(float))
    frame.loc[:17, "y"] = np.nan
    dataset = _register(h, tmp_path, "sample", frame)
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": ["iv", "ks"]})
    [option] = _card(h)["repair_options"]
    bins_before = dict(_step(h, plan_id, "analyze_feature_bins").inputs)
    _raise_on_retry(monkeypatch)
    _send(h, option["command"])
    assert "平台没有做任何改动" in _rollback_reply(h)
    assert _step(h, plan_id, "analyze_feature_bins").inputs == bins_before
    metrics = _step(h, plan_id, "compute_feature_metrics")
    assert metrics.status.value == "failed" and "drop_nan_labels" not in (metrics.inputs or {})


def test_target_redefinition_is_rolled_back_when_retry_raises(tmp_path, monkeypatch):
    h = _harness(tmp_path)
    dpd = RNG.choice([0, 0, 0, 5, 35, 95], N)
    dataset = _register(h, tmp_path, "sample", BASE.assign(y=dpd))
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": ["iv", "ks"]})
    option = next(item for item in _card(h)["repair_options"] if item["params"]["rule"]["threshold"] == 30)
    bins_before = dict(_step(h, plan_id, "analyze_feature_bins").inputs)
    metrics_before = dict(_step(h, plan_id, "compute_feature_metrics").inputs)
    _raise_on_retry(monkeypatch)
    _send(h, option["command"])
    reply = _rollback_reply(h)
    assert "派生数据集" in reply and "原样本没有改动" in reply
    assert _step(h, plan_id, "analyze_feature_bins").inputs == bins_before
    metrics = _step(h, plan_id, "compute_feature_metrics")
    assert metrics.inputs == metrics_before and metrics.status.value == "failed"
    original = h.registry.read_authenticated_parquet_snapshot(dataset.id)
    assert sorted(original["y"].unique().tolist()) == [0, 5, 35, 95]


def test_unrestorable_rollback_says_what_remains(tmp_path, monkeypatch):
    h = _harness(tmp_path)
    frame = BASE.assign(y=BASE.y.astype(float))
    frame.loc[:17, "y"] = np.nan
    dataset = _register(h, tmp_path, "sample", frame)
    plan_id = _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                           "features": ["x1", "x2"], "metrics": ["iv", "ks"]})
    [option] = _card(h)["repair_options"]
    _raise_on_retry(monkeypatch)

    def refuse(*args, **kwargs):
        raise RuntimeError("注入的还原失败")

    real_update = PlanRepository.update_step
    calls = []

    def update_once(self, step):
        calls.append(step.id)
        if len(calls) > 1:  # the repair's own write succeeds; the write-back is refused
            refuse()
        return real_update(self, step)

    monkeypatch.setattr(PlanRepository, "restore_failed_step", refuse)
    monkeypatch.setattr(PlanRepository, "update_step", update_once)
    _send(h, option["command"])
    reply = _messages(h)[-1]["content"]
    assert "没能完全还原" in reply and "平台没有做任何改动" not in reply
    assert "「" in reply
    [entry] = repair.load_repair_state(h.settings.tasks_dir, TASK_ID)["audit"]
    assert entry["remaining"]
    assert _step(h, plan_id, "analyze_feature_bins").inputs.get("drop_nan_labels") is True


# ---------- 5 / 9. fixtures: tuning timeout and IV floor ----------


def _stub_plan(h, *, tool, error, inputs, output=None, extra_steps=()):
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
    from marvis.plugins.manifest import ToolRef

    steps = [*extra_steps, PlanStep(
        id="s-fail", plan_id="p1", index=len(extra_steps), title="失败步骤",
        tool_ref=ToolRef("modeling", tool), inputs=inputs, depends_on=[s.id for s in extra_steps],
        post_checks=[], status=StepStatus.FAILED, error=error,
    )]
    plan = Plan(id="p1", task_id=TASK_ID, goal="g", source="template", template_id="modeling",
                steps=steps, autonomy_level=1, status=PlanStatus.FAILED)
    repo = SimpleNamespace(
        load_plan=lambda _pid: plan,
        load_step_output=lambda step_id: (output or {}).get(step_id, {}),
        load_bound_step_output=lambda step_id: (output or {}).get(step_id, {}),
        latest_failed_step_run_error_kind=lambda _sid: "unknown_effect",
    )
    return repair.FailureContext(settings=h.settings, task=h.tasks.get_task(TASK_ID), plan_repo=repo,
                                 plan=plan, step=steps[-1], error=error, error_kind="unknown_effect", run_seq=1)


def test_tuning_timeout_behind_reconciliation_is_explained_not_rerun(tmp_path):
    from marvis.orchestrator.contracts import PlanStep, StepStatus
    from marvis.plugins.manifest import ToolRef

    h = _harness(tmp_path, task_type="modeling")
    config = PlanStep(id="s-config", plan_id="p1", index=0, title="配置调参",
                      tool_ref=ToolRef("modeling", "configure_tuning"), inputs={}, depends_on=[], post_checks=[], status=StepStatus.DONE,
                      output_ref="ref")
    error = ("tool failure may have left side effects; explicit reconciliation required; original "
             "error_kind=timeout: tool modeling.tune_hyperparameters timed out after 130500s")
    context = _stub_plan(h, tool="tune_hyperparameters", error=error, inputs={},
                         output={"s-config": {"n_trials_by_recipe": {"lgb": 100, "xgb": 40}}},
                         extra_steps=(config,))
    diagnostic = repair.diagnose_workflow_failure(context, agent_mode=True)
    assert diagnostic["code"] == "modeling_tuning_timeout"
    assert diagnostic["retryable"] is False
    assert [item["tier"] for item in diagnostic["repair_options"]] == ["confirm"]
    assert diagnostic["recovery_actions"] == []
    assert "lgb 100→50" in diagnostic["actions"][0]
    facts = {item["label"]: item["value"] for item in diagnostic["evidence"]}
    assert facts["超时上限"] == "36 小时"


def test_iv_floor_offers_a_lower_threshold_verified_from_platform_scores(tmp_path):
    h = _harness(tmp_path, task_type="modeling")
    scores = {"x1": {"iv": 0.012}, "x2": {"iv": 0.004}, "x3": {"iv": 0.0001}}
    context = _stub_plan(h, tool="select_features", error="selected empty", inputs={"iv_min": 0.02},
                         output={"s-fail": {"selected": [], "scores": scores}})
    diagnostic = repair.diagnose_workflow_failure(context, agent_mode=True)
    assert diagnostic["code"] == "modeling_iv_floor_empty"
    [option] = diagnostic["repair_options"]
    assert option["params"] == {"iv_min": 0.01, "passing": 1, "previous": 0.02}
    assert "1 个特征能通过" in option["consequence"]


# ---------- budget, stale commands, unknown failures ----------


def test_auto_repair_budget_and_unknown_failures(tmp_path):
    h = _harness(tmp_path, tier="conservative")
    task = h.tasks.get_task(TASK_ID)
    diagnostic = {"repair_options": [{"id": "exclude_text_features", "tier": "auto", "fingerprint": "f0"}]}
    used = [{"metadata": {"repair_report": {"tier": "auto", "trigger": "auto", "fingerprint": f"f{i}"}}}
            for i in range(1, 3)]
    assert repair.plan_auto_repair(h.settings, task, diagnostic, used) is None
    assert repair.plan_auto_repair(h.settings, task, diagnostic, used[:1]) is not None
    context = _stub_plan(h, tool="train_models", error="RuntimeError: boom", inputs={})
    assert repair.diagnose_workflow_failure(context, agent_mode=True) is None


def test_old_card_command_is_answered_as_expired(tmp_path):
    h = _harness(tmp_path)
    dataset = _register(h, tmp_path, "sample", BASE.assign(channel=RNG.choice(["A", "B"], N)))
    _run(h, "feature_analysis", {"dataset_id": dataset.id, "target_col": "y",
                                  "features": ["x1", "channel"], "metrics": []})
    report = next(m for m in _messages(h) if (m.get("metadata") or {}).get("repair_report"))
    assert report  # auto-repaired; the card's command now names an expired option
    old_command = "这次先不算文本列 channel，其余特征照常计算"
    h.tasks.add_agent_message(TASK_ID, role="assistant", stage="chat", content="旧卡",
                              metadata={"error": True, "error_diagnostic": {
                                  "diagnosed_by": repair.DIAGNOSED_BY, "repair_options": [
                                      {"id": "exclude_text_features", "command": old_command}]}})
    h.tasks.add_agent_message(TASK_ID, role="assistant", stage="chat", content="进展",
                              metadata={"kind": "gate"})
    result = _send(h, old_command)
    assert result["status"] == "message_saved"
    assert _messages(h)[-1]["content"] == repair.STALE_COMMAND_REPLY


# ---------- route: GET /agent/messages carries undo_available ----------


def test_messages_route_annotates_undo_availability(tmp_path):
    from fastapi.testclient import TestClient

    from marvis.app import create_app

    settings = build_settings(tmp_path / "ws")
    client = TestClient(create_app(settings))
    task = TaskRepository(settings.db_path).create_task(TaskCreate(
        model_name="m", model_version="v", validator="t", source_dir=str(tmp_path),
        task_type="feature_analysis", run_mode="agent"))
    TaskRepository(settings.db_path).add_agent_message(task.id, role="assistant", stage="chat", content="x",
        metadata={"repair_report": {"schema_version": repair.WORKFLOW_REPAIR_SCHEMA, "repair_id": "r1",
                                    "verified": True, "undo_command": UNDO_COMMAND}})
    response = client.get(f"/api/tasks/{task.id}/agent/messages")
    assert response.status_code == 200
    report = response.json()["messages"][-1]["metadata"]["repair_report"]
    assert report["undo_available"] is False  # no applied record: nothing to undo


# ---------- strategy sample design ----------


def test_strategy_sample_design_gap_gets_a_card_and_a_corrected_utterance(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import tests.test_strategy_sample_design_turn_e2e as e2e
    from marvis.app import create_app

    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = e2e._create_task(client, tmp_path)
    e2e._register_and_activate(client, task_id, tmp_path)
    llm = e2e._SampleDesignLLM()
    llm.workflow_inputs["field_bindings"]["time_field"] = "APPLY_DATE"
    monkeypatch.setattr("marvis.agent.validation_app_service.driver_llm_client", lambda request, task: llm)
    utterance = e2e._utterance().replace("时间字段 apply_date", "时间字段 APPLY_DATE")
    response = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": utterance})
    assert response.status_code == 202, response.text
    last = response.json()["messages"][-1]
    assert "APPLY_DATE" in last["content"]  # the clarification text is unchanged
    diagnostic = last["metadata"]["sample_design_diagnostic"]
    assert diagnostic["code"] == "strategy_sample_design_field_missing"
    assert "error_diagnostic" not in last["metadata"]
    [reply] = last["metadata"]["quick_replies"]
    assert reply["label"] == "时间字段改用 apply_date"
    assert "时间字段 apply_date" in reply["command"] and "APPLY_DATE" not in reply["command"]


# ---------- batch ----------


def test_batch_payload_lists_each_child_repair_record():
    from marvis.routers.validation_batches import batch_child_repair_records

    messages = [
        {"metadata": {"repair_report": {"repair_id": "a", "finding": "f", "action": "X1 → x1",
                                        "verified": True, "tier": "auto"}}},
        {"metadata": {"repair_report": {"repair_id": "b", "finding": "g", "action": "试过",
                                        "verified": False, "trigger": "user"}}},
        {"metadata": {"repair_undo": {"repair_id": "a"}}},
    ]
    records = batch_child_repair_records(messages)
    assert [(r["repair_id"], r["verified"], r["undone"], r["trigger"]) for r in records] == [
        ("a", True, True, "auto"), ("b", False, False, "user")]
