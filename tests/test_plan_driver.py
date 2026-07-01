"""Generic PlanDriver loop: run -> pause at gate -> compose message from the
just-computed prior step -> confirm -> resume -> done. Driven against a REAL
PlanExecutor + PlanRepository with a fake tool runner returning canned outputs,
so the gate detection / dependency-output rendering / append-only messaging are
exercised deterministically without running real modeling tools.
"""

from __future__ import annotations

from dataclasses import replace as _dataclass_replace
from types import SimpleNamespace

import pytest

from marvis.agent.plan_driver import (
    DriverError,
    PlanDriver,
    _parse_dedup_instruction,
    is_confirm,
    render_tool_output,
)
from marvis.db import PlanRepository, init_db
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.reviewer import Reviewer
from marvis.plugins.manifest import ToolRef
from marvis.plugins.runner import ToolResult


class FakeLLM:
    def complete(self, **kwargs):
        return '{"summary": "done", "open_items": [], "goal_doubt": false, "goal_met": true}'


class FakeTools:
    def resolve(self, ref):
        return SimpleNamespace(failure_policy="fail")


class FakeRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self._tools = FakeTools()

    def invoke(self, ref, inputs, *, task_id):
        self.calls.append((ref.tool, inputs))
        return ToolResult(ok=True, output=self.outputs.pop(0), error=None, error_kind=None, duration_ms=1)


class FailingRunner:
    def __init__(self):
        self.calls = []
        self._tools = FakeTools()

    def invoke(self, ref, inputs, *, task_id):
        self.calls.append((ref.tool, inputs))
        return ToolResult(
            ok=False,
            output=None,
            error="bad threshold",
            error_kind="validation",
            duration_ms=1,
        )


class FakeHooks:
    def dispatch(self, event, payload, *, task_id):
        return []


def _step(step_id, *, index, tool, depends_on=None, needs_confirmation=False, phase=None):
    return PlanStep(
        id=step_id,
        plan_id="plan-1",
        index=index,
        title=step_id,
        tool_ref=ToolRef("modeling", tool),
        inputs={},
        depends_on=depends_on or [],
        post_checks=[],  # empty => deterministic check trivially passes on canned output
        needs_confirmation=needs_confirmation,
        phase=phase,
    )


def _gated_modeling_plan() -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("screen", index=0, tool="screen_features", phase="特征"),
            _step("tune", index=1, tool="tune_hyperparameters", depends_on=["screen"], needs_confirmation=True, phase="建模"),
            _step("train", index=2, tool="train_model", depends_on=["tune"], phase="建模"),
        ],
    )


def _gated_modeling_weight_plan() -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("spec", index=0, tool="choose_modeling_spec", phase="建模"),
            _step("screen", index=1, tool="screen_features", depends_on=["spec"], phase="特征"),
            _step(
                "tune",
                index=2,
                tool="tune_hyperparameters",
                depends_on=["spec", "screen"],
                needs_confirmation=True,
                phase="建模",
            ),
        ],
    )


def _gated_join_dedup_plan() -> Plan:
    steps = [
        _step("propose", index=0, tool="propose_join", phase="拼接"),
        _step("confirm", index=1, tool="confirm_join", depends_on=["propose"], phase="拼接"),
        _step(
            "execute",
            index=2,
            tool="execute_join",
            depends_on=["confirm"],
            needs_confirmation=True,
            phase="拼接",
        ),
    ]
    steps = [_dataclass_replace(step, plan_id="plan-join") for step in steps]
    plan = Plan(
        id="plan-join",
        task_id="task-join",
        goal="join",
        source="template",
        template_id="data_join",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=steps,
    )
    plan.steps[1].inputs = {"join_plan_id": "join-1", "dedup_strategies": {}}
    return plan


def _driver(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_gated_modeling_plan())
    runner = FakeRunner([
        {"selected": ["sig1", "sig2"], "leakage": [["leak_col", 0.55, 1]], "suspected": [["score_x", 0.3, 1]], "n_screened": 9,
         "ranked": [], "unusable": [], "scores": {}},
        {"best_params": {"num_leaves": 31}, "best_metrics": {"test_ks": 0.41}, "n_trials": 8},
        {"experiment_id": "exp-1", "artifact_id": "art-1", "metrics": {"oot_ks": 0.39, "oot_auc": 0.72}, "feature_importance": [["sig1", 120.0]]},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    return PlanDriver(repo, executor), repo


def test_driver_runs_to_first_gate_and_shows_prior_step(tmp_path):
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")  # VALIDATED -> CONFIRMED (start() does this; here we drive the loop directly)

    turn = driver._run_and_handle("plan-1", run_seq=0)

    # paused before the gated 'tune' step, having just run 'screen'
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(turn.messages) == 1
    msg = turn.messages[0]
    assert msg.stage == "gate"
    # message is composed from the screen (dependency) output, not the gate step
    assert "特征筛选完成" in msg.content
    assert "保留 **2** 个" in msg.content
    assert "泄漏" in msg.content
    assert msg.metadata["step_id"] == "tune"
    assert msg.metadata["plan_id"] == "plan-1"
    assert msg.metadata["run_seq"] == 0
    assert msg.metadata["output_refs"] == {"screen": "metrics:screen:v1"}
    assert msg.metadata["gate_envelope"]["allowed_actions"] == ["confirm", "adjust", "replan", "clarify", "halt"]
    assert msg.metadata["gate_envelope"]["target_step_id"] == "tune"
    assert any(t["title"].startswith("入选特征") for t in msg.metadata["tables"])
    # only screen has executed so far
    loaded = repo.load_plan("plan-1")
    statuses = {s.id: s.status for s in loaded.steps}
    assert statuses["screen"] == StepStatus.DONE
    assert statuses["tune"] == StepStatus.AWAITING_CONFIRM


def test_driver_resume_confirm_runs_to_done(tmp_path):
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)  # pause at tune gate

    turn = driver.resume(plan_id="plan-1", user_text="确认", run_seq=1)

    assert turn.status == PlanStatus.DONE.value
    done = turn.messages[-1]
    assert done.stage == "done"
    assert "完成" in done.content
    loaded = repo.load_plan("plan-1")
    assert loaded.status == PlanStatus.DONE
    assert all(s.status == StepStatus.DONE for s in loaded.steps)


def test_driver_failed_message_carries_retry_contract(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("screen", index=0, tool="screen_features", phase="特征"),
            _step("train", index=1, tool="train_model", depends_on=["screen"], phase="建模"),
        ],
    )
    plan.steps[0].inputs = {"leakage_ks": 0.4, "max_missing_rate": 0.95}
    repo.create_plan(plan)
    executor = PlanExecutor(repo, FailingRunner(), Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)
    repo.confirm_plan("plan-1")

    turn = driver._run_and_handle("plan-1", run_seq=3)

    assert turn.status == PlanStatus.FAILED.value
    msg = turn.messages[0]
    assert msg.stage == "error"
    envelope = msg.metadata["failure_envelope"]
    assert envelope["schema_version"] == "failure.v1"
    assert envelope["failed_step_id"] == "screen"
    assert envelope["error_kind"] == "validation"
    assert envelope["retryable"] is True
    assert envelope["stale_token"] == "plan-1:screen:3"
    assert envelope["downstream_reset_steps"] == ["screen", "train"]
    assert envelope["editable_input_schema"]["properties"]["leakage_ks"] == {
        "default": 0.4,
        "type": "number",
    }


def test_driver_resume_non_confirm_holds_at_gate(tmp_path):
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)

    turn = driver.resume(plan_id="plan-1", user_text="把阈值调高一点", run_seq=1)

    # stays awaiting; tune not yet executed (adjust is a later slice)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert "确认" in turn.messages[0].content
    loaded = repo.load_plan("plan-1")
    assert {s.id: s.status for s in loaded.steps}["tune"] == StepStatus.AWAITING_CONFIRM


def test_manual_dedup_instruction_respects_negation():
    assert _parse_dedup_instruction("用 first 去重") == "first"
    assert _parse_dedup_instruction("请用 last 去重") == "last"
    assert _parse_dedup_instruction("别用 first 去重") is None
    assert _parse_dedup_instruction("不要用 last 策略") is None
    assert _parse_dedup_instruction("do not use first dedup") is None


def test_driver_plan_overview_gate_waits_for_kaishi(tmp_path):
    """A freshly-built (VALIDATED) plan does not run until 开始 is confirmed
    (spec §9 #2 plan-level overview gate)."""
    driver, repo = _driver(tmp_path)
    assert repo.load_plan("plan-1").status == PlanStatus.VALIDATED  # built, not started

    # non-confirm at the overview gate → the plan does NOT start
    turn = driver.resume(plan_id="plan-1", user_text="先看看", run_seq=0)
    assert turn.status == PlanStatus.VALIDATED.value
    assert repo.load_plan("plan-1").status == PlanStatus.VALIDATED
    assert {s.id: s.status for s in repo.load_plan("plan-1").steps}["screen"] != StepStatus.DONE

    # 开始 → confirm_plan + run to the first gate
    turn = driver.resume(plan_id="plan-1", user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert "特征筛选完成" in turn.messages[0].content
    assert {s.id: s.status for s in repo.load_plan("plan-1").steps}["screen"] == StepStatus.DONE


def test_screen_gate_carries_structured_screen_payload(tmp_path):
    """The screening gate message carries a structured ``metadata.screen`` pass-through
    (selected/buckets/scores + the screen step id + the gating thresholds) so the
    frontend §4 interactive selection table is a thin consumer."""
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    turn = driver._run_and_handle("plan-1", run_seq=0)

    screen = turn.messages[0].metadata.get("screen")
    assert screen is not None
    assert screen["selected"] == ["sig1", "sig2"]
    assert screen["step_id"] == "screen"  # an edited selection is confirmed against this step
    assert [row[0] for row in screen["leakage"]] == ["leak_col"]
    assert screen["thresholds"] == {"leakage_ks": 0.40, "max_missing_rate": 0.95}


def test_modeling_screen_gate_carries_sample_weight_setup_payload(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_weight_plan()
    plan.steps[0].inputs = {"sample_weight_col": "", "feature_cols": ["x1", "x2"]}
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "target_type": "binary",
            "recipe": "lgb",
            "recipes": ["lgb"],
            "feature_count": 2,
            "n_trials": 12,
            "metric_policy": "oot_ks",
            "eligible_algorithms": ["lgb", "xgb"],
            "disabled_algorithms": [{"recipe": "lgb_regressor", "reason": "target mismatch"}],
            "pmml_supported_algorithms": ["lgb", "xgb"],
            "warnings": ["样本权重列已从入模特征中移除。"],
            "reason": "目标类型 `binary`,候选算法 lgb,主调参算法 `lgb`,选择指标 oot_ks。",
            "sample_weight_col": "",
            "sample_weight_candidates": ["weight", "sample_weight"],
            "sample_weight_diagnostics": [
                {
                    "column": "weight",
                    "valid": True,
                    "missing_rate": 0.0,
                    "min": 1.0,
                    "max": 2.0,
                    "mean": 1.2,
                    "reason": "",
                }
            ],
        },
        {"selected": ["x1"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    turn = driver._run_and_handle("plan-1", run_seq=0)

    setup = turn.messages[0].metadata.get("modeling_setup")
    guidance = setup.pop("override_guidance")
    assert [item["id"] for item in guidance] == [
        "target_type",
        "recipes",
        "disabled_algorithms",
        "n_trials",
        "sample_weight",
    ]
    assert guidance[0]["label"] == "目标类型"
    assert "0/1 风控标签" in guidance[0]["message"]
    assert guidance[1]["level"] == "info"
    assert "均可导出 PMML" in guidance[1]["message"]
    assert guidance[-1]["level"] == "info"
    assert "检测到候选权重列 weight, sample_weight" in guidance[-1]["message"]
    assert setup == {
        "step_id": "spec",
        "step_title": "spec",
        "target_type": "binary",
        "recipe": "lgb",
        "recipes": ["lgb"],
        "feature_count": 2,
        "n_trials": 12,
        "metric_policy": "oot_ks",
        "eligible_algorithms": ["lgb", "xgb"],
        "disabled_algorithms": [{"recipe": "lgb_regressor", "reason": "target mismatch"}],
        "pmml_supported_algorithms": ["lgb", "xgb"],
        "warnings": ["样本权重列已从入模特征中移除。"],
        "reason": "目标类型 `binary`,候选算法 lgb,主调参算法 `lgb`,选择指标 oot_ks。",
        "split_summary": None,
        "sample_weight_col": "",
        "sample_weight_candidates": ["weight", "sample_weight"],
        "sample_weight_diagnostics": [
            {
                "column": "weight",
                "valid": True,
                "missing_rate": 0.0,
                "min": 1.0,
                "max": 2.0,
                "mean": 1.2,
                "reason": "",
            }
        ],
    }
    envelope = turn.messages[0].metadata["gate_envelope"]
    assert "adjust" in envelope["allowed_actions"]
    control_ids = {control["id"] for control in envelope["controls"]}
    assert {"target_type", "recipes", "n_trials"}.issubset(control_ids)
    sample_weight_control = next(control for control in envelope["controls"] if control["id"] == "sample_weight_col")
    assert sample_weight_control["schema"]["enum"] == ["", "weight", "sample_weight"]


def test_modeling_setup_payload_includes_split_summary_and_algorithm_controls(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("split", index=0, tool="make_split", phase="特征"),
            _step("spec", index=1, tool="choose_modeling_spec", depends_on=["split"], phase="建模"),
            _step(
                "screen",
                index=2,
                tool="screen_features",
                depends_on=["split", "spec"],
                needs_confirmation=True,
                phase="特征",
            ),
        ],
    )
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "result_dataset_id": "ds-split",
            "split_col": "split",
            "holdout_values": ["oot"],
            "sample_analysis": {
                "split_counts": {"train": 90, "test": 10, "oot": 2},
                "total_rows": 102,
            },
        },
        {
            "target_type": "binary",
            "recipe": "lgb",
            "recipes": ["lgb", "xgb"],
            "feature_count": 8,
            "n_trials": 12,
            "metric_policy": "oot_ks",
            "eligible_algorithms": ["lgb", "xgb"],
            "disabled_algorithms": [{"recipe": "lgb_regressor", "reason": "target mismatch"}],
            "pmml_supported_algorithms": ["lgb", "xgb"],
            "sample_weight_col": "",
            "sample_weight_candidates": [],
        },
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    turn = driver._run_and_handle("plan-1", run_seq=0)

    setup = turn.messages[0].metadata["modeling_setup"]
    assert setup["feature_count"] == 8
    assert setup["n_trials"] == 12
    assert setup["metric_policy"] == "oot_ks"
    assert setup["eligible_algorithms"] == ["lgb", "xgb"]
    assert setup["disabled_algorithms"] == [{"recipe": "lgb_regressor", "reason": "target mismatch"}]
    assert setup["split_summary"]["split_counts"] == {"train": 90, "test": 10, "oot": 2}
    assert setup["split_summary"]["warnings"] == ["OOT 占比低于 5%,稳定性结论需谨慎。"]
    guidance_by_id = {item["id"]: item for item in setup["override_guidance"]}
    assert guidance_by_id["split_quality"]["level"] == "warning"
    assert "OOT 占比低于 5%" in guidance_by_id["split_quality"]["message"]
    assert guidance_by_id["n_trials"]["message"].startswith("当前调参轮数 12")
    controls = {control["id"]: control for control in turn.messages[0].metadata["gate_envelope"]["controls"]}
    assert controls["target_type"]["schema"]["enum"] == ["binary", "continuous", "multiclass"]
    assert controls["recipes"]["schema"]["enum"] == ["lgb", "xgb"]
    assert controls["n_trials"]["bounds"] == {"min": 1, "max": 200}


def test_modeling_selection_gate_carries_delivery_payload(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("compare", index=0, tool="compare_experiments", phase="建模"),
            _step("select", index=1, tool="select_experiment", depends_on=["compare"], phase="建模"),
            _step(
                "report",
                index=2,
                tool="generate_model_report",
                depends_on=["select"],
                needs_confirmation=True,
                phase="报告",
            ),
        ],
    )
    repo.create_plan(plan)
    runner = FakeRunner([
        {"experiments": [{"id": "exp-lgb", "recipe": "lgb", "artifact_id": "art-lgb", "oot_ks": 0.31}]},
        {
            "selected_experiment_id": "exp-lgb",
            "artifact_id": "art-lgb",
            "recipe": "lgb",
            "target_type": "binary",
            "selection_metric": "oot_ks",
            "selection_reason": "按 oot_ks 在 PMML/验证移交可用候选中自动选择。",
            "metrics": {
                "oot_ks": 0.31,
                "test_ks": 0.29,
                "psi_oot_vs_train": 0.06,
                "feature_count": 18,
            },
            "capabilities": {
                "pmml_supported": True,
                "handoff_supported": True,
                "native_model_supported": True,
                "reason": "",
            },
            "policy_decision": {
                "status": "accepted",
                "explicit_selection": False,
                "selected_experiment_id": "exp-lgb",
                "policy": {"require_pmml": True, "require_handoff": True},
                "profile": {
                    "pmml_supported": True,
                    "handoff_supported": True,
                    "monotonicity_declared": True,
                },
                "violations": [],
                "override_reason": "",
            },
            "experiments": [
                {
                    "id": "exp-lgb",
                    "recipe": "lgb",
                    "artifact_id": "art-lgb",
                    "oot_ks": 0.31,
                    "test_ks": 0.29,
                    "psi_oot_vs_train": 0.06,
                    "feature_count": 18,
                    "monotonic_constraints": {"age": 1},
                    "calibration": {"method": "sigmoid", "pmml_includes_calibration": False},
                    "capabilities": {
                        "pmml_supported": True,
                        "handoff_supported": True,
                        "native_model_supported": True,
                    },
                },
                {
                    "id": "exp-scorecard",
                    "recipe": "scorecard",
                    "artifact_id": "art-scorecard",
                    "oot_ks": 0.30,
                    "test_ks": 0.28,
                    "psi_oot_vs_train": 0.08,
                    "feature_count": 12,
                    "scorecard_table": [
                        {"feature": "age", "bin_label": "[20,30)", "points": 12, "monotonic_direction": "increasing"},
                    ],
                    "capabilities": {
                        "pmml_supported": True,
                        "handoff_supported": True,
                        "native_model_supported": True,
                    },
                },
                {
                    "id": "exp-mlp",
                    "recipe": "mlp",
                    "artifact_id": "art-mlp",
                    "oot_ks": 0.33,
                    "test_ks": 0.48,
                    "psi_oot_vs_train": 0.28,
                    "feature_count": 120,
                    "capabilities": {
                        "pmml_supported": False,
                        "handoff_supported": False,
                        "native_model_supported": True,
                        "reason": "DNN 仅支持原生模型。",
                    },
                },
            ],
        },
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    turn = driver._run_and_handle("plan-1", run_seq=0)

    delivery = turn.messages[0].metadata["model_delivery"]
    assert delivery["source_tool"] == "select_experiment"
    assert delivery["selected_experiment_id"] == "exp-lgb"
    assert delivery["artifact_id"] == "art-lgb"
    assert delivery["metrics"] == {"oot_ks": 0.31, "test_ks": 0.29, "psi_oot_vs_train": 0.06, "feature_count": 18}
    assert delivery["business_signals"] == {
        "feature_count": 18.0,
        "stability": "稳定",
        "stability_value": 0.06,
        "generalization_gap": pytest.approx(0.02),
        "overfit_flag": False,
        "calibration": "已校准(PMML不含)",
        "delivery": "可移交",
    }
    assert delivery["policy_signals"] == {
        "scorecard": "非评分卡",
        "scorecard_status": "neutral",
        "monotonicity": "已约束",
        "monotonicity_status": "ready",
        "approval": "建议可审批",
        "approval_status": "ready",
        "reasons": [],
    }
    assert delivery["policy_decision"] == {
        "status": "accepted",
        "explicit_selection": False,
        "selected_experiment_id": "exp-lgb",
        "policy": {"require_pmml": True, "require_handoff": True},
        "profile": {
            "pmml_supported": True,
            "handoff_supported": True,
            "monotonicity_declared": True,
        },
        "violations": [],
        "override_reason": "",
    }
    assert delivery["readiness"][0]["status"] == "ready"
    assert delivery["readiness"][1]["status"] == "ready"
    assert delivery["readiness"][2]["status"] == "ready"
    assert delivery["readiness"][3] == {
        "id": "approval_policy",
        "label": "审批策略",
        "status": "ready",
        "artifact": "",
        "reason": "策略门控已通过",
    }
    assert [row["selected"] for row in delivery["candidates"]] == [True, False, False]
    assert delivery["candidates"][0]["business_signals"]["calibration"] == "已校准(PMML不含)"
    assert delivery["candidates"][0]["policy_signals"]["monotonicity"] == "已约束"
    assert delivery["candidates"][1]["policy_signals"]["scorecard"] == "评分卡"
    assert delivery["candidates"][1]["policy_signals"]["monotonicity"] == "已约束"
    assert delivery["candidates"][1]["policy_signals"]["approval"] == "建议可审批"
    assert delivery["candidates"][2]["business_signals"]["stability"] == "高风险"
    assert delivery["candidates"][2]["business_signals"]["delivery"] == "仅原生"
    assert delivery["candidates"][2]["policy_signals"]["approval"] == "需业务复核"
    assert delivery["candidates"][2]["capabilities"]["reason"] == "DNN 仅支持原生模型。"


def test_model_delivery_policy_signals_warn_on_partial_scorecard_monotonicity():
    from marvis.agent.gate_payloads import _policy_signals

    signals = _policy_signals({
        "id": "exp-scorecard",
        "recipe": "scorecard",
        "capabilities": {"pmml_supported": True, "handoff_supported": True},
        "scorecard_table": [
            {"feature": "x1", "monotonic_direction": "increasing"},
            {"feature": "x2"},
        ],
    })

    assert signals["scorecard"] == "评分卡"
    assert signals["monotonicity"] == "需确认"
    assert signals["monotonicity_status"] == "warning"
    assert any("x2" in reason for reason in signals["reasons"])


def test_post_training_gate_merges_report_readiness(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("select", index=0, tool="select_experiment", phase="建模"),
            _step("report", index=1, tool="generate_model_report", depends_on=["select"], phase="报告"),
            _step(
                "post",
                index=2,
                tool="post_training_action",
                depends_on=["select", "report"],
                needs_confirmation=True,
                phase="交付",
            ),
        ],
    )
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "selected_experiment_id": "exp-lgb",
            "artifact_id": "art-lgb",
            "recipe": "lgb",
            "target_type": "binary",
            "selection_metric": "oot_ks",
            "metrics": {"oot_ks": 0.31},
            "capabilities": {
                "pmml_supported": True,
                "handoff_supported": True,
                "native_model_supported": True,
                "calibrated": True,
                "calibration": {"method": "sigmoid", "pmml_includes_calibration": False},
                "pmml_includes_calibration": False,
                "limitations": ["模型已进行 sigmoid 概率校准，但 PMML 产物不包含校准器。"],
            },
        },
        {
            "report_path": "/tmp/model_report.xlsx",
            "section_status": [
                {"section": "汇总", "available": True},
                {"section": "Vintage", "available": False, "reason": "缺少 MOB 列"},
            ],
        },
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    turn = driver._run_and_handle("plan-1", run_seq=0)

    delivery = turn.messages[0].metadata["model_delivery"]
    assert delivery["source_tool"] == "select_experiment"
    assert delivery["report"] == {
        "step_id": "report",
        "step_title": "report",
        "report_path": "/tmp/model_report.xlsx",
        "available_sections": 1,
        "total_sections": 2,
        "skipped_sections": 1,
        "status": "partial",
        "sections": [
            {"section": "汇总", "available": True, "reason": ""},
            {"section": "Vintage", "available": False, "reason": "缺少 MOB 列"},
        ],
    }
    report_readiness = next(item for item in delivery["readiness"] if item["id"] == "model_report")
    assert report_readiness["status"] == "partial"
    assert report_readiness["artifact"] == "/tmp/model_report.xlsx"
    assert report_readiness["reason"] == "报告章节 1/2 可生成"


def test_done_message_carries_post_training_delivery_payload(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("report", index=0, tool="generate_model_report", phase="报告"),
            _step("post", index=1, tool="post_training_action", depends_on=["report"], phase="交付"),
        ],
    )
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "report_path": "/tmp/model_report.xlsx",
            "section_status": [
                {"section": "汇总", "status": "ok"},
                {"section": "模型指标", "status": "ok"},
            ],
        },
        {
            "experiment_id": "exp-lgb",
            "artifact_id": "art-lgb",
            "native_model_path": "/tmp/model.pkl",
            "pmml_path": "/tmp/model.pmml",
            "validation_task_id": "task-validation",
            "challenger_task_id": "task-challenger",
            "challenger_package_path": "/tmp/challenger_backtest_plan.json",
            "challenger_package_markdown_path": "/tmp/challenger_backtest_plan.md",
            "approval_package_path": "/tmp/art-lgb.approval_package.json",
            "approval_package_markdown_path": "/tmp/art-lgb.approval_package.md",
            "model_card_path": "/tmp/art-lgb.model_card.json",
            "model_card_markdown_path": "/tmp/art-lgb.model_card.md",
            "model_card": {
                "schema_version": 1,
                "card_version": "model_card_v1",
                "artifact_id": "art-lgb",
            },
            "monitoring_policy_path": "/tmp/art-lgb.monitoring_policy.json",
            "monitoring_policy_markdown_path": "/tmp/art-lgb.monitoring_policy.md",
            "monitoring_policy": {
                "schema_version": 1,
                "policy_version": "model_monitoring_v1",
                "status": "pass",
                "recommendation": "可进入常规监控",
            },
            "challenger_comparison_path": "/tmp/art-lgb.champion_comparison.json",
            "challenger_comparison_markdown_path": "/tmp/art-lgb.champion_comparison.md",
            "challenger_comparison": {
                "schema_version": 1,
                "comparison_version": "champion_challenger_v1",
                "status": "warn",
                "recommendation": "Challenger 有指标弱于 Champion, 需业务复核差异 (1 项下降)",
                "summary": {"declined_count": 1, "comparable_metric_count": 3},
            },
            "capabilities": {
                "pmml_supported": True,
                "handoff_supported": True,
                "native_model_supported": True,
                "calibrated": True,
                "calibration": {"method": "sigmoid", "pmml_includes_calibration": False},
                "pmml_includes_calibration": False,
                "limitations": ["模型已进行 sigmoid 概率校准，但 PMML 产物不包含校准器。"],
            },
            "actions": [
                {"action": "export_pmml", "status": "succeeded", "pmml_path": "/tmp/model.pmml"},
                {
                    "action": "handoff_to_validation",
                    "status": "succeeded",
                    "validation_task_id": "task-validation",
                },
                {
                    "action": "create_challenger_backtest",
                    "status": "succeeded",
                    "challenger_task_id": "task-challenger",
                    "markdown_path": "/tmp/challenger_backtest_plan.md",
                },
            ],
        }
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    turn = driver._run_and_handle("plan-1", run_seq=0)

    delivery = turn.messages[0].metadata["model_delivery"]
    assert turn.messages[0].stage == "done"
    assert delivery["source_tool"] == "post_training_action"
    assert delivery["native_model_path"] == "/tmp/model.pkl"
    assert delivery["pmml_path"] == "/tmp/model.pmml"
    assert delivery["validation_task_id"] == "task-validation"
    assert delivery["challenger_task_id"] == "task-challenger"
    assert delivery["challenger_package_markdown_path"] == "/tmp/challenger_backtest_plan.md"
    assert delivery["approval_package_path"] == "/tmp/art-lgb.approval_package.json"
    assert delivery["approval_package_markdown_path"] == "/tmp/art-lgb.approval_package.md"
    assert delivery["model_card_path"] == "/tmp/art-lgb.model_card.json"
    assert delivery["model_card_markdown_path"] == "/tmp/art-lgb.model_card.md"
    assert delivery["model_card"]["card_version"] == "model_card_v1"
    assert delivery["monitoring_policy_path"] == "/tmp/art-lgb.monitoring_policy.json"
    assert delivery["monitoring_policy_markdown_path"] == "/tmp/art-lgb.monitoring_policy.md"
    assert delivery["monitoring_policy"]["status"] == "pass"
    assert delivery["challenger_comparison_path"] == "/tmp/art-lgb.champion_comparison.json"
    assert delivery["challenger_comparison_markdown_path"] == "/tmp/art-lgb.champion_comparison.md"
    assert delivery["challenger_comparison"]["status"] == "warn"
    assert delivery["capabilities"]["calibrated"] is True
    assert delivery["capabilities"]["pmml_includes_calibration"] is False
    assert delivery["capabilities"]["calibration"]["method"] == "sigmoid"
    assert delivery["report"]["status"] == "ready"
    assert delivery["report"]["available_sections"] == 2
    assert delivery["report"]["report_path"] == "/tmp/model_report.xlsx"
    assert [item["status"] for item in delivery["actions"]] == ["succeeded", "succeeded", "succeeded"]
    assert [item["status"] for item in delivery["readiness"]] == [
        "ready",
        "ready",
        "ready",
        "ready",
        "pass",
        "warn",
        "succeeded",
        "succeeded",
        "succeeded",
        "ready",
    ]
    assert delivery["readiness"][2]["id"] == "approval_package"
    assert delivery["readiness"][2]["artifact"] == "/tmp/art-lgb.approval_package.md"
    assert delivery["readiness"][3]["id"] == "model_card"
    assert delivery["readiness"][3]["artifact"] == "/tmp/art-lgb.model_card.md"
    assert delivery["readiness"][4]["id"] == "monitoring_policy"
    assert delivery["readiness"][4]["artifact"] == "/tmp/art-lgb.monitoring_policy.md"
    assert delivery["readiness"][5]["id"] == "challenger_comparison"
    assert delivery["readiness"][5]["artifact"] == "/tmp/art-lgb.champion_comparison.md"
    assert delivery["readiness"][6]["id"] == "pmml"
    assert "PMML" in delivery["readiness"][6]["reason"]
    assert "校准" in delivery["readiness"][6]["reason"]
    assert delivery["readiness"][8]["id"] == "challenger_backtest"
    assert delivery["readiness"][8]["artifact"] == "task-challenger"
    assert delivery["readiness"][-1]["id"] == "approval_policy"
    assert delivery["policy_signals"]["approval"] == "建议可审批"


def test_driver_sample_weight_adjust_reruns_modeling_spec_and_downstream_screen(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_weight_plan()
    plan.steps[0].inputs = {"sample_weight_col": "", "feature_cols": ["x1", "x2"]}
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "target_type": "binary",
            "recipes": ["lgb"],
            "sample_weight_col": "",
            "sample_weight_candidates": ["weight", "sample_weight"],
        },
        {"selected": ["x1"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
        {
            "target_type": "binary",
            "recipes": ["lgb"],
            "sample_weight_col": "weight",
            "sample_weight_candidates": ["weight", "sample_weight"],
        },
        {"selected": ["x1", "x2"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)
    turn = driver.resume(
        plan_id="plan-1",
        user_text="调整样本权重",
        run_seq=1,
        adjust_params={"sample_weight_col": "weight"},
        expected_step_id="tune",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [call[0] for call in runner.calls] == [
        "choose_modeling_spec",
        "screen_features",
        "choose_modeling_spec",
        "screen_features",
    ]
    assert runner.calls[2][1]["sample_weight_col"] == "weight"
    assert repo.load_plan("plan-1").steps[2].status == StepStatus.AWAITING_CONFIRM
    assert turn.messages[-1].metadata["modeling_setup"]["sample_weight_col"] == "weight"


def test_driver_modeling_setup_adjust_reruns_spec_and_downstream_screen(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_weight_plan()
    plan.steps[0].inputs = {
        "target_type": "binary",
        "recipes": ["lgb"],
        "n_trials": 12,
        "sample_weight_col": "",
        "feature_cols": ["x1", "x2"],
    }
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "target_type": "binary",
            "recipe": "lgb",
            "recipes": ["lgb"],
            "n_trials": 12,
            "sample_weight_col": "",
            "sample_weight_candidates": [],
        },
        {"selected": ["x1"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
        {
            "target_type": "continuous",
            "recipe": "lgb_regressor",
            "recipes": ["lgb_regressor"],
            "n_trials": 20,
            "sample_weight_col": "",
            "sample_weight_candidates": [],
        },
        {"selected": ["x1", "x2"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)
    turn = driver.resume(
        plan_id="plan-1",
        user_text="调整建模规格",
        run_seq=1,
        adjust_params={"target_type": "continuous", "recipes": ["lgb_regressor"], "n_trials": 20},
        expected_step_id="tune",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [call[0] for call in runner.calls] == [
        "choose_modeling_spec",
        "screen_features",
        "choose_modeling_spec",
        "screen_features",
    ]
    assert runner.calls[2][1]["target_type"] == "continuous"
    assert runner.calls[2][1]["recipes"] == ["lgb_regressor"]
    assert runner.calls[2][1]["n_trials"] == 20
    assert turn.messages[-1].metadata["modeling_setup"]["target_type"] == "continuous"
    assert turn.messages[-1].metadata["modeling_setup"]["recipes"] == ["lgb_regressor"]
    assert turn.messages[-1].metadata["modeling_setup"]["n_trials"] == 20


def test_driver_n_trials_only_adjust_requires_fresh_modeling_gate_token(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_weight_plan()
    plan.steps[0].inputs = {
        "target_type": "binary",
        "recipes": ["lgb"],
        "n_trials": 12,
        "sample_weight_col": "",
        "feature_cols": ["x1", "x2"],
    }
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "target_type": "binary",
            "recipe": "lgb",
            "recipes": ["lgb"],
            "n_trials": 12,
            "sample_weight_col": "",
            "sample_weight_candidates": [],
        },
        {"selected": ["x1"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
        {
            "target_type": "binary",
            "recipe": "lgb",
            "recipes": ["lgb"],
            "n_trials": 24,
            "sample_weight_col": "",
            "sample_weight_candidates": [],
        },
        {"selected": ["x1", "x2"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)

    with pytest.raises(DriverError, match="缺少待确认步骤校验"):
        driver.resume(
            plan_id="plan-1",
            user_text="调整调参轮数",
            run_seq=1,
            adjust_params={"n_trials": 24},
        )
    with pytest.raises(DriverError, match="待确认步骤已变化"):
        driver.resume(
            plan_id="plan-1",
            user_text="调整调参轮数",
            run_seq=1,
            adjust_params={"n_trials": 24},
            expected_step_id="old-gate",
        )
    assert [call[0] for call in runner.calls] == ["choose_modeling_spec", "screen_features"]

    turn = driver.resume(
        plan_id="plan-1",
        user_text="调整调参轮数",
        run_seq=1,
        adjust_params={"n_trials": 24},
        expected_step_id="tune",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [call[0] for call in runner.calls] == [
        "choose_modeling_spec",
        "screen_features",
        "choose_modeling_spec",
        "screen_features",
    ]
    assert runner.calls[2][1]["n_trials"] == 24
    assert turn.messages[-1].metadata["modeling_setup"]["n_trials"] == 24


def test_driver_sample_weight_adjust_rejects_unknown_candidate_without_reset(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_weight_plan()
    plan.steps[0].inputs = {"sample_weight_col": "", "feature_cols": ["x1", "x2"]}
    repo.create_plan(plan)
    runner = FakeRunner([
        {
            "target_type": "binary",
            "recipes": ["lgb"],
            "sample_weight_col": "",
            "sample_weight_candidates": ["weight"],
        },
        {"selected": ["x1"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)
    turn = driver.resume(
        plan_id="plan-1",
        user_text="调整样本权重",
        run_seq=1,
        adjust_params={"sample_weight_col": "not_a_candidate"},
        expected_step_id="tune",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(runner.calls) == 2
    assert "不在已检测候选列中" in turn.messages[-1].content
    loaded_spec = _dataclass_replace(repo.load_plan("plan-1").steps[0])
    assert loaded_spec.inputs["sample_weight_col"] == ""


def test_plan_overview_message_carries_gate_envelope(tmp_path):
    driver, _repo = _driver(tmp_path)

    msg = driver._plan_overview_message(_gated_modeling_plan())

    assert msg.metadata["gate_envelope"]["kind"] == "plan_overview"
    assert msg.metadata["gate_envelope"]["allowed_actions"] == ["confirm", "replan", "clarify", "halt"]


def test_render_screen_shows_metric_columns_and_buckets():
    """Enriched screen render: per-feature KS/IV/missing columns + leakage/suspected/
    unusable buckets with reasons (not just a list of feature names)."""
    text, tables = render_tool_output("screen_features", {
        "selected": ["sig1"],
        "ranked": [["sig1", 0.21]],
        "leakage": [["leak_col", 0.55, "univariate KS 0.550 >= 0.4 — suspected target leakage"]],
        "suspected": [["score_x", 0.30, "name looks like a model output/score"]],
        "unusable": [["const_col", "only 1 distinct non-null value(s)"]],
        "scores": {"sig1": {"ks": 0.21, "iv": 0.18, "missing_rate": 0.03}},
        "n_screened": 4,
    })

    assert "不可用" in text
    selected_table = next(t for t in tables if t["title"].startswith("入选特征"))
    assert selected_table["columns"] == ["特征", "KS", "IV", "缺失率"]
    assert selected_table["rows"][0][0] == "sig1"
    assert "3.0%" in selected_table["rows"][0]  # missing_rate rendered as a percentage
    titles = " ".join(t["title"] for t in tables)
    assert "疑似泄漏" in titles and "疑似模型输出" in titles and "不可用" in titles


def test_resume_with_selection_overrides_screen_output(tmp_path):
    """Confirming the screening gate with an edited selection overrides the screen
    step's proposed ``selected`` so downstream ``$ref:...output.selected`` trains on
    exactly the user's chosen features."""
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)  # pause at tune gate; screen DONE with [sig1, sig2]

    driver.resume(
        plan_id="plan-1",
        user_text="确认",
        run_seq=1,
        selection=["sig1"],
        expected_step_id="tune",
    )

    assert repo.load_step_output("screen")["selected"] == ["sig1"]
    screen_step = next(step for step in repo.load_plan("plan-1").steps if step.id == "screen")
    assert screen_step.output_ref == "metrics:screen:v2"


def test_resume_selection_constrained_to_known_and_allows_force_select(tmp_path):
    """An edited selection may re-pick among screened features — including force-selecting
    a flagged (leakage) column — but cannot inject a column the screen never saw."""
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)

    driver.resume(
        plan_id="plan-1", user_text="确认", run_seq=1,
        selection=["sig1", "leak_col", "never_screened"],  # force-select leakage; drop unknown
        expected_step_id="tune",
    )

    assert repo.load_step_output("screen")["selected"] == ["sig1", "leak_col"]


def test_resume_empty_or_unknown_selection_keeps_proposed(tmp_path):
    """A selection that resolves to nothing is ignored (keep the proposed set) rather
    than training on zero features."""
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)

    driver.resume(
        plan_id="plan-1",
        user_text="确认",
        run_seq=1,
        selection=["never_screened"],
        expected_step_id="tune",
    )

    assert repo.load_step_output("screen")["selected"] == ["sig1", "sig2"]


def test_resume_structured_screen_control_rejects_stale_or_missing_gate_token(tmp_path):
    driver, _repo = _driver(tmp_path)
    driver._repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)

    with pytest.raises(DriverError, match="缺少待确认步骤校验"):
        driver.resume(plan_id="plan-1", user_text="确认", run_seq=1, selection=["sig1"])
    with pytest.raises(DriverError, match="待确认步骤已变化"):
        driver.resume(
            plan_id="plan-1",
            user_text="确认",
            run_seq=1,
            selection=["sig1"],
            expected_step_id="old-gate",
        )


def test_resume_dedup_control_rejects_stale_or_missing_gate_token(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_gated_join_dedup_plan())
    runner = FakeRunner([
        {"joins": [{"feature_id": "feat-1"}]},
        {"needs_dedup": ["feat-1"]},
        {"needs_dedup": []},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-join")
    driver._run_and_handle("plan-join", run_seq=0)

    with pytest.raises(DriverError, match="缺少待确认步骤校验"):
        driver.resume(
            plan_id="plan-join",
            user_text="确认",
            run_seq=1,
            dedup_strategies={"feat-1": "first"},
        )
    with pytest.raises(DriverError, match="待确认步骤已变化"):
        driver.resume(
            plan_id="plan-join",
            user_text="确认",
            run_seq=1,
            dedup_strategies={"feat-1": "first"},
            expected_step_id="old-gate",
        )
    with pytest.raises(DriverError, match="不支持的去重策略"):
        driver.resume(
            plan_id="plan-join",
            user_text="确认",
            run_seq=1,
            dedup_strategies={"feat-1": "drop_all"},
            expected_step_id="execute",
        )
    assert [call[0] for call in runner.calls] == ["propose_join", "confirm_join"]

    turn = driver.resume(
        plan_id="plan-join",
        user_text="确认",
        run_seq=1,
        dedup_strategies={"feat-1": "first"},
        expected_step_id="execute",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [call[0] for call in runner.calls] == ["propose_join", "confirm_join", "confirm_join"]
    assert runner.calls[-1][1]["dedup_strategies"] == {"feat-1": "first"}


class FakeRouterLLM:
    """Returns a fixed instruction-route JSON (agent-mode gate instruction)."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_driver_adjust_reruns_analysis_step_with_new_params(tmp_path):
    """An agent-mode 'adjust' instruction at a gate re-runs the gate's analysis
    dependency with overridden parameters and re-pauses at the gate (spec §3 调整)."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_plan()
    # leakage_ks is a real declared input of screen_features; the template sets it, so it
    # is a legitimate override target (adjusting an UNDECLARED key would fail validation).
    plan.steps[0].inputs = {"leakage_ks": 0.4}
    repo.create_plan(plan)
    runner = FakeRunner([
        {"selected": ["sig1"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
        # re-run after the adjust keeps 3 features
        {"selected": ["sig1", "sig2", "sig3"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    llm = FakeRouterLLM('{"action":"adjust","params":{"leakage_ks":0.3},"constraint":"","reason":"放宽阈值重算"}')
    driver = PlanDriver(repo, executor, llm_client=llm)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)  # screen runs once, pause at the tune gate
    assert len(runner.calls) == 1

    turn = driver.resume(plan_id="plan-1", user_text="把泄漏阈值放宽到 0.3", run_seq=1)

    # screen re-ran with the overridden input and the plan re-paused at the same gate
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(runner.calls) == 2
    assert runner.calls[1][1].get("leakage_ks") == 0.3  # the declared override reached the tool
    assert "保留 **3** 个" in turn.messages[-1].content  # the recomputed screen output is shown
    assert any("调整参数" in m.content for m in turn.messages)


def test_driver_manual_adjust_params_reruns_without_llm_router(tmp_path):
    """Manual-mode structured controls can re-run a gate dependency without relying on
    an LLM to parse free text."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_plan()
    plan.steps[0].inputs = {"leakage_ks": 0.4, "max_missing_rate": 0.95}
    repo.create_plan(plan)
    runner = FakeRunner([
        {"selected": ["sig1"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
        {"selected": ["sig1", "sig2"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)
    turn = driver.resume(
        plan_id="plan-1",
        user_text="调整筛选阈值",
        run_seq=1,
        adjust_params={"leakage_ks": 0.3},
        expected_step_id="tune",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(runner.calls) == 2
    assert runner.calls[1][1]["leakage_ks"] == 0.3
    assert "保留 **2** 个" in turn.messages[-1].content


def test_driver_adjust_resets_downstream_outputs_before_final_gate(tmp_path):
    """Adjusting an upstream dependency at the final gate must re-run dependent
    train/compare steps, not mix new tune results with stale model outputs."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("tune", index=0, tool="tune_hyperparameters", phase="建模"),
            _step("train", index=1, tool="train_models", depends_on=["tune"], phase="建模"),
            _step("compare", index=2, tool="compare_experiments", depends_on=["train"], phase="建模"),
            _step(
                "report",
                index=3,
                tool="generate_model_report",
                depends_on=["tune", "train", "compare"],
                needs_confirmation=True,
                phase="报告",
            ),
        ],
    )
    plan.steps[0].inputs = {"n_trials": 8}
    repo.create_plan(plan)
    runner = FakeRunner([
        {"best_params": {"num_leaves": 31}, "best_metrics": {"test_ks": 0.41}, "n_trials": 8},
        {"best_experiment_id": "exp-old", "best_recipe": "lgb", "experiments": [
            {"experiment_id": "exp-old", "recipe": "lgb", "metrics": {"oot_ks": 0.39}},
        ]},
        {"experiments": [{"recipe": "lgb", "capabilities": {"pmml_supported": True}}]},
        {"best_params": {"num_leaves": 63}, "best_metrics": {"test_ks": 0.45}, "n_trials": 12},
        {"best_experiment_id": "exp-new", "best_recipe": "lgb", "experiments": [
            {"experiment_id": "exp-new", "recipe": "lgb", "metrics": {"oot_ks": 0.43}},
        ]},
        {"experiments": [{"recipe": "lgb", "capabilities": {"pmml_supported": True}}]},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)
    assert [call[0] for call in runner.calls] == ["tune_hyperparameters", "train_models", "compare_experiments"]

    turn = driver.resume(
        plan_id="plan-1",
        user_text="把调参轮数改成 12",
        run_seq=1,
        adjust_params={"n_trials": 12},
        expected_step_id="report",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [call[0] for call in runner.calls] == [
        "tune_hyperparameters",
        "train_models",
        "compare_experiments",
        "tune_hyperparameters",
        "train_models",
        "compare_experiments",
    ]
    assert runner.calls[3][1]["n_trials"] == 12
    assert repo.load_step_output("tune")["best_params"] == {"num_leaves": 63}
    assert repo.load_step_output("train")["best_experiment_id"] == "exp-new"
    loaded = repo.load_plan("plan-1")
    assert {step.id: step.status for step in loaded.steps}["report"] == StepStatus.AWAITING_CONFIRM


def test_driver_adjust_rejects_invalid_structured_threshold_without_reset(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_plan()
    plan.steps[0].inputs = {"leakage_ks": 0.4}
    repo.create_plan(plan)
    runner = FakeRunner([
        {"selected": ["sig1"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
        {"selected": ["sig1", "sig2"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)
    turn = driver.resume(
        plan_id="plan-1",
        user_text="调整筛选阈值",
        run_seq=1,
        adjust_params={"leakage_ks": 1.5},
        expected_step_id="tune",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(runner.calls) == 1
    loaded_screen = _dataclass_replace(repo.load_plan("plan-1").steps[0])
    assert loaded_screen.inputs["leakage_ks"] == 0.4
    assert "leakage_ks 必须是 0 到 1 之间的数字" in turn.messages[-1].content


def test_driver_adjust_with_unmatched_params_does_not_rerun_or_claim_success(tmp_path):
    """If the router extracts a parameter no dependency declares, the driver should ask
    for a clearer instruction instead of resetting steps and saying it adjusted."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_plan()
    plan.steps[0].inputs = {"leakage_ks": 0.4}
    repo.create_plan(plan)
    runner = FakeRunner([
        {"selected": ["sig1"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
        {"selected": ["sig1", "sig2"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    llm = FakeRouterLLM('{"action":"adjust","params":{"unknown_param":123},"constraint":"","reason":"调参数"}')
    driver = PlanDriver(repo, executor, llm_client=llm)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)
    turn = driver.resume(plan_id="plan-1", user_text="unknown_param 调成 123", run_seq=1)

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(runner.calls) == 1
    assert repo.load_step_output("screen")["selected"] == ["sig1"]
    assert "没有识别到可调整的参数" in turn.messages[-1].content
    assert "已按指令调整参数" not in turn.messages[-1].content


def test_driver_replan_instruction_routes_to_structural_replan(tmp_path):
    """An agent-mode 'replan' instruction routes to the structural-replan path (no longer
    the canned stub); with no planner wired it fails gracefully with a recoverable hint."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_gated_modeling_plan())
    runner = FakeRunner([
        {"selected": ["sig1"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))  # planner=None
    llm = FakeRouterLLM('{"action":"replan","params":{},"constraint":"去掉调参步骤","reason":"改流程"}')
    driver = PlanDriver(repo, executor, llm_client=llm)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)  # pause at the tune gate

    turn = driver.resume(plan_id="plan-1", user_text="把调参那步去掉", run_seq=1)

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value  # holds at the gate
    assert "重规划未成功" in turn.messages[-1].content
    assert "仍在接入" not in turn.messages[-1].content  # not the retired stub


class _FakeReplanPlanner:
    """Returns a revised plan (one fresh step) for the structural-replan success path."""

    def __init__(self):
        self.last_instruction = None

    def replan(self, plan, *, completed_summaries, observation, reason, tier, instruction=None):
        self.last_instruction = instruction
        new_step = PlanStep(
            id="new-a", plan_id=plan.id, index=0, title="新步骤A",
            tool_ref=ToolRef("modeling", "screen_features"), inputs={}, depends_on=[],
            post_checks=[], needs_confirmation=False, phase="特征",
        )
        return _dataclass_replace(plan, steps=[new_step], replan_count=plan.replan_count + 1)


def test_driver_replan_success_at_overview_shows_new_plan_and_stays_validated(tmp_path):
    """A 'replan' instruction at the VALIDATED overview regenerates the plan, shows the new
    overview, and stays VALIDATED — nothing runs until 开始 (spec §3 replan + §9#2 gate)."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_gated_modeling_plan())  # VALIDATED, not started
    planner = _FakeReplanPlanner()
    executor = PlanExecutor(
        repo, FakeRunner([]), Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo), planner=planner
    )
    llm = FakeRouterLLM('{"action":"replan","params":{},"constraint":"只跑一步A","reason":"改流程"}')
    driver = PlanDriver(repo, executor, llm_client=llm)

    turn = driver.resume(plan_id="plan-1", user_text="把流程改成只跑一步", run_seq=0)

    assert turn.status == PlanStatus.VALIDATED.value  # not started — still awaits 开始
    assert planner.last_instruction == "把流程改成只跑一步"
    loaded = repo.load_plan("plan-1")
    assert [step.id for step in loaded.steps] == ["new-a"]  # remaining steps replaced
    assert loaded.status == PlanStatus.VALIDATED
    assert any("重规划" in message.content for message in turn.messages)


def test_is_confirm_matches_common_phrasings():
    assert is_confirm("确认")
    assert is_confirm("ok 继续")
    assert not is_confirm("把 age 去掉")


def test_is_confirm_rejects_negated_or_contrasting_confirm_phrases():
    assert not is_confirm("好的但先别执行")
    assert not is_confirm("可以，不过先不要继续")
    assert not is_confirm("ok 先暂停一下")
    assert not is_confirm("不开始")
    assert not is_confirm("do not proceed")
    assert is_confirm("没问题，继续")


def test_render_registry_has_modeling_renderers_and_generic_fallback():
    text, tables = render_tool_output("screen_features", {"selected": ["a"], "leakage": [], "suspected": [], "n_screened": 3})
    assert "特征筛选完成" in text
    spec_text, spec_tables = render_tool_output(
        "choose_modeling_spec",
        {
            "target_type": "binary",
            "recipe": "lgb",
            "recipes": ["lgb", "catboost"],
            "sample_weight_col": "weight",
            "feature_count": 12,
            "n_trials": 9,
            "metric_policy": "higher OOT KS",
            "eligible_algorithms": ["lgb", "catboost"],
            "disabled_algorithms": [{"recipe": "lgb_regressor", "reason": "family mismatch"}],
        },
    )
    assert "建模规格已生成" in spec_text
    assert spec_tables[0]["title"] == "建模规格"
    tuning_text, tuning_tables = render_tool_output(
        "configure_tuning",
        {
            "recipe": "lgb",
            "target_type": "binary",
            "tune_enabled": True,
            "n_trials": 9,
            "sample_weight_col": "weight",
            "params": {"sample_weight_col": "weight"},
            "reason": "LightGBM 使用有界随机搜索。",
        },
    )
    assert "调参配置已生成" in tuning_text
    assert tuning_tables[0]["title"] == "调参配置"
    report_text, report_tables = render_tool_output(
        "generate_model_report",
        {
            "report_path": "/tmp/model_report.xlsx",
            "section_status": [
                {"section": "sample_analysis", "available": True, "reason": None},
                {"section": "vintage", "available": False, "reason": "缺少业务列/字典: mob_observe_cols"},
            ],
        },
    )
    assert "1 个缺输入/跳过" in report_text
    assert report_tables[0]["title"] == "报告章节状态"
    delivery_text, delivery_tables = render_tool_output(
        "post_training_action",
        {
            "native_model_path": "/tmp/model.pkl",
            "monitoring_policy_path": "/tmp/model.monitoring_policy.json",
            "monitoring_policy_markdown_path": "/tmp/model.monitoring_policy.md",
            "monitoring_policy": {"status": "warn", "recommendation": "需补充监控阈值"},
            "capabilities": {"pmml_supported": False, "handoff_supported": False, "native_model_supported": True, "reason": "CatBoost 不支持 PMML"},
            "actions": [
                {"action": "export_pmml", "status": "skipped", "reason": "CatBoost 不支持 PMML"},
                {"action": "handoff_to_validation", "status": "skipped", "reason": "sample_dataset_id is required"},
                {
                    "action": "create_challenger_backtest",
                    "status": "skipped",
                    "reason": "PMML-capable model is required",
                },
            ],
        },
    )
    assert "跳过 3 个" in delivery_text
    assert delivery_tables[0]["title"] == "训练后交付状态"
    assert any("CatBoost 不支持 PMML" in row for row in delivery_tables[0]["rows"])
    assert any("PMML-capable model is required" in row for row in delivery_tables[0]["rows"])
    assert any("监控策略" in row for row in delivery_tables[0]["rows"])
    strategy_text, strategy_tables = render_tool_output(
        "build_strategy",
        {
            "strategy_id": "strategy-1",
            "strategy_type": "approval",
            "score_col": "score",
            "default_decision": "approve",
            "rules": [{"condition": "score < 600", "decision": "reject", "value": None}],
        },
    )
    assert "策略候选已生成" in strategy_text
    assert strategy_tables[0]["title"] == "策略规则(按顺序命中)"
    vintage_text, vintage_tables = render_tool_output(
        "vintage_curve",
        {
            "cohorts": ["202601"],
            "mob_axis": [0, 1],
            "curves": {"202601": [0.0, 0.2]},
            "counts": {"202601": 10},
            "summary": {"trend": "stable", "at_ref": {"202601": 0.2}},
        },
    )
    assert "Vintage 曲线完成" in vintage_text
    assert vintage_tables[0]["title"] == "Vintage 累计坏账率"
    text2, _ = render_tool_output("some_unknown_tool", {"status": "ok", "rows": 10})
    assert "已完成" in text2


def test_render_propose_join_surfaces_fingerprint_consistency_column():
    """C2 shows a 指纹(raw=md5?) column per spec §5: ✓ when key formats match,
    ✗ raw≠md5 when one side is raw and the other md5 (transform_side != 'both')."""
    _text, tables = render_tool_output("propose_join", {
        "joins": [
            {
                "feature_id": "feat_ok",
                "key_pairs": [{"anchor_col": "mobile", "feature_col": "mobile", "transform_side": "both"}],
                "diagnostics": {"match_rate": 0.99, "feature_key_unique": True, "fan_out_detected": False},
            },
            {
                "feature_id": "feat_md5",
                "key_pairs": [{"anchor_col": "mobile", "feature_col": "mobile_md5", "transform_side": "feature"}],
                "diagnostics": {"match_rate": 0.95, "feature_key_unique": True, "fan_out_detected": False},
            },
        ],
    })
    table = next(t for t in tables if t["title"].startswith("拼接诊断"))
    assert "指纹(raw=md5?)" in table["columns"]
    fp_idx = table["columns"].index("指纹(raw=md5?)")
    cells = {row[0]: row[fp_idx] for row in table["rows"]}
    assert cells["feat_ok"] == "✓"
    assert "✗" in cells["feat_md5"] and "raw≠md5" in cells["feat_md5"]
    assert "键指纹不一致" in _text  # the mismatch warning fired


def test_render_make_split_surfaces_split_counts_and_group_distribution():
    """G1 split gate renders train/test/oot counts + a per month/channel distribution table
    so the user can sanity-check the split before screening/training."""
    _text, tables = render_tool_output("make_split", {
        "result_dataset_id": "ds_split",
        "split_col": "model_flag",
        "sample_analysis": {
            "split_counts": {"train": 300, "test": 120, "oot": 180},
            "total_rows": 600,
            "group_distributions": {
                "渠道": {"train": {"A": 200, "B": 100}, "oot": {"A": 180}},
            },
        },
    })
    assert "样本切分完成" in _text
    counts = next(t for t in tables if t["title"].startswith("切分计数"))
    assert ["train", 300, "0.5000"] == [counts["rows"][0][0], counts["rows"][0][1], counts["rows"][0][2]]
    dist = next(t for t in tables if "渠道" in t["title"])
    assert "A" in dist["columns"] and "B" in dist["columns"]


def test_render_propose_join_surfaces_key_relaxation_proposals():
    """C2 shows a 择键建议 table (spec §4/§5) when a low-match key can be relaxed by dropping
    an element — with the reduced key's match/uniqueness/fan-out, as a proposal only."""
    _text, tables = render_tool_output("propose_join", {
        "joins": [{
            "feature_id": "feat_lowmatch",
            "key_pairs": [
                {"anchor_col": "mobile", "feature_col": "mobile", "transform_side": "both"},
                {"anchor_col": "姓名", "feature_col": "姓名", "transform_side": "both"},
            ],
            "diagnostics": {
                "match_rate": 0.10, "feature_key_unique": True, "fan_out_detected": False,
                "key_alternatives": [
                    {"key_pairs": [["mobile", "mobile"]], "dropped": "姓名",
                     "match_rate": 0.98, "feature_key_unique": True, "fan_out_detected": False},
                ],
            },
        }],
    })
    relax = next(t for t in tables if t["title"].startswith("择键建议"))
    assert relax["rows"][0][0] == "feat_lowmatch"
    assert "减「姓名」" in relax["rows"][0][2]
    assert any("0.98" in str(c) for c in relax["rows"][0])
    assert "择键建议" in _text  # the relaxation hint fired


def test_render_tune_leaderboard_includes_full_per_trial_matrix():
    """The tune leaderboard surfaces each trial's train/test/oot × {KS, AUC} +
    head/tail lift at 5% AND 10% + overfit gaps (train-test, train-oot) — spec §5."""
    _text, tables = render_tool_output("tune_hyperparameters", {
        "n_trials": 2, "best_params": {}, "best_metrics": {"test_ks": 0.4, "test_auc": 0.72},
        "trials": [
            {"train_ks": 0.45, "test_ks": 0.40, "oot_ks": 0.38, "score": 0.38,
             "test_auc": 0.72, "oot_auc": 0.70,
             "lift_head_5": 3.4, "lift_head_10": 3.1, "lift_tail_5": 0.2, "lift_tail_10": 0.3,
             "overfit_gap_tt": 0.05, "overfit_gap_to": 0.07},
            {"train_ks": 0.50, "test_ks": 0.35, "oot_ks": 0.33, "score": 0.30,
             "test_auc": 0.69, "oot_auc": 0.67,
             "lift_head_5": 3.0, "lift_head_10": 2.8, "lift_tail_5": 0.25, "lift_tail_10": 0.32,
             "overfit_gap_tt": 0.15, "overfit_gap_to": 0.17},
        ],
    })
    board = next(table for table in tables if table["title"].startswith("trials 排行"))
    for col in ("test_auc", "oot_auc", "头部lift5%", "头部lift10%", "尾部lift5%", "尾部lift10%",
                "过拟合gap(tt)", "过拟合gap(to)"):
        assert col in board["columns"], col
    assert any("0.72" in str(cell) for cell in board["rows"][0])  # AUC reached the top row
    assert any("0.07" in str(cell) for cell in board["rows"][0])  # train-oot gap surfaced
