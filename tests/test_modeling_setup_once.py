"""The four-step modeling setup form is a one-time decision (eval S4).

S4 showed the same 建模协作台 three times: the setup form's 「确认方案，开始执行」
only re-split the sample and re-opened the same gate, and later gates rendered
the form again.  These tests pin the fixes: one click both applies and
confirms the setup, an unrelated later adjustment keeps the confirmed setup,
and the setup payload describes the existing partition, the task's OOT goal
and the whole-period nature of a time OOT from platform data only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from marvis.agent.gate_adapters import _oot_requirement, render_gate_dependencies
from marvis.agent.gate_payloads import build_modeling_setup_payload, split_adoption_note
from marvis.agent.modeling_setup import build_modeling_proposal
from marvis.agent.plan_driver import PlanDriver
from marvis.db import PlanRepository, init_db
from marvis.orchestrator.contracts import (
    Plan,
    PlanStatus,
    StepStatus,
    plan_fingerprint,
    plan_step_confirmation_fingerprint,
)
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.reviewer import Reviewer
from marvis.packs.modeling.prepare import _make_split
from marvis.packs.modeling.prepare_tools import _split_sample_analysis
from tests.test_modeling_auto_split_detection import _register_source
from tests.test_modeling_existing_split_contract import _frame
from tests.test_plan_driver import (
    FakeHooks,
    FakeLLM,
    FakeRunner,
    _register_result_dataset,
    _step,
)


def _setup_plan() -> Plan:
    """切分样本 -> 选择建模规格 -> 特征筛选(setup gate) -> 调参(gate)."""
    split = _step("split", index=0, tool="make_split", phase="特征")
    split.inputs = {
        "dataset_id": "ds-upload",
        "target_col": "y",
        "split_col": "split",
        "split_config": {},
    }
    spec = _step("spec", index=1, tool="choose_modeling_spec", depends_on=["split"], phase="特征")
    spec.inputs = {"n_trials": 1}
    screen = _step(
        "screen",
        index=2,
        tool="screen_features",
        depends_on=["split", "spec"],
        needs_confirmation=True,
        phase="特征",
    )
    screen.inputs = {"leakage_ks": 0.4}
    tune = _step(
        "tune",
        index=3,
        tool="tune_hyperparameters",
        depends_on=["spec", "screen"],
        needs_confirmation=True,
        phase="建模",
    )
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[split, spec, screen, tune],
    )


def _split_output(dataset_id: str, counts: dict, *, oot_months: dict | None = None) -> dict:
    distributions = {"apply_month": {"oot": oot_months}} if oot_months else {}
    return {
        "result_dataset_id": dataset_id,
        "split_col": "split",
        "split_values": {key: key for key in counts},
        "holdout_values": ["oot"] if "oot" in counts else [],
        "feature_cols": ["x1", "x2"],
        "available_columns": ["x1", "x2", "y", "split", "apply_month"],
        "sample_analysis": {
            "split_counts": counts,
            "total_rows": sum(counts.values()),
            "group_distributions": distributions,
        },
    }


_SPEC = {"target_type": "binary", "recipe": "lr", "recipes": ["lr"], "feature_count": 2, "n_trials": 1}
_SCREEN = {"selected": ["x1"], "leakage": [], "suspected": [], "n_screened": 2, "ranked": [], "unusable": [], "scores": {}}


def _driver(tmp_path, outputs):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    for dataset_id in ("ds-split-1", "ds-split-2"):
        _register_result_dataset(db_path, dataset_id)
    repo = PlanRepository(db_path)
    repo.create_plan(_setup_plan())
    runner = FakeRunner(outputs)
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    driver = PlanDriver(repo, executor)
    repo.confirm_plan("plan-1")
    first = driver._run_and_handle("plan-1", run_seq=0)
    assert first.messages[-1].metadata["step_id"] == "screen"
    return driver, repo, runner


def _ui_snapshot(repo, step_id: str) -> dict:
    plan = repo.load_plan("plan-1")
    gate = next(step for step in plan.steps if step.id == step_id)
    return {
        "expected_step_id": step_id,
        "expected_plan_status": plan.status.value,
        "expected_plan_revision": plan.replan_count,
        "expected_plan_fingerprint": plan_fingerprint(plan),
        "expected_step_fingerprint": plan_step_confirmation_fingerprint(gate, confirmed=False),
        "_trusted_ui_action": True,
    }


def test_setup_form_confirm_with_edits_runs_past_the_setup_gate_once(tmp_path):
    driver, repo, runner = _driver(tmp_path, [
        _split_output("ds-split-1", {"train": 360, "test": 120}),
        _SPEC,
        _split_output(
            "ds-split-2",
            {"train": 270, "test": 90, "oot": 120},
            oot_months={"2026-01": 120},
        ),
        _SPEC,
        _SCREEN,
    ])

    turn = driver.resume(
        plan_id="plan-1",
        user_text="确认",
        run_seq=1,
        adjust_params={
            "split_config": {"test_size": 0.25, "oot_by_time": "apply_month", "oot_size": 0.2},
        },
        _continue_after_adjust=True,
        **_ui_snapshot(repo, "screen"),
    )

    # One click: re-split, then screen ran without a second setup confirmation.
    assert [call[0] for call in runner.calls] == [
        "make_split",
        "choose_modeling_spec",
        "make_split",
        "choose_modeling_spec",
        "screen_features",
    ]
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    statuses = {step.id: step.status for step in repo.load_plan("plan-1").steps}
    assert statuses["screen"] == StepStatus.DONE
    assert statuses["tune"] == StepStatus.AWAITING_CONFIRM
    # The re-opened setup gate is not rendered again; the next gate is.
    gates = [message for message in turn.messages if message.stage == "gate"]
    assert [message.metadata["step_id"] for message in gates] == ["tune"]
    chat = "\n".join(message.content for message in turn.messages if message.stage == "chat")
    assert "已按你确认的方案切分样本：train 270 行、test 90 行、OOT 120 行（共 480 行）" in chat
    # Plain statement that whole months were cut: 120 rows = 25.0%, not the 20% target.
    assert "OOT 按 `apply_month` 整月切分：2026-01，共 120 行进入 OOT，占 25.0%" in chat
    assert "填写的 20% 只是目标" in chat


def test_setup_gate_adjust_without_the_one_click_flag_still_reopens_for_review(tmp_path):
    driver, repo, runner = _driver(tmp_path, [
        _split_output("ds-split-1", {"train": 360, "test": 120}),
        _SPEC,
        _split_output("ds-split-2", {"train": 300, "test": 100, "oot": 80}),
        _SPEC,
    ])

    turn = driver.resume(
        plan_id="plan-1",
        user_text="按 apply_month 切 OOT",
        run_seq=1,
        adjust_params={"split_config": {"oot_by_time": "apply_month"}},
        expected_step_id="screen",
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [call[0] for call in runner.calls][-1] == "choose_modeling_spec"
    assert turn.messages[-1].metadata["step_id"] == "screen"
    assert turn.messages[-1].metadata["modeling_setup"]


def test_later_threshold_adjust_keeps_the_confirmed_setup(tmp_path):
    driver, repo, runner = _driver(tmp_path, [
        _split_output("ds-split-1", {"train": 300, "test": 100, "oot": 80}),
        _SPEC,
        _SCREEN,
        _SCREEN,
    ])
    driver.resume(plan_id="plan-1", user_text="确认", run_seq=1, **_ui_snapshot(repo, "screen"))
    assert repo.load_plan("plan-1").steps[3].status == StepStatus.AWAITING_CONFIRM

    turn = driver.resume(
        plan_id="plan-1",
        user_text="调整筛选阈值",
        run_seq=2,
        adjust_params={"leakage_ks": 0.5},
        **_ui_snapshot(repo, "tune"),
    )

    # The screen re-ran with the new threshold, but split/spec were untouched,
    # so the setup form is not asked again; the user lands on the target gate.
    assert [call[0] for call in runner.calls] == [
        "make_split",
        "choose_modeling_spec",
        "screen_features",
        "screen_features",
    ]
    assert runner.calls[-1][1]["leakage_ks"] == 0.5
    gates = [message for message in turn.messages if message.stage == "gate"]
    assert [message.metadata["step_id"] for message in gates] == ["tune"]
    assert any("沿用你之前确认的建模方案" in message.content for message in turn.messages)


def test_later_adjust_that_changes_the_spec_reopens_the_setup_form(tmp_path):
    driver, repo, runner = _driver(tmp_path, [
        _split_output("ds-split-1", {"train": 300, "test": 100, "oot": 80}),
        _SPEC,
        _SCREEN,
        {**_SPEC, "n_trials": 5},
    ])
    driver.resume(plan_id="plan-1", user_text="确认", run_seq=1, **_ui_snapshot(repo, "screen"))

    turn = driver.resume(
        plan_id="plan-1",
        user_text="调参轮数改成 5",
        run_seq=2,
        adjust_params={"n_trials": 5},
        expected_step_id="tune",
    )

    # The modeling spec itself changed: that is a new setup to review.
    assert turn.messages[-1].metadata["step_id"] == "screen"
    assert turn.messages[-1].metadata["modeling_setup"]["n_trials"] == 5


def test_setup_payload_offers_existing_partition_and_task_oot_goal():
    split = _step("split", index=0, tool="make_split")
    split.inputs = {"dataset_id": "ds-upload", "split_col": "split", "split_config": {}}
    spec = _step("spec", index=1, tool="choose_modeling_spec", depends_on=["split"])
    gate = _step("screen", index=2, tool="screen_features", depends_on=["split", "spec"], needs_confirmation=True)
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        steps=[split, spec, gate],
        success_criteria=[{"metric": "oot_ks", "min": 0.3, "label": "OOT KS"}],
    )
    outputs = {
        "split": _split_output("ds-1", {"train": 300, "test": 60, "oot": 120}),
        "spec": _SPEC,
    }

    rendered = render_gate_dependencies(
        plan,
        gate,
        outputs.get,
        load_dataset=lambda dataset_id: type("Dataset", (), {"role": "sample"})(),
    )

    setup = rendered.modeling_setup
    assert setup["oot_requested"] is True
    assert setup["oot_requirement"] == "OOT KS ≥ 0.3"
    summary = setup["split_summary"]
    assert summary["existing_split"] == {"column": "split", "origin": "uploaded"}
    assert summary["has_oot"] is True
    assert summary["time_column_candidates"] == ["apply_month"]
    assert summary["notes"] == []


def test_split_summary_marks_platform_made_frames_and_ref_inputs_as_derived():
    split = _step("split", index=0, tool="make_split")
    split.inputs = {"dataset_id": "$ref:执行拼接.output.result_dataset_id", "split_col": "model_flag"}
    payload = build_modeling_setup_payload(
        _SPEC,
        _step("spec", index=1, tool="choose_modeling_spec"),
        split_output=_split_output("ds-1", {"train": 300, "test": 60}),
        split_step=split,
        split_source_role="derived",
    )
    assert payload["split_summary"]["existing_split"] == {"column": "model_flag", "origin": "derived"}
    assert payload["oot_requested"] is False


@pytest.mark.parametrize(
    "criteria,expected",
    [
        ([], None),
        ([{"metric": "test_ks", "min": 0.3}], None),
        ([{"metric": "oot_ks", "min": 0.3, "label": "OOT KS"}], "OOT KS ≥ 0.3"),
        (
            [{"schema_version": "v1", "criteria": [{"metric": "oot_auc", "minimum": 0.7}]}],
            "OOT AUC ≥ 0.7",
        ),
    ],
)
def test_oot_requirement_reads_only_the_tasks_own_criteria(criteria, expected):
    plan = Plan(
        id="p",
        task_id="t",
        goal="g",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        steps=[],
        success_criteria=criteria,
    )
    assert _oot_requirement(plan) == expected


def test_time_oot_note_states_whole_month_cut_from_real_split():
    months = ["2025-10", "2025-11", "2025-12", "2026-01"]
    frame = pd.DataFrame({
        "x1": range(480),
        "y": [0, 1] * 240,
        "apply_month": [month for month in months for _ in range(120)],
    })
    config = {"test_size": 0.25, "oot_by_time": "apply_month", "oot_size": 0.2}
    prepared = _make_split(frame, config, seed=7)
    analysis = _split_sample_analysis(prepared["split"], frame[["apply_month"]])
    split = _step("split", index=0, tool="make_split")
    split.inputs = {"split_config": config}

    note = split_adoption_note({"sample_analysis": analysis, "holdout_values": ["oot"]}, split)

    # Dense-rank time cut: the newest whole month, 120/480 = 25%, not the 20% asked.
    assert analysis["split_counts"]["oot"] == 120
    assert "OOT 按 `apply_month` 整月切分：2026-01，共 120 行进入 OOT，占 25.0%" in note
    assert "填写的 20% 只是目标，同一个月的样本不会被拆开" in note


def _default_time_col_frame():
    frame = _frame("split")
    frame["apply_month"] = ["2025-12"] * 360 + ["2026-01"] * 240
    frame["stmt_date"] = frame["apply_month"]
    return frame


def test_setup_note_does_not_name_the_schema_default_time_column(tmp_path):
    backend, registry, dataset = _register_source(tmp_path, _default_time_col_frame())
    proposal = build_modeling_proposal(
        registry, backend, dataset.task_id, tmp_path / "source",
        target_col="y", anchor_id=dataset.id, recipes=["lr"], n_trials=1,
        time_col="apply_month",  # TaskCreate default: nobody asked for it
    )
    assert proposal.split_col == "split"
    assert not any("apply_month" in note for note in proposal.notes)
    assert not any("请求的时间列" in note for note in proposal.notes)


def test_setup_note_still_names_an_explicitly_requested_time_column(tmp_path):
    backend, registry, dataset = _register_source(tmp_path, _default_time_col_frame())
    proposal = build_modeling_proposal(
        registry, backend, dataset.task_id, tmp_path / "source",
        target_col="y", anchor_id=dataset.id, recipes=["lr"], n_trials=1,
        time_col="stmt_date",
    )
    assert any("未按请求的时间列 `stmt_date`" in note for note in proposal.notes)


def test_setup_note_names_algorithms_for_this_target_only(tmp_path):
    # Eval S4: the note listed every recipe code (regressors included) for a
    # binary A card: 「算法:`lr`（可选 lgb/xgb/…/lgb_regressor/…）」.
    backend, registry, dataset = _register_source(tmp_path, _default_time_col_frame())
    proposal = build_modeling_proposal(
        registry, backend, dataset.task_id, tmp_path / "source",
        target_col="y", anchor_id=dataset.id, recipes=["lr"], n_trials=1,
    )
    [note] = [note for note in proposal.notes if note.startswith("算法")]
    assert note == "算法：逻辑回归（也可以改用 LightGBM、XGBoost、CatBoost、评分卡、神经网络）。"
