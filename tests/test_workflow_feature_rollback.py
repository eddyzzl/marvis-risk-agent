from __future__ import annotations

from types import SimpleNamespace

import pytest

from marvis.agent.plan_driver import DriverError, PlanDriver
from marvis.agent.driver_turn import DriverTurn
from marvis.agent.turn_handlers import _maybe_handle_workflow_recovery_turn
from marvis.agent.workflow_recovery import (
    parse_champion_refit_revision_intent,
    parse_tuning_budget_revision_intent,
    parse_workflow_rollback_intent,
)
from marvis.db import connect, init_db
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef
from marvis.repositories.plans import PlanRepository
from marvis.state_machine import ConflictError


REAL_FEATURE_ROLLBACK_COMMAND = (
    "诊断发现当前特征集包含标签派生的贷后结果字段。请排除 "
    "mob3_ever30、mob4_ever30、mob5_ever30、mob6_ever30 和 appl_seq_x，"
    "从“特征筛选”步骤重新执行，并重置后续步骤。保留数据切分和建模规格，"
    "源数据不做修改。"
)
EXCLUSIONS = [
    "mob3_ever30",
    "mob4_ever30",
    "mob5_ever30",
    "mob6_ever30",
    "appl_seq_x",
]
FEATURES = [*EXCLUSIONS, "safe_a", "safe_b"]
TUNING_BUDGET_COMMAND = (
    "修改当前计划的调参配置：lgb=1、xgb=1、catboost=1，总预算3轮。"
    "保留当前数据切分和187个特征，只更新配置并重置调参及后续步骤，"
    "不要立即执行。"
)
SKIP_CHAMPION_REFIT_COMMAND = (
    "重试当前失败的选择实验步骤，不做 train+test 全量重训，"
    "沿用已训练的 experiment_2b05015b52f64474a5b6bd7213765f43。"
)


def test_real_ui_feature_revision_sentence_parses_as_typed_rollback() -> None:
    intent = parse_workflow_rollback_intent(REAL_FEATURE_ROLLBACK_COMMAND)

    assert intent is not None
    assert intent.action == "revise_and_rerun"
    assert intent.root_step == "feature_screening"
    assert list(intent.excluded_features) == EXCLUSIONS


@pytest.mark.parametrize(
    "text",
    [
        "继续",
        "重试当前失败步骤",
        "为什么要排除 mob3_ever30 后，从特征筛选重新执行？",
        "不要排除 mob3_ever30，从特征筛选重新执行",
        "排除 mob3_ever30，从特征筛选重新执行，但是不要了",
        "从特征筛选重新执行",
        "排除 mob3_ever30",
    ],
)
def test_feature_revision_parser_rejects_retry_questions_negation_and_partial_commands(
    text: str,
) -> None:
    assert parse_workflow_rollback_intent(text) is None


@pytest.mark.parametrize(
    ("text", "expected_default", "expected_by_recipe"),
    [
        (TUNING_BUDGET_COMMAND, None, {"lgb": 1, "xgb": 1, "catboost": 1}),
        (
            "不用调参这么多轮，只需要流程跑通，调参每种算法1轮就行",
            1,
            {},
        ),
        (
            "把 LightGBM、XGBoost、CatBoost 的调参预算都改为 1 轮",
            1,
            {},
        ),
    ],
)
def test_tuning_budget_revision_parser_accepts_explicit_bounded_commands(
    text: str,
    expected_default: int | None,
    expected_by_recipe: dict[str, int],
) -> None:
    intent = parse_tuning_budget_revision_intent(text)

    assert intent is not None
    assert intent.default_n_trials == expected_default
    assert dict(intent.n_trials_by_recipe) == expected_by_recipe
    assert intent.execute is False


@pytest.mark.parametrize(
    "text",
    [
        "可以把调参改成1轮吗？",
        "调参少一点",
        "不要调参了",
        "调参预算 lgb=1、lgb=2",
        "调参改成0轮",
        "调参改成1.5轮",
        "把未知算法 foo 改成1轮",
        "从失败调参步骤继续，沿用 lgb、xgb、catboost 各 40 轮",
    ],
)
def test_tuning_budget_revision_parser_rejects_questions_cancellation_and_ambiguity(
    text: str,
) -> None:
    assert parse_tuning_budget_revision_intent(text) is None


def test_champion_refit_revision_parser_accepts_explicit_retry_and_experiment() -> None:
    intent = parse_champion_refit_revision_intent(SKIP_CHAMPION_REFIT_COMMAND)

    assert intent is not None
    assert intent.refit_on_train_plus_test is False
    assert (
        intent.selected_experiment_id
        == "experiment_2b05015b52f64474a5b6bd7213765f43"
    )


@pytest.mark.parametrize(
    "text",
    [
        "为什么要跳过全量重训？",
        "可以不做 refit 吗？",
        "不做 train+test 全量重训",
        "重试当前失败步骤",
        "重试当前失败步骤，但是继续做全量重训",
    ],
)
def test_champion_refit_revision_parser_rejects_questions_and_partial_commands(
    text: str,
) -> None:
    assert parse_champion_refit_revision_intent(text) is None


def _step(
    step_id: str,
    index: int,
    title: str,
    tool: str,
    *,
    depends_on: list[str] | None = None,
    status: StepStatus = StepStatus.DONE,
    inputs: dict | None = None,
    needs_confirmation: bool = False,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        plan_id="plan-rollback",
        index=index,
        title=title,
        tool_ref=ToolRef("modeling", tool),
        inputs=inputs or {},
        depends_on=depends_on or [],
        post_checks=[],
        needs_confirmation=needs_confirmation,
        status=status,
    )


def _failed_modeling_plan() -> Plan:
    return Plan(
        id="plan-rollback",
        task_id="task-rollback",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.FAILED,
        steps=[
            _step("split", 0, "切分样本", "make_split"),
            _step(
                "spec",
                1,
                "选择建模规格",
                "choose_modeling_spec",
                depends_on=["split"],
            ),
            _step(
                "screen",
                2,
                "特征筛选",
                "screen_features",
                depends_on=["split", "spec"],
                inputs={
                    "dataset_id": "$ref:split.output.result_dataset_id",
                    "features": "$ref:spec.output.feature_cols",
                    "target_col": "label_sqandzy_new",
                    "split_col": "$ref:split.output.split_col",
                },
                needs_confirmation=True,
            ),
            _step(
                "select",
                3,
                "精选特征",
                "select_features",
                depends_on=["screen", "spec"],
                inputs={"features": "$ref:screen.output.selected"},
                needs_confirmation=True,
            ),
            _step(
                "configure",
                4,
                "配置调参",
                "configure_tuning",
                depends_on=["select", "spec"],
                needs_confirmation=True,
            ),
            _step(
                "tune",
                5,
                "调参",
                "tune_hyperparameters",
                depends_on=["split", "select", "configure"],
                status=StepStatus.FAILED,
                inputs={"features": "$ref:select.output.selected"},
                needs_confirmation=True,
            ),
            _step(
                "train",
                6,
                "训练模型",
                "train_models",
                depends_on=["split", "spec", "select", "tune"],
                status=StepStatus.PENDING,
                inputs={"features": "$ref:select.output.selected"},
            ),
        ],
    )


def _persist_failed_plan(tmp_path) -> tuple[PlanRepository, Plan]:
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _failed_modeling_plan()
    repo.create_plan(plan)
    outputs = {
        "split": {
            "result_dataset_id": "ds-split",
            "feature_cols": FEATURES,
            "split_col": "split_tag",
        },
        "spec": {"feature_cols": FEATURES, "target_col": "label_sqandzy_new"},
        "screen": {"selected": FEATURES},
        "select": {"selected": FEATURES},
        "configure": {
            "recipe": "lgb",
            "recipes": ["lgb", "xgb", "catboost"],
            "n_trials_by_recipe": {"lgb": 40, "xgb": 40, "catboost": 40},
            "total_n_trials": 120,
            "reason": "test",
        },
    }
    for step in plan.steps:
        output = outputs.get(step.id)
        if output is None:
            continue
        step.output_ref = repo.store_step_output(step.id, output)
        repo.update_step(step)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET confirmed = 1, review_json = "
            "'[{\"reviewer\":\"test\",\"passed\":true,\"reasons\":[],"
            "\"at\":\"2026-07-22T00:00:00+00:00\",\"status\":\"evaluated\"}]' "
            "WHERE id IN ('screen', 'select', 'configure', 'tune')"
        )
        conn.execute("UPDATE plan_steps SET error = 'worker failed' WHERE id = 'tune'")
    return repo, plan


def test_repository_rolls_back_same_plan_atomically_and_versions_next_screen_output(
    tmp_path,
) -> None:
    repo, plan = _persist_failed_plan(tmp_path)
    revised_inputs = {
        **next(step for step in plan.steps if step.id == "screen").inputs,
        "features": ["safe_a", "safe_b"],
    }

    reset = repo.rollback_failed_plan_from_step(
        plan.id,
        "screen",
        "tune",
        root_inputs=revised_inputs,
        excluded_features=EXCLUSIONS,
        expected_plan_revision=0,
        expected_root_output_ref="metrics:screen:v1",
    )

    assert reset == ["screen", "select", "configure", "tune", "train"]
    revised = repo.load_plan(plan.id)
    assert revised.status == PlanStatus.RUNNING
    assert revised.replan_count == 1
    assert [step.status for step in revised.steps[:2]] == [StepStatus.DONE, StepStatus.DONE]
    assert [step.output_ref for step in revised.steps[:2]] == [
        "metrics:split:v1",
        "metrics:spec:v1",
    ]
    for step in revised.steps[2:]:
        assert step.status == StepStatus.PENDING
        assert step.output_ref is None
        assert step.error is None
        assert step.review_verdicts == []
    assert revised.steps[2].inputs["features"] == ["safe_a", "safe_b"]
    assert revised.steps[3].inputs["features"] == "$ref:screen.output.selected"
    assert revised.steps[5].inputs["features"] == "$ref:select.output.selected"
    with connect(repo.db_path) as conn:
        confirmations = conn.execute(
            "SELECT id, confirmed FROM plan_steps WHERE plan_id = ? ORDER BY idx",
            (plan.id,),
        ).fetchall()
    assert all(int(row["confirmed"]) == 0 for row in confirmations[2:])
    audit = repo.list_audit(kind="plan.step.rollback")[-1]
    assert audit["detail"]["excluded_features"] == EXCLUSIONS
    assert audit["detail"]["plan_revision_after"] == 1
    assert repo.store_step_output("screen", {"selected": ["safe_a", "safe_b"]}) == (
        "metrics:screen:v2"
    )


class _PauseAtFreshScreenExecutor:
    def __init__(self, repo: PlanRepository) -> None:
        self.repo = repo

    def run(self, plan_id: str):
        plan = self.repo.load_plan(plan_id)
        screen = next(step for step in plan.steps if step.id == "screen")
        assert screen.status == StepStatus.PENDING
        assert screen.needs_confirmation is True
        screen.status = StepStatus.AWAITING_CONFIRM
        self.repo.update_step(screen)
        self.repo.set_plan_status(plan_id, PlanStatus.AWAITING_CONFIRM)
        return SimpleNamespace(status=PlanStatus.AWAITING_CONFIRM)


class _PauseAtFreshTuningConfigExecutor:
    def __init__(self, repo: PlanRepository) -> None:
        self.repo = repo
        self.calls = 0

    def run(self, plan_id: str):
        self.calls += 1
        plan = self.repo.load_plan(plan_id)
        configure = next(step for step in plan.steps if step.id == "configure")
        assert configure.status == StepStatus.PENDING
        assert configure.needs_confirmation is True
        configure.status = StepStatus.AWAITING_CONFIRM
        self.repo.update_step(configure)
        self.repo.set_plan_status(plan_id, PlanStatus.AWAITING_CONFIRM)
        return SimpleNamespace(status=PlanStatus.AWAITING_CONFIRM)


def test_driver_revision_filters_from_upstream_and_requires_fresh_screen_confirmation(
    tmp_path,
) -> None:
    repo, plan = _persist_failed_plan(tmp_path)
    driver = PlanDriver(repo, _PauseAtFreshScreenExecutor(repo))

    turn = driver.rollback_failed_plan_to_feature_screen(
        plan.id,
        "tune",
        excluded_features=EXCLUSIONS,
        run_seq=4,
    )

    assert turn.plan_id == plan.id
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert repo.count_plans_for_task(plan.task_id) == 1
    revised = repo.load_plan(plan.id)
    assert revised.steps[2].status == StepStatus.AWAITING_CONFIRM
    assert revised.steps[2].inputs["features"] == ["safe_a", "safe_b"]
    assert EXCLUSIONS[0] not in revised.steps[2].inputs["features"]


def test_driver_tuning_budget_revision_preserves_features_and_reopens_config_gate(
    tmp_path,
) -> None:
    repo, plan = _persist_failed_plan(tmp_path)
    executor = _PauseAtFreshTuningConfigExecutor(repo)
    driver = PlanDriver(repo, executor)

    turn = driver.rollback_failed_plan_to_tuning_config(
        plan.id,
        "tune",
        default_n_trials=1,
        n_trials_by_recipe={},
        run_seq=5,
    )

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert executor.calls == 1
    setup = turn.messages[-1].metadata["modeling_setup"]
    assert setup["n_trials"] == 1
    assert setup["n_trials_by_recipe"] == {
        "lgb": 1,
        "xgb": 1,
        "catboost": 1,
    }
    spec_table = next(
        table
        for table in turn.messages[-1].metadata["tables"]
        if table.get("title") == "建模规格"
    )
    assert ["按算法调参预算", "lgb=1、xgb=1、catboost=1（总计 3 轮）"] in spec_table[
        "rows"
    ]
    revised = repo.load_plan(plan.id)
    assert [step.status for step in revised.steps[:4]] == [StepStatus.DONE] * 4
    assert revised.steps[4].status == StepStatus.AWAITING_CONFIRM
    assert revised.steps[4].inputs["n_trials_by_recipe"] == {
        "lgb": 1,
        "xgb": 1,
        "catboost": 1,
    }
    assert [step.status for step in revised.steps[5:]] == [StepStatus.PENDING] * 2
    assert repo.load_step_output("select")["selected"] == FEATURES
    audit = repo.list_audit(kind="plan.step.rollback")[-1]
    assert audit["detail"]["revision_kind"] == "tuning_budget"
    assert audit["detail"]["n_trials_by_recipe"] == {
        "lgb": 1,
        "xgb": 1,
        "catboost": 1,
    }


@pytest.mark.parametrize(
    "excluded, error",
    [
        (["unknown_feature"], "不在当前已确认"),
        (["label_sqandzy_new"], "目标列或切分列"),
        (["split_tag"], "目标列或切分列"),
        (FEATURES, "特征集合为空"),
    ],
)
def test_driver_revision_invalid_columns_leave_plan_unchanged(
    tmp_path,
    excluded: list[str],
    error: str,
) -> None:
    repo, plan = _persist_failed_plan(tmp_path)
    before = repo.load_plan(plan.id)

    with pytest.raises(DriverError, match=error):
        PlanDriver(repo, _PauseAtFreshScreenExecutor(repo)).rollback_failed_plan_to_feature_screen(
            plan.id,
            "tune",
            excluded_features=excluded,
        )

    after = repo.load_plan(plan.id)
    assert after.status == PlanStatus.FAILED
    assert after.replan_count == before.replan_count
    assert [step.status for step in after.steps] == [step.status for step in before.steps]
    assert [step.output_ref for step in after.steps] == [
        step.output_ref for step in before.steps
    ]
    assert repo.list_audit(kind="plan.step.rollback") == []


@pytest.mark.parametrize("concurrent_change", ["status", "revision", "not_ancestor"])
def test_repository_revision_concurrency_and_non_ancestor_guards_are_zero_write(
    tmp_path,
    concurrent_change: str,
) -> None:
    repo, plan = _persist_failed_plan(tmp_path)
    with connect(repo.db_path) as conn:
        if concurrent_change == "status":
            conn.execute("UPDATE plans SET status = 'running' WHERE id = ?", (plan.id,))
        elif concurrent_change == "revision":
            conn.execute("UPDATE plans SET replan_count = 1 WHERE id = ?", (plan.id,))
        else:
            conn.execute(
                "UPDATE plan_steps SET depends_on_json = '[\"spec\"]' "
                "WHERE id IN ('select', 'configure', 'tune')"
            )

    with pytest.raises(ConflictError):
        repo.rollback_failed_plan_from_step(
            plan.id,
            "screen",
            "tune",
            root_inputs={
                **next(step for step in plan.steps if step.id == "screen").inputs,
                "features": ["safe_a", "safe_b"],
            },
            excluded_features=EXCLUSIONS,
            expected_plan_revision=0,
            expected_root_output_ref="metrics:screen:v1",
        )

    assert repo.list_audit(kind="plan.step.rollback") == []
    with connect(repo.db_path) as conn:
        screen = conn.execute(
            "SELECT status, output_ref, inputs_json FROM plan_steps WHERE id = 'screen'"
        ).fetchone()
    assert screen["status"] == "done"
    assert screen["output_ref"] == "metrics:screen:v1"
    assert "safe_a\",\"safe_b" not in screen["inputs_json"]


class _MessageRepo:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = list(messages)

    def list_agent_messages(self, task_id: str) -> list[dict]:
        return list(self.messages)

    def add_agent_message(self, task_id: str, **message) -> dict:
        stored = {"id": f"m{len(self.messages) + 1}", **message}
        self.messages.append(stored)
        return stored


def test_real_ui_sentence_uses_same_plan_rollback_before_plain_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _failed_modeling_plan()

    class _PlanRepo:
        def list_plans_for_task(self, task_id: str):
            return [plan]

    class _Driver:
        def __init__(self) -> None:
            self.rollback_calls: list[dict] = []
            self.retry_calls: list[dict] = []

        def rollback_failed_plan_to_feature_screen(self, plan_id, step_id, **kwargs):
            self.rollback_calls.append(
                {"plan_id": plan_id, "step_id": step_id, **kwargs}
            )
            return DriverTurn(plan_id, PlanStatus.AWAITING_CONFIRM.value, [])

        def retry_failed_step(self, plan_id, step_id, **kwargs):
            self.retry_calls.append({"plan_id": plan_id, "step_id": step_id, **kwargs})
            raise AssertionError("typed rollback must take precedence over plain retry")

    driver = _Driver()
    monkeypatch.setattr("marvis.agent.turn_handlers._driver", lambda runtime: driver)
    messages = _MessageRepo(
        [
            {
                "id": "failure-1",
                "role": "assistant",
                "stage": "chat",
                "content": "调参失败。",
                "metadata": {
                    "error": True,
                    "error_diagnostic": {
                        "workflow": "modeling",
                        "code": "workflow_step_failed",
                        "summary": "调参失败。",
                        "cause": "worker interrupted",
                        "retryable": True,
                    },
                    "failure_envelope": {
                        "failed_step_id": "tune",
                        "retryable": True,
                        "run_seq": 3,
                    },
                },
            }
        ]
    )
    task = SimpleNamespace(
        id="task-rollback",
        task_type="modeling",
        run_mode="agent",
    )
    runtime = SimpleNamespace(plan_repo=_PlanRepo(), settings=None)

    response = _maybe_handle_workflow_recovery_turn(
        runtime,
        messages,
        task,
        user_text=REAL_FEATURE_ROLLBACK_COMMAND,
        selection=None,
        dedup_strategies=None,
        adjust_params=None,
        expected_step_id=None,
        recovery_bypass=False,
    )

    assert response is not None
    assert driver.rollback_calls == [
        {
            "plan_id": "plan-rollback",
            "step_id": "tune",
            "excluded_features": EXCLUSIONS,
            "run_seq": 4,
        }
    ]
    assert driver.retry_calls == []
    assert next(
        item
        for item in messages.messages
        if (item.get("metadata") or {}).get("intent") == "workflow_recovery_revision"
    )["metadata"]["plan_id"] == "plan-rollback"
    assert not any("join_c1" in (item.get("metadata") or {}) for item in messages.messages)


def test_real_ui_tuning_budget_sentence_reopens_configuration_not_plain_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _failed_modeling_plan()

    class _PlanRepo:
        def list_plans_for_task(self, task_id: str):
            return [plan]

    class _Driver:
        def __init__(self) -> None:
            self.budget_calls: list[dict] = []
            self.retry_calls: list[dict] = []

        def rollback_failed_plan_to_tuning_config(self, plan_id, step_id, **kwargs):
            self.budget_calls.append(
                {"plan_id": plan_id, "step_id": step_id, **kwargs}
            )
            return DriverTurn(plan_id, PlanStatus.AWAITING_CONFIRM.value, [])

        def retry_failed_step(self, plan_id, step_id, **kwargs):
            self.retry_calls.append({"plan_id": plan_id, "step_id": step_id, **kwargs})
            raise AssertionError("typed budget revision must not use plain retry")

    driver = _Driver()
    monkeypatch.setattr("marvis.agent.turn_handlers._driver", lambda runtime: driver)
    messages = _MessageRepo(
        [
            {
                "id": "failure-1",
                "role": "assistant",
                "stage": "chat",
                "content": "调参失败。",
                "metadata": {
                    "error": True,
                    "error_diagnostic": {
                        "workflow": "modeling",
                        "code": "workflow_step_failed",
                        "summary": "调参失败。",
                        "cause": "worker interrupted",
                        "retryable": True,
                    },
                    "failure_envelope": {
                        "failed_step_id": "tune",
                        "retryable": True,
                        "run_seq": 4,
                    },
                },
            }
        ]
    )
    task = SimpleNamespace(
        id="task-rollback",
        task_type="modeling",
        run_mode="agent",
    )
    runtime = SimpleNamespace(plan_repo=_PlanRepo(), settings=None)

    response = _maybe_handle_workflow_recovery_turn(
        runtime,
        messages,
        task,
        user_text=TUNING_BUDGET_COMMAND,
        selection=None,
        dedup_strategies=None,
        adjust_params=None,
        expected_step_id=None,
        recovery_bypass=False,
    )

    assert response is not None
    assert driver.budget_calls == [
        {
            "plan_id": "plan-rollback",
            "step_id": "tune",
            "default_n_trials": None,
            "n_trials_by_recipe": {"lgb": 1, "xgb": 1, "catboost": 1},
            "run_seq": 5,
        }
    ]
    assert driver.retry_calls == []
    revision = next(
        item
        for item in messages.messages
        if (item.get("metadata") or {}).get("intent")
        == "workflow_recovery_tuning_revision"
    )
    assert revision["metadata"]["n_trials_by_recipe"] == {
        "lgb": 1,
        "xgb": 1,
        "catboost": 1,
    }


def test_real_ui_skip_refit_sentence_retries_select_with_explicit_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_step = _step(
        "select-experiment",
        0,
        "选择实验",
        "select_experiment",
        status=StepStatus.FAILED,
        inputs={
            "experiment_ids": "$ref:train.output.experiment_ids",
            "target_type": "binary",
            "selection_policy": {"require_pmml": True, "require_handoff": True},
        },
        needs_confirmation=True,
    )
    plan = Plan(
        id="plan-select",
        task_id="task-select",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.FAILED,
        steps=[select_step],
    )

    class _PlanRepo:
        def list_plans_for_task(self, task_id: str):
            return [plan]

    class _Driver:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def retry_failed_step(self, plan_id, step_id, **kwargs):
            self.calls.append({"plan_id": plan_id, "step_id": step_id, **kwargs})
            return DriverTurn(plan_id, PlanStatus.AWAITING_CONFIRM.value, [])

    driver = _Driver()
    monkeypatch.setattr("marvis.agent.turn_handlers._driver", lambda runtime: driver)
    messages = _MessageRepo(
        [
            {
                "id": "failure-select",
                "role": "assistant",
                "stage": "chat",
                "content": "选择实验失败。",
                "metadata": {
                    "error": True,
                    "error_diagnostic": {
                        "workflow": "modeling",
                        "code": "workflow_step_failed",
                        "summary": "选择实验失败。",
                        "cause": "worker RSS exceeded memory limit",
                        "retryable": True,
                    },
                    "failure_envelope": {
                        "failed_step_id": "select-experiment",
                        "retryable": True,
                        "run_seq": 7,
                    },
                },
            }
        ]
    )
    task = SimpleNamespace(
        id="task-select",
        task_type="modeling",
        run_mode="agent",
    )
    runtime = SimpleNamespace(plan_repo=_PlanRepo(), settings=None)

    response = _maybe_handle_workflow_recovery_turn(
        runtime,
        messages,
        task,
        user_text=SKIP_CHAMPION_REFIT_COMMAND,
        selection=None,
        dedup_strategies=None,
        adjust_params=None,
        expected_step_id=None,
        recovery_bypass=False,
    )

    assert response is not None
    assert driver.calls == [
        {
            "plan_id": "plan-select",
            "step_id": "select-experiment",
            "run_seq": 8,
            "inputs": {
                "refit_on_train_plus_test": False,
                "selected_experiment_id": (
                    "experiment_2b05015b52f64474a5b6bd7213765f43"
                ),
            },
            "preserve_target_confirmation": False,
        }
    ]
    revision = next(
        item
        for item in messages.messages
        if (item.get("metadata") or {}).get("intent")
        == "workflow_recovery_champion_refit_revision"
    )
    assert revision["metadata"]["refit_on_train_plus_test"] is False
