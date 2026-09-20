"""Turn runner for governed Agent turns."""

from __future__ import annotations

from marvis.agent.join_setup import C1TargetValidationError
from marvis.agent.plan_driver import DriverError
from marvis.data.registry import AuthenticatedDatasetBinding
from marvis.data.registry import DatasetRegistry
from marvis.domain import TASK_TYPE_FEATURE_ANALYSIS
from marvis.domain import TaskRecord
from marvis.repositories.tasks import TaskRepository
import hashlib
from . import c1 as c1_lane
from . import c1_state as c1_state_lane
from . import contracts as contracts_lane
from . import responses as responses_lane
from . import shared as shared_lane
from . import strategy_contracts as strategy_contracts_lane
from . import typed_ui as typed_ui_lane


def _run_driver_turn(
    spec: contracts_lane._TurnHandlerSpec,
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    expected_plan_id: str | None = None,
    expected_plan_status: str | None = None,
    expected_plan_revision: int | None = None,
    expected_plan_fingerprint: str | None = None,
    expected_step_fingerprint: str | None = None,
    confirmation_source: str = "human",
    ui_action: str | None = None,
) -> dict:
    semantic_assignment: dict | None = None
    c1_target_binding: (
        tuple[
            DatasetRegistry,
            AuthenticatedDatasetBinding,
            str | None,
        ]
        | None
    ) = None
    feature_target_col: str | None = None
    feature_semantic_plan_start = False
    if user_text is not None and not ui_action:
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=spec.format_user_display(user_text),
            metadata={"intent": spec.intent},
        )
    try:
        active = shared_lane._active_plan(runtime.plan_repo, task.id)
        typed_ui_lane._validate_typed_ui_action_target(
            active,
            ui_action=ui_action,
            user_text=user_text,
            expected_plan_id=expected_plan_id,
            expected_step_id=expected_step_id,
            expected_plan_status=expected_plan_status,
            expected_plan_revision=expected_plan_revision,
            expected_plan_fingerprint=expected_plan_fingerprint,
            expected_step_fingerprint=expected_step_fingerprint,
        )
        if active is not None:
            stale_response = typed_ui_lane._terminate_stale_strategy_sample_plan(
                spec,
                runtime,
                repo,
                task,
                active,
            )
            if stale_response is not None:
                return stale_response
            driver = shared_lane._driver(runtime)
            # A rendered UI control is already a typed command whose plan/step
            # target was validated above.  Feed the driver its canonical token
            # rather than context-specific display copy such as “确认采纳” or
            # “开始模型验证”; free text with those words must remain on the LLM
            # semantic route and cannot confirm an unrelated live gate.
            driver_user_text = "确认" if ui_action is not None else (user_text or "")
            resume_kwargs = {
                "plan_id": active.id,
                "user_text": driver_user_text,
                "selection": selection,
                "dedup_strategies": dedup_strategies,
                "adjust_params": adjust_params,
                "expected_step_id": expected_step_id,
                "confirmation_source": confirmation_source,
            }
            if ui_action is not None:
                resume_kwargs.update(
                    {
                        "expected_plan_status": expected_plan_status,
                        "expected_plan_revision": expected_plan_revision,
                        "expected_plan_fingerprint": expected_plan_fingerprint,
                        "expected_step_fingerprint": expected_step_fingerprint,
                        "_trusted_ui_action": True,
                    }
                )
            turn = driver.resume(
                **resume_kwargs,
            )
            typed_ui_lane._append_successful_ui_action_messages(
                spec,
                repo,
                task,
                user_text=user_text,
                ui_action=ui_action,
                expected_plan_id=expected_plan_id,
                expected_step_id=expected_step_id,
            )
            typed_ui_lane._append_spec_messages(repo, task, turn, runtime)
            return responses_lane.join_turn_response(repo, task.id)
        setup_result = spec.run_setup(
            runtime,
            repo,
            task,
            user_text,
            confirmation_source,
        )
        if isinstance(setup_result, dict):
            return setup_result
        template_id, slots, start_kwargs = setup_result
        if spec.success_criteria is not None and "success_criteria" not in start_kwargs:
            criteria = spec.success_criteria(task)
            if criteria is not None:
                start_kwargs = {**start_kwargs, "success_criteria": criteria}
        semantic_assignment = start_kwargs.pop(
            "_post_start_c1_assignment",
            None,
        )
        c1_target_binding = start_kwargs.pop(
            "_post_start_c1_target_binding",
            None,
        )
        feature_target_col = start_kwargs.pop(
            "_post_start_feature_target_col",
            None,
        )
        post_start_messages = start_kwargs.pop(
            "_post_start_messages",
            [],
        )
        feature_semantic_plan_start = (
            spec.intent == TASK_TYPE_FEATURE_ANALYSIS
            and feature_target_col is not None
            and c1_state_lane._has_c1_semantic_authorization(semantic_assignment)
        )
        driver = shared_lane._driver(runtime)
        driver.start(
            task_id=task.id,
            template_id=template_id,
            slots=slots,
            tier=runtime.tier,
            _persist_start_turn=lambda conn, start_turn: (
                c1_lane._persist_start_turn_atomically(
                    conn,
                    spec,
                    repo,
                    task,
                    start_turn,
                    semantic_assignment=semantic_assignment,
                    c1_target_binding=c1_target_binding,
                    feature_target_col=feature_target_col,
                    post_start_messages=post_start_messages,
                    user_text=user_text,
                    ui_action=ui_action,
                    expected_plan_id=expected_plan_id,
                    expected_step_id=expected_step_id,
                )
            ),
            **start_kwargs,
        )
        # Plan, semantic authorization receipt, and overview were committed in
        # one SQLite transaction by ``_persist_start_turn_atomically`` above.
        return responses_lane.join_turn_response(repo, task.id)
    except strategy_contracts_lane._StrategySampleDesignRequiredError as exc:
        if spec.intent != "strategy":
            raise
        return responses_lane._strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
            fields=strategy_contracts_lane._STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS,
        )
    except C1TargetValidationError:
        raise
    except spec.setup_error_types as exc:
        return shared_lane.append_workflow_error(
            repo, task, spec, exc, setup_error=True
        )
    except DriverError:
        raise
    except Exception as exc:
        diagnostic_overrides = None
        if feature_semantic_plan_start:
            diagnostic_overrides = {
                "code": "feature_target_plan_start_rolled_back",
                "retry_instruction_sha256": hashlib.sha256(
                    str(user_text or "").strip().encode("utf-8")
                ).hexdigest(),
            }
        return shared_lane.append_workflow_error(
            repo,
            task,
            spec,
            exc,
            diagnostic_overrides=diagnostic_overrides,
        )
