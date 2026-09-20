"""Modeling for governed Agent turns."""

from __future__ import annotations

from marvis.agent.join_setup import AuthenticatedJoinSelection
from marvis.agent.join_setup import C1TargetValidationError
from marvis.agent.join_setup import JoinSetupError
from marvis.agent.join_setup import authenticate_join_selection
from marvis.agent.join_setup import build_join_proposal
from marvis.agent.memory_bridge import fetch_field_convention_hints
from marvis.agent.modeling_setup import ModelingSetupError
from marvis.agent.modeling_setup import build_modeling_proposal
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_AUTO
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN
from marvis.domain import TaskRecord
from marvis.repositories.tasks import TaskRepository
from . import c1 as c1_lane
from . import c1_state as c1_state_lane
from . import contracts as contracts_lane
from . import data_context as data_context_lane
from . import responses as responses_lane
from . import shared as shared_lane
from . import turn_runner as turn_runner_lane
from . import typed_ui as typed_ui_lane


def run_modeling_driver_turn(
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
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
    ui_action: str | None = None,
) -> dict:
    return turn_runner_lane._run_driver_turn(
        _MODELING_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        expected_plan_id=expected_plan_id,
        expected_plan_status=expected_plan_status,
        expected_plan_revision=expected_plan_revision,
        expected_plan_fingerprint=expected_plan_fingerprint,
        expected_step_fingerprint=expected_step_fingerprint,
        confirmation_source=confirmation_source,
        ui_action=ui_action,
    )


def _run_modeling_setup(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict | tuple:
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    conversation = repo.list_agent_messages(task.id)
    c1_state = c1_state_lane._latest_c1_state(conversation)
    c1_assignment = None
    c1_proposal = build_join_proposal(registry, task.id, task.source_dir)
    fresh_c1_state = c1_state_lane._c1_state_from_proposal(c1_proposal)
    c1_ingest_notices = list(c1_proposal.ingest_notices or [])
    anchor_file = next(
        (
            item
            for item in c1_proposal.files
            if item.dataset_id == c1_proposal.anchor_id
        ),
        None,
    )
    ambiguous_single_target = bool(
        c1_proposal.skip
        and c1_proposal.target_col is None
        and anchor_file is not None
        and len(getattr(anchor_file, "target_candidates", None) or []) > 1
    )
    if c1_state is None and c1_proposal.skip:
        # A single-table Agent turn can already carry the exact target column
        # in ordinary language.  Do not discard that governed schema binding
        # merely because there is no join role to confirm.  The downstream
        # modeling setup profiles the bound column and derives binary,
        # continuous, or multiclass deterministically.
        natural_assignment = c1_lane._parse_c1_reply(
            user_text,
            c1_state_lane._c1_state_from_proposal(c1_proposal),
            llm_client=runtime.llm_client,
            trusted_ui_action=(
                runtime.ui_action == "confirm_roles"
                or confirmation_source == CONFIRMATION_SOURCE_AUTO
            ),
            require_semantic_authorization=(
                runtime.require_semantic_text_authorization
            ),
        )
        if natural_assignment and natural_assignment.get("target_col"):
            c1_assignment = natural_assignment
    if not c1_proposal.skip or ambiguous_single_target:
        if c1_state is None or c1_state_lane._c1_snapshot(
            c1_state
        ) != c1_state_lane._c1_snapshot(fresh_c1_state):
            c1_lane._append_c1_message(repo, task.id, c1_proposal)
            return responses_lane.join_turn_response(repo, task.id)
        c1_assignment = c1_lane._parse_c1_reply(
            user_text,
            c1_state,
            llm_client=runtime.llm_client,
            trusted_ui_action=(
                runtime.ui_action == "confirm_roles"
                or confirmation_source == CONFIRMATION_SOURCE_AUTO
            ),
            require_semantic_authorization=(
                runtime.require_semantic_text_authorization
            ),
        )
        if c1_assignment is None:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="请先确认建模文件角色与目标列:无误就回复「确认」，或用下方控件调整后点「确认角色」。",
                metadata={
                    "join_c1": c1_state,
                    "tables": c1_state_lane._c1_table(c1_state),
                },
            )
            return responses_lane.join_turn_response(repo, task.id)
        if not c1_assignment["anchor_id"]:
            return responses_lane.append_join_error(
                repo, task.id, "请先指定建模样本主表（通常是含目标列的那张），再确认。"
            )
        if not c1_assignment.get("target_col"):
            candidates = list(getattr(anchor_file, "target_candidates", None) or [])
            candidate_text = "、".join(f"`{item}`" for item in candidates)
            return responses_lane.append_join_error(
                repo,
                task.id,
                "检测到多个合法目标列，不能只回复确认。请明确回复目标列名"
                + (f"（候选：{candidate_text}）" if candidate_text else "")
                + "。",
            )
    if c1_state_lane._has_c1_semantic_authorization(c1_assignment):
        current_c1_proposal = build_join_proposal(
            registry,
            task.id,
            task.source_dir,
        )
        current_c1_state = c1_state_lane._c1_state_from_proposal(current_c1_proposal)
        if not c1_state_lane._c1_semantic_snapshot_matches(
            c1_assignment, current_c1_state
        ):
            c1_lane._append_c1_message(repo, task.id, current_c1_proposal)
            return responses_lane.join_turn_response(repo, task.id)
    authenticated_selection: AuthenticatedJoinSelection | None = None
    reviewed_c1_hashes: dict[str, str] | None = None
    if c1_assignment is not None:
        reviewed_c1_state = c1_state or fresh_c1_state
        reviewed_c1_hashes = c1_state_lane._c1_expected_content_hashes(
            reviewed_c1_state
        )
        authenticated_selection = authenticate_join_selection(
            registry,
            task.id,
            anchor_id=c1_assignment["anchor_id"],
            feature_ids=c1_assignment["feature_ids"],
            expected_content_hashes=reviewed_c1_hashes,
        )
        try:
            c1_assignment["target_col"] = (
                typed_ui_lane._validated_authenticated_c1_target(
                    registry,
                    authenticated_selection,
                    c1_assignment.get("target_col"),
                )
            )
        except C1TargetValidationError as exc:
            if runtime.ui_action == "confirm_roles":
                raise
            return responses_lane.append_join_error(repo, task.id, str(exc))
    intake_params = data_context_lane._modeling_intake_params(runtime, task, user_text)
    intake_recipes = intake_params.get("recipes")
    intake_target_type = str(intake_params.get("target_type") or "").strip()
    intake_weight_col = (
        str(intake_params.get("sample_weight_col") or "").strip()
        if "sample_weight_col" in intake_params
        else None
    )
    proposal = build_modeling_proposal(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_type=intake_target_type or data_context_lane._modeling_target_type(task),
        n_trials=intake_params.get("n_trials"),
        recipes=(
            [str(item).strip() for item in intake_recipes if str(item).strip()]
            if isinstance(intake_recipes, list)
            else data_context_lane._modeling_recipes(task)
        ),
        sample_weight_col=(
            intake_weight_col
            if intake_weight_col is not None
            else (getattr(task, "sample_weight_col", "") or None)
        ),
        time_col=getattr(task, "time_col", "") or None,
        anchor_id=(c1_assignment or {}).get("anchor_id"),
        join_feature_ids=(c1_assignment or {}).get("feature_ids"),
        target_col=(c1_assignment or {}).get("target_col"),
        field_hints=fetch_field_convention_hints(
            runtime.settings,
            keywords=data_context_lane._modeling_field_hint_keywords(task, c1_proposal),
        ),
        authenticated_selection=authenticated_selection,
        c1_expected_content_hashes=reviewed_c1_hashes,
    )
    counts = proposal.counts
    bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    notices = shared_lane._merge_ingest_notices(
        c1_ingest_notices, proposal.ingest_notices
    )
    setup_message = {
        "role": "assistant",
        "stage": "chat",
        "content": (
            f"开始建模:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}，"
            f"切分 `{proposal.split_col}` train/test/oot="
            f"{counts.get('train', 0)}/{counts.get('test', 0)}/{counts.get('oot', 0)}，"
            f"候选特征 {len(proposal.feature_cols)} 个。先做泄漏感知特征筛选，随后请确认特征集。"
            f"{note_text}{shared_lane._ingest_notice_text(notices)}"
        ),
        "metadata": {"intent": "modeling", "ingest_notices": notices},
    }
    slots = proposal.template_slots()
    split_config = intake_params.get("split_config")
    if isinstance(split_config, dict):
        slots["split_config"] = dict(split_config)
    slots.setdefault("project_meta", data_context_lane._modeling_project_meta(task))
    return (
        proposal.template_id,
        slots,
        {
            "success_criteria": data_context_lane._modeling_success_criteria(task),
            **(
                {"_post_start_c1_assignment": c1_assignment}
                if c1_state_lane._has_c1_semantic_authorization(c1_assignment)
                else {}
            ),
            **(
                {
                    "_post_start_c1_target_binding": (
                        registry,
                        authenticated_selection.anchor,
                        c1_assignment.get("target_col"),
                    )
                }
                if authenticated_selection is not None and c1_assignment is not None
                else {}
            ),
            "_post_start_messages": [setup_message],
        },
    )


_MODELING_SPEC = contracts_lane._TurnHandlerSpec(
    intent="modeling",
    setup_error_types=(JoinSetupError, ModelingSetupError),
    error_label="建模出错",
    run_setup=_run_modeling_setup,
    format_user_display=c1_state_lane._c1_display_text,
)
