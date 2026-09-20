"""C1 for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from marvis.agent.join_setup import JoinSetupError
from marvis.agent.plan_driver import is_confirm
from marvis.data.registry import AuthenticatedDatasetBinding
from marvis.data.registry import DatasetRegistry
from marvis.domain import TaskRecord
from marvis.repositories.tasks import TaskRepository
import json
import sqlite3
from . import c1_state as c1_state_lane
from . import contracts as contracts_lane
from . import semantic_authorization as semantic_authorization_lane
from . import shared as shared_lane
from . import typed_ui as typed_ui_lane


def _append_c1_message(repo: TaskRepository, task_id: str, proposal) -> None:
    files = proposal.files
    anchor = next((f for f in files if f.proposed_role == "anchor"), None)
    feature_names = [f.name for f in files if f.proposed_role == "feature"]
    target_candidates = list(getattr(anchor, "target_candidates", None) or [])
    if proposal.skip:
        if len(target_candidates) > 1:
            candidate_text = "、".join(f"`{item}`" for item in target_candidates)
            target_text = (
                f"，检测到多个合法目标列：{candidate_text}；请直接回复要使用的目标列名"
            )
        elif proposal.target_col:
            target_text = f"，目标列 = `{proposal.target_col}`"
        else:
            target_text = "（未识别目标列，请指定）"
        text = (
            f"我发现 {len(files)} 个数据文件。提议**样本主表 = `{anchor.name if anchor else '?'}`**"
            + target_text
            + "。只有一张表，确认后将跳过拼接。"
        )
    else:
        text = (
            f"我发现 {len(files)} 个数据文件，先确认每张的**角色与目标列**（样本是锚，只贴列不改行，**1:1**）:\n"
            f"- 提议**样本主表** = `{anchor.name if anchor else '?'}`"
            + (
                f"（目标列 `{proposal.target_col}`）"
                if proposal.target_col
                else "（未识别目标列，请指定）"
            )
            + "\n- 提议**特征表** = "
            + (", ".join(f"`{name}`" for name in feature_names) or "（无）")
            + "\n确认无误回复「确认」；要改就用下方控件选好角色/目标列后点「确认角色」。"
        )
    notices = list(getattr(proposal, "ingest_notices", None) or [])
    text += shared_lane._ingest_notice_text(notices)
    c1_state = c1_state_lane._c1_state_from_proposal(proposal)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=text,
        metadata={
            "join_c1": c1_state,
            "tables": c1_state_lane._c1_table(c1_state),
            "ingest_notices": notices,
        },
    )


def _parse_c1_reply(
    user_text: str | None,
    c1_state: dict,
    *,
    llm_client=None,
    trusted_ui_action=False,
    require_semantic_authorization=False,
) -> dict | None:
    text = (user_text or "").strip()
    if text.startswith("[C1]"):
        if not trusted_ui_action:
            return None
        try:
            payload = json.loads(text[len("[C1]") :])
        except (ValueError, TypeError):
            return None
        anchor_ids = [aid for aid in (payload.get("anchor_ids") or []) if aid]
        if not anchor_ids:
            single_anchor_id = payload.get("anchor_id")
            anchor_ids = [single_anchor_id] if single_anchor_id else []
        anchor_ids = list(dict.fromkeys(anchor_ids))  # de-dup, preserve order
        if len(anchor_ids) > 1:
            names = c1_state_lane._c1_dataset_names(c1_state, anchor_ids)
            raise JoinSetupError(
                "样本主表只能有一个，请把 "
                + "、".join(names[1:])
                + " 改为「特征表」或「忽略」。"
            )
        anchor_id = anchor_ids[0] if anchor_ids else payload.get("anchor_id")
        feature_ids = [
            fid
            for fid in (payload.get("feature_ids") or [])
            if fid and fid != anchor_id
        ]
        return {
            "anchor_id": anchor_id,
            "feature_ids": feature_ids,
            "target_col": payload.get("target_col"),
        }
    proposed_assignment = None
    if is_confirm(text):
        proposed_assignment = {
            "anchor_id": c1_state.get("anchor_id"),
            "feature_ids": list(c1_state.get("feature_ids") or []),
            "target_col": c1_state.get("target_col"),
        }
    if proposed_assignment is None:
        proposed_assignment = c1_state_lane._natural_language_c1_assignment(
            text, c1_state
        )
    if proposed_assignment is None:
        natural_target = c1_state_lane._natural_language_c1_target(text, c1_state)
        if natural_target is None:
            natural_target = c1_state_lane._natural_language_c1_declared_target(text)
        if natural_target:
            proposed_assignment = {
                "anchor_id": c1_state.get("anchor_id"),
                "feature_ids": list(c1_state.get("feature_ids") or []),
                "target_col": natural_target,
            }
    if not require_semantic_authorization:
        return proposed_assignment
    if trusted_ui_action:
        return proposed_assignment
    if c1_state_lane._c1_authorization_is_explicitly_withheld(text):
        return None
    if proposed_assignment is None:
        proposed_assignment = {
            "anchor_id": c1_state.get("anchor_id"),
            "feature_ids": list(c1_state.get("feature_ids") or []),
            "target_col": c1_state.get("target_col"),
        }
    semantic_authorization = (
        semantic_authorization_lane._semantic_c1_recommendation_authorization(
            text,
            c1_state,
            llm_client,
            proposed_assignment=proposed_assignment,
        )
    )
    if semantic_authorization is not None:
        return {
            **proposed_assignment,
            "_semantic_authorization": semantic_authorization,
        }
    return None


def _persist_start_turn_atomically(
    conn: sqlite3.Connection,
    spec: contracts_lane._TurnHandlerSpec,
    repo: TaskRepository,
    task: TaskRecord,
    turn,
    *,
    semantic_assignment: dict | None,
    c1_target_binding: tuple[
        DatasetRegistry,
        AuthenticatedDatasetBinding,
        str | None,
    ]
    | None,
    feature_target_col: str | None,
    post_start_messages: Sequence[Mapping[str, object]],
    user_text: str | None,
    ui_action: str | None,
    expected_plan_id: str | None,
    expected_step_id: str | None,
) -> None:
    """Persist setup state, authorization evidence, and plan atomically."""

    if c1_target_binding is not None:
        registry, binding, target_col = c1_target_binding
        registry.persist_authenticated_target_on_connection(
            conn,
            binding,
            target_col,
        )
        repo.update_target_col_on_connection(
            conn,
            task.id,
            target_col,
        )

    if feature_target_col is not None:
        repo.update_target_col_on_connection(
            conn,
            task.id,
            feature_target_col,
        )

    typed_ui_lane._append_successful_ui_action_messages(
        spec,
        repo,
        task,
        user_text=user_text,
        ui_action=ui_action,
        expected_plan_id=expected_plan_id,
        expected_step_id=expected_step_id,
        conn=conn,
    )

    if semantic_assignment is not None:
        c1_state_lane._record_c1_semantic_authorization(
            repo,
            task.id,
            semantic_assignment,
            conn=conn,
        )
    for message in post_start_messages:
        repo.add_agent_message_on_connection(
            conn,
            task.id,
            role=str(message.get("role") or "assistant"),
            stage=str(message.get("stage") or "chat"),
            content=str(message.get("content") or ""),
            metadata=dict(message.get("metadata") or {}),
        )
    for message in turn.messages:
        repo.add_agent_message_on_connection(
            conn,
            task.id,
            role="assistant",
            stage="chat",
            content=message.content,
            metadata=dict(message.metadata),
        )
