"""c1 driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import re
import sqlite3
from marvis.agent.join_setup import JoinSetupError
from marvis.agent.plan_driver import is_confirm
from marvis.data.registry import AuthenticatedDatasetBinding, DatasetRegistry
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskRecord

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import _TurnHandlerSpec
    from . import _append_successful_ui_action_messages
    from . import _ingest_notice_text
    from . import _semantic_c1_recommendation_authorization

def _c1_display_text(user_text: str) -> str:
    return "已确认文件角色与目标列。" if user_text.startswith("[C1]") else user_text

def _latest_c1_state(conversation: list[dict]) -> dict | None:
    for message in reversed(conversation):
        if message.get("role") == "assistant":
            c1 = (message.get("metadata") or {}).get("join_c1")
            if isinstance(c1, dict):
                return c1
    return None

def _c1_table(c1_state: dict) -> list[dict]:
    rows = [
        [
            f.get("name", ""),
            str(f.get("row_count", "")),
            str(f.get("n_cols", "")),
            "是" if f.get("has_target") else "否",
            f.get("candidate_target") or "—",
            "样本主表" if f.get("proposed_role") == "anchor" else "特征表",
        ]
        for f in c1_state.get("files") or []
    ]
    return [
        {
            "title": "输入文件（请确认角色与目标列）",
            "columns": ["文件", "行数", "列数", "含目标列", "候选目标列", "提议角色"],
            "rows": rows,
        }
    ]

def _append_c1_message(repo: TaskRepository, task_id: str, proposal) -> None:
    files = proposal.files
    anchor = next((f for f in files if f.proposed_role == "anchor"), None)
    feature_names = [f.name for f in files if f.proposed_role == "feature"]
    target_candidates = list(getattr(anchor, "target_candidates", None) or [])
    if proposal.skip:
        if len(target_candidates) > 1:
            candidate_text = "、".join(f"`{item}`" for item in target_candidates)
            target_text = f"，检测到多个合法目标列：{candidate_text}；请直接回复要使用的目标列名"
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
    text += _ingest_notice_text(notices)
    c1_state = _c1_state_from_proposal(proposal)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=text,
        metadata={
            "join_c1": c1_state,
            "tables": _c1_table(c1_state),
            "ingest_notices": notices,
        },
    )

def _c1_state_from_proposal(proposal) -> dict:
    files = proposal.files
    return {
        "files": [
            {
                "dataset_id": f.dataset_id,
                "content_hash": f.content_hash,
                "name": f.name,
                "row_count": f.row_count,
                "n_cols": f.n_cols,
                "has_target": f.has_target,
                "candidate_target": f.candidate_target,
                "proposed_role": f.proposed_role,
                "columns": f.columns,
                "target_candidates": list(getattr(f, "target_candidates", None) or []),
            }
            for f in files
        ],
        "anchor_id": proposal.anchor_id,
        "feature_ids": proposal.feature_ids,
        "target_col": proposal.target_col,
        "skip": proposal.skip,
    }

def _c1_expected_content_hashes(c1_state: Mapping[str, object]) -> dict[str, str]:
    """Return the content hashes from the exact C1 card the user reviewed."""

    expected: dict[str, str] = {}
    for item in c1_state.get("files") or []:
        if not isinstance(item, Mapping):
            continue
        dataset_id = str(item.get("dataset_id") or "").strip()
        content_hash = str(item.get("content_hash") or "").strip()
        if not dataset_id or not content_hash:
            raise JoinSetupError(
                "C1 文件快照缺少数据集或内容指纹，请刷新后重新确认。"
            )
        expected[dataset_id] = content_hash
    if not expected:
        raise JoinSetupError("C1 文件快照为空，请刷新后重新确认。")
    return expected

def _c1_snapshot(c1_state: dict) -> str:
    """Canonical safe-data snapshot for an append-only C1 recommendation card."""

    files = [item for item in c1_state.get("files") or [] if isinstance(item, dict)]
    payload = {
        "files": [
            {
                "dataset_id": item.get("dataset_id"),
                "content_hash": item.get("content_hash"),
                "name": item.get("name"),
                "row_count": item.get("row_count"),
                "n_cols": item.get("n_cols"),
                "columns": list(item.get("columns") or []),
                "target_candidates": list(item.get("target_candidates") or []),
                "proposed_role": item.get("proposed_role"),
            }
            for item in files
        ],
        "anchor_id": c1_state.get("anchor_id"),
        "feature_ids": list(c1_state.get("feature_ids") or []),
        "target_col": c1_state.get("target_col"),
        "skip": bool(c1_state.get("skip")),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
            names = _c1_dataset_names(c1_state, anchor_ids)
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
        proposed_assignment = _natural_language_c1_assignment(text, c1_state)
    if proposed_assignment is None:
        natural_target = _natural_language_c1_target(text, c1_state)
        if natural_target is None:
            natural_target = _natural_language_c1_declared_target(text)
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
    if _c1_authorization_is_explicitly_withheld(text):
        return None
    if proposed_assignment is None:
        proposed_assignment = {
            "anchor_id": c1_state.get("anchor_id"),
            "feature_ids": list(c1_state.get("feature_ids") or []),
            "target_col": c1_state.get("target_col"),
        }
    semantic_authorization = _semantic_c1_recommendation_authorization(
        text,
        c1_state,
        llm_client,
        proposed_assignment=proposed_assignment,
    )
    if semantic_authorization is not None:
        return {
            **proposed_assignment,
            "_semantic_authorization": semantic_authorization,
        }
    return None

_C1_WITHHELD_AUTHORIZATION = re.compile(
    r"[?？]|(?:先别|先不要|暂不|不要|别)(?:再)?继续|(?:暂停|停止|暂缓|稍后再说)|"
    r"(?:如果|假如|若|只有|等到|待).{0,40}(?:才|就|再)继续|"
    r"\b(?:do\s+not|don't|dont)\s+(?:continue|proceed)|"
    r"\b(?:pause|hold|defer|later|if\b.{0,40}\bthen)\b",
    re.IGNORECASE,
)

def _c1_authorization_is_explicitly_withheld(text: str) -> bool:
    return bool(_C1_WITHHELD_AUTHORIZATION.search(text or ""))

def _has_c1_semantic_authorization(assignment: dict | None) -> bool:
    return isinstance((assignment or {}).get("_semantic_authorization"), dict)

def _c1_semantic_snapshot_matches(assignment: dict, c1_state: dict) -> bool:
    authorization = assignment.get("_semantic_authorization")
    if not isinstance(authorization, dict):
        return True
    expected_snapshot = str(authorization.get("c1_snapshot_sha256") or "")
    current_snapshot = hashlib.sha256(
        _c1_snapshot(c1_state).encode("utf-8")
    ).hexdigest()
    current_assignment = {
        "anchor_id": assignment.get("anchor_id"),
        "feature_ids": list(assignment.get("feature_ids") or []),
        "target_col": assignment.get("target_col"),
    }
    authorized_assignment = authorization.get("proposed_assignment")
    if not isinstance(authorized_assignment, dict):
        return False
    normalized_authorized_assignment = {
        "anchor_id": authorized_assignment.get("anchor_id"),
        "feature_ids": list(authorized_assignment.get("feature_ids") or []),
        "target_col": authorized_assignment.get("target_col"),
    }
    assignment_payload = json.dumps(
        current_assignment,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_assignment = str(
        authorization.get("proposed_assignment_sha256") or ""
    )
    current_assignment_hash = hashlib.sha256(
        assignment_payload.encode("utf-8")
    ).hexdigest()
    return (
        bool(expected_snapshot)
        and bool(expected_assignment)
        and hmac.compare_digest(expected_snapshot, current_snapshot)
        and hmac.compare_digest(expected_assignment, current_assignment_hash)
        and current_assignment == normalized_authorized_assignment
    )

def _record_c1_semantic_authorization(
    repo: TaskRepository,
    task_id: str,
    assignment: dict,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    authorization = assignment.pop("_semantic_authorization", None)
    if not isinstance(authorization, dict):
        return
    message = {
        "role": "assistant",
        "stage": "chat",
        "content": "已通过独立语义复核，确认采用当前文件角色与目标列建议。",
        "metadata": {
            "intent": "c1_semantic_authorization",
            "display_in_timeline": False,
            "semantic_authorization": authorization,
        },
    }
    if conn is None:
        repo.add_agent_message(task_id, **message)
    else:
        repo.add_agent_message_on_connection(conn, task_id, **message)

def _persist_start_turn_atomically(
    conn: sqlite3.Connection,
    spec: _TurnHandlerSpec,
    repo: TaskRepository,
    task: TaskRecord,
    turn,
    *,
    semantic_assignment: dict | None,
    c1_target_binding: tuple[
        DatasetRegistry,
        AuthenticatedDatasetBinding,
        str | None,
    ] | None,
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

    _append_successful_ui_action_messages(
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
        _record_c1_semantic_authorization(
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

def _natural_language_c1_assignment(text: str, c1_state: dict) -> dict | None:
    """Resolve Agent-mode file-role changes expressed in ordinary language.

    Agent mode intentionally makes the structured C1 controls evidence-only and
    tells users to describe role changes in chat.  Keep this parser narrow and
    deterministic: an exact file name must be mentioned next to a role term.
    """

    if not text:
        return None
    files = [item for item in c1_state.get("files") or [] if isinstance(item, dict)]
    anchor_matches = [
        item
        for item in files
        if _c1_file_role_is_mentioned(
            text,
            str(item.get("name") or ""),
            ("样本主表", "主样本", "主表", "锚点表"),
        )
    ]
    if len(anchor_matches) != 1:
        return None
    anchor_id = anchor_matches[0].get("dataset_id")
    if not anchor_id:
        return None

    explicit_feature_ids = [
        item.get("dataset_id")
        for item in files
        if item.get("dataset_id") != anchor_id
        and _c1_file_role_is_mentioned(
            text,
            str(item.get("name") or ""),
            ("特征表",),
        )
    ]
    ignores_remaining = bool(
        re.search(
            r"(?:其余|其他|剩余)[^。；\n]{0,80}(?:忽略|不使用|排除)"
            r"|(?:不要|不再)[^。；\n]{0,40}作为特征表",
            text,
        )
    )
    if ignores_remaining:
        feature_ids = explicit_feature_ids
    else:
        feature_ids = [
            dataset_id
            for dataset_id in c1_state.get("feature_ids") or []
            if dataset_id and dataset_id != anchor_id
        ]
        feature_ids.extend(explicit_feature_ids)

    current_anchor_id = c1_state.get("anchor_id")
    target_col = _natural_language_c1_target(text, c1_state, anchor_id=anchor_id)
    declared_target = _natural_language_c1_declared_target(text)
    if target_col is None and declared_target is not None:
        target_col = declared_target
    if target_col is None and anchor_id == current_anchor_id:
        target_col = c1_state.get("target_col")
    return {
        "anchor_id": anchor_id,
        "feature_ids": list(dict.fromkeys(feature_ids)),
        "target_col": target_col,
    }

def _c1_file_role_is_mentioned(text: str, file_name: str, role_terms: tuple[str, ...]) -> bool:
    if not file_name:
        return False
    name_pattern = re.escape(file_name)
    role_pattern = "(?:" + "|".join(re.escape(term) for term in role_terms) + ")"
    return bool(
        re.search(
            rf"{name_pattern}[^，,。；;\n]{{0,48}}{role_pattern}"
            rf"|{role_pattern}[^，,。；;\n]{{0,48}}{name_pattern}",
            text,
            re.IGNORECASE,
        )
    )

def _natural_language_c1_target(
    text: str,
    c1_state: dict,
    *,
    anchor_id: str | None = None,
) -> str | None:
    """Resolve an exact schema column mentioned in an Agent-mode reply."""

    if not text:
        return None
    anchor_id = anchor_id or c1_state.get("anchor_id")
    anchor = next(
        (item for item in c1_state.get("files") or [] if item.get("dataset_id") == anchor_id),
        None,
    )
    if not isinstance(anchor, dict):
        return None
    candidates = list(anchor.get("target_candidates") or anchor.get("columns") or [])
    matches = []
    for column in candidates:
        name = str(column)
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, re.IGNORECASE):
            matches.append(name)
    return matches[0] if len(matches) == 1 else None

def _natural_language_c1_declared_target(text: str) -> str | None:
    """Preserve an explicit target token so anchor validation can fail closed."""

    marker_first = re.search(
        r"(?:目标列|标签列)\s*(?:改为|改成|设为|设置为|指定为|选择|选用|使用|用|是|为|[:：=])"
        r"\s*[`'\"“”]?([^\s,，。；;:`'\"“”]+)",
        text or "",
        re.IGNORECASE,
    )
    if marker_first is not None:
        return str(marker_first.group(1) or "").strip() or None
    value_first = re.search(
        r"(?:把|将|用)?\s*[`'\"“”]?([^\s,，。；;:`'\"“”]+)[`'\"“”]?"
        r"\s*(?:作为|当作|设为|设置为|指定为|用作|当|是)\s*(?:目标列|标签列)",
        text or "",
        re.IGNORECASE,
    )
    if value_first is None:
        return None
    return str(value_first.group(1) or "").strip() or None

def _c1_dataset_names(c1_state: dict, dataset_ids: list[str]) -> list[str]:
    by_id = {f.get("dataset_id"): f.get("name") for f in c1_state.get("files") or []}
    return [by_id.get(dataset_id) or dataset_id for dataset_id in dataset_ids]

