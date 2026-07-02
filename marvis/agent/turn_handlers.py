from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from marvis.agent.auto_drive import decide_gate
from marvis.agent.feature_setup import FeatureSetupError, build_feature_proposal
from marvis.agent.join_setup import JoinSetupError, build_join_proposal
from marvis.agent.modeling_setup import ModelingSetupError, build_modeling_proposal
from marvis.agent.plan_driver import DriverError, PlanDriver, is_confirm
from marvis.agent.strategy_setup import StrategySetupError, build_strategy_proposal
from marvis.agent.vintage_setup import VintageSetupError, build_vintage_proposal
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    TaskRecord,
)
from marvis.llm_client import LLMClientError


DRIVER_AGENT_TASK_TYPES = frozenset(
    {
        TASK_TYPE_DATA_JOIN,
        TASK_TYPE_FEATURE_ANALYSIS,
        TASK_TYPE_MODELING,
        TASK_TYPE_STRATEGY,
        TASK_TYPE_VINTAGE,
    }
)

AGENT_MAX_GATES = 8
_TERMINAL_PLAN_STATUS_VALUES = frozenset({"done", "failed", "cancelled"})


@dataclass(frozen=True)
class DriverTurnRuntime:
    settings: Any
    plan_repo: Any
    plan_executor: Any
    planner: Any
    plan_validator: Any
    llm_client: Any | None
    tier: str


def run_join_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    driver = _driver(runtime)
    if user_text is not None:
        display = "已确认文件角色与目标列。" if user_text.startswith("[C1]") else user_text
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=display, metadata={"intent": "data_join"}
        )
    try:
        active = _active_plan(runtime.plan_repo, task.id)
        if active is not None:
            turn = driver.resume(
                plan_id=active.id,
                user_text=user_text or "",
                selection=selection,
                dedup_strategies=dedup_strategies,
                adjust_params=adjust_params,
                expected_step_id=expected_step_id,
            )
            append_driver_messages(repo, task.id, turn)
            return join_turn_response(repo, task.id)
        conversation = repo.list_agent_messages(task.id)
        c1_state = _latest_c1_state(conversation)
        _, registry = _modeling_data_runtime(runtime.settings)
        if c1_state is None:
            proposal = build_join_proposal(registry, task.id, task.source_dir)
            _append_c1_message(repo, task.id, proposal)
            return join_turn_response(repo, task.id)
        assignment = _parse_c1_reply(user_text, c1_state)
        if assignment is None:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="请确认文件角色与目标列:无误就回复「确认」,或用下方控件调整后点「确认角色」。",
                metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
            )
            return join_turn_response(repo, task.id)
        if not assignment["anchor_id"]:
            return append_join_error(repo, task.id, "请先指定样本锚表(通常是含目标列的那张),再确认。")
        if not assignment["feature_ids"]:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="已确认样本表与目标列。只有一张表,无需拼接(数据拼接阶段已跳过)。",
                metadata={"join_skip": True},
            )
            return join_turn_response(repo, task.id)
        turn = driver.start(
            task_id=task.id,
            template_id="data_join",
            slots={"anchor_id": assignment["anchor_id"], "feature_ids": assignment["feature_ids"]},
            tier=runtime.tier,
        )
        append_driver_messages(repo, task.id, turn)
        return join_turn_response(repo, task.id)
    except JoinSetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"数据拼接出错：{exc}")


def run_feature_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    driver = _driver(runtime)
    if user_text is not None:
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=user_text, metadata={"intent": "feature_analysis"}
        )
    try:
        active = _active_plan(runtime.plan_repo, task.id)
        if active is not None:
            turn = driver.resume(
                plan_id=active.id,
                user_text=user_text or "",
                selection=selection,
                dedup_strategies=dedup_strategies,
                adjust_params=adjust_params,
                expected_step_id=expected_step_id,
            )
            append_driver_messages(repo, task.id, turn)
            return join_turn_response(repo, task.id)
        backend, registry = _modeling_data_runtime(runtime.settings)
        proposal = build_feature_proposal(
            registry, backend, task.id, task.source_dir, metrics=_feature_metrics(task)
        )
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"分析数据集 `{proposal.dataset_name}`(目标列 `{proposal.target_col}`,"
                f"{len(proposal.features)} 个候选特征):"
            ),
            metadata={"intent": "feature_analysis"},
        )
        turn = driver.start(
            task_id=task.id,
            template_id=proposal.template_id,
            slots=proposal.template_slots(),
            tier=runtime.tier,
        )
        append_driver_messages(repo, task.id, turn)
        return join_turn_response(repo, task.id)
    except FeatureSetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"特征分析出错：{exc}")


def run_strategy_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    driver = _driver(runtime)
    if user_text is not None:
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=user_text, metadata={"intent": "strategy"}
        )
    try:
        active = _active_plan(runtime.plan_repo, task.id)
        if active is not None:
            turn = driver.resume(
                plan_id=active.id,
                user_text=user_text or "",
                selection=selection,
                dedup_strategies=dedup_strategies,
                adjust_params=adjust_params,
                expected_step_id=expected_step_id,
            )
            append_driver_messages(repo, task.id, turn)
            return join_turn_response(repo, task.id)
        backend, registry = _modeling_data_runtime(runtime.settings)
        proposal = build_strategy_proposal(
            registry,
            backend,
            task.id,
            task.source_dir,
            target_col=getattr(task, "target_col", "") or None,
            score_col=getattr(task, "score_col", "") or None,
        )
        note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
        bad = f"(坏率 {proposal.bad_rate:.2%})" if proposal.bad_rate is not None else ""
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"开始策略分析:样本 `{proposal.dataset_name}`,目标列 `{proposal.target_col}`{bad},"
                f"评分列 `{proposal.score_col}`。已生成默认审批策略候选,回测前会停下确认。"
                f"{note_text}"
            ),
            metadata={"intent": "strategy"},
        )
        turn = driver.start(
            task_id=task.id,
            template_id=proposal.template_id,
            slots=proposal.template_slots(),
            tier=runtime.tier,
        )
        append_driver_messages(repo, task.id, turn)
        return join_turn_response(repo, task.id)
    except StrategySetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"策略分析出错：{exc}")


def run_vintage_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    driver = _driver(runtime)
    if user_text is not None:
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=user_text, metadata={"intent": "vintage"}
        )
    try:
        active = _active_plan(runtime.plan_repo, task.id)
        if active is not None:
            turn = driver.resume(
                plan_id=active.id,
                user_text=user_text or "",
                selection=selection,
                dedup_strategies=dedup_strategies,
                adjust_params=adjust_params,
                expected_step_id=expected_step_id,
            )
            append_driver_messages(repo, task.id, turn)
            return join_turn_response(repo, task.id)
        backend, registry = _modeling_data_runtime(runtime.settings)
        proposal = build_vintage_proposal(
            registry,
            backend,
            task.id,
            task.source_dir,
            target_col=getattr(task, "target_col", "") or None,
            time_col=getattr(task, "time_col", "") or None,
        )
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"开始 Vintage 风险分析:样本 `{proposal.dataset_name}`,"
                f"cohort `{proposal.cohort_col}`,MOB `{proposal.mob_col}`,坏账列 `{proposal.bad_col}`。"
            ),
            metadata={"intent": "vintage"},
        )
        turn = driver.start(
            task_id=task.id,
            template_id=proposal.template_id,
            slots=proposal.template_slots(),
            tier=runtime.tier,
        )
        append_driver_messages(repo, task.id, turn)
        return join_turn_response(repo, task.id)
    except VintageSetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"Vintage 风险分析出错：{exc}")


def run_modeling_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    driver = _driver(runtime)
    if user_text is not None:
        display = "已确认文件角色与目标列。" if user_text.startswith("[C1]") else user_text
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=display, metadata={"intent": "modeling"}
        )
    try:
        active = _active_plan(runtime.plan_repo, task.id)
        if active is not None:
            turn = driver.resume(
                plan_id=active.id,
                user_text=user_text or "",
                selection=selection,
                dedup_strategies=dedup_strategies,
                adjust_params=adjust_params,
                expected_step_id=expected_step_id,
            )
            append_driver_messages(repo, task.id, turn)
            return join_turn_response(repo, task.id)
        backend, registry = _modeling_data_runtime(runtime.settings)
        conversation = repo.list_agent_messages(task.id)
        c1_state = _latest_c1_state(conversation)
        c1_assignment = None
        c1_proposal = build_join_proposal(registry, task.id, task.source_dir)
        if not c1_proposal.skip:
            if c1_state is None:
                _append_c1_message(repo, task.id, c1_proposal)
                return join_turn_response(repo, task.id)
            c1_assignment = _parse_c1_reply(user_text, c1_state)
            if c1_assignment is None:
                repo.add_agent_message(
                    task.id,
                    role="assistant",
                    stage="chat",
                    content="请先确认建模文件角色与目标列:无误就回复「确认」,或用下方控件调整后点「确认角色」。",
                    metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
                )
                return join_turn_response(repo, task.id)
            if not c1_assignment["anchor_id"]:
                return append_join_error(repo, task.id, "请先指定建模样本主表(通常是含目标列的那张),再确认。")
        proposal = build_modeling_proposal(
            registry,
            backend,
            task.id,
            task.source_dir,
            target_type=_modeling_target_type(task),
            recipes=_modeling_recipes(task),
            sample_weight_col=getattr(task, "sample_weight_col", "") or None,
            anchor_id=(c1_assignment or {}).get("anchor_id"),
            join_feature_ids=(c1_assignment or {}).get("feature_ids"),
            target_col=(c1_assignment or {}).get("target_col"),
        )
        counts = proposal.counts
        bad = f"(坏率 {proposal.bad_rate:.2%})" if proposal.bad_rate is not None else ""
        note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"开始建模:样本 `{proposal.dataset_name}`,目标列 `{proposal.target_col}`{bad},"
                f"切分 `{proposal.split_col}` train/test/oot="
                f"{counts.get('train', 0)}/{counts.get('test', 0)}/{counts.get('oot', 0)},"
                f"候选特征 {len(proposal.feature_cols)} 个。先做泄漏感知特征筛选,随后请确认特征集。"
                f"{note_text}"
            ),
            metadata={"intent": "modeling"},
        )
        slots = proposal.template_slots()
        slots.setdefault("project_meta", _modeling_project_meta(task))
        turn = driver.start(
            task_id=task.id,
            template_id=proposal.template_id,
            slots=slots,
            tier=runtime.tier,
        )
        append_driver_messages(repo, task.id, turn)
        return join_turn_response(repo, task.id)
    except (JoinSetupError, ModelingSetupError) as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"建模出错：{exc}")


DRIVER_TURN_FUNCS = {
    TASK_TYPE_MODELING: run_modeling_driver_turn,
    TASK_TYPE_DATA_JOIN: run_join_driver_turn,
    TASK_TYPE_FEATURE_ANALYSIS: run_feature_driver_turn,
    TASK_TYPE_STRATEGY: run_strategy_driver_turn,
    TASK_TYPE_VINTAGE: run_vintage_driver_turn,
}


def dispatch_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    agent_client,
    auto_accept_enabled: bool = False,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    result = DRIVER_TURN_FUNCS[task.task_type](
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
    )
    if agent_client is not None and auto_accept_enabled:
        agent_autodrive_turn(runtime, repo, task, client=agent_client)
        return join_turn_response(repo, task.id)
    return result


def agent_autodrive_turn(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, *, client
) -> None:
    turn_fn = DRIVER_TURN_FUNCS[task.task_type]
    for _ in range(AGENT_MAX_GATES):
        gate = latest_open_gate(repo.list_agent_messages(task.id))
        if gate is None:
            return
        try:
            decision = decide_gate(client, gate=gate)
        except LLMClientError as exc:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"⚠️ 自动决策失败（{exc}），请手动确认或重试。",
                metadata={"intent": "agent_error"},
            )
            return
        action = decision["action"]
        decision_meta = {"intent": "agent_decision", "action": action}
        for key in (
            "params",
            "selection",
            "dedup_strategies",
            "replan_goal",
            "clarifying_question",
            "confidence",
        ):
            if key in decision:
                decision_meta[key] = decision[key]
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=_auto_decision_content(decision),
            metadata=decision_meta,
        )
        gate_meta = gate.get("metadata") if isinstance(gate.get("metadata"), dict) else {}
        gate_step_id = gate_meta.get("step_id")
        if action == "confirm":
            turn_fn(runtime, repo, task, user_text="确认", expected_step_id=gate_step_id)
            continue
        if action == "adjust":
            params = decision.get("params") if isinstance(decision.get("params"), dict) else None
            selection = decision.get("selection") if isinstance(decision.get("selection"), list) else None
            dedup = decision.get("dedup_strategies") if isinstance(decision.get("dedup_strategies"), dict) else None
            if not (params or selection or dedup):
                return
            turn_fn(
                runtime,
                repo,
                task,
                user_text=decision["reason"],
                selection=selection,
                dedup_strategies=dedup,
                adjust_params=params,
                expected_step_id=gate_meta.get("step_id"),
            )
            continue
        if action == "replan":
            turn_fn(
                runtime,
                repo,
                task,
                user_text=decision.get("replan_goal") or decision["reason"],
                expected_step_id=gate_step_id,
            )
            continue
        return


def append_driver_messages(repo: TaskRepository, task_id: str, turn) -> None:
    for message in turn.messages:
        repo.add_agent_message(
            task_id,
            role="assistant",
            stage="chat",
            content=message.content,
            metadata=dict(message.metadata),
        )


def join_turn_response(repo: TaskRepository, task_id: str) -> dict:
    return {"task_id": task_id, "status": "ok", "messages": repo.list_agent_messages(task_id)}


def append_join_error(repo: TaskRepository, task_id: str, detail: str) -> dict:
    repo.add_agent_message(task_id, role="assistant", stage="chat", content=detail, metadata={"error": True})
    return {"task_id": task_id, "status": "error", "messages": repo.list_agent_messages(task_id)}


def latest_open_gate(messages: list[dict]) -> dict | None:
    last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if last_assistant is None:
        return None
    meta = last_assistant.get("metadata") or {}
    if meta.get("error") or meta.get("join_skip"):
        return None
    if meta.get("kind") in ("gate", "plan_overview") or "join_c1" in meta:
        return last_assistant
    return None


def _driver(runtime: DriverTurnRuntime) -> PlanDriver:
    return PlanDriver(
        runtime.plan_repo,
        runtime.plan_executor,
        planner=runtime.planner,
        validator=runtime.plan_validator,
        llm_client=runtime.llm_client,
    )


def _modeling_data_runtime(settings):
    datasets_root = getattr(settings, "datasets_dir", settings.workspace / "datasets")
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(data_repo, backend, datasets_root)
    return backend, registry


def _active_plan(plan_repo, task_id: str):
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        status = getattr(plan.status, "value", plan.status)
        if status not in _TERMINAL_PLAN_STATUS_VALUES:
            return plan
    return None


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
            "title": "输入文件(请确认角色与目标列)",
            "columns": ["文件", "行数", "列数", "含目标列", "候选目标列", "提议角色"],
            "rows": rows,
        }
    ]


def _append_c1_message(repo: TaskRepository, task_id: str, proposal) -> None:
    files = proposal.files
    anchor = next((f for f in files if f.proposed_role == "anchor"), None)
    feature_names = [f.name for f in files if f.proposed_role == "feature"]
    if proposal.skip:
        text = (
            f"我发现 {len(files)} 个数据文件。提议**样本主表 = `{anchor.name if anchor else '?'}`**"
            + (f",目标列 = `{proposal.target_col}`" if proposal.target_col else "(未识别目标列,请指定)")
            + "。只有一张表,确认后将跳过拼接。请确认,或用下方控件调整。"
        )
    else:
        text = (
            f"我发现 {len(files)} 个数据文件,先确认每张的**角色与目标列**(样本是锚,只贴列不改行,**1:1**):\n"
            f"- 提议**样本主表** = `{anchor.name if anchor else '?'}`"
            + (f"(目标列 `{proposal.target_col}`)" if proposal.target_col else "(未识别目标列,请指定)")
            + "\n- 提议**特征表** = "
            + (", ".join(f"`{name}`" for name in feature_names) or "(无)")
            + "\n确认无误回复「确认」;要改就用下方控件选好角色/目标列后点「确认角色」。"
        )
    c1_state = {
        "files": [
            {
                "dataset_id": f.dataset_id,
                "name": f.name,
                "row_count": f.row_count,
                "n_cols": f.n_cols,
                "has_target": f.has_target,
                "candidate_target": f.candidate_target,
                "proposed_role": f.proposed_role,
                "columns": f.columns,
            }
            for f in files
        ],
        "anchor_id": proposal.anchor_id,
        "feature_ids": proposal.feature_ids,
        "target_col": proposal.target_col,
        "skip": proposal.skip,
    }
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=text,
        metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
    )


def _parse_c1_reply(user_text: str | None, c1_state: dict) -> dict | None:
    text = (user_text or "").strip()
    if text.startswith("[C1]"):
        try:
            payload = json.loads(text[len("[C1]"):])
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
            fid for fid in (payload.get("feature_ids") or []) if fid and fid != anchor_id
        ]
        return {
            "anchor_id": anchor_id,
            "feature_ids": feature_ids,
            "target_col": payload.get("target_col"),
        }
    if is_confirm(text):
        return {
            "anchor_id": c1_state.get("anchor_id"),
            "feature_ids": list(c1_state.get("feature_ids") or []),
            "target_col": c1_state.get("target_col"),
        }
    return None


def _c1_dataset_names(c1_state: dict, dataset_ids: list[str]) -> list[str]:
    by_id = {f.get("dataset_id"): f.get("name") for f in c1_state.get("files") or []}
    return [by_id.get(dataset_id) or dataset_id for dataset_id in dataset_ids]


def _feature_metrics(task: TaskRecord) -> list[str]:
    return [str(item).strip() for item in (getattr(task, "metrics", None) or []) if str(item).strip()]


def _modeling_recipes(task: TaskRecord) -> list[str] | None:
    recipes = [str(item).strip() for item in (getattr(task, "recipes", None) or []) if str(item).strip()]
    return recipes or None


def _modeling_target_type(task: TaskRecord) -> str | None:
    target_type = str(getattr(task, "target_type", "") or "").strip()
    return target_type or None


def _modeling_project_meta(task: TaskRecord) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key, value in (
        ("模型名称", getattr(task, "model_name", "")),
        ("模型版本", getattr(task, "model_version", "")),
        ("验证人", getattr(task, "validator", "")),
    ):
        text = str(value or "").strip()
        if text:
            meta[key] = text
    return meta


def _auto_decision_content(decision: dict) -> str:
    reason = str(decision.get("reason") or "").strip() or "自动决策已生成。"
    if decision.get("action") == "clarify" and decision.get("clarifying_question"):
        return f"🤖 {reason}\n\n需要确认:{decision['clarifying_question']}"
    if decision.get("action") == "replan" and decision.get("replan_goal"):
        return f"🤖 {reason}\n\n重规划目标:{decision['replan_goal']}"
    return f"🤖 {reason}"
