from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json

from marvis.agent.service import REQUIRED_AGENT_REPORT_KEYS, agent_conclusions_confirmed
from marvis.agent.validation_messages import (
    add_and_stream_agent_message,
    agent_stage_label,
    agent_stage_opening_text,
    format_conclusion_values,
    model_metadata,
    stream_agent_message,
)
from marvis.agent.validation_service import raise_if_agent_cancelled
from marvis.agent_memory.api_support import (
    agent_memory_context_from_store,
    audit_agent_memory_use_from_store,
)
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import TaskRepository
from marvis.domain import TASK_TYPE_VALIDATION, TaskRecord, TaskStatus
from marvis.repositories.validation_contracts import (
    ValidationContractRepository,
    require_confirmed_validation_input_contract,
)


MAX_INPUT_CONFIRMATION_CANDIDATE_LINES = 24
MAX_INPUT_CONFIRMATION_CANDIDATES_PER_FIELD = 3
MAX_INPUT_CONFIRMATION_VALUE_CHARS = 160


def _require_ready_contract_for_v2(
    settings,
    task: TaskRecord,
) -> None:
    if (
        task.task_type == TASK_TYPE_VALIDATION
        and task.validation_workflow_version == 2
    ):
        require_confirmed_validation_input_contract(
            ValidationContractRepository(settings.db_path),
            task.id,
        )


@dataclass(frozen=True)
class ValidationStageDependencies:
    perform_scan_task: Callable
    run_notebook_stage: Callable
    run_pmml_scoring_stage: Callable
    run_metrics_stage: Callable
    run_report_stage: Callable
    agent_pipeline_settings: Callable
    agent_evidence_from_settings: Callable
    add_agent_report_ready_message: Callable
    is_metrics_failure: Callable[[TaskRecord], bool]
    compose_agent_start_message: Callable
    summarize_stage: Callable
    generate_word_conclusions: Callable
    failure_summary: Callable


def open_agent_stage(
    repo: TaskRepository,
    *,
    task: TaskRecord,
    task_id: str,
    stage: str,
    model_profile: dict,
    opening_message_id: str | None,
    auto_accept: bool = False,
    deps: ValidationStageDependencies,
) -> None:
    if auto_accept and stage != "scan":
        add_agent_auto_stage_start_message(
            repo,
            task_id=task_id,
            stage=stage,
            model_profile=model_profile,
            validation_workflow_version=task.validation_workflow_version,
        )
    if stage == "scan":
        if opening_message_id:
            stream_agent_message(
                repo,
                opening_message_id,
                task_id=task_id,
                model_profile=model_profile,
                producer=lambda on_delta: deps.compose_agent_start_message(
                    task=task,
                    model_profile=model_profile,
                    on_delta=on_delta,
                ),
                raise_if_cancelled=raise_if_agent_cancelled,
            )
            return
        add_and_stream_agent_message(
            repo,
            task_id,
            stage="chat",
            model_profile=model_profile,
            producer=lambda on_delta: deps.compose_agent_start_message(
                task=task,
                model_profile=model_profile,
                on_delta=on_delta,
            ),
            raise_if_cancelled=raise_if_agent_cancelled,
        )
        return
    if auto_accept:
        return
    finalize_agent_opening_message(
        repo,
        task_id=task_id,
        message_id=opening_message_id,
        model_profile=model_profile,
        content=agent_stage_opening_text(
            stage,
            validation_workflow_version=task.validation_workflow_version,
        ),
    )


def add_agent_auto_stage_start_message(
    repo: TaskRepository,
    *,
    task_id: str,
    stage: str,
    model_profile: dict,
    validation_workflow_version: int | None = None,
) -> None:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=(
            "接下来开始执行"
            f"{agent_stage_label(stage, validation_workflow_version=validation_workflow_version)}。"
        ),
        metadata={
            **model_metadata(model_profile),
            "auto_accept": True,
            "auto_stage_start": stage,
            "streaming": False,
        },
    )


def finalize_agent_opening_message(
    repo: TaskRepository,
    *,
    task_id: str,
    message_id: str | None,
    model_profile: dict,
    content: str,
) -> None:
    metadata = {**model_metadata(model_profile), "streaming": False}
    if message_id:
        repo.update_agent_message(message_id, content=content, metadata=metadata)
        return
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=content,
        metadata=metadata,
    )


def run_agent_scan_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
    deps: ValidationStageDependencies,
) -> bool:
    task = repo.get_task(task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="scan",
        content=(
            "正在调用材料识别工具 scan_materials：读取材料目录，识别 Notebook、样本数据、"
            "PMML 模型和数据字典，并检查 Notebook RMC 契约。"
        ),
        metadata={
            **model_metadata(model_profile),
            "tool_call": {
                "name": "scan_materials",
                "stage": "scan",
            },
        },
    )
    raise_if_agent_cancelled(task_id)
    scan_payload = deps.perform_scan_task(repo, task, settings)
    raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="材料完备性",
            error=task.status_message,
            model_profile=model_profile,
            deps=deps,
            evidence={"scan": scan_payload},
        )
        return False
    add_and_stream_agent_message(
        repo,
        task_id,
        stage="scan",
        model_profile=model_profile,
        producer=lambda on_delta: deps.summarize_stage(
            task=task,
            stage="scan",
            evidence=scan_payload,
            model_profile=model_profile,
            fallback="材料扫描完成，平台已识别必需验证材料。",
            on_delta=on_delta,
        ),
        raise_if_cancelled=raise_if_agent_cancelled,
    )
    raise_if_agent_cancelled(task_id)
    contract_payload = _pending_validation_contract_payload(scan_payload)
    if contract_payload is not None:
        add_agent_input_confirmation_prompt(
            repo,
            task_id=task_id,
            model_profile=model_profile,
            contract_payload=contract_payload,
        )
        return True
    if not auto_accept:
        add_agent_continue_prompt(
            repo,
            task_id,
            model_profile,
            next_stage="reproducibility",
            validation_workflow_version=task.validation_workflow_version,
        )
    return True


def _pending_validation_contract_payload(scan_payload: object) -> dict | None:
    if not isinstance(scan_payload, dict):
        return None
    payload = scan_payload.get("validation_input_contract")
    if not isinstance(payload, dict) or payload.get("status") != "pending_confirmation":
        return None
    return payload


def add_agent_input_confirmation_prompt(
    repo: TaskRepository,
    *,
    task_id: str,
    model_profile: dict,
    contract_payload: dict,
) -> None:
    status = str(contract_payload.get("status") or "missing")
    revision = contract_payload.get("revision")
    contract = contract_payload.get("contract")
    candidates = contract.get("candidates") if isinstance(contract, dict) else None
    candidate_rows: list[list[str]] = []
    omitted_candidates = 0
    if isinstance(candidates, dict):
        candidate_fields = sorted(candidates.items())
        for field_index, (field_name, values) in enumerate(candidate_fields):
            if not isinstance(values, list):
                continue
            visible = values[:MAX_INPUT_CONFIRMATION_CANDIDATES_PER_FIELD]
            omitted_candidates += len(values) - len(visible)
            for candidate_index, candidate in enumerate(visible):
                if len(candidate_rows) >= MAX_INPUT_CONFIRMATION_CANDIDATE_LINES:
                    omitted_candidates += len(visible) - candidate_index
                    break
                if not isinstance(candidate, dict) or "value" not in candidate:
                    continue
                rendered = json.dumps(
                    _candidate_preview_value(candidate["value"]),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if len(rendered) > MAX_INPUT_CONFIRMATION_VALUE_CHARS:
                    rendered = (
                        rendered[: MAX_INPUT_CONFIRMATION_VALUE_CHARS - 1] + "…"
                    )
                candidate_rows.append([str(field_name), rendered])
            if len(candidate_rows) >= MAX_INPUT_CONFIRMATION_CANDIDATE_LINES:
                omitted_candidates += sum(
                    len(value)
                    for _name, value in candidate_fields[field_index + 1 :]
                    if isinstance(value, list)
                )
                break
    if omitted_candidates:
        candidate_rows.append(["…", f"其余候选已省略（{omitted_candidates} 项）"])
    candidate_tables = (
        [
            {
                "title": "验证字段候选",
                "columns": ["验证字段", "候选值"],
                "rows": candidate_rows,
            }
        ]
        if candidate_rows
        else []
    )
    candidate_intro = (
        "候选字段已整理为下表。"
        if candidate_rows
        else "当前没有可自动确认的候选字段。"
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="input_confirmation",
        content=(
            f"验证输入契约状态为 {status}"
            f"（revision {revision}）。"
            f"{candidate_intro}"
            "请先在验证字段确认界面逐项确认验证字段并提交；契约状态变为 ready 后，"
            "再回复“继续”。在确认完成前，平台不会执行 PMML 打分测试。"
        ),
        metadata={
            **model_metadata(model_profile),
            "tables": candidate_tables,
            "awaiting_validation_input_confirmation": True,
            "validation_input_contract_ref": {
                "task_id": task_id,
                "revision": revision,
                "status": status,
                "needs_confirmation": bool(
                    contract_payload.get("needs_confirmation", True)
                ),
            },
        },
    )


def _candidate_preview_value(value: object, *, depth: int = 0) -> object:
    if isinstance(value, str):
        if len(value) <= MAX_INPUT_CONFIRMATION_VALUE_CHARS:
            return value
        return value[: MAX_INPUT_CONFIRMATION_VALUE_CHARS - 1] + "…"
    if depth >= 3:
        return "…"
    if isinstance(value, list):
        preview = [
            _candidate_preview_value(item, depth=depth + 1) for item in value[:5]
        ]
        if len(value) > 5:
            preview.append(f"… {len(value) - 5} more")
        return preview
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        preview = {
            str(key): _candidate_preview_value(item, depth=depth + 1)
            for key, item in items[:5]
        }
        if len(items) > 5:
            preview["…"] = f"{len(items) - 5} more"
        return preview
    return value


def run_agent_reproducibility_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
    deps: ValidationStageDependencies,
) -> bool:
    task = repo.get_task(task_id)
    _require_ready_contract_for_v2(settings, task)
    uses_pmml_scoring = (
        task.task_type == TASK_TYPE_VALIDATION
        and task.validation_workflow_version == 2
    )
    repo.update_status(
        task_id,
        TaskStatus.RUNNING,
        "agent PMML scoring queued" if uses_pmml_scoring else "agent notebook queued",
        expected={TaskStatus.SCANNED, TaskStatus.FAILED},
    )
    raise_if_agent_cancelled(task_id)
    stage_runner = (
        deps.run_pmml_scoring_stage if uses_pmml_scoring else deps.run_notebook_stage
    )
    stage_kwargs = {
        "task_id": task_id,
        "settings": deps.agent_pipeline_settings(settings, task),
        "stage_claimed": True,
    }
    if uses_pmml_scoring:
        stage_kwargs["cancellation_check"] = lambda: raise_if_agent_cancelled(
            task_id
        )
    stage_runner(**stage_kwargs)
    raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        evidence = deps.agent_evidence_from_settings(settings, task_id)
        add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="PMML打分测试" if uses_pmml_scoring else "模型可复现性",
            error=task.status_message,
            model_profile=model_profile,
            deps=deps,
            evidence=evidence,
        )
        return False
    evidence = deps.agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = agent_memory_context_from_store(
        memory_store,
        task,
        stage="reproducibility",
        evidence=evidence,
    )
    message = add_and_stream_agent_message(
        repo,
        task_id,
        stage="reproducibility",
        model_profile=model_profile,
        producer=lambda on_delta: deps.summarize_stage(
            task=task,
            stage="reproducibility",
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            fallback=(
                "PMML打分测试已完成，后续效果、稳定性和压力测试将使用本次评分结果。"
                if uses_pmml_scoring
                else "分数一致性阶段已完成，请查看可复现性证据明细。"
            ),
            on_delta=on_delta,
        ),
        raise_if_cancelled=raise_if_agent_cancelled,
    )
    audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    raise_if_agent_cancelled(task_id)
    if not auto_accept:
        add_agent_continue_prompt(repo, task_id, model_profile, next_stage="metrics")
    return True


def run_agent_metrics_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
    deps: ValidationStageDependencies,
) -> bool:
    task = repo.get_task(task_id)
    _require_ready_contract_for_v2(settings, task)
    if task.status == TaskStatus.FAILED and deps.is_metrics_failure(task):
        expected_statuses = {
            TaskStatus.FAILED,
            TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
        }
    else:
        expected_statuses = {
            TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
        }
    repo.update_status(
        task_id,
        TaskStatus.COMPUTING_METRICS,
        "agent metrics queued",
        expected=expected_statuses,
    )
    raise_if_agent_cancelled(task_id)
    metrics_kwargs = {
        "task_id": task_id,
        "settings": deps.agent_pipeline_settings(settings, task),
        "stage_claimed": True,
    }
    if task.validation_workflow_version == 2:
        metrics_kwargs["cancellation_check"] = lambda: raise_if_agent_cancelled(
            task_id
        )
    deps.run_metrics_stage(**metrics_kwargs)
    raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        evidence = deps.agent_evidence_from_settings(settings, task_id)
        add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="效果和稳定性",
            error=task.status_message,
            model_profile=model_profile,
            deps=deps,
            evidence=evidence,
        )
        return False
    evidence = deps.agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = agent_memory_context_from_store(
        memory_store,
        task,
        stage="metrics",
        evidence=evidence,
    )
    message = add_and_stream_agent_message(
        repo,
        task_id,
        stage="metrics",
        model_profile=model_profile,
        producer=lambda on_delta: deps.summarize_stage(
            task=task,
            stage="metrics",
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            fallback=(
                "效果、稳定性和模型压力测试结果已生成，请结合 OOT KS、PSI 和压力测试明细复核。"
                if task.validation_workflow_version == 2
                else "效果、稳定性和 Excel 指标产物已生成，请结合 OOT KS、PSI 和压力测试明细复核。"
            ),
            on_delta=on_delta,
        ),
        raise_if_cancelled=raise_if_agent_cancelled,
    )
    audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    raise_if_agent_cancelled(task_id)
    if not auto_accept:
        add_agent_continue_prompt(
            repo, task_id, model_profile, next_stage="word_conclusion_draft"
        )
    return True


def run_agent_word_conclusion_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    draft_message_id: str | None = None,
    *,
    auto_accept: bool = False,
    rewrite_instruction: str | None = None,
    deps: ValidationStageDependencies,
) -> bool:
    task = repo.get_task(task_id)
    _require_ready_contract_for_v2(settings, task)
    evidence = deps.agent_evidence_from_settings(settings, task_id)
    evidence = _word_conclusion_evidence_with_stage_summaries(repo, task_id, evidence)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = agent_memory_context_from_store(
        memory_store,
        task,
        stage="word_conclusion_draft",
        evidence=evidence,
        user_message=rewrite_instruction or "",
    )
    draft_result: dict[str, object] = {}

    def produce_draft(_on_delta):
        _, report_revision = repo.get_report_values(task_id)
        values, metadata = deps.generate_word_conclusions(
            task=task,
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            user_instruction=rewrite_instruction,
        )
        draft_result["values"] = values
        draft_result["metadata"] = metadata
        draft_result["report_revision"] = report_revision
        return (
            format_conclusion_values(values),
            {**metadata, "draft_values": values, "report_revision": report_revision},
        )

    if draft_message_id:
        message = stream_agent_message(
            repo,
            draft_message_id,
            task_id=task_id,
            model_profile=model_profile,
            producer=produce_draft,
            raise_if_cancelled=raise_if_agent_cancelled,
        )
    else:
        message = add_and_stream_agent_message(
            repo,
            task_id,
            stage="word_conclusion_draft",
            model_profile=model_profile,
            producer=produce_draft,
            raise_if_cancelled=raise_if_agent_cancelled,
        )
    audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    values = draft_result.get("values")
    if not isinstance(values, dict) or not agent_conclusions_confirmed(values):
        add_agent_word_draft_failure_message(
            repo,
            task_id=task_id,
            model_profile=model_profile,
            metadata=draft_result.get("metadata"),
        )
        return False
    if auto_accept:
        return auto_confirm_agent_report_conclusions(
            repo=repo,
            settings=settings,
            task_id=task_id,
            model_profile=model_profile,
            values=draft_result.get("values"),
            expected_revision=draft_result.get("report_revision"),
            deps=deps,
        )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
        metadata={**model_metadata(model_profile), "awaiting_confirmation": True},
    )
    return True


def _word_conclusion_evidence_with_stage_summaries(
    repo: TaskRepository,
    task_id: str,
    evidence: object,
) -> dict:
    payload = dict(evidence) if isinstance(evidence, dict) else {}
    summaries = _visible_stage_summaries_for_word_conclusion(
        repo.list_agent_messages(task_id)
    )
    if summaries:
        payload["visible_stage_summaries"] = summaries
    return payload


def _visible_stage_summaries_for_word_conclusion(messages: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    excluded_stages = {
        "chat",
        "word_conclusion_draft",
        "word_conclusion_confirmed",
        "word_report_ready",
    }
    for message in messages[-16:]:
        if message.get("role") != "assistant":
            continue
        stage = str(message.get("stage") or "")
        content = str(message.get("content") or "").strip()
        if not stage or stage in excluded_stages or not content:
            continue
        summaries.append({"stage": stage, "content": content})
    return summaries


def add_agent_word_draft_failure_message(
    repo: TaskRepository,
    *,
    task_id: str,
    model_profile: dict,
    metadata: object,
) -> None:
    llm_error = ""
    if isinstance(metadata, dict):
        llm_error = str(metadata.get("llm_error") or "").strip()
    detail = f"直接原因：{llm_error}" if llm_error else "直接原因：大模型未返回完整的三段 JSON 草稿。"
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=(
            "报告结论草稿生成失败，未生成可确认的三段 Word 结论，也不会写入 Word。"
            f"{detail} 请缩小输入、换用更大上下文窗口的模型，或重新生成报告结论草稿。"
        ),
        metadata={
            **model_metadata(model_profile),
            "word_draft_failed": True,
            **({"llm_error": llm_error} if llm_error else {}),
        },
    )


def auto_confirm_agent_report_conclusions(
    *,
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    values: object,
    expected_revision: object,
    deps: ValidationStageDependencies,
) -> bool:
    if (
        not isinstance(values, dict)
        or not agent_conclusions_confirmed(values)
        or not isinstance(expected_revision, int)
        or isinstance(expected_revision, bool)
    ):
        raise RuntimeError("agent report draft is incomplete; cannot auto-confirm report")
    conclusion_values = {
        key: str(values.get(key) or "").strip()
        for key in REQUIRED_AGENT_REPORT_KEYS
    }
    revision = repo.update_agent_report_conclusions_with_audit(
        task_id,
        conclusion_values,
        expected_revision=expected_revision,
        audit={
            "kind": "report.agent_conclusions.confirm",
            "target_ref": task_id,
            "outcome": "succeeded",
            "detail": {
                "keys": sorted(conclusion_values),
                "expected_revision": expected_revision,
                "auto_accept": True,
            },
        },
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_confirmed",
        content="三段报告结论已自动确认，正在生成最终 Word 报告。",
        metadata={
            **model_metadata(model_profile),
            "revision": revision,
            "confirmed_keys": sorted(REQUIRED_AGENT_REPORT_KEYS),
            "auto_accept": True,
        },
    )
    raise_if_agent_cancelled(task_id)
    deps.run_report_stage(
        task_id=task_id,
        settings=deps.agent_pipeline_settings(settings, repo.get_task(task_id)),
    )
    raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="报告生成",
            error=task.status_message,
            model_profile=model_profile,
            deps=deps,
        )
        return False
    deps.add_agent_report_ready_message(repo, task_id)
    return True


def add_agent_failure_summary(
    repo: TaskRepository,
    *,
    task_id: str,
    task: TaskRecord,
    stage_label: str,
    error: str,
    model_profile: dict,
    deps: ValidationStageDependencies,
    evidence: dict | None = None,
) -> None:
    add_and_stream_agent_message(
        repo,
        task_id,
        stage="failure",
        model_profile=model_profile,
        producer=lambda on_delta: deps.failure_summary(
            task=task,
            stage=stage_label,
            error=error,
            evidence=evidence,
            model_profile=model_profile,
            on_delta=on_delta,
        ),
        raise_if_cancelled=raise_if_agent_cancelled,
    )


def add_agent_continue_prompt(
    repo: TaskRepository,
    task_id: str,
    model_profile: dict,
    *,
    next_stage: str,
    validation_workflow_version: int | None = None,
) -> None:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=(
            "是否继续执行【"
            f"{agent_stage_label(next_stage, validation_workflow_version=validation_workflow_version)}"
            "】？"
            "你可以先继续提问；需要继续时，请明确回复“继续”。"
        ),
        metadata={**model_metadata(model_profile), "awaiting_next_stage": next_stage},
    )
