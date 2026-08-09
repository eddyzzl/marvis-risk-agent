from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import zipfile

from marvis.agent.service import (
    agent_conclusions_confirmed,
    fallback_word_conclusions,
    generate_word_conclusions,
)
from marvis.agent.validation_evidence import agent_evidence_from_settings
from marvis.api_scan_helpers import perform_scan_task
from marvis.api_stage_helpers import pipeline_settings_from_settings, run_stage_job
from marvis.api_task_helpers import allowed_material_roots
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskStatus
from marvis.job_heartbeat import heartbeat_job
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.pipeline import (
    run_metrics_stage,
    run_pmml_scoring_stage,
    run_report_stage,
)
from marvis.redaction import redact_text
from marvis.repositories.validation_batches import (
    ValidationBatchItemRecord,
    ValidationBatchRecord,
    ValidationBatchRepository,
)
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.safe_paths import prepare_task_output_dir, resolve_fixed_task_output
from marvis.validation.results import validation_results_from_dict
from marvis.validation.stress_risk import (
    stress_ks_risk,
    stress_psi_risk,
    worst_stress_risk,
)


BatchSummaryWriter = Callable[..., Any]
BatchRunner = Callable[..., ValidationBatchRecord]


@dataclass(frozen=True)
class ValidationBatchSummaryEntry:
    ordinal: int
    item_id: str
    child_task_id: str
    model_name: str
    model_version: str
    oot_ks: float | None
    oot_psi: float | None
    pmml_status: str
    stress_risk: str | None
    report_complete: bool
    outcome: str
    manual_review_url: str
    error_message: str


class ValidationBatchItemError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def run_validation_batch_job(
    *,
    job_id: str,
    settings,
    parent_task_id: str,
    model_id: str | None = None,
    effort: str | None = None,
    manual_review_base_url: str = "",
    summary_writer: BatchSummaryWriter | None = None,
    retry_failed_items: bool | None = None,
    runner: BatchRunner | None = None,
) -> None:
    """Execute one claimed parent job and always release its active-job slot."""

    task_repo = TaskRepository(settings.db_path)
    batch_repo = ValidationBatchRepository(settings.db_path)
    if not task_repo.mark_job_running(job_id):
        _recover_unstarted_batch_job(
            batch_repo=batch_repo,
            task_repo=task_repo,
            parent_task_id=parent_task_id,
        )
        return
    execute = runner or run_validation_batch
    try:
        with heartbeat_job(task_repo, job_id):
            final_batch = execute(
                settings=settings,
                parent_task_id=parent_task_id,
                model_id=model_id,
                effort=effort,
                manual_review_base_url=manual_review_base_url,
                summary_writer=summary_writer,
                retry_failed_items=retry_failed_items,
            )
        if final_batch.status == "failed":
            task_repo.finish_job(
                job_id,
                status="failed",
                error_name="ValidationBatchFailed",
                error_value=(
                    final_batch.error_message or "validation batch failed"
                ),
            )
        else:
            task_repo.finish_job(job_id, status="succeeded")
    except Exception as exc:
        detail = _bounded_public_error(exc, settings)
        try:
            batch_repo.update_batch(
                parent_task_id,
                status="failed",
                error_message=detail,
                finished=True,
            )
        except Exception:
            pass
        try:
            _finish_parent_task(
                task_repo,
                parent_task_id,
                TaskStatus.FAILED,
                "批次模型验证执行失败",
            )
            task_repo.add_agent_message(
                parent_task_id,
                role="assistant",
                stage="failure",
                content=f"批次模型验证执行失败：{detail}",
                metadata={
                    "batch_failed": True,
                    "error_code": exc.__class__.__name__,
                },
            )
        except Exception:
            pass
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=detail,
        )


def _recover_unstarted_batch_job(
    *,
    batch_repo: ValidationBatchRepository,
    task_repo: TaskRepository,
    parent_task_id: str,
) -> None:
    """Release a claimed batch when its queued background job never starts."""

    try:
        batch = batch_repo.get_batch(parent_task_id)
    except KeyError:
        return
    if batch.status != "running":
        return
    message = "批次后台任务未能启动，可修复后重试"
    batch_repo.update_batch(
        parent_task_id,
        status="partial_failure",
        error_message=message,
        finished=True,
    )
    parent = task_repo.get_task(parent_task_id)
    if parent.status is not TaskStatus.FAILED:
        _finish_parent_task(
            task_repo,
            parent_task_id,
            TaskStatus.FAILED,
            message,
        )
    task_repo.add_agent_message(
        parent_task_id,
        role="assistant",
        stage="failure",
        content=message,
        metadata={
            "batch_failed_to_start": True,
            "retryable": True,
        },
    )


def mark_validation_batch_parent_running(
    task_repo: TaskRepository,
    parent_task_id: str,
) -> None:
    """Move a parent task into the common CREATED -> SCANNED -> RUNNING flow."""

    parent = task_repo.get_task(parent_task_id)
    if parent.status is TaskStatus.CREATED:
        task_repo.update_status(
            parent_task_id,
            TaskStatus.SCANNED,
            "批次材料已登记，准备逐项验证",
            expected=TaskStatus.CREATED,
        )
        parent = task_repo.get_task(parent_task_id)
    if parent.status is TaskStatus.RUNNING:
        task_repo.update_status_message(
            parent_task_id,
            "批次模型验证正在逐项执行",
        )
        return
    task_repo.update_status(
        parent_task_id,
        TaskStatus.RUNNING,
        "批次模型验证正在逐项执行",
        expected={
            TaskStatus.SCANNED,
            TaskStatus.EXECUTED,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.FAILED,
            TaskStatus.WRITING_ARTIFACTS,
        },
    )


def run_validation_batch(
    *,
    settings,
    parent_task_id: str,
    model_id: str | None = None,
    effort: str | None = None,
    manual_review_base_url: str = "",
    summary_writer: BatchSummaryWriter | None = None,
    retry_failed_items: bool | None = None,
) -> ValidationBatchRecord:
    """Run validation children in ordinal order with per-item failure isolation."""

    batch_repo = ValidationBatchRepository(settings.db_path)
    task_repo = TaskRepository(settings.db_path)
    contract_repo = ValidationContractRepository(settings.db_path)
    mark_validation_batch_parent_running(task_repo, parent_task_id)
    initial_batch = batch_repo.get_batch(parent_task_id)
    retry_failed = (
        initial_batch.status in {"partial_failure", "failed"}
        if retry_failed_items is None
        else retry_failed_items
    )
    batch_repo.update_batch(
        parent_task_id,
        status="running",
        started=True,
        reopened=retry_failed,
    )
    model_profile = _optional_model_profile(settings, model_id, effort)

    for item in batch_repo.list_items(parent_task_id):
        if item.status in {"succeeded", "review_required"}:
            try:
                _record_completed_item(
                    batch_repo=batch_repo,
                    settings=settings,
                    item=item,
                    final_task=task_repo.get_task(item.child_task_id),
                )
            except Exception as exc:
                _record_item_failure(
                    batch_repo=batch_repo,
                    task_repo=task_repo,
                    settings=settings,
                    parent_task_id=parent_task_id,
                    item=item,
                    stage=getattr(exc, "stage", "summary"),
                    exc=exc,
                )
            continue
        if item.status == "cancelled":
            continue
        if item.status == "failed" and not retry_failed:
            continue
        stage = item.stage if item.status == "failed" else "scan"
        try:
            task = task_repo.get_task(item.child_task_id)
            prepare_task_output_dir(settings.tasks_dir, task.id)
            if item.status == "failed" and retry_failed:
                task = _prepare_failed_child_for_retry(task_repo, task, item)
            batch_repo.update_item(
                item.id,
                status="running",
                stage=stage,
                started=True,
                reopened=item.status == "failed",
                clear_results=item.status == "failed",
            )
            contract = contract_repo.get(task.id)
            if (
                task.status is TaskStatus.CREATED
                or contract is None
                or contract.status == "blocked"
            ):
                perform_scan_task(task_repo, task, settings)
                task = task_repo.get_task(task.id)
                contract = contract_repo.get(task.id)
            if (
                task.status is TaskStatus.SCANNED
                and contract is not None
                and contract.status == "pending_confirmation"
            ):
                batch_repo.update_item(
                    item.id,
                    status="awaiting_confirmation",
                    stage="input_confirmation",
                    outcome="",
                    error_code="",
                    error_message="",
                )
                continue
            if contract is None or contract.status != "ready":
                raise ValidationBatchItemError(stage, task.status_message)

            if task.status in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
                _record_completed_item(
                    batch_repo=batch_repo,
                    settings=settings,
                    item=item,
                    final_task=task,
                )
                continue
            resumable_stages = {
                TaskStatus.SCANNED: "pmml_scoring",
                TaskStatus.EXECUTED: "metrics",
                TaskStatus.WRITING_ARTIFACTS: "report_conclusion",
            }
            if task.status not in resumable_stages:
                raise ValidationBatchItemError(stage, task.status_message)

            stage = resumable_stages[task.status]
            batch_repo.update_item(item.id, status="running", stage=stage)
            job_id = contract_repo.start_ready_job(task.id, "validation_batch")
            run_stage_job(
                job_id,
                settings.db_path,
                _run_ready_item_pipeline,
                {
                    "settings": settings,
                    "task_id": task.id,
                    "model_profile": model_profile,
                    "cancellation_job_id": job_id,
                },
            )
            final_task = task_repo.get_task(task.id)
            if final_task.status not in {
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            }:
                raise ValidationBatchItemError(
                    "report",
                    final_task.status_message,
                )
            _record_completed_item(
                batch_repo=batch_repo,
                settings=settings,
                item=item,
                final_task=final_task,
            )
        except Exception as exc:
            failure_stage = getattr(exc, "stage", stage)
            _record_item_failure(
                batch_repo=batch_repo,
                task_repo=task_repo,
                settings=settings,
                parent_task_id=parent_task_id,
                item=item,
                stage=failure_stage,
                exc=exc,
            )
            continue

    final_items = batch_repo.list_items(parent_task_id)
    if any(item.status == "awaiting_confirmation" for item in final_items):
        _add_parent_waiting_message(task_repo, parent_task_id, final_items)
        waiting_batch = batch_repo.update_batch(
            parent_task_id,
            status="awaiting_confirmation",
        )
        task_repo.update_status(
            parent_task_id,
            TaskStatus.SCANNED,
            "批次等待逐项确认输入合同",
            expected=TaskStatus.RUNNING,
        )
        return waiting_batch

    return _finalize_batch(
        settings=settings,
        parent_task_id=parent_task_id,
        batch_repo=batch_repo,
        task_repo=task_repo,
        items=final_items,
        manual_review_base_url=manual_review_base_url,
        summary_writer=summary_writer,
    )


def _prepare_failed_child_for_retry(
    task_repo: TaskRepository,
    task,
    item: ValidationBatchItemRecord,
):
    """Rewind a failed child only as far as its recorded failed stage."""

    if task.status is not TaskStatus.FAILED:
        return task
    target_status = {
        "pmml_scoring": TaskStatus.SCANNED,
        "metrics": TaskStatus.EXECUTED,
        "report_conclusion": TaskStatus.WRITING_ARTIFACTS,
        "report": TaskStatus.WRITING_ARTIFACTS,
    }.get(item.stage, TaskStatus.CREATED)
    task_repo.reset_status_for_agent_rerun(
        task.id,
        target_status,
        f"批次修复后重试：{item.stage or 'scan'}",
        clear_agent_report_conclusions=True,
    )
    return task_repo.get_task(task.id)


def _run_ready_item_pipeline(
    *,
    settings,
    task_id: str,
    model_profile: dict,
    cancellation_job_id: str,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    pipeline_settings = pipeline_settings_from_settings(settings, task, None)
    if task.status is TaskStatus.SCANNED:
        _run_item_stage(
            "pmml_scoring",
            run_pmml_scoring_stage,
            task_id=task_id,
            settings=pipeline_settings,
            cancellation_job_id=cancellation_job_id,
        )
        task = repo.get_task(task_id)
    if task.status is TaskStatus.EXECUTED:
        _run_item_stage(
            "metrics",
            run_metrics_stage,
            task_id=task_id,
            settings=pipeline_settings,
            cancellation_job_id=cancellation_job_id,
        )
        task = repo.get_task(task_id)
    if task.status in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        _write_report_conclusions(
            repo=repo,
            settings=settings,
            task_id=task_id,
            model_profile=model_profile,
        )
        _run_item_stage(
            "report",
            run_report_stage,
            task_id=task_id,
            settings=pipeline_settings,
            cancellation_job_id=cancellation_job_id,
        )
        task = repo.get_task(task_id)
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise ValidationBatchItemError("pipeline", task.status_message)


def _run_item_stage(stage: str, func: Callable, **kwargs) -> None:
    try:
        func(**kwargs)
    except Exception as exc:
        raise ValidationBatchItemError(stage, str(exc)) from exc


def _write_report_conclusions(
    *,
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
) -> None:
    task = repo.get_task(task_id)
    evidence = agent_evidence_from_settings(settings, task_id)
    generated: dict[str, str] = {}
    if model_profile:
        try:
            generated, _metadata = generate_word_conclusions(
                task=task,
                evidence=evidence,
                model_profile=model_profile,
            )
        except Exception:
            generated = {}
    values, narrative_source = compose_batch_report_conclusions(
        generated=generated,
        fallback=fallback_word_conclusions(task=task, evidence=evidence),
    )
    if not agent_conclusions_confirmed(values):
        raise ValidationBatchItemError(
            "report_conclusion",
            "report conclusions are incomplete",
        )
    _current, revision = repo.get_report_values(task_id)
    repo.update_agent_report_conclusions_with_audit(
        task_id,
        values,
        expected_revision=revision,
        audit={
            "kind": "report.agent_conclusions.generated",
            "target_ref": task_id,
            "outcome": "succeeded",
            "detail": {
                "keys": sorted(values),
                "expected_revision": revision,
                "source": narrative_source,
                "batch": True,
            },
        },
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_generated",
        content=(
            "报告结论已自动生成，正在直接生成最终 Word 报告。"
            if narrative_source.startswith("agent_generated")
            else "平台已根据确定性指标生成保守报告结论，正在直接生成最终 Word 报告。"
        ),
        metadata={"narrative_source": narrative_source, "batch": True},
    )


def compose_batch_report_conclusions(
    *,
    generated: dict[str, str],
    fallback: dict[str, str],
) -> tuple[dict[str, str], str]:
    """Keep Agent pressure prose while reserving the final verdict for evidence."""

    if not agent_conclusions_confirmed(generated):
        return dict(fallback), "deterministic_fallback"
    values = dict(generated)
    values["TEXT:final_validation_conclusion"] = fallback.get(
        "TEXT:final_validation_conclusion",
        "",
    )
    return values, "agent_generated_with_deterministic_verdict"


def _record_completed_item(
    *,
    batch_repo: ValidationBatchRepository,
    settings,
    item: ValidationBatchItemRecord,
    final_task,
) -> None:
    results_path = resolve_fixed_task_output(
        settings.tasks_dir,
        item.child_task_id,
        "validation_results.json",
    )
    if results_path is None:
        raise ValidationBatchItemError(
            "summary",
            "validation results are unavailable or outside the task output directory",
        )
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        results = validation_results_from_dict(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationBatchItemError(
            "summary",
            f"validation results are unavailable: {exc}",
        ) from exc

    oot = next(
        (row for row in results.effectiveness.overall if row.split == "oot"),
        None,
    )
    oot_ks = None if oot is None else float(oot.ks)
    oot_psi = None if oot is None else float(oot.psi_vs_train)
    pmml_status = results.pmml_scoring.status if results.pmml_scoring else ""
    stress_risk = validation_batch_stress_risk(results)
    report_complete = _individual_reports_complete(
        settings.tasks_dir,
        item.child_task_id,
    )
    outcome, reasons = validation_batch_item_outcome(
        oot_ks=oot_ks,
        pmml_status=pmml_status,
        oot_psi=oot_psi,
        stress_risk=stress_risk,
        report_complete=report_complete,
        task_status=final_task.status.value,
    )
    batch_repo.update_item(
        item.id,
        status="succeeded" if outcome == "pass" else "review_required",
        stage="completed",
        oot_ks=oot_ks,
        oot_psi=oot_psi,
        pmml_status=pmml_status,
        stress_risk=stress_risk or "unknown",
        report_complete=report_complete,
        outcome=outcome,
        error_code="",
        error_message="；".join(reasons),
        finished=True,
    )


def validation_batch_item_outcome(
    *,
    oot_ks: float | None,
    pmml_status: str,
    oot_psi: float | None,
    stress_risk: str | None,
    report_complete: bool,
    task_status: str,
) -> tuple[str, tuple[str, ...]]:
    """Return the deterministic batch verdict; OOT KS is display-only evidence."""

    _ = oot_ks
    reasons: list[str] = []
    stability_risk = stress_psi_risk(oot_psi)
    if pmml_status != "pass":
        reasons.append("PMML 打分测试未通过")
    if oot_psi is None:
        reasons.append("缺少 OOT 稳定性 PSI")
    elif stability_risk is None:
        reasons.append("OOT 稳定性 PSI 无法判定")
    elif stability_risk in {"medium", "high"}:
        reasons.append("OOT PSI 达到人工复核阈值")
    if stress_risk != "low":
        reasons.append(
            "压力测试证据不完整或无法判定"
            if stress_risk not in {"medium", "high"}
            else "压力测试达到人工复核阈值"
        )
    if not report_complete:
        reasons.append("报告产物不完整")
    if task_status == TaskStatus.REVIEW_REQUIRED.value:
        reasons.append("单模型任务要求人工复核")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return ("pass", ()) if not unique_reasons else ("manual_review", unique_reasons)


def validation_batch_stress_risk(results) -> str | None:
    """Classify stress risk only when every required category is complete."""

    stress_test = results.stress_test
    categories = list(stress_test.per_category)
    if (
        stress_test.status != "completed"
        or not categories
        or getattr(stress_test, "unclassified_features", [])
        or int(getattr(stress_test, "category_source_counts", {}).get(
            "unresolved",
            0,
        ))
        != 0
    ):
        return None
    baseline_ks = float(stress_test.baseline.ks)
    risks: list[str] = []
    has_unclassified_metric = False
    for category in categories:
        if (
            category.status != "completed"
            or getattr(category, "error", None) is not None
            or category.ks_after is None
        ):
            return None
        ks_risk = stress_ks_risk(baseline_ks, float(category.ks_after))
        psi_risk = stress_psi_risk(category.psi_vs_baseline)
        category_risk = worst_stress_risk(ks_risk, psi_risk)
        if category_risk is None:
            return None
        if ks_risk is None or psi_risk is None:
            has_unclassified_metric = True
        risks.append(category_risk)
    overall_risk = worst_stress_risk(*risks)
    if has_unclassified_metric and overall_risk == "low":
        return None
    return overall_risk


def _individual_reports_complete(tasks_dir: Path, task_id: str) -> bool:
    paths = [
        resolve_fixed_task_output(tasks_dir, task_id, filename)
        for filename in ("validation_report.docx", "validation.xlsx")
    ]
    return all(
        path is not None
        and path.stat().st_size > 0
        and zipfile.is_zipfile(path)
        for path in paths
    )


def _record_item_failure(
    *,
    batch_repo: ValidationBatchRepository,
    task_repo: TaskRepository,
    settings,
    parent_task_id: str,
    item: ValidationBatchItemRecord,
    stage: str,
    exc: Exception,
) -> None:
    detail = _bounded_public_error(exc, settings)
    batch_repo.update_item(
        item.id,
        status="failed",
        stage=stage,
        outcome="failed",
        error_code=exc.__class__.__name__,
        error_message=detail,
        finished=True,
    )
    task_repo.add_agent_message(
        parent_task_id,
        role="assistant",
        stage="failure",
        content=(
            f"模型 {item.model_name} 在{stage}阶段失败：{detail}。"
            "该项已隔离，平台会继续处理后续模型。"
        ),
        metadata={
            "batch_item_failed": True,
            "item_id": item.id,
            "child_task_id": item.child_task_id,
            "failed_stage": stage,
            "error_code": exc.__class__.__name__,
        },
    )


def _add_parent_waiting_message(
    repo: TaskRepository,
    parent_task_id: str,
    items: list[ValidationBatchItemRecord],
) -> None:
    waiting_count = sum(item.status == "awaiting_confirmation" for item in items)
    repo.add_agent_message(
        parent_task_id,
        role="assistant",
        stage="input_confirmation",
        content=(
            f"批次中有 {waiting_count} 个模型等待确认输入合同。"
            "平台不会自动选择字段口径；请逐项确认后再次启动批次。"
        ),
        metadata={
            "awaiting_confirmation": True,
            "waiting_item_count": waiting_count,
        },
    )


def _finalize_batch(
    *,
    settings,
    parent_task_id: str,
    batch_repo: ValidationBatchRepository,
    task_repo: TaskRepository,
    items: list[ValidationBatchItemRecord],
    manual_review_base_url: str,
    summary_writer: BatchSummaryWriter | None,
) -> ValidationBatchRecord:
    batch = batch_repo.get_batch(parent_task_id)
    parent = task_repo.get_task(parent_task_id)
    output_path = (
        settings.tasks_dir
        / parent_task_id
        / "outputs"
        / "validation_batch_summary.xlsx"
    )
    rows = [
        ValidationBatchSummaryEntry(
            ordinal=item.ordinal,
            item_id=item.id,
            child_task_id=item.child_task_id,
            model_name=item.model_name,
            model_version=item.model_version,
            oot_ks=item.oot_ks,
            oot_psi=item.oot_psi,
            pmml_status=item.pmml_status,
            stress_risk=item.stress_risk or None,
            report_complete=item.report_complete,
            outcome=item.outcome or ("failed" if item.status == "failed" else "manual_review"),
            manual_review_url=_manual_review_url(
                manual_review_base_url,
                parent_task_id,
                item.child_task_id,
            ),
            error_message=item.error_message,
        )
        for item in sorted(items, key=lambda value: value.ordinal)
    ]
    writer = summary_writer or _late_bound_summary_writer
    try:
        output_path = (
            prepare_task_output_dir(settings.tasks_dir, parent_task_id)
            / "validation_batch_summary.xlsx"
        )
        written = writer(
            batch_name=parent.model_name,
            created_at=batch.created_at,
            rows=rows,
            output_path=output_path,
        )
    except Exception as exc:
        detail = _bounded_public_error(exc, settings)
        task_repo.add_agent_message(
            parent_task_id,
            role="assistant",
            stage="failure",
            content=f"批次汇总报告生成失败：{detail}",
            metadata={
                "batch_summary_failed": True,
                "error_code": exc.__class__.__name__,
            },
        )
        failed_batch = batch_repo.update_batch(
            parent_task_id,
            status="failed",
            error_message=detail,
            finished=True,
        )
        _finish_parent_task(
            task_repo,
            parent_task_id,
            TaskStatus.FAILED,
            "批次汇总报告生成失败",
        )
        return failed_batch

    failed_count = sum(item.status == "failed" for item in items)
    manual_review_count = sum(item.status == "review_required" for item in items)
    status = (
        "failed"
        if failed_count == len(items)
        else "partial_failure"
        if failed_count
        else "completed"
    )
    summary_path = str(Path(written) if written is not None else output_path)
    final = batch_repo.update_batch(
        parent_task_id,
        status=status,
        summary_path=summary_path,
        error_message="",
        finished=True,
    )
    parent_status = (
        TaskStatus.FAILED
        if status == "failed"
        else TaskStatus.REVIEW_REQUIRED
        if failed_count or manual_review_count
        else TaskStatus.SUCCEEDED
    )
    _finish_parent_task(
        task_repo,
        parent_task_id,
        parent_status,
        (
            "批次模型验证失败"
            if parent_status is TaskStatus.FAILED
            else "批次模型验证已完成，存在需人工复核的模型"
            if parent_status is TaskStatus.REVIEW_REQUIRED
            else "批次模型验证已完成"
        ),
    )
    task_repo.add_agent_message(
        parent_task_id,
        role="assistant",
        stage="batch_completed",
        content=(
            "批次模型验证已完成，汇总报告已生成。"
            if not failed_count and not manual_review_count
            else (
                "批次模型验证已完成；"
                f"{manual_review_count} 个模型需人工复核，"
                f"{failed_count} 个模型失败。汇总报告已保留。"
            )
        ),
        metadata={
            "batch_status": status,
            "summary_download_url": (
                f"/api/validation-batches/{parent_task_id}/summary/download"
            ),
            "failed_item_count": failed_count,
            "manual_review_count": manual_review_count,
        },
    )
    return final


def _finish_parent_task(
    task_repo: TaskRepository,
    parent_task_id: str,
    terminal_status: TaskStatus,
    message: str,
) -> None:
    parent = task_repo.get_task(parent_task_id)
    if parent.status is terminal_status:
        task_repo.update_status_message(parent_task_id, message)
        return
    if parent.status is not TaskStatus.RUNNING:
        mark_validation_batch_parent_running(task_repo, parent_task_id)
    task_repo.update_status(
        parent_task_id,
        TaskStatus.EXECUTED,
        "批次模型验证执行完成，正在收口结果",
        expected=TaskStatus.RUNNING,
    )
    task_repo.update_status(
        parent_task_id,
        terminal_status,
        message,
        expected=TaskStatus.EXECUTED,
    )


def _late_bound_summary_writer(**kwargs):
    from marvis.output.validation_batch_excel import write_validation_batch_excel

    return write_validation_batch_excel(**kwargs)


def _manual_review_url(base_url: str, parent_task_id: str, child_task_id: str) -> str:
    relative = f"/?task={parent_task_id}&item={child_task_id}"
    return f"{base_url.rstrip('/')}{relative}" if base_url else relative


def _optional_model_profile(settings, model_id: str | None, effort: str | None) -> dict:
    try:
        profile = resolve_llm_model(settings.workspace, model_id, role="narrative")
    except LLMSettingsError:
        return {}
    if effort is not None:
        normalized = str(effort).strip().lower()
        profile["reasoning_effort"] = (
            normalized
            if normalized in {"none", "low", "medium", "high"}
            else "high"
        )
    return profile


def _bounded_public_error(exc: Exception, settings) -> str:
    detail = redact_text(str(exc).strip() or exc.__class__.__name__)
    roots = sorted(
        (str(root) for root in allowed_material_roots(settings) if str(root)),
        key=len,
        reverse=True,
    )
    for root in roots:
        detail = detail.replace(root, "<material-root>")
    return detail[:500]


__all__ = [
    "BatchRunner",
    "BatchSummaryWriter",
    "ValidationBatchItemError",
    "ValidationBatchSummaryEntry",
    "compose_batch_report_conclusions",
    "mark_validation_batch_parent_running",
    "run_validation_batch",
    "run_validation_batch_job",
    "validation_batch_item_outcome",
    "validation_batch_stress_risk",
]
