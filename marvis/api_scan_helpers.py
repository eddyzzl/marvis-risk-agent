from __future__ import annotations

import json
from pathlib import Path

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db import TaskRepository
from marvis.domain import FileArtifact, FileRole, TaskRecord, TaskStatus
from marvis.files import (
    EXCEL_SUFFIXES,
    SAMPLE_SUFFIXES,
    scan_source_dir,
    write_json_atomic,
)
from marvis.notebook_contract import (
    NotebookContractError,
    inspect_notebook_contract,
    precheck_notebook_contract,
)
from marvis.notebook_steps import notebook_step_preview
from marvis.notebooks import close_live_notebook_session
from marvis.pipeline import SCAN_STAGE_FAILURE_PREFIX
from marvis.safe_paths import assert_within


REQUIRED_SCAN_MATERIALS = (
    (FileRole.NOTEBOOK, "Notebook 文件", "notebook_path"),
    (FileRole.SAMPLE, "样本数据", "sample_path"),
    (FileRole.MODEL_PMML, "PMML 模型", "pmml_path"),
    (FileRole.DATA_DICTIONARY, "数据字典", "dictionary_path"),
)
MATERIAL_SELECTION_SUFFIXES = {
    FileRole.NOTEBOOK: {".ipynb"},
    FileRole.SAMPLE: SAMPLE_SUFFIXES | EXCEL_SUFFIXES,
    FileRole.MODEL_PMML: {".pmml"},
    FileRole.DATA_DICTIONARY: SAMPLE_SUFFIXES | EXCEL_SUFFIXES,
}

RMC_CONTRACT_NAME_LABELS = {
    "RMC_SAMPLE_DF": "RMC_SAMPLE_DF（样本 DataFrame）",
    "RMC_SCORE_FN": "RMC_SCORE_FN（模型打分函数）",
    "RMC_TARGET_COL": "RMC_TARGET_COL（目标列）",
    "RMC_ALGORITHM": "RMC_ALGORITHM（模型算法）",
}
SCAN_FAILURE_PREFIX = SCAN_STAGE_FAILURE_PREFIX


def format_notebook_contract_error(exc: NotebookContractError) -> str:
    message = str(exc)
    missing_prefix = "Notebook contract check failed before execution: missing "
    if message.startswith(missing_prefix):
        missing_names = [
            name.strip()
            for name in message.removeprefix(missing_prefix).split(",")
            if name.strip()
        ]
        missing_text = "、".join(
            RMC_CONTRACT_NAME_LABELS.get(name, name) for name in missing_names
        )
        return f"Notebook RMC 契约检查失败：缺少 {missing_text}。请在 Notebook 顶层定义后重新扫描。"
    return f"Notebook RMC 契约检查失败：{message}"


def scan_error_checks(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    return [check for check in checks if check.get("status") == "error"]


def scan_status_message(checks: list[dict[str, str]]) -> str:
    messages = [
        check.get("message", "")
        for check in scan_error_checks(checks)
        if check.get("message")
    ]
    if messages:
        return f"{SCAN_FAILURE_PREFIX}{'；'.join(messages)}"
    return "材料扫描完成。"


def is_scan_failure(task: TaskRecord) -> bool:
    return task.status == TaskStatus.FAILED and task.status_message.startswith(
        SCAN_FAILURE_PREFIX
    )


def scan_hook_payload(payload: dict) -> dict:
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    failed_codes = [
        str(check.get("code") or "")
        for check in checks
        if isinstance(check, dict) and check.get("status") != "pass"
    ]
    return {
        "task_id": str(payload.get("task_id") or ""),
        "status": str(payload.get("status") or ""),
        "status_message": str(payload.get("status_message") or ""),
        "check_count": len(checks),
        "failed_check_codes": [code for code in failed_codes if code],
    }


def perform_scan_task(repo: TaskRepository, task: TaskRecord, settings) -> dict:
    # source_dir is normalized at task-create time, so pipeline and /scan agree.
    source_dir = Path(task.source_dir).resolve()
    artifacts = scan_source_dir(source_dir)
    checks = scan_preflight_checks(task, artifacts)
    task_dir = settings.tasks_dir / task.id
    uow = ArtifactUnitOfWork()
    staged_execution = uow.stage_directory(task_dir, "execution")
    staged_execution.path.mkdir(parents=True, exist_ok=True)
    uow.remove_path(task_dir / "outputs")
    uow.remove_path(task_dir / "images")
    notebook_steps = scan_notebook_steps(
        settings,
        task,
        artifacts,
        execution_dir=staged_execution.path,
    )
    notebook_contract = scan_notebook_contract(task, artifacts)
    scan_status = TaskStatus.FAILED if scan_error_checks(checks) else TaskStatus.SCANNED
    scan_message = scan_status_message(checks)
    payload = {
        "task_id": task.id,
        "status": scan_status.value,
        "status_message": scan_message,
        "artifacts": [artifact_payload(artifact) for artifact in artifacts],
        "ambiguities": artifact_ambiguities(artifacts),
        "selected_materials": material_selection_payload(task, source_dir),
        "checks": checks,
        "notebook_steps": notebook_steps,
        "notebook_contract": notebook_contract,
    }
    (staged_execution.path / "scan_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    close_live_notebook_session(task.id)
    expected_statuses = {
        TaskStatus.CREATED,
        TaskStatus.SCANNED,
        TaskStatus.FAILED,
        TaskStatus.EXECUTED,
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.SUCCEEDED,
        TaskStatus.REVIEW_REQUIRED,
    }

    def commit_scan(conn):
        repo.update_status_on_connection(
            conn,
            task.id,
            scan_status,
            scan_message,
            expected=expected_statuses,
            begin_immediate=True,
        )
        return payload

    def commit_scan_legacy():
        repo.update_status(
            task.id,
            scan_status,
            scan_message,
            expected=expected_statuses,
        )
        return payload

    if hasattr(repo, "transaction") and hasattr(repo, "update_status_on_connection"):
        uow.finalize_with_connection(repo.transaction, commit_scan)
    else:
        uow.finalize(commit_scan_legacy)
    return payload


def artifact_payload(artifact) -> dict:
    return {
        "role": artifact.role.value,
        "path": str(artifact.path),
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "risk_notes": artifact.risk_notes,
    }


def material_candidates_payload(task: TaskRecord) -> dict:
    source_dir = Path(task.source_dir).resolve()
    artifacts = scan_source_dir(source_dir)
    candidates: dict[str, list[dict]] = {
        role.value: [] for role, _, _ in REQUIRED_SCAN_MATERIALS
    }
    seen: dict[str, set[str]] = {role.value: set() for role, _, _ in REQUIRED_SCAN_MATERIALS}
    for artifact in artifacts:
        for role in material_candidate_roles(artifact):
            role_key = role.value
            candidate = material_candidate_payload(artifact, source_dir)
            key = candidate["relative_path"]
            if key in seen[role_key]:
                continue
            seen[role_key].add(key)
            candidates[role_key].append(candidate)
    for values in candidates.values():
        values.sort(key=lambda item: item["relative_path"])
    return {
        "task_id": task.id,
        "source_dir": str(source_dir),
        "selection": material_selection_payload(task, source_dir),
        "candidates": candidates,
        "artifacts": [
            material_candidate_payload(artifact, source_dir) for artifact in artifacts
        ],
        "required_roles": [
            {"role": role.value, "label": label, "field": field}
            for role, label, field in REQUIRED_SCAN_MATERIALS
        ],
    }


def material_candidate_roles(artifact: FileArtifact) -> list[FileRole]:
    roles: list[FileRole] = []
    if artifact.role in {
        FileRole.NOTEBOOK,
        FileRole.MODEL_PMML,
        FileRole.DATA_DICTIONARY,
        FileRole.SAMPLE,
    }:
        roles.append(artifact.role)
    suffix = artifact.path.suffix.lower()
    if suffix in MATERIAL_SELECTION_SUFFIXES[FileRole.SAMPLE]:
        roles.append(FileRole.SAMPLE)
    if suffix in MATERIAL_SELECTION_SUFFIXES[FileRole.DATA_DICTIONARY]:
        roles.append(FileRole.DATA_DICTIONARY)
    result: list[FileRole] = []
    for role in roles:
        if role not in result:
            result.append(role)
    return result


def material_candidate_payload(artifact: FileArtifact, source_dir: Path) -> dict:
    try:
        relative_path = artifact.path.resolve().relative_to(source_dir).as_posix()
    except ValueError:
        relative_path = artifact.path.name
    return {
        "role": artifact.role.value,
        "path": str(artifact.path),
        "relative_path": relative_path,
        "name": artifact.path.name,
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "risk_notes": artifact.risk_notes,
    }


def material_selection_payload(task: TaskRecord, source_dir: Path) -> dict[str, str]:
    return {
        field: material_selection_value(task, source_dir, field)
        for _, _, field in REQUIRED_SCAN_MATERIALS
    }


def material_selection_value(task: TaskRecord, source_dir: Path, field: str) -> str:
    value = getattr(task, field, None)
    if not value:
        return ""
    raw_path = Path(value)
    candidate = raw_path if raw_path.is_absolute() else source_dir / raw_path
    try:
        return candidate.resolve(strict=False).relative_to(source_dir).as_posix()
    except ValueError:
        return str(value)


def validate_material_selection(task: TaskRecord, selection: dict[str, str]) -> dict[str, str]:
    source_dir = Path(task.source_dir).resolve()
    normalized: dict[str, str] = {}
    for role, label, field in REQUIRED_SCAN_MATERIALS:
        normalized[field] = normalize_selected_material_path(
            source_dir=source_dir,
            role=role,
            label=label,
            value=selection.get(field, ""),
        )
    return normalized


def normalize_selected_material_path(
    *,
    source_dir: Path,
    role: FileRole,
    label: str,
    value: str,
) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        raise ValueError(f"请选择{label}。")
    raw_path = Path(raw_value)
    candidate = raw_path if raw_path.is_absolute() else source_dir / raw_path
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label}路径不存在：{raw_value}") from exc
    try:
        resolved = assert_within(source_dir, resolved)
    except PermissionError as exc:
        raise ValueError(f"{label}必须位于材料目录内：{raw_value}") from exc
    suffix = resolved.suffix.lower()
    allowed = MATERIAL_SELECTION_SUFFIXES[role]
    if suffix not in allowed:
        allowed_text = "、".join(sorted(allowed))
        raise ValueError(
            f"{label}文件格式不支持：{suffix or '无扩展名'}，支持 {allowed_text}"
        )
    return resolved.relative_to(source_dir).as_posix()


def artifact_ambiguities(artifacts) -> list[str]:
    role_counts: dict[str, int] = {}
    for artifact in artifacts:
        role_counts[artifact.role.value] = role_counts.get(artifact.role.value, 0) + 1
    return [
        f"{role} has {count} candidates; configure explicit path before validation"
        for role, count in sorted(role_counts.items())
        if count > 1
    ]


def scan_notebook_steps(
    settings,
    task: TaskRecord,
    artifacts: list[FileArtifact],
    *,
    execution_dir: Path | None = None,
) -> list[dict]:
    notebook_path, error = resolve_scan_material(
        task=task,
        artifacts=artifacts,
        role=FileRole.NOTEBOOK,
        label="Notebook 文件",
        task_field="notebook_path",
    )
    if error or notebook_path is None:
        return []
    try:
        steps = notebook_step_preview(notebook_path)
    except Exception:
        return []
    execution_dir = execution_dir or settings.tasks_dir / task.id / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(execution_dir / "notebook_steps.json", {"steps": steps, "cells": []})
    return steps


def scan_notebook_contract(
    task: TaskRecord,
    artifacts: list[FileArtifact],
) -> dict:
    notebook_path, error = resolve_scan_material(
        task=task,
        artifacts=artifacts,
        role=FileRole.NOTEBOOK,
        label="Notebook 文件",
        task_field="notebook_path",
    )
    if error or notebook_path is None:
        return {}
    try:
        return inspect_notebook_contract(notebook_path)
    except NotebookContractError as exc:
        return {
            "read_only": True,
            "source": "notebook_static_scan",
            "error": format_notebook_contract_error(exc),
        }
    except Exception as exc:
        return {
            "read_only": True,
            "source": "notebook_static_scan",
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def scan_preflight_checks(
    task: TaskRecord,
    artifacts: list[FileArtifact],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    resolved_paths: dict[FileRole, Path] = {}
    for role, label, task_field in REQUIRED_SCAN_MATERIALS:
        path, error = resolve_scan_material(
            task=task,
            artifacts=artifacts,
            role=role,
            label=label,
            task_field=task_field,
        )
        check_id = f"material_{role.value}"
        if error:
            checks.append(
                {
                    "id": check_id,
                    "label": label,
                    "status": "error",
                    "message": error,
                }
            )
            continue
        assert path is not None
        resolved_paths[role] = path
        checks.append(
            {
                "id": check_id,
                "label": label,
                "status": "success",
                "message": f"已识别：{path.name}",
            }
        )

    notebook_path = resolved_paths.get(FileRole.NOTEBOOK)
    if notebook_path is not None:
        try:
            result = precheck_notebook_contract(notebook_path)
        except NotebookContractError as exc:
            checks.append(
                {
                    "id": "notebook_contract",
                    "label": "Notebook RMC 契约",
                    "status": "error",
                    "message": format_notebook_contract_error(exc),
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "id": "notebook_contract",
                    "label": "Notebook RMC 契约",
                    "status": "error",
                    "message": f"Notebook 契约检查失败：{exc.__class__.__name__}: {exc}",
                }
            )
        else:
            target_note = f"，目标列 {result.target_col}" if result.target_col else ""
            algorithm_note = f"，算法 {result.algorithm}" if result.algorithm else ""
            if (
                result.algorithm
                and result.algorithm_raw
                and result.algorithm_raw != result.algorithm
            ):
                algorithm_note += f"（原始 {result.algorithm_raw}）"
            checks.append(
                {
                    "id": "notebook_contract",
                    "label": "Notebook RMC 契约",
                    "status": "success",
                    "message": (
                        "已定义 RMC_SAMPLE_DF / RMC_SCORE_FN / RMC_TARGET_COL / "
                        f"RMC_ALGORITHM{target_note}{algorithm_note}"
                    ),
                }
            )
    return checks


def resolve_scan_material(
    *,
    task: TaskRecord,
    artifacts: list[FileArtifact],
    role: FileRole,
    label: str,
    task_field: str,
) -> tuple[Path | None, str | None]:
    source_dir = Path(task.source_dir).resolve()
    explicit_value = getattr(task, task_field, None)
    if explicit_value:
        raw_path = Path(explicit_value)
        candidate = raw_path if raw_path.is_absolute() else source_dir / raw_path
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            return None, f"配置的 {label} 路径不存在：{candidate}"
        try:
            resolved = assert_within(source_dir, resolved)
        except PermissionError:
            return None, f"配置的 {label} 必须位于材料目录内：{resolved}"
        return resolved, None

    candidates = [artifact.path for artifact in artifacts if artifact.role == role]
    if not candidates:
        return None, f"缺少必需材料：{label}"
    if len(candidates) > 1:
        candidate_text = "、".join(path.name for path in candidates[:5])
        return None, f"{label} 有 {len(candidates)} 个候选，请在创建任务时显式指定：{candidate_text}"
    return candidates[0], None
