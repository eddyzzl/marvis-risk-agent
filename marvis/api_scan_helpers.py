from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import uuid

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db import TaskRepository
from marvis.domain import (
    TASK_TYPE_VALIDATION,
    FileArtifact,
    FileRole,
    TaskRecord,
    TaskStatus,
)
from marvis.files import (
    EXCEL_SUFFIXES,
    SAMPLE_SUFFIXES,
    scan_source_dir,
    sha256_file,
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
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.safe_paths import assert_within
from marvis.validation.feature_metadata import (
    FeatureMetadataSelection,
    inspect_feature_metadata,
    normalize_feature_metadata,
)
from marvis.validation.field_recognition import recognize_notebook_fields
from marvis.validation.input_contracts import (
    INPUT_CONTRACT_SCHEMA,
    FIELD_RECOGNITION_SCHEMA,
    FieldCandidate,
    FieldEvidence,
    FieldRecognitionResult,
    ValidationInputContract,
)
from marvis.validation.pmml_manifest import parse_pmml_input_manifest
from marvis.validation.sample_schema import inspect_sample_schema
from marvis.validation_materials import (
    ResolvedValidationMaterials,
    resolve_selected_validation_materials,
)


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


class SourceDirectoryScanError(ValueError):
    """A source-tree guardrail failure, distinct from v2 material validation."""


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
        result = f"Notebook RMC 契约检查失败：缺少 {missing_text}。请在 Notebook 顶层定义后重新扫描。"
    else:
        result = f"Notebook RMC 契约检查失败：{message}"
    details = []
    if exc.cell_index is not None and exc.line_number is not None:
        details.append(f"位置：代码单元 {exc.cell_index}，第 {exc.line_number} 行")
    if exc.notebook_path:
        details.append(f"文件：{exc.notebook_path}")
    if exc.notebook_sha256:
        details.append(f"SHA-256：{exc.notebook_sha256}")
    if exc.source_excerpt:
        details.append("代码：" + exc.source_excerpt.replace("\n", " | "))
    if details:
        separator = " " if result.endswith(("。", "！", "？")) else "；"
        result += separator + "；".join(details)
    return result


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
    try:
        artifacts = scan_source_dir(source_dir)
    except ValueError as exc:
        raise SourceDirectoryScanError(str(exc)) from exc
    is_v2_validation = _uses_v2_validation_contract(task)
    validation_contract: ValidationInputContract | None = None
    resolved_materials: ResolvedValidationMaterials | None = None
    if is_v2_validation:
        resolved_materials = resolve_selected_validation_materials(task)
        validation_contract = build_validation_input_contract(
            notebook_path=resolved_materials.notebook,
            sample_path=resolved_materials.sample,
            pmml_path=resolved_materials.pmml,
            dictionary_path=resolved_materials.dictionary,
            known_hashes={
                artifact.path.resolve(): artifact.sha256
                for artifact in artifacts
                if artifact.sha256
            },
        )
        checks = _v2_scan_checks(resolved_materials, validation_contract)
    else:
        checks = scan_preflight_checks(task, artifacts)
    task_dir = settings.tasks_dir / task.id
    uow = ArtifactUnitOfWork()
    staged_execution = uow.stage_directory(task_dir, "execution")
    staged_execution.path.mkdir(parents=True, exist_ok=True)
    previous_scan = _archive_previous_scan(task_dir / "execution", staged_execution.path)
    uow.remove_path(task_dir / "outputs")
    uow.remove_path(task_dir / "images")
    notebook_steps = scan_notebook_steps(
        settings,
        task,
        artifacts,
        execution_dir=staged_execution.path,
    )
    notebook_contract = (
        {
            "read_only": True,
            "source": "notebook_static_recognition",
        }
        if is_v2_validation
        else scan_notebook_contract(task, artifacts)
    )
    scan_status = TaskStatus.FAILED if scan_error_checks(checks) else TaskStatus.SCANNED
    scan_message = scan_status_message(checks)
    payload = {
        "scan_id": uuid.uuid4().hex,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task.id,
        "status": scan_status.value,
        "status_message": scan_message,
        "notebook_revision": notebook_revision_payload(task, artifacts),
        "previous_scan": previous_scan,
        "artifacts": [artifact_payload(artifact) for artifact in artifacts],
        "ambiguities": artifact_ambiguities(artifacts),
        "selected_materials": material_selection_payload(task, source_dir),
        "checks": checks,
        "notebook_steps": notebook_steps,
        "notebook_contract": notebook_contract,
    }
    if not is_v2_validation:
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
        if validation_contract is not None:
            record = ValidationContractRepository(
                settings.db_path
            ).replace_candidates_on_connection(
                conn,
                task.id,
                validation_contract,
            )
            payload["validation_input_contract"] = record.to_api_payload()
            (staged_execution.final_path / "scan_result.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
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


def _uses_v2_validation_contract(task: TaskRecord) -> bool:
    return (
        task.task_type == TASK_TYPE_VALIDATION
        and task.validation_workflow_version == 2
    )


def build_validation_input_contract(
    *,
    notebook_path: Path,
    sample_path: Path,
    pmml_path: Path,
    dictionary_path: Path,
    known_hashes: dict[Path, str] | None = None,
) -> ValidationInputContract:
    """Build bounded, read-only v2 candidates from four selected materials."""

    known = {Path(path).resolve(): digest for path, digest in (known_hashes or {}).items()}
    conflicts: list[str] = []

    try:
        sample_schema = inspect_sample_schema(sample_path)
        sample_digest = sample_schema.sha256
    except ValueError as exc:
        sample_schema = None
        sample_digest = _known_or_hash(sample_path, known)
        conflicts.append(f"sample: {exc}")

    try:
        fields = recognize_notebook_fields(notebook_path)
        notebook_digest = fields.notebook_sha256
    except ValueError as exc:
        notebook_digest = _known_or_hash(notebook_path, known)
        fields = FieldRecognitionResult(
            schema_version=FIELD_RECOGNITION_SCHEMA,
            notebook_sha256=notebook_digest,
            candidates={},
            transformations=(),
            conflicts=(f"Notebook: {exc}",),
            diagnostics=(),
        )

    try:
        manifest = parse_pmml_input_manifest(pmml_path)
    except ValueError as exc:
        manifest = None
        conflicts.append(f"PMML: {exc}")
    pmml_digest = _known_or_hash(pmml_path, known)

    metadata = None
    metadata_selections: tuple[FeatureMetadataSelection, ...] = ()
    if manifest is not None:
        try:
            inspection = inspect_feature_metadata(dictionary_path, manifest)
            metadata_selections = inspection.selections
            conflicts.extend(
                f"feature metadata: {message}"
                for message in inspection.blocking_errors
            )
            if len(metadata_selections) == 1:
                metadata = normalize_feature_metadata(
                    dictionary_path,
                    selection=metadata_selections[0],
                    manifest=manifest,
                )
        except ValueError as exc:
            conflicts.append(f"feature metadata: {exc}")
    else:
        conflicts.append("feature metadata: PMML manifest is unavailable")
    dictionary_digest = _known_or_hash(dictionary_path, known)

    return assemble_validation_input_contract(
        material_hashes={
            "notebook": notebook_digest,
            "sample": sample_digest,
            "pmml": pmml_digest,
            "dictionary": dictionary_digest,
        },
        sample_schema=sample_schema,
        fields=fields,
        manifest=manifest,
        metadata=metadata,
        metadata_selections=metadata_selections,
        conflicts=tuple(conflicts),
    )


def assemble_validation_input_contract(
    *,
    material_hashes: dict[str, str],
    sample_schema,
    fields: FieldRecognitionResult,
    manifest,
    metadata,
    metadata_selections: tuple[FeatureMetadataSelection, ...],
    conflicts: tuple[str, ...],
) -> ValidationInputContract:
    candidates = {key: tuple(value) for key, value in fields.candidates.items()}
    hard_conflicts = [*fields.conflicts, *conflicts]

    if manifest is not None:
        algorithm = str(manifest.algorithm).strip()
        if algorithm:
            candidates["algorithm"] = _append_candidate(
                candidates.get("algorithm", ()),
                _material_candidate(algorithm, "pmml_manifest", "PMML algorithm"),
            )
        else:
            hard_conflicts.append("PMML algorithm is unavailable")

        output_candidates = tuple(
            value for value in manifest.output_candidates if str(value).strip()
        )
        if output_candidates:
            candidates["pmml_output_field"] = tuple(
                [*candidates.get("pmml_output_field", ())]
                + [
                    _material_candidate(value, "pmml_output", "PMML output field")
                    for value in output_candidates
                ]
            )
        else:
            hard_conflicts.append("PMML output field is unavailable")
        if manifest.unsupported_derivations:
            hard_conflicts.append(
                "unsupported PMML stress dependencies: "
                + ", ".join(manifest.unsupported_derivations)
            )
        if sample_schema is not None:
            producible_fields = _candidate_producible_fields(
                sample_columns=frozenset(sample_schema.columns),
                transformations=fields.transformations,
            )
            missing_inputs = [
                name
                for name in manifest.raw_required_fields
                if name not in producible_fields
            ]
            if missing_inputs:
                hard_conflicts.append(
                    "sample missing required PMML inputs: " + ", ".join(missing_inputs)
                )

    if metadata_selections:
        candidates["feature_metadata_selection"] = tuple(
            _material_candidate(
                {
                    "metadata_sheet": selection.sheet_name,
                    "feature_col": selection.feature_col,
                    "category_col": selection.category_col,
                    "importance_col": selection.importance_col,
                },
                "feature_metadata",
                "feature metadata column selection",
            )
            for selection in metadata_selections
        )
    else:
        hard_conflicts.append("feature metadata selection is unavailable")

    return ValidationInputContract(
        schema_version=INPUT_CONTRACT_SCHEMA,
        material_hashes=material_hashes,
        status="blocked" if hard_conflicts else "pending_confirmation",
        candidates=candidates,
        sample_schema=sample_schema,
        pmml_manifest=manifest,
        feature_metadata=metadata,
        confirmed={},
        transformations=fields.transformations,
        conflicts=tuple(_bounded_conflicts(hard_conflicts)),
    )


def _known_or_hash(path: Path, known_hashes: dict[Path, str]) -> str:
    selected = Path(path).resolve()
    return known_hashes.get(selected) or sha256_file(selected)


def _material_candidate(value, source_kind: str, excerpt: str) -> FieldCandidate:
    return FieldCandidate(
        value=value,
        evidence=(
            FieldEvidence(
                source_kind=source_kind,
                notebook_cell=None,
                source_excerpt=excerpt,
                confidence=1.0,
            ),
        ),
    )


def _append_candidate(
    values: tuple[FieldCandidate, ...], candidate: FieldCandidate
) -> tuple[FieldCandidate, ...]:
    return (*values, candidate)


def _candidate_producible_fields(
    *, sample_columns: frozenset[str], transformations
) -> frozenset[str]:
    """Resolve transformation dependencies without recursion.

    A field is producible when it is present in the sample or at least one of its
    transformation alternatives has all inputs available.  The dependency queue
    handles long chains and leaves cyclic-only components unresolved.
    """
    producible = set(sample_columns)
    pending_inputs: list[set[str]] = []
    outputs: list[str] = []
    waiting_by_input: dict[str, list[int]] = {}
    ready: list[str] = []

    for index, spec in enumerate(transformations):
        dependencies = set(spec.input_fields) - producible
        pending_inputs.append(dependencies)
        outputs.append(spec.output_field)
        if not dependencies:
            ready.append(spec.output_field)
        for dependency in dependencies:
            waiting_by_input.setdefault(dependency, []).append(index)

    while ready:
        field = ready.pop()
        if field in producible:
            continue
        producible.add(field)
        for index in waiting_by_input.get(field, ()):
            dependencies = pending_inputs[index]
            dependencies.discard(field)
            if not dependencies:
                ready.append(outputs[index])

    return frozenset(producible)


def _bounded_conflicts(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values[:64]:
        text = str(value)
        bounded = text if len(text) <= 500 else text[:497] + "..."
        if bounded not in result:
            result.append(bounded)
    return result


def _v2_scan_checks(
    materials: ResolvedValidationMaterials,
    contract: ValidationInputContract,
) -> list[dict[str, str]]:
    checks = [
        {
            "id": f"material_{role.value}",
            "label": label,
            "status": "success",
            "message": f"已识别：{path.name}",
        }
        for (role, label, _field), path in zip(
            REQUIRED_SCAN_MATERIALS,
            (
                materials.notebook,
                materials.sample,
                materials.pmml,
                materials.dictionary,
            ),
            strict=True,
        )
    ]
    if contract.conflicts:
        checks.append(
            {
                "id": "validation_input_contract",
                "label": "验证输入契约",
                "status": "error",
                "message": _bounded_scan_error_message(contract.conflicts),
            }
        )
    else:
        checks.append(
            {
                "id": "validation_input_contract",
                "label": "验证输入契约",
                "status": "success",
                "message": "字段与元数据候选已生成，等待用户确认。",
            }
        )
    return checks


def _bounded_scan_error_message(conflicts: tuple[str, ...], limit: int = 2_000) -> str:
    message = "；".join(conflicts)
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def artifact_payload(artifact) -> dict:
    return {
        "role": artifact.role.value,
        "path": str(artifact.path),
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "risk_notes": artifact.risk_notes,
    }


def notebook_revision_payload(
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
    stat = notebook_path.stat()
    artifact = next(
        (item for item in artifacts if item.path.resolve() == notebook_path.resolve()),
        None,
    )
    digest = artifact.sha256 if artifact is not None else None
    return {
        "path": str(notebook_path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": digest or sha256_file(notebook_path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "mtime_ns": stat.st_mtime_ns,
    }


def _archive_previous_scan(previous_execution: Path, staged_execution: Path) -> dict | None:
    previous_history = previous_execution / "scan_history"
    staged_history = staged_execution / "scan_history"
    if previous_history.is_dir():
        shutil.copytree(previous_history, staged_history, dirs_exist_ok=True)

    previous_result = previous_execution / "scan_result.json"
    if not previous_result.is_file():
        return None
    raw = previous_result.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    staged_history.mkdir(parents=True, exist_ok=True)
    archive_path = staged_history / f"{sha256(raw).hexdigest()}.json"
    if not archive_path.exists():
        shutil.copy2(previous_result, archive_path)
    return _scan_history_summary(payload)


def _scan_history_summary(payload: dict) -> dict:
    scan_id = str(payload.get("scan_id") or "")
    if not scan_id:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        scan_id = f"legacy-{sha256(encoded).hexdigest()[:16]}"
    return {
        "scan_id": scan_id,
        "scanned_at": payload.get("scanned_at"),
        "status": payload.get("status"),
        "status_message": payload.get("status_message"),
        "notebook_revision": payload.get("notebook_revision") or {},
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
