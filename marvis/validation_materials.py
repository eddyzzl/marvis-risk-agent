from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marvis.domain import FileRole
from marvis.files import scan_source_dir


@dataclass(frozen=True)
class ResolvedValidationMaterials:
    notebook: Path
    sample: Path
    pmml: Path
    dictionary: Path


def resolve_selected_validation_materials(task: Any) -> ResolvedValidationMaterials:
    return resolve_validation_material_paths(
        source_dir=getattr(task, "source_dir", ""),
        notebook_path=getattr(task, "notebook_path", None),
        sample_path=getattr(task, "sample_path", None),
        pmml_path=getattr(task, "pmml_path", None),
        dictionary_path=getattr(task, "dictionary_path", None),
    )


def resolve_validation_material_paths(
    *,
    source_dir: str | Path,
    notebook_path: str | Path | None,
    sample_path: str | Path | None,
    pmml_path: str | Path | None,
    dictionary_path: str | Path | None,
) -> ResolvedValidationMaterials:
    raw_source_dir = Path(source_dir).expanduser()
    try:
        root = raw_source_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"task source directory does not exist: {raw_source_dir}") from exc
    if not root.is_dir():
        raise ValueError(f"task source directory is not a directory: {root}")

    def selected(raw_value: str | Path | None, label: str) -> Path:
        value = str(raw_value or "").strip()
        if not value:
            raise ValueError(f"selected {label} path is missing")
        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else root / raw
        try:
            path = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"selected {label} file does not exist") from exc
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"selected {label} escapes source directory") from exc
        if not path.is_file():
            raise ValueError(f"selected {label} is not a regular file")
        return path

    return ResolvedValidationMaterials(
        notebook=selected(notebook_path, "Notebook"),
        sample=selected(sample_path, "sample"),
        pmml=selected(pmml_path, "PMML"),
        dictionary=selected(dictionary_path, "feature metadata"),
    )


_DISCOVER_ROLES = (
    (FileRole.NOTEBOOK, "notebook", "Notebook"),
    (FileRole.SAMPLE, "sample", "样本数据"),
    (FileRole.MODEL_PMML, "pmml", "PMML"),
    (FileRole.DATA_DICTIONARY, "dictionary", "数据字典"),
)


def discover_validation_material_paths(source_dir: str | Path) -> ResolvedValidationMaterials:
    raw_source_dir = Path(source_dir).expanduser()
    try:
        root = raw_source_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"task source directory does not exist: {raw_source_dir}") from exc
    if not root.is_dir():
        raise ValueError(f"task source directory is not a directory: {root}")
    artifacts = scan_source_dir(root)
    grouped: dict[FileRole, list[Path]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.role, []).append(artifact.path)
    resolved: dict[str, Path] = {}
    problems: list[str] = []
    for role, field, label in _DISCOVER_ROLES:
        matches = grouped.get(role, [])
        if not matches:
            problems.append(f"缺少必需材料：{label}")
            continue
        if len(matches) > 1:
            names = "、".join(path.name for path in matches[:5])
            problems.append(f"{label} 有 {len(matches)} 个候选，请显式指定：{names}")
            continue
        resolved[field] = matches[0]
    if problems:
        raise ValueError("；".join(problems))
    return ResolvedValidationMaterials(
        notebook=resolved["notebook"],
        sample=resolved["sample"],
        pmml=resolved["pmml"],
        dictionary=resolved["dictionary"],
    )


__all__ = [
    "ResolvedValidationMaterials",
    "discover_validation_material_paths",
    "resolve_selected_validation_materials",
    "resolve_validation_material_paths",
]
