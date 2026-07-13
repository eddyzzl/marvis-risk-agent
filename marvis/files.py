import json
from hashlib import sha256
from pathlib import Path
import uuid
from typing import Any

from marvis.domain import FileArtifact, FileRole

EXCEL_SUFFIXES = {".xlsx", ".xls"}
SAMPLE_SUFFIXES = {".feather", ".csv", ".parquet"}
DICTIONARY_KEYWORDS = ("字典", "dictionary")
FEATURE_METADATA_KEYWORDS = (
    "元数据",
    "metadata",
    "特征重要性",
    "feature importance",
    "feature_importance",
)
FEATURE_METADATA_SUFFIXES = EXCEL_SUFFIXES | {".csv", ".parquet"}
EXCEL_SAMPLE_KEYWORDS = (
    "样本",
    "数据",
    "建模",
    "sample",
    "data",
    "modeling",
)


def classify_file(path: Path) -> FileRole:
    name = path.name
    if name.startswith((".~", "~$")) or name == ".DS_Store":
        return FileRole.UNKNOWN

    suffix = path.suffix.lower()
    lower_name = name.lower()

    if suffix == ".ipynb":
        return FileRole.NOTEBOOK
    if suffix in FEATURE_METADATA_SUFFIXES and any(
        keyword in name or keyword in lower_name
        for keyword in (*DICTIONARY_KEYWORDS, *FEATURE_METADATA_KEYWORDS)
    ):
        return FileRole.DATA_DICTIONARY
    if suffix in SAMPLE_SUFFIXES:
        return FileRole.SAMPLE
    if suffix in EXCEL_SUFFIXES and any(
        keyword in name or keyword in lower_name for keyword in EXCEL_SAMPLE_KEYWORDS
    ):
        return FileRole.SAMPLE
    if suffix == ".pmml":
        return FileRole.MODEL_PMML
    return FileRole.UNKNOWN


def scan_source_dir(
    source_dir: Path,
    hash_limit_bytes: int = 50 * 1024 * 1024,
    *,
    max_files: int = 2000,
    max_depth: int = 6,
    max_total_hash_bytes: int = 500 * 1024 * 1024,
    include_unknown_suffixes: set[str] | frozenset[str] | None = None,
) -> list[FileArtifact]:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

    included_unknown = {
        str(suffix).lower() for suffix in (include_unknown_suffixes or ())
    }
    artifacts: list[FileArtifact] = []
    scanned_files = 0
    total_hashed_bytes = 0
    for path in sorted(source_dir.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_dir)
        if _is_hidden_or_checkpoint_path(relative_path):
            continue
        if max_depth > 0 and len(relative_path.parent.parts) > max_depth:
            raise ValueError(
                f"source_dir is too deep: {relative_path} exceeds max_depth={max_depth}"
            )
        scanned_files += 1
        if max_files > 0 and scanned_files > max_files:
            raise ValueError(f"source_dir has too many files: max_files={max_files}")

        role = classify_file(path)
        if role == FileRole.UNKNOWN:
            if path.name.startswith((".~", "~$")) or path.name == ".DS_Store":
                continue
            if path.suffix.lower() not in included_unknown:
                continue

        size_bytes = path.stat().st_size
        should_hash = (
            size_bytes <= hash_limit_bytes
            and total_hashed_bytes + size_bytes <= max_total_hash_bytes
        )
        digest = sha256_file(path) if should_hash else None
        if should_hash:
            total_hashed_bytes += size_bytes
        risk_notes = []
        artifacts.append(
            FileArtifact(
                role=role,
                path=path,
                size_bytes=size_bytes,
                sha256=digest,
                risk_notes=risk_notes,
            )
        )
    return artifacts


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(content, encoding=encoding)
        temp_path.replace(target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    encoding: str = "utf-8",
) -> Path:
    return write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent),
        encoding=encoding,
    )


def _is_hidden_or_checkpoint_path(relative_path: Path) -> bool:
    return any(
        part.startswith(".") or part == ".ipynb_checkpoints"
        for part in relative_path.parts
    )
