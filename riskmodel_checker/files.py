from hashlib import sha256
from pathlib import Path

from riskmodel_checker.domain import FileArtifact, FileRole


def classify_file(path: Path) -> FileRole:
    name = path.name
    if name.startswith((".~", "~$")) or name == ".DS_Store":
        return FileRole.UNKNOWN

    suffix = path.suffix.lower()
    lower_name = name.lower()

    if suffix == ".ipynb":
        return FileRole.NOTEBOOK
    if suffix in {".xlsx", ".csv"} and ("字典" in name or "dictionary" in lower_name):
        return FileRole.DATA_DICTIONARY
    if suffix in {".feather", ".csv", ".parquet"}:
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
) -> list[FileArtifact]:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

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


def _is_hidden_or_checkpoint_path(relative_path: Path) -> bool:
    return any(
        part.startswith(".") or part == ".ipynb_checkpoints"
        for part in relative_path.parts
    )
