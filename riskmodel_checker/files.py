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
    source_dir: Path, hash_limit_bytes: int = 50 * 1024 * 1024
) -> list[FileArtifact]:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

    artifacts: list[FileArtifact] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if _is_hidden_or_checkpoint_path(path.relative_to(source_dir)):
            continue

        role = classify_file(path)
        if role == FileRole.UNKNOWN:
            continue

        size_bytes = path.stat().st_size
        digest = sha256_file(path) if size_bytes <= hash_limit_bytes else None
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
        for part in relative_path.parts[:-1]
    )
