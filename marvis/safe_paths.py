import re
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath


_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w一-鿿\-]+", re.UNICODE)
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def safe_filename_component(
    value: str,
    *,
    max_length: int = 64,
    fallback: str = "_",
) -> str:
    normalized = unicodedata.normalize("NFC", value)
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", normalized).strip("._")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:max_length]


def assert_within(parent: Path, candidate: Path) -> Path:
    resolved_parent = parent.resolve()
    resolved_candidate = candidate.resolve()
    if (
        resolved_candidate != resolved_parent
        and resolved_parent not in resolved_candidate.parents
    ):
        raise PermissionError(f"path escape: {candidate} not within {parent}")
    return resolved_candidate


def lexical_child_path(root: Path, relative_path: str) -> Path:
    """Return a lexical child without resolving its final component.

    This helper is for unlink/remove boundaries where resolving the final
    symlink would redirect deletion to its target.  Intermediate symlinks are
    rejected because operating on a descendant would still traverse them.
    Callers with an adversarial concurrent filesystem must additionally use
    descriptor-relative operations to close the check/use race.
    """

    root_path = Path(root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PermissionError(f"path escapes root: {relative_path}")
    current = root_path
    for part in relative.parts[:-1]:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise PermissionError(f"path traverses symlink: {relative_path}")
    return root_path.joinpath(*relative.parts)


def canonical_relative_posix_path(value: str) -> str:
    """Return the one persisted spelling for a workspace-relative path."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("relative path must be non-empty text")
    normalized = unicodedata.normalize("NFC", value.replace("\\", "/"))
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or not posix.parts
        or ".." in posix.parts
    ):
        raise ValueError("path must stay relative to its workspace root")
    return "/".join(part for part in posix.parts if part not in {"", "."})


def conservative_relative_path_identity(value: str) -> str:
    """Normalize aliases conservatively for deletion reference checks.

    Case-folding can retain an extra file on a case-sensitive filesystem, but
    prevents deleting a still-referenced file on case-insensitive macOS and
    Windows volumes.  Garbage collection prefers a bounded leak over data loss.
    """

    return canonical_relative_posix_path(value).casefold()


def resolve_fixed_task_output(
    tasks_dir: Path,
    task_id: str,
    filename: str,
) -> Path | None:
    """Resolve one fixed task artifact without following task-owned symlinks.

    ``tasks_dir`` is a trusted configured root.  Below it, the task directory,
    ``outputs`` directory, and fixed artifact must all be real filesystem
    entries.  This prevents one task from publishing a symlink to another task
    (or to a path outside the workspace) through a download endpoint.
    """

    if (
        not task_id
        or Path(task_id).name != task_id
        or not filename
        or Path(filename).name != filename
    ):
        return None

    configured_root = Path(tasks_dir)
    if configured_root.absolute().is_symlink():
        return None
    task_dir = configured_root / task_id
    output_dir = task_dir / "outputs"
    candidate = output_dir / filename
    try:
        trusted_root = configured_root.resolve(strict=True)
        if any(path.is_symlink() for path in (task_dir, output_dir, candidate)):
            return None
        resolved_output = output_dir.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    expected_output = trusted_root / task_id / "outputs"
    if (
        resolved_output != expected_output
        or resolved_candidate.parent != resolved_output
        or not resolved_candidate.is_file()
    ):
        return None
    return resolved_candidate


def prepare_task_output_dir(tasks_dir: Path, task_id: str) -> Path:
    """Create and verify the exact non-symlink output directory for a task."""

    if not task_id or Path(task_id).name != task_id:
        raise PermissionError("invalid task output identifier")

    configured_root = Path(tasks_dir)
    if configured_root.absolute().is_symlink():
        raise PermissionError("task output root must not be a symlink")
    try:
        trusted_root = configured_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PermissionError("task output root is unavailable") from exc

    task_dir = configured_root / task_id
    output_dir = task_dir / "outputs"
    if task_dir.is_symlink() or output_dir.is_symlink():
        raise PermissionError("task output path must not contain symlinks")
    task_dir.mkdir(exist_ok=True)
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise PermissionError("task directory is not a real directory")
    output_dir.mkdir(exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise PermissionError("task output directory is not a real directory")
    try:
        resolved_output = output_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PermissionError("task output directory is unavailable") from exc
    if resolved_output != trusted_root / task_id / "outputs":
        raise PermissionError("task output directory escaped its task root")
    return output_dir
