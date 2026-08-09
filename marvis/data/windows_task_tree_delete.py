"""Handle-bound recursive deletion of task-owned trees on Windows.

Directory enumeration is deliberately a thin, injectable boundary. Every
enumerated child is reopened without following reparse points, authenticated by
its handle, and deleted by handle only after its own children are removed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
import ntpath
import os
from os import PathLike
from typing import Protocol

from marvis.data.windows_handle_api import (
    CtypesWindowsHandleApi as _CtypesWindowsTaskTreeApi,
)


_DELETE = 0x00010000
_FILE_READ_ATTRIBUTES = 0x00000080
_FILE_WRITE_ATTRIBUTES = 0x00000100
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_DEVICE = 0x00000040
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_TYPE_DISK = 0x0001
_FILE_DISPOSITION_DELETE = 0x00000001
_FILE_DISPOSITION_POSIX_SEMANTICS = 0x00000002
_FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE = 0x00000010
_MISSING_ERROR_CODES = frozenset({2, 3})
_SHARING_ERROR_CODES = frozenset({32, 33})


class _WindowsTaskTreeApi(Protocol):
    def open_handle(
        self,
        path: str,
        *,
        desired_access: int,
        share_mode: int,
        creation_disposition: int,
        flags_and_attributes: int,
    ) -> int: ...

    def file_attributes(self, handle: int) -> int: ...

    def file_type(self, handle: int) -> int: ...

    def final_path(self, handle: int) -> str: ...

    def mark_delete(self, handle: int, *, flags: int) -> None: ...

    def close_handle(self, handle: int) -> None: ...


class WindowsTaskTreeDeleteUnsupportedError(RuntimeError):
    """Raised when the Windows-only deletion primitive is unavailable."""


class WindowsTaskTreeDeleteRetryableError(OSError):
    """Raised for a transient Windows handle or filesystem failure."""

    def __init__(self, error_code: int | None, message: str) -> None:
        super().__init__(error_code or 0, message)
        self.winerror = error_code


class UnsafeWindowsTaskTreeDeleteError(PermissionError):
    """Raised when a task tree cannot be proven safe by opened handles."""


def delete_task_tree_windows(
    target_path: str | PathLike[str],
    trusted_root: str | PathLike[str],
    *,
    native_api: _WindowsTaskTreeApi | None = None,
    enumerate_children: Callable[[str], Iterable[str]] | None = None,
) -> None:
    """Recursively delete one verified task directory through Windows handles."""

    if native_api is None:
        if os.name != "nt":
            raise WindowsTaskTreeDeleteUnsupportedError(
                "handle-bound task tree deletion requires Windows"
            )
        native_api = _load_native_api()
    enumerate_children = enumerate_children or _enumerate_child_names

    root_text = _windows_path_text(trusted_root, label="trusted root")
    target_text = _windows_path_text(target_path, label="task tree target")
    _assert_strict_windows_child(
        target_text,
        root_text,
        message="task tree target is outside the trusted root",
    )

    # Do not share DELETE.  The open directory handle must pin its name while
    # path-based enumeration opens each child; otherwise a rename/replace ABA
    # race could make a child with the same final path belong to a different
    # directory object.  The handle itself already requested DELETE access, so
    # SetFileInformationByHandle can still mark that object for deletion.
    share_mode = _FILE_SHARE_READ | _FILE_SHARE_WRITE
    try:
        with _opened_handle(
            native_api,
            root_text,
            desired_access=_FILE_READ_ATTRIBUTES,
            share_mode=share_mode,
        ) as root_handle:
            if root_handle is None:
                return
            final_root = _verified_entry_path(
                native_api,
                root_handle,
                expected_directory=True,
                description="trusted root",
            )
            with _opened_handle(
                native_api,
                target_text,
                desired_access=(
                    _DELETE | _FILE_READ_ATTRIBUTES | _FILE_WRITE_ATTRIBUTES
                ),
                share_mode=share_mode,
            ) as target_handle:
                if target_handle is None:
                    return
                _delete_opened_entry(
                    native_api,
                    target_text,
                    target_handle,
                    final_parent=final_root,
                    expected_directory=True,
                    enumerate_children=enumerate_children,
                    share_mode=share_mode,
                )
    except (
        UnsafeWindowsTaskTreeDeleteError,
        WindowsTaskTreeDeleteRetryableError,
        WindowsTaskTreeDeleteUnsupportedError,
    ):
        raise
    except OSError as exc:
        raise _retryable_error("Windows task tree operation failed", exc) from exc


def _delete_opened_entry(
    native_api: _WindowsTaskTreeApi,
    path: str,
    handle: int,
    *,
    final_parent: str,
    expected_directory: bool | None,
    enumerate_children: Callable[[str], Iterable[str]],
    share_mode: int,
) -> None:
    final_path = _verified_entry_path(
        native_api,
        handle,
        expected_directory=expected_directory,
        description="task tree entry",
    )
    _assert_strict_windows_child(
        final_path,
        final_parent,
        message="opened task tree entry escaped its verified parent",
    )
    attributes = native_api.file_attributes(handle)
    if attributes & _FILE_ATTRIBUTE_DIRECTORY:
        child_names = tuple(enumerate_children(path))
        for child_name in child_names:
            normalized_name = _validated_child_name(child_name)
            child_path = ntpath.join(path, normalized_name)
            with _opened_handle(
                native_api,
                child_path,
                desired_access=(
                    _DELETE | _FILE_READ_ATTRIBUTES | _FILE_WRITE_ATTRIBUTES
                ),
                share_mode=share_mode,
            ) as child_handle:
                if child_handle is None:
                    continue
                _delete_opened_entry(
                    native_api,
                    child_path,
                    child_handle,
                    final_parent=final_path,
                    expected_directory=None,
                    enumerate_children=enumerate_children,
                    share_mode=share_mode,
                )
    native_api.mark_delete(handle, flags=_delete_disposition_flags())


def _verified_entry_path(
    native_api: _WindowsTaskTreeApi,
    handle: int,
    *,
    expected_directory: bool | None,
    description: str,
) -> str:
    attributes = native_api.file_attributes(handle)
    is_directory = bool(attributes & _FILE_ATTRIBUTE_DIRECTORY)
    if (
        native_api.file_type(handle) != _FILE_TYPE_DISK
        or attributes & (_FILE_ATTRIBUTE_DEVICE | _FILE_ATTRIBUTE_REPARSE_POINT)
        or (expected_directory is not None and is_directory is not expected_directory)
    ):
        raise UnsafeWindowsTaskTreeDeleteError(
            f"{description} is not the expected disk object"
        )
    final_path = native_api.final_path(handle)
    _windows_path_text(final_path, label=f"opened {description}")
    return final_path


@contextmanager
def _opened_handle(
    native_api: _WindowsTaskTreeApi,
    path: str,
    *,
    desired_access: int,
    share_mode: int,
) -> Iterator[int | None]:
    try:
        handle = native_api.open_handle(
            path,
            desired_access=desired_access,
            share_mode=share_mode,
            creation_disposition=_OPEN_EXISTING,
            flags_and_attributes=(
                _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT
            ),
        )
    except OSError as exc:
        if _windows_error_code(exc) in _MISSING_ERROR_CODES:
            yield None
            return
        raise _retryable_error("Windows could not open task tree path", exc) from exc

    body_failed = False
    try:
        yield handle
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            native_api.close_handle(handle)
        except OSError as exc:
            if not body_failed:
                raise _retryable_error(
                    "Windows could not close task tree handle",
                    exc,
                ) from exc


def _enumerate_child_names(path: str) -> tuple[str, ...]:
    try:
        with os.scandir(path) as entries:
            return tuple(entry.name for entry in entries)
    except OSError as exc:
        raise _retryable_error(
            "Windows could not enumerate task tree directory",
            exc,
        ) from exc


def _validated_child_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or ":" in value
        or ntpath.isabs(value)
    ):
        raise UnsafeWindowsTaskTreeDeleteError(
            "directory enumeration returned an unsafe child name"
        )
    return value


def _load_native_api() -> _WindowsTaskTreeApi:
    try:
        return _CtypesWindowsTaskTreeApi()
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise WindowsTaskTreeDeleteUnsupportedError(
            "Windows task tree handle deletion API is unavailable"
        ) from exc


def _windows_path_text(value: str | PathLike[str], *, label: str) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise UnsafeWindowsTaskTreeDeleteError(f"{label} must be path-like") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise UnsafeWindowsTaskTreeDeleteError(
            f"{label} must be a non-empty Windows text path"
        )
    normalized = _normalize_windows_path(raw)
    if not ntpath.isabs(normalized):
        raise UnsafeWindowsTaskTreeDeleteError(f"{label} must be absolute")
    drive, tail = ntpath.splitdrive(normalized)
    if not drive:
        raise UnsafeWindowsTaskTreeDeleteError(
            f"{label} must be a fully qualified Windows path"
        )
    if ":" in tail:
        raise UnsafeWindowsTaskTreeDeleteError(
            f"{label} must not address an alternate data stream"
        )
    return raw


def _normalize_windows_path(value: str) -> str:
    path = value.replace("/", "\\")
    upper = path.upper()
    if upper.startswith("\\\\.\\"):
        raise UnsafeWindowsTaskTreeDeleteError(
            "Windows device namespace paths are not valid task tree paths"
        )
    if upper.startswith("\\\\?\\UNC\\"):
        path = "\\\\" + path[8:]
    elif upper.startswith("\\??\\UNC\\"):
        path = "\\\\" + path[8:]
    elif upper.startswith("\\\\?\\"):
        extended_path = path[4:]
        if len(extended_path) >= 3 and extended_path[1:3] == ":\\":
            path = extended_path
        elif not extended_path.upper().startswith("VOLUME{"):
            raise UnsafeWindowsTaskTreeDeleteError(
                "unsupported Windows extended path namespace"
            )
    elif upper.startswith("\\??\\"):
        path = path[4:]
    return ntpath.normcase(ntpath.normpath(path))


def _assert_strict_windows_child(
    candidate: str,
    root: str,
    *,
    message: str,
) -> None:
    normalized_candidate = _normalize_windows_path(candidate)
    normalized_root = _normalize_windows_path(root)
    candidate_drive, candidate_tail = ntpath.splitdrive(normalized_candidate)
    root_drive, root_tail = ntpath.splitdrive(normalized_root)
    if (
        not candidate_drive
        or not root_drive
        or not ntpath.isabs(normalized_candidate)
        or not ntpath.isabs(normalized_root)
        or ":" in candidate_tail
        or ":" in root_tail
    ):
        raise UnsafeWindowsTaskTreeDeleteError(message)
    try:
        common = ntpath.commonpath((normalized_root, normalized_candidate))
    except ValueError as exc:
        raise UnsafeWindowsTaskTreeDeleteError(message) from exc
    if common != normalized_root or normalized_candidate == normalized_root:
        raise UnsafeWindowsTaskTreeDeleteError(message)


def _delete_disposition_flags() -> int:
    return (
        _FILE_DISPOSITION_DELETE
        | _FILE_DISPOSITION_POSIX_SEMANTICS
        | _FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE
    )


def _windows_error_code(exc: OSError) -> int | None:
    winerror = getattr(exc, "winerror", None)
    if isinstance(winerror, int):
        return winerror
    return exc.errno if isinstance(exc.errno, int) else None


def _retryable_error(
    operation: str,
    exc: OSError,
) -> WindowsTaskTreeDeleteRetryableError:
    error_code = _windows_error_code(exc)
    failure_kind = (
        "sharing or lock conflict"
        if error_code in _SHARING_ERROR_CODES
        else "filesystem error"
    )
    return WindowsTaskTreeDeleteRetryableError(
        error_code,
        f"{operation}: {failure_kind}",
    )


__all__ = [
    "UnsafeWindowsTaskTreeDeleteError",
    "WindowsTaskTreeDeleteRetryableError",
    "WindowsTaskTreeDeleteUnsupportedError",
    "delete_task_tree_windows",
]
