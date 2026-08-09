"""Handle-bound deletion of dataset files on Windows.

The public operation is intentionally Windows-only.  Tests on other platforms
can inject the native boundary without loading ``kernel32``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import ntpath
import os
from os import PathLike
from typing import Protocol

from marvis.data.windows_handle_api import (
    CtypesWindowsHandleApi as _CtypesWindowsDeleteApi,
)


_DELETE = 0x00010000
_FILE_READ_ATTRIBUTES = 0x00000080
_FILE_WRITE_ATTRIBUTES = 0x00000100
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
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


class _WindowsDeleteApi(Protocol):
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


class WindowsDatasetDeleteUnsupportedError(RuntimeError):
    """Raised when the Windows-only deletion primitive is unavailable."""


class WindowsDatasetDeleteRetryableError(OSError):
    """Raised for a transient Windows handle or filesystem failure."""

    def __init__(self, error_code: int | None, message: str) -> None:
        super().__init__(error_code or 0, message)
        self.winerror = error_code


class UnsafeWindowsDatasetDeleteError(PermissionError):
    """Raised when the opened object is not a safe dataset file."""


def delete_dataset_file_windows(
    dataset_path: str | PathLike[str],
    datasets_root: str | PathLike[str],
    *,
    native_api: _WindowsDeleteApi | None = None,
) -> None:
    """Delete one dataset file through a verified Windows file handle."""

    if native_api is None:
        if os.name != "nt":
            raise WindowsDatasetDeleteUnsupportedError(
                "handle-bound dataset deletion requires Windows"
            )
        native_api = _load_native_api()

    root_text = _windows_path_text(datasets_root, label="datasets root")
    candidate_text = _windows_path_text(dataset_path, label="dataset path")
    _assert_strict_windows_child(
        candidate_text,
        root_text,
        message="dataset path is outside the configured datasets root",
    )

    share_mode = _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE
    try:
        with _opened_handle(
            native_api,
            root_text,
            desired_access=_FILE_READ_ATTRIBUTES,
            share_mode=share_mode,
            flags_and_attributes=(
                _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT
            ),
        ) as root_handle:
            if root_handle is None:
                return
            root_attributes = native_api.file_attributes(root_handle)
            if (
                native_api.file_type(root_handle) != _FILE_TYPE_DISK
                or not root_attributes & _FILE_ATTRIBUTE_DIRECTORY
                or root_attributes & _FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise UnsafeWindowsDatasetDeleteError(
                    "datasets root handle is not a real directory"
                )
            final_root = native_api.final_path(root_handle)

            with _opened_handle(
                native_api,
                candidate_text,
                desired_access=(
                    _DELETE | _FILE_READ_ATTRIBUTES | _FILE_WRITE_ATTRIBUTES
                ),
                share_mode=share_mode,
                flags_and_attributes=(
                    _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT
                ),
            ) as candidate_handle:
                if candidate_handle is None:
                    return
                attributes = native_api.file_attributes(candidate_handle)
                if native_api.file_type(
                    candidate_handle
                ) != _FILE_TYPE_DISK or attributes & (
                    _FILE_ATTRIBUTE_DIRECTORY
                    | _FILE_ATTRIBUTE_DEVICE
                    | _FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    raise UnsafeWindowsDatasetDeleteError(
                        "dataset delete target is not a regular non-reparse file"
                    )
                final_candidate = native_api.final_path(candidate_handle)
                _assert_strict_windows_child(
                    final_candidate,
                    final_root,
                    message="opened dataset file escaped the trusted datasets root",
                )
                native_api.mark_delete(
                    candidate_handle,
                    flags=(
                        _FILE_DISPOSITION_DELETE
                        | _FILE_DISPOSITION_POSIX_SEMANTICS
                        | _FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE
                    ),
                )
    except (
        UnsafeWindowsDatasetDeleteError,
        WindowsDatasetDeleteRetryableError,
    ):
        raise
    except OSError as exc:
        raise _retryable_error("Windows handle operation failed", exc) from exc


def _load_native_api() -> _WindowsDeleteApi:
    try:
        return _CtypesWindowsDeleteApi()
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise WindowsDatasetDeleteUnsupportedError(
            "Windows handle deletion API is unavailable"
        ) from exc


@contextmanager
def _opened_handle(
    native_api: _WindowsDeleteApi,
    path: str,
    *,
    desired_access: int,
    share_mode: int,
    flags_and_attributes: int,
) -> Iterator[int | None]:
    try:
        handle = native_api.open_handle(
            path,
            desired_access=desired_access,
            share_mode=share_mode,
            creation_disposition=_OPEN_EXISTING,
            flags_and_attributes=flags_and_attributes,
        )
    except OSError as exc:
        if _windows_error_code(exc) in _MISSING_ERROR_CODES:
            yield None
            return
        raise _retryable_error("Windows could not open dataset path", exc) from exc

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
                    "Windows could not close dataset handle",
                    exc,
                ) from exc


def _windows_path_text(
    value: str | PathLike[str],
    *,
    label: str,
) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise UnsafeWindowsDatasetDeleteError(f"{label} must be path-like") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise UnsafeWindowsDatasetDeleteError(
            f"{label} must be a non-empty Windows text path"
        )
    normalized = _normalize_windows_path(raw)
    if not ntpath.isabs(normalized):
        raise UnsafeWindowsDatasetDeleteError(f"{label} must be absolute")
    drive, tail = ntpath.splitdrive(normalized)
    if not drive:
        raise UnsafeWindowsDatasetDeleteError(
            f"{label} must be a fully qualified Windows path"
        )
    if ":" in tail:
        raise UnsafeWindowsDatasetDeleteError(
            f"{label} must not address an alternate data stream"
        )
    return raw


def _normalize_windows_path(value: str) -> str:
    path = value.replace("/", "\\")
    upper = path.upper()
    if upper.startswith("\\\\.\\"):
        raise UnsafeWindowsDatasetDeleteError(
            "Windows device namespace paths are not valid dataset paths"
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
            raise UnsafeWindowsDatasetDeleteError(
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
        raise UnsafeWindowsDatasetDeleteError(message)
    try:
        common = ntpath.commonpath((normalized_root, normalized_candidate))
    except ValueError as exc:
        raise UnsafeWindowsDatasetDeleteError(message) from exc
    if common != normalized_root or normalized_candidate == normalized_root:
        raise UnsafeWindowsDatasetDeleteError(message)


def _windows_error_code(exc: OSError) -> int | None:
    winerror = getattr(exc, "winerror", None)
    if isinstance(winerror, int):
        return winerror
    return exc.errno if isinstance(exc.errno, int) else None


def _retryable_error(
    operation: str,
    exc: OSError,
) -> WindowsDatasetDeleteRetryableError:
    error_code = _windows_error_code(exc)
    failure_kind = (
        "sharing or lock conflict"
        if error_code in _SHARING_ERROR_CODES
        else "filesystem error"
    )
    return WindowsDatasetDeleteRetryableError(
        error_code,
        f"{operation}: {failure_kind}",
    )


__all__ = [
    "UnsafeWindowsDatasetDeleteError",
    "WindowsDatasetDeleteRetryableError",
    "WindowsDatasetDeleteUnsupportedError",
    "delete_dataset_file_windows",
]
