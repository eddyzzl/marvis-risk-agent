from __future__ import annotations

import os
from pathlib import Path

import pytest

from marvis.data.windows_safe_delete import (
    UnsafeWindowsDatasetDeleteError,
    WindowsDatasetDeleteRetryableError,
    WindowsDatasetDeleteUnsupportedError,
    delete_dataset_file_windows,
)


class FakeWindowsDeleteApi:
    def __init__(self, entries):
        self.entries = entries
        self.handles = {}
        self.open_calls = []
        self.deleted = []
        self.closed = []
        self._next_handle = 100

    def open_handle(
        self,
        path,
        *,
        desired_access,
        share_mode,
        creation_disposition,
        flags_and_attributes,
    ):
        self.open_calls.append(
            {
                "path": path,
                "desired_access": desired_access,
                "share_mode": share_mode,
                "creation_disposition": creation_disposition,
                "flags_and_attributes": flags_and_attributes,
            }
        )
        entry = self.entries[path]
        if isinstance(entry, BaseException):
            raise entry
        handle = self._next_handle
        self._next_handle += 1
        self.handles[handle] = entry
        return handle

    def file_attributes(self, handle):
        return self.handles[handle]["attributes"]

    def file_type(self, handle):
        return self.handles[handle]["type"]

    def final_path(self, handle):
        return self.handles[handle]["final_path"]

    def mark_delete(self, handle, *, flags):
        self.deleted.append((handle, flags))
        error = self.handles[handle].get("delete_error")
        if error is not None:
            raise error

    def close_handle(self, handle):
        self.closed.append(handle)


@pytest.mark.skipif(os.name == "nt", reason="non-Windows contract")
def test_windows_dataset_delete_is_deterministically_unsupported_off_windows():
    with pytest.raises(
        WindowsDatasetDeleteUnsupportedError,
        match="requires Windows",
    ):
        delete_dataset_file_windows(
            Path(r"C:\workspace\datasets\task\sample.parquet"),
            Path(r"C:\workspace\datasets"),
        )


def test_windows_dataset_delete_uses_verified_delete_handle_for_regular_file():
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\task\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10,
                "type": 1,
                "final_path": rf"\\?\{root}",
            },
            candidate: {
                "attributes": 0x1 | 0x80,
                "type": 1,
                "final_path": rf"\\?\{candidate}",
            },
        }
    )

    delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.open_calls == [
        {
            "path": root,
            "desired_access": 0x80,
            "share_mode": 0x1 | 0x2 | 0x4,
            "creation_disposition": 3,
            "flags_and_attributes": 0x02000000 | 0x00200000,
        },
        {
            "path": candidate,
            "desired_access": 0x00010000 | 0x80 | 0x100,
            "share_mode": 0x1 | 0x2 | 0x4,
            "creation_disposition": 3,
            "flags_and_attributes": 0x02000000 | 0x00200000,
        },
    ]
    assert api.deleted == [(101, 0x1 | 0x2 | 0x10)]
    assert api.closed == [101, 100]


def test_windows_dataset_delete_opens_and_rejects_reparse_root():
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\task\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10 | 0x400,
                "type": 1,
                "final_path": rf"\\?\{root}",
            },
        }
    )

    with pytest.raises(
        UnsafeWindowsDatasetDeleteError,
        match="root handle is not a real directory",
    ):
        delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.open_calls[0]["flags_and_attributes"] == 0x02000000 | 0x00200000
    assert api.deleted == []
    assert api.closed == [100]


@pytest.mark.parametrize("error_code", [2, 3])
def test_windows_dataset_delete_is_a_noop_when_target_is_missing(error_code):
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\gone\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10,
                "type": 1,
                "final_path": rf"\\?\{root}",
            },
            candidate: OSError(error_code, "file disappeared"),
        }
    )

    delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.deleted == []
    assert api.closed == [100]


@pytest.mark.parametrize(
    ("attributes", "file_type"),
    [
        pytest.param(0x80 | 0x400, 1, id="reparse-point"),
        pytest.param(0x10, 1, id="directory"),
        pytest.param(0x80, 3, id="non-disk-object"),
    ],
)
def test_windows_dataset_delete_rejects_non_regular_objects(
    attributes,
    file_type,
):
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\task\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10,
                "type": 1,
                "final_path": rf"\\?\{root}",
            },
            candidate: {
                "attributes": attributes,
                "type": file_type,
                "final_path": rf"\\?\{candidate}",
            },
        }
    )

    with pytest.raises(
        UnsafeWindowsDatasetDeleteError,
        match="regular non-reparse file",
    ):
        delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.deleted == []
    assert api.closed == [101, 100]


def test_windows_dataset_delete_rejects_final_handle_path_at_sibling_prefix():
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\task\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10,
                "type": 1,
                "final_path": rf"\\?\{root}",
            },
            candidate: {
                "attributes": 0x80,
                "type": 1,
                "final_path": (r"\\?\C:\workspace\datasets-evil\task\sample.parquet"),
            },
        }
    )

    with pytest.raises(
        UnsafeWindowsDatasetDeleteError,
        match="escaped the trusted datasets root",
    ):
        delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.deleted == []


def test_windows_dataset_delete_final_path_check_is_case_insensitive():
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\task\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10,
                "type": 1,
                "final_path": r"\\?\C:\WORKSPACE\DATASETS",
            },
            candidate: {
                "attributes": 0x80,
                "type": 1,
                "final_path": (r"\\?\c:\workspace\datasets\TASK\sample.parquet"),
            },
        }
    )

    delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.deleted == [(101, 0x1 | 0x2 | 0x10)]


@pytest.mark.parametrize("error_code", [32, 33])
def test_windows_dataset_delete_classifies_sharing_and_lock_errors_as_retryable(
    error_code,
):
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\task\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10,
                "type": 1,
                "final_path": rf"\\?\{root}",
            },
            candidate: OSError(error_code, "file is in use"),
        }
    )

    with pytest.raises(
        WindowsDatasetDeleteRetryableError,
        match="sharing or lock conflict",
    ) as raised:
        delete_dataset_file_windows(candidate, root, native_api=api)

    assert raised.value.winerror == error_code
    assert api.deleted == []
    assert api.closed == [100]


def test_windows_dataset_delete_keeps_handle_delete_lock_failure_retryable():
    root = r"C:\workspace\datasets"
    candidate = rf"{root}\task\sample.parquet"
    api = FakeWindowsDeleteApi(
        {
            root: {
                "attributes": 0x10,
                "type": 1,
                "final_path": rf"\\?\{root}",
            },
            candidate: {
                "attributes": 0x80,
                "type": 1,
                "final_path": rf"\\?\{candidate}",
                "delete_error": OSError(33, "mapped file is locked"),
            },
        }
    )

    with pytest.raises(
        WindowsDatasetDeleteRetryableError,
        match="sharing or lock conflict",
    ):
        delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.closed == [101, 100]


def test_windows_dataset_delete_rejects_lexical_sibling_before_opening_handle():
    root = r"C:\workspace\datasets"
    candidate = r"C:\workspace\datasets-evil\sample.parquet"
    api = FakeWindowsDeleteApi({})

    with pytest.raises(
        UnsafeWindowsDatasetDeleteError,
        match="outside the configured datasets root",
    ):
        delete_dataset_file_windows(candidate, root, native_api=api)

    assert api.open_calls == []
