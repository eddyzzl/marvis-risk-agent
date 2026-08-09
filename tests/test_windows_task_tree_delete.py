from __future__ import annotations

import os

import pytest

from marvis.data.windows_handle_api import CtypesWindowsHandleApi
from marvis.data.windows_safe_delete import _CtypesWindowsDeleteApi
from marvis.data.windows_task_tree_delete import (
    _CtypesWindowsTaskTreeApi,
    UnsafeWindowsTaskTreeDeleteError,
    WindowsTaskTreeDeleteRetryableError,
    WindowsTaskTreeDeleteUnsupportedError,
    delete_task_tree_windows,
)


def test_windows_delete_boundaries_share_one_native_handle_adapter():
    assert _CtypesWindowsDeleteApi is CtypesWindowsHandleApi
    assert _CtypesWindowsTaskTreeApi is CtypesWindowsHandleApi


class FakeWindowsTaskTreeApi:
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
        self.handles[handle] = {"path": path, **entry}
        return handle

    def file_attributes(self, handle):
        return self.handles[handle]["attributes"]

    def file_type(self, handle):
        return self.handles[handle]["type"]

    def final_path(self, handle):
        return self.handles[handle]["final_path"]

    def mark_delete(self, handle, *, flags):
        self.deleted.append((self.handles[handle]["path"], flags))
        error = self.handles[handle].get("delete_error")
        if error is not None:
            raise error

    def close_handle(self, handle):
        self.closed.append(handle)


def _directory(path):
    return {
        "attributes": 0x10,
        "type": 1,
        "final_path": rf"\\?\{path}",
    }


def _file(path):
    return {
        "attributes": 0x80,
        "type": 1,
        "final_path": rf"\\?\{path}",
    }


def test_windows_task_tree_delete_uses_handles_children_before_parent():
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    result = rf"{target}\result.txt"
    nested = rf"{target}\nested"
    artifact = rf"{nested}\artifact.bin"
    api = FakeWindowsTaskTreeApi(
        {
            root: _directory(root),
            target: _directory(target),
            result: _file(result),
            nested: _directory(nested),
            artifact: _file(artifact),
        }
    )
    children = {
        target: ("result.txt", "nested"),
        nested: ("artifact.bin",),
    }

    delete_task_tree_windows(
        target,
        root,
        native_api=api,
        enumerate_children=lambda path: children[path],
    )

    delete_flags = 0x1 | 0x2 | 0x10
    assert api.deleted == [
        (result, delete_flags),
        (artifact, delete_flags),
        (nested, delete_flags),
        (target, delete_flags),
    ]
    assert all(
        call["flags_and_attributes"] == 0x02000000 | 0x00200000
        for call in api.open_calls
    )
    assert all(call["share_mode"] == 0x1 | 0x2 for call in api.open_calls)
    assert [call["path"] for call in api.open_calls] == [
        root,
        target,
        result,
        nested,
        artifact,
    ]


@pytest.mark.skipif(os.name == "nt", reason="non-Windows contract")
def test_windows_task_tree_delete_is_deterministically_unsupported_off_windows():
    with pytest.raises(
        WindowsTaskTreeDeleteUnsupportedError,
        match="requires Windows",
    ):
        delete_task_tree_windows(
            r"C:\workspace\tasks\task-1",
            r"C:\workspace\tasks",
        )


@pytest.mark.parametrize("error_code", [2, 3])
def test_windows_task_tree_delete_is_idempotent_when_target_is_missing(error_code):
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    api = FakeWindowsTaskTreeApi(
        {
            root: _directory(root),
            target: OSError(error_code, "target disappeared"),
        }
    )

    delete_task_tree_windows(
        target,
        root,
        native_api=api,
        enumerate_children=lambda _path: (),
    )

    assert api.deleted == []
    assert api.closed == [100]


@pytest.mark.parametrize(
    ("entry_name", "attributes", "file_type"),
    [
        pytest.param("root", 0x10 | 0x400, 1, id="reparse-root"),
        pytest.param("root", 0x10, 3, id="non-disk-root"),
        pytest.param("target", 0x80, 1, id="file-target"),
        pytest.param("target", 0x10 | 0x400, 1, id="reparse-target"),
    ],
)
def test_windows_task_tree_delete_rejects_unsafe_root_or_target(
    entry_name,
    attributes,
    file_type,
):
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    entries = {root: _directory(root), target: _directory(target)}
    path = root if entry_name == "root" else target
    entries[path] = {
        "attributes": attributes,
        "type": file_type,
        "final_path": rf"\\?\{path}",
    }
    api = FakeWindowsTaskTreeApi(entries)

    with pytest.raises(
        UnsafeWindowsTaskTreeDeleteError,
        match="expected disk object",
    ):
        delete_task_tree_windows(
            target,
            root,
            native_api=api,
            enumerate_children=lambda _path: (),
        )

    assert api.deleted == []


def test_windows_task_tree_delete_reopens_and_rejects_reparse_child():
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    child = rf"{target}\link"
    api = FakeWindowsTaskTreeApi(
        {
            root: _directory(root),
            target: _directory(target),
            child: {
                "attributes": 0x10 | 0x400,
                "type": 1,
                "final_path": rf"\\?\{child}",
            },
        }
    )

    with pytest.raises(
        UnsafeWindowsTaskTreeDeleteError,
        match="expected disk object",
    ):
        delete_task_tree_windows(
            target,
            root,
            native_api=api,
            enumerate_children=lambda path: ("link",) if path == target else (),
        )

    assert [call["path"] for call in api.open_calls] == [root, target, child]
    assert api.deleted == []


def test_windows_task_tree_delete_rejects_child_final_path_escape_after_enumeration():
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    child = rf"{target}\result.txt"
    api = FakeWindowsTaskTreeApi(
        {
            root: _directory(root),
            target: _directory(target),
            child: {
                "attributes": 0x80,
                "type": 1,
                "final_path": r"\\?\C:\workspace\tasks-evil\result.txt",
            },
        }
    )

    with pytest.raises(
        UnsafeWindowsTaskTreeDeleteError,
        match="escaped its verified parent",
    ):
        delete_task_tree_windows(
            target,
            root,
            native_api=api,
            enumerate_children=lambda path: (
                ("result.txt",) if path == target else ()
            ),
        )

    assert api.deleted == []


def test_windows_task_tree_delete_final_path_check_is_case_insensitive():
    root = r"C:\workspace\tasks"
    target = rf"{root}\Task-1"
    api = FakeWindowsTaskTreeApi(
        {
            root: {
                **_directory(root),
                "final_path": r"\\?\C:\WORKSPACE\TASKS",
            },
            target: {
                **_directory(target),
                "final_path": r"\\?\c:\workspace\tasks\task-1",
            },
        }
    )

    delete_task_tree_windows(
        target,
        root,
        native_api=api,
        enumerate_children=lambda _path: (),
    )

    assert [path for path, _flags in api.deleted] == [target]


@pytest.mark.parametrize("unsafe_name", ["..", r"nested\escape", "stream:alt"])
def test_windows_task_tree_delete_rejects_unsafe_enumeration_name(unsafe_name):
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    api = FakeWindowsTaskTreeApi(
        {root: _directory(root), target: _directory(target)}
    )

    with pytest.raises(
        UnsafeWindowsTaskTreeDeleteError,
        match="unsafe child name",
    ):
        delete_task_tree_windows(
            target,
            root,
            native_api=api,
            enumerate_children=lambda _path: (unsafe_name,),
        )

    assert [call["path"] for call in api.open_calls] == [root, target]
    assert api.deleted == []


@pytest.mark.parametrize("error_code", [32, 33])
def test_windows_task_tree_delete_classifies_sharing_errors_as_retryable(error_code):
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    api = FakeWindowsTaskTreeApi(
        {
            root: _directory(root),
            target: OSError(error_code, "directory is in use"),
        }
    )

    with pytest.raises(
        WindowsTaskTreeDeleteRetryableError,
        match="sharing or lock conflict",
    ) as raised:
        delete_task_tree_windows(
            target,
            root,
            native_api=api,
            enumerate_children=lambda _path: (),
        )

    assert raised.value.winerror == error_code
    assert api.deleted == []


def test_windows_task_tree_delete_keeps_other_os_errors_retryable():
    root = r"C:\workspace\tasks"
    target = rf"{root}\task-1"
    api = FakeWindowsTaskTreeApi(
        {
            root: _directory(root),
            target: {
                **_directory(target),
                "delete_error": OSError(145, "directory not empty"),
            },
        }
    )

    with pytest.raises(
        WindowsTaskTreeDeleteRetryableError,
        match="filesystem error",
    ) as raised:
        delete_task_tree_windows(
            target,
            root,
            native_api=api,
            enumerate_children=lambda _path: (),
        )

    assert raised.value.winerror == 145
