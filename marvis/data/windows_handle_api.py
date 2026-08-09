"""Shared ``kernel32`` handle adapter for governed Windows deletion.

The dataset-file and task-tree boundaries deliberately keep their policy,
path validation, retry taxonomy, and share-mode choices separate.  This module
owns only the native call surface so both policies receive the same Win32
compatibility and readonly-file fallback behavior.
"""

from __future__ import annotations


_EXTENDED_DISPOSITION_UNSUPPORTED_CODES = frozenset({1, 50, 87, 120})


class CtypesWindowsHandleApi:
    """Thin ``kernel32`` adapter kept behind injectable policy boundaries."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = (
                ("file_attributes", wintypes.DWORD),
                ("reparse_tag", wintypes.DWORD),
            )

        class FileBasicInfo(ctypes.Structure):
            _fields_ = (
                ("creation_time", ctypes.c_longlong),
                ("last_access_time", ctypes.c_longlong),
                ("last_write_time", ctypes.c_longlong),
                ("change_time", ctypes.c_longlong),
                ("file_attributes", wintypes.DWORD),
            )

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = (("delete_file", ctypes.c_ubyte),)

        class FileDispositionInfoEx(ctypes.Structure):
            _fields_ = (("flags", wintypes.DWORD),)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._file_attribute_tag_info = FileAttributeTagInfo
        self._file_basic_info = FileBasicInfo
        self._file_disposition_info = FileDispositionInfo
        self._file_disposition_info_ex = FileDispositionInfoEx

        self._create_file = kernel32.CreateFileW
        self._create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        self._create_file.restype = wintypes.HANDLE

        self._get_file_information = kernel32.GetFileInformationByHandleEx
        self._get_file_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self._get_file_information.restype = wintypes.BOOL

        self._get_file_type = kernel32.GetFileType
        self._get_file_type.argtypes = (wintypes.HANDLE,)
        self._get_file_type.restype = wintypes.DWORD

        self._get_final_path = kernel32.GetFinalPathNameByHandleW
        self._get_final_path.argtypes = (
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        self._get_final_path.restype = wintypes.DWORD

        self._set_file_information = kernel32.SetFileInformationByHandle
        self._set_file_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self._set_file_information.restype = wintypes.BOOL

        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL

    def open_handle(
        self,
        path: str,
        *,
        desired_access: int,
        share_mode: int,
        creation_disposition: int,
        flags_and_attributes: int,
    ) -> int:
        handle = self._create_file(
            path,
            desired_access,
            share_mode,
            None,
            creation_disposition,
            flags_and_attributes,
            None,
        )
        handle_value = self._ctypes.cast(handle, self._ctypes.c_void_p).value
        invalid_handle = self._ctypes.c_void_p(-1).value
        if handle_value is None or handle_value == invalid_handle:
            raise self._last_error("CreateFileW")
        return int(handle_value)

    def file_attributes(self, handle: int) -> int:
        information = self._file_attribute_tag_info()
        if not self._get_file_information(
            self._wintypes.HANDLE(handle),
            9,  # FileAttributeTagInfo
            self._ctypes.byref(information),
            self._ctypes.sizeof(information),
        ):
            raise self._last_error("GetFileInformationByHandleEx")
        return int(information.file_attributes)

    def file_type(self, handle: int) -> int:
        self._ctypes.set_last_error(0)
        file_type = int(self._get_file_type(self._wintypes.HANDLE(handle)))
        if file_type == 0 and self._ctypes.get_last_error():
            raise self._last_error("GetFileType")
        return file_type

    def final_path(self, handle: int) -> str:
        capacity = 512
        while capacity <= 32768:
            buffer = self._ctypes.create_unicode_buffer(capacity)
            length = int(
                self._get_final_path(
                    self._wintypes.HANDLE(handle),
                    buffer,
                    capacity,
                    0,  # FILE_NAME_NORMALIZED | VOLUME_NAME_DOS
                )
            )
            if length == 0:
                raise self._last_error("GetFinalPathNameByHandleW")
            if length < capacity:
                return str(buffer.value)
            capacity = length + 1
        raise OSError(206, "GetFinalPathNameByHandleW returned an invalid path")

    def mark_delete(self, handle: int, *, flags: int) -> None:
        extended_information = self._file_disposition_info_ex(flags=flags)
        if self._set_file_information(
            self._wintypes.HANDLE(handle),
            21,  # FileDispositionInfoEx
            self._ctypes.byref(extended_information),
            self._ctypes.sizeof(extended_information),
        ):
            return
        extended_error = self._last_error("SetFileInformationByHandle")
        if _windows_error_code(extended_error) not in (
            _EXTENDED_DISPOSITION_UNSUPPORTED_CODES
        ):
            raise extended_error
        self._mark_delete_legacy(handle)

    def close_handle(self, handle: int) -> None:
        if not self._close_handle(self._wintypes.HANDLE(handle)):
            raise self._last_error("CloseHandle")

    def _mark_delete_legacy(self, handle: int) -> None:
        basic_information = self._file_basic_info()
        if not self._get_file_information(
            self._wintypes.HANDLE(handle),
            0,  # FileBasicInfo
            self._ctypes.byref(basic_information),
            self._ctypes.sizeof(basic_information),
        ):
            raise self._last_error("GetFileInformationByHandleEx")

        original_attributes = int(basic_information.file_attributes)
        cleared_readonly = bool(original_attributes & 0x1)
        if cleared_readonly:
            basic_information.file_attributes = original_attributes & ~0x1
            if not self._set_file_information(
                self._wintypes.HANDLE(handle),
                0,  # FileBasicInfo
                self._ctypes.byref(basic_information),
                self._ctypes.sizeof(basic_information),
            ):
                raise self._last_error("SetFileInformationByHandle")

        disposition = self._file_disposition_info(delete_file=1)
        if self._set_file_information(
            self._wintypes.HANDLE(handle),
            4,  # FileDispositionInfo
            self._ctypes.byref(disposition),
            self._ctypes.sizeof(disposition),
        ):
            return
        deletion_error = self._last_error("SetFileInformationByHandle")
        if cleared_readonly:
            basic_information.file_attributes = original_attributes
            self._set_file_information(
                self._wintypes.HANDLE(handle),
                0,  # FileBasicInfo
                self._ctypes.byref(basic_information),
                self._ctypes.sizeof(basic_information),
            )
        raise deletion_error

    def _last_error(self, operation: str) -> OSError:
        error_code = int(self._ctypes.get_last_error())
        return OSError(error_code, f"{operation} failed")


def _windows_error_code(exc: OSError) -> int | None:
    winerror = getattr(exc, "winerror", None)
    if isinstance(winerror, int):
        return winerror
    errno = getattr(exc, "errno", None)
    return errno if isinstance(errno, int) else None


__all__ = ["CtypesWindowsHandleApi"]
