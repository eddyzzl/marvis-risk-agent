"""Recognise loopback API requests made from MARVIS-launched loopback sandboxes.

A ``loopback_only`` Notebook process tree may still open loopback sockets, so
without this check it could call MARVIS's own API as a trusted local client
(and use LLM, web-search or plugin endpoints to move data off the machine).

Design: every loopback launch (``local_execution.execution_command``) creates
a fresh marker file plus a control file in a per-user marker directory and
adds two rules to that launch's Seatbelt profile: the marker may not be read,
and nothing in the marker directory may be written. The sandbox is inherited
by every descendant and cannot be removed, even by a double-forked ``setsid``
orphan or a copied/renamed binary, so the rule travels with the whole tree.

A process belongs to a live MARVIS launch iff ``sandbox_check`` reports that
it may not read that launch's marker but may read the control file next to
it. Unsandboxed processes may read both; other sandboxes (browsers, agent
shells, a user's own ``sandbox-exec`` curl) either may read both or may read
neither. No executable path is trusted.

Cost: with no marker present (no loopback launch alive) the check returns
immediately. While launches are alive it asks the kernel one sandbox question
per process and inspects sockets only of matching processes.

Rejected alternatives (measured): Seatbelt port-specific denies are enforced
for some port numbers and ignored for others; any token the zero-configuration
index page hands to the browser is equally available to the Notebook.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import secrets
import sys
import time

_SANDBOX_LIB = "/usr/lib/libsandbox.1.dylib"
_SANDBOX_FILTER_PATH = 1
_CONTROL_SUFFIX = ".control"
_MARKER_PREFIX = "launch-"
# A marker that no live process carries is removed after this grace period
# (the launch may not have exec'd yet when the API checks).
_MARKER_GRACE_SECONDS = 30.0
_sandbox_check = None


def marker_directory() -> Path:
    override = os.environ.get("MARVIS_LOOPBACK_MARKER_DIR")
    base = Path(override) if override else Path.home() / "Library" / "Caches" / "MARVIS" / "loopback-launches"
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = base.resolve()
    if any(char in str(resolved) for char in '"\\\n'):
        raise OSError("loopback marker directory path is not representable in a sandbox profile")
    return resolved


def new_launch_marker() -> tuple[Path, Path]:
    """Create one launch's marker and control file; return (directory, marker)."""
    directory = marker_directory()
    marker = directory / f"{_MARKER_PREFIX}{os.getpid()}-{secrets.token_hex(8)}"
    for path in (marker, Path(str(marker) + _CONTROL_SUFFIX)):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    return directory, marker


def launch_marker_rules() -> str:
    """Profile rules marking a new launch; empty inside an already-marked launch.

    A launch started from within a marked sandbox (the isolated Notebook
    worker starting its kernel) cannot write a new marker; it repeats the
    inherited marker's rules instead. Any other failure fails closed.
    """
    try:
        directory, marker = new_launch_marker()
    except OSError:
        # Nested sandbox_apply is refused unless the inner profile keeps the
        # outer restrictions, so repeat the inherited marker rules verbatim.
        inherited = inherited_marker_rules()
        if inherited:
            return inherited
        raise
    return _marker_rules(directory, marker)


def inherited_marker_rules() -> str:
    """Rules of the launch markers this process already carries (else empty)."""
    if sys.platform != "darwin":
        return ""
    return "".join(_marker_rules(item.parent, item) for item in _active_markers()
                   if in_launch_sandbox(os.getpid(), item))


def _marker_rules(directory: Path, marker: Path) -> str:
    return (f'(deny file-read* (literal "{marker}"))\n'
            f'(deny file-write* (subpath "{directory}"))\n')


def _check(pid: int, path: str) -> int:
    global _sandbox_check
    if _sandbox_check is None:
        function = ctypes.CDLL(_SANDBOX_LIB).sandbox_check
        function.restype = ctypes.c_int
        # Fixed arguments only: the path is passed variadically (arm64 ABI).
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        _sandbox_check = function
    return _sandbox_check(int(pid), b"file-read-data", _SANDBOX_FILTER_PATH, ctypes.c_char_p(path.encode()))


def _active_markers() -> list[Path]:
    try:
        directory = marker_directory()
    except OSError:
        return []
    return sorted(path for path in directory.glob(_MARKER_PREFIX + "*")
                  if not path.name.endswith(_CONTROL_SUFFIX))


def in_launch_sandbox(pid: int, marker: Path) -> bool:
    return _check(pid, str(marker)) == 1 and _check(pid, str(marker) + _CONTROL_SUFFIX) == 0


def notebook_sandbox_peer(client_port: int | None, server_port: int | None) -> bool:
    """True when a process of a live MARVIS loopback launch owns the client socket."""
    if sys.platform != "darwin" or not client_port or not server_port:
        return False
    markers = _active_markers()
    if not markers:
        return False
    import psutil

    own = os.getpid()
    members: dict[Path, list[int]] = {marker: [] for marker in markers}
    for pid in psutil.pids():
        if pid in (0, 1, own):
            continue
        for marker in markers:
            if in_launch_sandbox(pid, marker):
                members[marker].append(pid)
    now = time.time()
    for marker, pids in members.items():
        if not pids:
            try:
                if now - marker.stat().st_mtime > _MARKER_GRACE_SECONDS:
                    marker.unlink(missing_ok=True)
                    Path(str(marker) + _CONTROL_SUFFIX).unlink(missing_ok=True)
            except OSError:
                pass
    for pid in {pid for pids in members.values() for pid in pids}:
        try:
            connections = psutil.Process(pid).net_connections(kind="tcp")
        except (psutil.Error, OSError):
            continue
        if any(connection.laddr and connection.laddr.port == client_port
               and connection.raddr and connection.raddr.port == server_port
               for connection in connections):
            return True
    return False
