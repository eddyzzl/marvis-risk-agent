"""Actual macOS OS syscalls and process execution, independent of Python guards."""

import ctypes
import json
import os
from pathlib import Path
import socket
import sys
import time

import pytest

from marvis.plugins.native_isolation import (
    capability,
    configured_mode,
    NativeIsolationError,
)
from marvis.plugins.manifest import ToolRef
from test_governed_capabilities import rt as rt, _archive, _propose, _confirm, _invoke

native = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="native macOS backend unavailable; no Linux/Windows verification",
)


def _enable(rt):
    response = rt[1].put(
        "/api/settings/execution-environment",
        json={"worker_isolation_mode": "os_sandbox"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["validation"]["worker_capability"]["startup_checked"]


@native
def test_actual_setting_and_signed_native_worker_compute_isolated(rt):
    _confirm(rt, _propose(rt, _archive(rt)))
    _enable(rt)
    result = _invoke(rt)
    assert result.ok, (result.error, result.stderr_tail)
    assert result.output == {"version": "1.0.0", "message": "synthetic"}
    proof = result.resource_limits["worker_isolation"]
    assert proof["enforced"] and proof["execution_boundary"] == "input_only_compute"
    assert not proof["host_workspace_readable"] and not proof["ipc_allowed"]
    assert proof["startup_checked"] and not proof["fork_allowed"]
    builtin = rt[0].state.tool_runner.invoke(
        ToolRef("_sample", "echo"), {"message": "trusted"}, task_id="synthetic"
    )
    assert builtin.ok, builtin.error
    assert (
        builtin.resource_limits["worker_isolation"]["execution_boundary"]
        == "trusted_builtin_host"
    )
    assert not builtin.resource_limits["worker_isolation"]["enforced"]
    assert builtin.resource_limits["worker_isolation"]["host_workspace_readable"]


_SOURCE = """
def echo(inputs, ctx):
    import ctypes, json, resource
    c = ctypes.CDLL(None, use_errno=True)
    raw = json.loads(inputs['message'])
    out = {}
    for key, path in raw['files'].items():
        fd = c.open(path.encode(), 0)
        out[key] = [fd, ctypes.get_errno()]
        if fd >= 0: c.close(fd)
    for key, path in raw['writes'].items():
        fd = c.open(path.encode(), 0x601, 0o600)
        out[key] = [fd, ctypes.get_errno()]
        if fd >= 0: c.close(fd)
    fd = c.open(__file__.encode(), 0)
    out['own_package_readable'] = fd >= 0
    if fd >= 0: c.close(fd)
    for key, args in raw['network'].items():
        family, kind, address = args
        fd = c.socket(family, kind, 0)
        pointer = ctypes.create_string_buffer(bytes(address))
        if kind == 2:
            r = c.sendto(fd, b'native', 6, 0, pointer, len(address))
        else:
            r = c.connect(fd, pointer, len(address))
        out[key] = [r, ctypes.get_errno()]
        c.close(fd)
    forked = c.fork()
    out['native_fork'] = [forked, ctypes.get_errno()]
    argv = (ctypes.c_char_p * 2)(b'/usr/bin/true', None)
    envp = (ctypes.c_char_p * 1)(None)
    pid = ctypes.c_int()
    spawned = c.posix_spawn(ctypes.byref(pid), b'/usr/bin/true', None, None, argv, envp)
    out['native_spawn'] = [spawned, pid.value]
    fd = c.shm_open(raw['shm'].encode(), 0, 0o600)
    out['posix_shm'] = [fd, ctypes.get_errno()]
    if fd >= 0: c.close(fd)
    port = ctypes.c_uint.in_dll(c, 'bootstrap_port').value
    target = ctypes.c_uint()
    out['mach_lookup'] = c.bootstrap_look_up(port, b'com.apple.system.notification_center', ctypes.byref(target))
    out['cpu_limits'] = list(resource.getrlimit(resource.RLIMIT_CPU))
    return {'version':'1.0.0', 'message':json.dumps(out)}
"""


@native
def test_native_ctypes_files_tcp4_tcp6_udp_unix_mach_shm_fork_spawn_denied(rt):
    import struct
    import uuid

    # Every target contains synthetic bytes. The admin path is only attempted
    # natively under confinement; no credential body is loaded or printed.
    host = rt[4] / "host-private"
    host.write_bytes(b"synthetic private marker")
    blind = rt[0].state.settings.workspace / ".blind_oot" / "synthetic"
    blind.parent.mkdir()
    blind.write_bytes(b"synthetic sealed marker")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix = Path("/private/tmp") / ("marvis-native-" + uuid.uuid4().hex[:18] + ".sock")
    sock.bind(str(unix))
    sock.listen()
    sock.settimeout(0.1)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(0.1)

    def ipv4(port):
        return list(
            bytes([16, 2])
            + struct.pack("!H", port)
            + socket.inet_aton("127.0.0.1")
            + bytes(8)
        )

    ipv6 = list(
        bytes([28, 30, 0, 9])
        + bytes(4)
        + socket.inet_pton(socket.AF_INET6, "::1")
        + bytes(4)
    )
    unixaddr = list(
        bytes([2 + len(str(unix).encode()) + 1, 1]) + str(unix).encode() + b"\0"
    )
    lib = ctypes.CDLL(None, use_errno=True)
    shm = "/m-" + uuid.uuid4().hex[:12]
    shared_fd = lib.shm_open(shm.encode(), os.O_CREAT | os.O_RDWR, 0o600)
    assert shared_fd >= 0
    try:
        target = ctypes.c_uint()
        assert (
            lib.bootstrap_look_up(
                ctypes.c_uint.in_dll(lib, "bootstrap_port").value,
                b"com.apple.system.notification_center",
                ctypes.byref(target),
            )
            == 0
        )
        _confirm(rt, _propose(rt, _archive(rt, source=_SOURCE)))
        _enable(rt)
        message = json.dumps(
            {
                "files": {
                    "host_read": str(host),
                    "db_read": str(rt[0].state.settings.db_path),
                    "admin_read": str(rt[0].state.settings.plugin_admin_token_path),
                    "blind_read": str(blind),
                },
                "writes": {
                    "host_write": str(host),
                    "host_new_write": str(rt[4] / "escape"),
                },
                "network": {
                    "tcp4": [2, 1, ipv4(9)],
                    "tcp6": [30, 1, ipv6],
                    "udp4": [2, 2, ipv4(udp.getsockname()[1])],
                    "udp6": [30, 2, ipv6],
                    "unix": [1, 1, unixaddr],
                },
                "shm": shm,
            }
        )
        result = rt[0].state.tool_runner.invoke(
            ToolRef("signed_echo", "echo"), {"message": message}, task_id="synthetic"
        )
        assert result.ok, (result.error, result.stderr_tail)
        out = json.loads(result.output["message"])
        for key in (
            "host_read",
            "db_read",
            "admin_read",
            "blind_read",
            "host_write",
            "host_new_write",
            "tcp4",
            "tcp6",
            "udp4",
            "udp6",
            "unix",
            "posix_shm",
            "native_fork",
        ):
            assert out[key] == [-1, 1], (key, out[key])
        assert out["native_spawn"][0] == 1 and out["native_spawn"][1] == 0
        assert out["mach_lookup"] != 0 and out["own_package_readable"]
        assert out["cpu_limits"][0] == out["cpu_limits"][1]
        assert host.read_bytes() == b"synthetic private marker"
        assert not (rt[4] / "escape").exists()
        with pytest.raises(socket.timeout):
            udp.recvfrom(20)
        with pytest.raises(socket.timeout):
            sock.accept()
    finally:
        lib.close(shared_fd)
        lib.shm_unlink(shm.encode())
        sock.close()
        udp.close()
        unix.unlink(missing_ok=True)


@native
@pytest.mark.parametrize(
    "permission",
    ["process:spawn", "network:optional", "read:dataset", "write:artifact"],
)
def test_declared_unmountable_needs_refuse_before_worker(rt, monkeypatch, permission):
    _confirm(rt, _propose(rt, _archive(rt, permissions=(permission,))))
    _enable(rt)
    monkeypatch.setattr(
        "marvis.plugins.runner._run_worker",
        lambda *_a, **_kw: pytest.fail("unsupported capability cannot start"),
    )
    result = _invoke(rt)
    assert not result.ok and result.error_kind == "isolation"
    assert result.error == "worker_isolation_needs_not_supported"


def test_corrupt_worker_mode_never_silently_downgrades(rt):
    path = rt[0].state.settings.workspace / "settings" / "execution_environment.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("{")
    with pytest.raises(NativeIsolationError):
        configured_mode(rt[0].state.settings.workspace)


def test_unverified_platform_is_explicit_and_does_not_persist_strong_mode(
    rt, monkeypatch
):
    monkeypatch.setattr("marvis.plugins.native_isolation.sys.platform", "linux")
    description = capability("os_sandbox", probe=True)
    assert not description["available"] and description["status"] == "unsupported"
    response = rt[1].put(
        "/api/settings/execution-environment",
        json={"worker_isolation_mode": "os_sandbox"},
    )
    assert response.status_code == 422
    assert configured_mode(rt[0].state.settings.workspace) == "trusted_local"


@native
def test_native_exec_same_pid_inherits_os_policy_and_hard_limits(rt):
    from marvis.plugins.contracts import PROTOCOL_VERSION, WORKER_RESULT_SENTINEL

    host = rt[4] / "exec-private"
    host.write_text("synthetic secret")
    script = (
        "import ctypes,json,os,resource; c=ctypes.CDLL(None,use_errno=True); "
        + "fd=c.open("
        + repr(str(host).encode())
        + ",0); e=ctypes.get_errno(); "
        + "s=c.socket(2,1,0); a=ctypes.create_string_buffer(bytes([16,2,0,9,127,0,0,1])+bytes(8)); "
        + "r=c.connect(s,a,16); n=ctypes.get_errno(); f=c.fork(); fe=ctypes.get_errno(); "
        + "proof={'file':[fd,e],'tcp':[r,n],'fork':[f,fe],'cpu':list(resource.getrlimit(resource.RLIMIT_CPU))}; "
        + "print("
        + repr(WORKER_RESULT_SENTINEL)
        + "+json.dumps({'ok':True,'worker_protocol_version':"
        + str(PROTOCOL_VERSION)
        + ",'output':{'version':'1.0.0','message':json.dumps(proof)}}))"
    )
    source = """
def echo(inputs,ctx):
    import ctypes,sys
    c=ctypes.CDLL(None,use_errno=True)
    argv=(ctypes.c_char_p*6)(sys.executable.encode(),b'-I',b'-B',b'-c',inputs['message'].encode(),None)
    env=(ctypes.c_char_p*2)(b'LANG=en_US.UTF-8',None)
    c.execve(sys.executable.encode(),argv,env)
    raise RuntimeError('exec was unexpectedly refused')
"""
    _confirm(rt, _propose(rt, _archive(rt, source=source)))
    _enable(rt)
    result = rt[0].state.tool_runner.invoke(
        ToolRef("signed_echo", "echo"), {"message": script}, task_id="synthetic"
    )
    assert result.ok, (result.error, result.stderr_tail)
    out = json.loads(result.output["message"])
    for key in ("file", "tcp", "fork"):
        assert out[key] == [-1, 1]
    assert out["cpu"][0] == out["cpu"][1] and out["cpu"][0] > 0
    assert result.resource_limits["worker_isolation"]["enforced"]


@native
@pytest.mark.parametrize("resource_case", ["native_rss", "native_stdout"])
def test_native_memory_or_raw_stdout_cannot_escape_host_budgets(rt, resource_case):
    if resource_case == "native_rss":
        source = """
def echo(inputs,ctx):
    import ctypes
    c=ctypes.CDLL(None); c.malloc.restype=ctypes.c_void_p
    c.malloc.argtypes=[ctypes.c_size_t]; c.memset.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
    pointer=c.malloc(256*1024*1024); c.memset(pointer,1,256*1024*1024)
    c.usleep(5000000)
    return {'version':'1.0.0','message':'must be killed'}
"""
    else:
        source = """
def echo(inputs,ctx):
    import ctypes
    c=ctypes.CDLL(None); buffer=ctypes.create_string_buffer(b'x'*65536)
    while True: c.write(1,buffer,65536)
"""
    _confirm(rt, _propose(rt, _archive(rt, source=source)))
    _enable(rt)
    rt[0].state.tool_runner._rss_memory_limit_mb = 96
    start = time.monotonic()
    result = _invoke(rt)
    assert not result.ok and result.error_kind == "resource_limit", (
        result.error,
        result.stderr_tail,
    )
    assert time.monotonic() - start < 8
    assert result.resource_limits["worker_isolation"]["enforced"]
    if resource_case == "native_rss":
        assert result.resource_limits["memory_limit_exceeded"]
        assert not result.resource_limits["monitor_degraded"]
    else:
        assert result.error_detail["kind"] == "worker_output_limit"
        assert result.resource_limits["captured_bytes"] <= 8 * 1024 * 1024


@native
def test_raw_launch_cpu_hard_limit_and_timeout_reap_same_pid_exec(rt):
    from marvis.plugins.contracts import PROTOCOL_VERSION
    from marvis.plugins.runner import _run_worker, WorkerTimeoutExpired

    source = rt[4] / "cpu_probe.py"
    source.write_text(
        "def run(inputs,ctx):\n import ctypes\n c=ctypes.CDLL(None)\n while True: c.getpid()\n"
    )
    job = {
        "protocol_version": PROTOCOL_VERSION,
        "module_path": str(source),
        "entrypoint": "run",
        "inputs": {},
        "task_id": "synthetic",
        "seed": None,
        "workspace": str(rt[0].state.settings.workspace),
        "datasets_root": str(rt[0].state.settings.datasets_dir),
        "builtin": False,
        "side_effects": [],
        "plugin_paths": [],
        "execution_profile": "standard",
        "worker_isolation_mode": "os_sandbox",
        "cpu_limit_seconds": 1,
        "file_size_limit_mb": 1,
        "memory_limit_mb": 256,
    }
    start = time.monotonic()
    completed = _run_worker(sys.executable, job, timeout=10, rss_limit_mb=256)
    assert completed.returncode in (-9, -24), (completed.returncode, completed.stderr)
    assert time.monotonic() - start < 6
    assert completed.isolation_capability["enforced"]
    job["cpu_limit_seconds"] = 100
    source.write_text(
        "def run(inputs,ctx):\n import ctypes\n ctypes.CDLL(None).usleep(10000000)\n return {}\n"
    )
    start = time.monotonic()
    with pytest.raises(WorkerTimeoutExpired) as info:
        _run_worker(sys.executable, job, timeout=1, rss_limit_mb=256)
    assert time.monotonic() - start < 6
    assert info.value.isolation_capability["enforced"]


@native
def test_os_rss_disabled_and_oversized_input_fail_before_user_tool(rt):
    _confirm(rt, _propose(rt, _archive(rt)))
    _enable(rt)
    rt[0].state.tool_runner._rss_memory_limit_mb = None
    result = _invoke(rt)
    assert not result.ok and result.error == "worker_isolation_rss_monitor_required"
    rt[0].state.tool_runner._rss_memory_limit_mb = 256
    result = rt[0].state.tool_runner.invoke(
        ToolRef("signed_echo", "echo"),
        {"message": "x" * (2 * 1024 * 1024)},
        task_id="synthetic",
    )
    assert not result.ok and result.error == "worker_isolation_input_budget_or_shape"
