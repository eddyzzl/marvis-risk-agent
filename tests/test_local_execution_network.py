import errno
import asyncio
import json
import socket
import subprocess
import sys
import threading
import shutil

import nbformat
import pytest

from marvis.local_execution import LocalExecutionIsolationError, loopback_only_command
from marvis.notebooks import run_notebook


_NATIVE_PROBE = """
import ctypes, errno, fcntl, json, os, socket, struct, subprocess, sys
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
assert fd >= 0
fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
addr = struct.pack('BBH4s8x', 16, socket.AF_INET, socket.htons(443), socket.inet_aton('203.0.113.1'))
rc = libc.connect(fd, addr, len(addr))
error = ctypes.get_errno()
libc.close(fd)
assert rc == -1 and error in (errno.EPERM, errno.EACCES), (rc, error)
print(json.dumps({'native_denied': error}))
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS native socket enforcement")
def test_os_boundary_denies_native_descendants_and_preserves_loopback():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(15)
        received = []

        def serve():
            conn, _ = server.accept()
            with conn:
                received.append(conn.recv(16))
                conn.sendall(b"local-ok")

        thread = threading.Thread(target=serve)
        thread.start()
        code = _NATIVE_PROBE + f"""
child = subprocess.run([sys.executable, '-c', {_NATIVE_PROBE!r}], capture_output=True, text=True, timeout=10)
assert child.returncode == 0, child.stderr
with socket.create_connection(('127.0.0.1', {server.getsockname()[1]}), timeout=5) as conn:
    conn.sendall(b'local-only')
    assert conn.recv(16) == b'local-ok'
print('descendant-and-loopback-ok')
"""
        result = subprocess.run(loopback_only_command([sys.executable, "-c", code]),
            capture_output=True, text=True, timeout=20)
        thread.join(16)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout.splitlines()[0])["native_denied"] in {errno.EPERM, errno.EACCES}
        assert "descendant-and-loopback-ok" in result.stdout
        assert received == [b"local-only"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS real Jupyter process trees")
@pytest.mark.parametrize("isolated", [False, True])
def test_notebook_kernel_and_nested_native_child_are_confined(tmp_path, isolated):
    code = _NATIVE_PROBE + f"""
child = subprocess.run([sys.executable, '-c', {_NATIVE_PROBE!r}], capture_output=True, text=True, timeout=10)
assert child.returncode == 0, child.stderr
print('notebook-compute-ok', sum(range(10)))
"""
    source = tmp_path / "synthetic.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), source)
    result = run_notebook(source, tmp_path / "executed.ipynb", tmp_path / "run.log",
        timeout=45, isolated=isolated)
    assert result.succeeded, (result.error_name, result.error_value)
    executed = nbformat.read(tmp_path / "executed.ipynb", as_version=4)
    text = "".join(item.get("text", "") for item in executed.cells[0].outputs)
    assert "notebook-compute-ok 45" in text


def test_unsupported_os_fails_before_launch(monkeypatch):
    monkeypatch.setattr("marvis.local_execution.sys.platform", "unsupported")
    with pytest.raises(LocalExecutionIsolationError, match="未启动"):
        loopback_only_command([sys.executable, "-c", "print('must not execute')"])


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS kernel boundary")
@pytest.mark.parametrize("provisioner", [
    {"provisioner_name": "remote-untrusted"},
    {"provisioner_name": "local-provisioner", "config": {"custom": True}},
])
def test_custom_kernel_provisioner_is_rejected_before_any_hook(provisioner, monkeypatch):
    from jupyter_client.kernelspec import KernelSpec
    from marvis.local_execution import notebook_kernel_manager_class

    manager = notebook_kernel_manager_class()()
    manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher"],
        metadata={"kernel_provisioner": provisioner})
    calls = []
    monkeypatch.setattr("jupyter_client.provisioning.factory.KernelProvisionerFactory.create_provisioner_instance",
        lambda *args, **kwargs: calls.append(True))
    with pytest.raises(LocalExecutionIsolationError, match="本地 kernel"):
        asyncio.run(manager._async_pre_start_kernel())
    assert calls == []
    assert manager.provisioner is None


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("java"), reason="macOS and local Java required")
def test_java_descendant_retains_network_boundary_and_local_service(tmp_path):
    source = tmp_path / "LocalNetworkProbe.java"
    source.write_text('''
import java.net.*;
import java.io.*;
class LocalNetworkProbe {
    public static void main(String[] args) throws Exception {
        try (Socket remote = new Socket()) {
            remote.connect(new InetSocketAddress("203.0.113.1", 443), 2000);
            throw new AssertionError("external connect was allowed");
        } catch (SocketException ex) {
            if (!ex.getMessage().toLowerCase().contains("permitted") &&
                !ex.getMessage().toLowerCase().contains("permission")) throw ex;
        }
        try (Socket local = new Socket("127.0.0.1", Integer.parseInt(args[0]))) {
            local.getOutputStream().write(7);
            if (local.getInputStream().read() != 11) throw new AssertionError("loopback failed");
        }
        System.out.println("java-local-ok-public-denied");
    }
}
''')
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(30)
        received = []

        def serve():
            conn, _ = server.accept()
            with conn:
                received.append(conn.recv(1))
                conn.sendall(bytes([11]))

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        command = [shutil.which("java"), str(source), str(server.getsockname()[1])]
        child_code = "import subprocess,sys; sys.exit(subprocess.call(" + repr(command) + "))"
        result = subprocess.run(loopback_only_command([sys.executable, "-c", child_code]),
            capture_output=True, text=True, timeout=35)
        thread.join(31)
        assert result.returncode == 0, result.stderr
        assert "java-local-ok-public-denied" in result.stdout
        assert received == [bytes([7])]
