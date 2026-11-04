import errno
import asyncio
import json
import os
from pathlib import Path
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
grandchild_code = 'import subprocess,sys; sys.exit(subprocess.call(' + repr([sys.executable, '-c', {_NATIVE_PROBE!r}]) + '))'
child = subprocess.run([sys.executable, '-c', grandchild_code], capture_output=True, text=True, timeout=10)
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
        timeout=45, isolated=isolated, network_mode="loopback_only")
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
def test_java_descendant_retains_network_boundary_and_local_service(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Xmx128m -Dmarvis.native.probe=retained")
    source = tmp_path / "LocalNetworkProbe.java"
    source.write_text('''
import java.net.*;
import java.io.*;
class LocalNetworkProbe {
    public static void main(String[] args) throws Exception {
        if (!"true".equals(System.getProperty("java.net.preferIPv4Stack")))
            throw new AssertionError("loopback JVM compatibility option missing");
        if (!"retained".equals(System.getProperty("marvis.native.probe")))
            throw new AssertionError("existing JVM options lost");
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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS loopback launch environment")
@pytest.mark.parametrize("options", [
    "", '-Xmx64m -Dmarvis.probe="kept value"',
    "-Xmx64m -Djava.net.preferIPv4Stack=true",
    "-Djava.net.preferIPv4Stack=false -Xmx64m",
])
def test_loopback_jvm_options_are_inherited_without_changing_host_or_other_options(monkeypatch, options):
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", options)
    code = "import os; print(os.environ['JAVA_TOOL_OPTIONS'])"
    child = "import subprocess,sys; sys.exit(subprocess.call(" + repr([sys.executable, "-c", code]) + "))"
    command = loopback_only_command([sys.executable, "-c", child])
    # Environment values, which can contain secrets, must never enter argv.
    if options:
        assert options not in " ".join(command)
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    actual = result.stdout.strip()
    assert actual.startswith(options)
    assert actual.endswith("-Djava.net.preferIPv4Stack=true")
    assert actual.count("-Djava.net.preferIPv4Stack=true") == 1
    assert os.environ["JAVA_TOOL_OPTIONS"] == options


@pytest.mark.parametrize("options", [None, "-Xmx64m -Djava.net.preferIPv4Stack=false"])
def test_trusted_launch_preserves_java_options(monkeypatch, options):
    from marvis.local_execution import execution_command

    if options is None:
        monkeypatch.delenv("JAVA_TOOL_OPTIONS", raising=False)
    else:
        monkeypatch.setenv("JAVA_TOOL_OPTIONS", options)
    command = [sys.executable, "-c", "import json,os; print(json.dumps(os.getenv('JAVA_TOOL_OPTIONS')))"]
    result = subprocess.run(execution_command(command), capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == options


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS native TCP/UDP enforcement")
@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
@pytest.mark.parametrize("socktype,port", [(socket.SOCK_STREAM, 443), (socket.SOCK_DGRAM, 53)])
def test_native_ipv4_ipv6_tcp_udp_and_dns_sockets_are_denied(family, socktype, port):
    code = f'''
import ctypes, errno, fcntl, os, socket, struct
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.socket({family}, {socktype}, 0)
assert fd >= 0
fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
if {family} == socket.AF_INET:
    addr = struct.pack('BBH4s8x', 16, {family}, socket.htons({port}), socket.inet_aton('203.0.113.1'))
else:
    addr = struct.pack('BBHI16sI', 28, {family}, socket.htons({port}), 0,
                       socket.inet_pton(socket.AF_INET6, '2001:db8::1'), 0)
if {socktype} == socket.SOCK_STREAM:
    rc = libc.connect(fd, addr, len(addr))
else:
    rc = libc.sendto(fd, b'dns-query', 9, 0, addr, len(addr))
error = ctypes.get_errno()
libc.close(fd)
assert rc == -1 and error in (errno.EPERM, errno.EACCES), (rc, error)
'''
    result = subprocess.run(loopback_only_command([sys.executable, "-c", code]),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS loopback transport")
@pytest.mark.parametrize("family,host", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")])
@pytest.mark.parametrize("socktype", [socket.SOCK_STREAM, socket.SOCK_DGRAM])
def test_loopback_ipv4_ipv6_tcp_and_udp_remain_available(family, host, socktype):
    with socket.socket(family, socktype) as server:
        server.bind((host, 0))
        server.settimeout(15)
        if socktype == socket.SOCK_STREAM:
            server.listen(1)
        received = []

        def serve():
            if socktype == socket.SOCK_STREAM:
                conn, _ = server.accept()
                with conn:
                    received.append(conn.recv(16))
                    conn.sendall(b"local-ok")
            else:
                data, address = server.recvfrom(16)
                received.append(data)
                server.sendto(b"local-ok", address)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        code = f'''
import socket
with socket.socket({family}, {socktype}) as client:
    client.settimeout(5)
    client.connect(({host!r}, {server.getsockname()[1]}))
    client.send(b'local-only')
    assert client.recv(16) == b'local-ok'
'''
        result = subprocess.run(loopback_only_command([sys.executable, "-c", code]),
                                capture_output=True, text=True, timeout=10)
        thread.join(16)
        assert result.returncode == 0, result.stderr
        assert received == [b"local-only"]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_trusted_mode_preserves_command_on_platforms_without_network_backend(monkeypatch, platform):
    from marvis.local_execution import execution_capability, execution_command

    monkeypatch.setattr("marvis.local_execution.sys.platform", platform)
    command = [sys.executable, "-c", "print('trusted')"]
    assert execution_command(command) == command
    assert execution_capability().available
    assert not execution_capability().network_restricted
    capability = execution_capability("loopback_only")
    assert capability.status == "unsupported"
    assert not capability.available
    assert not capability.network_restricted


@pytest.mark.parametrize("isolated", [False, True])
def test_unavailable_network_mode_reports_blocked_and_never_launches(tmp_path, monkeypatch, isolated):
    monkeypatch.setattr("marvis.local_execution.sys.platform", "unsupported")
    calls = []
    monkeypatch.setattr("marvis.notebooks.subprocess.Popen", lambda *a, **kw: calls.append("worker"))
    monkeypatch.setattr("marvis.notebooks.NotebookClient", lambda *a, **kw: calls.append("kernel"))
    source = tmp_path / "blocked.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise AssertionError('executed')")]), source)
    result = run_notebook(source, tmp_path / "executed.ipynb", tmp_path / "run.log",
                          progress_path=tmp_path / "progress.json", isolated=isolated,
                          network_mode="loopback_only")
    assert calls == []
    assert not result.succeeded
    assert result.error_name == "LocalExecutionIsolationError"
    assert "未启动" in result.error_value
    assert result.execution_policy["mode"] == "loopback_only"
    assert result.execution_policy["state"] == "blocked"
    assert not result.execution_policy["network_restricted"]
    assert json.loads((tmp_path / "progress.json").read_text())["execution_policy"] == result.execution_policy


def test_missing_or_non_executable_isolation_tool_has_actionable_preflight(tmp_path, monkeypatch):
    from marvis.local_execution import execution_capability, preflight_execution

    monkeypatch.setattr("marvis.local_execution.sys.platform", "darwin")
    tool = tmp_path / "missing-sandbox"
    monkeypatch.setattr("marvis.local_execution._SANDBOX_EXEC", tool)
    assert execution_capability("loopback_only").status == "missing_tool"
    tool.write_text("not executable")
    assert execution_capability("loopback_only").status == "missing_tool"
    with pytest.raises(LocalExecutionIsolationError, match="未启动"):
        preflight_execution("loopback_only")


def test_failed_policy_startup_never_claims_restricted_network(tmp_path, monkeypatch):
    from marvis.local_execution import execution_capability, preflight_execution

    tool = tmp_path / "sandbox-exec"
    tool.write_text("placeholder; startup is mocked\n")
    tool.chmod(0o700)
    monkeypatch.setattr("marvis.local_execution._SANDBOX_EXEC", tool)
    monkeypatch.setattr("marvis.local_execution.sys.platform", "darwin")
    calls = []

    def fail_startup(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=1)

    monkeypatch.setattr("marvis.local_execution.subprocess.run", fail_startup)
    capability = execution_capability("loopback_only", probe=True)
    assert capability.status == "startup_failed"
    assert capability.startup_checked
    assert not capability.available
    assert not capability.network_restricted
    assert len(calls) == 1
    assert calls[0][0] == str(tool)
    with pytest.raises(LocalExecutionIsolationError, match="启动预检失败"):
        preflight_execution("loopback_only")
    assert len(calls) == 2


@pytest.mark.parametrize("isolated", [False, True])
def test_default_trusted_notebook_records_actual_mode_and_succeeds(tmp_path, isolated):
    from marvis.notebooks import notebook_run_result_from_dict, notebook_run_result_to_dict

    source = tmp_path / "trusted.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(sum(range(10)))")]), source)
    result = run_notebook(source, tmp_path / "executed.ipynb", tmp_path / "run.log",
                          progress_path=tmp_path / "progress.json", timeout=45, isolated=isolated)
    assert result.succeeded, (result.error_name, result.error_value)
    assert result.execution_policy["mode"] == "trusted_local"
    assert result.execution_policy["backend"] == "none"
    assert not result.execution_policy["network_restricted"]
    assert result.execution_policy["subprocess_worker"] is isolated
    assert notebook_run_result_from_dict(notebook_run_result_to_dict(result)) == result


@pytest.mark.pmml_runtime
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS native Notebook/Java/PMML workflow")
@pytest.mark.parametrize("isolated", [False, True])
def test_loopback_mode_preserves_real_v1_contract_pmml_metrics_and_reports(tmp_path, monkeypatch, isolated):
    from docx import Document
    import pandas as pd

    from marvis.db import TaskRepository, init_db
    from marvis.domain import TaskCreate, TaskStatus
    from marvis.execution_environment import ExecutionEnvironmentSettings, save_execution_environment
    from marvis.pipeline import PipelineSettings, run_pipeline

    project = tmp_path / "materials"
    project.mkdir()
    sample = pd.DataFrame([
        {"x1": (i + 1) / 41, "x2": 0.0, "y": int(i >= 20), "split": split,
         "apply_month": "202601" if split == "train" else "202602"}
        for split in ("train", "test", "oot") for i in range(40)
    ])
    sample.to_csv(project / "sample.csv", index=False)
    shutil.copy(Path(__file__).parent / "fixtures" / "min_lr.pmml", project / "model.pmml")
    pd.DataFrame({"特征名": ["x1", "x2"], "类别": ["基础", "基础"], "importance": [0.8, 0.2]}).to_csv(
        project / "dictionary.csv", index=False)
    code = _NATIVE_PROBE + '''
import pandas as pd
import numpy as np
RMC_SAMPLE_DF = pd.read_csv('sample.csv')
RMC_TARGET_COL = 'y'
RMC_ALGORITHM = 'lr'
RMC_SPLIT_COL = 'split'
RMC_TIME_COL = 'apply_month'
class LocalModel:
    def score(self, frame):
        return 1 / (1 + np.exp(-(frame['x1'] - frame['x2'])))
local_model = LocalModel()
def RMC_SCORE_FN(frame):
    return local_model.score(frame)
RMC_FEATURE_IMPORTANCE = pd.DataFrame({'feature': ['x1', 'x2'], 'importance': [0.8, 0.2]})
RMC_MODEL_PARAMS = {'synthetic_fixture': True}
'''
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), project / "model.ipynb")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(TaskCreate(model_name="synthetic LR", model_version="test", validator="qa",
                                      source_dir=str(project), run_mode="manual"))
    with repo.transaction() as conn:
        conn.execute("UPDATE tasks SET validation_workflow_version = 1 WHERE id = ?", (task.id,))
    template_path = tmp_path / "template.docx"
    template = Document()
    template.add_paragraph("Synthetic validation report")
    template.save(template_path)
    save_execution_environment(workspace, ExecutionEnvironmentSettings(notebook_network_mode="loopback_only"))
    monkeypatch.setenv("MARVIS_ALLOW_LEGACY_LIVE_NOTEBOOK_EXECUTION", "1")
    settings = PipelineSettings(workspace=workspace, db_path=db_path, report_template_path=template_path,
                                random_sample_size=12, notebook_isolated_execution=isolated,
                                allow_legacy_live_notebook_execution=not isolated,
                                notebook_network_mode="loopback_only")
    run_pipeline(task_id=task.id, settings=settings)
    assert repo.get_task(task.id).status in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}
    task_dir = workspace / "tasks" / task.id
    evidence = json.loads((task_dir / "execution" / "notebook_execution.json").read_text())
    assert evidence["mode"] == "loopback_only"
    assert evidence["backend"] == "macos_seatbelt"
    assert evidence["state"] == "completed"
    assert evidence["network_restricted"]
    reproducibility = json.loads((task_dir / "outputs" / "reproducibility_result.json").read_text())
    assert reproducibility["summary"]["mismatch_count"] == 0
    assert reproducibility["summary"]["match_count"] == 12
    results = json.loads((task_dir / "outputs" / "validation_results.json").read_text())
    assert results["effectiveness"]["overall"]
    assert (task_dir / "outputs" / "validation.xlsx").stat().st_size > 0
    assert (task_dir / "outputs" / "validation_report.docx").stat().st_size > 0


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS network backend failure states")
@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("code,timeout,memory_limit,error_name", [
    ("import time; time.sleep(5)", 1, None, "CellTimeoutError"),
    ("import os; os._exit(17)", 15, None, "DeadKernelError"),
    ("import time; time.sleep(5)", 15, 1, "NotebookResourceLimitExceeded"),
])
def test_loopback_notebook_timeout_crash_and_resource_limit_remain_visible(
    tmp_path, isolated, code, timeout, memory_limit, error_name,
):
    source = tmp_path / "failure.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), source)
    result = run_notebook(source, tmp_path / "executed.ipynb", tmp_path / "run.log",
                          timeout=timeout, isolated=isolated, network_mode="loopback_only",
                          memory_limit_mb=memory_limit, resource_poll_interval_seconds=0.05)
    assert not result.succeeded
    assert result.error_name == error_name
    assert result.execution_policy["mode"] == "loopback_only"
    assert result.execution_policy["state"] == "failed"
    assert result.execution_policy["next_action"]
    assert (tmp_path / "executed.ipynb").is_file()
    assert error_name in (tmp_path / "run.log").read_text()
    if memory_limit is not None:
        assert result.resource_usage["memory_limit_exceeded"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS restricted network dependency")
@pytest.mark.parametrize("isolated", [False, True])
def test_notebook_external_network_dependency_fails_with_local_preparation_action(tmp_path, isolated):
    source = tmp_path / "network-dependency.ipynb"
    code = "import socket; socket.create_connection(('203.0.113.1', 443), timeout=1)"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), source)
    result = run_notebook(source, tmp_path / "executed.ipynb", tmp_path / "run.log",
                          timeout=15, isolated=isolated, network_mode="loopback_only")
    assert not result.succeeded
    assert result.failed_cell_index == 0
    assert result.error_name == "PermissionError"
    assert result.execution_policy["state"] == "failed"
    assert "外部依赖" in result.execution_policy["next_action"]
    assert "本地材料" in result.execution_policy["next_action"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS kernel startup rejection cleanup")
def test_rejected_provisioner_closes_without_deferred_atexit_failure(tmp_path):
    code = f'''
from pathlib import Path
import sys
import nbformat
from jupyter_client.kernelspec import KernelSpec
from marvis.local_execution import notebook_kernel_manager_class
from marvis.notebooks import NotebookExecutionSession
root = Path({str(tmp_path)!r})
source = root / 'source.ipynb'
nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise AssertionError('executed')")]), source)
session = NotebookExecutionSession(notebook_path=source, executed_path=root/'executed.ipynb',
                                   log_path=root/'run.log', network_mode='loopback_only')
manager = notebook_kernel_manager_class()()
manager._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher'],
    metadata={{'kernel_provisioner': {{'provisioner_name': 'remote-untrusted'}}}})
session.client.km = manager
result = session.execute_notebook(keep_alive=False)
assert not result.succeeded and result.error_name == 'LocalExecutionIsolationError', result
assert result.failed_cell_index is None
session.close()
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "Exception ignored in atexit callback" not in result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS AF_UNIX enforcement")
def test_loopback_mode_denies_af_unix_connect(tmp_path):
    import tempfile

    # AF_UNIX paths are limited to ~104 bytes; pytest tmp paths can exceed it.
    directory = Path(tempfile.mkdtemp(prefix="mx-"))
    path = directory / "s.sock"
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path))
        server.listen(1)
        code = f'''
import errno, socket
with socket.socket(socket.AF_UNIX) as client:
    try:
        client.connect({str(path)!r})
    except OSError as exc:
        assert exc.errno in (errno.EPERM, errno.EACCES), exc
        print("unix-denied")
    else:
        raise AssertionError("AF_UNIX connect was allowed")
'''
        result = subprocess.run(loopback_only_command([sys.executable, "-c", code]),
                                capture_output=True, text=True, timeout=10)
    shutil.rmtree(directory, ignore_errors=True)
    assert result.returncode == 0, result.stderr
    assert "unix-denied" in result.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS system resolver under the loopback policy")
def test_loopback_mode_system_resolver_resolves_only_local_names():
    code = '''
import socket
assert {item[4][0] for item in socket.getaddrinfo("localhost", 80)} & {"127.0.0.1", "::1"}
try:
    socket.getaddrinfo("example.com", 443)
except socket.gaierror:
    print("external-name-unresolved")
else:
    raise AssertionError("external name resolved inside the loopback policy")
'''
    result = subprocess.run(loopback_only_command([sys.executable, "-c", code]),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "external-name-unresolved" in result.stdout


@pytest.fixture
def live_marvis_api(tmp_path, monkeypatch):
    """A real loopback MARVIS server, so the peer is an actual socket."""
    import time
    import urllib.request

    import uvicorn

    from marvis.app import create_app

    monkeypatch.delenv("MARVIS_LOCAL_TOKEN", raising=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(tmp_path / "ws"), host="127.0.0.1",
                                           port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=2).read()
            break
        except OSError:
            time.sleep(0.05)
    yield base
    server.should_exit = True
    thread.join(10)


_API_PROBE = '''
import sys, urllib.error, urllib.request
try:
    urllib.request.urlopen(sys.argv[1] + "/api/tasks", timeout=10)
except urllib.error.HTTPError as exc:
    print("status", exc.code)
else:
    print("status", 200)
'''


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS loopback Notebook boundary")
def test_loopback_sandboxed_process_cannot_use_marvis_api_but_browser_and_cli_can(live_marvis_api):
    import urllib.request

    # A normal (unsandboxed) local client — browser or CLI tool — keeps working.
    with urllib.request.urlopen(live_marvis_api + "/api/tasks", timeout=10) as response:
        assert response.status == 200
    index = urllib.request.urlopen(live_marvis_api + "/", timeout=10).read().decode()
    assert "<html" in index.lower()
    plain = subprocess.run([sys.executable, "-c", _API_PROBE, live_marvis_api],
                           capture_output=True, text=True, timeout=20)
    assert "status 200" in plain.stdout, plain.stderr

    # The same request from inside the loopback policy (and a nested child) is refused.
    sandboxed = subprocess.run(loopback_only_command([sys.executable, "-c", _API_PROBE, live_marvis_api]),
                               capture_output=True, text=True, timeout=30)
    assert "status 403" in sandboxed.stdout, sandboxed.stderr
    nested = "import subprocess,sys; sys.exit(subprocess.call([sys.executable, '-c', sys.argv[1], sys.argv[2]]))"
    child = subprocess.run(loopback_only_command([sys.executable, "-c", nested, _API_PROBE, live_marvis_api]),
                           capture_output=True, text=True, timeout=30)
    assert "status 403" in child.stdout, child.stderr
    # Static assets carry no private state and stay reachable.
    static = subprocess.run(loopback_only_command([sys.executable, "-c",
        "import sys,urllib.request; print(urllib.request.urlopen(sys.argv[1] + '/static/app.js').status)",
        live_marvis_api]), capture_output=True, text=True, timeout=30)
    assert static.stdout.strip() == "200", static.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS real loopback Notebook kernel")
@pytest.mark.parametrize("isolated", [False, True])
def test_loopback_notebook_kernel_request_is_rejected_and_token_withheld(
    tmp_path, live_marvis_api, monkeypatch, isolated,
):
    monkeypatch.setenv("MARVIS_LOCAL_TOKEN", "TEST_ONLY_KERNEL_TOKEN")
    code = (
        "import os, urllib.error, urllib.request\n"
        f"base = {live_marvis_api!r}\n"
        "try:\n"
        "    urllib.request.urlopen(base + '/api/tasks', timeout=10)\n"
        "    status = 200\n"
        "except urllib.error.HTTPError as exc:\n"
        "    status = exc.code\n"
        "print('api-status', status, 'token-visible', 'MARVIS_LOCAL_TOKEN' in os.environ)\n"
    )
    source = tmp_path / "api-probe.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), source)
    result = run_notebook(source, tmp_path / "executed.ipynb", tmp_path / "run.log",
                          timeout=60, isolated=isolated, network_mode="loopback_only")
    assert result.succeeded, (result.error_name, result.error_value)
    executed = nbformat.read(tmp_path / "executed.ipynb", as_version=4)
    text = "".join(item.get("text", "") for item in executed.cells[0].outputs)
    assert "api-status 403 token-visible False" in text


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS loopback Notebook boundary")
def test_copied_binary_in_fake_app_bundle_inside_the_sandbox_is_still_refused(live_marvis_api, tmp_path):
    # Review B1: no executable path is trusted; the inherited sandbox decides.
    code = f'''
import os, shutil, subprocess, sys
bundle = {str(tmp_path)!r} + "/Fake.app/Contents/MacOS"
os.makedirs(bundle, exist_ok=True)
copy = os.path.join(bundle, "py")
shutil.copy2(os.path.realpath(sys.executable), copy)
# Relocated standalone Python cannot infer its original standard-library root.
# Keep the interpreter runnable so this probes the inherited network boundary,
# rather than passing/failing before urllib can make the protected API request.
env = {{**os.environ, "PYTHONHOME": sys.base_prefix}}
raise SystemExit(subprocess.call([copy, "-c", sys.argv[1], sys.argv[2]], env=env))
'''
    result = subprocess.run(loopback_only_command([sys.executable, "-c", code, _API_PROBE, live_marvis_api]),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "status 403" in result.stdout, (result.stdout, result.stderr)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS loopback Notebook boundary")
def test_orphaned_setsid_descendant_of_a_launch_is_still_refused(live_marvis_api):
    code = '''
import os, sys, time
read, write = os.pipe()
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        os.close(read)
        import urllib.error, urllib.request
        try:
            urllib.request.urlopen(sys.argv[1] + "/api/tasks", timeout=10)
            status = 200
        except urllib.error.HTTPError as exc:
            status = exc.code
        os.write(write, str(status).encode())
        os._exit(0)
    os._exit(0)
os.close(write)
print("status", os.read(read, 8).decode())
'''
    result = subprocess.run(loopback_only_command([sys.executable, "-c", code, live_marvis_api]),
                            capture_output=True, text=True, timeout=60)
    assert "status 403" in result.stdout, (result.stdout, result.stderr)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec")
def test_sandboxed_client_not_launched_by_marvis_is_allowed(live_marvis_api):
    # Review B2: a user's own sandboxed CLI (or an agent shell) is not a Notebook.
    profile = '(version 1)\n(allow default)\n(deny network-outbound (remote ip "*:443"))\n'
    result = subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, "/usr/bin/curl", "-s", "-o", "/dev/null",
                             "-w", "%{http_code}", live_marvis_api + "/api/tasks"],
                            capture_output=True, text=True, timeout=30)
    assert result.stdout.strip() == "200", result.stderr


_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


@pytest.mark.skipif(sys.platform != "darwin" or not Path(_CHROME).exists(), reason="macOS with Google Chrome")
def test_real_browser_is_allowed_while_a_loopback_launch_is_alive(live_marvis_api, tmp_path):
    # A live launch makes the API inspect peers; the browser must still pass.
    holder = subprocess.Popen(loopback_only_command([sys.executable, "-c", "import time; time.sleep(60)"]))
    try:
        result = subprocess.run([_CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={tmp_path}",
                                 "--dump-dom", live_marvis_api + "/api/tasks"],
                                capture_output=True, text=True, timeout=90)
    finally:
        holder.kill()
        holder.wait()
    # The task list (an empty JSON array) renders; a refusal would show the 403 detail.
    assert "<pre>[" in result.stdout, result.stderr[-500:]
    assert "Notebook" not in result.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS loopback launch markers")
def test_no_live_launch_means_no_peer_inspection(monkeypatch, tmp_path):
    # Review B3: with no loopback launch alive the check does no process walk.
    import marvis.local_peer as local_peer

    monkeypatch.setenv("MARVIS_LOOPBACK_MARKER_DIR", str(tmp_path / "markers"))
    monkeypatch.setattr("psutil.pids", lambda: pytest.fail("no process walk without a live launch"))
    assert local_peer.notebook_sandbox_peer(50000, 8000) is False
