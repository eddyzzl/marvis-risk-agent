from pathlib import Path

import marvis.kernel_probe as kernel_probe


def test_kernel_probe_connection_key_is_utf8_decodable():
    key = kernel_probe._connection_key()

    assert key.decode("utf-8")


def test_kernel_probe_environment_uses_target_runtime_and_ignores_host_python_paths(
    tmp_path: Path,
    monkeypatch,
):
    runtime = tmp_path / "validation-runtime"
    python = runtime / "python.exe"
    monkeypatch.setenv("PATH", r"C:\host-runtime")
    monkeypatch.setenv("PYTHONPATH", r"C:\host-python-packages")
    monkeypatch.setenv("PYTHONHOME", r"C:\host-python")

    env = kernel_probe._kernel_environment(python)

    assert env["PATH"].split(";")[:3] == [
        str(runtime),
        str(runtime / "Scripts"),
        str(runtime / "Library" / "bin"),
    ]
    assert env["CONDA_PREFIX"] == str(runtime)
    assert env["CONDA_DEFAULT_ENV"] == str(runtime)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env


def test_kernel_probe_reports_child_stderr_when_kernel_dies(
    tmp_path: Path,
    monkeypatch,
):
    events = []

    class FakeClient:
        def __init__(self, *, connection_file):
            events.append(("client", connection_file))

        def load_connection_file(self):
            events.append("load")

        def start_channels(self):
            events.append("start")

        def wait_for_ready(self, *, timeout):
            events.append(("wait", timeout))
            raise RuntimeError("Kernel died before replying to kernel_info")

        def stop_channels(self):
            events.append("stop")

    class FakeProcess:
        returncode = 3221225781

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return "", "ImportError: DLL load failed while importing _zmq"

    monkeypatch.setattr(kernel_probe, "BlockingKernelClient", FakeClient)
    monkeypatch.setattr(
        kernel_probe.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        kernel_probe,
        "write_connection_file",
        lambda path, **kwargs: (path, {}),
    )

    result = kernel_probe.probe_python_kernel(
        tmp_path / "validation-runtime" / "python.exe",
        timeout=7,
    )

    assert result.ok is False
    assert result.returncode == 3221225781
    assert "Kernel died before replying to kernel_info" in result.message
    assert "DLL load failed while importing _zmq" in result.message
    assert events[-1] == "stop"


def test_kernel_probe_succeeds_after_kernel_info_handshake(tmp_path: Path, monkeypatch):
    class FakeClient:
        def __init__(self, *, connection_file):
            pass

        def load_connection_file(self):
            pass

        def start_channels(self):
            pass

        def wait_for_ready(self, *, timeout):
            pass

        def stop_channels(self):
            pass

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(kernel_probe, "BlockingKernelClient", FakeClient)
    monkeypatch.setattr(
        kernel_probe.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        kernel_probe,
        "write_connection_file",
        lambda path, **kwargs: (path, {}),
    )

    result = kernel_probe.probe_python_kernel(
        tmp_path / "validation-runtime" / "python.exe",
        timeout=7,
    )

    assert result.ok is True
    assert result.message == "Jupyter kernel_info handshake succeeded"
