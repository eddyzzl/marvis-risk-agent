import logging
from pathlib import Path

import pytest

import marvis.execution_environment as execution_environment
from marvis.execution_environment import (
    ExecutionEnvironmentSettings,
    detect_execution_environment_options,
    execution_settings_unreadable,
    load_execution_environment,
    save_execution_environment,
    validate_execution_environment,
)
from marvis.local_execution import SETTINGS_UNREADABLE, SETTINGS_UNREADABLE_MESSAGE


def test_execution_environment_settings_round_trip(tmp_path: Path):
    settings = ExecutionEnvironmentSettings(
        execution_mode="jupyter_kernel",
        kernel_name="marvis-kernel",
        conda_env_name="",
        python_executable="",
    )

    save_execution_environment(tmp_path, settings)

    assert load_execution_environment(tmp_path) == settings


def test_execution_environment_defaults_to_python3_kernel(tmp_path: Path):
    settings = load_execution_environment(tmp_path)

    assert settings.execution_mode == "jupyter_kernel"
    assert settings.kernel_name == "python3"


def test_notebook_memory_limit_defaults_to_none_and_round_trips(tmp_path: Path):
    # Default is no cap (large datasets); a positive value is a real ceiling.
    assert load_execution_environment(tmp_path).notebook_memory_limit_mb is None

    settings = ExecutionEnvironmentSettings(notebook_memory_limit_mb=16384)
    save_execution_environment(tmp_path, settings)
    assert load_execution_environment(tmp_path).notebook_memory_limit_mb == 16384


def test_notebook_memory_limit_treats_zero_and_junk_as_no_cap(tmp_path: Path):
    normalize = execution_environment._normalize_notebook_memory_limit_mb
    assert normalize(None) is None
    assert normalize(0) is None
    assert normalize(-1) is None
    assert normalize("nonsense") is None
    assert normalize("8192") == 8192
    assert normalize(8192) == 8192


def test_validate_jupyter_kernel_reports_missing_kernel(monkeypatch):
    monkeypatch.setattr(
        "marvis.execution_environment.available_kernel_names",
        lambda: ["python3"],
    )
    settings = ExecutionEnvironmentSettings(
        execution_mode="jupyter_kernel",
        kernel_name="missing-kernel",
        conda_env_name="",
        python_executable="",
    )

    result = validate_execution_environment(settings)

    assert result.ok is False
    assert "missing-kernel" in result.message


def test_validate_jupyter_kernel_reports_available_kernel(monkeypatch):
    monkeypatch.setattr(
        "marvis.execution_environment.available_kernel_names",
        lambda: ["python3", "marvis-kernel"],
    )
    settings = ExecutionEnvironmentSettings(
        execution_mode="jupyter_kernel",
        kernel_name="marvis-kernel",
        conda_env_name="",
        python_executable="",
    )

    result = validate_execution_environment(settings)

    assert result.ok is True
    assert result.kernel_name == "marvis-kernel"


def test_detect_execution_environment_options_includes_registered_conda_kernel(
    tmp_path: Path,
    monkeypatch,
):
    conda_env = tmp_path / "miniconda3" / "envs" / "marvis"
    conda_python = conda_env / "bin" / "python"
    conda_python.parent.mkdir(parents=True)
    conda_python.write_text("", encoding="utf-8")
    current_python = tmp_path / "current" / "bin" / "python"
    current_python.parent.mkdir(parents=True)
    current_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "marvis.execution_environment.available_kernel_specs",
        lambda: {
            "python3": {
                "display_name": "Python 3",
                "argv": [str(current_python), "-m", "ipykernel_launcher"],
            },
            "marvis": {
                "display_name": "Python marvis",
                "argv": [str(conda_python), "-m", "ipykernel_launcher"],
            },
        },
    )
    monkeypatch.setattr(
        "marvis.execution_environment._conda_environment_paths",
        lambda: [conda_env],
    )
    monkeypatch.setattr(
        "marvis.execution_environment.sys.executable",
        str(current_python),
    )

    options = detect_execution_environment_options()
    conda_options = [
        option for option in options if option.execution_mode == "conda_env"
    ]

    assert any(option.id == "kernel:marvis" for option in options)
    assert conda_options
    assert conda_options[0].conda_env_name == "marvis"
    assert conda_options[0].kernel_name == "marvis"
    assert conda_options[0].python_executable == str(conda_python)
    assert conda_options[0].available is True


def test_detect_execution_environment_options_marks_conda_without_kernel_unavailable(
    tmp_path: Path,
    monkeypatch,
):
    conda_env = tmp_path / "miniconda3" / "envs" / "analysis"
    conda_python = conda_env / "bin" / "python"
    conda_python.parent.mkdir(parents=True)
    conda_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "marvis.execution_environment.available_kernel_specs",
        lambda: {"python3": {"display_name": "Python 3", "argv": []}},
    )
    monkeypatch.setattr(
        "marvis.execution_environment._conda_environment_paths",
        lambda: [conda_env],
    )

    options = detect_execution_environment_options()
    conda_option = next(
        option for option in options if option.conda_env_name == "analysis"
    )

    assert conda_option.available is False
    assert "Kernel" in conda_option.note


def test_python_path_for_conda_environment_uses_windows_candidate(tmp_path: Path, monkeypatch):
    env_path = tmp_path / "envs" / "marvis"
    windows_python = env_path / "python.exe"
    windows_python.parent.mkdir(parents=True)
    windows_python.write_text("", encoding="utf-8")
    (env_path / "bin").mkdir()
    (env_path / "bin" / "python").write_text("", encoding="utf-8")
    monkeypatch.setattr(execution_environment.sys, "platform", "win32")

    assert execution_environment._python_path_for_environment(env_path) == windows_python


def test_python_path_for_conda_environment_uses_posix_candidate(tmp_path: Path, monkeypatch):
    env_path = tmp_path / "envs" / "marvis"
    posix_python = env_path / "bin" / "python"
    posix_python.parent.mkdir(parents=True)
    posix_python.write_text("", encoding="utf-8")
    (env_path / "python.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(execution_environment.sys, "platform", "darwin")

    assert execution_environment._python_path_for_environment(env_path) == posix_python


def test_save_execution_environment_writes_atomically(tmp_path: Path, monkeypatch):
    calls = []
    original_replace = Path.replace

    def tracking_replace(self, target):
        calls.append((str(self), str(target)))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)

    settings = ExecutionEnvironmentSettings(
        execution_mode="jupyter_kernel",
        kernel_name="marvis-kernel",
        conda_env_name="",
        python_executable="",
    )

    save_execution_environment(tmp_path, settings)

    settings_path = tmp_path / "settings" / "execution_environment.json"
    assert calls, "expected save_execution_environment to use an atomic replace"
    temp_source, final_target = calls[-1]
    assert final_target == str(settings_path)
    assert temp_source != str(settings_path)
    assert settings_path.exists()
    # No leftover temp file after a successful atomic write.
    leftover_tmp_files = list(settings_path.parent.glob(".*.tmp"))
    assert leftover_tmp_files == []


def _write_unreadable_settings(tmp_path: Path, kind: str) -> Path:
    settings_path = tmp_path / "settings" / "execution_environment.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "truncated":
        settings_path.write_text('{"notebook_network_mode": "loopback_on', encoding="utf-8")
    elif kind == "empty":
        settings_path.write_text("", encoding="utf-8")
    elif kind == "not_object":
        settings_path.write_text('["loopback_only"]', encoding="utf-8")
    elif kind == "not_utf8":
        settings_path.write_bytes(b"\xff\xfe{")
    elif kind == "directory":
        settings_path.mkdir()
    return settings_path


@pytest.mark.parametrize("kind", ["truncated", "empty", "not_object", "not_utf8", "directory"])
def test_unreadable_settings_fail_closed_instead_of_trusted_local(tmp_path: Path, caplog, kind):
    _write_unreadable_settings(tmp_path, kind)

    with caplog.at_level(logging.WARNING, logger="marvis.execution_environment"):
        settings = load_execution_environment(tmp_path)

    assert settings.notebook_network_mode == SETTINGS_UNREADABLE
    assert settings.notebook_network_mode != "trusted_local"
    assert execution_settings_unreadable(settings)
    assert any("fails closed" in record.message for record in caplog.records)

    validation = validate_execution_environment(settings)
    assert validation.ok is False
    assert validation.message == SETTINGS_UNREADABLE_MESSAGE
    assert validation.kernel_name == ""
    assert validation.network_capability["status"] == SETTINGS_UNREADABLE
    assert validation.network_capability["available"] is False
    assert validation.network_capability["network_restricted"] is False
    assert "重新选择 Python 环境和 Notebook 网络模式" in validation.network_capability["next_action"]


def test_unreadable_settings_refuse_every_notebook_start_path(tmp_path: Path):
    from types import SimpleNamespace

    from marvis.api_stage_helpers import pipeline_settings_from_settings
    from marvis.local_execution import (
        LocalExecutionIsolationError, execution_capability, execution_command,
        notebook_kernel_manager_class, preflight_execution,
    )
    from marvis.pipeline_errors import PipelineError
    from marvis.pipeline_io import _execution_kernel_name

    _write_unreadable_settings(tmp_path, "truncated")
    for start in (
        lambda: execution_capability(SETTINGS_UNREADABLE),
        lambda: preflight_execution(SETTINGS_UNREADABLE),
        lambda: execution_command(["python", "-V"], SETTINGS_UNREADABLE),
        lambda: notebook_kernel_manager_class(SETTINGS_UNREADABLE),
    ):
        with pytest.raises(LocalExecutionIsolationError, match="无法读取"):
            start()

    task = SimpleNamespace(feature_columns=[])
    app_settings = SimpleNamespace(workspace=tmp_path, db_path=tmp_path / "marvis.sqlite",
                                   report_template_path=None)
    pipeline_settings = pipeline_settings_from_settings(app_settings, task, None)
    assert pipeline_settings.notebook_network_mode == SETTINGS_UNREADABLE
    with pytest.raises(PipelineError, match="无法读取.*重新选择 Python 环境和 Notebook 网络模式"):
        _execution_kernel_name(pipeline_settings)


def test_unreadable_settings_notebook_run_is_blocked_before_kernel_start(tmp_path: Path, monkeypatch):
    import nbformat

    from marvis import notebooks

    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")])
    source = tmp_path / "model.ipynb"
    nbformat.write(notebook, source)
    monkeypatch.setattr(notebooks, "_run_notebook_with_policy",
                        lambda *a, **kw: pytest.fail("kernel must not start"))

    result = notebooks.run_notebook(
        notebook_path=source, executed_path=tmp_path / "executed.ipynb",
        log_path=tmp_path / "notebook.log", network_mode=SETTINGS_UNREADABLE,
    )

    assert result.succeeded is False
    assert result.execution_policy["state"] == "blocked"
    assert result.execution_policy["available"] is False
    assert "无法读取" in result.error_value


def test_unreadable_settings_placeholder_is_never_persisted(tmp_path: Path):
    with pytest.raises(ValueError):
        save_execution_environment(tmp_path, ExecutionEnvironmentSettings(notebook_network_mode=SETTINGS_UNREADABLE))
    assert not (tmp_path / "settings" / "execution_environment.json").exists()


def test_settings_api_reports_unreadable_state_and_requires_explicit_repair(tmp_path: Path):
    from fastapi.testclient import TestClient

    from marvis.app import create_app

    settings_path = _write_unreadable_settings(tmp_path, "truncated")
    corrupted = settings_path.read_bytes()
    # Startup itself must not crash or rewrite the file.
    client = TestClient(create_app(tmp_path))
    assert settings_path.read_bytes() == corrupted

    for endpoint in ("/api/settings/execution-environment", "/api/settings/execution-environment/options"):
        payload = client.get(endpoint).json()
        # Review B4: every unknown choice is a placeholder, never a default.
        for field in ("execution_mode", "kernel_name", "conda_env_name", "python_executable",
                      "notebook_network_mode", "worker_isolation_mode"):
            assert payload["settings"][field] == SETTINGS_UNREADABLE, field
        assert payload["settings"]["settings_unreadable"] is True
        assert payload["settings"]["notebook_memory_limit_mb"] is None
        assert payload["validation"]["ok"] is False
        assert payload["validation"]["message"] == SETTINGS_UNREADABLE_MESSAGE
        assert "重新选择 Python 环境和 Notebook 网络模式" in payload["validation"]["network_capability"]["next_action"]

    # A kernel-only save (or one echoing the placeholder) must not invent a
    # boundary for the unknown choice; the corrupted file stays untouched.
    unknown = dict.fromkeys(("execution_mode", "kernel_name", "conda_env_name", "python_executable",
                             "notebook_network_mode", "worker_isolation_mode"), SETTINGS_UNREADABLE)
    for body in (
        {"execution_mode": "jupyter_kernel", "kernel_name": "python3"},
        {"execution_mode": "jupyter_kernel", "kernel_name": "python3",
         "notebook_network_mode": SETTINGS_UNREADABLE, "worker_isolation_mode": "trusted_local"},
        {"notebook_network_mode": "loopback_only"},
        # The UI echoing what GET reported, with only the network mode chosen:
        {**unknown, "notebook_network_mode": "loopback_only"},
        # Both boundaries chosen but the Python environment still unknown:
        {**unknown, "notebook_network_mode": "loopback_only", "worker_isolation_mode": "trusted_local"},
        {**unknown, "execution_mode": "jupyter_kernel", "kernel_name": "python3", "conda_env_name": "",
         "python_executable": "", "notebook_network_mode": "loopback_only"},
    ):
        response = client.put("/api/settings/execution-environment", json=body)
        assert response.status_code == 422, body
        detail = response.json()["detail"]
        assert detail["code"] == "execution_settings_unreadable"
        assert "无法读取" in detail["message"]
        assert settings_path.read_bytes() == corrupted

    repaired = client.put("/api/settings/execution-environment", json={
        "execution_mode": "jupyter_kernel", "kernel_name": "python3",
        "notebook_network_mode": "loopback_only", "worker_isolation_mode": "trusted_local",
    })
    assert repaired.status_code == 200
    assert repaired.json()["settings"]["notebook_network_mode"] == "loopback_only"
    assert load_execution_environment(tmp_path).notebook_network_mode == "loopback_only"


def test_network_mode_defaults_legacy_settings_to_trusted_and_round_trips(tmp_path):
    import json

    path = tmp_path / "settings" / "execution_environment.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"kernel_name": "python3"}))
    assert load_execution_environment(tmp_path).notebook_network_mode == "trusted_local"
    settings = ExecutionEnvironmentSettings(notebook_network_mode="loopback_only")
    save_execution_environment(tmp_path, settings)
    assert load_execution_environment(tmp_path) == settings


def test_loopback_preflight_reports_unavailable_without_downgrading(monkeypatch):
    monkeypatch.setattr("marvis.local_execution.sys.platform", "linux")
    result = validate_execution_environment(ExecutionEnvironmentSettings(notebook_network_mode="loopback_only"))
    assert not result.ok
    assert result.network_capability["mode"] == "loopback_only"
    assert result.network_capability["status"] == "unsupported"
    assert not result.network_capability["network_restricted"]
    assert "未启动" in result.message


def test_invalid_saved_network_mode_does_not_silently_become_trusted(tmp_path):
    save_execution_environment(tmp_path, ExecutionEnvironmentSettings(notebook_network_mode="invalid"))
    settings = load_execution_environment(tmp_path)
    assert settings.notebook_network_mode == "invalid"
    assert not validate_execution_environment(settings).ok


def test_loopback_environment_preflight_rejects_custom_provisioner_before_version_check(monkeypatch):
    from marvis.local_execution import LocalExecutionCapability

    monkeypatch.setattr(execution_environment, "execution_capability", lambda *a, **kw:
                        LocalExecutionCapability("loopback_only", "darwin", "macos_seatbelt", True,
                                                 "available", True))
    monkeypatch.setattr(execution_environment, "available_kernel_specs", lambda: {
        "python3": {"metadata": {"kernel_provisioner": {"provisioner_name": "remote"}}},
    })
    calls = []
    monkeypatch.setattr(execution_environment, "_run_version_check", lambda *a, **kw: calls.append(True))
    result = validate_execution_environment(ExecutionEnvironmentSettings(
        execution_mode="python_executable", python_executable="must-not-run",
        notebook_network_mode="loopback_only",
    ))
    assert not result.ok
    assert calls == []
    assert "provisioner" in result.message
