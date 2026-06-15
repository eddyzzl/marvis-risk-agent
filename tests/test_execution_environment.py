from pathlib import Path

import marvis.execution_environment as execution_environment
from marvis.execution_environment import (
    ExecutionEnvironmentSettings,
    detect_execution_environment_options,
    load_execution_environment,
    save_execution_environment,
    validate_execution_environment,
)


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
