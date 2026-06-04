from pathlib import Path

from riskmodel_checker.execution_environment import (
    ExecutionEnvironmentSettings,
    detect_execution_environment_options,
    load_execution_environment,
    save_execution_environment,
    validate_execution_environment,
)


def test_execution_environment_settings_round_trip(tmp_path: Path):
    settings = ExecutionEnvironmentSettings(
        execution_mode="jupyter_kernel",
        kernel_name="riskmodel-kernel",
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
        "riskmodel_checker.execution_environment.available_kernel_names",
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
        "riskmodel_checker.execution_environment.available_kernel_names",
        lambda: ["python3", "riskmodel-kernel"],
    )
    settings = ExecutionEnvironmentSettings(
        execution_mode="jupyter_kernel",
        kernel_name="riskmodel-kernel",
        conda_env_name="",
        python_executable="",
    )

    result = validate_execution_environment(settings)

    assert result.ok is True
    assert result.kernel_name == "riskmodel-kernel"


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
        "riskmodel_checker.execution_environment.available_kernel_specs",
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
        "riskmodel_checker.execution_environment._conda_environment_paths",
        lambda: [conda_env],
    )
    monkeypatch.setattr(
        "riskmodel_checker.execution_environment.sys.executable",
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
        "riskmodel_checker.execution_environment.available_kernel_specs",
        lambda: {"python3": {"display_name": "Python 3", "argv": []}},
    )
    monkeypatch.setattr(
        "riskmodel_checker.execution_environment._conda_environment_paths",
        lambda: [conda_env],
    )

    options = detect_execution_environment_options()
    conda_option = next(
        option for option in options if option.conda_env_name == "analysis"
    )

    assert conda_option.available is False
    assert "Kernel" in conda_option.note
