import subprocess
import sys

import pytest

from riskmodel_checker import __main__
from riskmodel_checker.settings import Settings


def test_cli_help_lists_validate_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        __main__.main(["--help"])

    assert exc.value.code == 0
    stdout = capsys.readouterr().out
    assert "serve" in stdout
    assert "validate" in stdout


@pytest.mark.parametrize("args", [["--help"], ["serve", "--help"]])
def test_cli_help_returns_without_loading_validation_stack(args):
    result = subprocess.run(
        [sys.executable, "-m", "riskmodel_checker", *args],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert "MARVIS Risk Agent" in result.stdout


def test_cli_validate_dispatches_pipeline(tmp_path, monkeypatch):
    calls = []

    def fake_build_settings(workspace):
        workspace = workspace.resolve()
        return Settings(workspace=workspace)

    def fake_init_db(db_path):
        calls.append(("init_db", db_path))

    def fake_run_staged_pipeline(*, task_id, settings):
        calls.append(("run_staged_pipeline", task_id, settings))

    class FakePipelineSettings:
        def __init__(
            self,
            *,
            workspace,
            db_path,
            report_template_path,
            feature_columns,
            notebook_kernel_name,
        ):
            self.workspace = workspace
            self.db_path = db_path
            self.report_template_path = report_template_path
            self.feature_columns = feature_columns
            self.notebook_kernel_name = notebook_kernel_name

    monkeypatch.setattr(
        __main__,
        "_load_validation_runtime",
        lambda: (
            fake_build_settings,
            fake_init_db,
            FakePipelineSettings,
            fake_run_staged_pipeline,
            lambda workspace: type("Environment", (), {"kernel_name": "riskmodel-kernel"})(),
        ),
    )

    __main__.main(
        [
            "validate",
            "task-1",
            "--workspace",
            str(tmp_path),
            "--feature-columns",
            "x1,x2",
        ]
    )

    assert calls[0] == ("init_db", tmp_path / "riskmodel_checker.sqlite")
    assert calls[1][0] == "run_staged_pipeline"
    assert calls[1][1] == "task-1"
    assert calls[1][2].feature_columns == ["x1", "x2"]
    assert calls[1][2].notebook_kernel_name == "riskmodel-kernel"


def test_cli_without_subcommand_defaults_to_serve(monkeypatch):
    calls = []

    def fake_serve(args):
        calls.append((args.host, args.port, args.workspace))

    monkeypatch.setattr(__main__, "_serve", fake_serve)

    __main__.main(["--host", "0.0.0.0", "--port", "9000"])

    assert calls == [("0.0.0.0", 9000, __main__.Path("./workspace"))]
