import sys
from pathlib import Path

import pytest

from riskmodel_checker import __main__ as cli


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_short_marvis_script_alias():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'marvis = "riskmodel_checker.__main__:main"' in text


def test_main_without_subcommand_starts_default_profile(monkeypatch):
    observed = []

    def fake_serve(args):
        observed.append(cli._resolve_serve_options(args))

    monkeypatch.setattr(cli, "_serve", fake_serve)

    cli.main([])

    assert observed[0].host == "127.0.0.1"
    assert observed[0].port == 8000
    assert observed[0].workspace == Path("./workspace")


def test_serve_profile_defaults_for_main_and_v1_1():
    main_args = cli._parse_args(["serve", "--profile", "main"])
    v1_args = cli._parse_args(["serve", "--profile", "v1-1"])

    main_options = cli._resolve_serve_options(main_args)
    v1_options = cli._resolve_serve_options(v1_args)

    assert main_options.port == 8000
    assert main_options.workspace == Path("./workspace-main")
    assert v1_options.port == 8001
    assert v1_options.workspace == Path("./workspace-v1-1")


def test_explicit_serve_options_override_profile_defaults():
    args = cli._parse_args(
        [
            "serve",
            "--profile",
            "v1-1",
            "--host",
            "0.0.0.0",
            "--port",
            "8017",
            "--workspace",
            "./custom-workspace",
        ]
    )

    options = cli._resolve_serve_options(args)

    assert options.host == "0.0.0.0"
    assert options.port == 8017
    assert options.workspace == Path("./custom-workspace")


def test_update_rejects_non_git_checkout(monkeypatch, tmp_path):
    def fake_git_output(repo, *args):
        raise RuntimeError("not a git worktree")

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    args = cli._parse_args(["update", "--repo", str(tmp_path)])

    with pytest.raises(RuntimeError, match="not a git clone"):
        cli._update(args)


def test_update_rejects_dirty_worktree(monkeypatch, tmp_path):
    def fake_git_output(repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short"):
            return " M README.md"
        raise AssertionError(args)

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    args = cli._parse_args(["update", "--repo", str(tmp_path)])

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        cli._update(args)


def test_update_fetches_fast_forwards_and_refreshes_editable_install(monkeypatch, tmp_path):
    git_commands = []
    process_commands = []

    def fake_git_output(repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short"):
            return ""
        if args == ("describe", "--tags", "--always"):
            return "V1.1.0"
        raise AssertionError(args)

    def fake_run_git(repo, *args):
        git_commands.append(args)

    def fake_run_process(command, *, cwd):
        process_commands.append((command, cwd))

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    monkeypatch.setattr(cli, "_run_git", fake_run_git)
    monkeypatch.setattr(cli, "_run_process", fake_run_process)

    args = cli._parse_args(["update", "--repo", str(tmp_path)])
    result = cli._update(args)

    assert git_commands == [
        ("fetch", "origin"),
        ("pull", "--ff-only", "origin", "main"),
    ]
    assert process_commands == [
        ([sys.executable, "-m", "pip", "install", "-e", "."], tmp_path)
    ]
    assert result["repo"] == str(tmp_path)
    assert result["version"] == "V1.1.0"
