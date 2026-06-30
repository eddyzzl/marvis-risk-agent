import sys
from pathlib import Path

import pytest

from marvis import __main__ as cli


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_short_marvis_script_alias():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'marvis = "marvis.__main__:main"' in text


def test_pyproject_declares_runtime_dependency_bounds():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for requirement in [
        '"pydantic>=2.7,<3"',
        '"pandas>=2.2,<3"',
        '"packaging>=16.8,<24"',
        '"nbclient>=0.8,<0.12"',
        '"nbformat>=5.9,<6"',
        '"ipykernel>=6.23.2,<7"',
        '"scikit-learn>=1.4,<2"',
    ]:
        assert requirement in text


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
        if args == ("status", "--short", "--untracked-files=no"):
            return " M README.md"
        raise AssertionError(args)

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    args = cli._parse_args(["update", "--repo", str(tmp_path)])

    with pytest.raises(RuntimeError, match="tracked uncommitted changes"):
        cli._update(args)


def test_update_ignores_untracked_files_when_checked_tree_is_clean(monkeypatch, tmp_path):
    git_commands = []
    process_commands = []

    def fake_git_output(repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short", "--untracked-files=no"):
            return ""
        if args == ("describe", "--tags", "--always"):
            return "V1.1.4"
        raise AssertionError(args)

    def fake_run_git(repo, *args):
        git_commands.append(args)

    def fake_run_process(command, *, cwd):
        process_commands.append((command, cwd))

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    monkeypatch.setattr(cli, "_run_git", fake_run_git)
    monkeypatch.setattr(cli, "_run_process", fake_run_process)
    monkeypatch.setattr(cli, "_current_python_is_conda_base", lambda: False)

    args = cli._parse_args(["update", "--repo", str(tmp_path)])
    result = cli._update(args)

    assert git_commands == [
        ("fetch", "origin"),
        ("pull", "--ff-only", "origin", "main"),
    ]
    assert process_commands == [
        ([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], tmp_path)
    ]
    assert result["repo"] == str(tmp_path)
    assert result["version"] == "V1.1.4"


def test_update_fetches_fast_forwards_and_refreshes_editable_install(monkeypatch, tmp_path):
    git_commands = []
    process_commands = []

    def fake_git_output(repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short", "--untracked-files=no"):
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
    monkeypatch.setattr(cli, "_current_python_is_conda_base", lambda: False)

    args = cli._parse_args(["update", "--repo", str(tmp_path)])
    result = cli._update(args)

    assert git_commands == [
        ("fetch", "origin"),
        ("pull", "--ff-only", "origin", "main"),
    ]
    assert process_commands == [
        ([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], tmp_path)
    ]
    assert result["repo"] == str(tmp_path)
    assert result["version"] == "V1.1.0"


def test_update_with_deps_refreshes_editable_install_and_dependencies(monkeypatch, tmp_path):
    process_commands = []

    def fake_git_output(repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short", "--untracked-files=no"):
            return ""
        if args == ("describe", "--tags", "--always"):
            return "V1.1.0"
        raise AssertionError(args)

    def fake_run_git(repo, *args):
        pass

    def fake_run_process(command, *, cwd):
        process_commands.append((command, cwd))

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    monkeypatch.setattr(cli, "_run_git", fake_run_git)
    monkeypatch.setattr(cli, "_run_process", fake_run_process)
    monkeypatch.setattr(cli, "_current_python_is_conda_base", lambda: False)

    args = cli._parse_args(["update", "--repo", str(tmp_path), "--with-deps"])
    cli._update(args)

    assert process_commands == [
        ([sys.executable, "-m", "pip", "install", "-e", "."], tmp_path)
    ]


def test_update_from_conda_base_creates_dedicated_env(monkeypatch, tmp_path):
    process_commands = []

    def fake_git_output(repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short", "--untracked-files=no"):
            return ""
        if args == ("describe", "--tags", "--always"):
            return "V1.1.0"
        raise AssertionError(args)

    def fake_run_process(command, *, cwd):
        process_commands.append((command, cwd))

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    monkeypatch.setattr(cli, "_run_git", lambda repo, *args: None)
    monkeypatch.setattr(cli, "_run_process", fake_run_process)
    monkeypatch.setattr(cli, "_current_python_is_conda_base", lambda: True)
    monkeypatch.setattr(cli, "_conda_env_exists", lambda env_name: False)
    monkeypatch.setattr(cli, "_conda_command", lambda: "conda")

    args = cli._parse_args(["update", "--repo", str(tmp_path)])
    result = cli._update(args)

    assert process_commands == [
        (["conda", "create", "-y", "-n", "marvis", "python=3.12", "pip"], tmp_path),
        (
            [
                "conda",
                "run",
                "-n",
                "marvis",
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                ".",
            ],
            tmp_path,
        ),
    ]
    assert result["install_target"] == "conda:marvis"


def test_update_from_conda_base_reuses_dedicated_env_without_deps(monkeypatch, tmp_path):
    process_commands = []

    def fake_git_output(repo, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short", "--untracked-files=no"):
            return ""
        if args == ("describe", "--tags", "--always"):
            return "V1.1.0"
        raise AssertionError(args)

    def fake_run_process(command, *, cwd):
        process_commands.append((command, cwd))

    monkeypatch.setattr(cli, "_git_output", fake_git_output)
    monkeypatch.setattr(cli, "_run_git", lambda repo, *args: None)
    monkeypatch.setattr(cli, "_run_process", fake_run_process)
    monkeypatch.setattr(cli, "_current_python_is_conda_base", lambda: True)
    monkeypatch.setattr(cli, "_conda_env_exists", lambda env_name: True)
    monkeypatch.setattr(cli, "_conda_command", lambda: "conda")

    args = cli._parse_args(["update", "--repo", str(tmp_path)])
    result = cli._update(args)

    assert process_commands == [
        (
            [
                "conda",
                "run",
                "-n",
                "marvis",
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                ".",
                "--no-deps",
            ],
            tmp_path,
        ),
    ]
    assert result["install_target"] == "conda:marvis"
