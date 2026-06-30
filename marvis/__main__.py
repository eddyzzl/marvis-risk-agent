import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zlib


DEFAULT_CONDA_ENV_NAME = "marvis"
DEFAULT_CONDA_PYTHON = "3.12"


@dataclass(frozen=True)
class ServeOptions:
    host: str
    port: int
    workspace: Path
    profile: str = ""


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        if args.command == "validate":
            _validate(args)
        elif args.command == "update":
            result = _update(args)
            print(f"MARVIS updated at {result['repo']} ({result['version']}).")
            if result.get("install_target", "").startswith("conda:"):
                env_name = result["install_target"].split(":", 1)[1]
                print(f"MARVIS was installed into conda environment '{env_name}'.")
                print(f"Run `conda activate {env_name}` before starting MARVIS.")
        elif args.command == "version":
            _print_version()
        else:
            _serve(_apply_serve_defaults(args))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="marvis",
        description="MARVIS-Agent CLI",
    )
    _add_serve_options(parser)

    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FastAPI app",
        description="MARVIS-Agent CLI",
    )
    _add_serve_options(serve_parser)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run the model-validation workflow for a task",
    )
    validate_parser.add_argument("task_id")
    validate_parser.add_argument("--profile", default="")
    validate_parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
    )
    validate_parser.add_argument(
        "--feature-columns",
        default="",
        help="Deprecated: comma-separated feature column names. v3 notebooks use RMC_SCORE_FN.",
    )

    update_parser = subparsers.add_parser(
        "update",
        help="Update a git-cloned MARVIS checkout",
    )
    update_parser.add_argument("--repo", type=Path, default=None)
    update_parser.add_argument("--remote", default="origin")
    update_parser.add_argument("--branch", default="main")
    update_parser.add_argument("--skip-install", action="store_true")
    update_parser.add_argument(
        "--env-name",
        default=DEFAULT_CONDA_ENV_NAME,
        help="Dedicated conda environment to create or reuse when update is run from conda base.",
    )
    update_parser.add_argument(
        "--no-dedicated-env",
        action="store_true",
        help="Allow update to refresh the current conda base environment.",
    )
    update_parser.add_argument(
        "--with-deps",
        action="store_true",
        help="Refresh MARVIS dependencies too. Default update only refreshes the editable install.",
    )

    subparsers.add_parser("version", help="Print the installed MARVIS version")

    return parser.parse_args(argv)


def _add_serve_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workspace", type=Path, default=None)


def _resolve_serve_options(args: argparse.Namespace) -> ServeOptions:
    defaults = _profile_defaults(getattr(args, "profile", ""))
    return ServeOptions(
        host=args.host or defaults.host,
        port=args.port if args.port is not None else defaults.port,
        workspace=args.workspace or defaults.workspace,
        profile=defaults.profile,
    )


def _apply_serve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    options = _resolve_serve_options(args)
    args.host = options.host
    args.port = options.port
    args.workspace = options.workspace
    args.profile = options.profile
    return args


def _profile_defaults(profile: str | None) -> ServeOptions:
    slug = _profile_slug(profile)
    if not slug:
        return ServeOptions("127.0.0.1", 8000, Path("./workspace"), "")
    if slug in {"main", "stable"}:
        return ServeOptions("127.0.0.1", 8000, Path("./workspace-main"), slug)
    if slug in {"dev", "current"}:
        return ServeOptions("127.0.0.1", 8001, Path("./workspace-dev"), slug)

    version_match = re.fullmatch(r"v(\d+)(?:-(\d+))?", slug)
    if version_match:
        major = int(version_match.group(1))
        minor = int(version_match.group(2) or 0)
        port = 8000 + minor if major == 1 else 8000 + major * 100 + minor
        return ServeOptions("127.0.0.1", port, Path(f"./workspace-{slug}"), slug)

    port = 8100 + zlib.crc32(slug.encode("utf-8")) % 800
    return ServeOptions("127.0.0.1", port, Path(f"./workspace-{slug}"), slug)


def _profile_slug(profile: str | None) -> str:
    value = str(profile or "").strip().lower().replace(".", "-").replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return re.sub(r"-+", "-", slug)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from marvis.app import create_app

    options = _resolve_serve_options(args)
    app = create_app(options.workspace)
    if options.profile:
        print(
            f"MARVIS profile '{options.profile}' uses workspace {options.workspace}"
            f" and port {options.port}."
        )
    print(f"MARVIS-Agent running at http://{options.host}:{options.port}")
    print("If running behind JupyterHub, try the matching /proxy/<port>/ URL.")
    uvicorn.run(app, host=options.host, port=options.port)


def _validate(args: argparse.Namespace) -> None:
    build_settings, init_db, PipelineSettings, run_staged_pipeline, load_execution_environment = (
        _load_validation_runtime()
    )
    workspace = args.workspace or _profile_defaults(getattr(args, "profile", "")).workspace
    settings = build_settings(workspace)
    environment = load_execution_environment(settings.workspace)
    init_db(settings.db_path)
    run_staged_pipeline(
        task_id=args.task_id,
        settings=PipelineSettings(
            workspace=settings.workspace,
            db_path=settings.db_path,
            report_template_path=settings.report_template_path,
            feature_columns=_parse_feature_columns(args.feature_columns),
            notebook_kernel_name=environment.kernel_name,
        ),
    )


def _load_validation_runtime():
    from marvis.db import init_db
    from marvis.execution_environment import load_execution_environment
    from marvis.pipeline import PipelineSettings, run_staged_pipeline
    from marvis.settings import build_settings

    return build_settings, init_db, PipelineSettings, run_staged_pipeline, load_execution_environment


def _parse_feature_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    return columns


def _update(args: argparse.Namespace) -> dict[str, str]:
    repo_hint = args.repo or _package_root()
    repo = _git_root(repo_hint)
    current_branch = _git_output(repo, "branch", "--show-current")
    if not current_branch:
        raise RuntimeError("marvis update requires a checked-out branch, not detached HEAD")
    if args.branch and current_branch != args.branch:
        raise RuntimeError(
            f"marvis update expected branch {args.branch}; current branch is {current_branch}"
        )
    if _git_output(repo, "status", "--short", "--untracked-files=no"):
        raise RuntimeError(
            "marvis update found tracked uncommitted changes; commit or stash them before updating"
        )

    _run_git(repo, "fetch", args.remote)
    _run_git(repo, "pull", "--ff-only", args.remote, args.branch or current_branch)
    install_target = "skipped"
    if not args.skip_install:
        if _should_use_dedicated_conda_env(args):
            created = _ensure_conda_env(args.env_name, cwd=repo)
            install_with_deps = args.with_deps or created
            _run_process(
                _conda_env_install_command(args.env_name, with_deps=install_with_deps),
                cwd=repo,
            )
            install_target = f"conda:{args.env_name}"
        else:
            _run_process(_editable_install_command(with_deps=args.with_deps), cwd=repo)
            install_target = "current-python"
    version = _git_output(repo, "describe", "--tags", "--always")
    return {"repo": str(repo), "version": version, "install_target": install_target}


def _editable_install_command(*, with_deps: bool) -> list[str]:
    command = [sys.executable, "-m", "pip", "install", "-e", "."]
    if not with_deps:
        command.append("--no-deps")
    return command


def _conda_env_install_command(env_name: str, *, with_deps: bool) -> list[str]:
    command = [
        _conda_command(),
        "run",
        "-n",
        env_name,
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        ".",
    ]
    if not with_deps:
        command.append("--no-deps")
    return command


def _should_use_dedicated_conda_env(args: argparse.Namespace) -> bool:
    if args.no_dedicated_env:
        return False
    if not args.env_name:
        return False
    return _current_python_is_conda_base()


def _current_python_is_conda_base() -> bool:
    if os.environ.get("CONDA_DEFAULT_ENV") == "base":
        return True

    conda_info = _conda_info()
    if not conda_info:
        return False
    root_prefix = str(conda_info.get("root_prefix") or conda_info.get("base_prefix") or "")
    if not root_prefix:
        return False
    return _safe_same_path(Path(sys.prefix), Path(root_prefix))


def _conda_info() -> dict:
    try:
        completed = subprocess.run(
            [_conda_command(), "info", "--json"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ensure_conda_env(env_name: str, *, cwd: Path) -> bool:
    if _conda_env_exists(env_name):
        return False
    _run_process(
        [
            _conda_command(),
            "create",
            "-y",
            "-n",
            env_name,
            f"python={DEFAULT_CONDA_PYTHON}",
            "pip",
        ],
        cwd=cwd,
    )
    return True


def _conda_env_exists(env_name: str) -> bool:
    try:
        completed = subprocess.run(
            [_conda_command(), "run", "-n", env_name, "python", "-V"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "marvis update detected conda base, but the conda command was not found; "
            "activate conda first or rerun with --no-dedicated-env"
        ) from exc
    return completed.returncode == 0


def _conda_command() -> str:
    return os.environ.get("CONDA_EXE") or "conda"


def _safe_same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def _git_root(path: Path) -> Path:
    try:
        return Path(_git_output(path, "rev-parse", "--show-toplevel"))
    except RuntimeError as exc:
        raise RuntimeError("marvis update target is not a git clone checkout") from exc


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "git command failed").strip()
        raise RuntimeError(message)
    return completed.stdout.strip()


def _run_git(repo: Path, *args: str) -> None:
    _run_process(["git", "-C", str(repo), *args], cwd=repo)


def _run_process(command: list[str], *, cwd: Path) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def _print_version() -> None:
    from marvis import __version__

    print(f"MARVIS-Agent {__version__}")


if __name__ == "__main__":
    main()
