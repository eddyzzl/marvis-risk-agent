import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import zlib


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
        elif args.command == "version":
            _print_version()
        else:
            _serve(_apply_serve_defaults(args))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="marvis",
        description="MARVIS Risk Agent CLI",
    )
    _add_serve_options(parser)

    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FastAPI app",
        description="MARVIS Risk Agent CLI",
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
    print(f"MARVIS Risk Agent running at http://{options.host}:{options.port}")
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
    if _git_output(repo, "status", "--short"):
        raise RuntimeError(
            "marvis update found uncommitted changes; commit or stash them before updating"
        )

    _run_git(repo, "fetch", args.remote)
    _run_git(repo, "pull", "--ff-only", args.remote, args.branch or current_branch)
    if not args.skip_install:
        _run_process([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo)
    version = _git_output(repo, "describe", "--tags", "--always")
    return {"repo": str(repo), "version": version}


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
    subprocess.run(command, cwd=cwd, check=True)


def _print_version() -> None:
    from marvis import __version__

    print(f"MARVIS Risk Agent {__version__}")


if __name__ == "__main__":
    main()
