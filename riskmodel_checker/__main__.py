import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="marvis-risk-agent",
        description="MARVIS Risk Agent CLI",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workspace", type=Path, default=Path("./workspace"))

    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FastAPI app",
        description="MARVIS Risk Agent CLI",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--workspace", type=Path, default=Path("./workspace"))

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run the model-validation workflow for a task",
    )
    validate_parser.add_argument("task_id")
    validate_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("./workspace"),
    )
    validate_parser.add_argument(
        "--feature-columns",
        default="",
        help="Deprecated: comma-separated feature column names. v3 notebooks use RMC_SCORE_FN.",
    )

    args = parser.parse_args(argv)
    if args.command == "validate":
        _validate(args)
    else:
        _serve(args)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from riskmodel_checker.app import create_app

    app = create_app(args.workspace)
    print(f"MARVIS Risk Agent running at http://{args.host}:{args.port}")
    print("If running behind JupyterHub, try the matching /proxy/<port>/ URL.")
    uvicorn.run(app, host=args.host, port=args.port)


def _validate(args: argparse.Namespace) -> None:
    build_settings, init_db, PipelineSettings, run_staged_pipeline = (
        _load_validation_runtime()
    )
    settings = build_settings(args.workspace)
    init_db(settings.db_path)
    run_staged_pipeline(
        task_id=args.task_id,
        settings=PipelineSettings(
            workspace=settings.workspace,
            db_path=settings.db_path,
            report_template_path=settings.report_template_path,
            feature_columns=_parse_feature_columns(args.feature_columns),
        ),
    )


def _load_validation_runtime():
    from riskmodel_checker.db import init_db
    from riskmodel_checker.pipeline import PipelineSettings, run_staged_pipeline
    from riskmodel_checker.settings import build_settings

    return build_settings, init_db, PipelineSettings, run_staged_pipeline


def _parse_feature_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    return columns


if __name__ == "__main__":
    main()
