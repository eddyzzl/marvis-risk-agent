from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from riskmodel_checker.api import router as api_router
from riskmodel_checker.branding import (
    load_branding,
    render_branded_index_html,
    resolve_branding_asset,
)
from riskmodel_checker.db import init_db
from riskmodel_checker.recovery import reclaim_stale_running_tasks
from riskmodel_checker.settings import Settings, build_settings
from riskmodel_checker.state_machine import IllegalTransition


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _is_local_client(host: str | None) -> bool:
    if not host:
        return False
    return host in _LOCAL_CLIENT_HOSTS or host.startswith("127.")


def create_app(workspace: str | Path | Settings) -> FastAPI:
    settings = workspace if isinstance(workspace, Settings) else build_settings(workspace)
    init_db(settings.db_path)
    reclaim_stale_running_tasks(settings.db_path)

    app = FastAPI(title="MARVIS Risk Agent")
    app.state.settings = settings

    @app.middleware("http")
    async def _local_unsafe_method_guard(request, call_next):
        if request.method.upper() not in _SAFE_METHODS and not _is_local_client(
            request.client.host if request.client else None
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "unsafe API methods are limited to local clients"},
            )
        return await call_next(request)

    app.include_router(api_router)

    @app.exception_handler(IllegalTransition)
    def _illegal_transition(_request, exc: IllegalTransition):
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "current_status": exc.current.value,
            },
        )

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/branding/assets/{asset_path:path}")
    def branding_asset(asset_path: str) -> FileResponse:
        asset = resolve_branding_asset(settings.workspace, asset_path)
        if asset is None:
            raise HTTPException(status_code=404, detail="branding asset not found")
        return FileResponse(asset)

    @app.get("/")
    def index() -> HTMLResponse:
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(
            render_branded_index_html(index_html, load_branding(settings.workspace))
        )

    return app
