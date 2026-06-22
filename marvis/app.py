import ipaddress
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from marvis import __version__
from marvis.api import router as api_router
from marvis.branding import (
    DEFAULT_BRANDING,
    load_branding,
    render_branded_index_html,
    resolve_branding_asset,
)
from marvis.db import init_db
from marvis.recovery import reclaim_stale_running_tasks
from marvis.settings import Settings, build_settings
from marvis.state_machine import IllegalTransition


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_NAMED_LOCAL_HOSTS = {"localhost", "testclient"}
_REMOTE_READ_ENV = "MARVIS_ALLOW_REMOTE_READ"
_TRUSTED_PROXY_ENV = "MARVIS_TRUSTED_PROXY_HOSTS"
_FORWARDED_CLIENT_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")


def _is_local_client(host: str | None) -> bool:
    if not host:
        return False
    if host in _NAMED_LOCAL_HOSTS:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    # IPv4-mapped IPv6 loopback (::ffff:127.0.0.1) reports as IPv6; unwrap it.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped.is_loopback
    return addr.is_loopback


def _trusted_proxy_hosts() -> frozenset[str]:
    raw = os.environ.get(_TRUSTED_PROXY_ENV, "")
    return frozenset(host.strip() for host in raw.split(",") if host.strip())


def _effective_client_host(request) -> str | None:
    """Originating client host.

    `request.client.host` is the direct TCP peer. Behind a same-host reverse
    proxy that peer is the proxy's loopback address, which would make every
    remote request look local. X-Forwarded-For is only consulted when the direct
    peer is an explicitly trusted proxy (MARVIS_TRUSTED_PROXY_HOSTS); forwarded
    headers from an untrusted loopback peer fail closed instead of inheriting
    local privileges.
    """
    direct = request.client.host if request.client else None
    trusted = _trusted_proxy_hosts()
    forwarded_header_present = any(
        bool(request.headers.get(header)) for header in _FORWARDED_CLIENT_HEADERS
    )
    if direct and trusted and direct in trusted:
        forwarded = request.headers.get("x-forwarded-for", "")
        first_hop = forwarded.split(",")[0].strip() if forwarded else ""
        if first_hop:
            return first_hop
    if forwarded_header_present and _is_local_client(direct):
        return None
    return direct


def _remote_read_enabled() -> bool:
    return os.environ.get(_REMOTE_READ_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_public_read_path(path: str) -> bool:
    return path == "/" or path == "/api/health" or path.startswith("/static/")


def _is_local_only_path(path: str) -> bool:
    """Paths that stay local even when remote read is enabled: system
    configuration and private branding are never exposed remotely. `/api/branding`
    is included because its JSON carries private workspace branding incl. validator
    aliases (real names); the branding asset files under `/branding/` are already
    local-only, so the metadata route must match them."""
    return (
        path == "/api/branding"
        or path.startswith("/api/settings")
        or path.startswith("/branding/")
    )


def create_app(workspace: str | Path | Settings) -> FastAPI:
    settings = workspace if isinstance(workspace, Settings) else build_settings(workspace)
    init_db(settings.db_path)
    reclaim_stale_running_tasks(settings.db_path)

    app = FastAPI(title="MARVIS-全能信贷风控智能体")
    app.state.settings = settings

    @app.middleware("http")
    async def _local_access_guard(request, call_next):
        method = request.method.upper()
        path = request.url.path
        is_local = _is_local_client(_effective_client_host(request))
        if not is_local:
            if method not in _SAFE_METHODS:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "unsafe API methods are limited to local clients"},
                )
            if _is_local_only_path(path):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "this endpoint is limited to local clients"},
                )
            if not _remote_read_enabled() and not _is_public_read_path(path):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "API access is limited to local clients"},
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
    def index(request: Request) -> HTMLResponse:
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        index_html = index_html.replace("__MARVIS_STATIC_VERSION__", __version__)
        branding = load_branding(settings.workspace)
        if not _is_local_client(_effective_client_host(request)):
            branding = dict(DEFAULT_BRANDING)
        return HTMLResponse(
            render_branded_index_html(index_html, branding)
        )

    return app
