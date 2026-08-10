import base64
import binascii
import hashlib
import hmac
from collections.abc import Callable, Mapping
from datetime import datetime
from html import escape
import ipaddress
import json
import logging
from pathlib import Path
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from marvis import __version__
from marvis.errors import not_found
from marvis.agent_memory.consolidation import (
    CONSOLIDATION_TRIGGERS,
    ConsolidationScheduler,
)
from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.evolution import EvolutionManager
from marvis.agent_memory.store import AgentMemoryStore
from marvis.api import router as api_router
from marvis.artifacts.recovery import reconcile_workspace_artifacts
from marvis.branding import (
    DEFAULT_BRANDING,
    load_branding,
    render_branded_index_html,
    resolve_branding_asset,
)
from marvis.data.backend import DUCKDB_TEMP_DIR_NAME, DataBackend, duckdb_health
from marvis.data.dataset_source_gc import (
    DatasetSourceGarbageCollector,
    DatasetSourceGcSweepReport,
    DatasetSourceGcWatchdog,
)
from marvis.data.task_filesystem_gc import (
    TaskFilesystemGarbageCollector,
    TaskFilesystemGcSweepReport,
    TaskFilesystemGcWatchdog,
)
from marvis.data.validation_batch_upload_gc import (
    ValidationBatchUploadGarbageCollector,
    ValidationBatchUploadRecoveryReport,
)
from marvis.db_schema import init_db, sqlite_health
from marvis.drafts.registry import DraftRegistry
from marvis.drafts.sandbox import DraftSandbox
from marvis.execution_environment import load_execution_environment
from marvis.governance.errors import AuthorizationError
from marvis.governance.repository import (
    DEFAULT_SESSION_TTL_SECONDS,
    GovernanceRepository,
)
from marvis.governance.service import GovernanceService
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import load_llm_settings, resolve_llm_model
from marvis.memory_policy import load_memory_policy
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.intent import IntentRouter
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.reviewer import Reviewer
from marvis.orchestrator.subagent import SubAgentDispatcher
from marvis.orchestrator.templates import clear_user_templates, load_builtin_templates
from marvis.orchestrator.templates.skills import load_user_skill_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.operations.integration import build_operations_runtime
from marvis.operations.router import router as operations_router
from marvis.operations.scheduler import MonitoringExecutor
from marvis.plugins.hooks import HookDispatcher
from marvis.plugins.loader import load_builtin_packs, sync_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.production_governance.router import router as production_governance_router
from marvis.production_governance.evidence import ActivationEvidenceVerifier
from marvis.job_watchdog import (
    JobHeartbeatWatchdog,
    heartbeat_timeout_seconds,
    sweep_heartbeat_lost_jobs,
)
from marvis.recovery import reclaim_running_plans, reclaim_stale_running_tasks
from marvis.repositories.drafts import DraftRepository
from marvis.repositories.llm_calls import record_llm_call
from marvis.repositories.plans import PlanRepository
from marvis.repositories.plugins import PluginRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.tasks import TaskRepository
from marvis.routers.agent_memory import router as agent_memory_router
from marvis.routers.llm import router as llm_router
from marvis.routers.artifacts import router as artifacts_router
from marvis.routers.audit import router as audit_router
from marvis.routers.branding import router as branding_router
from marvis.routers.data import router as data_router
from marvis.routers.data_analysis import router as data_analysis_router
from marvis.routers.drafts import router as drafts_router
from marvis.routers.evidence import router as evidence_router
from marvis.routers.materials import router as materials_router
from marvis.routers.modeling import router as modeling_router
from marvis.routers.plans import router as plans_router
from marvis.routers.plugins import ensure_plugin_admin_token, router as plugins_router
from marvis.routers.report_fields import router as report_fields_router
from marvis.routers.reports import router as reports_router
from marvis.routers.scans import router as scans_router
from marvis.routers.skills import router as skills_router
from marvis.routers.stage_controls import router as stage_controls_router
from marvis.routers.strategy_candidate_lab import router as strategy_candidate_lab_router
from marvis.routers.tasks import router as tasks_router
from marvis.routers.validation_agent import router as validation_agent_router
from marvis.routers.validation_batches import router as validation_batches_router
from marvis.routers.validation_contracts import router as validation_contracts_router
from marvis.routers.validation_stages import router as validation_stages_router
from marvis.settings import Settings, build_settings
from marvis.state_machine import IllegalTransition
from marvis.validation_batch_ingress import ValidationBatchIngressLimitMiddleware


logger = logging.getLogger(__name__)

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_NAMED_LOCAL_HOSTS = {"localhost", "testclient"}
_REMOTE_READ_ENV = "MARVIS_ALLOW_REMOTE_READ"
_TRUSTED_PROXY_ENV = "MARVIS_TRUSTED_PROXY_HOSTS"
_LOCAL_TOKEN_ENV = "MARVIS_LOCAL_TOKEN"
_LOCAL_TOKEN_HEADER = "x-marvis-token"
_LOCAL_BASIC_REALM = "MARVIS"
_LOCAL_SESSION_COOKIE = "marvis_local_session"
_FORWARDED_CLIENT_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")
# PERF-9: cache busting must cover every JS/CSS asset the frontend can load,
# not just the 4 entry files -- the version hash below is derived from a live
# rglob (see _static_asset_version) so newly added js/ or css/ files are
# picked up automatically; this constant only documents the globs scanned.
_STATIC_VERSION_GLOBS = ("*.js", "*.css")


def _has_configured_llm(workspace: Path) -> bool:
    """GAP-8: True if at least one enabled model with a resolvable api_key is
    saved -- a pure settings-file read, no network call."""
    try:
        settings_payload = load_llm_settings(workspace)
    except Exception:
        return False
    return any(
        model.get("enabled") and model.get("has_api_key")
        for model in settings_payload.get("enabled_models", [])
    )


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


def _is_forwarded_request_from_trusted_proxy(request) -> bool:
    """True only for the X-Forwarded-For flow this app explicitly trusts.

    A configured proxy is a transport boundary, not an authorization grant.
    The middleware below still requires ``MARVIS_LOCAL_TOKEN`` before a
    forwarded client can use the private workbench surface.
    """

    direct = request.client.host if request.client else None
    if not direct or direct not in _trusted_proxy_hosts():
        return False
    forwarded = request.headers.get("x-forwarded-for", "")
    return bool(forwarded.split(",")[0].strip())


def _remote_read_enabled() -> bool:
    return os.environ.get(_REMOTE_READ_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_local_token() -> str:
    """GAP-5: optional shared-host hardening. `_is_local_client` treats any
    loopback peer as trusted, which is correct for a single-user laptop but
    wrong on a shared JupyterHub-style host where any other logged-in user's
    process can also reach 127.0.0.1 and would otherwise inherit the private
    read/write surface (including installing plugins = arbitrary code
    execution). When MARVIS_LOCAL_TOKEN is set, the credential-bearing browser
    bootstrap and local reads use HTTP Basic (the token is the password) or
    X-Marvis-Token, while writes must present X-Marvis-Token explicitly. When
    unset (the default), behavior is unchanged."""
    return os.environ.get(_LOCAL_TOKEN_ENV, "").strip()


def _request_has_valid_local_token(
    request,
    expected_token: str,
    *,
    allow_basic: bool = False,
) -> bool:
    header_token = request.headers.get(_LOCAL_TOKEN_HEADER, "")
    expected_bytes = expected_token.encode("utf-8")
    header_valid = hmac.compare_digest(header_token.encode("utf-8"), expected_bytes)
    if not allow_basic:
        return header_valid
    basic_password = _request_basic_password(request)
    basic_valid = hmac.compare_digest(basic_password.encode("utf-8"), expected_bytes)
    return header_valid or basic_valid


def _request_basic_password(request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, separator, encoded = authorization.partition(" ")
    if not separator or scheme.lower() != "basic":
        return ""
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return ""
    _username, separator, password = decoded.partition(":")
    return password if separator else ""


def _local_basic_auth_challenge() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": "MARVIS authentication required"},
        headers={
            "WWW-Authenticate": f'Basic realm="{_LOCAL_BASIC_REALM}"',
            "Cache-Control": "no-store",
        },
    )


def _is_public_read_path(path: str) -> bool:
    return path == "/" or path == "/api/health" or path.startswith("/static/")


def _is_local_token_public_path(path: str) -> bool:
    """Assets needed to render a Basic-authenticated page without a header.

    This is deliberately narrower than ``_is_public_read_path``: the latter
    describes what a remote anonymous reader may receive, while a configured
    local token is specifically a shared-host confidentiality boundary.
    """

    return path.startswith("/static/")


def _static_asset_version(static_dir: Path) -> str:
    # PERF-9: bind the year-long immutable-cache key to every JS/CSS path and
    # byte, not to the largest mtime.  Build/copy tools may preserve mtimes,
    # and changing any file older than the current maximum would otherwise
    # leave the URL unchanged while the server advertises immutable content.
    assets = sorted(
        {
            path
            for glob in _STATIC_VERSION_GLOBS
            for path in static_dir.rglob(glob)
            if path.is_file()
        },
        key=lambda path: path.relative_to(static_dir).as_posix(),
    )
    digest = hashlib.sha256()
    hashed_assets = 0
    for path in assets:
        try:
            content = path.read_bytes()
        except OSError:
            continue
        relative_path = path.relative_to(static_dir).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        hashed_assets += 1
    if not hashed_assets:
        return __version__
    return f"{__version__}-{digest.hexdigest()}"


def _static_import_map(static_dir: Path, version: str) -> str:
    # PERF-9: map each module's fully resolved, document-relative URL to the
    # same URL with a version query. Import-map keys are resolved against the
    # document base, not against the importing module, so entries such as
    # "./js/api.js" do not match app.js resolving that specifier to
    # "./static/js/api.js". Both keys and values must also be URL-like; a
    # value such as "static/js/api.js" is a bare specifier and is ignored by
    # browsers. Keeping the paths document-relative preserves deployments
    # mounted below a reverse-proxy prefix.
    js_dir = static_dir / "js"
    v2_dir = js_dir / "v2"
    js_files = sorted(p.name for p in js_dir.glob("*.js") if p.is_file())
    v2_files = sorted(p.name for p in v2_dir.glob("*.js") if p.is_file()) if v2_dir.is_dir() else []

    module_paths = [f"js/{name}" for name in js_files]
    module_paths.extend(f"js/v2/{name}" for name in v2_files)
    imports = {
        f"./static/{relative}": f"./static/{relative}?v={version}"
        for relative in module_paths
    }
    import_map = {"imports": imports}
    return json.dumps(import_map, ensure_ascii=False, separators=(",", ":"))


def _is_local_only_path(path: str) -> bool:
    """Paths that stay local even when remote read is enabled: system
    configuration and private branding are never exposed remotely. `/api/branding`
    is included because its JSON carries private workspace branding incl. validator
    aliases (real names); the branding asset files under `/branding/` are already
    local-only, so the metadata route must match them."""
    return (
        path == "/api/branding"
        or path.startswith("/api/operations")
        or path.startswith("/api/production-governance")
        or path.startswith("/api/settings")
        or path == "/api/skills/reload"
        or path == "/api/skills/validate"
        or path == "/api/strategies/monitoring-due"
        or path.startswith("/branding/")
    )


def create_app(
    workspace: str | Path | Settings,
    *,
    operations_executor_allowlist: Mapping[str, MonitoringExecutor] | None = None,
    operations_clock: Callable[[], datetime] | None = None,
    production_activation_verifiers: Mapping[
        str,
        ActivationEvidenceVerifier,
    ]
    | None = None,
) -> FastAPI:
    settings = workspace if isinstance(workspace, Settings) else build_settings(workspace)
    logger.info("MARVIS starting up workspace=%s version=%s", settings.workspace, __version__)
    init_db(settings.db_path)
    validation_batch_upload_gc = ValidationBatchUploadGarbageCollector(
        settings.db_path,
        material_uploads_root=settings.workspace / "material_uploads",
    )
    try:
        validation_batch_upload_recovery = (
            validation_batch_upload_gc.reconcile_startup()
        )
    except Exception:
        logger.exception("startup validation batch upload recovery failed")
        validation_batch_upload_recovery = ValidationBatchUploadRecoveryReport(
            examined=0,
            removed=0,
            failed=1,
        )
    governance_repo = GovernanceRepository(settings.db_path)
    governance_reconciliation = governance_repo.reconcile_startup()
    reclaim_stale_running_tasks(settings.db_path, tasks_dir=settings.tasks_dir)
    artifact_recovery_report = reconcile_workspace_artifacts(settings)
    _recovery_actions = (
        artifact_recovery_report.removed_staging_dirs
        + artifact_recovery_report.removed_backups
        + artifact_recovery_report.restored_backups
        + artifact_recovery_report.removed_orphan_dirs
        + artifact_recovery_report.removed_orphan_tmp_files
    )
    if _recovery_actions or artifact_recovery_report.errors:
        logger.info(
            "startup artifact recovery: %d action(s), %d error(s)",
            _recovery_actions, len(artifact_recovery_report.errors),
        )
    dataset_source_gc = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
    )
    try:
        dataset_source_gc_startup_report = dataset_source_gc.sweep(limit=100)
        dataset_source_gc_startup_error = None
    except Exception as exc:
        logger.exception("startup dataset source GC sweep failed")
        dataset_source_gc_startup_error = (
            f"{exc.__class__.__name__}: {' '.join(str(exc).split())[:500]}"
        )
        dataset_source_gc_startup_report = DatasetSourceGcSweepReport(
            examined=0,
            deleted=0,
            cancelled_referenced=0,
            deferred=0,
            quarantined=0,
        )
    task_filesystem_gc = TaskFilesystemGarbageCollector(
        settings.db_path,
        tasks_root=settings.tasks_dir,
        datasets_root=settings.datasets_dir,
        material_uploads_root=settings.workspace / "material_uploads",
    )
    try:
        task_filesystem_gc_startup_report = task_filesystem_gc.sweep(limit=100)
        task_filesystem_gc_startup_error = None
    except Exception as exc:
        logger.exception("startup task filesystem GC sweep failed")
        task_filesystem_gc_startup_error = (
            f"{exc.__class__.__name__}: {' '.join(str(exc).split())[:500]}"
        )
        task_filesystem_gc_startup_report = TaskFilesystemGcSweepReport(
            examined=0,
            deleted=0,
            cancelled_referenced=0,
            deferred=0,
            quarantined=0,
        )

    app = FastAPI(title="MARVIS-Agent")
    app.add_middleware(ValidationBatchIngressLimitMiddleware, settings=settings)
    app.state.settings = settings
    app.state.validation_batch_upload_gc = validation_batch_upload_gc
    app.state.validation_batch_upload_recovery = validation_batch_upload_recovery
    app.state.dataset_source_gc = dataset_source_gc
    app.state.dataset_source_gc_startup_report = dataset_source_gc_startup_report
    app.state.dataset_source_gc_startup_error = dataset_source_gc_startup_error
    app.state.task_filesystem_gc = task_filesystem_gc
    app.state.task_filesystem_gc_startup_report = task_filesystem_gc_startup_report
    app.state.task_filesystem_gc_startup_error = task_filesystem_gc_startup_error
    app.state.operations_runtime = build_operations_runtime(
        settings,
        executor_allowlist=operations_executor_allowlist,
        clock=operations_clock,
    )
    app.state.production_activation_verifiers = dict(
        production_activation_verifiers or {}
    )
    app.state.governance_repo = governance_repo
    app.state.governance_reconciliation = governance_reconciliation
    # Per-workspace plugin-admin secret (replaces the old "local-dev" magic
    # header): minted into the workspace on first startup, stored 0600.
    app.state.plugin_admin_token = ensure_plugin_admin_token(settings.plugin_admin_token_path)
    app.state.artifact_recovery_report = artifact_recovery_report.to_dict()
    _configure_plugin_runtime(app, settings)
    _configure_orchestrator(app, settings)
    task_repo = TaskRepository(settings.db_path)
    reclaim_running_plans(
        app.state.plan_repo,
        app.state.reviewer,
        app.state.hook_dispatcher,
        app.state.harness_state,
        task_repo,
    )
    sweep_heartbeat_lost_jobs(task_repo)
    job_watchdog = JobHeartbeatWatchdog(task_repo)
    job_watchdog.start()
    app.state.job_watchdog = job_watchdog
    dataset_source_gc_watchdog = DatasetSourceGcWatchdog(dataset_source_gc)
    app.state.dataset_source_gc_watchdog = dataset_source_gc_watchdog
    task_filesystem_gc_watchdog = TaskFilesystemGcWatchdog(task_filesystem_gc)
    app.state.task_filesystem_gc_watchdog = task_filesystem_gc_watchdog
    app.router.add_event_handler("startup", dataset_source_gc_watchdog.start)
    app.router.add_event_handler("startup", task_filesystem_gc_watchdog.start)
    app.router.add_event_handler("shutdown", task_filesystem_gc_watchdog.stop)
    app.router.add_event_handler("shutdown", dataset_source_gc_watchdog.stop)
    app.router.add_event_handler("shutdown", job_watchdog.stop)
    logger.info("MARVIS startup complete workspace=%s", settings.workspace)

    @app.middleware("http")
    async def _local_access_guard(request, call_next):
        method = request.method.upper()
        path = request.url.path
        is_local = _is_local_client(_effective_client_host(request))
        is_forwarded_trusted_proxy_request = _is_forwarded_request_from_trusted_proxy(request)
        pending_session_token = None
        local_token_authenticated = False
        request.state.local_principal = None
        # GAP-5: on a shared host, "local" alone does not mean "the owning
        # user" -- any other logged-in user's process is also a loopback peer.
        # Keep both the credential-bearing bootstrap and every private read
        # behind the token. Unsafe requests require X-Marvis-Token explicitly:
        # accepting browser-cached Basic credentials there would make state
        # changes vulnerable to cross-site form requests. Left unconfigured
        # (the default), both checks are no-ops.
        local_token = _configured_local_token()
        if local_token and (is_local or is_forwarded_trusted_proxy_request):
            if method in {"GET", "HEAD"}:
                local_token_authenticated = _request_has_valid_local_token(
                    request,
                    local_token,
                    allow_basic=True,
                )
                if not _is_local_token_public_path(path) and not local_token_authenticated:
                    return _local_basic_auth_challenge()
            elif method not in _SAFE_METHODS:
                local_token_authenticated = _request_has_valid_local_token(request, local_token)
                if not local_token_authenticated:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "missing or invalid X-Marvis-Token"},
                    )
        elif not local_token:
            local_token_authenticated = True
        proxy_token_authenticated = bool(
            local_token
            and is_forwarded_trusted_proxy_request
            and local_token_authenticated
        )
        private_client_authenticated = bool(
            (is_local and local_token_authenticated) or proxy_token_authenticated
        )
        request.state.private_client_authenticated = private_client_authenticated
        if not is_local and not proxy_token_authenticated:
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
        # Do not allocate a governance session until the shared-host
        # credential boundary has passed. Otherwise an unauthenticated
        # loopback peer could create unbounded short-lived local principals.
        if (
            private_client_authenticated
            and not path.startswith("/static/")
        ):
            session_token = request.cookies.get(_LOCAL_SESSION_COOKIE)
            try:
                if not session_token:
                    raise AuthorizationError("local session cookie is missing")
                principal = governance_repo.resolve_local_session(session_token)
            except AuthorizationError:
                session = governance_repo.create_local_session()
                principal = session.principal
                pending_session_token = session.token
            request.state.local_principal = principal
        response = await call_next(request)
        if local_token and private_client_authenticated and not path.startswith("/static/"):
            # Authenticated task/data responses are private browser content,
            # not reusable shared-host cache entries.
            response.headers.setdefault("Cache-Control", "no-store")
        if pending_session_token is not None:
            response.set_cookie(
                _LOCAL_SESSION_COOKIE,
                pending_session_token,
                max_age=DEFAULT_SESSION_TTL_SECONDS,
                path="/",
                httponly=True,
                samesite="strict",
            )
        return response

    @app.middleware("http")
    async def _static_cache_control(request, call_next):
        response = await call_next(request)
        # PERF-9: every /static/* response now gets an explicit Cache-Control
        # instead of relying on StaticFiles' implicit (browser-heuristic)
        # caching. Requests that carry the app's ?v= cache-busting query
        # param are safe to cache for a year (the query string changes
        # whenever any JS/CSS file changes, see _static_asset_version); any
        # other /static request -- old cached HTML still pointing at an
        # unversioned URL, or a transitively-imported module reached without
        # a version param -- gets no-cache so the browser always revalidates
        # against the file's ETag/Last-Modified instead of silently reusing
        # a stale copy.
        if request.url.path.startswith("/static/"):
            if request.url.query and "v=" in request.url.query:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "no-cache"
        return response

    app.include_router(api_router)
    app.include_router(agent_memory_router)
    app.include_router(llm_router)
    app.include_router(artifacts_router)
    app.include_router(audit_router)
    app.include_router(branding_router)
    app.include_router(data_router)
    app.include_router(data_analysis_router)
    app.include_router(plugins_router)
    app.include_router(drafts_router)
    app.include_router(evidence_router)
    app.include_router(materials_router)
    app.include_router(modeling_router)
    app.include_router(operations_router)
    app.include_router(plans_router)
    app.include_router(production_governance_router)
    app.include_router(report_fields_router)
    app.include_router(scans_router)
    app.include_router(skills_router)
    app.include_router(stage_controls_router)
    app.include_router(strategy_candidate_lab_router)
    app.include_router(reports_router)
    app.include_router(tasks_router)
    app.include_router(validation_agent_router)
    app.include_router(validation_batches_router)
    app.include_router(validation_contracts_router)
    app.include_router(validation_stages_router)

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
    def health() -> dict[str, object]:
        # Constructing DataBackend ensures the workspace-scoped DuckDB temp
        # directory exists (PERF-8); duckdb_health() then opens a connection the
        # same way every DataBackend operation does (TST-9c: per-operation, not a
        # shared default connection) and reports its actually-effective config.
        duckdb_temp_directory = settings.datasets_dir.parent / DUCKDB_TEMP_DIR_NAME
        DataBackend(settings.datasets_dir)
        stuck_jobs = task_repo.count_heartbeat_stale_running_jobs(
            older_than_seconds=heartbeat_timeout_seconds()
        )
        # S5: adopted strategies past their monitoring cadence. Cheap read of the
        # DB + each plan file; surfaced so the health/task-workbench UI can show an
        # overdue badge (the stuck_jobs badge pattern).
        monitoring_overdue_count = len(StrategyRepository(settings.db_path).list_monitoring_due())
        # GAP-8: llm_configured is a cheap "at least one enabled model with an
        # api_key is saved" check -- it deliberately does NOT make a live LLM
        # call (that belongs to POST /api/settings/llm/test); it only answers
        # "has the user configured anything at all" so the frontend can
        # distinguish "never configured" from "configured but the model is dumb".
        return {
            "status": "ok",
            "stuck_jobs": stuck_jobs,
            "monitoring_overdue_count": monitoring_overdue_count,
            "llm_configured": _has_configured_llm(settings.workspace),
            **sqlite_health(settings.db_path),
            **duckdb_health(duckdb_temp_directory),
        }

    @app.get("/api/strategies/monitoring-due")
    def strategies_monitoring_due() -> dict[str, object]:
        # S5: overdue-strategy detail behind the same local-only guard as the rest
        # of the private surface (_is_local_only_path lists this path; the
        # _local_access_guard middleware rejects a non-loopback caller before the
        # handler runs). Returns the same rows list the health count is derived from.
        due = StrategyRepository(settings.db_path).list_monitoring_due()
        return {"monitoring_due": due, "count": len(due)}

    @app.get("/branding/assets/{asset_path:path}")
    def branding_asset(asset_path: str) -> FileResponse:
        asset = resolve_branding_asset(settings.workspace, asset_path)
        if asset is None:
            raise not_found("branding asset not found")
        return FileResponse(asset)

    @app.get("/")
    def index(request: Request) -> HTMLResponse:
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        static_version = _static_asset_version(static_dir)
        index_html = index_html.replace("__MARVIS_STATIC_VERSION__", static_version)
        index_html = index_html.replace(
            "__MARVIS_STATIC_IMPORT_MAP__", _static_import_map(static_dir, static_version)
        )
        is_private_client = bool(
            getattr(request.state, "private_client_authenticated", False)
        )
        branding = load_branding(settings.workspace)
        if not is_private_client:
            branding = dict(DEFAULT_BRANDING)
        # GAP-5: the access guard reaches this handler for a token-configured
        # local client only after HTTP Basic or X-Marvis-Token authentication.
        # Hand that authenticated UI its token so api.js can keep echoing the
        # existing API header on non-GET requests. Remote clients never receive
        # either credential, even when MARVIS_ALLOW_REMOTE_READ is enabled.
        local_token = _configured_local_token() if is_private_client else ""
        index_html = index_html.replace(
            "__MARVIS_LOCAL_TOKEN__", escape(local_token, quote=True)
        )
        # Hand the local UI its plugin-admin token so draft-tools-panel.js can
        # echo it via X-Marvis-Plugin-Admin on plugin/draft governance calls.
        # Never embedded for a remote client (same reasoning as the local token).
        plugin_admin_token = app.state.plugin_admin_token if is_private_client else ""
        index_html = index_html.replace(
            "__MARVIS_PLUGIN_ADMIN_TOKEN__", escape(plugin_admin_token, quote=True)
        )
        response_headers = None
        if _configured_local_token():
            # The response body contains credentials and varies by incoming
            # authentication, so it must never be stored by a browser/proxy.
            response_headers = {
                "Cache-Control": "no-store",
                "Vary": "Authorization",
            }
        return HTMLResponse(
            render_branded_index_html(index_html, branding),
            headers=response_headers,
        )

    return app


def _configure_plugin_runtime(app: FastAPI, settings: Settings) -> None:
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parent / "packs"
    sync_builtin_packs(plugin_repo, packs_root)
    plugin_registry.load_from_db()
    load_builtin_packs(plugin_registry, packs_root)
    tool_registry = ToolRegistry(plugin_registry)
    plan_repo = PlanRepository(settings.db_path)
    governance_service = GovernanceService(
        plan_repo=plan_repo,
        tool_registry=tool_registry,
        strategy_repo=StrategyRepository(settings.db_path),
        governance_repo=app.state.governance_repo,
    )
    environment = load_execution_environment(settings.workspace)
    python_executable = environment.python_executable or sys.executable
    tool_runner = ToolRunner(
        tool_registry,
        plugin_repo,
        python_executable=python_executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
        plugin_paths=[settings.plugins_dir],
        rss_memory_limit_mb=environment.rss_memory_limit_mb,
        governance=app.state.governance_repo,
        binding_resolver=governance_service,
    )
    hook_dispatcher = HookDispatcher(plugin_registry, tool_runner, plugin_repo)
    hook_dispatcher.rebuild_index()
    draft_repo = DraftRepository(settings.db_path)
    draft_registry = DraftRegistry(draft_repo)
    draft_sandbox = DraftSandbox(tool_runner, draft_registry, draft_repo)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_consolidation_scheduler = ConsolidationScheduler(
        DistillationEngine(memory_store, llm_factory=_llm_factory(settings, role="distill")),
        EvolutionManager(memory_store),
        memory_store,
        auto_enabled=lambda: load_memory_policy(settings.workspace).auto_distill,
    )
    for event in CONSOLIDATION_TRIGGERS:
        hook_dispatcher.register_listener(event, memory_consolidation_scheduler.on_event)
    app.state.plugin_repo = plugin_repo
    app.state.plugin_registry = plugin_registry
    app.state.tool_registry = tool_registry
    app.state.plan_repo = plan_repo
    app.state.governance_service = governance_service
    app.state.tool_runner = tool_runner
    app.state.hook_dispatcher = hook_dispatcher
    app.state.draft_repo = draft_repo
    app.state.draft_registry = draft_registry
    app.state.draft_sandbox = draft_sandbox
    app.state.memory_consolidation_scheduler = memory_consolidation_scheduler
    app.state.plugin_python_executable = python_executable
    app.state.plugin_paths = [settings.plugins_dir]
    app.state.plugin_rss_memory_limit_mb = environment.rss_memory_limit_mb


def _configure_orchestrator(app: FastAPI, settings: Settings) -> None:
    load_builtin_templates()
    clear_user_templates()
    plan_repo = app.state.plan_repo
    plan_validator = PlanValidator(app.state.tool_registry)
    skill_report = load_user_skill_templates(
        settings.workspace,
        app.state.tool_registry,
        plan_validator,
    )
    # LLM-4: each orchestrator role gets its own factory so a role_overrides
    # entry (settings/llm.json) can route it to a smaller/larger model; with no
    # override configured every factory still resolves to default_model_id,
    # i.e. today's single-model behavior is preserved unchanged.
    planner_llm_factory = _llm_factory(settings, role="planner")
    reviewer_llm_factory = _llm_factory(settings, role="critic")
    router_llm_factory = _llm_factory(settings, role="router_intent")
    intent_router = IntentRouter(router_llm_factory, app.state.tool_registry)
    planner = Planner(app.state.tool_registry, planner_llm_factory, plan_validator)
    reviewer = Reviewer(reviewer_llm_factory)
    harness_state = HarnessState(plan_repo)

    def executor_factory(restricted_registry):
        restricted_runner = ToolRunner(
            restricted_registry,
            app.state.plugin_repo,
            python_executable=app.state.plugin_python_executable,
            datasets_root=settings.datasets_dir,
            workspace=settings.workspace,
            plugin_paths=app.state.plugin_paths,
            rss_memory_limit_mb=app.state.plugin_rss_memory_limit_mb,
            governance=app.state.governance_repo,
            binding_resolver=app.state.governance_service,
        )
        return PlanExecutor(
            plan_repo,
            restricted_runner,
            reviewer,
            None,
            app.state.hook_dispatcher,
            harness_state,
            authorizer=app.state.governance_service,
        )

    def subagent_planner_factory(restricted_registry):
        return Planner(
            restricted_registry,
            planner_llm_factory,
            PlanValidator(restricted_registry),
        )

    subagent_dispatcher = SubAgentDispatcher(
        plan_repo,
        planner,
        executor_factory,
        app.state.tool_registry,
        intent_router,
        planner_factory=subagent_planner_factory,
    )
    plan_executor = PlanExecutor(
        plan_repo,
        app.state.tool_runner,
        reviewer,
        subagent_dispatcher,
        app.state.hook_dispatcher,
        harness_state,
        planner=planner,
        authorizer=app.state.governance_service,
    )
    app.state.plan_repo = plan_repo
    app.state.plan_validator = plan_validator
    app.state.skill_report = skill_report
    app.state.intent_router = intent_router
    app.state.planner = planner
    app.state.reviewer = reviewer
    app.state.harness_state = harness_state
    app.state.subagent_dispatcher = subagent_dispatcher
    app.state.plan_executor = plan_executor


def _llm_factory(settings: Settings, *, role: str | None = None):
    """Build an LLM client factory for one orchestrator role.

    LLM-4: ``role`` is resolved against settings/llm.json's role_overrides
    (falls back to default_model_id when unmapped — see
    marvis.llm_settings.resolve_llm_model), so different components (planner,
    critic, router_intent, distill, ...) can be routed to different models
    without touching call sites beyond this assembly point.
    """
    db_path = settings.db_path

    def _record(record: dict) -> None:
        try:
            record_llm_call(db_path, record)
        except Exception:
            # Observability writes must never break an orchestration call.
            pass

    def factory():
        return OpenAICompatibleLLMClient(
            resolve_llm_model(settings.workspace, role=role),
            on_call_recorded=_record,
        )

    return factory
