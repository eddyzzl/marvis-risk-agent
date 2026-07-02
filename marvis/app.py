import ipaddress
import json
import logging
from pathlib import Path
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from marvis import __version__
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
from marvis.data.backend import DataBackend, duckdb_health
from marvis.db import (
    DraftRepository,
    record_llm_call,
    PlanRepository,
    PluginRepository,
    TaskRepository,
    init_db,
    sqlite_health,
)
from marvis.drafts.registry import DraftRegistry
from marvis.drafts.sandbox import DraftSandbox
from marvis.execution_environment import load_execution_environment
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
from marvis.plugins.hooks import HookDispatcher
from marvis.plugins.loader import load_builtin_packs, sync_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.job_watchdog import (
    JobHeartbeatWatchdog,
    heartbeat_timeout_seconds,
    sweep_heartbeat_lost_jobs,
)
from marvis.recovery import reclaim_running_plans, reclaim_stale_running_tasks
from marvis.routers.agent_memory import router as agent_memory_router
from marvis.routers.llm import router as llm_router
from marvis.routers.artifacts import router as artifacts_router
from marvis.routers.branding import router as branding_router
from marvis.routers.data import router as data_router
from marvis.routers.drafts import router as drafts_router
from marvis.routers.evidence import router as evidence_router
from marvis.routers.materials import router as materials_router
from marvis.routers.modeling import router as modeling_router
from marvis.routers.plans import router as plans_router
from marvis.routers.plugins import router as plugins_router
from marvis.routers.report_fields import router as report_fields_router
from marvis.routers.reports import router as reports_router
from marvis.routers.scans import router as scans_router
from marvis.routers.skills import router as skills_router
from marvis.routers.stage_controls import router as stage_controls_router
from marvis.routers.tasks import router as tasks_router
from marvis.routers.validation_agent import router as validation_agent_router
from marvis.routers.validation_stages import router as validation_stages_router
from marvis.settings import Settings, build_settings
from marvis.state_machine import IllegalTransition


logger = logging.getLogger(__name__)

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_NAMED_LOCAL_HOSTS = {"localhost", "testclient"}
_REMOTE_READ_ENV = "MARVIS_ALLOW_REMOTE_READ"
_TRUSTED_PROXY_ENV = "MARVIS_TRUSTED_PROXY_HOSTS"
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


def _remote_read_enabled() -> bool:
    return os.environ.get(_REMOTE_READ_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_public_read_path(path: str) -> bool:
    return path == "/" or path == "/api/health" or path.startswith("/static/")


def _static_asset_version(static_dir: Path) -> str:
    # PERF-9: scan every JS/CSS file under static/ (recursively, so js/v2/*
    # and any future subdirectory are included) instead of a hardcoded
    # 4-file allowlist -- editing any module now changes the version string,
    # eliminating the "changed a v2 module, browser kept the old file"
    # staleness window described in the review.
    mtimes = []
    for glob in _STATIC_VERSION_GLOBS:
        for path in static_dir.rglob(glob):
            try:
                mtimes.append(path.stat().st_mtime_ns)
            except OSError:
                continue
    if not mtimes:
        return __version__
    return f"{__version__}-{max(mtimes)}"


def _static_import_map(static_dir: Path, version: str) -> str:
    # PERF-9: app.js and every js/v2/*.js controller import sibling modules
    # via bare relative specifiers (e.g. "./js/api.js", "../ui-utils.js")
    # with no version query string, so bumping _static_asset_version alone
    # does not change the URL the browser fetches those modules from --
    # only the entry point (app.js) gets a fresh URL, while everything it
    # imports keeps resolving to the same unversioned path and can keep
    # running stale code after an upgrade. A browser-native import map
    # rewrites every "./js/*.js" / "../*.js" specifier actually used in the
    # tree to a `?v=<version>` URL, so editing any module changes the URL
    # for every module that (transitively) imports it -- without editing
    # app.js's or the controllers' source, which a large slice of the
    # frontend test suite greps verbatim (test_frontend_shell_static.py's
    # import inventory, etc.) and would otherwise break wholesale.
    js_dir = static_dir / "js"
    v2_dir = js_dir / "v2"
    js_files = sorted(p.name for p in js_dir.glob("*.js") if p.is_file())
    v2_files = sorted(p.name for p in v2_dir.glob("*.js") if p.is_file()) if v2_dir.is_dir() else []

    def versioned(relative: str) -> str:
        return f"static/{relative}?v={version}"

    # Scope for modules importing from static/ itself (app.js).
    top_imports = {f"./js/{name}": versioned(f"js/{name}") for name in js_files}
    top_imports.update({f"./js/v2/{name}": versioned(f"js/v2/{name}") for name in v2_files})

    # Scope for modules importing from static/js/ (e.g. create-task-dialog.js).
    js_scope_imports = {f"./{name}": versioned(f"js/{name}") for name in js_files}

    # Scope for modules importing from static/js/v2/ (e.g. plan_rail_controller.js).
    v2_scope_imports = {f"./{name}": versioned(f"js/v2/{name}") for name in v2_files}
    v2_scope_imports.update({f"../{name}": versioned(f"js/{name}") for name in js_files})

    import_map = {
        "imports": top_imports,
        "scopes": {
            "static/js/": js_scope_imports,
            "static/js/v2/": v2_scope_imports,
        },
    }
    return json.dumps(import_map, ensure_ascii=False, separators=(",", ":"))


def _is_local_only_path(path: str) -> bool:
    """Paths that stay local even when remote read is enabled: system
    configuration and private branding are never exposed remotely. `/api/branding`
    is included because its JSON carries private workspace branding incl. validator
    aliases (real names); the branding asset files under `/branding/` are already
    local-only, so the metadata route must match them."""
    return (
        path == "/api/branding"
        or path.startswith("/api/settings")
        or path == "/api/skills/reload"
        or path == "/api/skills/validate"
        or path.startswith("/branding/")
    )


def create_app(workspace: str | Path | Settings) -> FastAPI:
    settings = workspace if isinstance(workspace, Settings) else build_settings(workspace)
    logger.info("MARVIS starting up workspace=%s version=%s", settings.workspace, __version__)
    init_db(settings.db_path)
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

    app = FastAPI(title="MARVIS-Agent")
    app.state.settings = settings
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
    logger.info("MARVIS startup complete workspace=%s", settings.workspace)

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
    app.include_router(branding_router)
    app.include_router(data_router)
    app.include_router(plugins_router)
    app.include_router(drafts_router)
    app.include_router(evidence_router)
    app.include_router(materials_router)
    app.include_router(modeling_router)
    app.include_router(plans_router)
    app.include_router(report_fields_router)
    app.include_router(scans_router)
    app.include_router(skills_router)
    app.include_router(stage_controls_router)
    app.include_router(reports_router)
    app.include_router(tasks_router)
    app.include_router(validation_agent_router)
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
        # Constructing DataBackend applies the DuckDB memory_limit / threads /
        # temp_directory PRAGMAs (PERF-8) if this process has not touched a
        # dataset yet, so health always reports the actually-effective config.
        DataBackend(settings.datasets_dir)
        stuck_jobs = task_repo.count_heartbeat_stale_running_jobs(
            older_than_seconds=heartbeat_timeout_seconds()
        )
        # GAP-8: llm_configured is a cheap "at least one enabled model with an
        # api_key is saved" check -- it deliberately does NOT make a live LLM
        # call (that belongs to POST /api/settings/llm/test); it only answers
        # "has the user configured anything at all" so the frontend can
        # distinguish "never configured" from "configured but the model is dumb".
        return {
            "status": "ok",
            "stuck_jobs": stuck_jobs,
            "llm_configured": _has_configured_llm(settings.workspace),
            **sqlite_health(settings.db_path),
            **duckdb_health(),
        }

    @app.get("/branding/assets/{asset_path:path}")
    def branding_asset(asset_path: str) -> FileResponse:
        asset = resolve_branding_asset(settings.workspace, asset_path)
        if asset is None:
            raise HTTPException(status_code=404, detail="branding asset not found")
        return FileResponse(asset)

    @app.get("/")
    def index(request: Request) -> HTMLResponse:
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        static_version = _static_asset_version(static_dir)
        index_html = index_html.replace("__MARVIS_STATIC_VERSION__", static_version)
        index_html = index_html.replace(
            "__MARVIS_STATIC_IMPORT_MAP__", _static_import_map(static_dir, static_version)
        )
        branding = load_branding(settings.workspace)
        if not _is_local_client(_effective_client_host(request)):
            branding = dict(DEFAULT_BRANDING)
        return HTMLResponse(
            render_branded_index_html(index_html, branding)
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
    plan_repo = PlanRepository(settings.db_path)
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
        )
        return PlanExecutor(
            plan_repo,
            restricted_runner,
            reviewer,
            None,
            app.state.hook_dispatcher,
            harness_state,
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
