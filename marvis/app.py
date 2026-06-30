import ipaddress
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
from marvis.db import DraftRepository, PlanRepository, PluginRepository, init_db, sqlite_health
from marvis.drafts.registry import DraftRegistry
from marvis.drafts.sandbox import DraftSandbox
from marvis.execution_environment import load_execution_environment
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import resolve_llm_model
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
from marvis.recovery import reclaim_stale_running_tasks
from marvis.routers.agent_memory import router as agent_memory_router
from marvis.routers.branding import router as branding_router
from marvis.routers.data import router as data_router
from marvis.routers.drafts import router as drafts_router
from marvis.routers.plans import router as plans_router
from marvis.routers.plugins import router as plugins_router
from marvis.routers.skills import router as skills_router
from marvis.routers.tasks import router as tasks_router
from marvis.settings import Settings, build_settings
from marvis.state_machine import IllegalTransition


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_NAMED_LOCAL_HOSTS = {"localhost", "testclient"}
_REMOTE_READ_ENV = "MARVIS_ALLOW_REMOTE_READ"
_TRUSTED_PROXY_ENV = "MARVIS_TRUSTED_PROXY_HOSTS"
_FORWARDED_CLIENT_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")
_STATIC_VERSION_FILES = (
    "app.js",
    "styles.css",
    "css/welcome.css",
    "css/v2-workbench.css",
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
    mtimes = []
    for relative_path in _STATIC_VERSION_FILES:
        try:
            mtimes.append((static_dir / relative_path).stat().st_mtime_ns)
        except OSError:
            continue
    if not mtimes:
        return __version__
    return f"{__version__}-{max(mtimes)}"


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
    init_db(settings.db_path)
    reclaim_stale_running_tasks(settings.db_path)
    artifact_recovery_report = reconcile_workspace_artifacts(settings)

    app = FastAPI(title="MARVIS-Agent")
    app.state.settings = settings
    app.state.artifact_recovery_report = artifact_recovery_report.to_dict()
    _configure_plugin_runtime(app, settings)
    _configure_orchestrator(app, settings)

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
    app.include_router(agent_memory_router)
    app.include_router(branding_router)
    app.include_router(data_router)
    app.include_router(plugins_router)
    app.include_router(drafts_router)
    app.include_router(plans_router)
    app.include_router(skills_router)
    app.include_router(tasks_router)

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
        return {"status": "ok", **sqlite_health(settings.db_path)}

    @app.get("/branding/assets/{asset_path:path}")
    def branding_asset(asset_path: str) -> FileResponse:
        asset = resolve_branding_asset(settings.workspace, asset_path)
        if asset is None:
            raise HTTPException(status_code=404, detail="branding asset not found")
        return FileResponse(asset)

    @app.get("/")
    def index(request: Request) -> HTMLResponse:
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        index_html = index_html.replace("__MARVIS_STATIC_VERSION__", _static_asset_version(static_dir))
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
    )
    hook_dispatcher = HookDispatcher(plugin_registry, tool_runner, plugin_repo)
    hook_dispatcher.rebuild_index()
    draft_repo = DraftRepository(settings.db_path)
    draft_registry = DraftRegistry(draft_repo)
    draft_sandbox = DraftSandbox(tool_runner, draft_registry, draft_repo)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_consolidation_scheduler = ConsolidationScheduler(
        DistillationEngine(memory_store),
        EvolutionManager(memory_store),
        memory_store,
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
    llm_factory = _llm_factory(settings)
    intent_router = IntentRouter(llm_factory, app.state.tool_registry)
    planner = Planner(app.state.tool_registry, llm_factory, plan_validator)
    reviewer = Reviewer(llm_factory)
    harness_state = HarnessState(plan_repo)

    def executor_factory(restricted_registry):
        restricted_runner = ToolRunner(
            restricted_registry,
            app.state.plugin_repo,
            python_executable=app.state.plugin_python_executable,
            datasets_root=settings.datasets_dir,
            workspace=settings.workspace,
            plugin_paths=app.state.plugin_paths,
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
            llm_factory,
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


def _llm_factory(settings: Settings):
    def factory():
        return OpenAICompatibleLLMClient(resolve_llm_model(settings.workspace))

    return factory
