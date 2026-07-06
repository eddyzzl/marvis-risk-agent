from dataclasses import asdict

from fastapi import APIRouter, Request

from marvis.errors import not_found, unprocessable

from marvis.api_schemas import (
    ExecutionEnvironmentRequest,
    LLMConnectionTestRequest,
    LLMSettingsRequest,
    MemoryPolicyRequest,
    model_payload,
)
from marvis.execution_environment import (
    ExecutionEnvironmentSettings,
    detect_execution_environment_options,
    load_execution_environment,
    save_execution_environment,
    validate_execution_environment,
)
from marvis.llm_client import test_llm_connection
from marvis.llm_settings import (
    LLMSettingsError,
    load_llm_settings,
    resolve_llm_model,
    save_llm_settings,
)
from marvis.memory_policy import (
    MemoryPolicySettings,
    load_memory_policy,
    save_memory_policy,
)


router = APIRouter()


@router.get("/settings/execution-environment")
def get_execution_environment_settings(request: Request) -> dict:
    settings = load_execution_environment(request.app.state.settings.workspace)
    validation = validate_execution_environment(settings)
    return {
        "settings": asdict(settings),
        "validation": asdict(validation),
    }


@router.get("/settings/execution-environment/options")
def get_execution_environment_options(request: Request) -> dict:
    settings = load_execution_environment(request.app.state.settings.workspace)
    validation = validate_execution_environment(settings)
    options = detect_execution_environment_options()
    return {
        "settings": asdict(settings),
        "validation": asdict(validation),
        "options": [asdict(option) for option in options],
    }


@router.put("/settings/execution-environment")
def update_execution_environment_settings(
    payload: ExecutionEnvironmentRequest,
    request: Request,
) -> dict:
    workspace = request.app.state.settings.workspace
    # Preserve the plugin RSS limit, which this request never carries -- otherwise
    # saving a kernel selection would silently reset it to the default.
    current = load_execution_environment(workspace)
    settings = ExecutionEnvironmentSettings(
        execution_mode=payload.execution_mode,
        kernel_name=payload.kernel_name,
        conda_env_name=payload.conda_env_name,
        python_executable=payload.python_executable,
        rss_memory_limit_mb=current.rss_memory_limit_mb,
        notebook_memory_limit_mb=payload.notebook_memory_limit_mb,
    )
    saved = save_execution_environment(workspace, settings)
    validation = validate_execution_environment(saved)
    return {
        "settings": asdict(saved),
        "validation": asdict(validation),
    }


@router.get("/settings/llm")
def get_llm_settings(request: Request) -> dict:
    try:
        return load_llm_settings(request.app.state.settings.workspace)
    except LLMSettingsError as exc:
        raise unprocessable(str(exc)) from exc


@router.put("/settings/llm")
def update_llm_settings(payload: LLMSettingsRequest, request: Request) -> dict:
    try:
        return save_llm_settings(
            request.app.state.settings.workspace,
            model_payload(payload),
        )
    except LLMSettingsError as exc:
        raise unprocessable(str(exc)) from exc


# GAP-8: preflight connection check, called either from the "add/edit model"
# dialog (inline candidate profile, not yet saved) or against an already-saved
# model_id -- so a mistyped base_url/port/model name is caught at
# configuration time instead of surfacing later as a silent agent degradation.
@router.post("/settings/llm/test")
def test_llm_settings_connection(payload: LLMConnectionTestRequest, request: Request) -> dict:
    workspace = request.app.state.settings.workspace
    profile: dict = {
        "api_base_url": payload.api_base_url,
        "model_name": payload.model_name,
        "api_key": payload.api_key,
    }
    if payload.model_id:
        try:
            saved = resolve_llm_model(workspace, payload.model_id)
        except LLMSettingsError as exc:
            raise not_found(str(exc)) from exc
        profile = {**saved, **{k: v for k, v in profile.items() if v}}
    result = test_llm_connection(profile, timeout_seconds=min(max(payload.timeout_seconds, 1), 60))
    return {
        "ok": result.ok,
        "latency_ms": result.latency_ms,
        "model_echo": result.model_echo,
        "error_kind": result.error_kind,
        "error_detail": result.error_detail,
    }


@router.get("/settings/memory-policy")
def get_memory_policy_settings(request: Request) -> dict:
    # Create-on-read safe: if no settings file exists, this returns defaults
    # (both flags on) without writing anything to disk.
    settings = load_memory_policy(request.app.state.settings.workspace)
    return {"settings": asdict(settings)}


@router.put("/settings/memory-policy")
def update_memory_policy_settings(
    payload: MemoryPolicyRequest,
    request: Request,
) -> dict:
    settings = MemoryPolicySettings(
        reference_cross_task=payload.reference_cross_task,
        auto_distill=payload.auto_distill,
    )
    saved = save_memory_policy(request.app.state.settings.workspace, settings)
    # Unlike execution-environment (which returns {settings, validation}),
    # memory-policy returns only {settings} -- there is nothing to validate.
    return {"settings": asdict(saved)}
