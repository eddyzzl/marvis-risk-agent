from __future__ import annotations

import json
import os
import re
from pathlib import Path
from uuid import uuid4


MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff ]{1,64}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

# LLM-4: per-task-tier role -> model_id routing. A role not present here (or not
# mapped in the saved role_overrides) simply falls back to default_model_id — this
# is the existing behavior, so an empty/absent role_overrides is a no-op.
# "unknown" covers the llm_client default `caller` tag for any call site that
# hasn't been given a more specific tag yet.
KNOWN_LLM_ROLES = (
    "planner",
    "gate",
    "router",
    "router_intent",
    "critic",
    "reviewer_summary",
    "author",
    "distill",
    "learn",
    "cross",
    "narrative",
    "unknown",
)


class LLMSettingsError(ValueError):
    pass


def load_llm_settings(workspace: str | Path) -> dict:
    return _public_settings(_load_private_settings(Path(workspace)))


def load_private_llm_settings(workspace: str | Path) -> dict:
    return _load_private_settings(Path(workspace))


def save_llm_settings(workspace: str | Path, payload: dict) -> dict:
    workspace_path = Path(workspace)
    existing = _load_private_settings(workspace_path)
    existing_by_id = {
        str(model.get("model_id")): model
        for model in existing.get("models", [])
        if isinstance(model, dict) and model.get("model_id")
    }
    models: list[dict] = []
    seen_ids: set[str] = set()
    for raw_model in payload.get("models", []):
        if not isinstance(raw_model, dict):
            raise LLMSettingsError("model profile must be an object")
        model_id = _model_id(raw_model)
        if model_id in seen_ids:
            raise LLMSettingsError(f"duplicate model_id: {model_id}")
        seen_ids.add(model_id)
        old_model = existing_by_id.get(model_id, {})
        api_key_env = _api_key_env_from_payload(raw_model, old_model)
        api_key = _api_key_from_payload(raw_model, old_model)
        if api_key_env:
            api_key = ""
        models.append(
            {
                "model_id": model_id,
                "enabled": bool(
                    raw_model["enabled"]
                    if "enabled" in raw_model
                    else old_model.get("enabled", True)
                ),
                "display_name": str(
                    raw_model.get("display_name")
                    or raw_model.get("model_name")
                    or model_id
                ).strip(),
                "provider": str(raw_model.get("provider") or "OpenAI Compatible").strip(),
                "api_base_url": str(raw_model.get("api_base_url") or "").strip().rstrip("/"),
                "model_name": str(raw_model.get("model_name") or "").strip(),
                "api_key": api_key,
                "api_key_env": api_key_env,
                "enable_thinking": bool(raw_model.get("enable_thinking")),
                "reasoning_effort": _reasoning_effort(raw_model.get("reasoning_effort")),
                "structured_output": _structured_output(raw_model.get("structured_output")),
                "thinking_style": _thinking_style(raw_model.get("thinking_style")),
                "timeout_seconds": _timeout_seconds(raw_model.get("timeout_seconds")),
                "context_window": _context_window(raw_model.get("context_window")),
                "max_output_tokens": _max_output_tokens(raw_model.get("max_output_tokens")),
            }
        )
    default_model_id = str(payload.get("default_model_id") or "").strip()
    if default_model_id and default_model_id not in seen_ids:
        default_model_id = ""
    if not default_model_id:
        default_model_id = next(
            (model["model_id"] for model in models if model["enabled"]),
            models[0]["model_id"] if models else "",
        )
    role_overrides = _role_overrides(
        payload.get("role_overrides")
        if "role_overrides" in payload
        else existing.get("role_overrides"),
        seen_ids,
    )
    private_settings = {
        "default_model_id": default_model_id,
        "capability_tier": str(
            payload.get("capability_tier") or existing.get("capability_tier") or ""
        ).strip(),
        "role_overrides": role_overrides,
        "models": models,
    }
    path = _settings_path(workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(private_settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _public_settings(private_settings)


def resolve_llm_model(
    workspace: str | Path,
    model_id: str | None = None,
    *,
    role: str | None = None,
) -> dict:
    """Resolve the model profile to use for one LLM call.

    LLM-4 task-tier routing: an explicit ``model_id`` always wins (e.g. a
    per-request model_id from the UI). Otherwise, when ``role`` is given and
    ``role_overrides`` maps it to a model_id, that model is used; falling back
    to ``default_model_id`` when the role is unmapped (today's behavior — an
    empty/absent role_overrides is a no-op, every role routes to the default
    model exactly as before this feature existed).
    """
    settings = _load_private_settings(Path(workspace))
    models = [
        model for model in settings.get("models", [])
        if isinstance(model, dict) and model.get("enabled")
    ]
    if not models:
        raise LLMSettingsError("请先在设置中配置至少一个启用的大模型")
    role_overrides = settings.get("role_overrides") or {}
    role_model_id = str(role_overrides.get(role) or "").strip() if role else ""
    selected_id = (
        model_id
        or role_model_id
        or settings.get("default_model_id")
        or ""
    ).strip()
    if not selected_id:
        selected_id = str(models[0].get("model_id") or "")
    selected = next(
        (model for model in models if model.get("model_id") == selected_id),
        None,
    )
    if selected is None:
        raise LLMSettingsError("当前选择的模型不可用，请在对话框中选择其他模型或到设置中检查配置")
    selected = dict(selected)
    selected["api_key"] = _resolved_api_key(selected)
    if not selected.get("api_key"):
        raise LLMSettingsError("当前选择的模型不可用，请在对话框中选择其他模型或到设置中检查配置")
    if not selected.get("api_base_url") or not selected.get("model_name"):
        raise LLMSettingsError("当前选择的模型缺少 API Base URL 或模型名")
    return selected


def _settings_path(workspace: Path) -> Path:
    return workspace / "settings" / "llm.json"


def _load_private_settings(workspace: Path) -> dict:
    path = _settings_path(workspace)
    if not path.exists():
        return {"default_model_id": "", "role_overrides": {}, "models": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMSettingsError(f"大模型配置读取失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise LLMSettingsError("大模型配置必须是 JSON 对象")
    models = [
        model for model in payload.get("models", [])
        if isinstance(model, dict)
    ]
    seen_ids = {
        str(model.get("model_id"))
        for model in models
        if model.get("model_id")
    }
    return {
        "default_model_id": str(payload.get("default_model_id") or ""),
        "capability_tier": str(payload.get("capability_tier") or ""),
        "role_overrides": _role_overrides(payload.get("role_overrides"), seen_ids),
        "models": models,
    }


def _public_settings(private_settings: dict) -> dict:
    models = [_public_model(model) for model in private_settings.get("models", [])]
    enabled_models = [model for model in models if model["enabled"]]
    return {
        "default_model_id": str(private_settings.get("default_model_id") or ""),
        "capability_tier": str(private_settings.get("capability_tier") or ""),
        "role_overrides": dict(private_settings.get("role_overrides") or {}),
        "models": models,
        "enabled_models": enabled_models,
    }


def _public_model(model: dict) -> dict:
    return {
        "model_id": str(model.get("model_id") or ""),
        "enabled": bool(model.get("enabled")),
        "display_name": str(model.get("display_name") or ""),
        "provider": str(model.get("provider") or ""),
        "api_base_url": str(model.get("api_base_url") or ""),
        "model_name": str(model.get("model_name") or ""),
        "api_key_env": str(model.get("api_key_env") or ""),
        "enable_thinking": bool(model.get("enable_thinking")),
        "reasoning_effort": _reasoning_effort(model.get("reasoning_effort")),
        "structured_output": _structured_output(model.get("structured_output")),
        "thinking_style": _thinking_style(model.get("thinking_style")),
        "timeout_seconds": int(model.get("timeout_seconds") or 60),
        "context_window": _context_window(model.get("context_window")),
        "max_output_tokens": _max_output_tokens(model.get("max_output_tokens")),
        "has_api_key": bool(_resolved_api_key(model)),
    }


def _model_id(raw_model: dict) -> str:
    model_id = str(raw_model.get("model_id") or "").strip()
    if not model_id:
        model_id = f"model-{uuid4().hex[:10]}"
    if not MODEL_ID_RE.match(model_id):
        raise LLMSettingsError(f"invalid model_id: {model_id}")
    return model_id


def _api_key_from_payload(raw_model: dict, old_model: dict) -> str:
    if "api_key" in raw_model:
        return str(raw_model.get("api_key") or "")
    if raw_model.get("has_api_key") and old_model.get("api_key"):
        return str(old_model.get("api_key") or "")
    return ""


def _api_key_env_from_payload(raw_model: dict, old_model: dict) -> str:
    if "api_key_env" in raw_model:
        value = str(raw_model.get("api_key_env") or "").strip()
    else:
        value = str(old_model.get("api_key_env") or "").strip()
    if value and not ENV_NAME_RE.match(value):
        raise LLMSettingsError(f"invalid api_key_env: {value}")
    return value


def _resolved_api_key(model: dict) -> str:
    api_key = str(model.get("api_key") or "")
    if api_key:
        return api_key
    api_key_env = str(model.get("api_key_env") or "").strip()
    if not api_key_env:
        return ""
    return os.environ.get(api_key_env, "")


def _timeout_seconds(value) -> int:
    try:
        timeout = int(value or 60)
    except (TypeError, ValueError):
        timeout = 60
    return min(max(timeout, 5), 300)


# LLM-5: context_window/max_output_tokens defaults mirror marvis.llm_client's
# DEFAULT_CONTEXT_WINDOW/DEFAULT_MAX_OUTPUT_TOKENS (32768 / 2048) so a profile
# saved without these fields behaves identically to the client-side default.
_DEFAULT_CONTEXT_WINDOW = 32768
_DEFAULT_MAX_OUTPUT_TOKENS = 2048
_MIN_CONTEXT_WINDOW = 1024
_MAX_CONTEXT_WINDOW = 2_000_000
_MIN_MAX_OUTPUT_TOKENS = 16
_MAX_MAX_OUTPUT_TOKENS = 200_000


def _context_window(value) -> int:
    try:
        window = int(value) if value else _DEFAULT_CONTEXT_WINDOW
    except (TypeError, ValueError):
        window = _DEFAULT_CONTEXT_WINDOW
    return min(max(window, _MIN_CONTEXT_WINDOW), _MAX_CONTEXT_WINDOW)


def _max_output_tokens(value) -> int:
    try:
        tokens = int(value) if value else _DEFAULT_MAX_OUTPUT_TOKENS
    except (TypeError, ValueError):
        tokens = _DEFAULT_MAX_OUTPUT_TOKENS
    return min(max(tokens, _MIN_MAX_OUTPUT_TOKENS), _MAX_MAX_OUTPUT_TOKENS)


def _role_overrides(value, known_model_ids: set[str]) -> dict[str, str]:
    """Validate a role_overrides payload: known roles + known model_ids only.

    Unknown roles/model_ids are silently dropped rather than raising — a saved
    override that refers to a since-deleted model_id should not brick the
    whole settings save; resolve_llm_model already falls back to
    default_model_id when a role is unmapped.
    """
    if not isinstance(value, dict):
        return {}
    overrides: dict[str, str] = {}
    for role, model_id in value.items():
        role_name = str(role or "").strip()
        target_id = str(model_id or "").strip()
        if role_name not in KNOWN_LLM_ROLES or not target_id:
            continue
        if known_model_ids and target_id not in known_model_ids:
            continue
        overrides[role_name] = target_id
    return overrides


_VALID_REASONING_EFFORTS = ("low", "medium", "high")


def _reasoning_effort(value) -> str:
    effort = str(value or "").strip().lower()
    return effort if effort in _VALID_REASONING_EFFORTS else "high"


_VALID_STRUCTURED_OUTPUTS = ("json_schema", "json_object", "none")


def _structured_output(value) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in _VALID_STRUCTURED_OUTPUTS else "json_object"


_VALID_THINKING_STYLES = ("qwen_chat_template", "openai_reasoning", "anthropic", "none")


def _thinking_style(value) -> str:
    style = str(value or "").strip().lower()
    return style if style in _VALID_THINKING_STYLES else "none"
