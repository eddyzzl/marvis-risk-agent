from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4


MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff ]{1,64}$")


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
        api_key = _api_key_from_payload(raw_model, old_model)
        models.append(
            {
                "model_id": model_id,
                "enabled": bool(raw_model.get("enabled", True)),
                "display_name": str(
                    raw_model.get("display_name")
                    or raw_model.get("model_name")
                    or model_id
                ).strip(),
                "provider": str(raw_model.get("provider") or "OpenAI Compatible").strip(),
                "api_base_url": str(raw_model.get("api_base_url") or "").strip().rstrip("/"),
                "model_name": str(raw_model.get("model_name") or "").strip(),
                "api_key": api_key,
                "timeout_seconds": _timeout_seconds(raw_model.get("timeout_seconds")),
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
    private_settings = {
        "default_model_id": default_model_id,
        "models": models,
    }
    path = _settings_path(workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(private_settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _public_settings(private_settings)


def resolve_llm_model(workspace: str | Path, model_id: str | None = None) -> dict:
    settings = _load_private_settings(Path(workspace))
    models = [
        model for model in settings.get("models", [])
        if isinstance(model, dict) and model.get("enabled")
    ]
    if not models:
        raise LLMSettingsError("请先在设置中配置至少一个启用的大模型")
    selected_id = (model_id or settings.get("default_model_id") or "").strip()
    if not selected_id:
        selected_id = str(models[0].get("model_id") or "")
    selected = next(
        (model for model in models if model.get("model_id") == selected_id),
        None,
    )
    if selected is None or not selected.get("api_key"):
        raise LLMSettingsError("当前选择的模型不可用，请在对话框中选择其他模型或到设置中检查配置")
    if not selected.get("api_base_url") or not selected.get("model_name"):
        raise LLMSettingsError("当前选择的模型缺少 API Base URL 或模型名")
    return dict(selected)


def _settings_path(workspace: Path) -> Path:
    return workspace / "settings" / "llm.json"


def _load_private_settings(workspace: Path) -> dict:
    path = _settings_path(workspace)
    if not path.exists():
        return {"default_model_id": "", "models": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMSettingsError(f"大模型配置读取失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise LLMSettingsError("大模型配置必须是 JSON 对象")
    return {
        "default_model_id": str(payload.get("default_model_id") or ""),
        "models": [
            model for model in payload.get("models", [])
            if isinstance(model, dict)
        ],
    }


def _public_settings(private_settings: dict) -> dict:
    models = [_public_model(model) for model in private_settings.get("models", [])]
    enabled_models = [model for model in models if model["enabled"]]
    return {
        "default_model_id": str(private_settings.get("default_model_id") or ""),
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
        "timeout_seconds": int(model.get("timeout_seconds") or 60),
        "has_api_key": bool(model.get("api_key")),
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


def _timeout_seconds(value) -> int:
    try:
        timeout = int(value or 60)
    except (TypeError, ValueError):
        timeout = 60
    return min(max(timeout, 5), 300)
