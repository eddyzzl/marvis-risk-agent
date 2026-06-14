from riskmodel_checker.llm_settings import (
    load_llm_settings,
    resolve_llm_model,
    save_llm_settings,
)


def test_llm_settings_round_trip_masks_api_key(tmp_path):
    saved = save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "enable_thinking": True,
                    "timeout_seconds": 45,
                },
                {
                    "model_id": "model-b",
                    "enabled": False,
                    "display_name": "备用模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "backup-gpt",
                    "api_key": "backup-secret",
                    "timeout_seconds": 30,
                },
            ],
        },
    )

    assert saved["default_model_id"] == "model-a"
    assert saved["models"][0]["has_api_key"] is True
    assert saved["models"][0]["enable_thinking"] is True
    assert "api_key" not in saved["models"][0]
    assert [model["model_id"] for model in saved["enabled_models"]] == ["model-a"]

    loaded = load_llm_settings(tmp_path)

    assert loaded == saved


def test_llm_settings_update_preserves_existing_key_when_masked(tmp_path):
    save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "display_name": "旧名称",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "old-model",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                }
            ],
        },
    )

    saved = save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "display_name": "新名称",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "new-model",
                    "has_api_key": True,
                    "timeout_seconds": 60,
                }
            ],
        },
    )

    assert saved["models"][0]["display_name"] == "新名称"
    assert saved["models"][0]["has_api_key"] is True
    raw = (tmp_path / "settings" / "llm.json").read_text(encoding="utf-8")
    assert "secret" in raw
    assert "api_key" not in saved["models"][0]


def test_llm_settings_update_preserves_enabled_when_payload_omits_it(tmp_path):
    save_llm_settings(
        tmp_path,
        {
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": False,
                    "display_name": "备用模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "backup-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                }
            ],
        },
    )

    saved = save_llm_settings(
        tmp_path,
        {
            "models": [
                {
                    "model_id": "model-a",
                    "display_name": "备用模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "backup-gpt",
                    "has_api_key": True,
                    "timeout_seconds": 45,
                }
            ],
        },
    )

    assert saved["models"][0]["enabled"] is False
    assert saved["enabled_models"] == []


def test_llm_settings_can_resolve_api_key_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MARVIS_TEST_LLM_KEY", "env-secret")

    saved = save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key_env": "MARVIS_TEST_LLM_KEY",
                    "timeout_seconds": 45,
                }
            ],
        },
    )

    raw = (tmp_path / "settings" / "llm.json").read_text(encoding="utf-8")
    resolved = resolve_llm_model(tmp_path, "model-a")

    assert saved["models"][0]["has_api_key"] is True
    assert saved["models"][0]["api_key_env"] == "MARVIS_TEST_LLM_KEY"
    assert "env-secret" not in raw
    assert resolved["api_key"] == "env-secret"
