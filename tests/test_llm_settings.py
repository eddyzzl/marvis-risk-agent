from marvis.llm_settings import (
    load_llm_settings,
    resolve_llm_model,
    save_llm_settings,
)


def test_llm_settings_round_trip_masks_api_key(tmp_path):
    saved = save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "capability_tier": "autonomous",
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
    assert saved["capability_tier"] == "autonomous"
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


def test_structured_output_defaults_and_validation(tmp_path):
    saved = save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "gpt",
                    "api_key": "secret",
                },
                {
                    "model_id": "model-b",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "gpt",
                    "api_key": "secret",
                    "structured_output": "json_schema",
                    "thinking_style": "qwen_chat_template",
                },
                {
                    "model_id": "model-c",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "gpt",
                    "api_key": "secret",
                    "structured_output": "bogus",
                    "thinking_style": "bogus",
                },
            ],
        },
    )

    by_id = {model["model_id"]: model for model in saved["models"]}
    assert by_id["model-a"]["structured_output"] == "json_object"
    assert by_id["model-a"]["thinking_style"] == "none"
    assert by_id["model-b"]["structured_output"] == "json_schema"
    assert by_id["model-b"]["thinking_style"] == "qwen_chat_template"
    assert by_id["model-c"]["structured_output"] == "json_object"
    assert by_id["model-c"]["thinking_style"] == "none"

    private = resolve_llm_model(tmp_path, "model-b")
    assert private["structured_output"] == "json_schema"
    assert private["thinking_style"] == "qwen_chat_template"


# --- LLM-4: per-caller-role model routing -----------------------------------
def test_role_overrides_route_to_mapped_model(tmp_path):
    save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "role_overrides": {"planner": "model-b", "gate": "unknown-model"},
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "default-model",
                    "api_key": "secret-a",
                },
                {
                    "model_id": "model-b",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "small-model",
                    "api_key": "secret-b",
                },
            ],
        },
    )

    # Known role with a valid mapping routes to the mapped model.
    planner_profile = resolve_llm_model(tmp_path, role="planner")
    assert planner_profile["model_id"] == "model-b"
    assert planner_profile["model_name"] == "small-model"

    # role_overrides entry pointing at an unknown model_id is dropped at save
    # time, so an unmapped role falls back to default_model_id exactly like an
    # absent role_overrides entry (today's behavior is preserved).
    gate_profile = resolve_llm_model(tmp_path, role="gate")
    assert gate_profile["model_id"] == "model-a"

    # No role given (or role unmapped) falls back to default_model_id.
    default_profile = resolve_llm_model(tmp_path)
    assert default_profile["model_id"] == "model-a"
    unmapped_role_profile = resolve_llm_model(tmp_path, role="critic")
    assert unmapped_role_profile["model_id"] == "model-a"


def test_explicit_model_id_wins_over_role_override(tmp_path):
    save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "role_overrides": {"planner": "model-b"},
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "default-model",
                    "api_key": "secret-a",
                },
                {
                    "model_id": "model-b",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "small-model",
                    "api_key": "secret-b",
                },
            ],
        },
    )

    profile = resolve_llm_model(tmp_path, "model-a", role="planner")
    assert profile["model_id"] == "model-a"


def test_role_overrides_reject_unknown_role_names(tmp_path):
    saved = save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "role_overrides": {"planner": "model-a", "not_a_real_role": "model-a"},
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "default-model",
                    "api_key": "secret-a",
                },
            ],
        },
    )

    assert saved["role_overrides"] == {"planner": "model-a"}


def test_role_overrides_round_trip_through_load(tmp_path):
    save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "role_overrides": {"critic": "model-a"},
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "default-model",
                    "api_key": "secret-a",
                },
            ],
        },
    )

    loaded = load_llm_settings(tmp_path)
    assert loaded["role_overrides"] == {"critic": "model-a"}


# --- LLM-5: context_window / max_output_tokens profile fields ---------------
def test_context_window_and_max_output_tokens_defaults_and_bounds(tmp_path):
    saved = save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "gpt",
                    "api_key": "secret",
                },
                {
                    "model_id": "model-b",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "gpt",
                    "api_key": "secret",
                    "context_window": 8192,
                    "max_output_tokens": 512,
                },
            ],
        },
    )

    by_id = {model["model_id"]: model for model in saved["models"]}
    assert by_id["model-a"]["context_window"] == 32768
    assert by_id["model-a"]["max_output_tokens"] == 2048
    assert by_id["model-b"]["context_window"] == 8192
    assert by_id["model-b"]["max_output_tokens"] == 512

    resolved = resolve_llm_model(tmp_path, "model-b")
    assert resolved["context_window"] == 8192
    assert resolved["max_output_tokens"] == 512
