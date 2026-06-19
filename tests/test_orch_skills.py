import json
from pathlib import Path

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import PluginRepository, init_db
from marvis.orchestrator.templates import (
    clear_user_templates,
    get_template,
    load_builtin_templates,
)
from marvis.orchestrator.templates.skills import (
    SkillTemplateError,
    load_user_skill_templates,
    parse_skill_template,
    validate_skill_template,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _skill_data(**overrides):
    data = {
        "id": "user_echo",
        "title": "User Echo",
        "goal_patterns": ["echo"],
        "default_autonomy": 1,
        "enabled": True,
        "slots": [
            {
                "name": "message",
                "required": True,
                "source": "user",
                "description": "Message",
            }
        ],
        "steps": [
            {
                "title": "Echo",
                "tool": {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": "{slot:message}"},
                "depends_on": [],
                "post_checks": [{"kind": "nonempty", "spec": {"field": "echoed"}}],
            }
        ],
    }
    data.update(overrides)
    return data


def _registry_and_validator(tmp_path: Path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    plugin_registry = PluginRegistry(repo)
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    plugin_registry.register(_metrics_manifest(), enabled=True)
    tool_registry = ToolRegistry(plugin_registry)
    return tool_registry, PlanValidator(tool_registry)


def _metrics_manifest():
    return parse_manifest(
        {
            "name": "metrics_pack",
            "version": "0.1.0",
            "display_name": "Metrics Pack",
            "description": "Skill validation metrics",
            "module": "metrics_pack.tools",
            "tools": [
                {
                    "name": "score_metrics",
                    "summary": "Compute metrics",
                    "input_schema": {
                        "type": "object",
                        "properties": {"dataset": {"type": "string"}},
                        "required": ["dataset"],
                        "additionalProperties": False,
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"ks": {"type": "number"}},
                        "required": ["ks"],
                        "additionalProperties": False,
                    },
                    "determinism": "deterministic",
                    "timeout_seconds": 10,
                    "failure_policy": "fail",
                    "entrypoint": "tool_score_metrics",
                }
            ],
            "hooks": [],
            "permissions": [],
        },
        builtin=True,
    )


def test_parse_skill_template_returns_user_workflow_template():
    template = parse_skill_template(_skill_data())

    assert template.id == "user_echo"
    assert template.source == "user"
    assert template.steps[0].tool_ref == ToolRef("_sample", "echo")
    assert template.steps[0].inputs_template == {"message": "{slot:message}"}


def test_parse_skill_template_rejects_bad_shape():
    data = _skill_data(steps=[])

    try:
        parse_skill_template(data)
    except SkillTemplateError as exc:
        assert "steps" in str(exc)
    else:
        raise AssertionError("expected SkillTemplateError")


def test_validate_skill_template_rejects_builtin_shadow_and_plan_problems(tmp_path):
    load_builtin_templates()
    tool_registry, plan_validator = _registry_and_validator(tmp_path)
    shadow = parse_skill_template(_skill_data(id="sample_echo"))
    metric_without_check = parse_skill_template(
        _skill_data(
            id="bad_metrics",
            steps=[
                {
                    "title": "Metrics",
                    "tool": {"plugin": "metrics_pack", "tool": "score_metrics"},
                    "inputs": {"dataset": "{slot:message}"},
                    "depends_on": [],
                    "post_checks": [],
                }
            ],
        )
    )

    shadow_problems = validate_skill_template(shadow, tool_registry, plan_validator)
    metric_problems = validate_skill_template(metric_without_check, tool_registry, plan_validator)

    assert any("shadows a builtin" in problem for problem in shadow_problems)
    assert any("ks" in problem for problem in metric_problems)


def test_validate_skill_template_rejects_unknown_post_check_kind(tmp_path):
    tool_registry, plan_validator = _registry_and_validator(tmp_path)
    template = parse_skill_template(
        _skill_data(
            id="bad_post_check",
            steps=[
                {
                    "title": "Echo",
                    "tool": {"plugin": "_sample", "tool": "echo"},
                    "inputs": {"message": "{slot:message}"},
                    "depends_on": [],
                    "post_checks": [{"kind": "mystery", "spec": {"field": "echoed"}}],
                }
            ],
        )
    )

    problems = validate_skill_template(template, tool_registry, plan_validator)

    assert any("unknown post_check kind mystery" in problem for problem in problems)


def test_load_user_skill_templates_registers_active_and_reports_rejected(tmp_path):
    clear_user_templates()
    load_builtin_templates()
    tool_registry, plan_validator = _registry_and_validator(tmp_path)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "active.json").write_text(json.dumps(_skill_data()), encoding="utf-8")
    (skills_dir / "disabled.json").write_text(
        json.dumps(_skill_data(id="disabled_echo", enabled=False)),
        encoding="utf-8",
    )
    (skills_dir / "rejected.json").write_text(
        json.dumps(_skill_data(id="sample_echo")),
        encoding="utf-8",
    )

    report = load_user_skill_templates(tmp_path, tool_registry, plan_validator)

    assert report.active == ["user_echo"]
    assert report.disabled == ["disabled_echo"]
    assert report.rejected[0][0] == "sample_echo"
    assert get_template("user_echo").source == "user"


def test_skills_api_lists_reloads_and_validates_user_skills(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)

    initial = client.get("/api/skills")
    assert initial.status_code == 200
    assert initial.json()["counts"] == {"active": 0, "disabled": 0, "rejected": 0}

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "active.json").write_text(json.dumps(_skill_data()), encoding="utf-8")
    (skills_dir / "disabled.json").write_text(
        json.dumps(_skill_data(id="disabled_echo", enabled=False)),
        encoding="utf-8",
    )
    (skills_dir / "rejected.json").write_text(
        json.dumps(_skill_data(id="sample_echo")),
        encoding="utf-8",
    )

    reloaded = client.post("/api/skills/reload")
    skills = reloaded.json()["skills"]

    assert reloaded.status_code == 200
    assert reloaded.json()["counts"] == {"active": 1, "disabled": 1, "rejected": 1}
    assert {"id": "user_echo", "status": "active", "problems": []} in skills
    assert any(skill["id"] == "sample_echo" and skill["status"] == "rejected" for skill in skills)

    valid = client.post("/api/skills/validate", json=_skill_data(id="preview_echo"))
    invalid = client.post("/api/skills/validate", json=_skill_data(id="sample_echo"))

    assert valid.json() == {"valid": True, "id": "preview_echo", "problems": []}
    assert invalid.json()["valid"] is False
    assert any("shadows a builtin" in problem for problem in invalid.json()["problems"])
