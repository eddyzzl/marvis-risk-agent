from __future__ import annotations

from fastapi import APIRouter, Request

from marvis.orchestrator.templates import clear_user_templates, list_templates, load_builtin_templates
from marvis.orchestrator.templates.skills import (
    SkillLoadReport,
    SkillTemplateError,
    load_user_skill_templates,
    parse_skill_template,
    validate_skill_template,
)


router = APIRouter(prefix="/api/skills", tags=["skills"])


@router.get("")
def list_skills(request: Request) -> dict:
    load_builtin_templates()
    return _report_payload(getattr(request.app.state, "skill_report", SkillLoadReport()))


@router.post("/reload")
def reload_skills(request: Request) -> dict:
    load_builtin_templates()
    clear_user_templates()
    report = load_user_skill_templates(
        request.app.state.settings.workspace,
        request.app.state.tool_registry,
        request.app.state.plan_validator,
    )
    request.app.state.skill_report = report
    return _report_payload(report)


@router.post("/validate")
def validate_skill(request: Request, body: dict) -> dict:
    skill_payload = body.get("skill") if isinstance(body.get("skill"), dict) else body
    try:
        template = parse_skill_template(skill_payload)
    except SkillTemplateError as exc:
        return {"valid": False, "problems": [str(exc)]}
    problems = validate_skill_template(
        template,
        request.app.state.tool_registry,
        request.app.state.plan_validator,
    )
    return {
        "valid": not problems,
        "id": template.id,
        "problems": problems,
    }


def _report_payload(report: SkillLoadReport) -> dict:
    templates = list_templates()
    titles = {template.id: template.title for template in templates}
    skills = []
    for skill_id in report.active:
        skills.append({"id": skill_id, "status": "active", "problems": [], "title": titles.get(skill_id, "")})
    for skill_id in report.disabled:
        skills.append({"id": skill_id, "status": "disabled", "problems": [], "title": titles.get(skill_id, "")})
    for skill_id, problems in report.rejected:
        skills.append({"id": skill_id, "status": "rejected", "problems": list(problems), "title": titles.get(skill_id, "")})
    return {
        "skills": skills,
        "counts": {
            "active": len(report.active),
            "disabled": len(report.disabled),
            "rejected": len(report.rejected),
        },
        "builtin": [
            _template_payload(template)
            for template in templates
            if template.source == "builtin"
        ],
    }


def _template_payload(template) -> dict:
    return {
        "id": template.id,
        "title": template.title,
        "goal_patterns": list(template.goal_patterns),
        "default_autonomy": template.default_autonomy,
        "slots": [
            {
                "name": slot.name,
                "required": slot.required,
                "source": slot.source,
                "description": slot.description,
            }
            for slot in template.slots
        ],
        "steps": [_step_payload(step) for step in template.steps],
        "success_criteria": [dict(item) for item in template.success_criteria],
    }


def _step_payload(step) -> dict:
    return {
        "title": step.title,
        "tool": {
            "plugin": step.tool_ref.plugin,
            "tool": step.tool_ref.tool,
            "version": step.tool_ref.version,
        },
        "inputs": step.inputs_template,
        "depends_on": list(step.depends_on_titles),
        "post_checks": [
            {"kind": check.kind, "spec": dict(check.spec)}
            for check in step.post_checks
        ],
        "needs_confirmation": step.needs_confirmation,
        "decision_point": step.decision_point,
        "sub_agent_scope": step.sub_agent_scope,
        "granted_tools": [
            {
                "plugin": ref.plugin,
                "tool": ref.tool,
                "version": ref.version,
            }
            for ref in step.granted_tools
        ],
        "phase": step.phase,
    }
