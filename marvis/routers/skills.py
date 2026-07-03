from __future__ import annotations

from fastapi import APIRouter, Request

from marvis.orchestrator.templates import clear_user_templates
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
    return _report_payload(getattr(request.app.state, "skill_report", SkillLoadReport()))


@router.post("/reload")
def reload_skills(request: Request) -> dict:
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
    skills = []
    for skill_id in report.active:
        skills.append({"id": skill_id, "status": "active", "problems": []})
    for skill_id in report.disabled:
        skills.append({"id": skill_id, "status": "disabled", "problems": []})
    for skill_id, problems in report.rejected:
        skills.append({"id": skill_id, "status": "rejected", "problems": list(problems)})
    return {
        "skills": skills,
        "counts": {
            "active": len(report.active),
            "disabled": len(report.disabled),
            "rejected": len(report.rejected),
        },
    }
