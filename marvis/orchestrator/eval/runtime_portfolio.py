"""Download registered native reports through the protected task route."""

import re
from urllib.parse import quote

from .runtime_contracts import digest


def download_registered_report(journey, action, *, family):
    from .runtime_runner import RuntimeJourneyError, _tool_name

    tool = {"portfolio": "analysis.portfolio_report", "feature": "feature.generate_feature_report"}[family]
    matches = [
        step for plan in journey.plans() for step in plan["steps"]
        if _tool_name(step["tool_ref"]) == tool
        and step["status"] == "done"
    ]
    if len(matches) != 1:
        raise RuntimeJourneyError(f"{family}_report_not_unique")
    step_id = matches[0]["id"]
    output = journey.json_request(
        "GET", f"/api/step-outputs/{step_id}", label=f"read_{family}_report_output"
    )
    artifact_id = output.get("artifact_id")
    content_hash = output.get("artifact_content_hash")
    if (not isinstance(artifact_id, str) or not artifact_id
            or not isinstance(content_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", content_hash)):
        raise RuntimeJourneyError(f"{family}_download_binding_invalid")
    # The caller cannot inject a filesystem path, URL, or artifact identity.
    # The route authenticates task ownership and registered file bytes.
    route = (f"/api/tasks/{quote(journey.task_id, safe='')}/task-artifacts/"
             f"{quote(artifact_id, safe='')}/download?expected_content_hash={content_hash}")
    journey.record_human_action(action)
    response = journey.request("GET", route, label=f"download_{family}_report")
    if (response.status_code != 200 or not response.content
            or digest(response.content) != content_hash):
        raise RuntimeJourneyError(f"{family}_download_integrity_failed")
    journey.events[-1].update(
        sha256=content_hash, size_bytes=len(response.content),
        artifact_id=artifact_id, step_id=step_id,
    )


def download_portfolio_report(journey, action):
    return download_registered_report(journey, action, family="portfolio")
