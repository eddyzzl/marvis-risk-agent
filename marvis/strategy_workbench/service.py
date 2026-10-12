"""User drafts never become deterministic evidence; all pointers bind current evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from html.parser import HTMLParser
import json
from pathlib import Path
import time

from marvis.db_schema import connect
from marvis.redaction import redact_text
from marvis.strategy_workbench.contracts import (
    SEARCH_TOOLS,
    WorkbenchError,
    canonical,
    digest,
)


def require_owner(conn, actor):
    row = conn.execute(
        "SELECT status,expires_at FROM local_principals WHERE id=?", (actor,)
    ).fetchone()
    if (
        not row
        or row["status"] != "active"
        or datetime.fromisoformat(row["expires_at"]).timestamp() <= time.time()
    ):
        raise WorkbenchError("workbench_session_required", 403)
    role = conn.execute(
        "SELECT status FROM production_principals WHERE local_principal_id=?", (actor,)
    ).fetchone()
    if role and role["status"] != "active":
        raise WorkbenchError("workbench_actor_inactive", 403)


_EVIDENCE_TABLES = (
    "datasets",
    "data_workspaces",
    "task_artifacts",
    "strategies",
    "strategy_candidate_pools",
    "strategy_candidate_pool_revisions",
    "strategy_project_context_revisions",
    "strategy_project_context_heads",
    "strategy_report_revisions",
    "strategy_report_heads",
    "strategy_pool_materializations",
)


def evidence_stamp(conn, task_id):
    payload = {}
    for table in _EVIDENCE_TABLES:
        # The names are a source-owned fixed whitelist, never caller identifiers.
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE task_id=? ORDER BY rowid", (task_id,)  # nosec B608 -- table iterates only source-owned _EVIDENCE_TABLES literal tuple; task id is bound
        ).fetchall()
        payload[table] = [dict(row) for row in rows]
    return digest(payload)


def authenticated_evidence(settings, task_id):
    from marvis.packs.strategy.candidate_lab_projection import (
        CandidateLabProjectionError,
        build_strategy_candidate_lab_projection,
    )

    with connect(settings.db_path) as conn:
        task = conn.execute(
            "SELECT task_type FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if not task:
            raise WorkbenchError("workbench_task_not_found", 404)
        if task["task_type"] != "strategy":
            raise WorkbenchError("workbench_requires_strategy_task", 422)
        before = evidence_stamp(conn, task_id)
    try:
        projection = build_strategy_candidate_lab_projection(settings, task_id)
    except CandidateLabProjectionError as exc:
        raise WorkbenchError("workbench_evidence_verification_failed") from exc
    with connect(settings.db_path) as conn:
        after = evidence_stamp(conn, task_id)
    if before != after:
        raise WorkbenchError("workbench_evidence_changed")
    stable = {
        k: v
        for k, v in projection.items()
        if k
        not in {
            "can_start",
            "blocked_reason",
            "active_plan",
            "open_gate",
            "evidence_drawer",
        }
    }
    return digest({"projection": stable, "registry_stamp": after}), after


def evidence_hash(settings, task_id):
    return authenticated_evidence(settings, task_id)[0]


class _FormFields(HTMLParser):
    def __init__(self):
        super().__init__()
        self.workflow = None
        self.fields = {}
        self.static_choices = {}
        self.select = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("data-candidate-lab-workflow"):
            self.workflow = attrs["data-candidate-lab-workflow"]
        field = attrs.get("data-candidate-lab-field")
        if self.workflow and field:
            kind = (
                "reference"
                if "readonly" in attrs
                else "select"
                if tag == "select"
                else "checks"
                if attrs.get("type") in {"checkbox", "radio"}
                else "narrative"
                if tag == "textarea" or attrs.get("type", "text") == "text"
                else "scalar"
            )
            self.fields[(self.workflow, field)] = kind
            key = (self.workflow, field)
            if tag == "select":
                self.select = key
            elif kind == "checks" and attrs.get("value"):
                self.static_choices.setdefault(key, set()).add(attrs["value"])
        if tag == "option" and self.select and attrs.get("value"):
            self.static_choices.setdefault(self.select, set()).add(attrs["value"])

    def handle_endtag(self, tag):
        if tag == "form":
            self.workflow = None
            self.select = None
        elif tag == "select":
            self.select = None


@lru_cache(maxsize=1)
def draft_form_contract():
    parser = _FormFields()
    parser.feed((Path(__file__).parents[1] / "static/index.html").read_text())
    parser.fields[("strategy_workbench_preferences", "field_display_mode")] = "select"
    parser.static_choices[("strategy_workbench_preferences", "field_display_mode")] = {
        "physical", "business", "both"
    }
    return parser


def draft_fields():
    return draft_form_contract().fields


def draft_static_choices():
    return [
        {"workflow": workflow, "field": field, "values": sorted(values)}
        for (workflow, field), values in sorted(draft_form_contract().static_choices.items())
    ]


def validate_draft(draft):
    value = draft.model_dump()
    if len(canonical(value).encode()) > 64 * 1024:
        raise WorkbenchError("workbench_draft_too_large", 413)
    fields = draft_fields()
    seen = set()
    for entry in draft.fields:
        key = (entry.workflow, entry.field)
        if key in seen or key not in fields or fields[key] != entry.kind:
            raise WorkbenchError("workbench_draft_field_invalid", 422)
        installed = draft_form_contract().static_choices.get(key)
        if installed and any(value and value not in installed for value in entry.values):
            raise WorkbenchError("workbench_draft_choice_invalid", 422)
        seen.add(key)
        if key == ("strategy_workbench_preferences", "field_display_mode") and (
            len(entry.values) != 1
            or entry.values[0] not in {"physical", "business", "both"}
        ):
            raise WorkbenchError("workbench_display_mode_invalid", 422)
        for text in entry.values:
            if len(text) > 4000 or "\x00" in text or redact_text(text) != text:
                raise WorkbenchError("workbench_draft_sensitive_or_oversized", 422)
    workflows = {key[0] for key in fields}
    if any(workflow not in workflows for workflow in draft.open_workflows):
        raise WorkbenchError("workbench_draft_workflow_invalid", 422)
    return value


class WorkbenchService:
    def __init__(self, settings):
        self.settings = settings
        self.db_path = settings.db_path

    def get(self, task_id, actor, revision=None):
        current = evidence_hash(self.settings, task_id)
        with connect(self.db_path) as conn:
            require_owner(conn, actor)
            if revision is None:
                row = conn.execute(
                    "SELECT * FROM strategy_workbench_drafts WHERE task_id=? AND actor_id=? ORDER BY revision DESC LIMIT 1",
                    (task_id, actor),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM strategy_workbench_drafts WHERE task_id=? AND actor_id=? AND revision=?",
                    (task_id, actor, revision),
                ).fetchone()
            history = [
                dict(r)
                for r in conn.execute(
                    "SELECT revision,evidence_hash,created_at FROM strategy_workbench_drafts WHERE task_id=? AND actor_id=? ORDER BY revision DESC LIMIT 30",
                    (task_id, actor),
                )
            ]
        invalidated = []
        draft = json.loads(row["payload"]) if row else None
        if row and row["evidence_hash"] != current:
            # A stale draft retains prose and knobs; evidence-dependent selection is
            # visibly cleared rather than rebound to another candidate or dataset.
            for field in draft["fields"]:
                meaningful = [value for value in field["values"] if value]
                installed = draft_form_contract().static_choices.get(
                    (field["workflow"], field["field"])
                )
                if (
                    field["kind"] in {"select", "checks", "reference"}
                    and meaningful
                    and not (installed and all(value in installed for value in meaningful))
                ):
                    invalidated.append(
                        {"workflow": field["workflow"], "field": field["field"]}
                    )
                    field["values"] = []
        return {
            "task_id": task_id,
            "actor_id": actor,
            "revision": row["revision"] if row else 0,
            "head_revision": history[0]["revision"] if history else 0,
            "evidence_hash": current,
            "saved_evidence_hash": row["evidence_hash"] if row else None,
            "draft": draft,
            "invalidated_selections": invalidated,
            "static_choices": draft_static_choices(),
            "history": history,
            "narrative_is_evidence": False,
        }

    def save(self, task_id, actor, request):
        value = validate_draft(request.draft)
        current, stamp = authenticated_evidence(self.settings, task_id)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_owner(conn, actor)
            if (
                evidence_stamp(conn, task_id) != stamp
                or request.evidence_hash != current
            ):
                raise WorkbenchError("workbench_evidence_changed")
            latest = conn.execute(
                "SELECT COALESCE(MAX(revision),0) FROM strategy_workbench_drafts WHERE task_id=? AND actor_id=?",
                (task_id, actor),
            ).fetchone()[0]
            if latest != request.expected_revision:
                raise WorkbenchError("workbench_draft_conflict")
            conn.execute(
                "INSERT INTO strategy_workbench_drafts VALUES(?,?,?,?,?,?)",
                (task_id, actor, latest + 1, current, canonical(value), time.time()),
            )
        return self.get(task_id, actor)

    def executions(self, task_id, actor):
        with connect(self.db_path) as conn:
            require_owner(conn, actor)
            rows = conn.execute(
                "SELECT * FROM strategy_workbench_searches WHERE task_id=? AND actor_id=? ORDER BY updated_at DESC LIMIT 50",
                (task_id, actor),
            ).fetchall()
        evidence_error = None
        try:
            current = evidence_hash(self.settings, task_id)
        except WorkbenchError as exc:
            if exc.status != 409:
                raise
            current, evidence_error = None, exc.code
        return {
            "evidence_error": evidence_error,
            "evidence_hash": current,
            "supported_tools": sorted(SEARCH_TOOLS),
            "pause_granularity": "inside_search_at_candidate_boundary",
            "individual_native_candidate_operation_interruptible": False,
            "items": [public_search(row) for row in rows],
        }


def public_search(row):
    return {
        key: row[key]
        for key in (
            "id",
            "task_id",
            "plan_id",
            "step_id",
            "tool_ref",
            "state",
            "revision",
            "evidence_hash",
            "completed",
            "total",
            "updated_at",
            "result_json",
        )
    }


def register_plan(settings, plan_id, actor):
    """Same typed/NL compiler and PlanValidator, before any Tool dispatch."""
    if not actor:
        return
    with connect(settings.db_path) as conn:
        steps = conn.execute(
            "SELECT s.id,p.task_id,s.tool_plugin || '.' || s.tool_name AS tool_ref FROM plan_steps s JOIN plans p ON p.id=s.plan_id WHERE s.plan_id=?",
            (plan_id,),
        ).fetchall()
        supported = [row for row in steps if row["tool_ref"] in SEARCH_TOOLS]
        if not supported:
            return
        require_owner(conn, actor)
    current = evidence_hash(settings, supported[0]["task_id"])
    with connect(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for row in supported:
            conn.execute(
                "INSERT OR IGNORE INTO strategy_workbench_searches(id,task_id,actor_id,plan_id,step_id,tool_ref,evidence_hash,state,revision,completed,total,updated_at) VALUES(?,?,?,?,?,?,?,'queued',0,0,0,?)",
                (
                    row["id"],
                    row["task_id"],
                    actor,
                    plan_id,
                    row["id"],
                    row["tool_ref"],
                    current,
                    time.time(),
                ),
            )
