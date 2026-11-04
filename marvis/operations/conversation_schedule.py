"""Monitoring schedules created from the conversation (V3 D-8).

The Agent fills a business form (what to monitor, how often, who to notify)
from the user's words. Nothing is proposed or published until the user
confirms the form; confirmation then runs the existing typed path
``SchedulePayload -> ScheduleContract -> propose_schedule -> confirm_schedule``.
Raw contract fields live under ``form["advanced"]``.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
import re
import uuid
from zoneinfo import ZoneInfo

from marvis.db_schema import connect
from marvis.llm_outbound_operations import FREQUENCIES, NOTIFY_CHANNELS


FORM_KIND = "monitoring_schedule_form"
CREATED_KIND = "monitoring_schedule_created"
_REFS = {"model": "modeling.monitor_run", "strategy": "strategy.run_strategy_monitoring"}
_INTERVAL_SECONDS = {"daily": 86400, "weekly": 7 * 86400}
_SCHEDULE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")
_MAX_OPTIONS = 200
DEFAULT_TIMEZONE = "Asia/Shanghai"


class ScheduleFormError(ValueError):
    def __init__(self, fields):
        self.fields = list(fields)
        super().__init__("monitoring schedule form is incomplete or invalid: " + ", ".join(self.fields))


def schedule_form_options(settings, task_id):
    """Platform-owned choice lists; the form can only select from these."""
    from marvis.repositories.datasets import DatasetRepository

    with connect(settings.db_path) as conn:
        experiments = conn.execute(
            "SELECT id, recipe_id FROM experiments WHERE task_id=? AND artifact_id IS NOT NULL "
            "ORDER BY created_at DESC, id LIMIT ?", (task_id, _MAX_OPTIONS)).fetchall()
        strategies = conn.execute(
            "SELECT id, strategy_type FROM strategies WHERE task_id=? ORDER BY created_at DESC, id LIMIT ?",
            (task_id, _MAX_OPTIONS)).fetchall()
    targets = [{"id": row["id"], "kind": "model", "label": f"模型 {row['recipe_id']} · {row['id'][:8]}"}
               for row in experiments]
    targets += [{"id": row["id"], "kind": "strategy", "label": f"策略 {row['strategy_type']} · {row['id'][:8]}"}
                for row in strategies]
    datasets = []
    for dataset in DatasetRepository(settings.db_path).list_datasets(task_id)[-_MAX_OPTIONS:]:
        columns = list(dataset.columns)
        datasets.append({
            "id": dataset.id, "row_count": dataset.row_count, "content_hash": dataset.content_hash,
            # Not "columns": the English UI layer relabels lists under that key.
            "column_names": [getattr(column, "name", column) for column in columns],
            "time_columns": [column.name for column in columns if "datetime" in str(getattr(column, "dtype", ""))],
        })
    return {"targets": targets[:_MAX_OPTIONS], "datasets": datasets,
            "frequencies": list(FREQUENCIES), "notify_channels": list(NOTIFY_CHANNELS)}


def _only(items):
    return items[0]["id"] if len(items) == 1 else None


def default_form(options, task_id, *, now):
    local = now.astimezone(ZoneInfo(DEFAULT_TIMEZONE))
    dataset_id = _only(options["datasets"])
    dataset = next((item for item in options["datasets"] if item["id"] == dataset_id), None)
    time_columns = dataset["time_columns"] if dataset else []
    return {
        "target_id": _only(options["targets"]),
        "dataset_id": dataset_id,
        "time_col": time_columns[0] if len(time_columns) == 1 else None,
        "frequency": "daily",
        "interval": 1,
        "start_date": (local.date() + timedelta(days=1)).isoformat(),
        "notify": NOTIFY_CHANNELS[0],
        "advanced": {
            "schedule_id": f"monitor-{task_id[:8]}-{uuid.uuid4().hex[:6]}",
            "timezone": DEFAULT_TIMEZONE,
            "complete_through": now.astimezone(UTC).replace(second=0, microsecond=0).isoformat(),
            "target_col": None,
            "label_maturity_days": None,
            "catch_up_budget": 1,
            "max_attempts": 1,
            "lease_seconds": 600,
        },
    }


def _llm_fill(client, instruction, options, *, today):
    """Return the model's enum-bounded field values, or ``{}`` on any failure."""
    from marvis.agent.json_reply import load_json_object
    from marvis.llm_client import LLMClientError
    from marvis.llm_outbound import OutboundContextError, application_schema
    from marvis.llm_outbound_operations import schedule_form_prompt
    from marvis.llm_prompts import MONITORING_SCHEDULE_FORM_SYS

    public = {"targets": options["targets"],
              "datasets": [{"id": item["id"], "row_count": item["row_count"], "columns": item["column_names"]}
                           for item in options["datasets"]]}
    prompt = schedule_form_prompt(instruction, public, today=today)
    try:
        raw = client.complete(system_prompt=MONITORING_SCHEDULE_FORM_SYS.text, user_prompt=prompt,
            response_format={"type": "json_object"},
            json_schema=application_schema("monitoring_schedule_form", options=public),
            caller="monitoring_schedule_form", prompt_name=MONITORING_SCHEDULE_FORM_SYS.name,
            prompt_version=MONITORING_SCHEDULE_FORM_SYS.version, stream=False)
        result, _ = load_json_object(raw)
    except (LLMClientError, OutboundContextError, ValueError, TypeError):
        return {}
    return result if isinstance(result, dict) else {}


def fill_schedule_form(settings, task_id, instruction, client, *, now=None):
    """Agent-filled form. Every model value is re-checked against the options."""
    now = now or datetime.now(UTC)
    options = schedule_form_options(settings, task_id)
    form = default_form(options, task_id, now=now)
    filled_by = "platform_default"
    if client is not None and options["targets"] and options["datasets"]:
        local_today = now.astimezone(ZoneInfo(DEFAULT_TIMEZONE)).date().isoformat()
        proposed = _llm_fill(client, instruction, options, today=local_today)
        accepted = {}
        for key in ("target_id", "dataset_id", "time_col", "frequency", "interval", "start_date", "notify"):
            value = proposed.get(key)
            if value is None:
                continue
            candidate = {**form, **accepted, key: value}
            _, errors = validate_form(candidate, options, fields=(key,), now=now)
            if not errors:
                accepted[key] = value
        if accepted:
            form.update(accepted)
            filled_by = "agent"
    _, missing = validate_form(form, options, now=now)
    return form, options, filled_by, missing


def _int(value, low, high):
    return type(value) is int and low <= value <= high


def _datetime(value, timezone):
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
    return parsed.astimezone(UTC)


def _local_today(now, timezone):
    try:
        zone = ZoneInfo(timezone)
    except (ValueError, KeyError, TypeError):
        zone = ZoneInfo(DEFAULT_TIMEZONE)
    return now.astimezone(zone).date()


def validate_form(form, options, *, fields=None, now=None):
    """Return ``(form, invalid_field_names)``; ``fields`` limits which are checked.

    With ``now``, two time facts are also enforced: the declared data coverage
    cannot be in the future (it asserts completeness that does not exist yet;
    ``ScheduleContract`` only requires a timezone), and the first run cannot be
    before today in the form's timezone (a past start would trigger immediate
    catch-up runs).
    """
    errors = []
    if not isinstance(form, dict):
        return form, ["form"]
    advanced = form.get("advanced") if isinstance(form.get("advanced"), dict) else {}
    targets = {item["id"] for item in options["targets"]}
    datasets = {item["id"]: item for item in options["datasets"]}
    dataset = datasets.get(form.get("dataset_id"))
    columns = set(dataset["column_names"]) if dataset else set()
    timezone = advanced.get("timezone", DEFAULT_TIMEZONE)
    checks = {
        "target_id": lambda: form.get("target_id") in targets,
        "dataset_id": lambda: dataset is not None,
        "time_col": lambda: form.get("time_col") in columns,
        "frequency": lambda: form.get("frequency") in FREQUENCIES,
        "interval": lambda: _int(form.get("interval"), 1, 12),
        "start_date": lambda: isinstance(form.get("start_date"), str) and bool(date.fromisoformat(form["start_date"]))
            and (now is None or date.fromisoformat(form["start_date"]) >= _local_today(now, timezone)),
        "notify": lambda: form.get("notify") in NOTIFY_CHANNELS,
    }
    if fields is None:
        checks.update({
            "advanced.schedule_id": lambda: isinstance(advanced.get("schedule_id"), str)
                and bool(_SCHEDULE_ID.fullmatch(advanced["schedule_id"])),
            "advanced.timezone": lambda: isinstance(timezone, str) and bool(ZoneInfo(timezone)),
            "advanced.complete_through": lambda: bool(_datetime(advanced.get("complete_through"), timezone))
                and (now is None or _datetime(advanced.get("complete_through"), timezone) <= now.astimezone(UTC)),
            "advanced.target_col": lambda: advanced.get("target_col") is None or advanced.get("target_col") in columns,
            "advanced.label_maturity_days": lambda: (
                advanced.get("label_maturity_days") is None if advanced.get("target_col") is None
                else _int(advanced.get("label_maturity_days"), 0, 3650)),
            "advanced.catch_up_budget": lambda: _int(advanced.get("catch_up_budget"), 1, 100),
            "advanced.max_attempts": lambda: _int(advanced.get("max_attempts"), 1, 10),
            "advanced.lease_seconds": lambda: _int(advanced.get("lease_seconds"), 60, 86400),
        })
    for name, check in checks.items():
        if fields is not None and name not in fields:
            continue
        try:
            valid = check()
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            errors.append(name)
    return form, errors


def first_run_date(form):
    """Calendar-month windows start on the 1st, so a monthly start rolls forward."""
    start = date.fromisoformat(form["start_date"])
    if form["frequency"] == "monthly" and start.day != 1:
        start = date(start.year + start.month // 12, start.month % 12 + 1, 1)
    return start


def form_to_schedule_payload(form, options, task_id, *, revision):
    """Deterministic business form -> installed ``SchedulePayload`` fields."""
    advanced = form["advanced"]
    timezone = advanced["timezone"]
    target = next(item for item in options["targets"] if item["id"] == form["target_id"])
    dataset = next(item for item in options["datasets"] if item["id"] == form["dataset_id"])
    anchor = datetime.combine(first_run_date(form), time(0, 0), ZoneInfo(timezone)).astimezone(UTC)
    if form["frequency"] == "monthly":
        calendar = {"anchor_at": anchor, "interval_months": form["interval"], "timezone": timezone}
    else:
        calendar = {"anchor_at": anchor, "interval_seconds": form["interval"] * _INTERVAL_SECONDS[form["frequency"]]}
    labeled = advanced.get("target_col") is not None
    attempts = advanced["max_attempts"]
    return {
        "schedule_id": advanced["schedule_id"],
        "revision": revision,
        "monitoring_ref": _REFS[target["kind"]],
        "active_from": anchor,
        "calendar": calendar,
        "catch_up_budget": advanced["catch_up_budget"],
        "lease_seconds": advanced["lease_seconds"],
        "retry_policy": {"max_attempts": attempts, "initial_backoff_seconds": 0.0 if attempts == 1 else 60.0,
                         "multiplier": 1.0 if attempts == 1 else 2.0,
                         "max_backoff_seconds": 0.0 if attempts == 1 else 600.0},
        "enabled": True,
        "monitoring_binding": {
            "task_id": task_id, "dataset_id": dataset["id"], "dataset_content_hash": dataset["content_hash"],
            "time_col": form["time_col"], "timezone": timezone,
            "complete_through": _datetime(advanced["complete_through"], timezone),
            "target_id": target["id"],
            "label_mode": "required" if labeled else "not_applicable",
            "target_col": advanced["target_col"] if labeled else None,
            "label_maturity_seconds": advanced["label_maturity_days"] * 86400 if labeled else None,
        },
    }


def draft_schedule_form_in_conversation(settings, repo, task_id, *, instruction, client, now=None):
    """Write the Agent-filled form card. Creates no proposal and no schedule."""
    form, options, filled_by, missing = fill_schedule_form(settings, task_id, instruction, client, now=now)
    if not options["targets"] or not options["datasets"]:
        return repo.add_agent_message(task_id, role="assistant", stage="chat",
            content="本任务还没有可监控的模型或策略，或没有监控数据，暂时无法创建定时监控。",
            metadata={"intent": "monitoring_schedule"})
    public = {**options, "datasets": [{k: v for k, v in item.items() if k != "content_hash"} for item in options["datasets"]]}
    content = "已按你的描述填好监控表单，核对后确认创建。" if not missing else "监控表单还缺几项，补齐后确认创建。"
    return repo.add_agent_message(task_id, role="assistant", stage="chat", content=content, metadata={
        "intent": "monitoring_schedule",
        "conversation_action": {"kind": FORM_KIND, "form": form, "options": public,
                                "filled_by": filled_by, "missing": missing},
    })


def _created_for(repo, task_id, message_id):
    for message in repo.list_agent_messages(task_id):
        action = (message.get("metadata") or {}).get("conversation_action") or {}
        if action.get("kind") == CREATED_KIND and action.get("source_message_id") == message_id:
            return message
    return None


def create_schedule_from_form(runtime, repo, task_id, *, message_id, form, actor_id):
    """User-confirmed form -> existing proposal + confirmation. Returns ``(message, replayed)``."""
    from marvis.operations.http_schemas import SchedulePayload
    from marvis.operations.router import _schedule_contract
    from marvis.operations.schedule_proposals import confirm_schedule, propose_schedule

    message = repo.get_agent_message(task_id, message_id)
    action = (message.get("metadata") or {}).get("conversation_action") or {}
    if message.get("role") != "assistant" or action.get("kind") != FORM_KIND:
        raise KeyError(message_id)
    existing = _created_for(repo, task_id, message_id)
    if existing is not None:
        return existing, True
    options = schedule_form_options(runtime.settings, task_id)
    form, errors = validate_form(form, options, now=runtime.store._clock())
    if errors:
        raise ScheduleFormError(errors)
    # This path only creates: an existing schedule id is never silently revised
    # (revisions go through the reviewed panel path with its expected revision).
    if runtime.store.get_schedule(form["advanced"]["schedule_id"]) is not None:
        raise ScheduleFormError(["advanced.schedule_id"])
    payload = SchedulePayload.model_validate(form_to_schedule_payload(form, options, task_id, revision=1))
    proposal = propose_schedule(runtime, _schedule_contract(payload), expected_revision=0,
                                instruction="", actor_id=actor_id)
    receipt = confirm_schedule(runtime, proposal["proposal_id"],
                               expected_contract_hash=proposal["contract_hash"], actor_id=actor_id)
    target = next(item for item in options["targets"] if item["id"] == form["target_id"])
    summary = {"target_label": target["label"], "dataset_id": form["dataset_id"], "frequency": form["frequency"],
               "interval": form["interval"], "start_date": first_run_date(form).isoformat(), "notify": form["notify"],
               "timezone": form["advanced"]["timezone"]}
    created = repo.add_agent_message(task_id, role="assistant", stage="chat",
        content=f"定时监控已创建：{receipt['schedule_id']}（版本 {receipt['revision']}）。", metadata={
            "intent": "monitoring_schedule",
            "conversation_action": {"kind": CREATED_KIND, "source_message_id": message_id,
                                    "proposal_id": proposal["proposal_id"], "schedule_id": receipt["schedule_id"],
                                    "revision": receipt["revision"], "contract_hash": receipt["contract_hash"],
                                    "summary": summary}})
    return created, False


__all__ = ["FORM_KIND", "CREATED_KIND", "ScheduleFormError", "schedule_form_options", "default_form",
           "fill_schedule_form", "validate_form", "form_to_schedule_payload",
           "draft_schedule_form_in_conversation", "create_schedule_from_form"]
