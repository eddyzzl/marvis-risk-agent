"""Human-reviewed typed schedules; natural text never publishes a schedule."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import re
import uuid

from marvis.db_schema import connect
from marvis.llm_outbound_instructions import user_instruction
from marvis.operations.contracts import FixedIntervalCalendar, ScheduleContract
from marvis.operations.month_calendar import CalendarMonthCalendar
from marvis.operations.repository import ScheduleRevisionConflict
from marvis.orchestrator.evidence import payload_hash


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def schedule_task(contract):
    if contract.monitoring_binding:
        return contract.monitoring_binding.task_id
    if contract.portfolio_binding:
        return contract.portfolio_binding.source.task_id
    if contract.feedback_binding:
        return contract.feedback_binding.task_id
    raise ValueError("typed scheduled source required")


def binding_identity(runtime, contract, *, conn=None, source_binding=None):
    executor = runtime._require_executor(contract.monitoring_ref)
    validator = getattr(executor, "validate", None)
    if not callable(validator):
        raise ValueError("proposal requires an installed authenticated domain executor")
    if conn is None:
        validator(contract)
    task_id = schedule_task(contract)
    from marvis.institution.policy import authorize, authorize_on_connection

    if conn is None:
        authorize(
            runtime.settings.db_path,
            "task",
            task_id,
            "execute",
            actor_id=contract.execution_actor_id,
        )
    else:
        authorize_on_connection(
            conn, "task", task_id, "execute", actor_id=contract.execution_actor_id
        )
    source = contract.monitoring_binding or (
        contract.portfolio_binding.source if contract.portfolio_binding else None
    )
    identity = {
        "binding": contract.to_dict().get("monitoring_binding")
        or contract.to_dict().get("portfolio_binding")
        or contract.to_dict().get("feedback_binding")
    }
    if source:
        authenticated = (
            source_binding
            or executor.registry.authenticate_dataset_binding(
                source.dataset_id,
                expected_task_id=task_id,
                expected_content_hash=source.dataset_content_hash,
            )
        )
        if conn is not None:
            executor.registry.verify_dataset_binding_on_connection(conn, authenticated)
        identity["source"] = {
            key: getattr(authenticated, key)
            for key in (
                "dataset_id",
                "task_id",
                "content_hash",
                "relative_path",
                "row_count",
            )
        }
    from marvis.plugins.manifest import ToolRef

    refs = [ToolRef(*contract.monitoring_ref.split(".", 1))]
    if contract.portfolio_binding:
        from marvis.operations.portfolio import _TOOLS

        refs = [ToolRef("analysis", name) for name in _TOOLS]
        if contract.portfolio_binding.experiment_id:
            refs.append(ToolRef("analysis", "score_stability_trend"))
    identity["installed_invocations"] = [
        executor.runner.prepare_invocation(ref) for ref in refs
    ]
    target = (
        contract.monitoring_binding.target_id
        if contract.monitoring_binding
        else contract.portfolio_binding.experiment_id
        if contract.portfolio_binding
        else None
    )
    if target:

        def target_identity(connection):
            if contract.monitoring_ref == "strategy.run_strategy_monitoring":
                row = connection.execute(
                    "SELECT id,task_id,version,dsl_content_hash,status FROM strategies WHERE id=? AND task_id=?",
                    (target, task_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM experiments WHERE id=? AND task_id=?",
                    (target, task_id),
                ).fetchone()
            if row is None:
                raise ValueError("authenticated scheduled target is unavailable")
            return payload_hash(dict(row))

        if conn is None:
            with connect(runtime.settings.db_path) as target_conn:
                identity["target_hash"] = target_identity(target_conn)
        else:
            identity["target_hash"] = target_identity(conn)
    return sha(identity)


def calendar_from_instruction(instruction, contract):
    """Admit explicit intervals and anchors; ambiguous language asks for details."""
    # The shared scanner checks the complete active-user instruction. No text
    # is shortened, sent to a provider, or used as source evidence.
    user_instruction(instruction)
    patterns = (
        r"每\s*(\d+)\s*(分钟|小时|天|个月)",
        r"every\s+(\d+)\s+(minutes?|hours?|days?|months?)",
    )
    matches = [
        m
        for pattern in patterns
        for m in re.finditer(pattern, instruction, re.IGNORECASE)
    ]
    if len(matches) != 1:
        return None, ["calendar_interval"]
    count = int(matches[0][1])
    unit = matches[0][2].lower()
    if not 1 <= count <= 120:
        return None, ["calendar_interval"]
    dates = re.findall(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})", instruction
    )
    if len(dates) != 1:
        return None, ["calendar_anchor_with_timezone"]
    anchor = datetime.fromisoformat(dates[0].replace("Z", "+00:00"))
    if unit in {"个月", "month", "months"}:
        if not isinstance(contract.calendar, CalendarMonthCalendar):
            return None, ["calendar_timezone"]
        calendar = CalendarMonthCalendar(anchor, count, contract.calendar.timezone)
    else:
        seconds = (
            60
            if unit in {"分钟", "minute", "minutes"}
            else 3600
            if unit in {"小时", "hour", "hours"}
            else 86400
        )
        calendar = FixedIntervalCalendar(anchor, count * seconds)
    return replace(contract, calendar=calendar, active_from=anchor), []


def propose_schedule(
    runtime, contract, *, expected_revision, instruction, actor_id, clock=None
):
    now = (clock or runtime.store._clock)().astimezone(UTC)
    contract = replace(contract, execution_actor_id=actor_id)
    proposed, missing = (
        calendar_from_instruction(instruction, contract)
        if instruction
        else (contract, [])
    )
    if missing:
        return {
            "state": "clarification",
            "missing_fields": missing,
            "published": False,
            "automatic_action_permitted": False,
        }
    if proposed.revision != expected_revision + 1:
        raise ScheduleRevisionConflict(
            "proposal revision must follow the reviewed head"
        )
    current = runtime.store.get_schedule(proposed.schedule_id)
    if current and schedule_task(current.contract) != schedule_task(proposed):
        raise ScheduleRevisionConflict("schedule task identity cannot be rebound")
    if (current.contract.revision if current else 0) != expected_revision:
        raise ScheduleRevisionConflict("schedule changed before proposal")
    proof = binding_identity(runtime, proposed)
    value = proposed.to_dict()
    value["runtime_binding_hash"] = proof
    proposed = ScheduleContract.from_dict(value)
    proposal_id = "proposal_" + uuid.uuid4().hex
    raw = canonical(proposed.to_dict())
    digest = hashlib.sha256(raw.encode()).hexdigest()
    with connect(runtime.settings.db_path) as conn:
        conn.execute(
            "INSERT INTO operations_schedule_proposals VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                proposal_id,
                actor_id,
                schedule_task(proposed),
                expected_revision,
                raw,
                digest,
                hashlib.sha256(instruction.encode()).hexdigest(),
                proof,
                now.isoformat(),
                (now + timedelta(minutes=30)).isoformat(),
            ),
        )
    return {
        "state": "needs_confirmation",
        "proposal_id": proposal_id,
        "contract_hash": digest,
        "binding_hash": proof,
        "expected_revision": expected_revision,
        "contract": proposed.to_dict(),
        "expires_at": (now + timedelta(minutes=30)).isoformat(),
        "published": False,
        "automatic_action_permitted": False,
    }


def confirm_schedule(runtime, proposal_id, *, expected_contract_hash, actor_id):
    # Prepare any content-addressed pin before taking the writer lock. The
    # final check below reuses that exact capability on the same connection.
    with connect(runtime.settings.db_path) as read_conn:
        proposed = read_conn.execute(
            "SELECT * FROM operations_schedule_proposals WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
    if proposed is None:
        raise KeyError(proposal_id)
    if (
        proposed["actor_id"] != actor_id
        or proposed["contract_hash"] != expected_contract_hash
    ):
        raise ScheduleRevisionConflict("proposal actor or reviewed contract changed")
    with connect(runtime.settings.db_path) as read_conn:
        from marvis.institution.policy import authorize_on_connection

        authorize_on_connection(
            read_conn, "task", proposed["task_id"], "execute", actor_id=actor_id
        )
        receipt = read_conn.execute(
            "SELECT * FROM operations_schedule_confirmations WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
    if receipt:
        return {
            "state": "published",
            "replayed": True,
            "schedule_id": receipt["schedule_id"],
            "revision": receipt["revision"],
            "contract_hash": receipt["contract_hash"],
            "automatic_action_permitted": False,
        }
    prepared = ScheduleContract.from_dict(json.loads(proposed["contract_json"]))
    executor = runtime._require_executor(prepared.monitoring_ref)
    binding_identity(runtime, prepared)
    source = prepared.monitoring_binding or (
        prepared.portfolio_binding.source if prepared.portfolio_binding else None
    )
    snapshot = (
        executor.registry.authenticate_dataset_binding(
            source.dataset_id,
            expected_task_id=source.task_id,
            expected_content_hash=source.dataset_content_hash,
        )
        if source
        else None
    )
    with connect(runtime.settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM operations_schedule_proposals WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise KeyError(proposal_id)
        if (
            row["actor_id"] != actor_id
            or row["contract_hash"] != expected_contract_hash
        ):
            raise ScheduleRevisionConflict(
                "proposal actor or reviewed contract changed"
            )
        contract = ScheduleContract.from_dict(json.loads(row["contract_json"]))
        from marvis.institution.policy import authorize_on_connection

        authorize_on_connection(
            conn, "task", row["task_id"], "execute", actor_id=actor_id
        )
        receipt = conn.execute(
            "SELECT * FROM operations_schedule_confirmations WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
        if receipt:
            return {
                "state": "published",
                "replayed": True,
                "schedule_id": receipt["schedule_id"],
                "revision": receipt["revision"],
                "contract_hash": receipt["contract_hash"],
                "automatic_action_permitted": False,
            }
        if datetime.fromisoformat(
            row["expires_at"]
        ) <= runtime.store._clock().astimezone(UTC):
            raise ScheduleRevisionConflict("proposal expired")
        head = (
            conn.execute(
                "SELECT max(revision) FROM operations_schedules WHERE schedule_id=?",
                (contract.schedule_id,),
            ).fetchone()[0]
            or 0
        )
        if head:
            previous = conn.execute(
                "SELECT contract_json FROM operations_schedules WHERE schedule_id=? AND revision=?",
                (contract.schedule_id, head),
            ).fetchone()
            if (
                schedule_task(
                    ScheduleContract.from_dict(json.loads(previous["contract_json"]))
                )
                != row["task_id"]
            ):
                raise ScheduleRevisionConflict(
                    "schedule task identity cannot be rebound"
                )
        if (
            head != row["expected_revision"]
            or binding_identity(runtime, contract, conn=conn, source_binding=snapshot)
            != row["binding_hash"]
        ):
            raise ScheduleRevisionConflict(
                "schedule or source/target binding changed before confirmation"
            )
        now = runtime.store._clock().astimezone(UTC).isoformat()
        conn.execute(
            "INSERT INTO operations_schedules VALUES(?,?,?,?,?,?)",
            (
                contract.schedule_id,
                contract.revision,
                contract.schema_version,
                row["contract_json"],
                row["contract_hash"],
                now,
            ),
        )
        conn.execute(
            "INSERT INTO operations_schedule_confirmations VALUES(?,?,?,?,?,?)",
            (
                proposal_id,
                contract.schedule_id,
                contract.revision,
                row["contract_hash"],
                actor_id,
                now,
            ),
        )
        if (
            binding_identity(runtime, contract, conn=conn, source_binding=snapshot)
            != row["binding_hash"]
        ):
            raise ScheduleRevisionConflict("source changed while confirming schedule")
    return {
        "state": "published",
        "replayed": False,
        "schedule_id": contract.schedule_id,
        "revision": contract.revision,
        "contract_hash": row["contract_hash"],
        "automatic_action_permitted": False,
    }
