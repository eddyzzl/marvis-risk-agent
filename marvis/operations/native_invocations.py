"""Durable schedule-run admission into the existing native ToolRunner.

Only the installed scheduler adapters mint this binding after a run has been
claimed. It is neither a caller-supplied actor nor an operations-prefix grant.
"""

from datetime import UTC, datetime

from marvis.db_schema import connect
from marvis.institution.policy import current_actor, authorize_on_connection
from marvis.institution.store import identity_mode
from marvis.operations.schedule_proposals import sha, schedule_task
from marvis.operations.monitoring import MonitoringExecutionError


def allowed_tools(contract):
    if contract.portfolio_binding:
        from marvis.operations.portfolio import _TOOLS

        return {"analysis." + name for name in _TOOLS} | (
            {"analysis.score_stability_trend"}
            if contract.portfolio_binding.experiment_id
            else set()
        )
    if contract.feedback_binding:
        return {"analysis.decision_ledger_feedback"}
    if contract.monitoring_binding:
        return {contract.monitoring_ref}
    return set()


def invoke_scheduled(executor, request, ref, inputs, *, suffix=""):
    schedule = executor.store.get_schedule(
        request.schedule_id, revision=request.schedule_revision
    )
    if schedule is None or schedule.contract.monitoring_ref != request.monitoring_ref:
        raise MonitoringExecutionError("scheduled_binding_changed")
    contract = schedule.contract
    task_id = schedule_task(contract)
    name = ref.plugin + "." + ref.tool
    if name not in allowed_tools(contract):
        raise MonitoringExecutionError("scheduled_tool_not_admitted")
    prepared = executor.runner.prepare_invocation(ref)
    invocation_id = f"operations:{request.run_id}" + (":" + suffix if suffix else "")
    values = (
        invocation_id,
        request.run_id,
        request.schedule_id,
        request.schedule_revision,
        task_id,
        contract.execution_actor_id or "trusted-local-legacy-scheduler",
        name,
        sha(prepared),
        sha(inputs),
        datetime.now(UTC).isoformat(),
    )
    with connect(executor.settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT r.*,p.state AS current_state,p.lease_token AS current_token,p.lease_expires_at AS current_expires
            FROM operations_runs r JOIN operations_periods p ON p.schedule_id=r.schedule_id AND p.period_key=r.period_key
            WHERE r.run_id=?""",
            (request.run_id,),
        ).fetchone()
        if (
            row is None
            or row["schedule_id"] != request.schedule_id
            or row["schedule_revision"] != request.schedule_revision
            or row["current_state"] != "running"
            or row["lease_token"] != row["current_token"]
            or datetime.fromisoformat(row["current_expires"]) <= executor.clock()
        ):
            raise MonitoringExecutionError("scheduled_dispatch_lease_changed")
        if (
            identity_mode(conn) == "institutional"
            and current_actor(executor.settings.db_path) != contract.execution_actor_id
        ):
            raise MonitoringExecutionError("scheduled_dispatch_actor_changed")
        authorize_on_connection(
            conn, "task", task_id, "execute", actor_id=contract.execution_actor_id
        )
        old = conn.execute(
            "SELECT * FROM operations_native_invocations WHERE invocation_id=?",
            (invocation_id,),
        ).fetchone()
        if old:
            if tuple(old)[:9] != values[:9]:
                raise MonitoringExecutionError("scheduled_dispatch_identity_changed")
            # A native call identity is consumed at host dispatch. Read-side
            # evidence reconciliation, never an automatic reinvocation, owns an
            # uncertain completion. Scheduler retries mint a new leased run ID.
            raise MonitoringExecutionError("scheduled_dispatch_already_consumed")
        conn.execute(
            "INSERT INTO operations_native_invocations VALUES(?,?,?,?,?,?,?,?,?,?)",
            values,
        )
    result = executor.runner.invoke(
        ref,
        inputs,
        task_id=task_id,
        invocation_id=invocation_id,
        expected_invocation=prepared,
        cancellation_check=executor._check_cancelled,
    )
    return result, prepared
