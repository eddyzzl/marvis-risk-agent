"""Task-bound visibility for schedules and their historical evidence."""

from marvis.db_schema import connect
from marvis.institution.policy import authorize_on_connection
from marvis.institution.store import IdentityError, identity_mode
from marvis.operations.schedule_proposals import schedule_task


def contract_access(db_path, contract, action="read", *, actor_id=None):
    with connect(db_path) as conn:
        if identity_mode(conn) != "institutional":
            return
        try:
            task_id = schedule_task(contract)
        except ValueError as exc:
            raise IdentityError("operations_unbound_schedule_forbidden", 403) from exc
        authorize_on_connection(conn, "task", task_id, action, actor_id=actor_id)


def visible_contract(db_path, contract):
    try:
        contract_access(db_path, contract)
        return True
    except IdentityError as exc:
        if exc.status in (403, 404):
            return False
        raise
