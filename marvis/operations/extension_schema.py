"""Migration 45: additive immutable operations approvals and feedback evidence."""


def create_operations_extensions_schema(conn):
    statements = (
        """CREATE TABLE IF NOT EXISTS operations_native_invocations(
          invocation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES operations_runs(run_id),
          schedule_id TEXT NOT NULL, schedule_revision INTEGER NOT NULL, task_id TEXT NOT NULL,
          actor_id TEXT NOT NULL, tool_ref TEXT NOT NULL, installed_invocation_hash TEXT NOT NULL,
          inputs_hash TEXT NOT NULL, dispatch_started_at TEXT NOT NULL,
          FOREIGN KEY(schedule_id,schedule_revision) REFERENCES operations_schedules(schedule_id,revision))""",
        """CREATE TABLE IF NOT EXISTS operations_schedule_proposals (
          proposal_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, task_id TEXT NOT NULL,
          expected_revision INTEGER NOT NULL CHECK(expected_revision>=0), contract_json TEXT NOT NULL,
          contract_hash TEXT NOT NULL CHECK(length(contract_hash)=64), instruction_hash TEXT NOT NULL,
          binding_hash TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS operations_schedule_confirmations (
          proposal_id TEXT PRIMARY KEY REFERENCES operations_schedule_proposals(proposal_id),
          schedule_id TEXT NOT NULL, revision INTEGER NOT NULL, contract_hash TEXT NOT NULL,
          actor_id TEXT NOT NULL, confirmed_at TEXT NOT NULL, UNIQUE(schedule_id,revision))""",
        """CREATE TABLE IF NOT EXISTS operations_disposition_claims (
          idempotency_key TEXT PRIMARY KEY, outcome_id TEXT NOT NULL UNIQUE,
          actor_id TEXT NOT NULL, outcome_hash TEXT NOT NULL, request_hash TEXT NOT NULL,
          request_json TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('claimed','completed','failed')),
          result_json TEXT, result_hash TEXT, created_at TEXT NOT NULL, finished_at TEXT,
          CHECK((state='claimed' AND result_json IS NULL AND result_hash IS NULL AND finished_at IS NULL)
            OR (state IN ('completed','failed') AND result_json IS NOT NULL AND result_hash IS NOT NULL AND finished_at IS NOT NULL)))""",
        """CREATE TABLE IF NOT EXISTS operations_decision_feedback (
          environment TEXT NOT NULL, request_id TEXT NOT NULL, task_id TEXT NOT NULL,
          decision_output_hash TEXT NOT NULL, package_hash TEXT NOT NULL, label INTEGER NOT NULL CHECK(label IN (0,1)),
          event_at TEXT NOT NULL, available_at TEXT NOT NULL, source_dataset_id TEXT NOT NULL,
          source_content_hash TEXT NOT NULL, source_row INTEGER NOT NULL CHECK(source_row>=0),
          actor_id TEXT NOT NULL, feedback_hash TEXT NOT NULL, recorded_at TEXT NOT NULL,
          PRIMARY KEY(environment,request_id))""",
        """CREATE INDEX IF NOT EXISTS operations_feedback_window ON operations_decision_feedback
          (task_id,environment,package_hash,event_at,available_at)""",
        """CREATE TRIGGER IF NOT EXISTS operations_disposition_identity_immutable BEFORE UPDATE ON operations_disposition_claims
          WHEN NEW.idempotency_key IS NOT OLD.idempotency_key OR NEW.outcome_id IS NOT OLD.outcome_id
          OR NEW.actor_id IS NOT OLD.actor_id OR NEW.outcome_hash IS NOT OLD.outcome_hash
          OR NEW.request_hash IS NOT OLD.request_hash OR NEW.request_json IS NOT OLD.request_json
          OR NEW.created_at IS NOT OLD.created_at OR OLD.state != 'claimed'
          OR NEW.state NOT IN ('completed','failed')
          BEGIN SELECT RAISE(ABORT,'operations disposition claim is immutable'); END""",
    )
    for statement in statements:
        conn.execute(statement)
    for table in (
        "operations_schedule_proposals",
        "operations_schedule_confirmations",
        "operations_decision_feedback",
        "operations_native_invocations",
    ):
        conn.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT,'operations evidence is immutable'); END"
        )
    for table in (
        "operations_schedule_proposals",
        "operations_schedule_confirmations",
        "operations_decision_feedback",
        "operations_disposition_claims",
        "operations_native_invocations",
    ):
        conn.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} BEGIN SELECT RAISE(ABORT,'operations evidence is retained'); END"
        )
