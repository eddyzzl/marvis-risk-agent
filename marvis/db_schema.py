import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MIGRATION_TABLES = frozenset({
    "tasks",
    "jobs",
    "plans",
    "plan_steps",
    "plan_step_outputs",
    "plan_step_output_versions",
    "plan_step_runs",
    "model_artifacts",
    "llm_calls",
    "datasets",
})


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL DEFAULT 'validation',
                    model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                validator TEXT NOT NULL,
                source_dir TEXT NOT NULL,
                algorithm TEXT NOT NULL DEFAULT 'lgb',
                run_mode TEXT NOT NULL DEFAULT 'manual',
                target_col TEXT NOT NULL DEFAULT 'y',
                score_col TEXT NOT NULL DEFAULT 'pred',
                split_col TEXT NOT NULL DEFAULT 'split',
                time_col TEXT NOT NULL DEFAULT 'apply_month',
                target_type TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                status_message TEXT NOT NULL,
                status_reason_code TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(
            conn,
            table="tasks",
            column="task_type",
            definition="TEXT NOT NULL DEFAULT 'validation'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="algorithm",
            definition="TEXT NOT NULL DEFAULT 'lgb'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="run_mode",
            definition="TEXT NOT NULL DEFAULT 'manual'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="target_col",
            definition="TEXT NOT NULL DEFAULT 'y'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="score_col",
            definition="TEXT NOT NULL DEFAULT 'pred'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="split_col",
            definition="TEXT NOT NULL DEFAULT 'split'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="time_col",
            definition="TEXT NOT NULL DEFAULT 'apply_month'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="feature_columns_json",
            definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="recipes_json",
            definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="target_type",
            definition="TEXT NOT NULL DEFAULT ''",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="sample_weight_col",
            definition="TEXT NOT NULL DEFAULT ''",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="oot_ks_min",
            definition="REAL",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="metrics_json",
            definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="capability_tier",
            definition="TEXT NOT NULL DEFAULT ''",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="notebook_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="sample_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="pmml_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="dictionary_path",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="report_values_json",
            definition="TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="report_values_revision",
            definition="INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            conn,
            table="tasks",
            column="status_reason_code",
            definition="TEXT NOT NULL DEFAULT ''",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC, id DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_message TEXT NOT NULL DEFAULT '',
                error_name TEXT,
                error_value TEXT,
                traceback TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                log_path TEXT,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_column(
            conn,
            table="jobs",
            column="heartbeat_at",
            definition="TEXT",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_task ON jobs(task_id, kind, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)"
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_active_task
                ON jobs(task_id)
             WHERE status IN ('queued', 'running')
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                role TEXT NOT NULL,
                stage TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_messages_task
                ON agent_messages(task_id, created_at, id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plugins (
                name TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                module TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                checksum TEXT NOT NULL DEFAULT '',
                builtin INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                installed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                plugin TEXT NOT NULL,
                name TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                input_schema_json TEXT NOT NULL,
                output_schema_json TEXT NOT NULL,
                determinism TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL,
                failure_policy TEXT NOT NULL,
                side_effects_json TEXT NOT NULL DEFAULT '[]',
                entrypoint TEXT NOT NULL DEFAULT '',
                memory_limit_mb INTEGER NOT NULL DEFAULT 2048,
                PRIMARY KEY (plugin, name),
                FOREIGN KEY(plugin) REFERENCES plugins(name) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                actor TEXT,
                target_ref TEXT,
                inputs_hash TEXT,
                outcome TEXT,
                detail_json TEXT NOT NULL DEFAULT '{}',
                at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                caller TEXT NOT NULL,
                model_id TEXT,
                prompt_chars INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                latency_ms INTEGER,
                ok INTEGER NOT NULL,
                error_kind TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                streamed INTEGER NOT NULL DEFAULT 0,
                prompt_name TEXT,
                prompt_version INTEGER,
                truncated INTEGER NOT NULL DEFAULT 0,
                at TEXT NOT NULL
            )
            """
        )
        # LLM-10: prompt_name/prompt_version trace which marvis.llm_prompts
        # PromptSpec was live for a call. LLM-5: truncated flags a call whose
        # prompt was cut down to fit the model's context_window budget.
        _ensure_column(
            conn,
            table="llm_calls",
            column="prompt_name",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="llm_calls",
            column="prompt_version",
            definition="INTEGER",
        )
        _ensure_column(
            conn,
            table="llm_calls",
            column="truncated",
            definition="INTEGER NOT NULL DEFAULT 0",
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                source TEXT NOT NULL,
                template_id TEXT,
                autonomy_level INTEGER NOT NULL,
                status TEXT NOT NULL,
                novel_mode TEXT NOT NULL DEFAULT 'plan_ahead',
                tier TEXT NOT NULL DEFAULT 'balanced',
                replan_count INTEGER NOT NULL DEFAULT 0,
                loop_events_json TEXT NOT NULL DEFAULT '[]',
                success_criteria_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(
            conn,
            table="plans",
            column="novel_mode",
            definition="TEXT NOT NULL DEFAULT 'plan_ahead'",
        )
        _ensure_column(
            conn,
            table="plans",
            column="tier",
            definition="TEXT NOT NULL DEFAULT 'balanced'",
        )
        _ensure_column(
            conn,
            table="plans",
            column="replan_count",
            definition="INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            conn,
            table="plans",
            column="loop_events_json",
            definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table="plans",
            column="success_criteria_json",
            definition="TEXT NOT NULL DEFAULT '[]'",
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_steps (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                title TEXT NOT NULL,
                tool_plugin TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_version TEXT,
                inputs_json TEXT NOT NULL,
                depends_on_json TEXT NOT NULL,
                post_checks_json TEXT NOT NULL,
                needs_confirmation INTEGER NOT NULL,
                decision_point INTEGER NOT NULL DEFAULT 0,
                sub_agent_scope TEXT,
                granted_tools_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                sub_agent_id TEXT,
                output_ref TEXT,
                review_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                phase TEXT,
                confirmed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_column(
            conn,
            table="plan_steps",
            column="phase",
            definition="TEXT",
        )
        _ensure_column(
            conn,
            table="plan_steps",
            column="decision_point",
            definition="INTEGER NOT NULL DEFAULT 0",
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_step_outputs (
                step_id TEXT PRIMARY KEY,
                output_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(step_id) REFERENCES plan_steps(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_column(
            conn,
            table="plan_step_outputs",
            column="evidence_json",
            definition="TEXT NOT NULL DEFAULT '{}'",
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_step_output_versions (
                step_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                output_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                PRIMARY KEY(step_id, version),
                FOREIGN KEY(step_id) REFERENCES plan_steps(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_column(
            conn,
            table="plan_step_output_versions",
            column="evidence_json",
            definition="TEXT NOT NULL DEFAULT '{}'",
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO plan_step_output_versions(step_id, version, output_json, created_at)
            SELECT step_id, 1, output_json, created_at FROM plan_step_outputs
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_step_runs (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                tool_ref TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL DEFAULT '{}',
                output_ref TEXT,
                error TEXT,
                error_kind TEXT,
                duration_ms INTEGER,
                side_effects_json TEXT NOT NULL DEFAULT '[]',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE,
                FOREIGN KEY(step_id) REFERENCES plan_steps(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_step_runs_step ON plan_step_runs(step_id, attempt)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_summaries (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sub_agents (
                id TEXT PRIMARY KEY,
                parent_task_id TEXT NOT NULL,
                parent_step_id TEXT,
                scope TEXT NOT NULL,
                granted_tools_json TEXT NOT NULL DEFAULT '[]',
                context_budget INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_ref TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                role TEXT NOT NULL,
                source_path TEXT NOT NULL,
                format TEXT NOT NULL,
                sheet TEXT,
                row_count INTEGER NOT NULL,
                columns_json TEXT NOT NULL,
                has_target INTEGER NOT NULL,
                target_col TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(
            conn,
            table="datasets",
            column="content_hash",
            definition="TEXT",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_content_hash ON datasets(content_hash)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS joins (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                anchor_dataset_id TEXT NOT NULL,
                joins_json TEXT NOT NULL,
                status TEXT NOT NULL,
                result_dataset_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT,
                artifact_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_artifacts (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                model_path TEXT NOT NULL,
                pmml_path TEXT,
                feature_list_json TEXT NOT NULL,
                feature_importance_json TEXT NOT NULL DEFAULT '[]',
                params_json TEXT NOT NULL,
                woe_maps_json TEXT,
                scorecard_table_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                rules_json TEXT NOT NULL,
                score_col TEXT,
                default_decision_json TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS backtests (
                id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_notes (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                distilled TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS draft_tools (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                name TEXT NOT NULL,
                summary TEXT NOT NULL,
                code TEXT NOT NULL,
                input_schema_json TEXT NOT NULL,
                output_schema_json TEXT NOT NULL,
                determinism TEXT NOT NULL,
                source TEXT NOT NULL,
                learning_note_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS draft_runs (
                id TEXT PRIMARY KEY,
                draft_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                inputs_hash TEXT NOT NULL,
                ok INTEGER NOT NULL,
                output_json TEXT,
                error TEXT,
                at TEXT NOT NULL,
                FOREIGN KEY (draft_id) REFERENCES draft_tools(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tools_plugin ON tools(plugin)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_kind_at ON audit(kind, at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_kind_at_id ON audit(kind, at, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_at_id ON audit(at, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_target_ref_at ON audit(target_ref, at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_caller_at ON llm_calls(caller, at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_task ON datasets(task_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_task ON experiments(task_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_artifacts_experiment ON model_artifacts(experiment_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_artifacts_experiment_created ON model_artifacts(experiment_id, created_at, id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_artifacts_created ON model_artifacts(created_at, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strategies_task ON strategies(task_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtests_strategy ON backtests(strategy_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_draft_tools_task ON draft_tools(task_id, status)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_draft_tools_task_order ON draft_tools(task_id, created_at, id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_draft_tools_task_created ON draft_tools(task_id, status, created_at, id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_draft_tools_order ON draft_tools(created_at, id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_draft_tools_created ON draft_tools(status, created_at, id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_draft_runs_draft ON draft_runs(draft_id)")
        _ensure_column(conn, "model_artifacts", "feature_importance_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "model_artifacts", "scorecard_table_json", "TEXT NOT NULL DEFAULT '[]'")
        # S1a: direction metadata (nullable -- old rows stay NULL, no backfill).
        _ensure_column(conn, "model_artifacts", "score_direction", "TEXT")
        _ensure_column(conn, "model_artifacts", "points_direction", "TEXT")
        # S1b: training-time baseline distribution snapshot (nullable JSON text --
        # old rows stay NULL, no backfill; monitor_run treats NULL as "no baseline").
        _ensure_column(conn, "model_artifacts", "baseline_distributions_json", "TEXT")
        from marvis.agent_memory.store import ensure_agent_memory_schema

        ensure_agent_memory_schema(conn)



def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    table_sql = _migration_table_identifier(table)
    column_sql = _sql_identifier(column)
    existing_columns = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_sql})").fetchall()
    }
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table_sql} ADD COLUMN {column_sql} {definition}")


def _migration_table_identifier(table: str) -> str:
    if table not in _MIGRATION_TABLES:
        raise ValueError(f"unsupported migration table: {table}")
    return _sql_identifier(table)


def _sql_identifier(identifier: str) -> str:
    if not _SQL_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'



# PERF-6: journal_mode is persisted *in the database file* by SQLite -- once a
# connection has set (or confirmed) WAL for a given db_path, every later
# connection to that same file already opens in WAL mode without re-issuing the
# pragma (verified: a fresh connection reports journal_mode=wal with no pragma
# call at all). The other four pragmas below (synchronous/busy_timeout/
# foreign_keys/temp_store) are connection-scoped, not file-persisted -- SQLite
# resets each to its default on every new connection, so those must keep running
# unconditionally on every connect() or correctness silently regresses (foreign
# keys stop being enforced, busy_timeout drops to 0, etc). This cache only ever
# removes the one redundant "PRAGMA journal_mode=WAL" round-trip per db file,
# never the correctness-critical pragmas.
_WAL_CONFIRMED_LOCK = threading.Lock()
_WAL_CONFIRMED_PATHS: set[str] = set()


def _configure_connection(conn: sqlite3.Connection, *, db_key: str | None = None) -> None:
    already_confirmed = False
    if db_key is not None:
        with _WAL_CONFIRMED_LOCK:
            already_confirmed = db_key in _WAL_CONFIRMED_PATHS
    if already_confirmed:
        mode_row = None
    else:
        mode_row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        # WAL is requested for concurrent readers/writers. It silently degrades on
        # read-only or networked filesystems; surface that instead of assuming the
        # concurrency guarantees hold. In-memory databases legitimately report
        # "memory" and are exempt.
        mode = str(mode_row[0]).lower() if mode_row is not None else None
        if mode is not None and mode not in ("wal", "memory"):
            logger.warning("Failed to enable WAL journal mode; got %r", mode_row[0])
        elif db_key is not None and mode == "wal":
            with _WAL_CONFIRMED_LOCK:
                _WAL_CONFIRMED_PATHS.add(db_key)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")


def sqlite_health(db_path: Path) -> dict[str, object]:
    with connect(db_path) as conn:
        mode_row = conn.execute("PRAGMA journal_mode").fetchone()
        busy_row = conn.execute("PRAGMA busy_timeout").fetchone()
    journal_mode = str(mode_row[0] if mode_row is not None else "unknown").lower()
    busy_timeout_ms = int(busy_row[0]) if busy_row is not None else 0
    return {
        "sqlite_journal_mode": journal_mode,
        "sqlite_wal_degraded": journal_mode not in {"wal", "memory"},
        "sqlite_busy_timeout_ms": busy_timeout_ms,
    }


@contextmanager
def connect(db_path: Path):
    conn = sqlite3.connect(db_path, timeout=5.0, isolation_level="DEFERRED")
    conn.row_factory = sqlite3.Row
    try:
        _configure_connection(conn, db_key=str(db_path))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
