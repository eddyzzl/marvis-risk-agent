import json
import logging
import re
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from marvis.strategy_lifecycle import (
    ASSET_STATUS_ADOPTED_LOCAL,
    ASSET_STATUS_DRAFT,
    ASSET_STATUS_RETIRED,
    LEGACY_STATUS_ADOPTED,
    LEGACY_STATUS_RETIRED,
    StrategyLifecycleError,
    asset_status_from_legacy,
    validate_lifecycle_pair,
)

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
    "strategy_artifacts",
    "validation_input_contracts",
    "data_analysis_runs",
    "data_transform_runs",
    "dataset_lineage_edges",
})

# ARCH-10: schema_version mechanism.
#
# Why PRAGMA user_version (not a schema_version table): user_version is a
# 4-byte integer stored directly in the SQLite file header (offset 60), so
# reading it never depends on whether any of our own tables exist yet --
# unlike a "SELECT version FROM schema_version" table, which itself needs a
# chicken-and-egg bootstrap migration. It participates in the surrounding
# transaction like any other write (verified: a rolled-back transaction that
# set PRAGMA user_version reverts it along with the DDL), so "run migration
# N's DDL, then stamp PRAGMA user_version = N" is atomic per migration. A
# fresh, never-initialized database reports user_version = 0.
#
# _MIGRATIONS is the explicit, ordered, one-directional upgrade path: each
# entry is (version, callable). init_db reads the current user_version and
# runs every migration whose version is greater than it, strictly in order,
# each in its own transaction immediately followed by stamping that
# migration's version. This is the extension point for future schema changes
# -- append a new (version, function) tuple; never edit a migration that has
# already shipped.
#
# _migration_001_baseline freezes the entire pre-ARCH-10 init_db body
# (30+ CREATE TABLE IF NOT EXISTS + all _ensure_column add-column probes +
# the plan_step_output_versions backfill + agent_memory schema) as migration
# 1. It is written the same way it always ran: every statement idempotent
# (IF NOT EXISTS / _ensure_column's existing-column probe / INSERT OR
# IGNORE), so running it against a pre-existing database with any subset of
# these tables/columns already present is a no-op for what's already there
# and additive for what's missing -- byte-for-byte the same behavior as the
# unversioned init_db this replaces. A pre-ARCH-10 database has user_version
# = 0 (SQLite default, since nothing ever set it), so it naturally re-runs
# migration 1 exactly once on the first init_db call after upgrading, then
# is stamped to version 1 and never runs it again.
#
# _migration_002_strategy_versioning (S2) adds strategy version/status
# lifecycle columns and the strategy_artifacts table. It only ever runs on a
# database already stamped at version 1 (which therefore already has the
# strategies table from migration 1), but is written idempotently anyway (a
# table_info probe before each ADD COLUMN, CREATE TABLE/INDEX IF NOT EXISTS)
# so it is safe against any partially-migrated database.
#
# _migration_003_validation_input_contracts adds the immutable validation
# workflow-version discriminator and the normalized, revisioned input contract.
# It backfills only historical validation rows that still carry version 0.
#
# _migration_004_strategy_task_input adds the optional, governed strategy
# business contract. Historical rows remain NULL so callers can distinguish a
# legacy task from an explicitly supplied (possibly still incomplete) contract.
#
# _migration_005_governance_authorization adds the Phase 0B immutable policy
# snapshot plus server-issued local principals, immutable human DecisionRecords,
# one-shot ApprovalRecords, and the crash-safe effect execution ledger used by
# ToolRunner.
#
# _migration_006_strategy_dsl adds the canonical, versioned Strategy DSL payload.
# The columns are nullable on purpose: historical rules_json rows are adapted at
# the repository boundary and are not rewritten during database migration.
#
# _migration_007_pending_strategy_requests moves confirmed-but-not-yet-consumed
# natural-language strategy drafts out of agent message metadata.  The request
# row is task-scoped, integrity-bound and one-shot so a repeated confirmation
# cannot replay the same draft into a second plan.
#
# _migration_008_strategy_monitoring_ledger adds append-only monitoring plan
# revisions and evidence-bound monitoring runs. Runtime tools are wired in a
# later slice; this migration only establishes the durable governance boundary.
#
# _migration_009_strategy_asset_lifecycle adds the canonical strategy asset
# lifecycle while preserving the legacy status field and wire token.
#
# _migration_010_task_artifact_registry adds one immutable, task-owned registry
# for downloadable workflow outputs.  The registry intentionally does not
# backfill legacy workflow-specific artifact tables because those rows do not
# carry the content hash and provenance required by this boundary.
#
# _migration_011_verified_strategy_artifacts adds nullable integrity metadata to
# historical strategy artifacts. Existing rows remain explicit legacy records;
# new verified rows are immutable and content-addressed.
#
# _migration_012_data_workspaces adds the task-scoped, CAS-updated data workspace
# used by the V2 data/semantics experience. Computed analysis stays outside this
# row and is invalidated through its server-owned analysis_generation counter.
#
# _migration_013_data_analysis_runs adds the task-owned, evidence-bound execution
# ledger for deterministic data-analysis artifacts.  The row is idempotent by
# canonical computational input and keeps the originating workspace revision as
# immutable provenance.
#
# _migration_014_data_transform_lineage adds immutable, task-owned successful
# transform records plus explicit parent->child dataset lineage.  A transform
# is bound to the exact source workspace generation and the activated result
# workspace generation; result bytes and structured evidence are content-bound.
SCHEMA_VERSION = 14


def _migration_001_baseline(conn: sqlite3.Connection) -> None:
    """Baseline schema as of ARCH-10 (schema_version introduction). Idempotent:
    every statement uses CREATE TABLE/INDEX IF NOT EXISTS, _ensure_column's
    existing-column probe, or INSERT OR IGNORE, so it is safe to run against
    an empty database, a pre-ARCH-10 database with any subset of these
    tables/columns already present, or (defensively) a database already at
    this version."""
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


def _migration_002_strategy_versioning(conn: sqlite3.Connection) -> None:
    """S2: strategy version/status lifecycle + strategy_artifacts table.

    Runs only against a database already stamped at version 1 (so the
    strategies table from migration 1 already exists), but every statement is
    idempotent -- a table_info probe guards each ADD COLUMN, and the new table
    and its index use CREATE ... IF NOT EXISTS -- so it is safe against any
    partially-migrated database and re-runnable without error."""
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(strategies)").fetchall()
    }
    for column, definition in (
        ("version", "INTEGER NOT NULL DEFAULT 1"),
        ("status", "TEXT NOT NULL DEFAULT 'draft'"),
        ("adopted_at", "TEXT"),
        ("adoption_reason", "TEXT"),
        ("parent_strategy_id", "TEXT"),
    ):
        if column not in existing:
            conn.execute(f"ALTER TABLE strategies ADD COLUMN {column} {definition}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_artifacts (
            id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategy_artifacts_strategy"
        " ON strategy_artifacts(strategy_id, created_at, id)"
    )


def _migration_003_validation_input_contracts(conn: sqlite3.Connection) -> None:
    """Add immutable workflow versions and normalized validation contracts.

    A version-2 test/repair database may contain only a subset of baseline tables.
    Guard the tasks backfill so such a database can still advance atomically; a
    normal database always has tasks before this migration runs.
    """
    tasks_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
    ).fetchone()
    if tasks_exists is not None:
        _ensure_column(
            conn,
            table="tasks",
            column="validation_workflow_version",
            definition="INTEGER NOT NULL DEFAULT 0",
        )
        task_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "task_type" in task_columns:
            conn.execute(
                "UPDATE tasks SET validation_workflow_version = 1 "
                "WHERE task_type = 'validation' AND validation_workflow_version = 0"
            )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS validation_input_contracts (
            task_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            candidate_json TEXT NOT NULL,
            confirmed_json TEXT NOT NULL DEFAULT '{}',
            material_hashes_json TEXT NOT NULL,
            sample_schema_json TEXT NOT NULL,
            pmml_manifest_json TEXT NOT NULL,
            metadata_resolution_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
        """
    )


def _migration_004_strategy_task_input(conn: sqlite3.Connection) -> None:
    """Add the optional serialized StrategyTaskInput to task records.

    Versioned test/repair databases can contain only a subset of baseline tables,
    so mirror migration 3's defensive tasks-table probe.
    """
    tasks_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
    ).fetchone()
    if tasks_exists is None:
        return
    _ensure_column(
        conn,
        table="tasks",
        column="strategy_input_json",
        definition="TEXT",
    )


def _migration_005_governance_authorization(conn: sqlite3.Connection) -> None:
    plan_steps_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'plan_steps'"
    ).fetchone()
    if plan_steps_exists is not None:
        _ensure_column(
            conn,
            table="plan_steps",
            column="policy_json",
            definition=(
                "TEXT NOT NULL DEFAULT "
                "'{\"schema_version\":\"tool-policy.v1\","
                "\"human_decision_gate\":\"none\","
                "\"effect_authorization\":\"none\"}'"
            ),
        )
        # ALTER TABLE gives every historical row the permissive default. Raise
        # (never lower) pre-upgrade explicit gates before stamping migration 5,
        # otherwise a later replan could silently discard them.
        legacy_gate_rows = conn.execute(
            "SELECT id, policy_json FROM plan_steps WHERE needs_confirmation = 1"
        ).fetchall()
        for row in legacy_gate_rows:
            try:
                policy = json.loads(str(row["policy_json"] or "{}"))
            except (TypeError, ValueError):
                policy = {}
            if not isinstance(policy, dict):
                policy = {}
            policy["schema_version"] = "tool-policy.v1"
            policy["human_decision_gate"] = "required"
            policy.setdefault("effect_authorization", "none")
            conn.execute(
                "UPDATE plan_steps SET policy_json = ? WHERE id = ?",
                (
                    json.dumps(
                        policy,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    row["id"],
                ),
            )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS local_principals (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK(kind = 'local_session'),
            display_name TEXT NOT NULL,
            session_token_hash TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK(status IN ('active', 'expired', 'revoked')),
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_records (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            plan_id TEXT NOT NULL,
            plan_revision INTEGER NOT NULL,
            step_id TEXT NOT NULL,
            tool_ref TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('approve', 'reject')),
            reason TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            evidence_hash TEXT NOT NULL,
            effect_target_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(principal_id) REFERENCES local_principals(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_decision_records_step "
        "ON decision_records(plan_id, step_id, created_at)"
    )
    # A DecisionRecord is evidence, not mutable application state. Reversal is
    # represented by a later DecisionRecord/Approval revocation; silently
    # rewriting or deleting the original would destroy the audit chain.
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_decision_records_immutable_update
        BEFORE UPDATE ON decision_records
        BEGIN
            SELECT RAISE(ABORT, 'decision_records are immutable');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_decision_records_immutable_delete
        BEFORE DELETE ON decision_records
        BEGIN
            SELECT RAISE(ABORT, 'decision_records are immutable');
        END
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_records (
            id TEXT PRIMARY KEY,
            decision_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            plan_id TEXT NOT NULL,
            plan_revision INTEGER NOT NULL,
            step_id TEXT NOT NULL,
            tool_ref TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            evidence_hash TEXT NOT NULL,
            effect_target_json TEXT NOT NULL,
            nonce TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK(
                status IN ('issued', 'reserved', 'consumed', 'expired', 'revoked')
            ),
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            reserved_at TEXT,
            reservation_id TEXT,
            consumed_at TEXT,
            revoked_at TEXT,
            revoke_reason TEXT,
            FOREIGN KEY(decision_id) REFERENCES decision_records(id),
            FOREIGN KEY(principal_id) REFERENCES local_principals(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_approval_records_step "
        "ON approval_records(plan_id, step_id, issued_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_approval_records_status "
        "ON approval_records(status, expires_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS effect_executions (
            id TEXT PRIMARY KEY,
            approval_id TEXT NOT NULL,
            reservation_id TEXT NOT NULL UNIQUE,
            runtime_generation TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN ('prepared', 'dispatched', 'committed', 'uncertain')
            ),
            prepared_at TEXT NOT NULL,
            dispatched_at TEXT,
            committed_at TEXT,
            uncertain_at TEXT,
            released_at TEXT,
            release_reason TEXT,
            uncertain_reason TEXT,
            result_hash TEXT,
            detail_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(approval_id) REFERENCES approval_records(id)
        )
        """
    )
    # An approval is one-shot. A preparation that provably never dispatched may
    # be released (released_at != NULL), after which the same still-valid
    # approval can be reserved again. Dispatched/uncertain/committed executions
    # keep the uniqueness lock permanently and can never be blindly replayed.
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_effect_executions_active_approval
            ON effect_executions(approval_id)
         WHERE released_at IS NULL
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_effect_executions_status "
        "ON effect_executions(status, prepared_at)"
    )


def _migration_006_strategy_dsl(conn: sqlite3.Connection) -> None:
    """Add the canonical Strategy DSL without mutating historical definitions."""

    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'strategies'"
    ).fetchone()
    if table is None:
        # Some compatibility tests and repaired historical databases carry a
        # legitimate schema version but only the task subset. There is no strategy
        # definition to migrate in that shape, so advancing the version is safe.
        return
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(strategies)").fetchall()
    }
    for column, definition in (
        ("dsl_json", "TEXT"),
        ("dsl_schema_version", "TEXT"),
    ):
        if column not in existing:
            conn.execute(f"ALTER TABLE strategies ADD COLUMN {column} {definition}")


def _migration_007_pending_strategy_requests(conn: sqlite3.Connection) -> None:
    """Add one-shot storage for validated natural-language strategy drafts.

    Only validated control-plane JSON belongs here.  Dataset identity is a
    fingerprint/locator object; sample rows are never materialized into this
    table.  Status is constrained at the database boundary because the
    repository relies on a conditional ``pending -> terminal`` update for
    replay protection.
    """

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_strategy_requests (
            id TEXT PRIMARY KEY,
            nonce TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL,
            validated_draft_json TEXT NOT NULL,
            dataset_identity_json TEXT NOT NULL,
            target_col TEXT,
            payload_sha256 TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ('pending', 'consumed', 'cancelled', 'invalidated')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pending_strategy_requests_task_status
            ON pending_strategy_requests(task_id, status, created_at, id)
        """
    )


def _migration_008_strategy_monitoring_ledger(conn: sqlite3.Connection) -> None:
    """Add immutable plan revisions and evidence-bound monitoring runs."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_monitoring_plans (
            id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            strategy_version INTEGER NOT NULL CHECK(strategy_version >= 1),
            revision INTEGER NOT NULL CHECK(revision >= 1),
            schema_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL CHECK(length(payload_hash) = 64),
            supersedes_plan_id TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(strategy_id, revision),
            UNIQUE(id, strategy_id),
            FOREIGN KEY(strategy_id) REFERENCES strategies(id) ON DELETE CASCADE,
            FOREIGN KEY(supersedes_plan_id)
                REFERENCES strategy_monitoring_plans(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_monitoring_plans_latest
            ON strategy_monitoring_plans(strategy_id, revision DESC, created_at DESC, id DESC)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_monitoring_plans_payload_hash
            ON strategy_monitoring_plans(strategy_id, payload_hash)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_strategy_monitoring_plans_immutable_update
        BEFORE UPDATE ON strategy_monitoring_plans
        BEGIN
            SELECT RAISE(ABORT, 'strategy_monitoring_plans are immutable');
        END
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_monitoring_runs (
            id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            monitoring_plan_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            dataset_content_hash TEXT NOT NULL CHECK(length(dataset_content_hash) = 64),
            strategy_effect_hash TEXT NOT NULL CHECK(length(strategy_effect_hash) = 64),
            economics_binding_hash TEXT NOT NULL CHECK(length(economics_binding_hash) = 64),
            result_json TEXT NOT NULL,
            result_hash TEXT NOT NULL CHECK(length(result_hash) = 64),
            overall_level TEXT NOT NULL
                CHECK(overall_level IN ('green', 'amber', 'red', 'n/a')),
            created_at TEXT NOT NULL,
            UNIQUE(
                monitoring_plan_id,
                dataset_content_hash,
                strategy_effect_hash,
                economics_binding_hash
            ),
            FOREIGN KEY(strategy_id) REFERENCES strategies(id) ON DELETE CASCADE,
            FOREIGN KEY(monitoring_plan_id, strategy_id)
                REFERENCES strategy_monitoring_plans(id, strategy_id)
                ON DELETE CASCADE,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_monitoring_runs_strategy
            ON strategy_monitoring_runs(strategy_id, created_at, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_monitoring_runs_plan
            ON strategy_monitoring_runs(monitoring_plan_id, created_at, id)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_strategy_monitoring_runs_immutable_update
        BEFORE UPDATE ON strategy_monitoring_runs
        BEGIN
            SELECT RAISE(ABORT, 'strategy_monitoring_runs are immutable');
        END
        """
    )


def _migration_009_strategy_asset_lifecycle(conn: sqlite3.Connection) -> None:
    """Add and backfill canonical lifecycle state without inventing deployment."""

    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'strategies'"
    ).fetchone()
    if table is None:
        # Defensive compatibility for tests/tools that stamp a deliberately
        # partial historical schema. A real MARVIS v8 database owns this table.
        return
    strategy_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(strategies)").fetchall()
    }
    if "status" not in strategy_columns:
        # The migration test/repair boundary permits deliberately partial
        # historical schemas. A real MARVIS v8 strategies table owns status.
        return
    rows = conn.execute("SELECT id, status FROM strategies ORDER BY id").fetchall()
    for row in rows:
        try:
            asset_status_from_legacy(row["status"])
        except StrategyLifecycleError as exc:
            raise StrategyLifecycleError(
                f"unknown legacy strategy status for {row['id']}: {row['status']!r}"
            ) from exc

    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(strategies)").fetchall()
    }
    if "asset_status" not in columns:
        conn.execute(
            """
            ALTER TABLE strategies ADD COLUMN asset_status
                TEXT NOT NULL DEFAULT 'draft'
                CHECK(asset_status IN ('draft', 'validated', 'adopted_local', 'retired'))
            """
        )

    # The default is intentionally ``draft`` so old draft rows remain stable.
    # The two updates also complete an idempotent, partially applied migration.
    conn.execute(
        "UPDATE strategies SET asset_status = ? "
        "WHERE status = ? AND asset_status = ?",
        (ASSET_STATUS_ADOPTED_LOCAL, LEGACY_STATUS_ADOPTED, ASSET_STATUS_DRAFT),
    )
    conn.execute(
        "UPDATE strategies SET asset_status = ? "
        "WHERE status = ? AND asset_status = ?",
        (ASSET_STATUS_RETIRED, LEGACY_STATUS_RETIRED, ASSET_STATUS_DRAFT),
    )

    rows = conn.execute(
        "SELECT id, status, asset_status FROM strategies ORDER BY id"
    ).fetchall()
    for row in rows:
        try:
            validate_lifecycle_pair(row["status"], row["asset_status"])
        except StrategyLifecycleError as exc:
            raise StrategyLifecycleError(
                "strategy lifecycle drift for "
                f"{row['id']}: status={row['status']!r}, "
                f"asset_status={row['asset_status']!r}"
            ) from exc


def _migration_010_task_artifact_registry(conn: sqlite3.Connection) -> None:
    """Add the immutable, task-scoped artifact provenance registry."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_artifacts (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            content_hash TEXT NOT NULL
                CHECK(length(content_hash) = 64)
                CHECK(content_hash NOT GLOB '*[^0-9a-f]*'),
            origin_tool TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(task_id, kind, path),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_task_artifacts_task_created
            ON task_artifacts(task_id, created_at, id)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_task_artifacts_immutable_update
        BEFORE UPDATE ON task_artifacts
        BEGIN
            SELECT RAISE(ABORT, 'task_artifacts are immutable');
        END
        """
    )


def _migration_011_verified_strategy_artifacts(conn: sqlite3.Connection) -> None:
    """Add immutable integrity metadata without fabricating legacy hashes."""

    table = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type = 'table' AND name = 'strategy_artifacts'"
    ).fetchone()
    if table is None:
        # Synthetic partial-schema migration tests may not own the V2 strategy
        # tables. A normal database always created this table in migration 2.
        return
    _ensure_column(
        conn,
        "strategy_artifacts",
        "content_hash",
        "TEXT CHECK(content_hash IS NULL OR "
        "(length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'))",
    )
    _ensure_column(
        conn,
        "strategy_artifacts",
        "content_size",
        "INTEGER CHECK(content_size IS NULL OR content_size >= 0)",
    )
    _ensure_column(conn, "strategy_artifacts", "provenance_json", "TEXT")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_artifacts_verified_content
            ON strategy_artifacts(strategy_id, kind, content_hash)
         WHERE content_hash IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_artifacts_verified_path
            ON strategy_artifacts(strategy_id, kind, path)
         WHERE content_hash IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_strategy_artifacts_integrity_triplet
        BEFORE INSERT ON strategy_artifacts
        WHEN NOT (
            (NEW.content_hash IS NULL
             AND NEW.content_size IS NULL
             AND NEW.provenance_json IS NULL)
            OR
            (NEW.content_hash IS NOT NULL
             AND NEW.content_size IS NOT NULL
             AND NEW.provenance_json IS NOT NULL)
        )
        BEGIN
            SELECT RAISE(ABORT, 'strategy artifact integrity metadata must be complete');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_strategy_artifacts_immutable_update
        BEFORE UPDATE ON strategy_artifacts
        BEGIN
            SELECT RAISE(ABORT, 'strategy_artifacts are immutable');
        END
        """
    )


def _migration_012_data_workspaces(conn: sqlite3.Connection) -> None:
    """Add the canonical, task-scoped data-workspace snapshot."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_workspaces (
            task_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL
                CHECK(schema_version = 'data-workspace.v1'),
            revision INTEGER NOT NULL CHECK(revision >= 1),
            active_dataset_id TEXT,
            active_dataset_content_hash TEXT
                CHECK(active_dataset_content_hash IS NULL OR
                    (length(active_dataset_content_hash) = 64
                     AND active_dataset_content_hash NOT GLOB '*[^0-9a-f]*')),
            analysis_generation INTEGER NOT NULL
                CHECK(analysis_generation >= 0),
            page TEXT NOT NULL
                CHECK(page IN ('overview', 'fields', 'semantics',
                               'history', 'statistics')),
            selected_field TEXT,
            semantic_mapping_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(
                (active_dataset_id IS NULL AND active_dataset_content_hash IS NULL)
                OR
                (active_dataset_id IS NOT NULL
                 AND active_dataset_content_hash IS NOT NULL)
            ),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(active_dataset_id) REFERENCES datasets(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_data_workspaces_active_dataset
            ON data_workspaces(active_dataset_id)
        """
    )


def _migration_013_data_analysis_runs(conn: sqlite3.Connection) -> None:
    """Add idempotent, task-owned data-analysis execution records."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_analysis_runs (
            id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL
                CHECK(schema_version = 'data-analysis.v1'),
            task_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            dataset_content_hash TEXT NOT NULL
                CHECK(length(dataset_content_hash) = 64)
                CHECK(dataset_content_hash NOT GLOB '*[^0-9a-f]*'),
            workspace_revision INTEGER NOT NULL CHECK(workspace_revision >= 0),
            analysis_generation INTEGER NOT NULL CHECK(analysis_generation >= 0),
            semantic_mapping_hash TEXT NOT NULL
                CHECK(length(semantic_mapping_hash) = 64)
                CHECK(semantic_mapping_hash NOT GLOB '*[^0-9a-f]*'),
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL
                CHECK(length(config_hash) = 64)
                CHECK(config_hash NOT GLOB '*[^0-9a-f]*'),
            producer_version TEXT NOT NULL,
            input_hash TEXT NOT NULL
                CHECK(length(input_hash) = 64)
                CHECK(input_hash NOT GLOB '*[^0-9a-f]*'),
            job_id TEXT,
            status TEXT NOT NULL
                CHECK(status IN (
                    'queued', 'running', 'succeeded', 'failed', 'cancelled'
                )),
            result_artifact_id TEXT,
            result_content_hash TEXT
                CHECK(result_content_hash IS NULL OR
                    (length(result_content_hash) = 64
                     AND result_content_hash NOT GLOB '*[^0-9a-f]*')),
            error_kind TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            UNIQUE(task_id, input_hash),
            CHECK(
                (result_artifact_id IS NULL AND result_content_hash IS NULL)
                OR
                (result_artifact_id IS NOT NULL AND result_content_hash IS NOT NULL)
            ),
            CHECK(
                (error_kind IS NULL AND error_message IS NULL)
                OR
                (error_kind IS NOT NULL AND error_message IS NOT NULL)
            ),
            CHECK(
                (status = 'queued'
                 AND started_at IS NULL
                 AND completed_at IS NULL
                 AND result_artifact_id IS NULL
                 AND error_kind IS NULL)
                OR
                (status = 'running'
                 AND started_at IS NOT NULL
                 AND completed_at IS NULL
                 AND result_artifact_id IS NULL
                 AND error_kind IS NULL)
                OR
                (status = 'succeeded'
                 AND started_at IS NOT NULL
                 AND completed_at IS NOT NULL
                 AND result_artifact_id IS NOT NULL
                 AND error_kind IS NULL)
                OR
                (status IN ('failed', 'cancelled')
                 AND completed_at IS NOT NULL
                 AND result_artifact_id IS NULL
                 AND error_kind IS NOT NULL)
            ),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE SET NULL,
            FOREIGN KEY(result_artifact_id) REFERENCES task_artifacts(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_data_analysis_runs_task
            ON data_analysis_runs(task_id, created_at DESC, id DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_data_analysis_runs_current
            ON data_analysis_runs(
                task_id, dataset_id, dataset_content_hash,
                analysis_generation, semantic_mapping_hash,
                config_hash, producer_version, status
            )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_data_analysis_runs_status
            ON data_analysis_runs(status, created_at, id)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_data_analysis_runs_identity_immutable
        BEFORE UPDATE ON data_analysis_runs
        WHEN NEW.id IS NOT OLD.id
          OR NEW.schema_version IS NOT OLD.schema_version
          OR NEW.task_id IS NOT OLD.task_id
          OR NEW.dataset_id IS NOT OLD.dataset_id
          OR NEW.dataset_content_hash IS NOT OLD.dataset_content_hash
          OR NEW.workspace_revision IS NOT OLD.workspace_revision
          OR NEW.analysis_generation IS NOT OLD.analysis_generation
          OR NEW.semantic_mapping_hash IS NOT OLD.semantic_mapping_hash
          OR NEW.config_json IS NOT OLD.config_json
          OR NEW.config_hash IS NOT OLD.config_hash
          OR NEW.producer_version IS NOT OLD.producer_version
          OR NEW.input_hash IS NOT OLD.input_hash
          OR NEW.created_at IS NOT OLD.created_at
        BEGIN
            SELECT RAISE(ABORT, 'data_analysis_runs identity is immutable');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_data_analysis_runs_succeeded_immutable
        BEFORE UPDATE ON data_analysis_runs
        WHEN OLD.status = 'succeeded'
         AND (
             NEW.job_id IS NOT OLD.job_id
             OR NEW.status IS NOT OLD.status
             OR NEW.result_artifact_id IS NOT OLD.result_artifact_id
             OR NEW.result_content_hash IS NOT OLD.result_content_hash
             OR NEW.error_kind IS NOT OLD.error_kind
             OR NEW.error_message IS NOT OLD.error_message
             OR NEW.updated_at IS NOT OLD.updated_at
             OR NEW.started_at IS NOT OLD.started_at
             OR NEW.completed_at IS NOT OLD.completed_at
         )
        BEGIN
            SELECT RAISE(ABORT, 'succeeded data_analysis_run is immutable');
        END
        """
    )


def _migration_014_data_transform_lineage(conn: sqlite3.Connection) -> None:
    """Add immutable data-transform evidence and parent/child lineage."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_transform_runs (
            id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL
                CHECK(schema_version = 'data-transform.v1'),
            task_id TEXT NOT NULL,
            source_dataset_id TEXT NOT NULL,
            source_content_hash TEXT NOT NULL
                CHECK(length(source_content_hash) = 64)
                CHECK(source_content_hash NOT GLOB '*[^0-9a-f]*'),
            workspace_revision INTEGER NOT NULL CHECK(workspace_revision >= 0),
            analysis_generation INTEGER NOT NULL CHECK(analysis_generation >= 0),
            semantic_mapping_hash TEXT NOT NULL
                CHECK(length(semantic_mapping_hash) = 64)
                CHECK(semantic_mapping_hash NOT GLOB '*[^0-9a-f]*'),
            operations_json TEXT NOT NULL,
            operations_hash TEXT NOT NULL
                CHECK(length(operations_hash) = 64)
                CHECK(operations_hash NOT GLOB '*[^0-9a-f]*'),
            producer_version TEXT NOT NULL,
            input_hash TEXT NOT NULL
                CHECK(length(input_hash) = 64)
                CHECK(input_hash NOT GLOB '*[^0-9a-f]*'),
            result_dataset_id TEXT NOT NULL,
            result_content_hash TEXT NOT NULL
                CHECK(length(result_content_hash) = 64)
                CHECK(result_content_hash NOT GLOB '*[^0-9a-f]*'),
            result_artifact_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            result_hash TEXT NOT NULL
                CHECK(length(result_hash) = 64)
                CHECK(result_hash NOT GLOB '*[^0-9a-f]*'),
            result_workspace_revision INTEGER NOT NULL
                CHECK(result_workspace_revision > workspace_revision),
            result_analysis_generation INTEGER NOT NULL
                CHECK(result_analysis_generation = analysis_generation + 1),
            created_at TEXT NOT NULL,
            UNIQUE(task_id, input_hash),
            CHECK(source_dataset_id <> result_dataset_id),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(source_dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(result_dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(result_artifact_id) REFERENCES task_artifacts(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_data_transform_runs_task
            ON data_transform_runs(task_id, created_at DESC, id DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_data_transform_runs_source
            ON data_transform_runs(task_id, source_dataset_id, created_at, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_data_transform_runs_result
            ON data_transform_runs(task_id, result_dataset_id)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_data_transform_runs_immutable
        BEFORE UPDATE ON data_transform_runs
        BEGIN
            SELECT RAISE(ABORT, 'data_transform_runs are immutable');
        END
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_lineage_edges (
            id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL
                CHECK(schema_version = 'dataset-lineage.v1'),
            task_id TEXT NOT NULL,
            parent_dataset_id TEXT NOT NULL,
            child_dataset_id TEXT NOT NULL,
            transform_run_id TEXT NOT NULL,
            relation_kind TEXT NOT NULL CHECK(relation_kind = 'transform'),
            edge_order INTEGER NOT NULL CHECK(edge_order >= 0),
            created_at TEXT NOT NULL,
            UNIQUE(task_id, parent_dataset_id, child_dataset_id, relation_kind),
            CHECK(parent_dataset_id <> child_dataset_id),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(child_dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(transform_run_id) REFERENCES data_transform_runs(id)
                ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dataset_lineage_edges_task
            ON dataset_lineage_edges(task_id, created_at, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dataset_lineage_edges_parent
            ON dataset_lineage_edges(task_id, parent_dataset_id, edge_order, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dataset_lineage_edges_child
            ON dataset_lineage_edges(task_id, child_dataset_id, edge_order, id)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_dataset_lineage_edges_matches_run
        BEFORE INSERT ON dataset_lineage_edges
        WHEN NOT EXISTS (
            SELECT 1
              FROM data_transform_runs AS run
             WHERE run.id = NEW.transform_run_id
               AND run.task_id = NEW.task_id
               AND run.source_dataset_id = NEW.parent_dataset_id
               AND run.result_dataset_id = NEW.child_dataset_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'dataset lineage edge does not match transform run');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_dataset_lineage_edges_immutable
        BEFORE UPDATE ON dataset_lineage_edges
        BEGIN
            SELECT RAISE(ABORT, 'dataset_lineage_edges are immutable');
        END
        """
    )


# Ordered, append-only migration registry. Each entry is
# (version, migration_function). To add a new migration: write a new
# _migration_NNN_description(conn) function, append (NNN, that function) to
# this list with NNN == SCHEMA_VERSION + 1, and bump SCHEMA_VERSION to match.
# Never edit or reorder an existing entry once it has shipped -- databases
# already stamped at that version will never run it again, so a later edit
# would silently diverge from what already-upgraded databases have.
_MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migration_001_baseline),
    (2, _migration_002_strategy_versioning),
    (3, _migration_003_validation_input_contracts),
    (4, _migration_004_strategy_task_input),
    (5, _migration_005_governance_authorization),
    (6, _migration_006_strategy_dsl),
    (7, _migration_007_pending_strategy_requests),
    (8, _migration_008_strategy_monitoring_ledger),
    (9, _migration_009_strategy_asset_lifecycle),
    (10, _migration_010_task_artifact_registry),
    (11, _migration_011_verified_strategy_artifacts),
    (12, _migration_012_data_workspaces),
    (13, _migration_013_data_analysis_runs),
    (14, _migration_014_data_transform_lineage),
]


class SchemaDowngradeError(RuntimeError):
    """Raised when a database's schema_version is newer than this code knows
    how to handle -- i.e. an older marvis checkout opened a database that a
    newer checkout already migrated further. There is no downgrade migration
    path (ARCH-10 explicitly does not attempt one): silently proceeding could
    misinterpret columns/tables a later migration added, so we refuse to
    touch the database instead of risking silent corruption."""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        current_version = _read_schema_version(conn)
        if current_version > SCHEMA_VERSION:
            raise SchemaDowngradeError(
                f"database {db_path} 的 schema_version={current_version} 高于当前代码支持的"
                f" SCHEMA_VERSION={SCHEMA_VERSION}；这通常意味着该数据库被更新版本的 marvis 迁移过，"
                f"而当前运行的是旧代码。请升级 marvis 后再打开此数据库，不支持降级迁移。"
            )
        for version, migration in _MIGRATIONS:
            if version <= current_version:
                continue
            migration(conn)
            conn.execute(f"PRAGMA user_version = {int(version)}")
            conn.commit()
            current_version = version


def _read_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


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
        schema_version = _read_schema_version(conn)
    journal_mode = str(mode_row[0] if mode_row is not None else "unknown").lower()
    busy_timeout_ms = int(busy_row[0]) if busy_row is not None else 0
    return {
        "sqlite_journal_mode": journal_mode,
        "sqlite_wal_degraded": journal_mode not in {"wal", "memory"},
        "sqlite_busy_timeout_ms": busy_timeout_ms,
        "schema_version": schema_version,
        "schema_version_expected": SCHEMA_VERSION,
        "schema_version_stale": schema_version < SCHEMA_VERSION,
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
