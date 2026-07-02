from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path
import uuid

from marvis.db import _now, connect
from marvis.domain import (
    TASK_STATUS_REASON_SERVER_RESTART,
    TaskStatus,
)
from marvis.orchestrator.contracts import PlanStatus
from marvis.orchestrator.errors import PlanNotFoundError
from marvis.orchestrator.plan_recovery import PlanStepRecovery
from marvis.pipeline import METRICS_STAGE_FAILURE_PREFIX
from marvis.state_machine import ConflictError


logger = logging.getLogger(__name__)

RECLAIM_SERVER_RESTART_MESSAGE = "reclaimed: server restart while running"
METRICS_RECLAIM_RESUME_MESSAGE = (
    METRICS_STAGE_FAILURE_PREFIX + "服务重启中断，可从指标阶段重试"
)


ORPHAN_RECLAIM_STATUSES = frozenset(
    {
        TaskStatus.RUNNING,
        TaskStatus.COMPUTING_METRICS,
    }
)


def last_completed_step(task_dir: Path) -> str | None:
    execution_dir = task_dir / "execution"
    outputs_dir = task_dir / "outputs"
    if (
        (outputs_dir / "validation.xlsx").exists()
        and (outputs_dir / "validation_report.docx").exists()
    ):
        return "artifacts"
    if (
        (execution_dir / "code_model_scores.csv").exists()
        and (execution_dir / "runtime_contract.json").exists()
        and (execution_dir / "model_meta.json").exists()
    ):
        return "notebook"
    if execution_dir.exists():
        return "scan"
    return None


def reclaim_stale_running_tasks(
    db_path: Path,
    *,
    tasks_dir: Path | None = None,
    stale_after_seconds: int = 0,
) -> int:
    cutoff = (
        datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
    ).isoformat()
    orphan_placeholders = ",".join(["?"] * len(ORPHAN_RECLAIM_STATUSES))
    with connect(db_path) as conn:
        reclaimed_task_ids = _stale_task_ids(conn, orphan_placeholders, cutoff)
        metrics_resumable_task_ids = _metrics_resumable_task_ids(
            conn, cutoff, tasks_dir=tasks_dir
        )
        agent_task_ids = _interrupted_agent_task_ids(
            conn, orphan_placeholders, cutoff
        )
        cursor = conn.execute(
            f"""
            UPDATE tasks
               SET status = ?,
                   status_message = ?,
                   status_reason_code = ?,
                   updated_at = ?
             WHERE updated_at <= ?
               AND status IN ({orphan_placeholders})
            """,
            (
                TaskStatus.FAILED.value,
                RECLAIM_SERVER_RESTART_MESSAGE,
                TASK_STATUS_REASON_SERVER_RESTART,
                _now(),
                cutoff,
                *(status.value for status in ORPHAN_RECLAIM_STATUSES),
            ),
        )
        if metrics_resumable_task_ids:
            placeholders = ",".join(["?"] * len(metrics_resumable_task_ids))
            conn.execute(
                f"""
                UPDATE tasks
                   SET status_message = ?,
                       updated_at = ?
                 WHERE id IN ({placeholders})
                   AND status = ?
                """,
                (
                    METRICS_RECLAIM_RESUME_MESSAGE,
                    _now(),
                    *metrics_resumable_task_ids,
                    TaskStatus.FAILED.value,
                ),
            )
        _finalize_interrupted_agent_messages(conn, agent_task_ids)
        _add_agent_restart_notices(conn, agent_task_ids)
        _fail_interrupted_jobs(
            conn,
            task_ids=sorted(set(reclaimed_task_ids) | set(agent_task_ids)),
            cutoff=cutoff,
        )
        if cursor.rowcount:
            logger.info(
                "startup recovery reclaimed %d stale running task(s) as failed",
                cursor.rowcount,
            )
        return cursor.rowcount


def _stale_task_ids(conn, orphan_placeholders: str, cutoff: str) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT id
          FROM tasks
         WHERE updated_at <= ?
           AND status IN ({orphan_placeholders})
        """,
        (cutoff, *(status.value for status in ORPHAN_RECLAIM_STATUSES)),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _metrics_resumable_task_ids(
    conn,
    cutoff: str,
    *,
    tasks_dir: Path | None,
) -> list[str]:
    """Tasks reclaimed out of COMPUTING_METRICS whose on-disk execution
    artifacts already completed the notebook step. These should resume via
    the metrics-only retry path instead of a full notebook re-run."""
    if tasks_dir is None:
        return []
    rows = conn.execute(
        """
        SELECT id
          FROM tasks
         WHERE updated_at <= ?
           AND status = ?
        """,
        (cutoff, TaskStatus.COMPUTING_METRICS.value),
    ).fetchall()
    resumable = []
    for row in rows:
        task_id = str(row[0])
        if last_completed_step(tasks_dir / task_id) == "notebook":
            resumable.append(task_id)
    return resumable


def _interrupted_agent_task_ids(
    conn,
    orphan_placeholders: str,
    cutoff: str,
) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT id
          FROM tasks
         WHERE run_mode = 'agent'
           AND status IN ({orphan_placeholders})
           AND updated_at <= ?
        UNION
        SELECT tasks.id
          FROM tasks
          JOIN jobs ON jobs.task_id = tasks.id
         WHERE tasks.run_mode = 'agent'
           AND jobs.status IN ('queued', 'running')
           AND tasks.updated_at <= ?
        """,
        (*(status.value for status in ORPHAN_RECLAIM_STATUSES), cutoff, cutoff),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _fail_interrupted_jobs(conn, *, task_ids: list[str], cutoff: str) -> None:
    params = [_now()]
    if task_ids:
        placeholders = ",".join(["?"] * len(task_ids))
        scope = f"(task_id IN ({placeholders}) OR created_at <= ?)"
        params.extend(task_ids)
        params.append(cutoff)
    else:
        scope = "created_at <= ?"
        params.append(cutoff)
    conn.execute(
        f"""
        UPDATE jobs
           SET status = 'failed',
               error_name = 'ServerRestart',
               error_value = 'process exited while job was running',
               finished_at = ?
         WHERE status IN ('queued', 'running')
           AND {scope}
        """,
        params,
    )


def _finalize_interrupted_agent_messages(
    conn,
    task_ids: list[str],
) -> None:
    if not task_ids:
        return
    placeholders = ",".join(["?"] * len(task_ids))
    rows = conn.execute(
        f"""
        SELECT id, content, metadata_json
          FROM agent_messages
         WHERE task_id IN ({placeholders})
        """,
        task_ids,
    ).fetchall()
    for message_id, content, metadata_json in rows:
        metadata = _load_metadata(metadata_json)
        if metadata.get("streaming") is not True:
            continue
        metadata["streaming"] = False
        metadata["interrupted_by_restart"] = True
        next_content = _interrupted_agent_content(str(content or ""))
        conn.execute(
            """
            UPDATE agent_messages
               SET content = ?,
                   metadata_json = ?
             WHERE id = ?
            """,
            (
                next_content,
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                message_id,
            ),
        )


def _add_agent_restart_notices(
    conn,
    task_ids: list[str],
) -> None:
    for task_id in task_ids:
        existing = conn.execute(
            """
            SELECT 1
              FROM agent_messages
             WHERE task_id = ?
               AND metadata_json LIKE '%"interrupted_by_restart":true%'
             LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        if existing is not None:
            continue
        now = _now()
        conn.execute(
            """
            INSERT INTO agent_messages
            (id, task_id, role, stage, content, created_at, metadata_json)
            VALUES (?, ?, 'assistant', 'failure', ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                task_id,
                "服务器重启，上一轮 Agent 执行已中断；已保留此前已写入的 Agent 对话和平台产物。你可以继续提问，或根据当前状态重新发送“继续”。",
                now,
                json.dumps(
                    {"interrupted_by_restart": True, "streaming": False},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        )


def _interrupted_agent_content(content: str) -> str:
    if not content.strip():
        return "服务器重启，上一轮 Agent 输出已中断。已保留此前写入的对话和验证结果，你可以继续提问或重新发送“继续”。"
    return content.rstrip() + "\n\n（服务器重启，Agent 输出在此处中断。已保留当前已写入内容。）"


def _load_metadata(value: str | None) -> dict:
    try:
        payload = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


PLAN_RESTART_NOTICE_MARKER = '"plan_interrupted_by_restart":true'


def reclaim_running_plans(
    plan_repo,
    reviewer,
    hook_dispatcher,
    harness_state,
    task_repo,
) -> int:
    """Reclaim V2 plans left in RUNNING by a crash/restart (REL-4).

    Startup-only counterpart to ``PlanExecutor._recover_inflight_steps``: a V2
    driver turn runs synchronously with no job/lock (REL-1), so a crash mid-turn
    leaves ``plans.status='running'`` forever — the task itself never enters
    tasks.status=RUNNING, so ``reclaim_stale_running_tasks`` above never sees it,
    and the plan spins until the user happens to resume() it. This reuses the
    exact same step-run-ledger recovery semantics (``PlanStepRecovery``, shared
    with the executor) for the RUNNING/CHECKING steps, then fails the plan itself,
    drops a Chinese restart notice into the owning task's conversation, and
    releases any orphaned driver job for that task.
    """
    step_recovery = PlanStepRecovery(plan_repo, reviewer, hook_dispatcher, harness_state)
    reclaimed = 0
    for plan in plan_repo.list_plans_by_status(PlanStatus.RUNNING):
        try:
            _reclaim_one_running_plan(plan_repo, step_recovery, task_repo, plan)
        except (PlanNotFoundError, ConflictError):
            # Plan was concurrently resumed/finished between the scan and the
            # reclaim attempt (e.g. a request already in flight when the
            # process restarted mid-request); leave it to the in-flight caller.
            continue
        except Exception:
            logger.exception("failed to reclaim running plan %s", plan.id)
            continue
        reclaimed += 1
    if reclaimed:
        logger.info("startup recovery reclaimed %d running plan(s) as failed", reclaimed)
    return reclaimed


def _reclaim_one_running_plan(plan_repo, step_recovery, task_repo, plan) -> None:
    step_recovery.recover_inflight_steps(plan)
    plan_repo.set_plan_status(plan.id, PlanStatus.FAILED)
    _fail_orphan_task_jobs(task_repo, plan.task_id)
    _add_plan_restart_notice(task_repo, plan.task_id, plan.id)


def _fail_orphan_task_jobs(task_repo, task_id: str) -> None:
    """Release a driver job stuck queued/running for this task (the process
    that owned it died with the plan mid-RUNNING); frees idx_jobs_active_task
    so the task isn't wedged behind a 409 after the restart notice invites a
    retry."""
    with connect(task_repo.db_path) as conn:
        rows = conn.execute(
            """
            SELECT id
              FROM jobs
             WHERE task_id = ?
               AND status IN ('queued', 'running')
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                UPDATE jobs
                   SET status = 'failed',
                       error_name = 'ServerRestart',
                       error_value = 'process exited while job was running',
                       finished_at = ?
                 WHERE id = ?
                   AND status IN ('queued', 'running')
                """,
                (_now(), row["id"]),
            )


def _add_plan_restart_notice(task_repo, task_id: str, plan_id: str) -> None:
    with connect(task_repo.db_path) as conn:
        existing = conn.execute(
            """
            SELECT 1
              FROM agent_messages
             WHERE task_id = ?
               AND metadata_json LIKE ?
             LIMIT 1
            """,
            (task_id, f'%{PLAN_RESTART_NOTICE_MARKER}%"plan_id":"{plan_id}"%'),
        ).fetchone()
        if existing is not None:
            return
        conn.execute(
            """
            INSERT INTO agent_messages
            (id, task_id, role, stage, content, created_at, metadata_json)
            VALUES (?, ?, 'assistant', 'failure', ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                task_id,
                "服务已重启，计划已暂停在当前步骤；中间产物已保留，可点击『重试步骤』从失败步继续。",
                _now(),
                json.dumps(
                    {
                        "plan_interrupted_by_restart": True,
                        "plan_id": plan_id,
                        "streaming": False,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        )
