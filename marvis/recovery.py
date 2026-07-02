from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import uuid

from marvis.db import _now, connect
from marvis.domain import (
    TASK_STATUS_REASON_SERVER_RESTART,
    TaskStatus,
)
from marvis.pipeline import METRICS_STAGE_FAILURE_PREFIX

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
