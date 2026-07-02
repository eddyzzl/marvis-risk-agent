from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from marvis.db_schema import connect
from marvis.repositories.audit import _write_audit_row

_ERROR_KINDS = frozenset(
    {
        "http_4xx",
        "http_5xx",
        "http_error",
        "timeout",
        "connection",
        "stream_interrupted",
        "context_length_exceeded",
    }
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_error_kind(value) -> str | None:
    if value is None:
        return None
    kind = str(value)
    return kind if kind in _ERROR_KINDS else "other"


def record_llm_call(db_path: Path, record: dict) -> None:
    """Persist one LLM call's metadata to ``llm_calls`` plus an audit checkpoint.

    ``record`` is the lightweight dict emitted by
    ``OpenAICompatibleLLMClient.complete``'s ``on_call_recorded`` hook. This is
    deliberately DB-agnostic on the client side so the client stays offline-
    testable; the wiring lives here.
    """
    caller = str(record.get("caller") or "unknown")
    model_id = record.get("model_id")
    prompt_chars = _as_int(record.get("prompt_chars"))
    prompt_tokens = _as_int(record.get("prompt_tokens"))
    completion_tokens = _as_int(record.get("completion_tokens"))
    latency_ms = _as_int(record.get("latency_ms"))
    ok = 1 if record.get("ok") else 0
    error_kind = _coerce_error_kind(record.get("error_kind"))
    retry_count = _as_int(record.get("retry_count")) or 0
    streamed = 1 if record.get("streamed") else 0
    # LLM-10: which marvis.llm_prompts PromptSpec (name/version) was live for
    # this call. LLM-5: whether the prompt was truncated to fit context_window.
    prompt_name = record.get("prompt_name")
    prompt_name = str(prompt_name) if prompt_name else None
    prompt_version = _as_int(record.get("prompt_version"))
    truncated = 1 if record.get("truncated") else 0
    at = _now()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO llm_calls(
                id, caller, model_id, prompt_chars, prompt_tokens,
                completion_tokens, latency_ms, ok, error_kind, retry_count,
                streamed, prompt_name, prompt_version, truncated, at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                caller,
                model_id,
                prompt_chars,
                prompt_tokens,
                completion_tokens,
                latency_ms,
                ok,
                error_kind,
                retry_count,
                streamed,
                prompt_name,
                prompt_version,
                truncated,
                at,
            ),
        )
        _write_audit_row(
            conn,
            kind="llm.call",
            target_ref=caller,
            outcome="ok" if ok else "error",
            detail={
                "model_id": model_id,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "retry_count": retry_count,
                "error_kind": error_kind,
                "streamed": bool(streamed),
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "truncated": bool(truncated),
            },
        )


def llm_usage_summary(db_path: Path, *, days: int | None = None) -> list[dict]:
    """Aggregate call counts / latency / failure & retry rates grouped by caller."""
    query = (
        "SELECT caller,"
        " COUNT(*) AS calls,"
        " SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failures,"
        " AVG(latency_ms) AS avg_latency_ms,"
        " AVG(retry_count) AS avg_retry_count,"
        " SUM(retry_count) AS total_retries,"
        " SUM(COALESCE(prompt_tokens, 0)) AS prompt_tokens,"
        " SUM(COALESCE(completion_tokens, 0)) AS completion_tokens"
        " FROM llm_calls"
    )
    params: list[object] = []
    if days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=int(days))).isoformat()
        query += " WHERE at >= ?"
        params.append(cutoff)
    query += " GROUP BY caller ORDER BY caller"
    with connect(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    summary: list[dict] = []
    for row in rows:
        calls = int(row["calls"] or 0)
        failures = int(row["failures"] or 0)
        avg_latency = row["avg_latency_ms"]
        summary.append(
            {
                "caller": row["caller"],
                "calls": calls,
                "failures": failures,
                "failure_rate": (failures / calls) if calls else 0.0,
                "avg_latency_ms": (
                    round(float(avg_latency), 1) if avg_latency is not None else None
                ),
                "avg_retry_count": (
                    round(float(row["avg_retry_count"]), 3)
                    if row["avg_retry_count"] is not None
                    else 0.0
                ),
                "total_retries": int(row["total_retries"] or 0),
                "prompt_tokens": int(row["prompt_tokens"] or 0),
                "completion_tokens": int(row["completion_tokens"] or 0),
            }
        )
    return summary


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
