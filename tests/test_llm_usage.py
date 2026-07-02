from pathlib import Path

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import init_db, llm_usage_summary, record_llm_call


def _db(tmp_path: Path) -> Path:
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    return db_path


def test_record_llm_call_and_usage_summary(tmp_path):
    db_path = _db(tmp_path)
    record_llm_call(
        db_path,
        {
            "caller": "gate",
            "model_id": "m1",
            "prompt_chars": 100,
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "latency_ms": 200,
            "ok": True,
            "error_kind": None,
            "retry_count": 0,
            "streamed": False,
        },
    )
    record_llm_call(
        db_path,
        {
            "caller": "gate",
            "model_id": "m1",
            "prompt_chars": 100,
            "latency_ms": 400,
            "ok": False,
            "error_kind": "timeout",
            "retry_count": 1,
            "streamed": False,
        },
    )
    record_llm_call(
        db_path,
        {
            "caller": "planner",
            "model_id": "m1",
            "latency_ms": 50,
            "ok": True,
            "retry_count": 0,
            "streamed": True,
        },
    )

    summary = {row["caller"]: row for row in llm_usage_summary(db_path)}
    gate = summary["gate"]
    assert gate["calls"] == 2
    assert gate["failures"] == 1
    assert gate["failure_rate"] == 0.5
    assert gate["avg_latency_ms"] == 300.0
    assert gate["total_retries"] == 1
    assert gate["prompt_tokens"] == 40
    assert summary["planner"]["calls"] == 1
    assert summary["planner"]["failures"] == 0


def test_record_llm_call_writes_audit_row(tmp_path):
    from marvis.db import _list_audit_rows

    db_path = _db(tmp_path)
    record_llm_call(
        db_path,
        {
            "caller": "router",
            "model_id": "m1",
            "latency_ms": 12,
            "ok": True,
            "retry_count": 0,
            "streamed": False,
        },
    )
    rows = _list_audit_rows(db_path, kind="llm.call")
    assert len(rows) == 1
    assert rows[0]["target_ref"] == "router"
    assert rows[0]["outcome"] == "ok"
    assert rows[0]["detail"]["latency_ms"] == 12


def test_llm_usage_endpoint_returns_aggregate(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    db_path = tmp_path / "marvis.sqlite"
    record_llm_call(
        db_path,
        {
            "caller": "critic",
            "model_id": "m1",
            "latency_ms": 25,
            "ok": True,
            "retry_count": 0,
            "streamed": False,
        },
    )

    response = client.get("/api/llm/usage")
    assert response.status_code == 200, response.text
    payload = response.json()
    callers = {row["caller"]: row for row in payload["callers"]}
    assert callers["critic"]["calls"] == 1
    assert callers["critic"]["failures"] == 0
