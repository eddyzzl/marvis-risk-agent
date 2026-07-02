from __future__ import annotations

from fastapi import APIRouter, Request

from marvis.db import llm_usage_summary


router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/usage")
def get_llm_usage(request: Request, days: int | None = None) -> dict:
    """Per-caller aggregate of recorded LLM calls (count / latency / failure / retry)."""
    bounded_days = None
    if days is not None:
        try:
            bounded_days = max(1, int(days))
        except (TypeError, ValueError):
            bounded_days = None
    summary = llm_usage_summary(
        request.app.state.settings.db_path,
        days=bounded_days,
    )
    return {"days": bounded_days, "callers": summary}
