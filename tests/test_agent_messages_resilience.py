"""The conversation list survives a plan whose result binding is untrusted.

2026-10-19: a historical completion message whose plan step binding fails the
trusted-binding check used to raise ConflictError out of the message list and
turn the whole conversation into a 500.
"""

from __future__ import annotations

from types import SimpleNamespace

from marvis.routers import validation_agent as router
from marvis.state_machine import ConflictError


def test_untrusted_result_binding_skips_the_download_not_the_conversation(monkeypatch):
    plan = SimpleNamespace(id="plan-1")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(plan_repo=SimpleNamespace(list_plans_for_task=lambda _task: [plan])))
    )

    class Composer:
        def latest_result_dataset_metadata(self, _plan):
            raise ConflictError("step plan-1-step-3 dependency plan-1-step-1 result binding is invalid")

    monkeypatch.setattr(router, "_plan_message_composer", lambda _request: Composer())
    messages = [
        {
            "role": "assistant",
            "content": "计划已全部完成。",
            "metadata": {"plan_id": "plan-1", "result_dataset": {"dataset_id": "ds-stale"}},
        },
        {"role": "assistant", "content": "其他消息", "metadata": {}},
    ]
    router._enrich_historical_result_download(request, "task-1", messages)
    # The untrusted download is withheld; the rest of the conversation is intact.
    assert "result_dataset" not in messages[0]["metadata"]
    assert messages[0]["metadata"]["plan_id"] == "plan-1"
    assert messages[1]["content"] == "其他消息"
