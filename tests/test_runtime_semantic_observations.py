"""Safe semantic failure evidence survives isolated benchmark workspace cleanup."""

import json

from marvis.db_schema import init_db
from marvis.domain import TaskCreate
from marvis.orchestrator.eval.runtime_runner import _receipts
from marvis.repositories.tasks import TaskRepository
from marvis.settings import build_settings


def test_runtime_retains_only_filtered_assistant_diagnostics(tmp_path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="test",
            model_version="test",
            validator="qa",
            source_dir=str(tmp_path),
            task_type="portfolio",
        )
    )
    diagnostics = {
        "schema_version": "semantic-diagnostics.v1",
        "stage": "authorization",
        "failure_code": "unauthorized",
        "accepted": False,
        "private": "PRIVATE_CONTRACT",
        "passes": [
            {
                "stage": "authorization",
                "attempts": [
                    {
                        "parse_valid": True,
                        "first_token": "object",
                        "decision": {
                            "verdict": "authorize",
                            "confidence": "medium",
                            "reason": "PRIVATE_REASON",
                            "evidence_quote": "PRIVATE_QUOTE",
                        },
                        "raw": "PRIVATE_RESPONSE",
                        "nested": {"api_key": "PRIVATE_KEY"},
                    }
                ],
            }
        ],
    }
    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content="PRIVATE_USER",
        metadata={"semantic_diagnostics": diagnostics},
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="PRIVATE_ASSISTANT",
        metadata={"semantic_diagnostics": diagnostics},
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="PRIVATE_FUTURE",
        metadata={"semantic_diagnostics": {**diagnostics, "schema_version": "future"}},
    )
    evidence, private = _receipts(tmp_path, task.id)
    observations = evidence["semantic_observations"]
    assert len(observations) == 1 and observations[0]["message_ordinal"] == 2
    assert observations[0]["failure_code"] == "unauthorized"
    assert observations[0]["passes"][0]["attempts"][0]["decision"] == {
        "verdict": "authorize",
        "confidence": "medium",
    }
    assert "PRIVATE" not in json.dumps(evidence)
    assert len(private["messages"]) == 3
    assert "PRIVATE_ASSISTANT" in json.dumps(private)
    assert evidence["plans"] == [] and evidence["steps"] == []


def test_legacy_receipts_without_diagnostics_do_not_gain_empty_evidence(tmp_path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="test",
            model_version="test",
            validator="qa",
            source_dir=str(tmp_path),
            task_type="portfolio",
        )
    )
    repo.add_agent_message(
        task.id, role="assistant", stage="chat", content="old", metadata={}
    )
    evidence, _ = _receipts(tmp_path, task.id)
    assert "semantic_observations" not in evidence
