from riskmodel_checker.agent.service import (
    agent_conclusions_confirmed,
    generate_word_conclusions,
)
from riskmodel_checker.domain import TaskRecord, TaskStatus
from riskmodel_checker.llm_client import LLMClientError


def _task() -> TaskRecord:
    return TaskRecord(
        id="task-1",
        model_name="A卡",
        model_version="v1",
        validator="qa",
        source_dir="/tmp/materials",
        algorithm="lgb",
        run_mode="agent",
        target_col="y",
        score_col="pred",
        split_col="split",
        time_col="apply_month",
        feature_columns=[],
        notebook_path=None,
        sample_path=None,
        pmml_path=None,
        dictionary_path=None,
        report_values_revision=0,
        status=TaskStatus.WRITING_ARTIFACTS,
        status_message="metrics generated",
        created_at="2026-05-31T00:00:00",
        updated_at="2026-05-31T00:00:00",
    )


def test_word_conclusion_llm_error_returns_non_confirmable_empty_values(monkeypatch):
    class FailingClient:
        def complete(self, **_kwargs):
            raise LLMClientError("offline")

    monkeypatch.setattr(
        "riskmodel_checker.agent.service._client",
        lambda _profile: FailingClient(),
    )

    values, metadata = generate_word_conclusions(
        task=_task(),
        evidence={},
        model_profile={"api_base_url": "http://llm", "model_name": "m", "api_key": "k"},
    )

    assert values == {}
    assert agent_conclusions_confirmed(values) is False
    assert metadata["fallback"] is True
    assert metadata["confirmable"] is False
    assert "offline" in metadata["llm_error"]
