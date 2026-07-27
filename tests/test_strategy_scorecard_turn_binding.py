"""Agent-side exact evidence binding for governed Scorecard workflows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _candidate_selection_artifact_slots,
    _scorecard_band_build_plan_slots,
    _scorecard_cutoff_selection_plan_slots,
)
import marvis.agent.turn_handlers as turn_handlers
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.score_evidence import (
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
)
from marvis.packs.modeling.score_evidence_tools import (
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
)
from marvis.packs.strategy.scorecard_candidate import (
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
)


ASSET_ID = "scorecard-band-asset-" + "a" * 32
ASSET_HASH = "b" * 64
CUTOFF_ID = "scorecard-cutoff-" + "c" * 32
SELECTION_ID = "scorecard-cutoff-selection-" + "d" * 32
SELECTION_HASH = "e" * 64
SAMPLE_REF = {
    "membership_artifact_id": "1" * 64,
    "expected_membership_artifact_content_hash": "2" * 64,
    "bundle_artifact_id": "3" * 64,
    "expected_bundle_artifact_content_hash": "4" * 64,
    "expected_bundle_id": "sample-bundle-" + "5" * 32,
    "expected_sample_design_id": "sample-design-" + "6" * 32,
    "expected_sample_design_content_hash": "7" * 64,
}


class _Artifacts:
    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.list_calls = 0

    def list_for_task(self, _task_id: str) -> list[dict]:
        self.list_calls += 1
        return [dict(record) for record in self.records]


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(db_path="/tmp/unused", tasks_dir="/tmp/tasks")
    )


def _task() -> SimpleNamespace:
    return SimpleNamespace(id="task-scorecard", task_type="strategy")


def _score_record(
    *,
    evidence_id: str,
    evidence_hash: str,
    vector_id: str,
    vector_hash: str,
) -> dict:
    return {
        "id": evidence_id,
        "kind": MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        "origin_tool": MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
        "content_hash": evidence_hash,
        "created_at": evidence_id,
        "provenance": {
            "score_vector_artifact_id": vector_id,
            "score_vector_artifact_content_hash": vector_hash,
        },
    }


def _score_binding(algorithm: str) -> SimpleNamespace:
    metadata = (
        {
            "score_product": "raw_native_uncalibrated_bad_probability",
            "score_direction": "higher_is_riskier",
            "points_direction": "higher_is_better",
            "calibration_status": "not_applied",
            "scorecard_table": [{"feature": "age"}],
        }
        if algorithm == "scorecard"
        else {}
    )
    return SimpleNamespace(
        training=SimpleNamespace(
            experiment=SimpleNamespace(recipe_id=algorithm),
            model_artifact=SimpleNamespace(algorithm=algorithm),
            evidence={
                "experiment": {"recipe_id": algorithm},
                "model_artifact": {
                    "algorithm": algorithm,
                    "scoring_metadata": metadata,
                },
            },
        ),
        envelope={
            "score_product": "raw_native_uncalibrated_bad_probability",
            "scoring_contract": {"score_direction": "higher_is_riskier"},
        },
    )


def test_scorecard_build_binds_only_latest_score_and_sample_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _score_record(
        evidence_id="8" * 64,
        evidence_hash="9" * 64,
        vector_id="a" * 64,
        vector_hash="b" * 64,
    )
    newest = _score_record(
        evidence_id="c" * 64,
        evidence_hash="d" * 64,
        vector_id="e" * 64,
        vector_hash="f" * 64,
    )
    artifacts = _Artifacts([old, newest])
    read_runtime = SimpleNamespace(task_artifacts=artifacts)
    load_calls: list[dict] = []
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: read_runtime,
    )
    monkeypatch.setattr(
        turn_handlers,
        "_latest_verified_strategy_sample_design_v2_binding",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_sample_ref",
        lambda _sample: dict(SAMPLE_REF),
    )

    def load_score(_runtime, **kwargs):
        load_calls.append(dict(kwargs))
        return _score_binding("scorecard")

    monkeypatch.setattr(
        turn_handlers,
        "load_model_score_evidence_artifacts",
        load_score,
    )
    monkeypatch.setattr(
        turn_handlers,
        "build_training_evidence_ref",
        lambda _training: {"sample_design_ref": dict(SAMPLE_REF)},
    )
    draft = StandardWorkflowRequestDraft(
        workflow="scorecard_band_build",
        workflow_inputs={"bin_count": 7},
    )

    slots = _scorecard_band_build_plan_slots(_runtime(), _task(), draft)

    assert load_calls == [
        {
            "task_id": "task-scorecard",
            "evidence_artifact_id": newest["id"],
            "expected_evidence_artifact_content_hash": newest["content_hash"],
            "score_vector_artifact_id": (
                newest["provenance"]["score_vector_artifact_id"]
            ),
            "expected_score_vector_artifact_content_hash": (
                newest["provenance"][
                    "score_vector_artifact_content_hash"
                ]
            ),
        }
    ]
    assert slots == {
        "score_evidence_ref": {
            "evidence_artifact_id": newest["id"],
            "expected_evidence_artifact_content_hash": newest["content_hash"],
            "score_vector_artifact_id": (
                newest["provenance"]["score_vector_artifact_id"]
            ),
            "expected_score_vector_artifact_content_hash": (
                newest["provenance"][
                    "score_vector_artifact_content_hash"
                ]
            ),
        },
        "sample_design_ref": SAMPLE_REF,
        "banding": {"method": "equal_frequency", "bin_count": 7},
    }
    assert artifacts.list_calls == 2


def test_scorecard_build_skips_authenticated_newer_non_scorecard_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorecard = _score_record(
        evidence_id="8" * 64,
        evidence_hash="9" * 64,
        vector_id="a" * 64,
        vector_hash="b" * 64,
    )
    newer_lgb = _score_record(
        evidence_id="c" * 64,
        evidence_hash="d" * 64,
        vector_id="e" * 64,
        vector_hash="f" * 64,
    )
    artifacts = _Artifacts([scorecard, newer_lgb])
    read_runtime = SimpleNamespace(task_artifacts=artifacts)
    load_calls: list[str] = []
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: read_runtime,
    )
    monkeypatch.setattr(
        turn_handlers,
        "_latest_verified_strategy_sample_design_v2_binding",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_sample_ref",
        lambda _sample: dict(SAMPLE_REF),
    )

    def load_score(_runtime, **kwargs):
        evidence_id = kwargs["evidence_artifact_id"]
        load_calls.append(evidence_id)
        return _score_binding(
            "lightgbm" if evidence_id == newer_lgb["id"] else "scorecard"
        )

    monkeypatch.setattr(
        turn_handlers,
        "load_model_score_evidence_artifacts",
        load_score,
    )
    monkeypatch.setattr(
        turn_handlers,
        "build_training_evidence_ref",
        lambda _training: {"sample_design_ref": dict(SAMPLE_REF)},
    )

    slots = _scorecard_band_build_plan_slots(
        _runtime(),
        _task(),
        StandardWorkflowRequestDraft(
            workflow="scorecard_band_build",
            workflow_inputs={},
        ),
    )

    assert load_calls == [newer_lgb["id"], scorecard["id"]]
    assert slots["score_evidence_ref"]["evidence_artifact_id"] == scorecard["id"]


def test_scorecard_build_does_not_fallback_when_newest_score_is_damaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _score_record(
        evidence_id="8" * 64,
        evidence_hash="9" * 64,
        vector_id="a" * 64,
        vector_hash="b" * 64,
    )
    newest = _score_record(
        evidence_id="c" * 64,
        evidence_hash="d" * 64,
        vector_id="e" * 64,
        vector_hash="f" * 64,
    )
    read_runtime = SimpleNamespace(task_artifacts=_Artifacts([old, newest]))
    load_calls: list[str] = []
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: read_runtime,
    )
    monkeypatch.setattr(
        turn_handlers,
        "_latest_verified_strategy_sample_design_v2_binding",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_sample_ref",
        lambda _sample: dict(SAMPLE_REF),
    )

    def damaged(_runtime, **kwargs):
        load_calls.append(kwargs["evidence_artifact_id"])
        raise ModelingError("damaged newest score evidence")

    monkeypatch.setattr(
        turn_handlers,
        "load_model_score_evidence_artifacts",
        damaged,
    )

    with pytest.raises(
        StrategySetupError,
        match="最新.*评分证据|最新.*分数证据",
    ):
        _scorecard_band_build_plan_slots(
            _runtime(),
            _task(),
            StandardWorkflowRequestDraft(
                workflow="scorecard_band_build",
                workflow_inputs={},
            ),
        )

    assert load_calls == [newest["id"]]


def test_scorecard_cutoff_selection_binds_exact_full_band_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {
        "id": "8" * 64,
        "kind": SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        "origin_tool": SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        "content_hash": "9" * 64,
        "created_at": "2026-07-25T00:00:00Z",
        "provenance": {
            "schema_version": SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
            "asset_id": ASSET_ID,
            "asset_hash": ASSET_HASH,
        },
    }
    artifacts = _Artifacts([record])
    read_runtime = SimpleNamespace(task_artifacts=artifacts)
    loader_calls: list[dict] = []
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: read_runtime,
    )

    def load_band(_runtime, **kwargs):
        loader_calls.append(dict(kwargs))
        return SimpleNamespace(
            artifact_id=record["id"],
            content_hash=record["content_hash"],
            asset={
                "asset_id": ASSET_ID,
                "asset_hash": ASSET_HASH,
                "cutoffs": [{"cutoff_id": CUTOFF_ID}],
            },
        )

    monkeypatch.setattr(
        turn_handlers,
        "load_scorecard_band_asset_artifact",
        load_band,
    )
    draft = StandardWorkflowRequestDraft(
        workflow="scorecard_cutoff_selection",
        workflow_inputs={
            "asset_id": ASSET_ID,
            "cutoff_id": CUTOFF_ID,
            "reason": "人工确认进入后续影响评审",
        },
    )

    slots = _scorecard_cutoff_selection_plan_slots(
        _runtime(),
        task_id="task-scorecard",
        draft=draft,
    )

    assert loader_calls == [
        {
            "task_id": "task-scorecard",
            "artifact_id": record["id"],
            "expected_artifact_content_hash": record["content_hash"],
            "expected_asset_id": ASSET_ID,
            "expected_asset_hash": ASSET_HASH,
        }
    ]
    assert slots == {
        "source_artifact_id": record["id"],
        "expected_source_artifact_content_hash": record["content_hash"],
        "expected_asset_id": ASSET_ID,
        "expected_asset_hash": ASSET_HASH,
        "cutoff_id": CUTOFF_ID,
        "reason": "人工确认进入后续影响评审",
    }
    assert artifacts.list_calls == 2


def test_pool_selection_strictly_replays_scorecard_pointer_to_full_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {
        "id": "8" * 64,
        "kind": SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        "origin_tool": SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        "content_hash": "9" * 64,
        "created_at": "2026-07-25T00:00:00Z",
        "provenance": {
            "schema_version": (
                SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
            ),
            "selection_id": SELECTION_ID,
            "selection_hash": SELECTION_HASH,
        },
    }
    artifacts = _Artifacts([record])
    read_runtime = SimpleNamespace(task_artifacts=artifacts)
    source = SimpleNamespace(
        asset={"asset_id": ASSET_ID, "asset_hash": ASSET_HASH},
        to_domain_binding=lambda: {"source": "verified"},
    )
    verified = SimpleNamespace(
        artifact_id=record["id"],
        content_hash=record["content_hash"],
        selection={"selection_id": SELECTION_ID, "cutoff_id": CUTOFF_ID},
        source_asset_binding=source,
        to_domain_binding=lambda: {"selection": "verified"},
    )
    load_calls: list[dict] = []
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: read_runtime,
    )

    def load_selection(_runtime, **kwargs):
        load_calls.append(dict(kwargs))
        return verified

    monkeypatch.setattr(
        turn_handlers,
        "load_scorecard_cutoff_selection_artifact",
        load_selection,
    )
    monkeypatch.setattr(
        turn_handlers,
        "scorecard_cutoff_selection_to_verified_candidate_fragment",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        turn_handlers,
        "verified_fragment_pool_parts",
        lambda _fragment: (
            {
                "asset_id": ASSET_ID,
                "asset_hash": ASSET_HASH,
                "fragment_id": "scorecard-fragment-" + "f" * 32,
            },
            "scorecard-rule-" + "a" * 32,
            {"condition": {}},
        ),
    )

    slots, fragment_id = _candidate_selection_artifact_slots(
        _runtime(),
        task_id="task-scorecard",
        selection_id=SELECTION_ID,
    )

    assert load_calls == [
        {
            "task_id": "task-scorecard",
            "artifact_id": record["id"],
            "expected_artifact_content_hash": record["content_hash"],
            "expected_selection_id": SELECTION_ID,
            "expected_selection_hash": SELECTION_HASH,
        }
    ]
    assert slots == {
        "source_artifact_id": record["id"],
        "expected_artifact_content_hash": record["content_hash"],
        "expected_asset_id": ASSET_ID,
        "expected_asset_hash": ASSET_HASH,
    }
    assert fragment_id == "scorecard-fragment-" + "f" * 32
    assert artifacts.list_calls == 2


def test_llm_free_scorecard_request_enters_the_same_slot_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[dict] = []

    class _Repo:
        def list_agent_messages(self, _task_id: str) -> list[dict]:
            return list(messages)

        def add_agent_message(self, _task_id: str, **message) -> dict:
            stored = {"id": f"message-{len(messages) + 1}", **message}
            messages.append(stored)
            return stored

    runtime = SimpleNamespace(
        settings=SimpleNamespace(db_path="/tmp/unused"),
        plan_repo=SimpleNamespace(list_plans_for_task=lambda _task_id: []),
    )
    resolver_calls: list[dict] = []
    plan_calls: list[dict] = []
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_dataset_preview",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StrategySetupError("no generic dataset preview")
        ),
    )

    def resolve(_runtime, task, draft):
        resolver_calls.append(
            {"task_id": task.id, "draft": draft.to_dict()}
        )
        return {
            "score_evidence_ref": {"server": "score"},
            "sample_design_ref": {"server": "sample"},
        }

    monkeypatch.setattr(
        turn_handlers,
        "_scorecard_band_build_plan_slots",
        resolve,
    )

    def start(_runtime, _repo, _task, **kwargs):
        plan_calls.append(dict(kwargs))
        return {"status": "started", **kwargs}

    monkeypatch.setattr(
        turn_handlers,
        "_start_confirmed_strategy_plan",
        start,
    )

    response = turn_handlers._handle_structured_strategy_request_turn(
        runtime,
        _Repo(),
        _task(),
        user_text="Candidate Lab 手工构建评分卡分档",
        strategy_request={
            "request_kind": "standard_workflow",
            "workflow": "scorecard_band_build",
            "workflow_inputs": {},
        },
    )

    assert {
        "scorecard_band_build",
        "scorecard_cutoff_selection",
    }.issubset(turn_handlers._MANUAL_STRATEGY_WORKFLOWS)
    assert resolver_calls == [
        {
            "task_id": "task-scorecard",
            "draft": {
                "request_kind": "standard_workflow",
                "workflow": "scorecard_band_build",
                "workflow_inputs": {},
            },
        }
    ]
    assert plan_calls == [
        {
            "template_id": "strategy_scorecard_band_build",
            "slots": {
                "score_evidence_ref": {"server": "score"},
                "sample_design_ref": {"server": "sample"},
            },
            "auto_start": True,
        }
    ]
    assert response["status"] == "started"
