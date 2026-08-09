"""Characterize the six legacy routes before canonical spec migration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent import turn_handlers


def _draft(workflow: str, workflow_inputs: dict[str, object]):
    return StandardWorkflowRequestDraft(
        workflow=workflow,
        workflow_inputs=workflow_inputs,
    )


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        (
            _draft("univariate_candidate_analysis", {}),
            (True, True, True),
        ),
        (
            _draft(
                "univariate_candidate_refinement",
                {
                    "feature": "score",
                    "method": "equal_width",
                    "selection": {
                        "risk_threshold": {"operator": ">=", "value": 0.2}
                    },
                },
            ),
            (True, True, True),
        ),
        (
            _draft(
                "univariate_candidate_refinement",
                {
                    "source_candidate_id": "candidate-" + "a" * 32,
                    "feature": "score",
                    "method": "equal_width",
                    "selection": {"source_bin_ids": ["regular:1"]},
                },
            ),
            (False, False, False),
        ),
        (
            _draft("candidate_monthly_stability", {}),
            (False, False, False),
        ),
        (
            _draft("scorecard_model_score_evidence_build", {}),
            (False, False, False),
        ),
        (
            _draft("scorecard_band_build", {}),
            (False, False, False),
        ),
        (
            _draft("scorecard_cutoff_selection", {}),
            (False, False, False),
        ),
    ],
)
def test_legacy_requirement_matrix_is_characterized(draft, expected) -> None:
    assert (
        turn_handlers._strategy_request_requires_dataset(draft),
        turn_handlers._strategy_request_requires_target(draft),
        turn_handlers._strategy_request_requires_complete_labels(draft),
    ) == expected


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs", "template_id"),
    [
        (
            "univariate_candidate_analysis",
            {
                "features": ["score"],
                "methods": ["equal_width"],
                "bin_count": 5,
                "min_bin_pct": 0.02,
                "sentinel_values": [-999],
            },
            "strategy_univariate_candidate_analysis",
        ),
        (
            "univariate_candidate_refinement",
            {
                "features": ["score"],
                "methods": ["equal_width"],
                "bin_count": 5,
                "min_bin_pct": 0.02,
                "sentinel_values": [],
                "feature": "score",
                "method": "equal_width",
                "merge_groups": [],
                "selection": {
                    "risk_threshold": {"operator": ">=", "value": 0.2}
                },
            },
            "strategy_univariate_candidate_refinement",
        ),
    ],
)
def test_legacy_fresh_univariate_plan_shape_is_characterized(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    workflow_inputs: dict[str, object],
    template_id: str,
) -> None:
    started: dict[str, object] = {}
    sample_ref = {"artifact_id": "sample-artifact", "content_hash": "a" * 64}
    monkeypatch.setattr(
        turn_handlers,
        "_latest_matching_strategy_sample_design_ref",
        lambda *_args, **_kwargs: dict(sample_ref),
    )
    monkeypatch.setattr(
        turn_handlers,
        "_start_confirmed_strategy_plan",
        lambda _runtime, _repo, _task, **kwargs: started.update(kwargs) or started,
    )
    context = SimpleNamespace(
        dataset_id="dataset-1",
        dataset_content_hash="b" * 64,
        workspace_revision=7,
        analysis_generation=3,
        semantic_mapping_hash="c" * 64,
        target_col="bad",
    )

    turn_handlers._run_validated_strategy_request(
        SimpleNamespace(),
        object(),
        SimpleNamespace(id="task-1"),
        _draft(workflow, workflow_inputs),
        context=context,
        auto_start=True,
        drop_nan_labels=True,
    )

    assert started == {
        "template_id": template_id,
        "slots": {
            "dataset_id": "dataset-1",
            **workflow_inputs,
            "expected_content_hash": "b" * 64,
            "workspace_revision": 7,
            "analysis_generation": 3,
            "semantic_mapping_hash": "c" * 64,
            "target_col": "bad",
            "sample_design_ref": sample_ref,
            "drop_nan_labels": True,
        },
        "auto_start": True,
    }


def test_legacy_existing_refinement_plan_shape_is_characterized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: dict[str, object] = {}
    source_id = "candidate-" + "a" * 32
    source_ref = {
        "source_artifact_id": "artifact-1",
        "expected_artifact_content_hash": "b" * 64,
        "expected_candidate_id": source_id,
        "expected_evidence_hash": "c" * 64,
    }
    inputs = {
        "source_candidate_id": source_id,
        "feature": "score",
        "method": "equal_width",
        "merge_groups": [["regular:1", "regular:2"]],
        "selection": {"source_bin_ids": ["regular:1"]},
        "selection_reason": "人工确认",
    }
    monkeypatch.setattr(
        turn_handlers,
        "_bind_candidate_source_artifact_evidence",
        lambda *_args, **_kwargs: dict(source_ref),
    )
    monkeypatch.setattr(
        turn_handlers,
        "_start_confirmed_strategy_plan",
        lambda _runtime, _repo, _task, **kwargs: started.update(kwargs) or started,
    )

    turn_handlers._run_validated_strategy_request(
        SimpleNamespace(),
        object(),
        SimpleNamespace(id="task-1"),
        _draft("univariate_candidate_refinement", inputs),
        context=None,
        auto_start=False,
        drop_nan_labels=False,
    )

    assert started == {
        "template_id": "strategy_univariate_candidate_refinement_existing",
        "slots": {
            "feature": "score",
            "method": "equal_width",
            "merge_groups": [["regular:1", "regular:2"]],
            "selection": {"source_bin_ids": ["regular:1"]},
            "selection_reason": "人工确认",
            **source_ref,
        },
        "auto_start": False,
    }


@pytest.mark.parametrize(
    (
        "workflow",
        "workflow_inputs",
        "helper_name",
        "evidence_slots",
        "bound_slots",
        "template_id",
    ),
    [
        (
            "candidate_monthly_stability",
            {"asset_id": "candidate-asset-" + "a" * 32},
            "_bind_candidate_monthly_stability_evidence",
            {
                "expected_asset_id": "candidate-asset-" + "a" * 32,
                "dataset_id": "dataset-1",
                "month_col": "month",
            },
            {
                "source_kind": "univariate_asset",
                "expected_asset_id": "candidate-asset-" + "a" * 32,
                "dataset_id": "dataset-1",
                "month_col": "month",
            },
            "strategy_candidate_monthly_stability",
        ),
        (
            "scorecard_model_score_evidence_build",
            {
                "features": ["score"],
                "seed": 42,
                "max_iter": 100,
                "scorecard_max_bins": 8,
            },
            "_bind_scorecard_model_score_evidence",
            {"sample_design_ref": {"bundle_artifact_id": "bundle-1"}},
            {
                "sample_design_ref": {"bundle_artifact_id": "bundle-1"},
                "features": ["score"],
                "params": {"max_iter": 100, "scorecard_max_bins": 8},
                "seed": 42,
            },
            "strategy_scorecard_model_score_evidence_build",
        ),
        (
            "scorecard_band_build",
            {"bin_count": 7},
            "_bind_scorecard_band_evidence",
            {
                "score_evidence_ref": {"evidence_artifact_id": "evidence-1"},
                "sample_design_ref": {"bundle_artifact_id": "bundle-1"},
            },
            {
                "score_evidence_ref": {"evidence_artifact_id": "evidence-1"},
                "sample_design_ref": {"bundle_artifact_id": "bundle-1"},
                "banding": {"method": "equal_frequency", "bin_count": 7},
            },
            "strategy_scorecard_band_build",
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "reason": "人工确认",
            },
            "_bind_scorecard_cutoff_evidence",
            {
                "source_artifact_id": "artifact-1",
                "expected_source_artifact_content_hash": "c" * 64,
                "expected_asset_id": "scorecard-band-asset-" + "a" * 32,
                "expected_asset_hash": "d" * 64,
            },
            {
                "source_artifact_id": "artifact-1",
                "expected_source_artifact_content_hash": "c" * 64,
                "expected_asset_id": "scorecard-band-asset-" + "a" * 32,
                "expected_asset_hash": "d" * 64,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "reason": "人工确认",
            },
            "strategy_scorecard_cutoff_selection",
        ),
    ],
)
def test_legacy_evidence_bound_plan_shapes_are_characterized(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    workflow_inputs: dict[str, object],
    helper_name: str,
    evidence_slots: dict[str, object],
    bound_slots: dict[str, object],
    template_id: str,
) -> None:
    started: dict[str, object] = {}
    monkeypatch.setattr(
        turn_handlers,
        helper_name,
        lambda *_args, **_kwargs: dict(evidence_slots),
    )
    monkeypatch.setattr(
        turn_handlers,
        "_start_confirmed_strategy_plan",
        lambda _runtime, _repo, _task, **kwargs: started.update(kwargs) or started,
    )

    turn_handlers._run_validated_strategy_request(
        SimpleNamespace(),
        object(),
        SimpleNamespace(id="task-1"),
        _draft(workflow, workflow_inputs),
        context=None,
        auto_start=True,
        drop_nan_labels=False,
    )

    assert started == {
        "template_id": template_id,
        "slots": bound_slots,
        "auto_start": True,
    }
