from __future__ import annotations

import pytest
from pydantic import ValidationError

from marvis.api_schemas import ManualStrategyRequest


def _request(inputs: dict[str, object]) -> dict[str, object]:
    return {
        "request_kind": "standard_workflow",
        "workflow": "univariate_candidate_refinement",
        "workflow_inputs": inputs,
    }


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "feature": "score",
            "method": "equal_width",
            "selection": {
                "risk_threshold": {"operator": ">=", "value": 0.2}
            },
            "bin_count": None,
        },
        {
            "feature": "score",
            "method": "equal_width",
            "source_candidate_id": "candidate-" + "a" * 32,
            "selection": {"source_bin_ids": ["bin-1"]},
            "merge_groups": None,
        },
    ],
)
def test_refinement_schema_rejects_explicit_null_optional_fields(
    inputs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="must be omitted instead of null"):
        ManualStrategyRequest.model_validate(_request(inputs))


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "feature": "score",
            "method": "manual",
            "selection": {
                "risk_threshold": {"operator": ">=", "value": 0.2}
            },
            "manual_breakpoints": {"score": [500, 400]},
        },
        {
            "feature": "score",
            "method": "equal_width",
            "source_candidate_id": "candidate-" + "a" * 32,
            "selection": {"source_bin_ids": ["bin-1", "bin-1"]},
        },
        {
            "feature": "score",
            "method": "equal_width",
            "source_candidate_id": "candidate-" + "a" * 32,
            "selection": {"source_bin_ids": ["bin-1"]},
            "merge_groups": [["bin-1", "bin-2"], ["bin-2", "bin-3"]],
        },
    ],
)
def test_refinement_schema_rejects_contracts_the_compiler_would_reject(
    inputs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(_request(inputs))


def test_refinement_schema_accepts_canonical_existing_controls() -> None:
    request = ManualStrategyRequest.model_validate(
        _request(
            {
                "feature": "score",
                "method": "equal_width",
                "source_candidate_id": "candidate-" + "a" * 32,
                "selection": {"source_bin_ids": ["bin-1", "bin-2"]},
                "merge_groups": [["bin-3", "bin-4"]],
                "selection_reason": "人工复核后保留高风险箱",
            }
        )
    )

    assert request.workflow_inputs["selection"]["source_bin_ids"] == [
        "bin-1",
        "bin-2",
    ]
