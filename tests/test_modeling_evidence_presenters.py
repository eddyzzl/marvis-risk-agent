from __future__ import annotations

from copy import deepcopy

import pytest

from marvis.agent.presenters import modeling_evidence as presenters
from marvis.canonical_results import (
    CanonicalResultAuthenticationError,
    authenticate_canonical_result,
)
from tests.test_model_score_evidence_tool import _run_score, _score_inputs
from tests.test_modeling_training_evidence_tool import (
    _fixture,
    _run as run_training,
)


EXPECTED_TOOLS = {
    "train_model_with_evidence_v2",
    "materialize_model_score_evidence_v2",
}


def _drifted_inputs(tool: str, value: dict) -> dict:
    changed = deepcopy(value)
    if tool == "train_model_with_evidence_v2":
        changed["seed"] = int(changed["seed"]) + 1
    else:
        changed["training_evidence_ref"]["expected_evidence_id"] += "-other"
    return changed


def test_modeling_evidence_registry_authenticates_real_tool_outputs_and_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    training = run_training(fixture)
    score = _run_score(fixture, training)
    outputs = {
        "train_model_with_evidence_v2": training,
        "materialize_model_score_evidence_v2": score,
    }
    inputs = {
        "train_model_with_evidence_v2": fixture["inputs"],
        "materialize_model_score_evidence_v2": _score_inputs(fixture, training),
    }

    assert set(presenters.MODELING_EVIDENCE_PRESENTERS) == EXPECTED_TOOLS
    for tool, output in outputs.items():
        assert authenticate_canonical_result(
            tool,
            output,
            trusted_inputs=inputs[tool],
            task_id=fixture["task"].id,
            workspace=fixture["runtime"].settings.workspace,
        ) is True
        with pytest.raises(CanonicalResultAuthenticationError):
            authenticate_canonical_result(
                tool,
                output,
                trusted_inputs=_drifted_inputs(tool, inputs[tool]),
                task_id=fixture["task"].id,
                workspace=fixture["runtime"].settings.workspace,
            )
        text, tables = presenters.MODELING_EVIDENCE_PRESENTERS[tool](
            output,
            runtime=fixture["runtime"],
            task_id=fixture["task"].id,
            trusted_inputs=inputs[tool],
        )
        assert all(status in text for status in ("未入池", "未采纳", "未部署"))
        assert tables
        rendered = repr(tables)
        assert output["evidence_id"] in rendered
        for artifact in output["artifacts"].values():
            assert artifact["artifact_id"] in rendered
            assert artifact["content_hash"] in rendered

        with pytest.raises(presenters.CanonicalPresenterIntegrityError):
            presenters.MODELING_EVIDENCE_PRESENTERS[tool](
                output,
                runtime=fixture["runtime"],
                task_id=fixture["task"].id,
                trusted_inputs=_drifted_inputs(tool, inputs[tool]),
            )

        forged = deepcopy(output)
        forged["unexpected_presenter_field"] = "must-not-render"
        with pytest.raises(presenters.CanonicalPresenterIntegrityError):
            presenters.MODELING_EVIDENCE_PRESENTERS[tool](
                forged,
                runtime=fixture["runtime"],
                task_id=fixture["task"].id,
                trusted_inputs=inputs[tool],
            )

        partial = deepcopy(output)
        partial.pop("schema_version")
        with pytest.raises(presenters.CanonicalPresenterIntegrityError):
            presenters.MODELING_EVIDENCE_PRESENTERS[tool](
                partial,
                runtime=fixture["runtime"],
                task_id=fixture["task"].id,
                trusted_inputs=inputs[tool],
            )

    validator_names = {
        "train_model_with_evidence_v2": (
            "validate_train_model_with_evidence_v2_tool_output"
        ),
        "materialize_model_score_evidence_v2": (
            "validate_materialize_model_score_evidence_v2_tool_output"
        ),
    }
    for tool, validator_name in validator_names.items():
        with monkeypatch.context() as patch:
            patch.setattr(
                presenters,
                validator_name,
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("injected validator failure")
                ),
            )
            with pytest.raises(
                presenters.CanonicalPresenterIntegrityError,
                match="canonical Tool output",
            ):
                presenters.MODELING_EVIDENCE_PRESENTERS[tool](
                    outputs[tool],
                    runtime=fixture["runtime"],
                    task_id=fixture["task"].id,
                    trusted_inputs=inputs[tool],
                )


@pytest.mark.parametrize("missing", ["runtime", "task_id"])
def test_modeling_evidence_presenters_require_live_trusted_context(missing: str) -> None:
    kwargs = {"runtime": object(), "task_id": "task-1", "trusted_inputs": {}}
    kwargs[missing] = None
    with pytest.raises(presenters.CanonicalPresenterIntegrityError):
        presenters.MODELING_EVIDENCE_PRESENTERS[
            "train_model_with_evidence_v2"
        ]({}, **kwargs)
