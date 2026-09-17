from __future__ import annotations

from dataclasses import replace

from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.validation.input_contracts import FieldCandidate, FieldEvidence
from marvis.validation.suggested_confirmation import (
    confirm_unambiguous_contract,
    suggested_confirmation_from_contract,
)
from tests.validation_builders import make_candidate_contract


def _evidence() -> FieldEvidence:
    return FieldEvidence("rmc_literal", 0, "RMC", 1.0)


def _unique_contract(**overrides: object):
    contract = make_candidate_contract()
    candidates = {
        **contract.candidates,
        "split_col": (FieldCandidate("split", (_evidence(),)),),
        "time_col": (FieldCandidate("apply_month", (_evidence(),)),),
        "pmml_output_field": (FieldCandidate("probability_1", (_evidence(),)),),
        "model_params": (FieldCandidate({}, (_evidence(),)),),
        **overrides,
    }
    return replace(contract, candidates=candidates)


def test_suggested_confirmation_uses_unique_candidates():
    confirmation = suggested_confirmation_from_contract(_unique_contract())
    assert confirmation is not None
    assert confirmation.target_col == "y"
    assert confirmation.split_col == "split"
    assert confirmation.time_col == "apply_month"
    assert confirmation.time_granularity == "month"
    assert confirmation.pmml_output_field == "probability_1"
    assert confirmation.feature_col == "feature"
    assert confirmation.importance_col == "importance"


def test_suggested_confirmation_none_when_field_conflicts():
    contract = _unique_contract(
        target_col=(
            FieldCandidate("y", (_evidence(),)),
            FieldCandidate("label", (_evidence(),)),
        ),
    )
    assert suggested_confirmation_from_contract(contract) is None


def _task_with_pending_contract(tmp_path, contract):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = TaskRepository(db_path).create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(tmp_path),
        )
    )
    ValidationContractRepository(db_path).replace_candidates(task.id, contract)
    return db_path, task.id


def test_confirm_unambiguous_contract_false_when_conflicts(tmp_path):
    db_path, task_id = _task_with_pending_contract(tmp_path, make_candidate_contract())
    assert confirm_unambiguous_contract(db_path=db_path, task_id=task_id) is False


def test_confirm_unambiguous_contract_true_when_unique(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from marvis.validation.input_confirmation import ValidatedConfirmation

    db_path, task_id = _task_with_pending_contract(tmp_path, _unique_contract())
    confirmed: list[str] = []
    monkeypatch.setattr(
        "marvis.validation.suggested_confirmation.resolve_selected_validation_materials",
        lambda _child: SimpleNamespace(sample=Path("s"), dictionary=Path("d")),
    )
    monkeypatch.setattr(
        "marvis.validation.suggested_confirmation.validate_confirmation_against_materials",
        lambda **kwargs: ValidatedConfirmation(
            values=kwargs["requested"],
            sample_schema=kwargs["contract"].sample_schema,
            feature_metadata=kwargs["contract"].feature_metadata,
        ),
    )

    def fake_confirm(self, task_id, values, **kwargs):
        confirmed.append(task_id)
        return SimpleNamespace(status="ready")

    monkeypatch.setattr(ValidationContractRepository, "confirm", fake_confirm)
    assert confirm_unambiguous_contract(db_path=db_path, task_id=task_id) is True
    assert confirmed == [task_id]
