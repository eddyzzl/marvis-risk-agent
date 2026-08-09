from __future__ import annotations

from pathlib import Path

import pytest

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_evidence_verifier import validate_pool_ref


def _valid_ref() -> dict[str, object]:
    return {
        "artifact_id": "a" * 64,
        "expected_artifact_content_hash": "b" * 64,
        "expected_pool_id": "strategy-pool-" + "c" * 32,
        "expected_revision": 3,
        "expected_revision_id": "strategy-pool-revision-" + "d" * 32,
        "expected_snapshot_hash": "e" * 64,
    }


def test_pool_ref_verifier_returns_a_detached_canonical_binding() -> None:
    source = _valid_ref()

    verified = validate_pool_ref(source)

    assert verified == source
    assert verified is not source
    source["expected_revision"] = 99
    assert verified["expected_revision"] == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra="forbidden"),
        lambda value: value.pop("expected_snapshot_hash"),
        lambda value: value.update(expected_pool_id="pool-guess"),
        lambda value: value.update(expected_revision_id="revision-guess"),
        lambda value: value.update(expected_revision=True),
        lambda value: value.update(expected_artifact_content_hash="A" * 64),
        lambda value: value.update(artifact_id=Path("not-json")),
    ],
)
def test_pool_ref_verifier_fails_closed_for_invalid_identity(mutate) -> None:
    value = _valid_ref()
    mutate(value)

    with pytest.raises(StrategyError):
        validate_pool_ref(value)


def test_pool_tools_delegate_to_the_single_pool_ref_verifier() -> None:
    from marvis.packs.strategy import impact_cube_tools, pool_validation_tools

    assert "_validate_pool_ref" not in impact_cube_tools.__dict__
    assert "_validate_pool_ref" not in pool_validation_tools.__dict__
    assert impact_cube_tools.validate_pool_ref is validate_pool_ref
    assert pool_validation_tools.validate_pool_ref is validate_pool_ref
