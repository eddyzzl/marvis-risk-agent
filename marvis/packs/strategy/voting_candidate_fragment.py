"""Trusted adapter from one verified Voting asset to the generic Pool seam.

The caller must independently verify the task-owned TaskArtifact row, canonical
path, provenance, bytes, parent Pool revision, and all parent candidate
lineages.  This pure adapter only freezes that verified binding into the
generic fragment contract consumed by Strategy Pool.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marvis.packs.strategy.candidate_fragment import (
    CandidateFragmentError,
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
    VOTING_CANDIDATE_ASSET_TYPE,
    validate_voting_candidate_asset,
)


VOTING_CANDIDATE_ARTIFACT_KIND = "strategy_voting_candidate_json"
VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION = (
    "strategy.voting-candidate-artifact.v1"
)
VOTING_CANDIDATE_ORIGIN_TOOL = "strategy.build_voting_candidate"

_BINDING_FIELDS = frozenset(
    {
        "artifact_id",
        "task_id",
        "kind",
        "content_hash",
        "origin_tool",
        "artifact_schema_version",
        "asset_id",
        "asset_hash",
    }
)


class VotingCandidateFragmentError(StrategyError):
    """A verified Voting asset could not cross the generic fragment seam."""


def voting_candidate_to_verified_fragment(
    asset_payload: Mapping[str, Any],
    *,
    artifact_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one independently verified Voting asset without adding action."""

    try:
        asset = validate_voting_candidate_asset(asset_payload)
    except StrategyError as exc:
        raise VotingCandidateFragmentError(
            "Voting candidate asset failed strict validation"
        ) from exc
    binding = _binding(artifact_binding, asset=asset)
    lifecycle = asset["lifecycle"]
    fragment = asset["fragment"]
    evidence = asset["candidate_evidence"]
    try:
        return build_verified_candidate_fragment(
            artifact={
                "artifact_id": binding["artifact_id"],
                "artifact_kind": VOTING_CANDIDATE_ARTIFACT_KIND,
                "artifact_schema_version": VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
                "artifact_content_hash": binding["content_hash"],
                "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
            },
            asset={
                "schema_version": VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
                "asset_id": asset["asset_id"],
                "asset_hash": asset["asset_hash"],
                "asset_type": VOTING_CANDIDATE_ASSET_TYPE,
            },
            fragment_type=fragment["fragment_type"],
            fragment_id=fragment["fragment_id"],
            rule_id=fragment["rule_id"],
            condition=fragment["condition"],
            requirements=fragment["requirements"],
            effect_id=fragment["effect_id"],
            evidence_id=evidence["candidate_id"],
            evidence_hash=evidence["evidence_hash"],
            evidence_identity=asset["evidence_identity"],
            candidate_stage=lifecycle["candidate_stage"],
            observation_stage=lifecycle["observation_stage"],
            validation_status=lifecycle["validation_status"],
        )
    except CandidateFragmentError as exc:
        raise VotingCandidateFragmentError(
            "Voting candidate failed generic fragment projection"
        ) from exc


def _binding(
    value: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise VotingCandidateFragmentError("artifact_binding must be an object")
    missing = sorted(_BINDING_FIELDS - set(value))
    unexpected = sorted(set(value) - _BINDING_FIELDS)
    if missing or unexpected:
        raise VotingCandidateFragmentError(
            "artifact_binding fields are invalid"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + (f" (unsupported: {', '.join(unexpected)})" if unexpected else "")
        )
    normalized = {key: _text(value[key], key) for key in _BINDING_FIELDS}
    expected = {
        "task_id": asset["pool_ref"]["task_id"],
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
        "artifact_schema_version": VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
    }
    for field, expected_value in expected.items():
        if normalized[field] != expected_value:
            raise VotingCandidateFragmentError(
                f"artifact_binding {field} does not match Voting asset"
            )
    return normalized


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
    ):
        raise VotingCandidateFragmentError(
            f"artifact_binding {field} must be canonical text"
        )
    return value


__all__ = [
    "VOTING_CANDIDATE_ARTIFACT_KIND",
    "VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "VOTING_CANDIDATE_ORIGIN_TOOL",
    "VotingCandidateFragmentError",
    "voting_candidate_to_verified_fragment",
]
