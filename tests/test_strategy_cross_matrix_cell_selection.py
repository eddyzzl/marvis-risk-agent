from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from marvis.feature.iv import _smoothed_woe_iv
from marvis.packs.strategy.candidate_fragment import (
    validate_verified_candidate_fragment,
)
from marvis.packs.strategy.cross_matrix_candidate import (
    canonical_cross_matrix_candidate_asset_json,
)
from marvis.packs.strategy.cross_matrix_cell_selection import (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION,
    CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
    CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
    CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
    CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION,
    CrossMatrixCellSelectionError,
    build_cross_matrix_cell_selection,
    canonical_cross_matrix_cell_selection_json,
    cross_matrix_cell_selection_content_hash,
    cross_matrix_cell_selection_to_verified_candidate_fragment,
    derive_cross_matrix_cell_group_facts,
    validate_cross_matrix_cell_selection,
)
from tests.test_strategy_cross_matrix_candidate import _build


HASH_C = "c" * 64
HASH_D = "d" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _binding(asset: dict) -> dict:
    canonical_bytes = canonical_cross_matrix_candidate_asset_json(asset).encode("utf-8")
    content_hash = hashlib.sha256(canonical_bytes).hexdigest()
    sample = asset["sample_identity"]
    provenance = {
        "schema_version": CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION,
        "producer_version": asset["producer_version"],
        "asset_schema_version": asset["schema_version"],
        "asset_type": asset["asset_type"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "parent_candidate_id": asset["parent"]["candidate_id"],
        "parent_evidence_hash": asset["parent"]["evidence_hash"],
        "candidate_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "source_artifact_id": "artifact-parent-evidence",
        "source_artifact_content_hash": HASH_C,
        "task_id": sample["task_id"],
        "dataset_id": sample["dataset_id"],
        "dataset_content_hash": sample["dataset_content_hash"],
        "registry_metadata_hash": HASH_D,
        "workspace_revision": sample["workspace_revision"],
        "workspace_generation": sample["workspace_generation"],
        "semantic_mapping_hash": sample["semantic_mapping_hash"],
        "sample_context_hash": sample["sample_context_hash"],
        "target_col": sample["target_col"],
        "labeled_row_count": sample["row_count"],
        "row_axis": {
            "feature": asset["axes"][0]["feature"],
            "method": asset["axes"][0]["method"],
        },
        "column_axis": {
            "feature": asset["axes"][1]["feature"],
            "method": asset["axes"][1]["method"],
        },
        "cell_count": asset["matrix"]["cell_count"],
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "budget": asset["budget"]["limit"],
        "truncated": False,
    }
    return {
        "artifact_id": "artifact-cross-matrix",
        "task_id": sample["task_id"],
        "kind": CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
        "artifact_schema_version": CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION,
        "content_hash": content_hash,
        "origin_tool": CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
        "path": (
            f"/tasks/{sample['task_id']}/strategy_cross_matrix_candidates/"
            f"{asset['asset_id']}_{content_hash[:12]}.json"
        ),
        "provenance_hash": hashlib.sha256(
            _canonical_json(provenance).encode("utf-8")
        ).hexdigest(),
        "provenance": provenance,
        "canonical_bytes": canonical_bytes,
    }


def _selection(
    asset: dict,
    cell_ids: list[str],
    *,
    reason: str | None = None,
) -> dict:
    return build_cross_matrix_cell_selection(
        asset,
        source_artifact_binding=_binding(asset),
        cell_ids=cell_ids,
        selection_reason=reason,
    )


def _selection_binding(
    selection: dict, artifact_id: str = "artifact-selection"
) -> dict:
    source = selection["source_artifact"]
    asset = selection["source_asset"]
    candidate = selection["source_candidate"]
    return {
        "artifact_id": artifact_id,
        "task_id": source["task_id"],
        "kind": CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
        "content_hash": cross_matrix_cell_selection_content_hash(selection),
        "origin_tool": CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
        "artifact_schema_version": (
            CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": selection["producer_version"],
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "group_id": selection["group_id"],
        "source_artifact_id": source["artifact_id"],
        "source_artifact_kind": source["kind"],
        "source_artifact_schema_version": source["artifact_schema_version"],
        "source_artifact_content_hash": source["content_hash"],
        "source_artifact_origin_tool": source["origin_tool"],
        "source_artifact_path": source["path"],
        "source_artifact_provenance_hash": source["provenance_hash"],
        "source_asset_schema_version": asset["schema_version"],
        "source_asset_id": asset["asset_id"],
        "source_asset_hash": asset["asset_hash"],
        "source_asset_type": asset["asset_type"],
        "source_candidate_id": candidate["candidate_id"],
        "source_evidence_hash": candidate["evidence_hash"],
        "source_evidence_identity": candidate["evidence_identity"],
        "cell_ids": selection["cell_ids"],
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value))
    return set()


def test_single_and_multi_selection_are_pointer_only_and_row_major() -> None:
    asset = _build()
    cells = asset["matrix"]["cells"]
    populated = [cell for cell in cells if cell["effect"]["count"] > 0]

    single = _selection(asset, [populated[0]["cell_id"]])
    multi = _selection(
        asset,
        [populated[2]["cell_id"], populated[0]["cell_id"]],
    )

    assert single["cell_ids"] == [populated[0]["cell_id"]]
    expected = [
        cell["cell_id"]
        for cell in cells
        if cell["cell_id"] in {populated[0]["cell_id"], populated[2]["cell_id"]}
    ]
    assert multi["cell_ids"] == expected
    assert multi["group_id"].startswith("cross-matrix-cell-group-")
    assert json.loads(canonical_cross_matrix_cell_selection_json(multi)) == multi
    assert validate_cross_matrix_cell_selection(multi) == multi
    assert {
        "condition",
        "rule",
        "effect",
        "metrics",
        "count",
        "good",
        "bad",
        "action",
        "lifecycle",
        "candidate_stage",
        "observation_stage",
        "validation_status",
        "canonical_bytes",
        "provenance",
    }.isdisjoint(_all_keys(multi))


def test_reason_changes_selection_not_group_rule_effect_or_fragment_identity() -> None:
    asset = _build()
    cell_ids = [
        cell["cell_id"]
        for cell in asset["matrix"]["cells"]
        if cell["effect"]["count"] > 0
    ][:2]
    plain = _selection(asset, cell_ids)
    reasoned = _selection(asset, cell_ids, reason="  analyst\t review  ")

    assert reasoned["selection_reason"] == "analyst review"
    assert reasoned["selection_id"] != plain["selection_id"]
    assert reasoned["selection_hash"] != plain["selection_hash"]
    assert reasoned["group_id"] == plain["group_id"]
    plain_fragment = cross_matrix_cell_selection_to_verified_candidate_fragment(
        plain,
        asset,
        selection_artifact_binding=_selection_binding(plain, "artifact-plain"),
        source_artifact_binding=_binding(asset),
    )
    reasoned_fragment = cross_matrix_cell_selection_to_verified_candidate_fragment(
        reasoned,
        asset,
        selection_artifact_binding=_selection_binding(reasoned, "artifact-reasoned"),
        source_artifact_binding=_binding(asset),
    )
    assert plain_fragment["fragment"] == reasoned_fragment["fragment"]
    assert plain_fragment["fragment"]["fragment_id"] == plain["group_id"]
    assert plain_fragment["fragment"]["fragment_type"] == ("cross_matrix_cell_group")


def test_replay_aggregates_primary_facts_then_derives_coalesced_metrics() -> None:
    asset = _build()
    populated_indexes = [
        index
        for index, cell in enumerate(asset["measurement"]["cells"])
        if cell["count"] > 0
    ][:2]
    ids = [asset["matrix"]["cells"][index]["cell_id"] for index in populated_indexes]

    facts = derive_cross_matrix_cell_group_facts(asset, cell_ids=list(reversed(ids)))
    primary = [asset["measurement"]["cells"][index] for index in populated_indexes]
    effect = facts["effect"]
    count = sum(cell["count"] for cell in primary)
    good = sum(cell["good"] for cell in primary)
    bad = sum(cell["bad"] for cell in primary)
    groups = asset["matrix"]["cell_count"] - len(ids) + 1
    expected_woe, expected_iv = _smoothed_woe_iv(
        bad,
        good,
        asset["measurement"]["bad"],
        asset["measurement"]["good"],
        groups,
        smoothing=asset["parent"]["smoothing"],
    )

    assert facts["cell_ids"] == ids
    assert facts["fragment"]["condition"] == {
        "op": "or",
        "args": [
            asset["matrix"]["cells"][index]["rule"]["condition"]
            for index in populated_indexes
        ],
    }
    assert (effect["count"], effect["good"], effect["bad"]) == (count, good, bad)
    assert effect["woe_group_count"] == groups
    assert effect["woe"] == expected_woe
    assert effect["iv_contribution"] == expected_iv
    assert effect["amount_metrics"]["loan_amount"]["value"] == sum(
        cell["amounts"]["loan_amount"]["value"] for cell in primary
    )

    selection = _selection(asset, list(reversed(ids)))
    verified = cross_matrix_cell_selection_to_verified_candidate_fragment(
        selection,
        asset,
        selection_artifact_binding=_selection_binding(selection),
        source_artifact_binding=_binding(asset),
    )
    assert validate_verified_candidate_fragment(verified) == verified
    assert (
        verified["evidence"]["evidence_id"]
        == asset["candidate_evidence"]["candidate_id"]
    )
    assert verified["evidence"]["identity"] == {
        key: asset["sample_identity"][key]
        for key in (
            "dataset_id",
            "dataset_content_hash",
            "workspace_revision",
            "workspace_generation",
            "semantic_mapping_hash",
            "sample_context_hash",
        )
    }
    assert verified["candidate_stage"] == "development"
    assert verified["observation_stage"] == "backtested"
    assert verified["validation_status"] == "unvalidated"


def test_duplicate_unknown_empty_and_zero_count_groups_fail_closed() -> None:
    asset = _build()
    populated = next(
        cell for cell in asset["matrix"]["cells"] if cell["effect"]["count"] > 0
    )
    empty = next(
        cell for cell in asset["matrix"]["cells"] if cell["effect"]["count"] == 0
    )
    for cell_ids, message in (
        ([], "at least one"),
        ([populated["cell_id"], populated["cell_id"]], "duplicates"),
        (["cross-cell-" + "0" * 32], "unknown"),
        ([empty["cell_id"]], "positive total count"),
    ):
        with pytest.raises(CrossMatrixCellSelectionError, match=message):
            _selection(asset, cell_ids)


def test_authenticated_pointer_and_live_source_tamper_fail_closed() -> None:
    asset = _build()
    cell_id = next(
        cell["cell_id"]
        for cell in asset["matrix"]["cells"]
        if cell["effect"]["count"] > 0
    )
    selection = _selection(asset, [cell_id])
    changed = deepcopy(selection)
    changed["cell_ids"] = ["cross-cell-" + "0" * 32]
    with pytest.raises(CrossMatrixCellSelectionError, match="selection"):
        validate_cross_matrix_cell_selection(changed)

    bad_binding = {**_binding(asset), "canonical_bytes": b"{}"}
    with pytest.raises(CrossMatrixCellSelectionError, match="canonical bytes"):
        cross_matrix_cell_selection_to_verified_candidate_fragment(
            selection,
            asset,
            selection_artifact_binding=_selection_binding(selection),
            source_artifact_binding=bad_binding,
        )


def test_source_provenance_requires_exact_boolean_truncated_flag() -> None:
    asset = _build()
    cell_id = next(
        cell["cell_id"]
        for cell in asset["matrix"]["cells"]
        if cell["effect"]["count"] > 0
    )
    binding = _binding(asset)
    binding["provenance"] = {**binding["provenance"], "truncated": 0}
    binding["provenance_hash"] = hashlib.sha256(
        _canonical_json(binding["provenance"]).encode("utf-8")
    ).hexdigest()

    with pytest.raises(CrossMatrixCellSelectionError, match="truncated"):
        build_cross_matrix_cell_selection(
            asset,
            source_artifact_binding=binding,
            cell_ids=[cell_id],
        )


def test_pure_reason_limit_is_fail_closed() -> None:
    asset = _build()
    cell_id = next(
        cell["cell_id"]
        for cell in asset["matrix"]["cells"]
        if cell["effect"]["count"] > 0
    )
    with pytest.raises(CrossMatrixCellSelectionError, match="at most 500"):
        _selection(asset, [cell_id], reason="x" * 501)
