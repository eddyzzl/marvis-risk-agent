from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

import marvis.packs.strategy.pool_stability as stability_module
from marvis.packs.strategy.impact_cube import (
    build_strategy_impact_cube,
    canonical_strategy_impact_cube_json,
)
from marvis.packs.strategy.impact_cube_tools import IMPACT_CUBE_ARTIFACT_KIND
from marvis.packs.strategy.pool_stability import (
    MAX_POOL_STABILITY_JSON_BYTES,
    POOL_STABILITY_SCHEMA_VERSION,
    PoolStabilityError,
    build_strategy_pool_stability,
    canonical_strategy_pool_stability_json,
    strategy_pool_stability_content_hash,
    validate_strategy_pool_stability,
)
from marvis.repositories.task_artifacts import stable_task_artifact_id
from test_strategy_impact_cube import (
    _build as _build_impact_cube,
    _dataset_binding,
    _frame,
    _legacy_ref,
    _pool,
    _pool_artifact_ref,
    _sample_design_v2_ref,
)


def _impact_cube_ref(cube: dict) -> dict:
    logical_path = (
        "/governed/tasks/"
        f"{cube['identity']['task_id']}/strategy_impact_cubes/"
        f"{cube['cube_id']}.json"
    )
    return {
        "artifact_id": stable_task_artifact_id(
            task_id=cube["identity"]["task_id"],
            kind=IMPACT_CUBE_ARTIFACT_KIND,
            path=logical_path,
        ),
        "expected_artifact_content_hash": hashlib.sha256(
            canonical_strategy_impact_cube_json(cube).encode("utf-8")
        ).hexdigest(),
        "expected_cube_id": cube["cube_id"],
        "expected_cube_content_hash": cube["content_hash"],
    }


def _build(strategy_type: str = "approval") -> dict:
    cube = _build_impact_cube(strategy_type)
    return build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )


def _build_cube_from_frames(
    strategy_type: str,
    frames: dict[str, pd.DataFrame],
) -> dict:
    counts = {name: len(frame) for name, frame in frames.items()}
    sample_ref = _sample_design_v2_ref(
        approval_counts=counts,
        risk_counts=counts,
    )
    sample_ref["analysis_universe_row_count"] = sum(counts.values())
    return build_strategy_impact_cube(
        pool=_pool(strategy_type),
        approval_partition_frames={
            name: frame.reset_index(drop=True)
            for name, frame in frames.items()
        },
        partition_frames={
            name: frame.reset_index(drop=True)
            for name, frame in frames.items()
        },
        pool_artifact_ref=_pool_artifact_ref(),
        sample_design_v2_ref=sample_ref,
        dataset_binding=_dataset_binding(),
        legacy_development_ref=_legacy_ref(),
        target_col="bad",
        target_bad_value=1,
        month_col=None,
        group_col=None,
        segment_col=None,
        current_strategy_spec=None,
        current_strategy_ref=None,
        economics_bindings=None,
    )


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _rehash(document: dict) -> dict:
    body = {
        key: value
        for key, value in document.items()
        if key not in {"stability_id", "content_hash"}
    }
    document["stability_id"] = (
        "strategy-pool-stability-"
        + hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()[:24]
    )
    without_hash = {
        key: value for key, value in document.items() if key != "content_hash"
    }
    document["content_hash"] = hashlib.sha256(
        _canonical(without_hash).encode("utf-8")
    ).hexdigest()
    return document


def _distribution(
    artifact: dict,
    *,
    population_role: str,
    basis: str,
    partition: str = "validation",
) -> dict:
    population = next(
        row
        for row in artifact["populations"]
        if row["population_role"] == population_role
    )
    comparison = next(
        row
        for row in population["comparisons"]
        if row["partition"] == partition
    )
    return next(
        row for row in comparison["distributions"] if row["basis"] == basis
    )


def test_builds_canonical_cross_partition_pool_stability() -> None:
    cube = _build_impact_cube("approval")
    first = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )
    second = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )

    assert first == second
    assert first["schema_version"] == POOL_STABILITY_SCHEMA_VERSION
    assert first["identity"] == cube["identity"]
    assert first["source_bindings"] == {
        "impact_cube": _impact_cube_ref(cube),
        "sample_design_v2": cube["source_bindings"]["sample_design_v2"],
        "dataset": cube["source_bindings"]["dataset"],
    }
    assert first["baseline_partition"] == "development"
    assert first["comparison_partitions"] == ["validation"]
    assert [row["population_role"] for row in first["populations"]] == [
        "approval",
        "risk",
    ]
    assert first["lifecycle"] == {
        "read_only": True,
        "effect_validation": False,
        "automatic_promotion": False,
        "mutates_pool": False,
        "creates_strategy": False,
        "adopts_strategy": False,
        "promotes_strategy": False,
        "deploys_strategy": False,
    }

    waterfall = _distribution(
        first,
        population_role="risk",
        basis="waterfall_incremental",
    )
    assert waterfall["development_sample_count"] == 3
    assert waterfall["comparison_sample_count"] == 3
    assert [
        (
            row["category"]["kind"],
            row["development_count"],
            row["comparison_count"],
        )
        for row in waterfall["categories"]
    ] == [
        ("pool_entry_incremental", 2, 0),
        ("pool_entry_incremental", 1, 1),
        ("default_unmatched", 0, 2),
    ]
    assert waterfall["psi"] > 0.25
    assert waterfall["severity"] == "material"
    assert waterfall["max_abs_share_delta"] == pytest.approx(2 / 3)

    action = _distribution(
        first,
        population_role="risk",
        basis="new_action",
    )
    assert sum(row["development_count"] for row in action["categories"]) == 3
    assert sum(row["comparison_count"] for row in action["categories"]) == 3
    assert {row["category"]["action"]["type"] for row in action["categories"]} == {
        "approval",
        "reject",
        "review",
    }
    assert action["severity"] == "material"

    raw = canonical_strategy_pool_stability_json(first)
    assert json.loads(raw) == first
    assert raw == canonical_strategy_pool_stability_json(json.loads(raw))
    assert validate_strategy_pool_stability(first) == first
    assert strategy_pool_stability_content_hash(first) == first["content_hash"]
    assert hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in first.items()
                if key != "content_hash"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest() == first["content_hash"]


def test_pool_stability_accepts_parallel_risk_population_larger_than_approval(
) -> None:
    frame = _frame()
    cube = build_strategy_impact_cube(
        pool=_pool("approval"),
        approval_partition_frames={
            "development": frame.iloc[:2].reset_index(drop=True),
            "validation": frame.iloc[3:5].reset_index(drop=True),
        },
        partition_frames={
            "development": frame.iloc[:3].reset_index(drop=True),
            "validation": frame.iloc[3:].reset_index(drop=True),
        },
        pool_artifact_ref=_pool_artifact_ref(),
        sample_design_v2_ref=_sample_design_v2_ref(
            approval_counts={"development": 2, "validation": 2},
            risk_counts={"development": 3, "validation": 3},
        ),
        dataset_binding=_dataset_binding(),
        legacy_development_ref=_legacy_ref(),
        target_col="bad",
        target_bad_value=1,
        month_col=None,
        group_col=None,
        segment_col=None,
        current_strategy_spec=None,
        current_strategy_ref=None,
        economics_bindings=None,
    )

    stability = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )

    assert stability["source_bindings"]["sample_design_v2"][
        "population_partition_counts"
    ] == {
        "approval": {"development": 2, "validation": 2},
        "risk": {"development": 3, "validation": 3},
    }


@pytest.mark.parametrize(
    "strategy_type",
    ["approval", "reject", "limit", "pricing", "segmentation"],
)
def test_supports_all_five_pool_types_with_canonical_typed_actions(
    strategy_type: str,
) -> None:
    cube = _build_impact_cube(strategy_type)
    artifact = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )

    assert artifact["identity"]["strategy_type"] == strategy_type
    assert validate_strategy_pool_stability(
        artifact,
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    ) == artifact
    for role in ("approval", "risk"):
        action = _distribution(
            artifact,
            population_role=role,
            basis="new_action",
        )
        assert sum(
            row["development_count"] for row in action["categories"]
        ) == action["development_sample_count"]
        assert sum(
            row["comparison_count"] for row in action["categories"]
        ) == action["comparison_sample_count"]
        action_tokens = [
            _canonical(row["category"]["action"])
            for row in action["categories"]
        ]
        assert action_tokens == sorted(action_tokens)


def test_uses_category_union_zero_buckets_and_real_zero_psi() -> None:
    base = _frame().iloc[:3].reset_index(drop=True)
    identical = _build_cube_from_frames(
        "approval",
        {
            "development": base,
            "validation": base.copy(),
        },
    )
    stable = build_strategy_pool_stability(
        impact_cube=identical,
        impact_cube_ref=_impact_cube_ref(identical),
    )
    for role in ("approval", "risk"):
        for basis in ("waterfall_incremental", "new_action"):
            distribution = _distribution(
                stable,
                population_role=role,
                basis=basis,
            )
            assert distribution["psi"] == 0.0
            assert distribution["max_abs_share_delta"] == 0.0
            assert distribution["severity"] == "stable"

    shifted = _build()
    for role in ("approval", "risk"):
        waterfall = _distribution(
            shifted,
            population_role=role,
            basis="waterfall_incremental",
        )
        assert any(
            row["development_count"] == 0
            and row["comparison_count"] > 0
            for row in waterfall["categories"]
        )
        assert any(
            row["development_count"] > 0
            and row["comparison_count"] == 0
            for row in waterfall["categories"]
        )
        action = _distribution(
            shifted,
            population_role=role,
            basis="new_action",
        )
        assert any(
            row["development_count"] == 0
            and row["comparison_count"] > 0
            for row in action["categories"]
        )
        assert any(
            row["development_count"] > 0
            and row["comparison_count"] == 0
            for row in action["categories"]
        )


def test_applies_fixed_warning_band_between_stable_and_material() -> None:
    development = pd.DataFrame(
        {"x": [0] * 5 + [5] * 5, "bad": [0] * 10}
    )
    validation = pd.DataFrame(
        {"x": [0] * 7 + [5] * 3, "bad": [0] * 10}
    )
    cube = _build_cube_from_frames(
        "approval",
        {
            "development": development,
            "validation": validation,
        },
    )
    artifact = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )

    for role in ("approval", "risk"):
        for basis in ("waterfall_incremental", "new_action"):
            distribution = _distribution(
                artifact,
                population_role=role,
                basis=basis,
            )
            assert 0.10 <= distribution["psi"] < 0.25
            assert distribution["severity"] == "warning"


def test_preserves_validation_and_oot_as_distinct_comparisons() -> None:
    frame = _frame()
    cube = _build_cube_from_frames(
        "approval",
        {
            "development": frame.iloc[:2],
            "validation": frame.iloc[2:4],
            "oot": frame.iloc[4:6],
        },
    )
    artifact = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )

    assert artifact["comparison_partitions"] == ["validation", "oot"]
    for population in artifact["populations"]:
        assert [row["partition"] for row in population["comparisons"]] == [
            "validation",
            "oot",
        ]
        for partition in ("validation", "oot"):
            for basis in ("waterfall_incremental", "new_action"):
                distribution = _distribution(
                    artifact,
                    population_role=population["population_role"],
                    partition=partition,
                    basis=basis,
                )
                assert distribution["development_sample_count"] == 2
                assert distribution["comparison_sample_count"] == 2


def test_waterfall_basis_uses_incremental_not_standalone_counts() -> None:
    cube = _build_impact_cube("approval")
    overall = next(
        row
        for row in cube["slices"]
        if row["population_role"] == "risk"
        and row["family"] == "overall"
        and row["dimensions"]["partition"]["value"] == "development"
    )
    second_entry = overall["waterfall"]["value"]["entries"][1]
    assert second_entry["standalone"]["count"] == 3
    assert second_entry["incremental"]["count"] == 1

    artifact = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=_impact_cube_ref(cube),
    )
    waterfall = _distribution(
        artifact,
        population_role="risk",
        basis="waterfall_incremental",
    )
    assert waterfall["categories"][1]["development_count"] == 1


def test_validator_recomputes_derived_values_after_coherent_rehash() -> None:
    artifact = _build()

    drifted = copy.deepcopy(artifact)
    distribution = _distribution(
        drifted,
        population_role="risk",
        basis="waterfall_incremental",
    )
    distribution["psi"] = 0.0
    _rehash(drifted)
    with pytest.raises(PoolStabilityError, match="psi.*derived"):
        validate_strategy_pool_stability(drifted)

    drifted = copy.deepcopy(artifact)
    distribution = _distribution(
        drifted,
        population_role="risk",
        basis="waterfall_incremental",
    )
    distribution["categories"][0]["comparison_count"] = 1
    _rehash(drifted)
    with pytest.raises(
        PoolStabilityError,
        match="derived evidence|counts do not conserve",
    ):
        validate_strategy_pool_stability(drifted)

    drifted = copy.deepcopy(artifact)
    distribution = _distribution(
        drifted,
        population_role="risk",
        basis="waterfall_incremental",
    )
    distribution["severity"] = "stable"
    _rehash(drifted)
    with pytest.raises(PoolStabilityError, match="severity"):
        validate_strategy_pool_stability(drifted)


def test_source_rebinding_rejects_coherently_rehashed_source_drift() -> None:
    cube = _build_impact_cube("approval")
    ref = _impact_cube_ref(cube)
    artifact = build_strategy_pool_stability(
        impact_cube=cube,
        impact_cube_ref=ref,
    )

    drifted = copy.deepcopy(artifact)
    drifted["identity"]["pool_id"] = "another-pool"
    _rehash(drifted)
    assert validate_strategy_pool_stability(drifted) == drifted
    with pytest.raises(PoolStabilityError, match="authenticated ImpactCube"):
        validate_strategy_pool_stability(
            drifted,
            impact_cube=cube,
            impact_cube_ref=ref,
        )

    wrong_ref = dict(ref)
    wrong_ref["expected_cube_content_hash"] = "0" * 64
    with pytest.raises(PoolStabilityError, match="does not bind"):
        build_strategy_pool_stability(
            impact_cube=cube,
            impact_cube_ref=wrong_ref,
        )

    wrong_ref = dict(ref)
    wrong_ref["artifact_id"] = "not-a-hash"
    with pytest.raises(PoolStabilityError, match="artifact_id"):
        build_strategy_pool_stability(
            impact_cube=cube,
            impact_cube_ref=wrong_ref,
        )


@pytest.mark.parametrize(
    ("partitions", "message"),
    [
        (("development",), "validation and/or OOT"),
        (("validation",), "development baseline"),
    ],
)
def test_requires_development_and_at_least_one_comparison_partition(
    partitions: tuple[str, ...],
    message: str,
) -> None:
    frame = _frame()
    frames = {
        partition: frame.iloc[:3].copy()
        for partition in partitions
    }
    cube = _build_cube_from_frames("approval", frames)

    with pytest.raises(PoolStabilityError, match=message):
        build_strategy_pool_stability(
            impact_cube=cube,
            impact_cube_ref=_impact_cube_ref(cube),
        )


def test_rejects_non_overall_source_misuse_and_untrusted_cube() -> None:
    cube = copy.deepcopy(_build_impact_cube("approval"))
    overall = next(
        row
        for row in cube["slices"]
        if row["population_role"] == "risk"
        and row["family"] == "overall"
        and row["dimensions"]["partition"]["value"] == "development"
    )
    overall["family"] = "month"

    with pytest.raises(PoolStabilityError, match="ImpactCube is invalid"):
        build_strategy_pool_stability(
            impact_cube=cube,
            impact_cube_ref=_impact_cube_ref(
                _build_impact_cube("approval")
            ),
        )


def test_rejects_non_finite_bool_pseudonumbers_and_noncanonical_actions() -> None:
    artifact = _build("limit")

    invalid = copy.deepcopy(artifact)
    distribution = _distribution(
        invalid,
        population_role="risk",
        basis="waterfall_incremental",
    )
    distribution["categories"][0]["development_count"] = True
    _rehash(invalid)
    with pytest.raises(PoolStabilityError, match="JSON-safe integer"):
        validate_strategy_pool_stability(invalid)

    invalid = copy.deepcopy(artifact)
    distribution = _distribution(
        invalid,
        population_role="risk",
        basis="new_action",
    )
    distribution["categories"][0]["category"]["action"]["value"] = True
    _rehash(invalid)
    with pytest.raises(PoolStabilityError, match="action is invalid"):
        validate_strategy_pool_stability(invalid)

    invalid = copy.deepcopy(artifact)
    distribution = _distribution(
        invalid,
        population_role="risk",
        basis="waterfall_incremental",
    )
    distribution["psi"] = float("inf")
    with pytest.raises(PoolStabilityError, match="finite JSON"):
        validate_strategy_pool_stability(invalid)


def test_strict_fields_lifecycle_hashes_and_byte_budget_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build()

    extra = copy.deepcopy(artifact)
    extra["unexpected"] = True
    with pytest.raises(PoolStabilityError, match="unsupported fields"):
        validate_strategy_pool_stability(extra)

    invalid = copy.deepcopy(artifact)
    invalid["lifecycle"]["effect_validation"] = True
    _rehash(invalid)
    with pytest.raises(PoolStabilityError, match="lifecycle"):
        validate_strategy_pool_stability(invalid)

    invalid = copy.deepcopy(artifact)
    invalid["content_hash"] = "0" * 64
    with pytest.raises(PoolStabilityError, match="content_hash"):
        validate_strategy_pool_stability(invalid)

    assert len(_canonical(artifact).encode("utf-8")) < (
        MAX_POOL_STABILITY_JSON_BYTES
    )
    monkeypatch.setattr(
        stability_module,
        "MAX_POOL_STABILITY_JSON_BYTES",
        len(_canonical(artifact).encode("utf-8")) - 1,
    )
    with pytest.raises(PoolStabilityError, match="byte budget"):
        validate_strategy_pool_stability(artifact)
