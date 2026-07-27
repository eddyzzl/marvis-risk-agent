from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marvis.packs.modeling.score_evidence_tools import (
    load_model_score_evidence_artifacts,
)
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
    hydrate_requirement_fields,
    model_score_virtual_field,
    normalize_pool_requirements,
    pool_requirement_bindings_provenance,
    project_pool_entry_requirements,
    require_resolved_pool_requirements_on_connection,
    resolve_pool_requirements,
    validate_pool_requirement_bindings_provenance,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    StrategySampleDesignV2ArtifactBinding,
    load_any_strategy_sample_design_v2_artifacts,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    run_materialize_sample_design_v2_native,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_model_score_evidence_tool import (
    _fixture,
    _run_score,
)
from tests.test_modeling_training_evidence_tool import _run as run_training


@pytest.fixture(scope="module")
def governed_score(tmp_path_factory: pytest.TempPathFactory) -> dict:
    fx = _fixture(tmp_path_factory.mktemp("pool-requirement-resolver"))
    score_output = _run_score(fx, run_training(fx))
    binding = load_model_score_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        evidence_artifact_id=score_output["artifacts"]["score_evidence"][
            "artifact_id"
        ],
        expected_evidence_artifact_content_hash=score_output["artifacts"][
            "score_evidence"
        ]["content_hash"],
        score_vector_artifact_id=score_output["artifacts"]["score_vector"][
            "artifact_id"
        ],
        expected_score_vector_artifact_content_hash=score_output["artifacts"][
            "score_vector"
        ]["content_hash"],
    )
    return {"fx": fx, "output": score_output, "binding": binding}


def _requirement(governed: dict) -> dict:
    binding = governed["binding"]
    vector_id = str(binding.vector_record["id"])
    return {
        "type": "model_score_vector.v1",
        "virtual_field": model_score_virtual_field(vector_id),
        "score_product": "raw_native_uncalibrated_bad_probability",
        "score_evidence_artifact_id": str(binding.evidence_record["id"]),
        "score_evidence_artifact_content_hash": str(
            binding.evidence_record["content_hash"]
        ),
        "score_vector_artifact_id": vector_id,
        "score_vector_artifact_content_hash": str(
            binding.vector_record["content_hash"]
        ),
    }


def _compiled(*requirements: dict) -> dict:
    return {
        "requirements": [
            {
                "rule_id": f"rule-{index}",
                "fragment_id": f"fragment-{index}",
                "requirement": requirement,
            }
            for index, requirement in enumerate(requirements)
        ]
    }


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _native_sample_binding(fx: dict, *, target_bad_value: int):
    sample = run_materialize_sample_design_v2_native(
        {
            "source_mode": "native_active_dataset",
            "dataset_id": fx["dataset"].id,
            "expected_dataset_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "workspace_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": fx["sample"]["bundle"]["sample_design"][
                "identity"
            ]["workspace_ref"]["semantic_mapping_hash"],
            "target_col": "bad",
            "target_bad_value": target_bad_value,
            "drop_nan_labels": False,
            "relationship": "nested_same_cohort",
            "scope": "strategy_development",
            "approval_population": {
                "inclusion": None,
                "exclusion": None,
            },
            "risk_population": {
                "inclusion": _eq("risk_flag", 1),
                "exclusion": None,
            },
            "partitioning": {
                "method": "predicate_ast",
                "selectors": {
                    "development": _eq(
                        "sample_partition",
                        "development",
                    ),
                    "validation": _eq("sample_partition", "validation"),
                    "oot": _eq("sample_partition", "oot"),
                },
            },
            "maturity": {
                "status": "confirmed_matured",
                "performance_window_days": 30,
                "cutoff_date": "2026-04-30",
                "reason": None,
            },
            "performance_window": {"status": "provided", "days": 30},
            "observation_window": {
                "status": "provided",
                "start": "2026-01-01",
                "end": "2026-04-30",
            },
            "field_bindings": {
                "entity_field": "customer_id",
                "time_field": "apply_date",
                "group_field": "channel",
                "month_field": "apply_month",
                "weight_field": "weight",
                "loan_amount_field": "loan_amount",
                "overdue_amount_field": "overdue_amount",
            },
            "historical_score": {
                "status": "available",
                "column": "legacy_score",
                "direction": "higher_is_riskier",
                "reason": None,
            },
            "policy": {
                "minimum_partition_count": 1,
                "minimum_bad_count": 1,
                "minimum_label_coverage": 1.0,
                "minimum_historical_score_coverage": 1.0,
                "maximum_group_coverage_gap": 1.0,
                "diagnostic_severities": {
                    "entity_overlap": "fail",
                    "temporal_oot": "fail",
                    "risk_outside_approval": "fail",
                    "maturity": "fail",
                    "label_coverage": "fail",
                    "historical_score_coverage": "warn",
                    "group_coverage_gap": "warn",
                    "sufficiency": "fail",
                },
            },
        },
        fx["ctx"],
        strategy_tools._runtime(fx["ctx"]),
    )
    records = TaskArtifactRepository(
        fx["settings"].db_path
    ).list_for_task(fx["task"].id)
    membership = next(
        item
        for item in records
        if item["kind"]
        == SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
        and item["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
    )
    bundle = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
        and item["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
    )
    return load_any_strategy_sample_design_v2_artifacts(
        strategy_tools._runtime(fx["ctx"]),
        task_id=fx["task"].id,
        membership_artifact_id=membership["id"],
        expected_membership_artifact_content_hash=membership["content_hash"],
        bundle_artifact_id=bundle["id"],
        expected_bundle_artifact_content_hash=bundle["content_hash"],
        expected_bundle_id=sample["bundle_id"],
        expected_sample_design_id=sample["sample_design_id"],
        expected_sample_design_content_hash=sample[
            "sample_design_content_hash"
        ],
    )


def _nested_requirement(
    requirement: dict,
    *,
    depth: int,
) -> dict:
    nested = requirement
    for index in range(depth):
        nested = {
            "entry_id": f"parent-entry-{index}",
            "rule_id": f"parent-rule-{index}",
            "fragment_id": f"parent-fragment-{index}",
            "requirement": nested,
        }
    return nested


def test_requirement_projection_flattens_voting_and_preserves_multiplicity() -> None:
    vector_id = "0123456789abcdef" + "a" * 48
    requirement = {
        "type": "model_score_vector.v1",
        "virtual_field": model_score_virtual_field(vector_id),
        "score_product": "raw_native_uncalibrated_bad_probability",
        "score_evidence_artifact_id": "1" * 64,
        "score_evidence_artifact_content_hash": "2" * 64,
        "score_vector_artifact_id": vector_id,
        "score_vector_artifact_content_hash": "3" * 64,
    }
    nested = _nested_requirement(requirement, depth=2)
    entries = [
        {
            "rule_id": "voting-rule",
            "source": {"fragment_id": "voting-fragment"},
            "execution": {"requirements": [nested, nested]},
        }
    ]

    projected = project_pool_entry_requirements(entries)

    expected = {
        "rule_id": "parent-rule-0",
        "fragment_id": "parent-fragment-0",
        "requirement": requirement,
    }
    assert projected == (expected, expected)
    assert normalize_pool_requirements(projected) == projected


@pytest.mark.parametrize(
    ("requirements", "error"),
    [
        (
            [
                {
                    "rule_id": "outer-rule",
                    "fragment_id": "outer-fragment",
                    "requirement": {
                        "entry_id": "missing-lineage-fields",
                        "requirement": {},
                    },
                }
            ],
            "lineage envelope",
        ),
        (
            [
                {
                    "rule_id": "outer-rule",
                    "fragment_id": "outer-fragment",
                    "requirement": _nested_requirement(
                        {
                            "type": "model_score_vector.v1",
                            "virtual_field": (
                                "__marvis_model_pd_0123456789abcdef"
                            ),
                            "score_product": (
                                "raw_native_uncalibrated_bad_probability"
                            ),
                            "score_evidence_artifact_id": "1" * 64,
                            "score_evidence_artifact_content_hash": "2" * 64,
                            "score_vector_artifact_id": (
                                "0123456789abcdef" + "a" * 48
                            ),
                            "score_vector_artifact_content_hash": "3" * 64,
                        },
                        depth=9,
                    ),
                }
            ],
            "depth",
        ),
    ],
)
def test_requirement_projection_rejects_malformed_or_excessive_envelopes(
    requirements: list[dict],
    error: str,
) -> None:
    with pytest.raises(StrategyError, match=error):
        normalize_pool_requirements(requirements)


def test_model_score_virtual_field_is_deterministic_and_strict() -> None:
    artifact_id = "0123456789abcdef" + "a" * 48

    assert model_score_virtual_field(artifact_id) == (
        "__marvis_model_pd_0123456789abcdef"
    )

    for invalid in ("0123456789abcdef", "A" * 64, "g" * 64, 123):
        with pytest.raises(StrategyError, match="score-vector artifact id"):
            model_score_virtual_field(invalid)  # type: ignore[arg-type]


@pytest.mark.slow
def test_resolver_authenticates_exact_score_reference_and_deduplicates(
    governed_score: dict,
) -> None:
    fx = governed_score["fx"]
    binding = governed_score["binding"]
    requirement = _requirement(governed_score)
    compiled = _compiled(requirement, requirement)
    strategy_runtime = SimpleNamespace(
        **{
            key: value
            for key, value in vars(fx["runtime"]).items()
            if key not in {"experiments", "modeling_repo"}
        }
    )

    resolved = resolve_pool_requirements(
        strategy_runtime,
        task_id=fx["task"].id,
        compiled_design=compiled,
        sample_design=binding.training.sample,
    )

    assert isinstance(resolved, ResolvedPoolRequirements)
    assert resolved.task_id == fx["task"].id
    assert resolved.requirements == tuple(compiled["requirements"])
    assert resolved.requirements_hash == hashlib.sha256(
        json.dumps(
            compiled["requirements"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert resolved.virtual_fields == (requirement["virtual_field"],)
    assert len(resolved.field_bindings) == 1
    field, loaded = resolved.field_bindings[0]
    assert field == requirement["virtual_field"]
    assert loaded.vector_record["id"] == binding.vector_record["id"]
    assert loaded.evidence_record["id"] == binding.evidence_record["id"]
    assert resolved.evidence_bindings == (loaded,)
    assert binding.vector.row_count == binding.training.sample.source_binding.row_count
    np.testing.assert_array_equal(
        binding.vector.row_ordinals,
        np.arange(binding.vector.row_count, dtype=np.int64),
    )
    provenance = pool_requirement_bindings_provenance(resolved)
    assert validate_pool_requirement_bindings_provenance(provenance) == provenance
    assert provenance["requirements_hash"] == resolved.requirements_hash
    assert provenance["requirements"] == compiled["requirements"]
    assert provenance["virtual_fields"] == [requirement["virtual_field"]]

    tampered = json.loads(json.dumps(provenance))
    tampered["requirements"][0]["requirement"][
        "score_vector_artifact_id"
    ] = "f" * 64
    with pytest.raises(StrategyError, match="changed|differ|match"):
        validate_pool_requirement_bindings_provenance(tampered)


@pytest.mark.slow
def test_resolver_binds_full_row_score_vector_to_native_sample(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path, target_bad_value=0)
    score_output = _run_score(fx, run_training(fx))
    score = load_model_score_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        evidence_artifact_id=score_output["artifacts"]["score_evidence"][
            "artifact_id"
        ],
        expected_evidence_artifact_content_hash=score_output["artifacts"][
            "score_evidence"
        ]["content_hash"],
        score_vector_artifact_id=score_output["artifacts"]["score_vector"][
            "artifact_id"
        ],
        expected_score_vector_artifact_content_hash=score_output["artifacts"][
            "score_vector"
        ]["content_hash"],
    )
    selected_sample = _native_sample_binding(fx, target_bad_value=0)
    requirement = _requirement({"binding": score})

    resolved = resolve_pool_requirements(
        strategy_tools._runtime(fx["ctx"]),
        task_id=fx["task"].id,
        compiled_design=_compiled(requirement),
        sample_design=selected_sample,
    )

    assert resolved.virtual_fields == (requirement["virtual_field"],)
    assert resolved.evidence_bindings[0].vector.row_count == (
        selected_sample.source_binding.row_count
    )
    assert selected_sample.bundle["sample_design"]["compatibility"] == {
        "source_mode": "native_active_dataset",
        "development_partition": "risk/development",
    }
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_resolved_pool_requirements_on_connection(conn, resolved)
        conn.rollback()

    wrong_target = replace(
        selected_sample,
        source_binding=replace(
            selected_sample.source_binding,
            target_bad_value=1,
        ),
    )
    with pytest.raises(StrategyError, match="native sample semantics"):
        resolve_pool_requirements(
            strategy_tools._runtime(fx["ctx"]),
            task_id=fx["task"].id,
            compiled_design=_compiled(requirement),
            sample_design=wrong_target,
        )

    wrong_dataset = replace(
        selected_sample,
        source_binding=replace(
            selected_sample.source_binding,
            dataset_id="different-native-dataset",
        ),
    )
    with pytest.raises(StrategyError, match="dataset/workspace"):
        resolve_pool_requirements(
            strategy_tools._runtime(fx["ctx"]),
            task_id=fx["task"].id,
            compiled_design=_compiled(requirement),
            sample_design=wrong_dataset,
        )


@pytest.mark.slow
def test_hydration_preserves_raw_row_order_and_cas_revalidates(
    governed_score: dict,
) -> None:
    fx = governed_score["fx"]
    binding = governed_score["binding"]
    requirement = _requirement(governed_score)
    resolved = resolve_pool_requirements(
        fx["runtime"],
        task_id=fx["task"].id,
        compiled_design=_compiled(requirement),
        sample_design=binding.training.sample,
    )
    frame = fx["runtime"].backend.read_frame(
        binding.training.sample.source_binding.dataset_path
    )
    original = frame.copy(deep=True)

    hydrated = hydrate_requirement_fields(frame, resolved=resolved)

    assert hydrated is not frame
    pd.testing.assert_frame_equal(frame, original)
    assert requirement["virtual_field"] not in frame.columns
    np.testing.assert_array_equal(
        hydrated[requirement["virtual_field"]].to_numpy(),
        binding.vector.scores,
    )
    assert not np.shares_memory(
        hydrated[requirement["virtual_field"]].to_numpy(),
        binding.vector.scores,
    )
    assert binding.vector.scores.flags.writeable is False

    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_resolved_pool_requirements_on_connection(conn, resolved)
        conn.rollback()

    with pytest.raises(StrategyError, match="disappeared before commit"):
        with fx["runtime"].task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM task_artifacts WHERE id = ?",
                (binding.vector_record["id"],),
            )
            require_resolved_pool_requirements_on_connection(conn, resolved)


def test_empty_requirements_are_a_deterministic_copy_only_noop(
) -> None:
    sample = StrategySampleDesignV2ArtifactBinding(
        task_id="task-empty-requirements",
        membership_artifact_id="1" * 64,
        membership_path=Path("/unused/membership.bin"),
        membership_artifact_content_hash="2" * 64,
        bundle_artifact_id="3" * 64,
        bundle_path=Path("/unused/bundle.json"),
        bundle_artifact_content_hash="4" * 64,
        provenance={},
        membership_provenance={},
        membership={},
        bundle={},
        source_binding=SimpleNamespace(),
    )
    resolved = resolve_pool_requirements(
        None,
        task_id=sample.task_id,
        compiled_design={"requirements": []},
        sample_design=sample,
    )
    frame = pd.DataFrame({"x": [1, 2]}, index=pd.Index([10, 20]))

    hydrated = hydrate_requirement_fields(frame, resolved=resolved)

    assert resolved.requirements == ()
    assert resolved.field_bindings == ()
    assert resolved.virtual_fields == ()
    assert resolved.evidence_bindings == ()
    assert resolved.requirements_hash == hashlib.sha256(b"[]").hexdigest()
    assert hydrated is not frame
    pd.testing.assert_frame_equal(hydrated, frame)


@pytest.mark.slow
def test_resolver_rejects_non_exact_or_mismatched_requirements(
    governed_score: dict,
) -> None:
    fx = governed_score["fx"]
    sample = governed_score["binding"].training.sample
    valid = _requirement(governed_score)

    invalid_requirements = [
        {**valid, "type": "future-score-vector.v1"},
        {**valid, "unexpected": True},
        {**valid, "virtual_field": "__marvis_model_pd_forged"},
        {**valid, "score_product": "calibrated_probability"},
    ]
    for requirement in invalid_requirements:
        with pytest.raises(StrategyError):
            resolve_pool_requirements(
                fx["runtime"],
                task_id=fx["task"].id,
                compiled_design=_compiled(requirement),
                sample_design=sample,
            )

    outer = _compiled(valid)
    outer["requirements"][0]["unexpected"] = True
    with pytest.raises(StrategyError, match="fields must be exactly"):
        resolve_pool_requirements(
            fx["runtime"],
            task_id=fx["task"].id,
            compiled_design=outer,
            sample_design=sample,
        )

    collision_vector_id = valid["score_vector_artifact_id"][:16] + "b" * 48
    collision = {
        **valid,
        "score_vector_artifact_id": collision_vector_id,
        "score_vector_artifact_content_hash": "b" * 64,
    }
    assert collision["virtual_field"] == model_score_virtual_field(
        collision_vector_id
    )
    with pytest.raises(StrategyError, match="cannot bind different"):
        resolve_pool_requirements(
            fx["runtime"],
            task_id=fx["task"].id,
            compiled_design=_compiled(valid, collision),
            sample_design=sample,
        )


@pytest.mark.slow
def test_resolver_rejects_physical_field_and_sample_identity_conflicts(
    governed_score: dict,
) -> None:
    fx = governed_score["fx"]
    sample = governed_score["binding"].training.sample
    requirement = _requirement(governed_score)
    source = sample.source_binding
    physical_conflict = replace(
        sample,
        source_binding=replace(
            source,
            columns=(*source.columns, requirement["virtual_field"]),
        ),
    )
    with pytest.raises(StrategyError, match="physical dataset column"):
        resolve_pool_requirements(
            fx["runtime"],
            task_id=fx["task"].id,
            compiled_design=_compiled(requirement),
            sample_design=physical_conflict,
        )

    wrong_dataset = replace(
        sample,
        source_binding=replace(source, dataset_id="different-dataset"),
    )
    with pytest.raises(StrategyError, match="dataset/workspace"):
        resolve_pool_requirements(
            fx["runtime"],
            task_id=fx["task"].id,
            compiled_design=_compiled(requirement),
            sample_design=wrong_dataset,
        )

    wrong_sample = replace(sample, membership_artifact_id="f" * 64)
    with pytest.raises(StrategyError, match="SampleDesign V2"):
        resolve_pool_requirements(
            fx["runtime"],
            task_id=fx["task"].id,
            compiled_design=_compiled(requirement),
            sample_design=wrong_sample,
        )


@pytest.mark.slow
def test_hydration_rejects_existing_field_length_and_row_order_ambiguity(
    governed_score: dict,
) -> None:
    fx = governed_score["fx"]
    binding = governed_score["binding"]
    requirement = _requirement(governed_score)
    resolved = resolve_pool_requirements(
        fx["runtime"],
        task_id=fx["task"].id,
        compiled_design=_compiled(requirement),
        sample_design=binding.training.sample,
    )
    frame = fx["runtime"].backend.read_frame(
        binding.training.sample.source_binding.dataset_path
    )

    existing = frame.copy()
    existing[requirement["virtual_field"]] = 0.0
    with pytest.raises(StrategyError, match="already exists"):
        hydrate_requirement_fields(existing, resolved=resolved)

    with pytest.raises(StrategyError, match="length"):
        hydrate_requirement_fields(
            frame.iloc[:-1].reset_index(drop=True),
            resolved=resolved,
        )

    ambiguous = frame.copy()
    ambiguous.index = pd.Index(range(1, len(ambiguous) + 1))
    with pytest.raises(StrategyError, match="RangeIndex"):
        hydrate_requirement_fields(ambiguous, resolved=resolved)

    with pytest.raises(StrategyError, match="bindings do not match"):
        hydrate_requirement_fields(
            frame,
            resolved=replace(resolved, field_bindings=()),
        )
