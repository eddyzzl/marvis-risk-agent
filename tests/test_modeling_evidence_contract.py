from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from marvis.feature.metrics import bootstrap_ks_ci
from marvis.packs.modeling._common import BINARY_MODELING_RECIPES
from marvis.packs.modeling.contracts import (
    Experiment,
    ModelArtifact,
    ModelMetrics,
    TrainConfig,
)
from marvis.packs.modeling.evidence import (
    LEGACY_RAW_CLASS_ONE_SCORE_PRODUCT,
    MAX_TRAINING_MASK_BYTES,
    MAX_TRAINING_MASK_ROWS,
    MODEL_TARGET_ENCODING_RULE,
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
    MODELING_TRAINING_EVIDENCE_SCHEMA_VERSION,
    NON_FINITE_BOUNDARY_TAG,
    RAW_BAD_PROBABILITY_SCORE_PRODUCT,
    SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND,
    SAMPLE_MEMBERSHIP_ARTIFACT_KIND,
    ModelingTrainingEvidenceError,
    build_model_binary_artifact_ref,
    build_modeling_training_evidence,
    build_task_artifact_ref,
    build_training_split_mask_hashes,
    canonical_modeling_training_evidence_json,
    decode_modeling_scoring_woe_maps_boundaries,
    modeling_scoring_metadata_from_artifact,
    modeling_training_evidence_from_json,
    validate_modeling_training_evidence,
)
from marvis.packs.strategy.sample_membership import (
    MAX_MEMBERSHIP_PAYLOAD_BYTES,
    MEMBERSHIP_MASK_ORDER,
)
from marvis.packs.strategy.sample_design_v2 import (
    build_metric_definitions_v2,
    build_metric_observation_v2,
    build_sample_population_v2,
    build_strategy_sample_design_v2,
    build_strategy_sample_design_v2_bundle,
    build_target_selector_v2,
    canonical_strategy_sample_design_v2_bundle_json,
)
from tests.test_strategy_sample_design_v2 import (
    _bundle as build_sample_design_v2_fixture,
    _components,
    _decoded_membership,
    _design_kwargs,
    _diagnostic_statistics,
    _metric_observations,
    _policy,
    _source_ref,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _config(**overrides) -> TrainConfig:
    defaults = {
        "dataset_id": "dataset-v2",
        "features": ("income", "age"),
        "target_col": "target",
        "split_col": "model_split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "params": {"learning_rate": 0.05, "num_leaves": 16},
        "seed": 42,
        "early_stopping_rounds": 20,
        "recipe_id": "lgb",
        "scenario_id": None,
        "target_type": "binary",
        "eval_metric": "ks_auc",
        "drop_nan_labels": True,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _metrics(**overrides) -> ModelMetrics:
    train_ks = 0.42
    test_ks = 0.38
    oot_ks = 0.35
    defaults = {
        "train_ks": train_ks,
        "test_ks": test_ks,
        "oot_ks": oot_ks,
        "train_auc": 0.76,
        "test_auc": 0.73,
        "oot_auc": 0.71,
        "psi_test_vs_train": 0.03,
        "psi_oot_vs_train": 0.07,
        "overfit_train_test_gap": abs(train_ks - test_ks) / train_ks,
        "overfit_train_oot_gap": abs(train_ks - oot_ks),
        "overfit_flag": True,
    }
    defaults.update(overrides)
    return ModelMetrics(**defaults)


def _artifact(**overrides) -> ModelArtifact:
    defaults = {
        "id": "artifact_0123456789abcdef0123456789abcdef",
        "experiment_id": "experiment_0123456789abcdef0123456789abcdef",
        "algorithm": "lgb",
        "model_path": "artifact/model.txt",
        "pmml_path": None,
        "feature_list": ("income", "age"),
        "params": {"num_leaves": 16, "learning_rate": 0.05},
        "woe_maps": None,
        "created_at": "2026-07-22T01:02:03+00:00",
        "feature_importance": (("income", 0.7), ("age", 0.3)),
        "score_direction": "higher_is_riskier",
        "points_direction": None,
    }
    defaults.update(overrides)
    return ModelArtifact(**defaults)


def _experiment(**overrides) -> Experiment:
    defaults = {
        "id": "experiment_0123456789abcdef0123456789abcdef",
        "task_id": "task-v2",
        "recipe_id": "lgb",
        "config": _config(),
        "metrics": _metrics(),
        "artifact_id": "artifact_0123456789abcdef0123456789abcdef",
        "status": "trained",
        "created_at": "2026-07-22T01:01:01+00:00",
    }
    defaults.update(overrides)
    return Experiment(**defaults)


def _sample_refs(bundle):
    bundle_hash = hashlib.sha256(
        canonical_strategy_sample_design_v2_bundle_json(bundle).encode("utf-8")
    ).hexdigest()
    return {
        "membership_artifact_ref": build_task_artifact_ref(
            artifact_id=_hash("membership-artifact-id"),
            kind=SAMPLE_MEMBERSHIP_ARTIFACT_KIND,
            content_hash=_hash("membership-artifact-bytes"),
        ),
        "sample_design_bundle_artifact_ref": build_task_artifact_ref(
            artifact_id=_hash("bundle-artifact-id"),
            kind=SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND,
            content_hash=bundle_hash,
        ),
    }


def _training_split_mask_hashes(bundle):
    counts = bundle["membership"]["counts"]["risk"]
    decoded = _decoded_membership(
        risk_outside_approval=(
            bundle["membership"]["counts"]["relationship"][
                "risk_outside_approval"
            ]["total"]
            > 0
        ),
        empty_development=counts["development"] == 0,
        empty_oot=counts["oot"] == 0,
    )
    membership = {
        "train": decoded["masks"]["risk/development"].tolist(),
        "test": decoded["masks"]["risk/validation"].tolist(),
        "oot": decoded["masks"]["risk/oot"].tolist(),
    }
    risk_union = [
        any(values)
        for values in zip(
            membership["train"],
            membership["test"],
            membership["oot"],
            strict=True,
        )
    ]
    return build_training_split_mask_hashes(
        sample_design_bundle=bundle,
        selector_masks=membership,
        membership_masks=membership,
        risk_membership_mask=risk_union,
    )


def _custom_bundle(*, good_value=0, bad_value=1, oot_mode="default"):
    decoded = _decoded_membership()
    approval, risk, _target, historical = _components(decoded)
    if oot_mode == "unlabeled":
        maturity = deepcopy(risk["maturity_evidence"])
        maturity["labeled_count"] -= decoded["header"]["counts"]["risk"]["oot"]
        risk = build_sample_population_v2(
            role="risk",
            membership_header=decoded["header"],
            inclusion_predicate_ref=risk["inclusion_predicate_ref"],
            exclusion_predicate_ref=risk["exclusion_predicate_ref"],
            maturity_evidence=maturity,
            source_refs=risk["source_refs"],
        )
    target = build_target_selector_v2(
        status="resolved",
        column="target",
        good_value=good_value,
        bad_value=bad_value,
        drop_missing=True,
        source_refs=[_source_ref("target")],
    )
    design_kwargs = _design_kwargs()
    design = build_strategy_sample_design_v2(
        task_id="task-v2",
        membership_header=decoded["header"],
        relationship="nested_same_cohort",
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_policy(),
        source_refs=[_source_ref("design-b"), _source_ref("design-a")],
        **design_kwargs,
    )
    observations = _metric_observations(
        decoded,
        design,
        maturity_status="confirmed_matured",
    )
    if oot_mode in {"single_class", "unlabeled"}:
        definitions = {
            item["metric_key"]: item for item in build_metric_definitions_v2()
        }
        keys_by_id = {
            item["metric_definition_id"]: item["metric_key"]
            for item in definitions.values()
        }
        if oot_mode == "single_class":
            updates = {
                ("oot", "bad_count"): ("present", 2, 2, 2),
                ("oot", "bad_rate"): ("present", 1.0, 2, 2),
                ("overall", "bad_count"): ("present", 4, 4, 6),
                ("overall", "bad_rate"): ("present", 4 / 6, 4, 6),
            }
        else:
            updates = {
                ("oot", "labeled_count"): ("present", 0, 0, 2),
                ("oot", "label_coverage"): ("present", 0.0, 0, 2),
                ("oot", "bad_count"): ("present", 0, 0, 0),
                ("oot", "bad_rate"): ("insufficient_data", None, None, None),
                ("overall", "labeled_count"): ("present", 4, 4, 6),
                ("overall", "label_coverage"): ("present", 4 / 6, 4, 6),
                ("overall", "bad_count"): ("present", 2, 2, 4),
                ("overall", "bad_rate"): ("present", 0.5, 2, 4),
            }
        rebuilt = []
        design_ref = {
            "sample_design_id": design["sample_design_id"],
            "content_hash": design["content_hash"],
        }
        for observation in observations:
            metric_key = keys_by_id[
                observation["metric_definition_ref"]["metric_definition_id"]
            ]
            update = updates.get((observation["partition"], metric_key))
            if observation["population"] != "risk" or update is None:
                rebuilt.append(observation)
                continue
            status, value, numerator, denominator = update
            rebuilt.append(
                build_metric_observation_v2(
                    sample_design_ref=design_ref,
                    metric_definition=definitions[metric_key],
                    population="risk",
                    partition=observation["partition"],
                    status=status,
                    value=value,
                    numerator=numerator,
                    denominator=denominator,
                    sample_count=observation["sample_count"],
                    source_refs=observation["source_refs"],
                )
            )
        observations = rebuilt
    return build_strategy_sample_design_v2_bundle(
        task_id="task-v2",
        membership_header=decoded["header"],
        membership_masks=decoded["masks"],
        relationship="nested_same_cohort",
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_policy(),
        diagnostic_statistics=_diagnostic_statistics(decoded),
        metric_observations=observations,
        source_refs=[_source_ref("design-a"), _source_ref("design-b")],
        **design_kwargs,
    )


def _evidence(
    *,
    bundle=None,
    experiment=None,
    artifact=None,
    nan_labels_dropped=0,
    **overrides,
):
    sample_bundle = bundle or build_sample_design_v2_fixture()
    kwargs = {
        "experiment": experiment or _experiment(),
        "model_artifact": artifact or _artifact(),
        "sample_design_bundle": sample_bundle,
        **_sample_refs(sample_bundle),
        "model_binary_artifact_ref": build_model_binary_artifact_ref(
            artifact_id=_hash("model-binary-artifact-id"),
            model_artifact_id=(artifact or _artifact()).id,
            content_hash=_hash("native-model-bytes"),
        ),
        "training_split_mask_hashes": _training_split_mask_hashes(
            sample_bundle
        ),
        "nan_labels_dropped": nan_labels_dropped,
    }
    kwargs.update(overrides)
    return build_modeling_training_evidence(**kwargs)


def test_builds_task_owned_content_addressed_training_evidence():
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)

    assert evidence["schema_version"] == MODELING_TRAINING_EVIDENCE_SCHEMA_VERSION
    assert evidence["artifact_kind"] == MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND
    assert evidence["task_id"] == "task-v2"
    assert evidence["experiment"]["status"] == "trained"
    assert evidence["model_artifact"]["model_binary_ref"] == {
        "artifact_id": _hash("model-binary-artifact-id"),
        "kind": "modeling_model_binary",
        "content_hash": _hash("native-model-bytes"),
        "model_artifact_id": evidence["model_artifact"]["artifact_id"],
    }
    metadata = evidence["model_artifact"]["scoring_metadata"]
    assert metadata["score_product"] == RAW_BAD_PROBABILITY_SCORE_PRODUCT
    assert metadata["calibration_status"] == "not_applied"
    assert len(evidence["model_artifact"]["scoring_metadata_hash"]) == 64
    binding = evidence["sample_design_binding"]
    assert binding["bundle_ref"]["bundle_id"] == bundle["bundle_id"]
    assert binding["membership_ref"]["payload_hash"] == bundle["membership"][
        "payload_hash"
    ]
    assert binding["dataset_ref"]["dataset_id"] == "dataset-v2"
    assert binding["workspace_ref"]["revision"] == 7


def test_freezes_canonical_train_config_target_and_verified_mask_proof():
    evidence = _evidence()
    training = evidence["training_contract"]

    assert training["train_config"]["features"] == ["income", "age"]
    assert training["features"] == ["income", "age"]
    assert training["seed"] == 42
    assert training["early_stopping_rounds"] == 20
    assert training["label_handling"] == {
        "drop_nan_labels": True,
        "nan_labels_dropped": 0,
    }
    assert training["weighting"] == {"used": False, "column": None}
    assert training["target"] == {
        "column": "target",
        "good_value": 0,
        "bad_value": 1,
        "encoded_good_value": 0,
        "encoded_bad_value": 1,
        "encoding_rule": MODEL_TARGET_ENCODING_RULE,
        "encoding_content_hash": training["target"][
            "encoding_content_hash"
        ],
        "drop_missing": True,
    }
    assert len(training["target"]["encoding_content_hash"]) == 64
    splits = training["split_proof"]["splits"]
    assert [item["name"] for item in splits] == ["train", "test", "oot"]
    assert [item["partition"] for item in splits] == [
        "development",
        "validation",
        "oot",
    ]
    assert [item["membership_ref"]["mask_name"] for item in splits] == [
        "risk/development",
        "risk/validation",
        "risk/oot",
    ]
    assert all(item["bad_count"] + item["good_count"] == item["labeled_count"] for item in splits)
    assert all(
        item["selector_mask_content_hash"]
        == item["membership_mask_content_hash"]
        for item in splits
    )
    assert training["split_proof"]["selector_union_mask_content_hash"] == (
        training["split_proof"]["risk_membership_mask_content_hash"]
    )
    assert set(training["split_proof"]["pairwise_overlap_counts"].values()) == {
        0
    }


def test_freezes_complete_metrics_snapshot_hash_and_feature_importance():
    evidence = _evidence()
    snapshot = evidence["metrics_snapshot"]

    assert snapshot["values"]["test_auc"] == 0.73
    assert snapshot["values"]["train_rmse"] is None
    assert len(snapshot["content_hash"]) == 64
    assert evidence["feature_importance"] == [
        {"feature": "income", "importance": 0.7},
        {"feature": "age", "importance": 0.3},
    ]


def test_external_validator_rejects_bare_metrics_without_address_recompute():
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    evidence["metrics_snapshot"] = evidence["metrics_snapshot"]["values"]

    with pytest.raises(
        ModelingTrainingEvidenceError,
        match="metrics_snapshot fields are invalid",
    ):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=bundle,
        )


def test_builder_is_deterministic_across_mapping_and_importance_order():
    first = _evidence()
    config = _config(params={"num_leaves": 16, "learning_rate": 0.05})
    experiment = _experiment(config=config)
    artifact = _artifact(
        params={"learning_rate": 0.05, "num_leaves": 16},
        feature_importance=(("age", 0.3), ("income", 0.7)),
    )

    second = _evidence(experiment=experiment, artifact=artifact)

    assert second == first


def test_canonical_json_round_trip_is_byte_exact():
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    raw = canonical_modeling_training_evidence_json(
        evidence,
        sample_design_bundle=bundle,
    )

    assert modeling_training_evidence_from_json(
        raw,
        sample_design_bundle=bundle,
    ) == evidence
    assert json.loads(raw)["evidence_id"] == evidence["evidence_id"]
    with pytest.raises(ModelingTrainingEvidenceError, match="canonical encoding"):
        modeling_training_evidence_from_json(
            raw + "\n",
            sample_design_bundle=bundle,
        )


def test_parser_rejects_duplicate_keys_and_non_object_json():
    bundle = build_sample_design_v2_fixture()
    raw = canonical_modeling_training_evidence_json(
        _evidence(bundle=bundle),
        sample_design_bundle=bundle,
    )
    duplicated = raw.replace(
        '"artifact_kind":',
        '"artifact_kind":"ambiguous","artifact_kind":',
        1,
    )

    with pytest.raises(ModelingTrainingEvidenceError, match="duplicate key"):
        modeling_training_evidence_from_json(
            duplicated,
            sample_design_bundle=bundle,
        )
    with pytest.raises(ModelingTrainingEvidenceError, match="must contain an object"):
        modeling_training_evidence_from_json("[]", sample_design_bundle=bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evidence_id", "modeling-training-evidence-" + "0" * 24, "evidence_id"),
        ("content_hash", "0" * 64, "content_hash"),
    ],
)
def test_rejects_wrong_address_id_and_hash(field, value, message):
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    evidence[field] = value

    with pytest.raises(ModelingTrainingEvidenceError, match=message):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=bundle,
        )


def test_rejects_unknown_and_ambiguous_result_style_fields():
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    evidence["result"] = {"auc": 0.99, "ks": 0.88}

    with pytest.raises(ModelingTrainingEvidenceError, match="extra=.*result"):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=bundle,
        )


def test_rejects_cross_task_and_cross_experiment_bindings():
    with pytest.raises(ModelingTrainingEvidenceError, match="task_id does not match"):
        _evidence(experiment=_experiment(task_id="task-other"))

    with pytest.raises(ModelingTrainingEvidenceError, match="does not belong"):
        _evidence(artifact=_artifact(experiment_id="experiment-other"))

    with pytest.raises(ModelingTrainingEvidenceError, match="artifact_id"):
        _evidence(experiment=_experiment(artifact_id="artifact-other"))


def test_rejects_untrained_or_untyped_training_results():
    with pytest.raises(ModelingTrainingEvidenceError, match="trained experiment"):
        _evidence(experiment=_experiment(status="created"))

    with pytest.raises(ModelingTrainingEvidenceError, match="metrics must be ModelMetrics"):
        _evidence(experiment=_experiment(metrics=None))


def test_rejects_not_matured_or_exploration_only_samples():
    bundle = build_sample_design_v2_fixture(maturity_status="not_matured")

    with pytest.raises(ModelingTrainingEvidenceError, match="strategy_development|confirmed-matured"):
        _evidence(bundle=bundle)


@pytest.mark.parametrize(
    "bundle",
    [
        build_sample_design_v2_fixture(single_class_development=True),
        build_sample_design_v2_fixture(single_class_validation=True),
    ],
)
def test_rejects_single_class_train_or_test_samples(bundle):
    with pytest.raises(ModelingTrainingEvidenceError, match="both good and bad"):
        _evidence(bundle=bundle)


def test_rejects_empty_training_sample():
    empty = build_sample_design_v2_fixture(empty_development=True)
    with pytest.raises(ModelingTrainingEvidenceError, match="must not be empty"):
        _evidence(bundle=empty)


def test_split_mask_hashes_require_rowwise_equality_exclusivity_and_sample_binding():
    bundle = build_sample_design_v2_fixture()
    decoded = _decoded_membership()
    membership = {
        "train": decoded["masks"]["risk/development"].tolist(),
        "test": decoded["masks"]["risk/validation"].tolist(),
        "oot": decoded["masks"]["risk/oot"].tolist(),
    }
    risk_union = [
        any(values)
        for values in zip(*membership.values(), strict=True)
    ]
    selector = deepcopy(membership)
    selector["train"][0] = not selector["train"][0]
    with pytest.raises(ModelingTrainingEvidenceError, match="selector mask does not equal"):
        build_training_split_mask_hashes(
            sample_design_bundle=bundle,
            selector_masks=selector,
            membership_masks=membership,
            risk_membership_mask=risk_union,
        )

    overlapping = deepcopy(membership)
    overlapping["test"] = list(overlapping["train"])
    with pytest.raises(ModelingTrainingEvidenceError, match="pairwise exclusive|risk membership"):
        build_training_split_mask_hashes(
            sample_design_bundle=bundle,
            selector_masks=overlapping,
            membership_masks=overlapping,
            risk_membership_mask=risk_union,
        )

    alternate = build_sample_design_v2_fixture(empty_oot=True)
    alternate_hashes = _training_split_mask_hashes(alternate)
    with pytest.raises(ModelingTrainingEvidenceError, match="different sample design"):
        _evidence(
            bundle=bundle,
            training_split_mask_hashes=alternate_hashes,
        )


def test_split_mask_hash_contract_rejects_selector_hash_or_overlap_drift():
    bundle = build_sample_design_v2_fixture()
    hashes = _training_split_mask_hashes(bundle)
    hashes["splits"]["train"]["selector_mask_content_hash"] = _hash(
        "alternate-selector-mask"
    )
    with pytest.raises(ModelingTrainingEvidenceError, match="must be identical"):
        _evidence(bundle=bundle, training_split_mask_hashes=hashes)


def test_training_mask_budget_matches_sample_membership_v2_codec():
    assert MAX_TRAINING_MASK_BYTES == MAX_MEMBERSHIP_PAYLOAD_BYTES
    assert MAX_TRAINING_MASK_ROWS == (
        MAX_MEMBERSHIP_PAYLOAD_BYTES // len(MEMBERSHIP_MASK_ORDER)
    ) * 8


class _OversizedBooleanSequence(Sequence):
    def __len__(self):
        return MAX_TRAINING_MASK_ROWS + 1

    def __getitem__(self, index):
        raise AssertionError("oversized masks must be rejected before reading rows")


def test_training_mask_helper_rejects_oversized_input_before_conversion():
    oversized = _OversizedBooleanSequence()
    masks = {name: oversized for name in ("train", "test", "oot")}

    with pytest.raises(ModelingTrainingEvidenceError, match="row limit|byte budget"):
        build_training_split_mask_hashes(
            sample_design_bundle=build_sample_design_v2_fixture(),
            selector_masks=masks,
            membership_masks=masks,
            risk_membership_mask=oversized,
        )


def test_training_mask_helper_hashes_numpy_and_list_inputs_identically():
    bundle = build_sample_design_v2_fixture()
    decoded = _decoded_membership()
    numpy_masks = {
        "train": decoded["masks"]["risk/development"],
        "test": decoded["masks"]["risk/validation"],
        "oot": decoded["masks"]["risk/oot"],
    }
    risk_union = numpy_masks["train"] | numpy_masks["test"] | numpy_masks["oot"]

    numpy_result = build_training_split_mask_hashes(
        sample_design_bundle=bundle,
        selector_masks=numpy_masks,
        membership_masks=numpy_masks,
        risk_membership_mask=risk_union,
    )
    list_result = build_training_split_mask_hashes(
        sample_design_bundle=bundle,
        selector_masks={name: mask.tolist() for name, mask in numpy_masks.items()},
        membership_masks={name: mask.tolist() for name, mask in numpy_masks.items()},
        risk_membership_mask=risk_union.tolist(),
    )

    assert numpy_result == list_result

    hashes = _training_split_mask_hashes(bundle)
    hashes["pairwise_overlap_counts"]["train_test"] = 1
    with pytest.raises(ModelingTrainingEvidenceError, match="pairwise exclusive"):
        _evidence(bundle=bundle, training_split_mask_hashes=hashes)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(dataset_id="dataset-other"), "dataset_id"),
        (_config(target_col="other_target"), "target_col"),
        (_config(features=("income", "target")), "leak"),
        (_config(split_values={"train": "x", "test": "x", "oot": "z"}), "distinct"),
        (_config(split_values={"train": 1, "test": True, "oot": 2.0}), "canonical text"),
        (_config(split_values={"train": "", "test": "test", "oot": "oot"}), "canonical text"),
        (_config(target_type="continuous"), "target_type=binary"),
        (_config(recipe_id=None), "recipe_id"),
        (_config(drop_nan_labels=False), "drop_missing"),
    ],
)
def test_rejects_train_config_structural_contradictions(config, message):
    experiment = _experiment(config=config)

    with pytest.raises(ModelingTrainingEvidenceError, match=message):
        _evidence(experiment=experiment)


def test_rejects_ungoverned_sample_weight_column():
    config = _config(params={"sample_weight_col": "weight"})

    with pytest.raises(ModelingTrainingEvidenceError, match="governed.*weight field"):
        _evidence(experiment=_experiment(config=config))


def test_accepts_reversed_raw_target_with_explicit_bad_probability_encoding():
    bundle = _custom_bundle(good_value=1, bad_value=0)
    evidence = _evidence(bundle=bundle)
    target = evidence["training_contract"]["target"]

    assert target["good_value"] == 1
    assert target["bad_value"] == 0
    assert target["encoded_good_value"] == 0
    assert target["encoded_bad_value"] == 1
    assert target["encoding_rule"] == MODEL_TARGET_ENCODING_RULE
    assert len(target["encoding_content_hash"]) == 64
    assert (
        evidence["model_artifact"]["scoring_metadata"]["score_product"]
        == RAW_BAD_PROBABILITY_SCORE_PRODUCT
    )


def test_rejects_legacy_ambiguous_class_one_score_product():
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    evidence["model_artifact"]["scoring_metadata"]["score_product"] = (
        LEGACY_RAW_CLASS_ONE_SCORE_PRODUCT
    )

    with pytest.raises(
        ModelingTrainingEvidenceError,
        match="legacy.*class-one.*republish",
    ):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=bundle,
        )


def test_rejects_ensemble_and_calibrated_score_products():
    ensemble_config = _config(recipe_id="ensemble")
    with pytest.raises(ModelingTrainingEvidenceError, match="does not support ensemble"):
        _evidence(
            experiment=_experiment(
                recipe_id="ensemble",
                config=ensemble_config,
            ),
            artifact=_artifact(algorithm="ensemble"),
        )

    calibrated_config = _config(
        params={"calibration_method": "isotonic"}
    )
    with pytest.raises(ModelingTrainingEvidenceError, match="calibration parameter"):
        _evidence(experiment=_experiment(config=calibrated_config))

    with pytest.raises(ModelingTrainingEvidenceError, match="calibration parameter"):
        _evidence(
            artifact=_artifact(params={"calibration": {"method": "platt"}})
        )


@pytest.mark.parametrize(
    "algorithm",
    ["lgb_regressor", "lgb_multiclass", "unknown_binary_recipe", "ensemble"],
)
def test_rejects_algorithms_without_native_binary_class_one_probability(
    algorithm,
):
    config = _config(recipe_id=algorithm)

    with pytest.raises(
        ModelingTrainingEvidenceError,
        match="binary modeling recipe",
    ):
        _evidence(
            experiment=_experiment(recipe_id=algorithm, config=config),
            artifact=_artifact(algorithm=algorithm),
        )


@pytest.mark.parametrize("algorithm", sorted(BINARY_MODELING_RECIPES))
def test_accepts_every_authoritative_binary_modeling_recipe(algorithm):
    config = _config(recipe_id=algorithm)
    artifact_overrides = {"algorithm": algorithm}
    if algorithm == "scorecard":
        artifact_overrides["points_direction"] = "higher_is_better"

    evidence = _evidence(
        experiment=_experiment(recipe_id=algorithm, config=config),
        artifact=_artifact(**artifact_overrides),
    )

    assert evidence["model_artifact"]["algorithm"] == algorithm


def test_scorecard_woe_boundaries_are_tagged_strict_json_and_reversible():
    config = _config(recipe_id="scorecard")
    artifact = _artifact(
        algorithm="scorecard",
        points_direction="higher_is_better",
        woe_maps={
            "income": {
                "feature": "income",
                "edges": (float("-inf"), 0.0, float("inf")),
                "woe_by_bin": (-0.25, 0.4),
                "na_woe": 0.05,
            },
            "age": {
                "feature": "age",
                "edges": (float("-inf"), 35.0, float("inf")),
                "woe_by_bin": (0.3, -0.2),
                "na_woe": 0.0,
            },
        },
    )
    evidence = _evidence(
        experiment=_experiment(recipe_id="scorecard", config=config),
        artifact=artifact,
    )
    metadata = evidence["model_artifact"]["scoring_metadata"]
    income_edges = metadata["woe_maps"]["income"]["edges"]

    assert income_edges == [
        {NON_FINITE_BOUNDARY_TAG: "negative_infinity"},
        0.0,
        {NON_FINITE_BOUNDARY_TAG: "positive_infinity"},
    ]
    assert modeling_scoring_metadata_from_artifact(artifact) == metadata

    raw = canonical_modeling_training_evidence_json(
        evidence,
        sample_design_bundle=build_sample_design_v2_fixture(),
    )
    assert "Infinity" not in raw
    assert "-Infinity" not in raw
    assert (
        modeling_training_evidence_from_json(
            raw,
            sample_design_bundle=build_sample_design_v2_fixture(),
        )
        == evidence
    )

    decoded = decode_modeling_scoring_woe_maps_boundaries(
        metadata["woe_maps"]
    )
    assert np.isneginf(decoded["income"]["edges"][0])
    assert decoded["income"]["edges"][1] == 0.0
    assert np.isposinf(decoded["income"]["edges"][-1])
    assert decoded["income"]["woe_by_bin"] == [-0.25, 0.4]


@pytest.mark.parametrize(
    "woe_maps",
    [
        {
            "income": {
                "edges": (float("-inf"), float("nan"), float("inf")),
                "woe_by_bin": (0.1, 0.2),
            }
        },
        {
            "income": {
                "edges": (float("-inf"), 0.0, float("inf")),
                "woe_by_bin": (float("inf"), 0.2),
            }
        },
        {
            "income": {
                "edges": (
                    {
                        NON_FINITE_BOUNDARY_TAG: "not_a_supported_boundary",
                    },
                    0.0,
                    {NON_FINITE_BOUNDARY_TAG: "positive_infinity"},
                ),
                "woe_by_bin": (0.1, 0.2),
            }
        },
    ],
)
def test_scorecard_woe_boundary_tags_reject_ambiguous_non_finite_values(
    woe_maps,
):
    config = _config(recipe_id="scorecard")
    artifact = _artifact(
        algorithm="scorecard",
        points_direction="higher_is_better",
        woe_maps=woe_maps,
    )

    with pytest.raises(
        ModelingTrainingEvidenceError,
        match="unsupported non-finite|invalid numeric-boundary tag",
    ):
        _evidence(
            experiment=_experiment(recipe_id="scorecard", config=config),
            artifact=artifact,
        )


def test_rejects_non_raw_score_metadata_and_non_task_model_binary_ref():
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    evidence["model_artifact"]["scoring_metadata"]["score_product"] = (
        "calibrated_probability"
    )
    with pytest.raises(ModelingTrainingEvidenceError, match="raw native"):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=bundle,
        )

    wrong_binary_ref = {
        "artifact_id": _hash("model-binary-artifact-id"),
        "kind": "native_model_binary",
        "content_hash": _hash("native-model-bytes"),
        "model_artifact_id": _artifact().id,
    }
    with pytest.raises(ModelingTrainingEvidenceError, match="modeling_model_binary"):
        _evidence(bundle=bundle, model_binary_artifact_ref=wrong_binary_ref)

    invalid_binary_ref = {
        **wrong_binary_ref,
        "artifact_id": "not-a-task-artifact-id",
        "kind": "modeling_model_binary",
    }
    with pytest.raises(ModelingTrainingEvidenceError, match="SHA-256"):
        _evidence(bundle=bundle, model_binary_artifact_ref=invalid_binary_ref)


def test_rejects_nan_inf_and_invalid_feature_importance():
    with pytest.raises(ModelingTrainingEvidenceError, match="non-finite"):
        _evidence(
            experiment=_experiment(config=_config(params={"learning_rate": float("nan")}))
        )

    with pytest.raises(ModelingTrainingEvidenceError, match="finite number"):
        _evidence(experiment=_experiment(metrics=_metrics(train_auc=float("inf"))))

    with pytest.raises(ModelingTrainingEvidenceError, match="non-negative"):
        _evidence(artifact=_artifact(feature_importance=(("income", -0.1),)))

    with pytest.raises(ModelingTrainingEvidenceError, match="outside TrainConfig"):
        _evidence(artifact=_artifact(feature_importance=(("unknown", 0.1),)))

    with pytest.raises(ModelingTrainingEvidenceError, match="points_direction"):
        _evidence(artifact=_artifact(points_direction="higher_is_better"))


def test_rejects_wrong_sample_pair_kind_id_or_bundle_hash():
    bundle = build_sample_design_v2_fixture()
    refs = _sample_refs(bundle)
    wrong_kind = deepcopy(refs["membership_artifact_ref"])
    wrong_kind["kind"] = "strategy_sample_design_v2_json"
    with pytest.raises(ModelingTrainingEvidenceError, match="kind must be"):
        _evidence(bundle=bundle, membership_artifact_ref=wrong_kind)

    bad_id = deepcopy(refs["membership_artifact_ref"])
    bad_id["artifact_id"] = "membership-not-a-hash"
    with pytest.raises(ModelingTrainingEvidenceError, match="SHA-256"):
        _evidence(bundle=bundle, membership_artifact_ref=bad_id)

    wrong_hash = deepcopy(refs["sample_design_bundle_artifact_ref"])
    wrong_hash["content_hash"] = _hash("wrong-bundle")
    with pytest.raises(ModelingTrainingEvidenceError, match="canonical bundle bytes"):
        _evidence(bundle=bundle, sample_design_bundle_artifact_ref=wrong_hash)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("training_contract", "train_config_hash"), "0" * 64, "train_config_hash"),
        (("training_contract", "split_proof", "content_hash"), "0" * 64, "split_proof.content_hash"),
        (("metrics_snapshot", "content_hash"), "0" * 64, "metrics_snapshot.content_hash"),
        (
            ("model_artifact", "scoring_metadata_hash"),
            "0" * 64,
            "scoring_metadata_hash",
        ),
        (
            (
                "training_contract",
                "target",
                "encoding_content_hash",
            ),
            "0" * 64,
            "encoding_content_hash",
        ),
        (
            ("model_artifact", "model_binary_ref", "content_hash"),
            "0" * 64,
            "evidence_id|content_hash",
        ),
    ],
)
def test_rejects_nested_hash_drift(path, value, message):
    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ModelingTrainingEvidenceError, match=message):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=bundle,
        )


def test_rejects_label_drop_and_mask_summary_drift():
    with pytest.raises(ModelingTrainingEvidenceError, match="nan_labels_dropped"):
        _evidence(nan_labels_dropped=1)

    bundle = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=bundle)
    evidence["training_contract"]["split_proof"]["splits"][0][
        "membership_ref"
    ]["mask_name"] = "risk/validation"
    with pytest.raises(ModelingTrainingEvidenceError, match="fields are invalid|mask"):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=bundle,
        )


def test_rejects_metric_range_availability_and_internal_consistency():
    with pytest.raises(ModelingTrainingEvidenceError, match="within"):
        _evidence(experiment=_experiment(metrics=_metrics(test_auc=1.1)))

    with pytest.raises(ModelingTrainingEvidenceError, match="overfit_train_test_gap"):
        _evidence(
            experiment=_experiment(metrics=_metrics(overfit_train_test_gap=0.01))
        )

    with pytest.raises(ModelingTrainingEvidenceError, match="unweighted"):
        _evidence(
            experiment=_experiment(metrics=_metrics(weighted_train_auc=0.75))
        )


def test_accepts_real_degenerate_bootstrap_ks_ci_output():
    runtime_ci = bootstrap_ks_ci(
        np.asarray([0.1, 0.9], dtype=float),
        np.asarray([0, 1], dtype=int),
        seed=42,
    )
    test_ks = float(runtime_ci["ks"])
    metrics = _metrics(
        test_ks=test_ks,
        overfit_train_test_gap=abs(0.42 - test_ks) / 0.42,
        test_ks_ci_low=runtime_ci["ci_low"],
        test_ks_ci_high=runtime_ci["ci_high"],
        test_ks_ci_std=runtime_ci["std"],
        ks_ci_n_boot=runtime_ci["n_boot"],
    )

    evidence = _evidence(experiment=_experiment(metrics=metrics))
    values = evidence["metrics_snapshot"]["values"]
    assert values["ks_ci_n_boot"] == 0
    assert values["test_ks_ci_low"] == values["test_ks"]
    assert values["test_ks_ci_high"] == values["test_ks"]
    assert values["test_ks_ci_std"] == 0.0


def test_accepts_general_test_ks_ci_with_positive_bootstrap_count():
    positive_boot_ci = _metrics(
        test_ks_ci_low=0.30,
        test_ks_ci_high=0.44,
        test_ks_ci_std=0.03,
        ks_ci_n_boot=200,
    )
    evidence = _evidence(experiment=_experiment(metrics=positive_boot_ci))
    assert evidence["metrics_snapshot"]["values"]["ks_ci_n_boot"] == 200


@pytest.mark.parametrize(
    ("low", "high", "std"),
    [
        (0.30, 0.44, 0.0),
        (0.30, 0.30, 0.0),
        (0.38, 0.38, 0.01),
    ],
)
def test_rejects_fake_zero_bootstrap_test_ks_ci(low, high, std):
    metrics = _metrics(
        test_ks_ci_low=low,
        test_ks_ci_high=high,
        test_ks_ci_std=std,
        ks_ci_n_boot=0,
    )

    with pytest.raises(
        ModelingTrainingEvidenceError,
        match="zero-bootstrap test KS confidence interval",
    ):
        _evidence(experiment=_experiment(metrics=metrics))


def test_null_bootstrap_count_only_allows_absent_test_ks_ci():
    with pytest.raises(ModelingTrainingEvidenceError, match="must be positive"):
        _evidence(
            experiment=_experiment(
                metrics=_metrics(
                    test_ks_ci_low=0.30,
                    test_ks_ci_high=0.44,
                    test_ks_ci_std=0.03,
                    ks_ci_n_boot=None,
                )
            )
        )


def test_zero_bootstrap_count_only_allows_absent_test_ks_ci():
    evidence = _evidence(
        experiment=_experiment(metrics=_metrics(ks_ci_n_boot=0))
    )

    values = evidence["metrics_snapshot"]["values"]
    assert values["ks_ci_n_boot"] == 0
    assert values["test_ks_ci_low"] is None
    assert values["test_ks_ci_high"] is None
    assert values["test_ks_ci_std"] is None


def test_labeled_single_class_oot_keeps_psi_but_forbids_label_metrics():
    bundle = _custom_bundle(oot_mode="single_class")
    single_class_metrics = _metrics(
        oot_ks=None,
        oot_auc=None,
        overfit_train_oot_gap=None,
        overfit_flag=False,
    )

    evidence = _evidence(
        bundle=bundle,
        experiment=_experiment(metrics=single_class_metrics),
    )
    values = evidence["metrics_snapshot"]["values"]
    assert values["psi_oot_vs_train"] == 0.07
    assert values["oot_ks"] is None
    assert values["oot_auc"] is None
    assert values["oot_ks_ci_low"] is None
    assert values["oot_lift_head_10"] is None

    with pytest.raises(
        ModelingTrainingEvidenceError,
        match="availability.*OOT label support",
    ):
        _evidence(
            bundle=bundle,
            experiment=_experiment(
                metrics=replace(single_class_metrics, oot_auc=0.5)
            ),
        )


def test_empty_oot_allows_no_label_metrics_and_rejects_fake_ci_or_lift():
    bundle = build_sample_design_v2_fixture(empty_oot=True)
    empty_oot_metrics = _metrics(
        oot_ks=None,
        oot_auc=None,
        psi_oot_vs_train=None,
        overfit_train_oot_gap=None,
        overfit_flag=False,
    )

    evidence = _evidence(
        bundle=bundle,
        experiment=_experiment(metrics=empty_oot_metrics),
    )
    assert evidence["metrics_snapshot"]["values"]["oot_ks"] is None

    fake_ci = replace(
        empty_oot_metrics,
        oot_ks_ci_low=0.1,
        oot_ks_ci_high=0.3,
        oot_ks_ci_std=0.05,
    )
    with pytest.raises(ModelingTrainingEvidenceError, match="confidence intervals and lift"):
        _evidence(bundle=bundle, experiment=_experiment(metrics=fake_ci))

    fake_lift = replace(empty_oot_metrics, oot_lift_head_10=1.5)
    with pytest.raises(ModelingTrainingEvidenceError, match="confidence intervals and lift"):
        _evidence(bundle=bundle, experiment=_experiment(metrics=fake_lift))


def test_fully_unlabeled_oot_allows_score_psi_but_no_label_evidence():
    bundle = _custom_bundle(oot_mode="unlabeled")
    unlabeled_metrics = _metrics(
        oot_ks=None,
        oot_auc=None,
        overfit_train_oot_gap=None,
        overfit_flag=False,
    )

    evidence = _evidence(
        bundle=bundle,
        experiment=_experiment(metrics=unlabeled_metrics),
    )
    values = evidence["metrics_snapshot"]["values"]
    assert values["psi_oot_vs_train"] == 0.07
    assert values["oot_ks"] is None
    assert values["oot_lift_head_10"] is None

    fake_lift = replace(unlabeled_metrics, oot_lift_head_10=1.2)
    with pytest.raises(ModelingTrainingEvidenceError, match="confidence intervals and lift"):
        _evidence(bundle=bundle, experiment=_experiment(metrics=fake_lift))


def test_external_sample_bundle_must_match_every_bound_reference():
    original = build_sample_design_v2_fixture()
    evidence = _evidence(bundle=original)
    other = build_sample_design_v2_fixture(empty_oot=True)

    with pytest.raises(
        ModelingTrainingEvidenceError,
        match="different sample design|bundle artifact hash|derived",
    ):
        validate_modeling_training_evidence(
            evidence,
            sample_design_bundle=other,
        )
