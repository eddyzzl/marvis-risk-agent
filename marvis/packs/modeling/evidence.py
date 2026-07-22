"""Content-addressed training evidence for governed Strategy V2 consumers.

The contract freezes facts that were already produced by the modeling runtime.
It does not read a database or file, train a model, score rows, or calculate a
reported model metric.  A future Tool must still resolve the referenced
experiment, model artifact, SampleDesign V2 artifact pair, and model binary in
the same task and compare their live bytes/records with this snapshot.

The supplied StrategySampleDesign V2 bundle is deliberately an external trust
input to validation.  This keeps the evidence payload compact while preventing
a self-consistent JSON document from inventing maturity, sample membership, or
dataset provenance.  The Tool that publishes this contract must, before the
training transaction starts, load the verified dataset and decoded membership,
evaluate the three split selectors row by row, compare every selector mask with
its governed membership mask, and call ``build_training_split_mask_hashes``.
Caller-supplied hashes are never evidence of that comparison by themselves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import hmac
import json
import math
import re
from typing import Any

import numpy as np

from marvis.packs.modeling._common import BINARY_MODELING_RECIPES
from marvis.packs.modeling.contracts import (
    Experiment,
    ModelArtifact,
    ModelMetrics,
    TrainConfig,
)
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_membership import (
    MAX_MEMBERSHIP_PAYLOAD_BYTES,
    MEMBERSHIP_MASK_ORDER,
)
from marvis.packs.strategy.sample_design_v2 import (
    canonical_strategy_sample_design_v2_bundle_json,
    validate_strategy_sample_design_v2_bundle,
)


MODELING_TRAINING_EVIDENCE_SCHEMA_VERSION = "modeling.training-evidence.v2"
MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND = "modeling_training_evidence_json"
MODELING_TRAINING_EVIDENCE_PRODUCER_VERSION = (
    "marvis.modeling.training-evidence/2"
)
SAMPLE_MEMBERSHIP_ARTIFACT_KIND = "strategy_sample_membership_v2_binary"
SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND = "strategy_sample_design_v2_json"
MODEL_BINARY_REF_KIND = "modeling_model_binary"
TRAINING_MASK_HASH_ALGORITHM = "sha256-packed-bool-little-v1"
RAW_SCORE_PRODUCT = "raw_native_uncalibrated_probability_p_class_1"

MAX_TRAINING_EVIDENCE_JSON_BYTES = 4 * 1024 * 1024
MAX_TRAINING_EVIDENCE_JSON_DEPTH = 32
MAX_TRAINING_EVIDENCE_JSON_NODES = 100_000
MAX_TRAINING_FEATURES = 10_000
# Match the SampleMembership V2 codec: its byte budget covers six equally
# sized little-bit packed masks.  A row count above this value cannot belong to
# a valid task-owned V2 membership artifact and is rejected before conversion.
MAX_TRAINING_MASK_BYTES = MAX_MEMBERSHIP_PAYLOAD_BYTES
MAX_TRAINING_MASK_ROWS = (
    MAX_TRAINING_MASK_BYTES // len(MEMBERSHIP_MASK_ORDER)
) * 8

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_ID_RE = re.compile(
    r"^modeling-training-evidence-[0-9a-f]{24}$"
)

_OUTER_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "artifact_kind",
        "evidence_id",
        "task_id",
        "experiment",
        "model_artifact",
        "sample_design_binding",
        "training_contract",
        "metrics_snapshot",
        "feature_importance",
        "content_hash",
    }
)
_BODY_FIELDS = _OUTER_FIELDS - {"evidence_id", "content_hash"}
_EXPERIMENT_FIELDS = frozenset(
    {
        "experiment_id",
        "task_id",
        "recipe_id",
        "status",
        "artifact_id",
        "created_at",
    }
)
_MODEL_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "experiment_id",
        "algorithm",
        "created_at",
        "scoring_metadata",
        "scoring_metadata_hash",
        "model_binary_ref",
    }
)
_MODEL_BINARY_REF_FIELDS = frozenset(
    {"artifact_id", "kind", "content_hash", "model_artifact_id"}
)
_SCORING_METADATA_FIELDS = frozenset(
    {
        "feature_list",
        "params",
        "woe_maps",
        "scorecard_table",
        "score_direction",
        "points_direction",
        "score_product",
        "calibration_status",
    }
)
_TASK_ARTIFACT_REF_FIELDS = frozenset(
    {"artifact_id", "kind", "content_hash"}
)
_ARTIFACT_PAIR_FIELDS = frozenset({"membership", "bundle"})
_SAMPLE_BINDING_FIELDS = frozenset(
    {
        "task_id",
        "artifact_pair",
        "bundle_ref",
        "sample_design_ref",
        "membership_ref",
        "dataset_ref",
        "workspace_ref",
    }
)
_BUNDLE_REF_FIELDS = frozenset({"bundle_id", "content_hash"})
_DESIGN_REF_FIELDS = frozenset({"sample_design_id", "content_hash"})
_MEMBERSHIP_REF_FIELDS = frozenset(
    {"membership_id", "content_hash", "payload_hash"}
)
_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash", "role"})
_WORKSPACE_REF_FIELDS = frozenset(
    {"revision", "generation", "semantic_mapping_hash"}
)
_TRAINING_CONTRACT_FIELDS = frozenset(
    {
        "train_config",
        "train_config_hash",
        "features",
        "seed",
        "target",
        "split_proof",
        "label_handling",
        "weighting",
        "early_stopping_rounds",
    }
)
_TRAIN_CONFIG_FIELDS = frozenset(field.name for field in fields(TrainConfig))
_TARGET_FIELDS = frozenset(
    {"column", "good_value", "bad_value", "drop_missing"}
)
_LABEL_HANDLING_FIELDS = frozenset(
    {"drop_nan_labels", "nan_labels_dropped"}
)
_WEIGHTING_FIELDS = frozenset({"used", "column"})
_SPLIT_PROOF_FIELDS = frozenset(
    {
        "membership_id",
        "membership_content_hash",
        "membership_payload_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "row_ordinal",
        "mask_hash_algorithm",
        "selector_union_mask_content_hash",
        "membership_union_mask_content_hash",
        "risk_membership_mask_content_hash",
        "pairwise_overlap_counts",
        "splits",
        "content_hash",
    }
)
_TRAINING_SPLIT_MASK_HASH_FIELDS = frozenset(
    {
        "membership_id",
        "membership_content_hash",
        "membership_payload_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "row_count",
        "mask_hash_algorithm",
        "selector_union_mask_content_hash",
        "membership_union_mask_content_hash",
        "risk_membership_mask_content_hash",
        "pairwise_overlap_counts",
        "splits",
    }
)
_TRAINING_SPLIT_MASK_ITEMS = frozenset({"train", "test", "oot"})
_TRAINING_SPLIT_MASK_ITEM_FIELDS = frozenset(
    {
        "selector_mask_content_hash",
        "membership_mask_content_hash",
        "selected_count",
    }
)
_PAIRWISE_OVERLAP_FIELDS = frozenset(
    {"train_test", "train_oot", "test_oot"}
)
_ROW_ORDINAL_FIELDS = frozenset({"start", "stop", "step"})
_SPLIT_FIELDS = frozenset(
    {
        "name",
        "population",
        "partition",
        "split_value",
        "membership_ref",
        "row_count",
        "labeled_count",
        "bad_count",
        "good_count",
        "selector_mask_content_hash",
        "membership_mask_content_hash",
    }
)
_PARTITION_MEMBERSHIP_REF_FIELDS = frozenset(
    {"membership_id", "membership_content_hash", "mask_name"}
)
_METRICS_SNAPSHOT_FIELDS = frozenset({"values", "content_hash"})
_MODEL_METRIC_FIELDS = frozenset(field.name for field in fields(ModelMetrics))
_FEATURE_IMPORTANCE_FIELDS = frozenset({"feature", "importance"})

_SPLIT_PARTITIONS = (
    ("train", "development"),
    ("test", "validation"),
    ("oot", "oot"),
)

_RATIO_METRICS = frozenset(
    {
        "train_ks",
        "test_ks",
        "oot_ks",
        "train_auc",
        "test_auc",
        "oot_auc",
        "weighted_train_ks",
        "weighted_test_ks",
        "weighted_oot_ks",
        "weighted_train_auc",
        "weighted_test_auc",
        "weighted_oot_auc",
        "test_ks_ci_low",
        "test_ks_ci_high",
        "oot_ks_ci_low",
        "oot_ks_ci_high",
        "train_macro_auc",
        "test_macro_auc",
        "oot_macro_auc",
        "train_accuracy",
        "test_accuracy",
        "oot_accuracy",
    }
)
_NON_NEGATIVE_METRICS = frozenset(
    {
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "weighted_psi_test_vs_train",
        "weighted_psi_oot_vs_train",
        "test_ks_ci_std",
        "oot_ks_ci_std",
        "test_lift_head_5",
        "test_lift_tail_5",
        "test_lift_head_10",
        "test_lift_tail_10",
        "oot_lift_head_5",
        "oot_lift_tail_5",
        "oot_lift_head_10",
        "oot_lift_tail_10",
        "train_rmse",
        "test_rmse",
        "oot_rmse",
        "train_mae",
        "test_mae",
        "oot_mae",
        "train_logloss",
        "test_logloss",
        "oot_logloss",
    }
)
_WEIGHTED_METRICS = frozenset(
    {
        "weighted_train_ks",
        "weighted_test_ks",
        "weighted_oot_ks",
        "weighted_train_auc",
        "weighted_test_auc",
        "weighted_oot_auc",
        "weighted_psi_test_vs_train",
        "weighted_psi_oot_vs_train",
    }
)
_NON_BINARY_METRICS = frozenset(
    {
        "train_rmse",
        "test_rmse",
        "oot_rmse",
        "train_mae",
        "test_mae",
        "oot_mae",
        "train_r2",
        "test_r2",
        "oot_r2",
        "train_macro_auc",
        "test_macro_auc",
        "oot_macro_auc",
        "train_logloss",
        "test_logloss",
        "oot_logloss",
        "train_accuracy",
        "test_accuracy",
        "oot_accuracy",
    }
)


class ModelingTrainingEvidenceError(ModelingError):
    """Training evidence violates the exact Strategy V2 structural contract."""


def build_task_artifact_ref(
    *, artifact_id: str, kind: str, content_hash: str
) -> dict[str, str]:
    """Build a strict task-artifact pointer; existence is verified by a Tool."""

    return _task_artifact_ref(
        {
            "artifact_id": artifact_id,
            "kind": kind,
            "content_hash": content_hash,
        },
        name="task artifact ref",
        expected_kind=kind,
    )


def build_model_binary_artifact_ref(
    *,
    artifact_id: str,
    model_artifact_id: str,
    content_hash: str,
) -> dict[str, str]:
    """Build the task-owned native model-binary pointer used by this contract."""

    return _model_binary_ref(
        {
            "artifact_id": artifact_id,
            "kind": MODEL_BINARY_REF_KIND,
            "content_hash": content_hash,
            "model_artifact_id": model_artifact_id,
        }
    )


def build_training_split_mask_hashes(
    *,
    sample_design_bundle: Mapping[str, Any],
    selector_masks: Mapping[str, object],
    membership_masks: Mapping[str, object],
    risk_membership_mask: object,
) -> dict[str, Any]:
    """Hash masks only after proving selector/member equality and conservation.

    This helper is pure.  The publishing Tool owns the trust boundary: it must
    obtain ``selector_masks`` from selectors evaluated against the verified
    training dataset and ``membership_masks`` from the independently decoded
    task-owned membership artifact.  Passing caller-provided masks or hashes to
    this helper does not establish provenance.
    """

    context = _sample_context(sample_design_bundle)
    expected_row_count = context["bundle"]["membership"]["row_count"]
    _validate_training_mask_row_budget(expected_row_count)
    selector = _three_boolean_masks(
        selector_masks,
        "selector_masks",
        expected_row_count=expected_row_count,
    )
    membership = _three_boolean_masks(
        membership_masks,
        "membership_masks",
        expected_row_count=expected_row_count,
    )
    risk = _boolean_mask(
        risk_membership_mask,
        "risk_membership_mask",
        expected_row_count=expected_row_count,
    )
    row_count = int(risk.size)
    if row_count == 0:
        raise ModelingTrainingEvidenceError("training masks must not be empty")
    for name in ("train", "test", "oot"):
        if not np.array_equal(selector[name], membership[name]):
            raise ModelingTrainingEvidenceError(
                f"{name} selector mask does not equal governed membership mask"
            )
    overlaps = {
        "train_test": _mask_overlap_count(selector["train"], selector["test"]),
        "train_oot": _mask_overlap_count(selector["train"], selector["oot"]),
        "test_oot": _mask_overlap_count(selector["test"], selector["oot"]),
    }
    if any(overlaps.values()):
        raise ModelingTrainingEvidenceError(
            "training selector masks must be pairwise exclusive"
        )
    selector_union = _mask_union(
        selector["train"], selector["test"], selector["oot"]
    )
    membership_union = _mask_union(
        membership["train"], membership["test"], membership["oot"]
    )
    if not np.array_equal(
        selector_union, membership_union
    ) or not np.array_equal(selector_union, risk):
        raise ModelingTrainingEvidenceError(
            "training split union does not equal governed risk membership"
        )
    bundle = context["bundle"]
    header = bundle["membership"]
    design = bundle["sample_design"]
    return {
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "membership_payload_hash": header["payload_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "row_count": row_count,
        "mask_hash_algorithm": TRAINING_MASK_HASH_ALGORITHM,
        "selector_union_mask_content_hash": _boolean_mask_content_hash(
            selector_union
        ),
        "membership_union_mask_content_hash": _boolean_mask_content_hash(
            membership_union
        ),
        "risk_membership_mask_content_hash": _boolean_mask_content_hash(risk),
        "pairwise_overlap_counts": overlaps,
        "splits": {
            name: {
                "selector_mask_content_hash": _boolean_mask_content_hash(
                    selector[name]
                ),
                "membership_mask_content_hash": _boolean_mask_content_hash(
                    membership[name]
                ),
                "selected_count": int(np.count_nonzero(membership[name])),
            }
            for name in ("train", "test", "oot")
        },
    }


def build_modeling_training_evidence(
    *,
    experiment: Experiment,
    model_artifact: ModelArtifact,
    sample_design_bundle: Mapping[str, Any],
    membership_artifact_ref: Mapping[str, Any],
    sample_design_bundle_artifact_ref: Mapping[str, Any],
    model_binary_artifact_ref: Mapping[str, Any],
    training_split_mask_hashes: Mapping[str, Any],
    nan_labels_dropped: int,
    producer_version: str = MODELING_TRAINING_EVIDENCE_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Freeze one trained binary model for later Strategy V2 consumption."""

    if not isinstance(experiment, Experiment):
        raise ModelingTrainingEvidenceError("experiment must be an Experiment")
    if not isinstance(model_artifact, ModelArtifact):
        raise ModelingTrainingEvidenceError(
            "model_artifact must be a ModelArtifact"
        )
    if not isinstance(experiment.config, TrainConfig):
        raise ModelingTrainingEvidenceError(
            "experiment.config must be a TrainConfig"
        )
    if not isinstance(experiment.metrics, ModelMetrics):
        raise ModelingTrainingEvidenceError(
            "trained experiment.metrics must be ModelMetrics"
        )

    config = _train_config(asdict(experiment.config))
    scoring_metadata = _scoring_metadata_from_model_artifact(model_artifact)
    body = {
        "schema_version": MODELING_TRAINING_EVIDENCE_SCHEMA_VERSION,
        "producer_version": producer_version,
        "artifact_kind": MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        "task_id": experiment.task_id,
        "experiment": {
            "experiment_id": experiment.id,
            "task_id": experiment.task_id,
            "recipe_id": experiment.recipe_id,
            "status": experiment.status,
            "artifact_id": experiment.artifact_id,
            "created_at": experiment.created_at,
        },
        "model_artifact": {
            "artifact_id": model_artifact.id,
            "experiment_id": model_artifact.experiment_id,
            "algorithm": model_artifact.algorithm,
            "created_at": model_artifact.created_at,
            "scoring_metadata": scoring_metadata,
            "scoring_metadata_hash": _sha256(
                _canonical_json(scoring_metadata)
            ),
            "model_binary_ref": model_binary_artifact_ref,
        },
        "sample_design_binding": _expected_sample_binding(
            _sample_context(sample_design_bundle),
            membership_artifact_ref=membership_artifact_ref,
            bundle_artifact_ref=sample_design_bundle_artifact_ref,
        ),
        "training_contract": _build_training_contract(
            config=config,
            sample_context=_sample_context(sample_design_bundle),
            training_split_mask_hashes=training_split_mask_hashes,
            nan_labels_dropped=nan_labels_dropped,
        ),
        "metrics_snapshot": _build_metrics_snapshot(asdict(experiment.metrics)),
        "feature_importance": [
            {"feature": feature, "importance": importance}
            for feature, importance in model_artifact.feature_importance
        ],
    }
    normalized = _normalize_body(body, sample_design_bundle=sample_design_bundle)
    return validate_modeling_training_evidence(
        _address(normalized),
        sample_design_bundle=sample_design_bundle,
    )


def validate_modeling_training_evidence(
    payload: Mapping[str, Any],
    *,
    sample_design_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate evidence against the independently authenticated sample bundle."""

    obj = _object(payload, "modeling training evidence")
    _preflight_json_tree(obj, name="modeling training evidence")
    _exact_fields(obj, _OUTER_FIELDS, "modeling training evidence")
    normalized_body = _normalize_body(
        {key: obj[key] for key in obj if key not in {"evidence_id", "content_hash"}},
        sample_design_bundle=sample_design_bundle,
    )
    normalized = _validate_addressed(obj, normalized_body)
    if len(_canonical_json(normalized).encode("utf-8")) > MAX_TRAINING_EVIDENCE_JSON_BYTES:
        raise ModelingTrainingEvidenceError(
            "modeling training evidence exceeds byte budget"
        )
    return normalized


def canonical_modeling_training_evidence_json(
    payload: Mapping[str, Any],
    *,
    sample_design_bundle: Mapping[str, Any],
) -> str:
    """Return the sole canonical JSON encoding of valid training evidence."""

    return _canonical_json(
        validate_modeling_training_evidence(
            payload,
            sample_design_bundle=sample_design_bundle,
        )
    )


def modeling_training_evidence_from_json(
    raw: str | bytes | bytearray,
    *,
    sample_design_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse canonical evidence JSON while rejecting duplicate keys and drift."""

    if not isinstance(raw, (str, bytes, bytearray)):
        raise ModelingTrainingEvidenceError(
            "modeling training evidence JSON must be text or bytes"
        )
    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if len(encoded) > MAX_TRAINING_EVIDENCE_JSON_BYTES:
        raise ModelingTrainingEvidenceError(
            "modeling training evidence JSON exceeds byte budget"
        )
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except ModelingTrainingEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise ModelingTrainingEvidenceError(
            "modeling training evidence is not valid bounded JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelingTrainingEvidenceError(
            "modeling training evidence JSON must contain an object"
        )
    normalized = validate_modeling_training_evidence(
        payload,
        sample_design_bundle=sample_design_bundle,
    )
    if encoded != _canonical_json(normalized).encode("utf-8"):
        raise ModelingTrainingEvidenceError(
            "modeling training evidence JSON must use canonical encoding"
        )
    return normalized


def _normalize_body(
    value: object,
    *,
    sample_design_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    obj = _object(value, "modeling training evidence body")
    _exact_fields(obj, _BODY_FIELDS, "modeling training evidence body")
    if obj["schema_version"] != MODELING_TRAINING_EVIDENCE_SCHEMA_VERSION:
        raise ModelingTrainingEvidenceError("schema_version is invalid")
    if obj["artifact_kind"] != MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND:
        raise ModelingTrainingEvidenceError(
            "artifact_kind must be modeling_training_evidence_json"
        )
    producer = _text(obj["producer_version"], "producer_version")
    context = _sample_context(sample_design_bundle)
    task_id = _text(obj["task_id"], "task_id")
    if task_id != context["task_id"]:
        raise ModelingTrainingEvidenceError(
            "evidence task_id does not match StrategySampleDesign V2"
        )

    experiment = _experiment(obj["experiment"])
    if experiment["task_id"] != task_id:
        raise ModelingTrainingEvidenceError(
            "experiment belongs to a different task"
        )
    if experiment["status"] != "trained":
        raise ModelingTrainingEvidenceError(
            "training evidence requires a trained experiment"
        )

    training = _training_contract(
        obj["training_contract"],
        sample_context=context,
    )
    config = training["train_config"]
    if config["recipe_id"] != experiment["recipe_id"]:
        raise ModelingTrainingEvidenceError(
            "TrainConfig recipe_id does not match experiment recipe_id"
        )

    model = _model_artifact(obj["model_artifact"], config=config)
    if model["experiment_id"] != experiment["experiment_id"]:
        raise ModelingTrainingEvidenceError(
            "model artifact does not belong to the experiment"
        )
    if model["artifact_id"] != experiment["artifact_id"]:
        raise ModelingTrainingEvidenceError(
            "experiment artifact_id does not match model artifact"
        )
    if model["algorithm"] != experiment["recipe_id"]:
        raise ModelingTrainingEvidenceError(
            "model artifact algorithm does not match experiment recipe"
        )

    binding_obj = _object(obj["sample_design_binding"], "sample_design_binding")
    _exact_fields(binding_obj, _SAMPLE_BINDING_FIELDS, "sample_design_binding")
    pair = _artifact_pair(binding_obj["artifact_pair"])
    expected_binding = _expected_sample_binding(
        context,
        membership_artifact_ref=pair["membership"],
        bundle_artifact_ref=pair["bundle"],
    )
    normalized_binding = _sample_binding(binding_obj)
    if normalized_binding != expected_binding:
        raise ModelingTrainingEvidenceError(
            "sample_design_binding is not derived from the supplied V2 artifact pair"
        )
    registered_ids = {
        normalized_binding["artifact_pair"]["membership"]["artifact_id"],
        normalized_binding["artifact_pair"]["bundle"]["artifact_id"],
        model["model_binary_ref"]["artifact_id"],
    }
    if len(registered_ids) != 3:
        raise ModelingTrainingEvidenceError(
            "model binary and sample artifact refs must be distinct task artifacts"
        )

    metrics = _metrics_snapshot(obj["metrics_snapshot"])
    _validate_binary_metrics(
        metrics["values"],
        split_proof=training["split_proof"],
        weighting=training["weighting"],
    )
    importance = _feature_importance(
        obj["feature_importance"],
        features=config["features"],
    )

    return {
        "schema_version": MODELING_TRAINING_EVIDENCE_SCHEMA_VERSION,
        "producer_version": producer,
        "artifact_kind": MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        "task_id": task_id,
        "experiment": experiment,
        "model_artifact": model,
        "sample_design_binding": normalized_binding,
        "training_contract": training,
        "metrics_snapshot": metrics,
        "feature_importance": importance,
    }


def _experiment(value: object) -> dict[str, str]:
    obj = _object(value, "experiment")
    _exact_fields(obj, _EXPERIMENT_FIELDS, "experiment")
    artifact_id = _text(obj["artifact_id"], "experiment.artifact_id")
    return {
        "experiment_id": _text(obj["experiment_id"], "experiment.experiment_id"),
        "task_id": _text(obj["task_id"], "experiment.task_id"),
        "recipe_id": _text(obj["recipe_id"], "experiment.recipe_id"),
        "status": _text(obj["status"], "experiment.status"),
        "artifact_id": artifact_id,
        "created_at": _text(obj["created_at"], "experiment.created_at"),
    }


def _model_artifact(
    value: object,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    obj = _object(value, "model_artifact")
    _exact_fields(obj, _MODEL_ARTIFACT_FIELDS, "model_artifact")
    artifact_id = _text(obj["artifact_id"], "model_artifact.artifact_id")
    algorithm = _text(obj["algorithm"], "model_artifact.algorithm")
    _require_binary_modeling_recipe(algorithm, name="model_artifact.algorithm")
    scoring_metadata = _scoring_metadata(
        obj["scoring_metadata"],
        algorithm=algorithm,
    )
    if scoring_metadata["feature_list"] != config["features"]:
        raise ModelingTrainingEvidenceError(
            "model scoring feature_list does not match TrainConfig features"
        )
    scoring_metadata_hash = _hash(
        obj["scoring_metadata_hash"],
        "model_artifact.scoring_metadata_hash",
    )
    if not hmac.compare_digest(
        scoring_metadata_hash,
        _sha256(_canonical_json(scoring_metadata)),
    ):
        raise ModelingTrainingEvidenceError(
            "model_artifact.scoring_metadata_hash does not match metadata"
        )
    model_binary = _model_binary_ref(obj["model_binary_ref"])
    if model_binary["model_artifact_id"] != artifact_id:
        raise ModelingTrainingEvidenceError(
            "model binary ref does not match model artifact"
        )
    return {
        "artifact_id": artifact_id,
        "experiment_id": _text(
            obj["experiment_id"], "model_artifact.experiment_id"
        ),
        "algorithm": algorithm,
        "created_at": _text(obj["created_at"], "model_artifact.created_at"),
        "scoring_metadata": scoring_metadata,
        "scoring_metadata_hash": scoring_metadata_hash,
        "model_binary_ref": model_binary,
    }


def _scoring_metadata_from_model_artifact(
    model_artifact: ModelArtifact,
) -> dict[str, Any]:
    return _scoring_metadata(
        {
            "feature_list": list(model_artifact.feature_list),
            "params": model_artifact.params,
            "woe_maps": model_artifact.woe_maps,
            "scorecard_table": list(model_artifact.scorecard_table),
            "score_direction": model_artifact.score_direction,
            "points_direction": model_artifact.points_direction,
            "score_product": RAW_SCORE_PRODUCT,
            "calibration_status": "not_applied",
        },
        algorithm=model_artifact.algorithm,
    )


def _scoring_metadata(
    value: object,
    *,
    algorithm: str,
) -> dict[str, Any]:
    obj = _object(value, "model scoring_metadata")
    _exact_fields(obj, _SCORING_METADATA_FIELDS, "model scoring_metadata")
    features = _text_array(
        obj["feature_list"],
        "model scoring_metadata.feature_list",
        required=True,
    )
    if len(features) != len(set(features)):
        raise ModelingTrainingEvidenceError(
            "model scoring feature_list contains duplicates"
        )
    params_obj = _object(obj["params"], "model scoring_metadata.params")
    params = _json_value(params_obj, "model scoring_metadata.params")
    assert isinstance(params, dict)
    _reject_calibration_parameters(params, "model scoring_metadata.params")
    woe_maps = (
        None
        if obj["woe_maps"] is None
        else _json_value(
            _object(obj["woe_maps"], "model scoring_metadata.woe_maps"),
            "model scoring_metadata.woe_maps",
        )
    )
    scorecard_table = _json_value(
        _array(
            obj["scorecard_table"],
            "model scoring_metadata.scorecard_table",
            required=False,
        ),
        "model scoring_metadata.scorecard_table",
    )
    assert isinstance(scorecard_table, list)
    score_direction = _text(
        obj["score_direction"], "model scoring_metadata.score_direction"
    )
    if score_direction != "higher_is_riskier":
        raise ModelingTrainingEvidenceError(
            "binary Strategy V2 evidence requires higher_is_riskier score direction"
        )
    points_direction = _optional_text(
        obj["points_direction"], "model scoring_metadata.points_direction"
    )
    expected_points_direction = (
        "higher_is_better" if algorithm == "scorecard" else None
    )
    if points_direction != expected_points_direction:
        raise ModelingTrainingEvidenceError(
            "model scoring points_direction contradicts its algorithm"
        )
    if algorithm != "scorecard" and scorecard_table:
        raise ModelingTrainingEvidenceError(
            "non-scorecard model cannot carry a scorecard_table"
        )
    if obj["score_product"] != RAW_SCORE_PRODUCT:
        raise ModelingTrainingEvidenceError(
            "first training-evidence vertical requires raw native P(class=1) score"
        )
    if obj["calibration_status"] != "not_applied":
        raise ModelingTrainingEvidenceError(
            "calibrated score products are not supported in the first vertical"
        )
    return {
        "feature_list": features,
        "params": params,
        "woe_maps": woe_maps,
        "scorecard_table": scorecard_table,
        "score_direction": score_direction,
        "points_direction": points_direction,
        "score_product": RAW_SCORE_PRODUCT,
        "calibration_status": "not_applied",
    }


def _model_binary_ref(value: object) -> dict[str, str]:
    obj = _object(value, "model_binary_ref")
    _exact_fields(obj, _MODEL_BINARY_REF_FIELDS, "model_binary_ref")
    if obj["kind"] != MODEL_BINARY_REF_KIND:
        raise ModelingTrainingEvidenceError(
            "model_binary_ref.kind must be modeling_model_binary"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"], "model_binary_ref.artifact_id"
        ),
        "kind": MODEL_BINARY_REF_KIND,
        "content_hash": _hash(
            obj["content_hash"], "model_binary_ref.content_hash"
        ),
        "model_artifact_id": _text(
            obj["model_artifact_id"], "model_binary_ref.model_artifact_id"
        ),
    }


def _sample_context(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        bundle = validate_strategy_sample_design_v2_bundle(value)
    except StrategyError as exc:
        raise ModelingTrainingEvidenceError(
            "sample_design_bundle failed strict StrategySampleDesign V2 validation"
        ) from exc
    design = bundle["sample_design"]
    identity = design["identity"]
    risk = next(item for item in bundle["populations"] if item["role"] == "risk")
    if design["sample_semantics"]["scope"] != "strategy_development":
        raise ModelingTrainingEvidenceError(
            "training evidence requires strategy_development sample scope"
        )
    if risk["maturity_evidence"]["status"] != "confirmed_matured":
        raise ModelingTrainingEvidenceError(
            "training evidence requires confirmed-matured risk samples"
        )
    if design["target_selector"]["status"] != "resolved":
        raise ModelingTrainingEvidenceError(
            "training evidence requires a resolved binary target"
        )
    if design["sample_semantics"]["split_definition"]["status"] != "available":
        raise ModelingTrainingEvidenceError(
            "training evidence requires an available split definition"
        )
    return {
        "bundle": bundle,
        "task_id": identity["task_id"],
        "dataset_ref": identity["dataset_ref"],
        "workspace_ref": identity["workspace_ref"],
        "target": design["target_selector"],
        "field_bindings": design["sample_semantics"]["field_bindings"],
        "risk": risk,
    }


def _expected_sample_binding(
    context: Mapping[str, Any],
    *,
    membership_artifact_ref: Mapping[str, Any],
    bundle_artifact_ref: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = context["bundle"]
    design = bundle["sample_design"]
    header = bundle["membership"]
    membership_artifact = _task_artifact_ref(
        membership_artifact_ref,
        name="sample artifact pair membership",
        expected_kind=SAMPLE_MEMBERSHIP_ARTIFACT_KIND,
    )
    bundle_artifact = _task_artifact_ref(
        bundle_artifact_ref,
        name="sample artifact pair bundle",
        expected_kind=SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND,
    )
    expected_bundle_file_hash = _sha256(
        canonical_strategy_sample_design_v2_bundle_json(bundle)
    )
    if not hmac.compare_digest(
        bundle_artifact["content_hash"], expected_bundle_file_hash
    ):
        raise ModelingTrainingEvidenceError(
            "sample-design bundle artifact hash does not match canonical bundle bytes"
        )
    if (
        membership_artifact["artifact_id"] == bundle_artifact["artifact_id"]
        or membership_artifact["content_hash"] == bundle_artifact["content_hash"]
    ):
        raise ModelingTrainingEvidenceError(
            "sample-design membership and bundle artifact refs are interchangeable"
        )
    return {
        "task_id": context["task_id"],
        "artifact_pair": {
            "membership": membership_artifact,
            "bundle": bundle_artifact,
        },
        "bundle_ref": {
            "bundle_id": bundle["bundle_id"],
            "content_hash": bundle["content_hash"],
        },
        "sample_design_ref": {
            "sample_design_id": design["sample_design_id"],
            "content_hash": design["content_hash"],
        },
        "membership_ref": {
            "membership_id": header["membership_id"],
            "content_hash": header["content_hash"],
            "payload_hash": header["payload_hash"],
        },
        "dataset_ref": dict(context["dataset_ref"]),
        "workspace_ref": dict(context["workspace_ref"]),
    }


def _sample_binding(value: object) -> dict[str, Any]:
    obj = _object(value, "sample_design_binding")
    _exact_fields(obj, _SAMPLE_BINDING_FIELDS, "sample_design_binding")
    pair = _artifact_pair(obj["artifact_pair"])
    bundle_ref = _object(obj["bundle_ref"], "sample_design_binding.bundle_ref")
    _exact_fields(bundle_ref, _BUNDLE_REF_FIELDS, "sample_design_binding.bundle_ref")
    design_ref = _object(
        obj["sample_design_ref"], "sample_design_binding.sample_design_ref"
    )
    _exact_fields(
        design_ref,
        _DESIGN_REF_FIELDS,
        "sample_design_binding.sample_design_ref",
    )
    membership_ref = _object(
        obj["membership_ref"], "sample_design_binding.membership_ref"
    )
    _exact_fields(
        membership_ref,
        _MEMBERSHIP_REF_FIELDS,
        "sample_design_binding.membership_ref",
    )
    dataset_ref = _dataset_ref(obj["dataset_ref"])
    workspace_ref = _workspace_ref(obj["workspace_ref"])
    return {
        "task_id": _text(obj["task_id"], "sample_design_binding.task_id"),
        "artifact_pair": pair,
        "bundle_ref": {
            "bundle_id": _text(
                bundle_ref["bundle_id"], "sample_design_binding.bundle_ref.bundle_id"
            ),
            "content_hash": _hash(
                bundle_ref["content_hash"],
                "sample_design_binding.bundle_ref.content_hash",
            ),
        },
        "sample_design_ref": {
            "sample_design_id": _text(
                design_ref["sample_design_id"],
                "sample_design_binding.sample_design_ref.sample_design_id",
            ),
            "content_hash": _hash(
                design_ref["content_hash"],
                "sample_design_binding.sample_design_ref.content_hash",
            ),
        },
        "membership_ref": {
            "membership_id": _text(
                membership_ref["membership_id"],
                "sample_design_binding.membership_ref.membership_id",
            ),
            "content_hash": _hash(
                membership_ref["content_hash"],
                "sample_design_binding.membership_ref.content_hash",
            ),
            "payload_hash": _hash(
                membership_ref["payload_hash"],
                "sample_design_binding.membership_ref.payload_hash",
            ),
        },
        "dataset_ref": dataset_ref,
        "workspace_ref": workspace_ref,
    }


def _artifact_pair(value: object) -> dict[str, dict[str, str]]:
    obj = _object(value, "sample artifact pair")
    _exact_fields(obj, _ARTIFACT_PAIR_FIELDS, "sample artifact pair")
    return {
        "membership": _task_artifact_ref(
            obj["membership"],
            name="sample artifact pair membership",
            expected_kind=SAMPLE_MEMBERSHIP_ARTIFACT_KIND,
        ),
        "bundle": _task_artifact_ref(
            obj["bundle"],
            name="sample artifact pair bundle",
            expected_kind=SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND,
        ),
    }


def _task_artifact_ref(
    value: object,
    *,
    name: str,
    expected_kind: str,
) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, _TASK_ARTIFACT_REF_FIELDS, name)
    artifact_id = _hash(obj["artifact_id"], f"{name}.artifact_id")
    kind = _text(obj["kind"], f"{name}.kind")
    if kind != expected_kind:
        raise ModelingTrainingEvidenceError(
            f"{name}.kind must be {expected_kind}"
        )
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "content_hash": _hash(obj["content_hash"], f"{name}.content_hash"),
    }


def _dataset_ref(value: object) -> dict[str, str]:
    obj = _object(value, "dataset_ref")
    _exact_fields(obj, _DATASET_REF_FIELDS, "dataset_ref")
    if obj["role"] != "active":
        raise ModelingTrainingEvidenceError("dataset_ref.role must be active")
    return {
        "dataset_id": _text(obj["dataset_id"], "dataset_ref.dataset_id"),
        "content_hash": _hash(obj["content_hash"], "dataset_ref.content_hash"),
        "role": "active",
    }


def _workspace_ref(value: object) -> dict[str, Any]:
    obj = _object(value, "workspace_ref")
    _exact_fields(obj, _WORKSPACE_REF_FIELDS, "workspace_ref")
    return {
        "revision": _non_negative_int(obj["revision"], "workspace_ref.revision"),
        "generation": _non_negative_int(
            obj["generation"], "workspace_ref.generation"
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"], "workspace_ref.semantic_mapping_hash"
        ),
    }


def _build_training_contract(
    *,
    config: Mapping[str, Any],
    sample_context: Mapping[str, Any],
    training_split_mask_hashes: Mapping[str, Any],
    nan_labels_dropped: int,
) -> dict[str, Any]:
    weighting = _weighting_from_config(config, sample_context=sample_context)
    target = _target_from_context(sample_context)
    split_proof = _expected_split_proof(
        config,
        sample_context=sample_context,
        training_split_mask_hashes=training_split_mask_hashes,
    )
    return {
        "train_config": dict(config),
        "train_config_hash": _sha256(_canonical_json(config)),
        "features": list(config["features"]),
        "seed": config["seed"],
        "target": target,
        "split_proof": split_proof,
        "label_handling": {
            "drop_nan_labels": config["drop_nan_labels"],
            "nan_labels_dropped": nan_labels_dropped,
        },
        "weighting": weighting,
        "early_stopping_rounds": config["early_stopping_rounds"],
    }


def _training_contract(
    value: object,
    *,
    sample_context: Mapping[str, Any],
) -> dict[str, Any]:
    obj = _object(value, "training_contract")
    _exact_fields(obj, _TRAINING_CONTRACT_FIELDS, "training_contract")
    config = _train_config(obj["train_config"])
    config_hash = _hash(
        obj["train_config_hash"], "training_contract.train_config_hash"
    )
    expected_config_hash = _sha256(_canonical_json(config))
    if not hmac.compare_digest(config_hash, expected_config_hash):
        raise ModelingTrainingEvidenceError(
            "training_contract.train_config_hash does not match TrainConfig"
        )
    if config["dataset_id"] != sample_context["dataset_ref"]["dataset_id"]:
        raise ModelingTrainingEvidenceError(
            "TrainConfig dataset_id does not match StrategySampleDesign V2"
        )
    target = _target(obj["target"])
    if target != _target_from_context(sample_context):
        raise ModelingTrainingEvidenceError(
            "training target does not match StrategySampleDesign V2"
        )
    if config["target_col"] != target["column"]:
        raise ModelingTrainingEvidenceError(
            "TrainConfig target_col does not match governed target"
        )
    if target["drop_missing"] != config["drop_nan_labels"]:
        raise ModelingTrainingEvidenceError(
            "SampleDesign target.drop_missing must equal TrainConfig.drop_nan_labels"
        )
    features = _text_array(
        obj["features"], "training_contract.features", required=True
    )
    if features != config["features"]:
        raise ModelingTrainingEvidenceError(
            "training_contract.features does not match TrainConfig"
        )
    seed = _non_negative_int(obj["seed"], "training_contract.seed")
    if seed != config["seed"]:
        raise ModelingTrainingEvidenceError(
            "training_contract.seed does not match TrainConfig"
        )
    early_stopping = _optional_positive_int(
        obj["early_stopping_rounds"],
        "training_contract.early_stopping_rounds",
    )
    if early_stopping != config["early_stopping_rounds"]:
        raise ModelingTrainingEvidenceError(
            "early_stopping_rounds does not match TrainConfig"
        )
    weighting = _weighting(obj["weighting"])
    expected_weighting = _weighting_from_config(
        config, sample_context=sample_context
    )
    if weighting != expected_weighting:
        raise ModelingTrainingEvidenceError(
            "training weighting does not match TrainConfig and sample semantics"
        )
    split_proof = _split_proof(obj["split_proof"])
    expected_proof = _expected_split_proof(
        config,
        sample_context=sample_context,
        training_split_mask_hashes=_mask_hash_input_from_split_proof(split_proof),
    )
    if split_proof != expected_proof:
        raise ModelingTrainingEvidenceError(
            "split_proof does not match Tool-verified risk partition mask evidence"
        )
    label_handling = _label_handling(obj["label_handling"])
    if label_handling["drop_nan_labels"] != config["drop_nan_labels"]:
        raise ModelingTrainingEvidenceError(
            "label handling does not match TrainConfig"
        )
    expected_dropped = _expected_nan_labels_dropped(split_proof)
    if label_handling["nan_labels_dropped"] != expected_dropped:
        raise ModelingTrainingEvidenceError(
            "nan_labels_dropped does not reconcile with verified split mask evidence"
        )
    _validate_label_policy(
        split_proof,
        drop_nan_labels=config["drop_nan_labels"],
    )
    return {
        "train_config": config,
        "train_config_hash": config_hash,
        "features": features,
        "seed": seed,
        "target": target,
        "split_proof": split_proof,
        "label_handling": label_handling,
        "weighting": weighting,
        "early_stopping_rounds": early_stopping,
    }


def _train_config(value: object) -> dict[str, Any]:
    obj = _object(value, "TrainConfig")
    _exact_fields(obj, _TRAIN_CONFIG_FIELDS, "TrainConfig")
    features = _text_array(obj["features"], "TrainConfig.features", required=True)
    if len(features) > MAX_TRAINING_FEATURES:
        raise ModelingTrainingEvidenceError("TrainConfig.features exceeds limit")
    if len(set(features)) != len(features):
        raise ModelingTrainingEvidenceError("TrainConfig.features contains duplicates")
    target_col = _text(obj["target_col"], "TrainConfig.target_col")
    split_col = _text(obj["split_col"], "TrainConfig.split_col")
    if target_col == split_col or target_col in features or split_col in features:
        raise ModelingTrainingEvidenceError(
            "target and split columns must not leak into model features"
        )
    split_values_obj = _object(obj["split_values"], "TrainConfig.split_values")
    _exact_fields(
        split_values_obj,
        frozenset({"train", "test", "oot"}),
        "TrainConfig.split_values",
    )
    split_values = {
        name: _text(
            split_values_obj[name],
            f"TrainConfig.split_values.{name}",
        )
        for name, _partition in _SPLIT_PARTITIONS
    }
    if len(set(split_values.values())) != 3:
        raise ModelingTrainingEvidenceError(
            "TrainConfig.split_values must be three distinct non-empty strings"
        )
    params_obj = _object(obj["params"], "TrainConfig.params")
    params = _json_value(params_obj, "TrainConfig.params")
    assert isinstance(params, dict)
    _reject_calibration_parameters(params, "TrainConfig.params")
    seed = _non_negative_int(obj["seed"], "TrainConfig.seed")
    if seed > 2**32 - 1:
        raise ModelingTrainingEvidenceError("TrainConfig.seed exceeds uint32")
    early = _optional_positive_int(
        obj["early_stopping_rounds"], "TrainConfig.early_stopping_rounds"
    )
    recipe_id = _text(obj["recipe_id"], "TrainConfig.recipe_id")
    _require_binary_modeling_recipe(recipe_id, name="TrainConfig.recipe_id")
    scenario_id = _optional_text(obj["scenario_id"], "TrainConfig.scenario_id")
    target_type = _text(obj["target_type"], "TrainConfig.target_type")
    if target_type != "binary":
        raise ModelingTrainingEvidenceError(
            "Strategy V2 training evidence currently requires target_type=binary"
        )
    eval_metric = _text(obj["eval_metric"], "TrainConfig.eval_metric")
    drop_nan_labels = _boolean(
        obj["drop_nan_labels"], "TrainConfig.drop_nan_labels"
    )
    weight_col = params.get("sample_weight_col")
    if weight_col is not None:
        weight_name = _text(weight_col, "TrainConfig.params.sample_weight_col")
        params["sample_weight_col"] = weight_name
        if weight_name in {target_col, split_col, *features}:
            raise ModelingTrainingEvidenceError(
                "sample weight column must not be target, split, or a feature"
            )
    return {
        "dataset_id": _text(obj["dataset_id"], "TrainConfig.dataset_id"),
        "features": features,
        "target_col": target_col,
        "split_col": split_col,
        "split_values": split_values,
        "params": params,
        "seed": seed,
        "early_stopping_rounds": early,
        "recipe_id": recipe_id,
        "scenario_id": scenario_id,
        "target_type": target_type,
        "eval_metric": eval_metric,
        "drop_nan_labels": drop_nan_labels,
    }


def _require_binary_modeling_recipe(value: str, *, name: str) -> None:
    if value in BINARY_MODELING_RECIPES:
        return
    if value == "ensemble":
        raise ModelingTrainingEvidenceError(
            f"{name} does not support ensemble; expected an authoritative "
            "binary modeling recipe that emits native P(class=1)"
        )
    raise ModelingTrainingEvidenceError(
        f"{name} must be an authoritative binary modeling recipe that emits "
        "native P(class=1)"
    )


def _target_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    target = context["target"]
    return {
        "column": target["column"],
        "good_value": target["good_value"],
        "bad_value": target["bad_value"],
        "drop_missing": target["drop_missing"],
    }


def _target(value: object) -> dict[str, Any]:
    obj = _object(value, "training target")
    _exact_fields(obj, _TARGET_FIELDS, "training target")
    good = _binary_value(obj["good_value"], "training target.good_value")
    bad = _binary_value(obj["bad_value"], "training target.bad_value")
    if good != 0 or bad != 1:
        raise ModelingTrainingEvidenceError(
            "first training-evidence vertical requires good_value=0 and bad_value=1"
        )
    return {
        "column": _text(obj["column"], "training target.column"),
        "good_value": good,
        "bad_value": bad,
        "drop_missing": _boolean(
            obj["drop_missing"], "training target.drop_missing"
        ),
    }


def _weighting_from_config(
    config: Mapping[str, Any],
    *,
    sample_context: Mapping[str, Any],
) -> dict[str, Any]:
    raw = config["params"].get("sample_weight_col")
    column = None if raw is None else _text(raw, "sample weight column")
    if column is not None and sample_context["field_bindings"]["weight_field"] != column:
        raise ModelingTrainingEvidenceError(
            "sample weight column is not the governed SampleDesign V2 weight field"
        )
    return {"used": column is not None, "column": column}


def _weighting(value: object) -> dict[str, Any]:
    obj = _object(value, "training weighting")
    _exact_fields(obj, _WEIGHTING_FIELDS, "training weighting")
    used = _boolean(obj["used"], "training weighting.used")
    column = _optional_text(obj["column"], "training weighting.column")
    if used != (column is not None):
        raise ModelingTrainingEvidenceError(
            "training weighting used/column pair is inconsistent"
        )
    return {"used": used, "column": column}


def _label_handling(value: object) -> dict[str, Any]:
    obj = _object(value, "label_handling")
    _exact_fields(obj, _LABEL_HANDLING_FIELDS, "label_handling")
    return {
        "drop_nan_labels": _boolean(
            obj["drop_nan_labels"], "label_handling.drop_nan_labels"
        ),
        "nan_labels_dropped": _non_negative_int(
            obj["nan_labels_dropped"], "label_handling.nan_labels_dropped"
        ),
    }


def _expected_split_proof(
    config: Mapping[str, Any],
    *,
    sample_context: Mapping[str, Any],
    training_split_mask_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = sample_context["bundle"]
    header = bundle["membership"]
    design = bundle["sample_design"]
    risk = sample_context["risk"]
    mask_hashes = _training_split_mask_hashes(training_split_mask_hashes)
    expected_mask_binding = {
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "membership_payload_hash": header["payload_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "row_count": header["row_count"],
    }
    if any(
        mask_hashes[field] != expected
        for field, expected in expected_mask_binding.items()
    ):
        raise ModelingTrainingEvidenceError(
            "training split mask hashes belong to a different sample design or membership"
        )
    observations = _risk_metric_observations(bundle)
    partitions = {item["name"]: item for item in risk["partitions"]}
    split_rows: list[dict[str, Any]] = []
    for name, partition in _SPLIT_PARTITIONS:
        population_row = partitions[partition]
        metrics = observations[partition]
        labeled = _present_count(metrics["labeled_count"], f"risk/{partition} labeled_count")
        bad = _present_count(metrics["bad_count"], f"risk/{partition} bad_count")
        row_count = population_row["row_count"]
        if mask_hashes["splits"][name]["selected_count"] != row_count:
            raise ModelingTrainingEvidenceError(
                f"{name} verified mask count does not match governed membership count"
            )
        if labeled > row_count or bad > labeled:
            raise ModelingTrainingEvidenceError(
                f"risk/{partition} label counts exceed exact membership"
            )
        split_rows.append(
            {
                "name": name,
                "population": "risk",
                "partition": partition,
                "split_value": config["split_values"][name],
                "membership_ref": dict(population_row["membership_ref"]),
                "row_count": row_count,
                "labeled_count": labeled,
                "bad_count": bad,
                "good_count": labeled - bad,
                "selector_mask_content_hash": mask_hashes["splits"][name][
                    "selector_mask_content_hash"
                ],
                "membership_mask_content_hash": mask_hashes["splits"][name][
                    "membership_mask_content_hash"
                ],
            }
        )
    proof_body = {
        "membership_id": header["membership_id"],
        "membership_content_hash": header["content_hash"],
        "membership_payload_hash": header["payload_hash"],
        "sample_design_id": design["sample_design_id"],
        "sample_design_content_hash": design["content_hash"],
        "row_ordinal": dict(header["row_ordinal"]),
        "mask_hash_algorithm": mask_hashes["mask_hash_algorithm"],
        "selector_union_mask_content_hash": mask_hashes[
            "selector_union_mask_content_hash"
        ],
        "membership_union_mask_content_hash": mask_hashes[
            "membership_union_mask_content_hash"
        ],
        "risk_membership_mask_content_hash": mask_hashes[
            "risk_membership_mask_content_hash"
        ],
        "pairwise_overlap_counts": mask_hashes["pairwise_overlap_counts"],
        "splits": split_rows,
    }
    return {**proof_body, "content_hash": _sha256(_canonical_json(proof_body))}


def _training_split_mask_hashes(value: object) -> dict[str, Any]:
    obj = _object(value, "training_split_mask_hashes")
    _exact_fields(
        obj,
        _TRAINING_SPLIT_MASK_HASH_FIELDS,
        "training_split_mask_hashes",
    )
    if obj["mask_hash_algorithm"] != TRAINING_MASK_HASH_ALGORITHM:
        raise ModelingTrainingEvidenceError(
            "training split mask hash algorithm is unsupported"
        )
    raw_splits = _object(
        obj["splits"], "training_split_mask_hashes.splits"
    )
    _exact_fields(
        raw_splits,
        _TRAINING_SPLIT_MASK_ITEMS,
        "training_split_mask_hashes.splits",
    )
    split_hashes: dict[str, dict[str, str]] = {}
    for name in ("train", "test", "oot"):
        item = _object(
            raw_splits[name], f"training_split_mask_hashes.splits.{name}"
        )
        _exact_fields(
            item,
            _TRAINING_SPLIT_MASK_ITEM_FIELDS,
            f"training_split_mask_hashes.splits.{name}",
        )
        selector_hash = _hash(
            item["selector_mask_content_hash"],
            f"training_split_mask_hashes.splits.{name}.selector_mask_content_hash",
        )
        membership_hash = _hash(
            item["membership_mask_content_hash"],
            f"training_split_mask_hashes.splits.{name}.membership_mask_content_hash",
        )
        if not hmac.compare_digest(selector_hash, membership_hash):
            raise ModelingTrainingEvidenceError(
                f"{name} selector and membership mask hashes must be identical"
            )
        split_hashes[name] = {
            "selector_mask_content_hash": selector_hash,
            "membership_mask_content_hash": membership_hash,
            "selected_count": _non_negative_int(
                item["selected_count"],
                f"training_split_mask_hashes.splits.{name}.selected_count",
            ),
        }
    overlaps = _pairwise_overlap_counts(obj["pairwise_overlap_counts"])
    if any(overlaps.values()):
        raise ModelingTrainingEvidenceError(
            "training split masks must be pairwise exclusive"
        )
    selector_union_hash = _hash(
        obj["selector_union_mask_content_hash"],
        "training_split_mask_hashes.selector_union_mask_content_hash",
    )
    membership_union_hash = _hash(
        obj["membership_union_mask_content_hash"],
        "training_split_mask_hashes.membership_union_mask_content_hash",
    )
    risk_hash = _hash(
        obj["risk_membership_mask_content_hash"],
        "training_split_mask_hashes.risk_membership_mask_content_hash",
    )
    if not (
        hmac.compare_digest(selector_union_hash, membership_union_hash)
        and hmac.compare_digest(selector_union_hash, risk_hash)
    ):
        raise ModelingTrainingEvidenceError(
            "training split union hashes must equal governed risk membership hash"
        )
    return {
        "membership_id": _text(
            obj["membership_id"], "training_split_mask_hashes.membership_id"
        ),
        "membership_content_hash": _hash(
            obj["membership_content_hash"],
            "training_split_mask_hashes.membership_content_hash",
        ),
        "membership_payload_hash": _hash(
            obj["membership_payload_hash"],
            "training_split_mask_hashes.membership_payload_hash",
        ),
        "sample_design_id": _text(
            obj["sample_design_id"],
            "training_split_mask_hashes.sample_design_id",
        ),
        "sample_design_content_hash": _hash(
            obj["sample_design_content_hash"],
            "training_split_mask_hashes.sample_design_content_hash",
        ),
        "row_count": _positive_int(
            obj["row_count"], "training_split_mask_hashes.row_count"
        ),
        "mask_hash_algorithm": TRAINING_MASK_HASH_ALGORITHM,
        "selector_union_mask_content_hash": selector_union_hash,
        "membership_union_mask_content_hash": membership_union_hash,
        "risk_membership_mask_content_hash": risk_hash,
        "pairwise_overlap_counts": overlaps,
        "splits": split_hashes,
    }


def _pairwise_overlap_counts(value: object) -> dict[str, int]:
    obj = _object(value, "pairwise_overlap_counts")
    _exact_fields(obj, _PAIRWISE_OVERLAP_FIELDS, "pairwise_overlap_counts")
    return {
        key: _non_negative_int(obj[key], f"pairwise_overlap_counts.{key}")
        for key in sorted(_PAIRWISE_OVERLAP_FIELDS)
    }


def _mask_hash_input_from_split_proof(
    split_proof: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "membership_id": split_proof["membership_id"],
        "membership_content_hash": split_proof["membership_content_hash"],
        "membership_payload_hash": split_proof["membership_payload_hash"],
        "sample_design_id": split_proof["sample_design_id"],
        "sample_design_content_hash": split_proof["sample_design_content_hash"],
        "row_count": split_proof["row_ordinal"]["stop"],
        "mask_hash_algorithm": split_proof["mask_hash_algorithm"],
        "selector_union_mask_content_hash": split_proof[
            "selector_union_mask_content_hash"
        ],
        "membership_union_mask_content_hash": split_proof[
            "membership_union_mask_content_hash"
        ],
        "risk_membership_mask_content_hash": split_proof[
            "risk_membership_mask_content_hash"
        ],
        "pairwise_overlap_counts": split_proof["pairwise_overlap_counts"],
        "splits": {
            item["name"]: {
                "selector_mask_content_hash": item[
                    "selector_mask_content_hash"
                ],
                "membership_mask_content_hash": item[
                    "membership_mask_content_hash"
                ],
                "selected_count": item["row_count"],
            }
            for item in split_proof["splits"]
        },
    }


def _split_proof(value: object) -> dict[str, Any]:
    obj = _object(value, "split_proof")
    _exact_fields(obj, _SPLIT_PROOF_FIELDS, "split_proof")
    row_ordinal_obj = _object(obj["row_ordinal"], "split_proof.row_ordinal")
    _exact_fields(row_ordinal_obj, _ROW_ORDINAL_FIELDS, "split_proof.row_ordinal")
    row_ordinal = {
        "start": _non_negative_int(row_ordinal_obj["start"], "row_ordinal.start"),
        "stop": _positive_int(row_ordinal_obj["stop"], "row_ordinal.stop"),
        "step": _positive_int(row_ordinal_obj["step"], "row_ordinal.step"),
    }
    if row_ordinal != {"start": 0, "stop": row_ordinal["stop"], "step": 1}:
        raise ModelingTrainingEvidenceError(
            "split_proof row ordinal must be positive zero-based unit-step"
        )
    raw_splits = _array(obj["splits"], "split_proof.splits", required=True)
    if len(raw_splits) != 3:
        raise ModelingTrainingEvidenceError(
            "split_proof must contain train, test, and oot"
        )
    splits = [
        _split_row(item, expected_name=name, expected_partition=partition)
        for item, (name, partition) in zip(raw_splits, _SPLIT_PARTITIONS, strict=True)
    ]
    proof_body = {
        "membership_id": _text(obj["membership_id"], "split_proof.membership_id"),
        "membership_content_hash": _hash(
            obj["membership_content_hash"], "split_proof.membership_content_hash"
        ),
        "membership_payload_hash": _hash(
            obj["membership_payload_hash"], "split_proof.membership_payload_hash"
        ),
        "sample_design_id": _text(
            obj["sample_design_id"], "split_proof.sample_design_id"
        ),
        "sample_design_content_hash": _hash(
            obj["sample_design_content_hash"],
            "split_proof.sample_design_content_hash",
        ),
        "row_ordinal": row_ordinal,
        "mask_hash_algorithm": _text(
            obj["mask_hash_algorithm"], "split_proof.mask_hash_algorithm"
        ),
        "selector_union_mask_content_hash": _hash(
            obj["selector_union_mask_content_hash"],
            "split_proof.selector_union_mask_content_hash",
        ),
        "membership_union_mask_content_hash": _hash(
            obj["membership_union_mask_content_hash"],
            "split_proof.membership_union_mask_content_hash",
        ),
        "risk_membership_mask_content_hash": _hash(
            obj["risk_membership_mask_content_hash"],
            "split_proof.risk_membership_mask_content_hash",
        ),
        "pairwise_overlap_counts": _pairwise_overlap_counts(
            obj["pairwise_overlap_counts"]
        ),
        "splits": splits,
    }
    _training_split_mask_hashes(
        _mask_hash_input_from_split_proof(proof_body)
    )
    content_hash = _hash(obj["content_hash"], "split_proof.content_hash")
    if not hmac.compare_digest(content_hash, _sha256(_canonical_json(proof_body))):
        raise ModelingTrainingEvidenceError(
            "split_proof.content_hash does not match proof content"
        )
    return {**proof_body, "content_hash": content_hash}


def _split_row(
    value: object,
    *,
    expected_name: str,
    expected_partition: str,
) -> dict[str, Any]:
    obj = _object(value, f"split_proof {expected_name}")
    _exact_fields(obj, _SPLIT_FIELDS, f"split_proof {expected_name}")
    if obj["name"] != expected_name or obj["partition"] != expected_partition:
        raise ModelingTrainingEvidenceError(
            "split_proof must use canonical train/development, test/validation, oot/oot order"
        )
    if obj["population"] != "risk":
        raise ModelingTrainingEvidenceError(
            "model training split proof must use the risk population"
        )
    membership = _partition_membership_ref(obj["membership_ref"])
    expected_mask = f"risk/{expected_partition}"
    if membership["mask_name"] != expected_mask:
        raise ModelingTrainingEvidenceError(
            f"split_proof {expected_name} mask_name must be {expected_mask}"
        )
    row_count = _non_negative_int(
        obj["row_count"], f"split_proof {expected_name}.row_count"
    )
    labeled = _non_negative_int(
        obj["labeled_count"], f"split_proof {expected_name}.labeled_count"
    )
    bad = _non_negative_int(
        obj["bad_count"], f"split_proof {expected_name}.bad_count"
    )
    good = _non_negative_int(
        obj["good_count"], f"split_proof {expected_name}.good_count"
    )
    if labeled > row_count or bad + good != labeled:
        raise ModelingTrainingEvidenceError(
            f"split_proof {expected_name} label counts are inconsistent"
        )
    return {
        "name": expected_name,
        "population": "risk",
        "partition": expected_partition,
        "split_value": _text(
            obj["split_value"], f"split_proof {expected_name}.split_value"
        ),
        "membership_ref": membership,
        "row_count": row_count,
        "labeled_count": labeled,
        "bad_count": bad,
        "good_count": good,
        "selector_mask_content_hash": _hash(
            obj["selector_mask_content_hash"],
            f"split_proof {expected_name}.selector_mask_content_hash",
        ),
        "membership_mask_content_hash": _hash(
            obj["membership_mask_content_hash"],
            f"split_proof {expected_name}.membership_mask_content_hash",
        ),
    }


def _partition_membership_ref(value: object) -> dict[str, str]:
    obj = _object(value, "partition membership ref")
    _exact_fields(
        obj,
        _PARTITION_MEMBERSHIP_REF_FIELDS,
        "partition membership ref",
    )
    return {
        "membership_id": _text(
            obj["membership_id"], "partition membership ref.membership_id"
        ),
        "membership_content_hash": _hash(
            obj["membership_content_hash"],
            "partition membership ref.membership_content_hash",
        ),
        "mask_name": _text(
            obj["mask_name"], "partition membership ref.mask_name"
        ),
    }


def _risk_metric_observations(
    bundle: Mapping[str, Any],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    metric_keys = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for observation in bundle["metric_observations"]:
        if observation["population"] != "risk" or observation["partition"] == "overall":
            continue
        metric_key = metric_keys[
            observation["metric_definition_ref"]["metric_definition_id"]
        ]
        result.setdefault(observation["partition"], {})[metric_key] = observation
    for _name, partition in _SPLIT_PARTITIONS:
        if partition not in result or not {"labeled_count", "bad_count"} <= result[partition].keys():
            raise ModelingTrainingEvidenceError(
                f"risk/{partition} sample statistics are incomplete"
            )
    return result


def _present_count(value: Mapping[str, Any], name: str) -> int:
    observed = value.get("value")
    if value.get("status") != "present" or isinstance(observed, bool) or not isinstance(observed, int):
        raise ModelingTrainingEvidenceError(f"{name} must be a present integer")
    if observed < 0:
        raise ModelingTrainingEvidenceError(f"{name} must be non-negative")
    return observed


def _expected_nan_labels_dropped(split_proof: Mapping[str, Any]) -> int:
    by_name = {item["name"]: item for item in split_proof["splits"]}
    required = sum(
        by_name[name]["row_count"] - by_name[name]["labeled_count"]
        for name in ("train", "test")
    )
    oot = by_name["oot"]
    partial_oot = (
        oot["row_count"] - oot["labeled_count"]
        if 0 < oot["labeled_count"] < oot["row_count"]
        else 0
    )
    return required + partial_oot


def _validate_label_policy(
    split_proof: Mapping[str, Any],
    *,
    drop_nan_labels: bool,
) -> None:
    by_name = {item["name"]: item for item in split_proof["splits"]}
    for name in ("train", "test"):
        split = by_name[name]
        if split["row_count"] == 0:
            raise ModelingTrainingEvidenceError(
                f"{name} split must not be empty"
            )
        if split["labeled_count"] != split["row_count"] and not drop_nan_labels:
            raise ModelingTrainingEvidenceError(
                f"{name} missing labels require drop_nan_labels confirmation"
            )
        if split["bad_count"] == 0 or split["good_count"] == 0:
            raise ModelingTrainingEvidenceError(
                f"{name} split requires both good and bad labels"
            )
    oot = by_name["oot"]
    if 0 < oot["labeled_count"] < oot["row_count"] and not drop_nan_labels:
        raise ModelingTrainingEvidenceError(
            "partially labeled oot requires drop_nan_labels confirmation"
        )
    if oot["labeled_count"] > 0 and (
        oot["bad_count"] == 0 or oot["good_count"] == 0
    ):
        raise ModelingTrainingEvidenceError(
            "labeled oot requires both good and bad labels; single-class OOT is insufficient data"
        )


def _build_metrics_snapshot(value: object) -> dict[str, Any]:
    values = _metric_values(value)
    return {
        "values": values,
        "content_hash": _sha256(_canonical_json(values)),
    }


def _metrics_snapshot(value: object) -> dict[str, Any]:
    obj = _object(value, "metrics_snapshot")
    _exact_fields(obj, _METRICS_SNAPSHOT_FIELDS, "metrics_snapshot")
    values = _metric_values(obj["values"])
    supplied_hash = _hash(
        obj["content_hash"], "metrics_snapshot.content_hash"
    )
    content_hash = _sha256(_canonical_json(values))
    if not hmac.compare_digest(supplied_hash, content_hash):
        raise ModelingTrainingEvidenceError(
            "metrics_snapshot.content_hash does not match metric values"
        )
    return {"values": values, "content_hash": content_hash}


def _metric_values(value: object) -> dict[str, Any]:
    values_obj = _object(value, "metrics_snapshot.values")
    _exact_fields(values_obj, _MODEL_METRIC_FIELDS, "metrics_snapshot.values")
    values: dict[str, Any] = {}
    for name in sorted(_MODEL_METRIC_FIELDS):
        raw = values_obj[name]
        if name == "overfit_flag":
            values[name] = _boolean(raw, f"metrics_snapshot.values.{name}")
        elif name == "ks_ci_n_boot":
            values[name] = (
                None
                if raw is None
                else _non_negative_int(raw, f"metrics_snapshot.values.{name}")
            )
        elif name == "overfit_train_test_gap":
            values[name] = float(
                _finite_number(raw, f"metrics_snapshot.values.{name}")
            )
        else:
            values[name] = (
                None
                if raw is None
                else float(_finite_number(raw, f"metrics_snapshot.values.{name}"))
            )
        observed = values[name]
        if observed is not None and name in _RATIO_METRICS and not 0 <= observed <= 1:
            raise ModelingTrainingEvidenceError(
                f"metrics_snapshot.values.{name} must be within [0, 1]"
            )
        if observed is not None and name in _NON_NEGATIVE_METRICS and observed < 0:
            raise ModelingTrainingEvidenceError(
                f"metrics_snapshot.values.{name} must be non-negative"
            )
    return values


def _validate_binary_metrics(
    values: Mapping[str, Any],
    *,
    split_proof: Mapping[str, Any],
    weighting: Mapping[str, Any],
) -> None:
    if any(values[name] is not None for name in _NON_BINARY_METRICS):
        raise ModelingTrainingEvidenceError(
            "binary training evidence cannot contain regression or multiclass metrics"
        )
    for name in ("train_ks", "test_ks", "train_auc", "test_auc", "psi_test_vs_train"):
        if values[name] is None:
            raise ModelingTrainingEvidenceError(
                f"binary training evidence requires {name}"
            )
    by_name = {item["name"]: item for item in split_proof["splits"]}
    oot = by_name["oot"]
    oot_has_scores = oot["row_count"] > 0
    oot_has_binary_labels = oot["bad_count"] > 0 and oot["good_count"] > 0
    if (values["psi_oot_vs_train"] is not None) != oot_has_scores:
        raise ModelingTrainingEvidenceError(
            "OOT PSI availability does not match exact OOT membership"
        )
    for name in ("oot_ks", "oot_auc"):
        if (values[name] is not None) != oot_has_binary_labels:
            raise ModelingTrainingEvidenceError(
                f"{name} availability does not match OOT label support"
            )
    oot_optional_label_metrics = (
        "oot_ks_ci_low",
        "oot_ks_ci_high",
        "oot_ks_ci_std",
        "oot_lift_head_5",
        "oot_lift_tail_5",
        "oot_lift_head_10",
        "oot_lift_tail_10",
    )
    if not oot_has_binary_labels and any(
        values[name] is not None for name in oot_optional_label_metrics
    ):
        raise ModelingTrainingEvidenceError(
            "OOT confidence intervals and lift require both good and bad labels"
        )
    if weighting["used"]:
        for name in (
            "weighted_train_ks",
            "weighted_test_ks",
            "weighted_train_auc",
            "weighted_test_auc",
            "weighted_psi_test_vs_train",
        ):
            if values[name] is None:
                raise ModelingTrainingEvidenceError(
                    f"weighted training requires {name}"
                )
        for name in ("weighted_oot_ks", "weighted_oot_auc"):
            if (values[name] is not None) != oot_has_binary_labels:
                raise ModelingTrainingEvidenceError(
                    f"{name} availability does not match OOT label support"
                )
        if (values["weighted_psi_oot_vs_train"] is not None) != oot_has_scores:
            raise ModelingTrainingEvidenceError(
                "weighted OOT PSI availability does not match OOT membership"
            )
    elif any(values[name] is not None for name in _WEIGHTED_METRICS):
        raise ModelingTrainingEvidenceError(
            "unweighted TrainConfig cannot contain weighted metrics"
        )

    train_ks = float(values["train_ks"])
    test_ks = float(values["test_ks"])
    oot_ks = values["oot_ks"]
    expected_train_test_gap = (
        abs(train_ks - test_ks) / abs(train_ks)
        if abs(train_ks) > 1e-12
        else 0.0
    )
    expected_train_oot_gap = (
        None if oot_ks is None else abs(train_ks - float(oot_ks))
    )
    if not math.isclose(
        values["overfit_train_test_gap"],
        expected_train_test_gap,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ModelingTrainingEvidenceError(
            "overfit_train_test_gap is inconsistent with KS snapshot"
        )
    if (values["overfit_train_oot_gap"] is None) != (expected_train_oot_gap is None):
        raise ModelingTrainingEvidenceError(
            "overfit_train_oot_gap availability is inconsistent with OOT KS"
        )
    if expected_train_oot_gap is not None and not math.isclose(
        values["overfit_train_oot_gap"],
        expected_train_oot_gap,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ModelingTrainingEvidenceError(
            "overfit_train_oot_gap is inconsistent with KS snapshot"
        )
    expected_flag = expected_train_test_gap > 0.10 or (
        expected_train_oot_gap is not None and expected_train_oot_gap > 0.05
    )
    if values["overfit_flag"] is not expected_flag:
        raise ModelingTrainingEvidenceError(
            "overfit_flag is inconsistent with governed thresholds"
        )
    _validate_ci(values, prefix="test")
    _validate_ci(values, prefix="oot")


def _validate_ci(values: Mapping[str, Any], *, prefix: str) -> None:
    low = values[f"{prefix}_ks_ci_low"]
    high = values[f"{prefix}_ks_ci_high"]
    std = values[f"{prefix}_ks_ci_std"]
    if len({item is None for item in (low, high, std)}) != 1:
        raise ModelingTrainingEvidenceError(
            f"{prefix} KS confidence interval must be wholly present or absent"
        )
    if low is not None and low > high:
        raise ModelingTrainingEvidenceError(
            f"{prefix} KS confidence interval low exceeds high"
        )
    if prefix == "test":
        n_boot = values["ks_ci_n_boot"]
        if low is None and n_boot not in {None, 0}:
            raise ModelingTrainingEvidenceError(
                "ks_ci_n_boot must be null or zero when test KS confidence "
                "interval is absent"
            )
        if low is not None and n_boot is None:
            raise ModelingTrainingEvidenceError(
                "ks_ci_n_boot must be positive when test KS confidence "
                "interval is present"
            )
        if low is not None and n_boot == 0:
            test_ks = values["test_ks"]
            is_runtime_degenerate_ci = (
                math.isclose(low, test_ks, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(high, test_ks, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(std, 0.0, rel_tol=1e-12, abs_tol=1e-12)
            )
            if not is_runtime_degenerate_ci:
                raise ModelingTrainingEvidenceError(
                    "zero-bootstrap test KS confidence interval must be "
                    "zero-width at test_ks with zero standard deviation"
                )


def _feature_importance(
    value: object,
    *,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    items = _array(value, "feature_importance", required=False)
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(items):
        obj = _object(raw, f"feature_importance[{index}]")
        _exact_fields(
            obj,
            _FEATURE_IMPORTANCE_FIELDS,
            f"feature_importance[{index}]",
        )
        feature = _text(obj["feature"], f"feature_importance[{index}].feature")
        importance = float(
            _finite_number(
                obj["importance"], f"feature_importance[{index}].importance"
            )
        )
        if importance < 0:
            raise ModelingTrainingEvidenceError(
                "feature importance must be non-negative"
            )
        if feature not in features:
            raise ModelingTrainingEvidenceError(
                "feature importance references a feature outside TrainConfig"
            )
        normalized.append({"feature": feature, "importance": importance})
    names = [item["feature"] for item in normalized]
    if len(names) != len(set(names)):
        raise ModelingTrainingEvidenceError(
            "feature_importance contains duplicate features"
        )
    normalized.sort(key=lambda item: (-item["importance"], item["feature"]))
    return normalized


def _reject_calibration_parameters(value: object, name: str) -> None:
    calibration_fragments = ("calibrat", "isotonic", "platt")
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in calibration_fragments):
                raise ModelingTrainingEvidenceError(
                    f"{name} contains unsupported calibration parameter: {key}"
                )
            _reject_calibration_parameters(child, f"{name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_calibration_parameters(child, f"{name}[{index}]")
        return
    if isinstance(value, str) and value.lower() in {
        "calibrated",
        "isotonic",
        "platt",
        "platt_scaling",
    }:
        raise ModelingTrainingEvidenceError(
            f"{name} requests an unsupported calibrated score product"
        )


def _three_boolean_masks(
    value: object,
    name: str,
    *,
    expected_row_count: int,
) -> dict[str, np.ndarray]:
    obj = _object(value, name)
    _exact_fields(obj, _TRAINING_SPLIT_MASK_ITEMS, name)
    return {
        split: _boolean_mask(
            obj[split],
            f"{name}.{split}",
            expected_row_count=expected_row_count,
        )
        for split in ("train", "test", "oot")
    }


def _boolean_mask(
    value: object,
    name: str,
    *,
    expected_row_count: int,
) -> np.ndarray:
    if isinstance(value, np.ndarray):
        row_count = len(value) if value.ndim > 0 else 0
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        row_count = len(value)
    else:
        raise ModelingTrainingEvidenceError(f"{name} must be a boolean array")
    _validate_training_mask_row_budget(row_count)
    if row_count == 0:
        raise ModelingTrainingEvidenceError(f"{name} must not be empty")
    if row_count != expected_row_count:
        raise ModelingTrainingEvidenceError(
            f"{name} does not match SampleDesign V2 analysis-universe row count"
        )
    array = value if isinstance(value, np.ndarray) else np.asarray(value)
    if array.ndim != 1 or array.dtype.kind != "b":
        raise ModelingTrainingEvidenceError(
            f"{name} must be a one-dimensional boolean array"
        )
    return np.ascontiguousarray(array, dtype=np.bool_)


def _validate_training_mask_row_budget(row_count: int) -> None:
    if row_count > MAX_TRAINING_MASK_ROWS:
        raise ModelingTrainingEvidenceError(
            "training mask row limit exceeds SampleMembership V2 byte budget"
        )


def _mask_overlap_count(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.count_nonzero(first & second))


def _mask_union(*masks: np.ndarray) -> np.ndarray:
    union = masks[0].copy()
    for mask in masks[1:]:
        np.logical_or(union, mask, out=union)
    return union


def _boolean_mask_content_hash(mask: np.ndarray) -> str:
    packed = np.packbits(mask, bitorder="little").tobytes()
    payload = (
        b"MARVIS_BOOL_MASK_V1\x00"
        + int(mask.size).to_bytes(8, byteorder="little", signed=False)
        + packed
    )
    return hashlib.sha256(payload).hexdigest()


def _address(body: Mapping[str, Any]) -> dict[str, Any]:
    evidence_id = "modeling-training-evidence-" + _sha256(
        _canonical_json(body)
    )[:24]
    without_hash = {**body, "evidence_id": evidence_id}
    return {
        **without_hash,
        "content_hash": _sha256(_canonical_json(without_hash)),
    }


def _validate_addressed(
    original: Mapping[str, Any],
    body: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_id = _text(original["evidence_id"], "evidence_id")
    if not _EVIDENCE_ID_RE.fullmatch(evidence_id):
        raise ModelingTrainingEvidenceError("evidence_id is invalid")
    expected_id = "modeling-training-evidence-" + _sha256(
        _canonical_json(body)
    )[:24]
    if not hmac.compare_digest(evidence_id, expected_id):
        raise ModelingTrainingEvidenceError(
            "evidence_id does not match canonical body"
        )
    without_hash = {**body, "evidence_id": evidence_id}
    content_hash = _hash(original["content_hash"], "content_hash")
    if not hmac.compare_digest(
        content_hash, _sha256(_canonical_json(without_hash))
    ):
        raise ModelingTrainingEvidenceError(
            "content_hash does not match evidence content"
        )
    return {**without_hash, "content_hash": content_hash}


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelingTrainingEvidenceError(f"{name} must be an object")
    return value


def _array(value: object, name: str, *, required: bool) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ModelingTrainingEvidenceError(f"{name} must be an array")
    result = list(value)
    if required and not result:
        raise ModelingTrainingEvidenceError(f"{name} must not be empty")
    return result


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ModelingTrainingEvidenceError(
            f"{name} fields are invalid; missing={missing}, extra={extra}"
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ModelingTrainingEvidenceError(
            f"{name} must be non-empty canonical text"
        )
    if "\x00" in value:
        raise ModelingTrainingEvidenceError(f"{name} must not contain NUL")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _text_array(value: object, name: str, *, required: bool) -> list[str]:
    result = [_text(item, f"{name}[{index}]") for index, item in enumerate(_array(value, name, required=required))]
    return result


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ModelingTrainingEvidenceError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ModelingTrainingEvidenceError(f"{name} must be a boolean")
    return value


def _finite_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ModelingTrainingEvidenceError(f"{name} must be a finite number")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingTrainingEvidenceError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise ModelingTrainingEvidenceError(f"{name} must be positive")
    return result


def _optional_positive_int(value: object, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def _binary_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise ModelingTrainingEvidenceError(f"{name} must be integer 0 or 1")
    return value


def _json_scalar(value: object, name: str) -> str | bool | int | float:
    if isinstance(value, str):
        return _text(value, name)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    raise ModelingTrainingEvidenceError(
        f"{name} must be a finite JSON scalar"
    )


def _json_value(value: object, name: str) -> Any:
    _preflight_json_tree(value, name=name)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _text(value, name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_number(value, name)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            canonical_key = _text(key, f"{name} key")
            normalized[canonical_key] = _json_value(value[key], f"{name}.{canonical_key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ModelingTrainingEvidenceError(
        f"{name} contains unsupported {type(value).__name__}"
    )


@dataclass(frozen=True)
class _Leave:
    identity: int


def _preflight_json_tree(value: object, *, name: str) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    active: set[int] = set()
    nodes = 0
    while stack:
        current, depth = stack.pop()
        if isinstance(current, _Leave):
            active.discard(current.identity)
            continue
        nodes += 1
        if nodes > MAX_TRAINING_EVIDENCE_JSON_NODES:
            raise ModelingTrainingEvidenceError(f"{name} exceeds node budget")
        if depth > MAX_TRAINING_EVIDENCE_JSON_DEPTH:
            raise ModelingTrainingEvidenceError(f"{name} exceeds depth budget")
        if current is None or isinstance(current, (str, bool)):
            continue
        if isinstance(current, (int, float)):
            if isinstance(current, float) and not math.isfinite(current):
                raise ModelingTrainingEvidenceError(
                    f"{name} contains non-finite number"
                )
            continue
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in active:
                raise ModelingTrainingEvidenceError(f"{name} contains a cycle")
            if any(not isinstance(key, str) for key in current):
                raise ModelingTrainingEvidenceError(
                    f"{name} keys must be strings"
                )
            active.add(identity)
            stack.append((_Leave(identity), depth))
            stack.extend((child, depth + 1) for child in current.values())
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            identity = id(current)
            if identity in active:
                raise ModelingTrainingEvidenceError(f"{name} contains a cycle")
            active.add(identity)
            stack.append((_Leave(identity), depth))
            stack.extend((child, depth + 1) for child in current)
            continue
        raise ModelingTrainingEvidenceError(
            f"{name} contains unsupported {type(current).__name__}"
        )


def _canonical_json(value: object) -> str:
    _preflight_json_tree(value, name="canonical JSON")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ModelingTrainingEvidenceError(
            "value is not finite canonical JSON"
        ) from exc


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelingTrainingEvidenceError(
                f"modeling training evidence JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "MAX_TRAINING_EVIDENCE_JSON_BYTES",
    "MAX_TRAINING_EVIDENCE_JSON_DEPTH",
    "MAX_TRAINING_EVIDENCE_JSON_NODES",
    "MAX_TRAINING_MASK_BYTES",
    "MAX_TRAINING_MASK_ROWS",
    "MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND",
    "MODELING_TRAINING_EVIDENCE_PRODUCER_VERSION",
    "MODELING_TRAINING_EVIDENCE_SCHEMA_VERSION",
    "MODEL_BINARY_REF_KIND",
    "RAW_SCORE_PRODUCT",
    "SAMPLE_DESIGN_BUNDLE_ARTIFACT_KIND",
    "SAMPLE_MEMBERSHIP_ARTIFACT_KIND",
    "TRAINING_MASK_HASH_ALGORITHM",
    "ModelingTrainingEvidenceError",
    "build_model_binary_artifact_ref",
    "build_modeling_training_evidence",
    "build_task_artifact_ref",
    "build_training_split_mask_hashes",
    "canonical_modeling_training_evidence_json",
    "modeling_training_evidence_from_json",
    "validate_modeling_training_evidence",
]
