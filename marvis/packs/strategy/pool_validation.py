"""Partition-neutral independent evidence for one exact Strategy Pool.

The governed Tool authenticates the current Pool, StrategySampleDesign V2
membership/bundle pair, and live dataset.  This persistence-free module only
receives the exact selected ``risk/validation`` or ``risk/oot`` rows and turns
the existing deterministic first-match kernel into a distinct independent
evidence contract.  Development lineage remains explicit provenance; it is
never presented as the observed population or lifecycle stage.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import re
from typing import Any

import pandas as pd

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import validate_strategy_pool
from marvis.packs.strategy.pool_impact import (
    STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
    STRATEGY_POOL_IMPACT_SCHEMA_VERSION,
    build_strategy_pool_impact_assessment,
    validate_strategy_pool_impact_assessment,
)
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentRef,
)


STRATEGY_POOL_VALIDATION_SCHEMA_VERSION = "strategy.pool-validation-evidence.v1"
STRATEGY_POOL_VALIDATION_PRODUCER_VERSION = (
    "marvis.strategy.pool-validation/1"
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_ID_RE = re.compile(r"^strategy-pool-validation-[0-9a-f]{24}$")
_PARTITIONS = frozenset({"validation", "oot"})
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "evidence_id",
        "identity",
        "source_bindings",
        "partition",
        "population",
        "comparison_mode",
        "lifecycle",
        "population_metrics",
        "overall",
        "waterfall",
        "default_unmatched",
        "monthly",
        "conservation",
        "red_flags",
        "content_hash",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "pool_id",
        "task_id",
        "strategy_type",
        "revision",
        "revision_id",
        "snapshot_hash",
        "design_hash",
        "strategy_spec_hash",
    }
)
_SOURCE_BINDING_FIELDS = frozenset(
    {
        "pool_artifact",
        "sample_design_v2",
        "dataset",
        "development_lineage",
        "target",
        "fields",
    }
)
_POOL_ARTIFACT_REF_FIELDS = frozenset(
    {"artifact_id", "artifact_content_hash"}
)
_SAMPLE_DESIGN_V2_REF_FIELDS = frozenset(
    {
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "membership_id",
        "membership_content_hash",
        "bundle_artifact_id",
        "bundle_artifact_content_hash",
        "bundle_id",
        "bundle_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition_key",
        "partition_count",
        "analysis_universe_row_count",
    }
)
_DATASET_BINDING_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "dataset_source_path",
        "dataset_registry_metadata_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    }
)
_SAMPLE_BINDING_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
)
_DEVELOPMENT_LINEAGE_FIELDS = frozenset(
    {"legacy_development_ref", "sample_binding"}
)
_TARGET_FIELDS = frozenset(
    {"column", "good_value", "bad_value", "missing_policy"}
)
_FIELD_BINDING_FIELDS = frozenset(
    {"month_col", "loan_amount_col", "overdue_amount_col"}
)
_LIFECYCLE_FIELDS = frozenset(
    {
        "stage",
        "validation_status",
        "mutates_pool",
        "creates_strategy",
        "adopts_strategy",
        "promotes_strategy",
        "deploys_strategy",
    }
)
_CONSERVATION_FIELDS = frozenset(
    {
        "standalone_equals_incremental_plus_shadowed",
        "incremental_plus_default_equals_population",
        "monthly_rolls_to_overall",
        "selected_partition_equals_membership_count",
        "risk_partition_excludes_development",
    }
)
_LEGACY_CONSERVATION_FIELDS = (
    "standalone_equals_incremental_plus_shadowed",
    "incremental_plus_default_equals_population",
    "monthly_rolls_to_overall",
)
_LEGACY_WATERFALL_SOURCE_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "asset_id",
        "asset_hash",
        "fragment_id",
        "sample_design_ref",
    }
)
_VALIDATION_WATERFALL_SOURCE_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "asset_id",
        "asset_hash",
        "fragment_id",
        "development_lineage_ref",
    }
)


def build_strategy_pool_validation_evidence(
    *,
    pool: Mapping[str, Any],
    frame: pd.DataFrame,
    pool_artifact_ref: Mapping[str, Any],
    sample_design_v2_ref: Mapping[str, Any],
    dataset_binding: Mapping[str, Any],
    legacy_development_ref: Mapping[str, Any],
    partition: str,
    population: str,
    comparison_mode: str,
    target_col: str,
    target_bad_value: int,
    month_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
    development_rows_excluded: bool,
) -> dict[str, Any]:
    """Build aggregate-only validation/OOT evidence from exact selected rows."""

    current_pool = validate_strategy_pool(pool)
    if current_pool["strategy_type"] in {
        "limit",
        "pricing",
        "segmentation",
    }:
        from marvis.packs.strategy.pool_validation_typed import (
            build_typed_strategy_pool_validation_evidence,
        )

        return build_typed_strategy_pool_validation_evidence(
            pool=current_pool,
            frame=frame,
            pool_artifact_ref=pool_artifact_ref,
            sample_design_v2_ref=sample_design_v2_ref,
            dataset_binding=dataset_binding,
            legacy_development_ref=legacy_development_ref,
            partition=partition,
            population=population,
            comparison_mode=comparison_mode,
            target_col=target_col,
            target_bad_value=target_bad_value,
            month_col=month_col,
            loan_amount_col=loan_amount_col,
            overdue_amount_col=overdue_amount_col,
            development_rows_excluded=development_rows_excluded,
        )
    selected_partition = _partition(partition)
    if population != "risk":
        raise StrategyError("Strategy Pool validation population must be risk")
    if comparison_mode != "absolute":
        raise StrategyError(
            "Strategy Pool validation comparison_mode must be absolute"
        )
    if development_rows_excluded is not True:
        raise StrategyError(
            "Strategy Pool validation must exclude risk/development rows"
        )
    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("Strategy Pool validation rows must be a DataFrame")
    if frame.empty:
        raise StrategyError(
            f"Strategy Pool {selected_partition} partition is empty"
        )

    pool_artifact = _pool_artifact_ref(pool_artifact_ref)
    sample_v2 = _sample_design_v2_ref(
        sample_design_v2_ref,
        partition=selected_partition,
    )
    if len(frame) != sample_v2["partition_count"]:
        raise StrategyError(
            "Strategy Pool validation selected rows do not match membership count"
        )
    dataset = _dataset_binding(dataset_binding)
    if dataset["task_id"] != current_pool["task_id"]:
        raise StrategyError(
            "Strategy Pool validation dataset belongs to another task"
        )
    development_ref = _risk_development_ref(
        legacy_development_ref
    )
    compatibility_ref = _legacy_impact_compatibility_ref(development_ref)
    sample_binding = _pool_sample_binding(
        current_pool,
        task_id=current_pool["task_id"],
    )
    for field in (
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    ):
        if sample_binding[field] != dataset[field]:
            raise StrategyError(
                f"Strategy Pool development lineage {field} does not match "
                "the V2 dataset binding"
            )

    target = _target_binding(
        {
            "column": target_col,
            "good_value": 1 - _target_bad_value(target_bad_value),
            "bad_value": target_bad_value,
            "missing_policy": "retain_population_exclude_risk_denominator",
        }
    )
    fields = _field_bindings(
        {
            "month_col": month_col,
            "loan_amount_col": loan_amount_col,
            "overdue_amount_col": overdue_amount_col,
        },
        target_col=target["column"],
    )
    legacy = build_strategy_pool_impact_assessment(
        pool=current_pool,
        frame=frame,
        sample_binding=sample_binding,
        sample_design_ref=compatibility_ref,
        target_col=target["column"],
        target_bad_value=target["bad_value"],
        month_col=fields["month_col"],
        loan_amount_col=fields["loan_amount_col"],
        overdue_amount_col=fields["overdue_amount_col"],
        comparison_mode="absolute",
    )
    if legacy["population"]["population_count"] != sample_v2["partition_count"]:
        raise StrategyError(
            "Strategy Pool validation population does not conserve membership"
        )

    waterfall = _validation_waterfall(
        legacy["waterfall"],
        legacy_development_ref=compatibility_ref,
        development_ref=development_ref,
    )
    body = {
        "schema_version": STRATEGY_POOL_VALIDATION_SCHEMA_VERSION,
        "producer_version": STRATEGY_POOL_VALIDATION_PRODUCER_VERSION,
        "identity": dict(legacy["identity"]),
        "source_bindings": {
            "pool_artifact": pool_artifact,
            "sample_design_v2": sample_v2,
            "dataset": dataset,
            "development_lineage": {
                "legacy_development_ref": development_ref,
                "sample_binding": sample_binding,
            },
            "target": target,
            "fields": fields,
        },
        "partition": selected_partition,
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle": _expected_lifecycle(selected_partition),
        "population_metrics": dict(legacy["population"]),
        "overall": dict(legacy["overall"]),
        "waterfall": waterfall,
        "default_unmatched": dict(legacy["default_unmatched"]),
        "monthly": dict(legacy["monthly"]),
        "conservation": {
            **dict(legacy["conservation"]),
            "selected_partition_equals_membership_count": True,
            "risk_partition_excludes_development": True,
        },
        "red_flags": list(legacy["red_flags"]),
    }
    evidence_id = "strategy-pool-validation-" + _sha256(
        _canonical_json(body)
    )[:24]
    document = {**body, "evidence_id": evidence_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return validate_strategy_pool_validation_evidence(document)


def validate_strategy_pool_validation_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate hashes, source binding shape, aggregates, and conservation."""

    obj = _json_object(payload, "Strategy Pool validation evidence")
    if (
        obj.get("schema_version")
        == "strategy.pool-validation-evidence.v2"
    ):
        from marvis.packs.strategy.pool_validation_typed import (
            validate_typed_strategy_pool_validation_evidence,
        )

        return validate_typed_strategy_pool_validation_evidence(obj)
    _exact_fields(obj, _TOP_LEVEL_FIELDS, "Strategy Pool validation evidence")
    if obj["schema_version"] != STRATEGY_POOL_VALIDATION_SCHEMA_VERSION:
        raise StrategyError(
            "Strategy Pool validation evidence schema_version is invalid"
        )
    if obj["producer_version"] != STRATEGY_POOL_VALIDATION_PRODUCER_VERSION:
        raise StrategyError(
            "Strategy Pool validation evidence producer_version is invalid"
        )
    evidence_id = _text(obj["evidence_id"], "evidence_id")
    if _EVIDENCE_ID_RE.fullmatch(evidence_id) is None:
        raise StrategyError("Strategy Pool validation evidence_id is invalid")
    content_hash = _hash(obj["content_hash"], "content_hash")
    without_hash = {
        key: value for key, value in obj.items() if key != "content_hash"
    }
    if not hmac.compare_digest(
        content_hash,
        _sha256(_canonical_json(without_hash)),
    ):
        raise StrategyError(
            "Strategy Pool validation content_hash does not match content"
        )
    body = {
        key: value
        for key, value in without_hash.items()
        if key != "evidence_id"
    }
    expected_id = "strategy-pool-validation-" + _sha256(
        _canonical_json(body)
    )[:24]
    if not hmac.compare_digest(evidence_id, expected_id):
        raise StrategyError(
            "Strategy Pool validation evidence_id does not match content"
        )

    partition = _partition(obj["partition"])
    if obj["population"] != "risk":
        raise StrategyError("Strategy Pool validation population must be risk")
    if obj["comparison_mode"] != "absolute":
        raise StrategyError(
            "Strategy Pool validation comparison_mode must be absolute"
        )
    if obj["lifecycle"] != _expected_lifecycle(partition):
        raise StrategyError(
            "Strategy Pool validation lifecycle must remain independent evidence"
        )
    conservation = _json_object(
        obj["conservation"],
        "Strategy Pool validation conservation",
    )
    _exact_fields(
        conservation,
        _CONSERVATION_FIELDS,
        "Strategy Pool validation conservation",
    )
    if not all(value is True for value in conservation.values()):
        raise StrategyError(
            "Strategy Pool validation conservation checks must all pass"
        )

    identity = _identity(obj["identity"])
    if identity["strategy_type"] not in {"approval", "reject"}:
        raise StrategyError(
            "Strategy Pool validation V1 supports approval/reject only"
        )
    sources = _json_object(
        obj["source_bindings"],
        "Strategy Pool validation source_bindings",
    )
    _exact_fields(
        sources,
        _SOURCE_BINDING_FIELDS,
        "Strategy Pool validation source_bindings",
    )
    _pool_artifact_ref(sources["pool_artifact"])
    sample_v2 = _sample_design_v2_ref(
        sources["sample_design_v2"],
        partition=partition,
    )
    dataset = _dataset_binding(sources["dataset"])
    development = _development_lineage(sources["development_lineage"])
    target = _target_binding(sources["target"])
    fields = _field_bindings(sources["fields"], target_col=target["column"])
    if identity["task_id"] != dataset["task_id"]:
        raise StrategyError(
            "Strategy Pool validation task and dataset bindings disagree"
        )
    for field in (
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    ):
        if development["sample_binding"][field] != dataset[field]:
            raise StrategyError(
                "Strategy Pool validation development and V2 dataset "
                f"bindings disagree on {field}"
            )

    legacy_waterfall = _legacy_waterfall(
        obj["waterfall"],
        legacy_development_ref=development["legacy_development_ref"],
    )
    compatibility_ref = _legacy_impact_compatibility_ref(
        development["legacy_development_ref"]
    )
    legacy_assessment = _legacy_assessment(
        identity=identity,
        sample_binding=development["sample_binding"],
        legacy_development_ref=compatibility_ref,
        target=target,
        fields=fields,
        population=obj["population_metrics"],
        overall=obj["overall"],
        waterfall=legacy_waterfall,
        default_unmatched=obj["default_unmatched"],
        monthly=obj["monthly"],
        conservation={
            field: conservation[field]
            for field in _LEGACY_CONSERVATION_FIELDS
        },
        red_flags=obj["red_flags"],
    )
    validated_legacy = validate_strategy_pool_impact_assessment(
        legacy_assessment
    )
    if (
        validated_legacy["population"]["population_count"]
        != sample_v2["partition_count"]
    ):
        raise StrategyError(
            "Strategy Pool validation population does not match V2 membership"
        )
    return obj


def canonical_strategy_pool_validation_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the byte-stable JSON representation of valid evidence."""

    return _canonical_json(validate_strategy_pool_validation_evidence(payload))


def _pool_sample_binding(
    pool: Mapping[str, Any],
    *,
    task_id: str,
) -> dict[str, Any]:
    identities = [
        _sample_binding(entry["source"]["evidence_identity"])
        for entry in pool["entries"]
    ]
    if not identities:
        raise StrategyError("cannot validate an empty Strategy Pool")
    if any(identity != identities[0] for identity in identities[1:]):
        raise StrategyError(
            "Strategy Pool entries do not share one development sample identity"
        )
    return {"task_id": _text(task_id, "task_id"), **identities[0]}


def _sample_binding(value: object) -> dict[str, Any]:
    obj = _json_object(value, "development sample binding")
    expected = _SAMPLE_BINDING_FIELDS - {"task_id"}
    if set(obj) == _SAMPLE_BINDING_FIELDS:
        task_id = _text(obj["task_id"], "development sample task_id")
    elif set(obj) == expected:
        task_id = None
    else:
        raise StrategyError(
            "development sample binding must contain exact governed fields"
        )
    normalized = {
        "dataset_id": _text(obj["dataset_id"], "development sample dataset_id"),
        "dataset_content_hash": _hash(
            obj["dataset_content_hash"],
            "development sample dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            obj["workspace_revision"],
            "development sample workspace_revision",
        ),
        "workspace_generation": _non_negative_int(
            obj["workspace_generation"],
            "development sample workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"],
            "development sample semantic_mapping_hash",
        ),
        "sample_context_hash": _hash(
            obj["sample_context_hash"],
            "development sample sample_context_hash",
        ),
    }
    return (
        normalized
        if task_id is None
        else {"task_id": task_id, **normalized}
    )


def _pool_artifact_ref(value: object) -> dict[str, str]:
    obj = _json_object(value, "Pool artifact ref")
    _exact_fields(obj, _POOL_ARTIFACT_REF_FIELDS, "Pool artifact ref")
    return {
        "artifact_id": _hash(obj["artifact_id"], "Pool artifact_id"),
        "artifact_content_hash": _hash(
            obj["artifact_content_hash"],
            "Pool artifact_content_hash",
        ),
    }


def _sample_design_v2_ref(
    value: object,
    *,
    partition: str,
) -> dict[str, Any]:
    obj = _json_object(value, "StrategySampleDesign V2 ref")
    _exact_fields(
        obj,
        _SAMPLE_DESIGN_V2_REF_FIELDS,
        "StrategySampleDesign V2 ref",
    )
    for field in (
        "membership_artifact_id",
        "membership_artifact_content_hash",
        "membership_content_hash",
        "bundle_artifact_id",
        "bundle_artifact_content_hash",
        "bundle_content_hash",
        "sample_design_content_hash",
    ):
        _hash(obj[field], f"StrategySampleDesign V2 ref.{field}")
    for field in ("membership_id", "bundle_id", "sample_design_id"):
        _text(obj[field], f"StrategySampleDesign V2 ref.{field}")
    expected_key = f"risk/{partition}"
    if obj["partition_key"] != expected_key:
        raise StrategyError(
            f"StrategySampleDesign V2 ref.partition_key must be {expected_key}"
        )
    partition_count = _non_negative_int(
        obj["partition_count"],
        "StrategySampleDesign V2 ref.partition_count",
    )
    if partition_count == 0:
        raise StrategyError(f"Strategy Pool {partition} partition is empty")
    universe = _positive_int(
        obj["analysis_universe_row_count"],
        "StrategySampleDesign V2 ref.analysis_universe_row_count",
    )
    if partition_count > universe:
        raise StrategyError(
            "StrategySampleDesign V2 partition_count exceeds analysis universe"
        )
    return {
        field: obj[field]
        for field in _SAMPLE_DESIGN_V2_REF_FIELDS
    }


def _dataset_binding(value: object) -> dict[str, Any]:
    obj = _json_object(value, "V2 dataset binding")
    _exact_fields(obj, _DATASET_BINDING_FIELDS, "V2 dataset binding")
    return {
        "task_id": _text(obj["task_id"], "V2 dataset task_id"),
        "dataset_id": _text(obj["dataset_id"], "V2 dataset dataset_id"),
        "dataset_content_hash": _hash(
            obj["dataset_content_hash"],
            "V2 dataset content_hash",
        ),
        "dataset_source_path": _text(
            obj["dataset_source_path"],
            "V2 dataset source_path",
        ),
        "dataset_registry_metadata_hash": _hash(
            obj["dataset_registry_metadata_hash"],
            "V2 dataset registry_metadata_hash",
        ),
        "workspace_revision": _non_negative_int(
            obj["workspace_revision"],
            "V2 dataset workspace_revision",
        ),
        "workspace_generation": _non_negative_int(
            obj["workspace_generation"],
            "V2 dataset workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            obj["semantic_mapping_hash"],
            "V2 dataset semantic_mapping_hash",
        ),
    }


def _development_lineage(value: object) -> dict[str, Any]:
    obj = _json_object(value, "development_lineage")
    _exact_fields(
        obj,
        _DEVELOPMENT_LINEAGE_FIELDS,
        "development_lineage",
    )
    return {
        "legacy_development_ref": _risk_development_ref(
            obj["legacy_development_ref"]
        ),
        "sample_binding": _sample_binding(obj["sample_binding"]),
    }


def _risk_development_ref(value: object) -> dict[str, str]:
    reference = StrategyRiskDevelopmentRef.from_value(value)
    if reference.partition not in {"development", "risk/development"}:
        raise StrategyError(
            "Strategy Pool validation development partition is invalid"
        )
    return reference.to_ref_dict()


def _legacy_impact_compatibility_ref(
    value: Mapping[str, Any],
) -> dict[str, str]:
    """Project a generic development ref into the legacy impact kernel.

    The Pool impact kernel still validates the historical ``development``
    spelling.  Validation evidence retains the exact generic source ref; only
    this in-memory deterministic calculation receives the compatibility
    projection.
    """

    reference = _risk_development_ref(value)
    return {**reference, "partition": "development"}


def _target_binding(value: object) -> dict[str, Any]:
    obj = _json_object(value, "validation target binding")
    _exact_fields(obj, _TARGET_FIELDS, "validation target binding")
    bad = _target_bad_value(obj["bad_value"])
    good = _target_bad_value(obj["good_value"])
    if {good, bad} != {0, 1}:
        raise StrategyError(
            "validation target good_value and bad_value must be complementary"
        )
    if obj["missing_policy"] != (
        "retain_population_exclude_risk_denominator"
    ):
        raise StrategyError("validation target missing_policy is invalid")
    return {
        "column": _text(obj["column"], "validation target column"),
        "good_value": good,
        "bad_value": bad,
        "missing_policy": obj["missing_policy"],
    }


def _field_bindings(
    value: object,
    *,
    target_col: str,
) -> dict[str, str | None]:
    obj = _json_object(value, "validation field bindings")
    _exact_fields(obj, _FIELD_BINDING_FIELDS, "validation field bindings")
    normalized = {
        field: _optional_text(obj[field], f"validation {field}")
        for field in sorted(_FIELD_BINDING_FIELDS)
    }
    selected = [
        target_col,
        *(item for item in normalized.values() if item is not None),
    ]
    if len(selected) != len(set(selected)):
        raise StrategyError("validation field bindings must be distinct")
    return normalized


def _identity(value: object) -> dict[str, Any]:
    obj = _json_object(value, "Strategy Pool validation identity")
    _exact_fields(obj, _IDENTITY_FIELDS, "Strategy Pool validation identity")
    normalized = {
        "pool_id": _text(obj["pool_id"], "identity.pool_id"),
        "task_id": _text(obj["task_id"], "identity.task_id"),
        "strategy_type": _text(
            obj["strategy_type"],
            "identity.strategy_type",
        ),
        "revision": _positive_int(obj["revision"], "identity.revision"),
        "revision_id": _text(obj["revision_id"], "identity.revision_id"),
        "snapshot_hash": _hash(
            obj["snapshot_hash"],
            "identity.snapshot_hash",
        ),
        "design_hash": _hash(obj["design_hash"], "identity.design_hash"),
        "strategy_spec_hash": _hash(
            obj["strategy_spec_hash"],
            "identity.strategy_spec_hash",
        ),
    }
    if normalized["strategy_type"] not in {
        "approval",
        "reject",
        "limit",
        "pricing",
        "segmentation",
    }:
        raise StrategyError("Strategy Pool validation type is unsupported")
    return normalized


def _validation_waterfall(
    value: object,
    *,
    legacy_development_ref: Mapping[str, Any],
    development_ref: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise StrategyError("legacy Pool impact waterfall must be a list")
    output: list[dict[str, Any]] = []
    for row in value:
        item = _json_object(row, "legacy Pool impact waterfall row")
        source = _json_object(
            item.get("source_ref"),
            "legacy Pool impact waterfall source_ref",
        )
        _exact_fields(
            source,
            _LEGACY_WATERFALL_SOURCE_FIELDS,
            "legacy Pool impact waterfall source_ref",
        )
        if source["sample_design_ref"] != dict(legacy_development_ref):
            raise StrategyError(
                "Pool waterfall source does not match legacy development lineage"
            )
        output.append(
            {
                **item,
                "source_ref": {
                    key: value
                    for key, value in source.items()
                    if key != "sample_design_ref"
                },
            }
        )
        output[-1]["source_ref"]["development_lineage_ref"] = dict(
            development_ref
        )
    return output


def _legacy_waterfall(
    value: object,
    *,
    legacy_development_ref: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise StrategyError(
            "Strategy Pool validation waterfall must be a non-empty list"
        )
    output: list[dict[str, Any]] = []
    for row in value:
        item = _json_object(row, "Strategy Pool validation waterfall row")
        source = _json_object(
            item.get("source_ref"),
            "Strategy Pool validation waterfall source_ref",
        )
        _exact_fields(
            source,
            _VALIDATION_WATERFALL_SOURCE_FIELDS,
            "Strategy Pool validation waterfall source_ref",
        )
        if source["development_lineage_ref"] != dict(
            legacy_development_ref
        ):
            raise StrategyError(
                "Strategy Pool validation waterfall lineage changed"
            )
        output.append(
            {
                **item,
                "source_ref": {
                    key: value
                    for key, value in source.items()
                    if key != "development_lineage_ref"
                },
            }
        )
        output[-1]["source_ref"]["sample_design_ref"] = dict(
            _legacy_impact_compatibility_ref(legacy_development_ref)
        )
    return output


def _legacy_assessment(
    *,
    identity: Mapping[str, Any],
    sample_binding: Mapping[str, Any],
    legacy_development_ref: Mapping[str, Any],
    target: Mapping[str, Any],
    fields: Mapping[str, Any],
    population: object,
    overall: object,
    waterfall: object,
    default_unmatched: object,
    monthly: object,
    conservation: object,
    red_flags: object,
) -> dict[str, Any]:
    body = {
        "schema_version": STRATEGY_POOL_IMPACT_SCHEMA_VERSION,
        "producer_version": STRATEGY_POOL_IMPACT_PRODUCER_VERSION,
        "identity": dict(identity),
        "bindings": {
            "sample": dict(sample_binding),
            "sample_design_ref": dict(legacy_development_ref),
            "target_col": target["column"],
            "target_bad_value": target["bad_value"],
            "month_col": fields["month_col"],
            "loan_amount_col": fields["loan_amount_col"],
            "overdue_amount_col": fields["overdue_amount_col"],
            "comparison_mode": "absolute",
        },
        "lifecycle": {
            "candidate_stage": "development",
            "observation_stage": "backtested",
            "validation_status": "unvalidated",
            "creates_strategy": False,
            "adopted": False,
            "deployed": False,
        },
        "population": population,
        "overall": overall,
        "waterfall": waterfall,
        "default_unmatched": default_unmatched,
        "monthly": monthly,
        "baseline": {
            "status": "not_requested",
            "binding": None,
            "overall": None,
        },
        "conservation": conservation,
        "red_flags": red_flags,
    }
    assessment_id = "strategy-impact-assessment-" + _sha256(
        _canonical_json(body)
    )[:24]
    document = {**body, "assessment_id": assessment_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return document


def _expected_lifecycle(partition: str) -> dict[str, Any]:
    return {
        "stage": partition,
        "validation_status": "independent_evidence",
        "mutates_pool": False,
        "creates_strategy": False,
        "adopts_strategy": False,
        "promotes_strategy": False,
        "deploys_strategy": False,
    }


def _partition(value: object) -> str:
    normalized = _text(value, "partition")
    if normalized not in _PARTITIONS:
        raise StrategyError("partition must be validation or oot")
    return normalized


def _target_bad_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise StrategyError("target bad_value must be integer 0 or 1")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        unexpected = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        details: list[str] = []
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise StrategyError(f"{name} has invalid fields ({'; '.join(details)})")


def _json_object(value: object, name: str) -> dict[str, Any]:
    try:
        encoded = _canonical_json(value)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    if normalized != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized < 1:
        raise StrategyError(f"{name} must be at least 1")
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "STRATEGY_POOL_VALIDATION_PRODUCER_VERSION",
    "STRATEGY_POOL_VALIDATION_SCHEMA_VERSION",
    "build_strategy_pool_validation_evidence",
    "canonical_strategy_pool_validation_json",
    "validate_strategy_pool_validation_evidence",
]
