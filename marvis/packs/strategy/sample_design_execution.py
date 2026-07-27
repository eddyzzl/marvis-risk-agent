"""Source-agnostic governed strategy-development execution binding.

The public seam accepts either the stable legacy sample design or a native
StrategySampleDesign V2 bundle.  Native execution always consumes the
persisted ``risk/development`` membership mask; predicates are replayed only
while authenticating the immutable evidence and are never used to select the
consumer frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hmac
import json
import math
from numbers import Real
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignExecutionBinding,
    bind_strategy_development_frame,
    load_historical_strategy_sample_design_execution_binding,
    load_strategy_sample_design_execution_binding,
    require_historical_strategy_sample_design_execution_binding_on_connection,
    require_strategy_sample_design_execution_binding_on_connection,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    StrategySampleDesignV2NativeArtifactBinding,
    authenticate_native_strategy_sample_design_v2_bundle_record,
    load_native_strategy_sample_design_v2_artifacts,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
)


_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StrategyRiskDevelopmentRef:
    """Exact reference to the governed strategy-development population."""

    artifact_id: str
    artifact_content_hash: str
    sample_design_id: str
    sample_design_content_hash: str
    partition: str

    @classmethod
    def from_value(cls, value: object) -> "StrategyRiskDevelopmentRef":
        obj = _object(value, "sample_design_ref")
        if set(obj) != _REF_FIELDS:
            missing = sorted(_REF_FIELDS - set(obj))
            unexpected = sorted(set(obj) - _REF_FIELDS)
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unsupported: " + ", ".join(unexpected))
            raise StrategyError(
                "sample_design_ref has invalid fields ("
                + "; ".join(details)
                + ")"
            )
        return cls(
            artifact_id=_hash(
                obj["artifact_id"],
                "sample_design_ref.artifact_id",
            ),
            artifact_content_hash=_hash(
                obj["artifact_content_hash"],
                "sample_design_ref.artifact_content_hash",
            ),
            sample_design_id=_text(
                obj["sample_design_id"],
                "sample_design_ref.sample_design_id",
            ),
            sample_design_content_hash=_hash(
                obj["sample_design_content_hash"],
                "sample_design_ref.sample_design_content_hash",
            ),
            partition=_text(
                obj["partition"],
                "sample_design_ref.partition",
            ),
        )

    def to_ref_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_content_hash": self.artifact_content_hash,
            "sample_design_id": self.sample_design_id,
            "sample_design_content_hash": self.sample_design_content_hash,
            "partition": self.partition,
        }


@dataclass(frozen=True)
class StrategyRiskDevelopmentExecutionBinding:
    """Authenticated execution semantics for one risk/development sample."""

    reference: StrategyRiskDevelopmentRef
    source_mode: str
    task_id: str
    dataset_id: str
    dataset_content_hash: str
    workspace_revision: int
    workspace_generation: int
    semantic_mapping_hash: str
    target_col: str
    target_bad_value: int
    drop_nan_labels: bool
    split_column: str | None
    partition_columns: tuple[str, ...]
    population_filter_columns: tuple[str, ...]
    excluded_feature_columns: tuple[str, ...]
    development_population_count: int
    active_population_count: int
    month_col: str | None
    weight_col: str | None
    loan_amount_col: str | None
    overdue_amount_col: str | None
    _legacy: StrategySampleDesignExecutionBinding | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _native: StrategySampleDesignV2NativeArtifactBinding | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_ref_dict(self) -> dict[str, str]:
        return self.reference.to_ref_dict()

    @property
    def source_ref(self) -> dict[str, str]:
        if self._legacy is not None:
            return self._legacy.source_ref
        return {
            "kind": "strategy_sample_design_v2",
            **self.reference.to_ref_dict(),
        }

    @property
    def source_ref_token(self) -> str:
        if self._legacy is not None:
            return self._legacy.source_ref_token
        return "strategy-sample-design:" + _canonical_json(self.source_ref)


def load_strategy_risk_development_execution_binding(
    runtime,
    *,
    task_id: str,
    sample_design_ref: object,
    dataset_id: str,
    dataset_content_hash: str,
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    target_col: str,
    drop_nan_labels: bool,
    month_col: str | None = None,
    weight_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
) -> StrategyRiskDevelopmentExecutionBinding:
    """Load a current legacy or native risk/development execution binding."""

    return _load_strategy_risk_development_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=sample_design_ref,
        dataset_id=dataset_id,
        dataset_content_hash=dataset_content_hash,
        workspace_revision=workspace_revision,
        workspace_generation=workspace_generation,
        semantic_mapping_hash=semantic_mapping_hash,
        target_col=target_col,
        drop_nan_labels=drop_nan_labels,
        month_col=month_col,
        weight_col=weight_col,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
        require_current_workspace=True,
    )


def load_historical_strategy_risk_development_execution_binding(
    runtime,
    *,
    task_id: str,
    sample_design_ref: object,
    dataset_id: str,
    dataset_content_hash: str,
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    target_col: str,
    drop_nan_labels: bool,
    month_col: str | None = None,
    weight_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
) -> StrategyRiskDevelopmentExecutionBinding:
    """Load immutable execution evidence without requiring workspace head."""

    return _load_strategy_risk_development_execution_binding(
        runtime,
        task_id=task_id,
        sample_design_ref=sample_design_ref,
        dataset_id=dataset_id,
        dataset_content_hash=dataset_content_hash,
        workspace_revision=workspace_revision,
        workspace_generation=workspace_generation,
        semantic_mapping_hash=semantic_mapping_hash,
        target_col=target_col,
        drop_nan_labels=drop_nan_labels,
        month_col=month_col,
        weight_col=weight_col,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
        require_current_workspace=False,
    )


def _load_strategy_risk_development_execution_binding(
    runtime,
    *,
    task_id: str,
    sample_design_ref: object,
    dataset_id: str,
    dataset_content_hash: str,
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    target_col: str,
    drop_nan_labels: bool,
    month_col: str | None,
    weight_col: str | None,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
    require_current_workspace: bool,
) -> StrategyRiskDevelopmentExecutionBinding:
    reference = StrategyRiskDevelopmentRef.from_value(sample_design_ref)
    task = _text(task_id, "task_id")
    expected = {
        "dataset_id": _text(dataset_id, "dataset_id"),
        "dataset_content_hash": _hash(
            dataset_content_hash,
            "dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            workspace_revision,
            "workspace_revision",
        ),
        "workspace_generation": _non_negative_int(
            workspace_generation,
            "workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            semantic_mapping_hash,
            "semantic_mapping_hash",
        ),
        "target_col": _text(target_col, "target_col"),
    }
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("drop_nan_labels must be boolean")
    optional = {
        "month_col": _optional_text(month_col, "month_col"),
        "weight_col": _optional_text(weight_col, "weight_col"),
        "loan_amount_col": _optional_text(
            loan_amount_col,
            "loan_amount_col",
        ),
        "overdue_amount_col": _optional_text(
            overdue_amount_col,
            "overdue_amount_col",
        ),
    }
    record = _artifact_record(
        runtime,
        task_id=task,
        reference=reference,
    )
    kind_origin = (record["kind"], record["origin_tool"])
    if kind_origin == (
        SAMPLE_DESIGN_ARTIFACT_KIND,
        SAMPLE_DESIGN_ORIGIN_TOOL,
    ):
        if reference.partition != "development":
            raise StrategyError(
                "legacy sample_design_ref.partition must be development"
            )
        loader = (
            load_strategy_sample_design_execution_binding
            if require_current_workspace
            else load_historical_strategy_sample_design_execution_binding
        )
        legacy = loader(
            runtime,
            task_id=task,
            sample_design_ref=reference.to_ref_dict(),
            dataset_id=expected["dataset_id"],
            dataset_content_hash=expected["dataset_content_hash"],
            workspace_revision=expected["workspace_revision"],
            workspace_generation=expected["workspace_generation"],
            semantic_mapping_hash=expected["semantic_mapping_hash"],
            target_col=expected["target_col"],
            drop_nan_labels=drop_nan_labels,
            **optional,
        )
        partition_columns = (
            () if legacy.split_column is None else (legacy.split_column,)
        )
        return StrategyRiskDevelopmentExecutionBinding(
            reference=reference,
            source_mode="legacy_anchored",
            task_id=legacy.task_id,
            dataset_id=legacy.dataset_id,
            dataset_content_hash=legacy.dataset_content_hash,
            workspace_revision=legacy.workspace_revision,
            workspace_generation=legacy.workspace_generation,
            semantic_mapping_hash=legacy.semantic_mapping_hash,
            target_col=legacy.target_col,
            target_bad_value=legacy.target_bad_value,
            drop_nan_labels=legacy.drop_nan_labels,
            split_column=legacy.split_column,
            partition_columns=partition_columns,
            population_filter_columns=(),
            excluded_feature_columns=tuple(
                sorted({legacy.target_col, *partition_columns})
            ),
            development_population_count=(
                legacy.development_population_count
            ),
            active_population_count=legacy.active_population_count,
            month_col=legacy.month_col,
            weight_col=legacy.weight_col,
            loan_amount_col=legacy.loan_amount_col,
            overdue_amount_col=legacy.overdue_amount_col,
            _legacy=legacy,
        )
    if kind_origin != (
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    ):
        raise StrategyError(
            "sample_design_ref artifact is not an exact supported "
            "sample-design kind/origin"
        )
    if reference.partition != "risk/development":
        raise StrategyError(
            "native sample_design_ref.partition must be risk/development"
        )
    authenticated = authenticate_native_strategy_sample_design_v2_bundle_record(
        runtime,
        task_id=task,
        record=record,
    )
    design = authenticated.bundle["sample_design"]
    if (
        authenticated.artifact_content_hash
        != reference.artifact_content_hash
        or design["sample_design_id"] != reference.sample_design_id
        or not hmac.compare_digest(
            design["content_hash"],
            reference.sample_design_content_hash,
        )
    ):
        raise StrategyError(
            "native sample_design_ref does not match the authenticated bundle"
        )
    provenance = authenticated.provenance
    if require_current_workspace:
        native = load_native_strategy_sample_design_v2_artifacts(
            runtime,
            task_id=task,
            membership_artifact_id=provenance["membership_artifact_id"],
            expected_membership_artifact_content_hash=provenance[
                "membership_artifact_content_hash"
            ],
            bundle_artifact_id=reference.artifact_id,
            expected_bundle_artifact_content_hash=(
                reference.artifact_content_hash
            ),
            expected_bundle_id=authenticated.bundle["bundle_id"],
            expected_sample_design_id=reference.sample_design_id,
            expected_sample_design_content_hash=(
                reference.sample_design_content_hash
            ),
        )
    else:
        from marvis.packs.strategy.sample_design_v2_native_tools import (
            load_historical_native_strategy_sample_design_v2_artifacts,
        )

        native = load_historical_native_strategy_sample_design_v2_artifacts(
            runtime,
            task_id=task,
            membership_artifact_id=provenance["membership_artifact_id"],
            expected_membership_artifact_content_hash=provenance[
                "membership_artifact_content_hash"
            ],
            bundle_artifact_id=reference.artifact_id,
            expected_bundle_artifact_content_hash=(
                reference.artifact_content_hash
            ),
            expected_bundle_id=authenticated.bundle["bundle_id"],
            expected_sample_design_id=reference.sample_design_id,
            expected_sample_design_content_hash=(
                reference.sample_design_content_hash
            ),
        )
    return _native_execution_binding(
        reference=reference,
        native=native,
        expected=expected,
        drop_nan_labels=drop_nan_labels,
        optional=optional,
    )


def _native_execution_binding(
    *,
    reference: StrategyRiskDevelopmentRef,
    native: StrategySampleDesignV2NativeArtifactBinding,
    expected: Mapping[str, Any],
    drop_nan_labels: bool,
    optional: Mapping[str, str | None],
) -> StrategyRiskDevelopmentExecutionBinding:
    source = native.source_binding
    design = native.bundle["sample_design"]
    semantics = design["sample_semantics"]
    target = design["target_selector"]
    designed_optional = semantics["field_bindings"]
    actual = {
        "dataset_id": source.dataset_id,
        "dataset_content_hash": source.dataset_content_hash,
        "workspace_revision": source.workspace_revision,
        "workspace_generation": source.workspace_generation,
        "semantic_mapping_hash": source.semantic_mapping_hash,
        "target_col": source.target_col,
    }
    for name, expected_value in expected.items():
        actual_value = actual[name]
        matched = (
            hmac.compare_digest(actual_value, expected_value)
            if isinstance(actual_value, str)
            and isinstance(expected_value, str)
            else type(actual_value) is type(expected_value)
            and actual_value == expected_value
        )
        if not matched:
            raise StrategyError(
                f"native sample-design {name} does not match execution binding"
            )
    if target["drop_missing"] is not drop_nan_labels:
        raise StrategyError(
            "native sample-design drop_nan_labels does not match execution binding"
        )
    if semantics["scope"] != "strategy_development":
        raise StrategyError(
            "native sample-design scope must be strategy_development"
        )
    optional_mapping = {
        "month_col": "month_field",
        "weight_col": "weight_field",
        "loan_amount_col": "loan_amount_field",
        "overdue_amount_col": "overdue_amount_field",
    }
    for request_name, design_name in optional_mapping.items():
        requested = optional[request_name]
        if requested is not None and designed_optional[design_name] != requested:
            raise StrategyError(
                "native sample-design "
                f"{request_name} does not match execution binding"
            )
    request = native.provenance["request"]
    partition_columns = tuple(
        sorted(_partition_columns(request["partitioning"]))
    )
    population_columns = tuple(
        sorted(
            {
                *(_predicate_columns(
                    request["approval_population"]["inclusion"]
                )),
                *(_predicate_columns(
                    request["approval_population"]["exclusion"]
                )),
                *(_predicate_columns(
                    request["risk_population"]["inclusion"]
                )),
                *(_predicate_columns(
                    request["risk_population"]["exclusion"]
                )),
            }
        )
    )
    return StrategyRiskDevelopmentExecutionBinding(
        reference=reference,
        source_mode="native_active_dataset",
        task_id=source.task_id,
        dataset_id=source.dataset_id,
        dataset_content_hash=source.dataset_content_hash,
        workspace_revision=source.workspace_revision,
        workspace_generation=source.workspace_generation,
        semantic_mapping_hash=source.semantic_mapping_hash,
        target_col=source.target_col,
        target_bad_value=source.target_bad_value,
        drop_nan_labels=source.drop_nan_labels,
        split_column=None,
        partition_columns=partition_columns,
        population_filter_columns=population_columns,
        excluded_feature_columns=tuple(
            sorted(
                {
                    source.target_col,
                    *partition_columns,
                    *population_columns,
                }
            )
        ),
        development_population_count=int(
            native.membership["header"]["counts"]["risk"]["development"]
        ),
        active_population_count=int(
            native.membership["header"]["row_count"]
        ),
        month_col=designed_optional["month_field"],
        weight_col=designed_optional["weight_field"],
        loan_amount_col=designed_optional["loan_amount_field"],
        overdue_amount_col=designed_optional["overdue_amount_field"],
        _native=native,
    )


def bind_strategy_risk_development_frame(
    frame: pd.DataFrame,
    *,
    binding: StrategyRiskDevelopmentExecutionBinding,
    normalize_target: bool = True,
) -> pd.DataFrame:
    """Select the authenticated development rows in original source order."""

    if not isinstance(binding, StrategyRiskDevelopmentExecutionBinding):
        raise StrategyError(
            "strategy risk-development execution binding is invalid"
        )
    if binding._legacy is not None:
        return bind_strategy_development_frame(
            frame,
            binding=binding._legacy,
            normalize_target=normalize_target,
        )
    if binding._native is None:
        raise StrategyError(
            "strategy risk-development execution source is unavailable"
        )
    if not isinstance(frame, pd.DataFrame):
        raise StrategyError(
            "strategy execution frame must be a pandas DataFrame"
        )
    if not isinstance(normalize_target, bool):
        raise StrategyError("normalize_target must be boolean")
    if len(frame) != binding.active_population_count:
        raise StrategyError(
            "strategy execution frame population does not match sample design"
        )
    if binding.target_col not in frame.columns:
        raise StrategyError(
            "strategy execution frame is missing sample-design columns: "
            + binding.target_col
        )
    raw_mask = binding._native.membership["masks"].get(
        "risk/development"
    )
    mask = np.asarray(raw_mask)
    if (
        mask.dtype != np.bool_
        or mask.ndim != 1
        or len(mask) != len(frame)
        or int(mask.sum()) != binding.development_population_count
    ):
        raise StrategyError(
            "native strategy risk/development membership changed"
        )
    selected = frame.iloc[np.flatnonzero(mask)].copy().reset_index(drop=True)
    if not normalize_target:
        return selected
    selected[binding.target_col] = _normalized_target(
        selected[binding.target_col],
        bad_value=binding.target_bad_value,
    )
    return selected


def revalidate_strategy_risk_development_execution_binding(
    runtime,
    binding: StrategyRiskDevelopmentExecutionBinding,
) -> StrategyRiskDevelopmentExecutionBinding:
    """Re-authenticate a current execution binding before persistence."""

    _require_binding(binding)
    return load_strategy_risk_development_execution_binding(
        runtime,
        task_id=binding.task_id,
        sample_design_ref=binding.to_ref_dict(),
        dataset_id=binding.dataset_id,
        dataset_content_hash=binding.dataset_content_hash,
        workspace_revision=binding.workspace_revision,
        workspace_generation=binding.workspace_generation,
        semantic_mapping_hash=binding.semantic_mapping_hash,
        target_col=binding.target_col,
        drop_nan_labels=binding.drop_nan_labels,
        month_col=binding.month_col,
        weight_col=binding.weight_col,
        loan_amount_col=binding.loan_amount_col,
        overdue_amount_col=binding.overdue_amount_col,
    )


def require_strategy_risk_development_execution_binding_on_connection(
    conn,
    binding: StrategyRiskDevelopmentExecutionBinding,
) -> None:
    """Recheck a current source while the caller holds a writer lock."""

    _require_binding(binding)
    if binding._legacy is not None:
        require_strategy_sample_design_execution_binding_on_connection(
            conn,
            binding._legacy,
        )
        return
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        require_native_strategy_sample_design_v2_artifact_binding_on_connection,
    )

    require_native_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding._native,
    )


def require_historical_strategy_risk_development_execution_binding_on_connection(
    conn,
    binding: StrategyRiskDevelopmentExecutionBinding,
) -> None:
    """Recheck immutable source evidence without requiring workspace head."""

    _require_binding(binding)
    if binding._legacy is not None:
        require_historical_strategy_sample_design_execution_binding_on_connection(
            conn,
            binding._legacy,
        )
        return
    from marvis.packs.strategy.sample_design_v2_native_tools import (
        require_historical_native_strategy_sample_design_v2_artifact_binding_on_connection,
    )

    require_historical_native_strategy_sample_design_v2_artifact_binding_on_connection(
        conn,
        binding._native,
    )


def _artifact_record(
    runtime,
    *,
    task_id: str,
    reference: StrategyRiskDevelopmentRef,
) -> dict[str, Any]:
    try:
        record = runtime.task_artifacts.get_for_task(
            task_id,
            reference.artifact_id,
        )
    except (TaskArtifactDataError, TaskArtifactNotFoundError) as exc:
        raise StrategyError(str(exc)) from exc
    if (
        not isinstance(record, Mapping)
        or record.get("id") != reference.artifact_id
        or record.get("task_id") != task_id
        or record.get("content_hash") != reference.artifact_content_hash
    ):
        raise StrategyError("sample_design_ref artifact binding changed")
    return dict(record)


def _partition_columns(partitioning: Mapping[str, Any]) -> set[str]:
    if partitioning["method"] == "time_ranges":
        return {_text(partitioning["column"], "partitioning.column")}
    columns: set[str] = set()
    for predicate in partitioning["selectors"].values():
        columns.update(_predicate_columns(predicate))
    return columns


def _predicate_columns(value: object) -> set[str]:
    columns: set[str] = set()
    if isinstance(value, Mapping):
        column = value.get("column")
        if isinstance(column, str):
            columns.add(column)
        for child in value.values():
            columns.update(_predicate_columns(child))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            columns.update(_predicate_columns(child))
    return columns


def _normalized_target(series: pd.Series, *, bad_value: int) -> pd.Series:
    values: list[int | float] = []
    for index, raw in enumerate(series.tolist()):
        if _is_missing(raw):
            values.append(float("nan"))
            continue
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, Real)
            or not math.isfinite(float(raw))
            or float(raw) not in {0.0, 1.0}
        ):
            raise StrategyError(
                "strategy execution target must contain only 0, 1, or "
                f"missing values (row {index})"
            )
        values.append(1 if int(float(raw)) == bad_value else 0)
    return pd.Series(
        values,
        dtype=(
            "float64"
            if any(pd.isna(value) for value in values)
            else "int64"
        ),
    )


def _require_binding(binding: object) -> None:
    if not isinstance(binding, StrategyRiskDevelopmentExecutionBinding):
        raise StrategyError(
            "strategy risk-development execution binding is invalid"
        )
    if (binding._legacy is None) == (binding._native is None):
        raise StrategyError(
            "strategy risk-development execution source is invalid"
        )


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise StrategyError(f"{name} must be an object")
    return dict(value)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "strategy execution binding is not canonical JSON"
        ) from exc


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise StrategyError(f"{name} must be non-empty text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _hash(value: object, name: str) -> str:
    text = _text(value, name)
    if not _HASH_RE.fullmatch(text):
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return text


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyError(f"{name} must be a non-negative integer")
    return value


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


__all__ = [
    "StrategyRiskDevelopmentExecutionBinding",
    "StrategyRiskDevelopmentRef",
    "bind_strategy_risk_development_frame",
    "load_historical_strategy_risk_development_execution_binding",
    "load_strategy_risk_development_execution_binding",
    "require_historical_strategy_risk_development_execution_binding_on_connection",
    "require_strategy_risk_development_execution_binding_on_connection",
    "revalidate_strategy_risk_development_execution_binding",
]
