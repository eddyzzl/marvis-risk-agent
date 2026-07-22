"""Shared fail-closed binding from strategy execution to one sample design.

The sample-design artifact is the governed authority for the development
population and target polarity.  Downstream strategy tools may select columns,
but they must not reinterpret the sample, maturity, split, or target semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Real
import re

import numpy as np
import pandas as pd

from marvis.data.workspace import (
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
)
from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
    StrategySampleDesignArtifactBinding,
    load_strategy_sample_design_artifact,
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
_OPTIONAL_COLUMN_FIELDS = {
    "month_col": "month_field",
    "weight_col": "weight_field",
    "loan_amount_col": "loan_amount_field",
    "overdue_amount_col": "overdue_amount_field",
}
_MAX_SAFE_JSON_INTEGER = 2**53 - 1


@dataclass(frozen=True)
class StrategySampleDesignRef:
    """Exact caller-confirmed reference to one immutable design partition."""

    artifact_id: str
    artifact_content_hash: str
    sample_design_id: str
    sample_design_content_hash: str
    partition: str

    @classmethod
    def from_value(cls, value: object) -> "StrategySampleDesignRef":
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            raise StrategyError("sample_design_ref must be an object")
        if set(value) != _REF_FIELDS:
            missing = sorted(_REF_FIELDS - set(value))
            unexpected = sorted(set(value) - _REF_FIELDS)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unsupported: " + ", ".join(unexpected))
            raise StrategyError(
                "sample_design_ref has invalid fields (" + "; ".join(details) + ")"
            )
        partition = _text(value["partition"], "sample_design_ref.partition")
        if partition != "development":
            raise StrategyError(
                "sample_design_ref.partition must be development for strategy development"
            )
        return cls(
            artifact_id=_hash(value["artifact_id"], "sample_design_ref.artifact_id"),
            artifact_content_hash=_hash(
                value["artifact_content_hash"],
                "sample_design_ref.artifact_content_hash",
            ),
            sample_design_id=_text(
                value["sample_design_id"], "sample_design_ref.sample_design_id"
            ),
            sample_design_content_hash=_hash(
                value["sample_design_content_hash"],
                "sample_design_ref.sample_design_content_hash",
            ),
            partition=partition,
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
class StrategySampleDesignExecutionBinding:
    """Verified design plus the exact execution semantics it controls."""

    reference: StrategySampleDesignRef
    artifact: StrategySampleDesignArtifactBinding
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
    development_values: tuple[str | bool | int | float, ...]
    development_population_count: int
    active_population_count: int
    month_col: str | None
    weight_col: str | None
    loan_amount_col: str | None
    overdue_amount_col: str | None

    def to_ref_dict(self) -> dict[str, str]:
        return self.reference.to_ref_dict()

    @property
    def source_ref(self) -> dict[str, str]:
        return {
            "kind": "strategy_sample_design",
            **self.reference.to_ref_dict(),
        }

    @property
    def source_ref_token(self) -> str:
        return "strategy-sample-design:" + json.dumps(
            self.source_ref,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def load_strategy_sample_design_execution_binding(
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
) -> StrategySampleDesignExecutionBinding:
    """Load and match one mature development sample to live tool bindings."""

    reference = StrategySampleDesignRef.from_value(sample_design_ref)
    normalized_task_id = _text(task_id, "task_id")
    normalized_dataset_id = _text(dataset_id, "dataset_id")
    normalized_dataset_hash = _hash(dataset_content_hash, "dataset_content_hash")
    normalized_revision = _non_negative_int(workspace_revision, "workspace_revision")
    normalized_generation = _non_negative_int(
        workspace_generation, "workspace_generation"
    )
    normalized_semantic_hash = _hash(
        semantic_mapping_hash, "semantic_mapping_hash"
    )
    normalized_target = _text(target_col, "target_col")
    if not isinstance(drop_nan_labels, bool):
        raise StrategyError("drop_nan_labels must be boolean")
    requested_optional = {
        "month_col": _optional_text(month_col, "month_col"),
        "weight_col": _optional_text(weight_col, "weight_col"),
        "loan_amount_col": _optional_text(loan_amount_col, "loan_amount_col"),
        "overdue_amount_col": _optional_text(
            overdue_amount_col, "overdue_amount_col"
        ),
    }

    artifact = load_strategy_sample_design_artifact(
        runtime,
        task_id=normalized_task_id,
        artifact_id=reference.artifact_id,
        expected_artifact_content_hash=reference.artifact_content_hash,
        expected_sample_design_id=reference.sample_design_id,
        expected_sample_design_content_hash=reference.sample_design_content_hash,
    )
    design = artifact.bundle["sample_design"]
    identity = design["identity"]
    dataset_ref = identity["dataset_ref"]
    workspace_ref = identity["workspace_ref"]
    target_definition = design["target_definition"]
    expected_bindings: tuple[tuple[str, object, object], ...] = (
        ("task_id", identity["task_id"], normalized_task_id),
        ("dataset_id", dataset_ref["dataset_id"], normalized_dataset_id),
        (
            "dataset_content_hash",
            dataset_ref["content_hash"],
            normalized_dataset_hash,
        ),
        ("workspace_revision", workspace_ref["revision"], normalized_revision),
        (
            "workspace_generation",
            workspace_ref["generation"],
            normalized_generation,
        ),
        (
            "semantic_mapping_hash",
            workspace_ref["semantic_mapping_hash"],
            normalized_semantic_hash,
        ),
        ("target_col", target_definition["column"], normalized_target),
        (
            "drop_nan_labels",
            target_definition["drop_nan_labels"],
            drop_nan_labels,
        ),
    )
    for field, designed, requested in expected_bindings:
        if isinstance(designed, str) and isinstance(requested, str):
            matched = hmac.compare_digest(designed, requested)
        else:
            matched = type(designed) is type(requested) and designed == requested
        if not matched:
            raise StrategyError(
                f"strategy sample-design {field} does not match execution binding"
            )
    if design["maturity"] != "confirmed_matured":
        raise StrategyError(
            "strategy sample-design must be confirmed_matured for strategy development"
        )
    if design["scope"] != "strategy_development":
        raise StrategyError(
            "strategy sample-design scope must be strategy_development"
        )

    designed_optional = design["optional_fields"]
    for request_field, design_field in _OPTIONAL_COLUMN_FIELDS.items():
        requested = requested_optional[request_field]
        if requested is not None and designed_optional[design_field] != requested:
            raise StrategyError(
                f"strategy sample-design {request_field} does not match execution binding"
            )

    split = design["split_definition"]
    if split["status"] == "available":
        split_column = split["column"]
        development_values = tuple(split["development_values"])
        split_counts = design["split_population_counts"]
        if not development_values or not isinstance(split_counts, Mapping):
            raise StrategyError(
                "strategy sample-design development partition is unavailable"
            )
        development_count = int(split_counts["development"])
    else:
        split_column = None
        development_values = ()
        development_count = int(design["active_dataset_boundary"]["population_count"])

    return StrategySampleDesignExecutionBinding(
        reference=reference,
        artifact=artifact,
        task_id=normalized_task_id,
        dataset_id=normalized_dataset_id,
        dataset_content_hash=normalized_dataset_hash,
        workspace_revision=normalized_revision,
        workspace_generation=normalized_generation,
        semantic_mapping_hash=normalized_semantic_hash,
        target_col=normalized_target,
        target_bad_value=int(target_definition["bad_value"]),
        drop_nan_labels=drop_nan_labels,
        split_column=split_column,
        development_values=development_values,
        development_population_count=development_count,
        active_population_count=int(
            design["active_dataset_boundary"]["population_count"]
        ),
        month_col=designed_optional["month_field"],
        weight_col=designed_optional["weight_field"],
        loan_amount_col=designed_optional["loan_amount_field"],
        overdue_amount_col=designed_optional["overdue_amount_field"],
    )


def revalidate_strategy_sample_design_execution_binding(
    runtime,
    binding: StrategySampleDesignExecutionBinding,
) -> StrategySampleDesignExecutionBinding:
    """Re-authenticate an execution binding immediately before persistence."""

    return load_strategy_sample_design_execution_binding(
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


def require_strategy_sample_design_execution_binding_on_connection(
    conn,
    binding: StrategySampleDesignExecutionBinding,
) -> None:
    """Recheck artifact and workspace state while a caller holds a DB lock."""

    row = conn.execute(
        """
        SELECT task_id, kind, path, content_hash, origin_tool, provenance_json
          FROM task_artifacts
         WHERE task_id = ? AND id = ?
        """,
        (binding.task_id, binding.reference.artifact_id),
    ).fetchone()
    if row is None:
        raise StrategyError("strategy sample-design artifact disappeared")
    try:
        provenance = json.loads(str(row["provenance_json"]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "strategy sample-design artifact provenance is invalid"
        ) from exc
    expected_artifact = binding.artifact
    if (
        str(row["task_id"]) != binding.task_id
        or str(row["kind"]) != SAMPLE_DESIGN_ARTIFACT_KIND
        or str(row["path"]) != str(expected_artifact.path)
        or str(row["content_hash"]) != expected_artifact.content_hash
        or str(row["origin_tool"]) != SAMPLE_DESIGN_ORIGIN_TOOL
        or _canonical_json(provenance) != _canonical_json(expected_artifact.provenance)
        or sha256_file(expected_artifact.path) != expected_artifact.content_hash
    ):
        raise StrategyError("strategy sample-design artifact binding changed")

    workspace = conn.execute(
        """
        SELECT revision, active_dataset_id, active_dataset_content_hash,
               analysis_generation, semantic_mapping_json
          FROM data_workspaces
         WHERE task_id = ?
        """,
        (binding.task_id,),
    ).fetchone()
    if workspace is None:
        raise StrategyError("strategy sample-design DataWorkspace disappeared")
    try:
        mapping = data_semantic_mapping_from_dict(
            json.loads(str(workspace["semantic_mapping_json"]))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError(
            "strategy sample-design semantic mapping is invalid"
        ) from exc
    if (
        int(workspace["revision"]) != binding.workspace_revision
        or int(workspace["analysis_generation"]) != binding.workspace_generation
        or str(workspace["active_dataset_id"]) != binding.dataset_id
        or str(workspace["active_dataset_content_hash"])
        != binding.dataset_content_hash
        or not hmac.compare_digest(
            data_semantic_mapping_hash(mapping),
            binding.semantic_mapping_hash,
        )
        or mapping.target_col != binding.target_col
    ):
        raise StrategyError("strategy sample-design DataWorkspace binding changed")


def bind_strategy_development_frame(
    frame: pd.DataFrame,
    *,
    binding: StrategySampleDesignExecutionBinding,
    normalize_target: bool = True,
) -> pd.DataFrame:
    """Select development rows and optionally normalize target to internal ``1=bad``."""

    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("strategy execution frame must be a pandas DataFrame")
    if not isinstance(normalize_target, bool):
        raise StrategyError("normalize_target must be boolean")
    if len(frame) != binding.active_population_count:
        raise StrategyError(
            "strategy execution frame population does not match sample design"
        )
    required = {binding.target_col}
    if binding.split_column is not None:
        required.add(binding.split_column)
    missing = sorted(required - set(str(column) for column in frame.columns))
    if missing:
        raise StrategyError(
            "strategy execution frame is missing sample-design columns: "
            + ", ".join(missing)
        )

    if binding.split_column is None:
        selected = frame.copy()
    else:
        allowed = {_scalar_identity(value) for value in binding.development_values}
        mask: list[bool] = []
        for index, raw in enumerate(frame[binding.split_column].tolist()):
            try:
                identity = _scalar_identity(_json_scalar(raw))
            except StrategyError as exc:
                raise StrategyError(
                    "strategy execution split column contains an invalid value "
                    f"at row {index}"
                ) from exc
            mask.append(identity in allowed)
        selected = frame.loc[pd.Series(mask, index=frame.index, dtype=bool)].copy()
    if len(selected) != binding.development_population_count:
        raise StrategyError(
            "strategy execution development population does not match sample design"
        )
    selected = selected.reset_index(drop=True)
    if not normalize_target:
        return selected

    normalized_target: list[int | float] = []
    for index, raw in enumerate(selected[binding.target_col].tolist()):
        if _is_missing_scalar(raw):
            normalized_target.append(float("nan"))
            continue
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, Real)
            or not math.isfinite(float(raw))
            or float(raw) not in {0.0, 1.0}
        ):
            raise StrategyError(
                "strategy execution target must contain only 0, 1, or missing "
                f"values (row {index})"
            )
        normalized_target.append(
            1 if int(float(raw)) == binding.target_bad_value else 0
        )
    selected[binding.target_col] = pd.Series(
        normalized_target,
        dtype=("float64" if any(pd.isna(value) for value in normalized_target) else "int64"),
    )
    return selected


def sample_design_ref_hash(value: object) -> str:
    """Return the canonical SHA-256 identity of an exact sample-design ref."""

    reference = StrategySampleDesignRef.from_value(value)
    raw = json.dumps(
        reference.to_ref_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
        raise StrategyError("sample-design binding is not canonical JSON") from exc


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


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _json_scalar(value: object) -> str | bool | int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if _is_missing_scalar(value) or not isinstance(value, (str, bool, int, float)):
        raise StrategyError("split value must be a non-null JSON scalar")
    if isinstance(value, str):
        if not value or "\x00" in value:
            raise StrategyError("split value must be non-empty text")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise StrategyError("split value exceeds exact JSON numeric range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise StrategyError("split value must be a finite exact JSON number")
        if value == 0 or value.is_integer():
            return int(value)
    return value


def _scalar_identity(value: str | bool | int | float) -> str:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    else:
        kind = "string"
    return json.dumps(
        [kind, value],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "StrategySampleDesignExecutionBinding",
    "StrategySampleDesignRef",
    "bind_strategy_development_frame",
    "load_strategy_sample_design_execution_binding",
    "require_strategy_sample_design_execution_binding_on_connection",
    "revalidate_strategy_sample_design_execution_binding",
    "sample_design_ref_hash",
]
