from __future__ import annotations

import hashlib
import json
import math
import uuid
from decimal import Decimal
from pathlib import Path

import numpy as np

from marvis.artifacts import ArtifactUnitOfWork
from marvis.feature.preprocessing import (
    read_preprocessing_chain,
    sidecar_path,
    write_preprocessing_chain,
)
from marvis.files import sha256_file
from marvis.packs.modeling._common import _effective_seed, _jsonable
from marvis.packs.modeling._runtime import _runtime, _task_dataset
from marvis.packs.modeling.errors import (
    ModelingError,
    SpecialValueDecisionRequiredError,
)


SPECIAL_VALUE_POLICY_VERSION = "1.0"
_ALLOWED_ACTIONS = frozenset({"mask", "retain", "drop"})


def tool_resolve_special_values(inputs: dict, ctx) -> dict:
    """Resolve every selected sentinel-bearing feature before refinement/training.

    ``screen_features`` only detects suspicious values.  This tool turns that
    observation into an auditable policy:

    * ``mask`` writes a real derived parquet and an exact replayable ``sentinel``
      preprocessing step;
    * ``retain`` freezes an explicit human confirmation, reason and source-data
      fingerprint;
    * ``drop`` removes the feature from the downstream feature list.

    Missing or incomplete decisions raise a typed gate error.  Consequently AUTO
    mode cannot silently accept a risky value merely because it can advance
    ordinary non-HITL steps.
    """

    runtime = _runtime(ctx)
    dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
    dataset_path = runtime.registry.resolve_path(dataset.id)
    features = _unique_strings(inputs.get("features") or [])
    detected = _relevant_sentinel_columns(
        inputs.get("sentinel_columns"),
        features=features,
    )
    if not detected:
        return {
            "result_dataset_id": dataset.id,
            "selected": features,
            "governance": {},
            "policy_fingerprint": "",
            "masked": [],
            "retained": [],
            "dropped": [],
        }

    decisions = inputs.get("decisions")
    decisions = decisions if isinstance(decisions, dict) else {}
    normalized, problems = _normalize_decisions(detected, decisions)
    if problems:
        raise SpecialValueDecisionRequiredError(
            columns=sorted(problems),
            sentinel_columns=detected,
            problems=problems,
        )

    dataset_hash = str(dataset.content_hash or "") or sha256_file(dataset_path)
    governance = _governance_evidence(
        normalized,
        dataset_id=dataset.id,
        dataset_content_hash=dataset_hash,
    )
    policy_fingerprint = _fingerprint({
        "version": SPECIAL_VALUE_POLICY_VERSION,
        "dataset_id": dataset.id,
        "dataset_content_hash": dataset_hash,
        "features": features,
        "governance": governance,
    })
    for evidence in governance.values():
        evidence["policy_fingerprint"] = policy_fingerprint

    masked = [
        column for column, decision in normalized.items()
        if decision["action"] == "mask"
    ]
    retained = [
        column for column, decision in normalized.items()
        if decision["action"] == "retain"
    ]
    dropped = [
        column for column, decision in normalized.items()
        if decision["action"] == "drop"
    ]
    selected = [feature for feature in features if feature not in set(dropped)]
    if not selected:
        raise ModelingError("特殊值治理后没有剩余特征；请至少保留一个可建模特征。")

    result_dataset_id = dataset.id
    if masked:
        registered = _write_masked_dataset(
            runtime,
            ctx,
            dataset=dataset,
            dataset_path=dataset_path,
            mask_values={
                column: normalized[column]["values"]
                for column in masked
            },
            governance=governance,
            policy_fingerprint=policy_fingerprint,
            seed=_effective_seed(inputs, ctx),
        )
        result_dataset_id = registered.id

    for evidence in governance.values():
        evidence["resolved_dataset_id"] = result_dataset_id
    return {
        "result_dataset_id": result_dataset_id,
        "selected": selected,
        "governance": _jsonable(governance),
        "policy_fingerprint": policy_fingerprint,
        "masked": masked,
        "retained": retained,
        "dropped": dropped,
    }


def _relevant_sentinel_columns(value, *, features: list[str]) -> dict[str, list[list[float]]]:
    if not isinstance(value, dict):
        return {}
    selected = set(features)
    out: dict[str, list[list[float]]] = {}
    for raw_column, rows in value.items():
        column = str(raw_column)
        if column not in selected:
            continue
        normalized_rows: list[list[float]] = []
        for row in rows if isinstance(rows, (list, tuple)) else []:
            raw_value = row[0] if isinstance(row, (list, tuple)) and row else row
            share = row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else None
            try:
                number = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            normalized_rows.append(
                [number, float(share)]
                if share is not None
                else [number]
            )
        if normalized_rows:
            out[column] = normalized_rows
    return out


def _normalize_decisions(
    detected: dict[str, list[list[float]]],
    decisions: dict,
) -> tuple[dict[str, dict], dict[str, str]]:
    normalized: dict[str, dict] = {}
    problems: dict[str, str] = {}
    for column, rows in detected.items():
        detected_values = _detected_values(rows)
        raw = decisions.get(column)
        if not isinstance(raw, dict):
            problems[column] = "missing_decision"
            continue
        action = str(raw.get("action") or "").strip().lower()
        if action not in _ALLOWED_ACTIONS:
            problems[column] = "invalid_action"
            continue
        supplied = raw.get("values")
        try:
            decided_values = (
                detected_values
                if supplied is None
                else _normalized_numeric_values(supplied)
            )
        except (TypeError, ValueError):
            problems[column] = "invalid_values"
            continue
        if decided_values != detected_values:
            problems[column] = "values_must_exactly_match_detection"
            continue
        confirmed = raw.get("confirmed") is True
        reason = str(raw.get("reason") or "").strip()
        if action == "retain" and not confirmed:
            problems[column] = "retain_requires_explicit_confirmation"
            continue
        if action == "retain" and not reason:
            problems[column] = "retain_requires_reason"
            continue
        normalized[column] = {
            "action": action,
            "values": decided_values,
            "confirmed": confirmed,
            "reason": reason,
        }
    return normalized, problems


def _governance_evidence(
    decisions: dict[str, dict],
    *,
    dataset_id: str,
    dataset_content_hash: str,
) -> dict[str, dict]:
    evidence: dict[str, dict] = {}
    for column in sorted(decisions):
        decision = decisions[column]
        row = {
            "policy_version": SPECIAL_VALUE_POLICY_VERSION,
            "column": column,
            "action": decision["action"],
            "detected_values": list(decision["values"]),
            "confirmed": bool(decision["confirmed"]),
            "reason": str(decision["reason"]),
            "source_dataset_id": dataset_id,
            "source_dataset_content_hash": dataset_content_hash,
        }
        row["decision_fingerprint"] = special_value_decision_fingerprint(row)
        row["fingerprint"] = row["decision_fingerprint"]
        evidence[column] = row
    return evidence


_DECISION_FINGERPRINT_FIELDS = (
    "policy_version",
    "column",
    "action",
    "detected_values",
    "confirmed",
    "reason",
    "source_dataset_id",
    "source_dataset_content_hash",
)


def special_value_decision_fingerprint(evidence: dict) -> str:
    """Return the canonical fingerprint for one auditable column decision."""

    payload = {
        field: evidence.get(field)
        for field in _DECISION_FINGERPRINT_FIELDS
    }
    return _fingerprint(payload)


def _write_masked_dataset(
    runtime,
    ctx,
    *,
    dataset,
    dataset_path: Path,
    mask_values: dict[str, list[float]],
    governance: dict[str, dict],
    policy_fingerprint: str,
    seed: int,
):
    out_dir = runtime.datasets_root / str(ctx.task_id) / "modeling"
    out_name = f"{dataset.id}_special_values_{uuid.uuid4().hex[:8]}.parquet"
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(out_dir, out_name)
    sidecar_artifact = uow.stage_file(
        out_dir,
        sidecar_path(Path(out_name)).name,
    )
    try:
        _stream_mask_to_parquet(
            runtime.backend,
            dataset_path,
            artifact.path,
            mask_values,
        )
        source_chain = read_preprocessing_chain(dataset_path)
        sentinel_step = {
            "kind": "sentinel",
            "columns": sorted(mask_values),
            "params": {
                column: list(mask_values[column])
                for column in sorted(mask_values)
            },
        }
        write_preprocessing_chain(
            sidecar_artifact.path,
            [*source_chain, sentinel_step],
        )

        def audit_factory(registered):
            return {
                "kind": "modeling.special_values.resolved",
                "target_ref": registered.id,
                "outcome": "succeeded",
                "detail": {
                    "source_dataset_id": dataset.id,
                    "policy_fingerprint": policy_fingerprint,
                    "governance": governance,
                },
            }

        register_on_connection = getattr(
            runtime.registry,
            "register_existing_with_audit_on_connection",
            None,
        )
        transaction = getattr(runtime.registry, "transaction", None)
        if callable(register_on_connection) and callable(transaction):
            return uow.finalize_with_connection(
                transaction,
                lambda conn: register_on_connection(
                    conn,
                    artifact.final_path,
                    audit_factory=audit_factory,
                    task_id=str(ctx.task_id),
                    role="derived",
                    anchor_target=dataset.id,
                    seed=seed,
                ),
            )
        return uow.finalize(
            lambda: runtime.registry.register_existing_with_audit(
                artifact.final_path,
                audit_factory=audit_factory,
                task_id=str(ctx.task_id),
                role="derived",
                anchor_target=dataset.id,
                seed=seed,
            )
        )
    except Exception:
        uow.rollback()
        raise


def _stream_mask_to_parquet(
    backend,
    source_path: Path,
    out_path: Path,
    mask_values: dict[str, list[float]],
) -> None:
    source_path = Path(source_path)
    if source_path.suffix.lower() != ".parquet":
        frame = backend.read_frame(source_path)
        for column, values in mask_values.items():
            frame.loc[frame[column].isin(values), column] = np.nan
        frame.to_parquet(out_path, index=False)
        return

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(source_path)
    raw_names = [str(name) for name in parquet_file.schema_arrow.names]
    canonical_names = [str(name) for name in backend.column_names(source_path)]
    canonical_to_raw = dict(zip(canonical_names, raw_names, strict=True))
    missing = sorted(set(mask_values) - set(canonical_to_raw))
    if missing:
        raise ModelingError(f"特殊值治理列不存在：{', '.join(missing)}")
    raw_to_values = {
        canonical_to_raw[column]: values
        for column, values in mask_values.items()
    }
    writer = None
    try:
        for batch in parquet_file.iter_batches(batch_size=4_096, use_threads=False):
            arrays = []
            for raw_name, array in zip(raw_names, batch.columns, strict=True):
                values = raw_to_values.get(raw_name)
                if values:
                    value_set = pa.array(
                        [_arrow_scalar_value(value, array.type) for value in values],
                        type=array.type,
                    )
                    is_sentinel = pc.is_in(array, value_set=value_set)
                    array = pc.if_else(
                        is_sentinel,
                        pa.scalar(None, type=array.type),
                        array,
                    )
                arrays.append(array)
            canonical_batch = pa.RecordBatch.from_arrays(arrays, names=canonical_names)
            if writer is None:
                writer = pq.ParquetWriter(
                    out_path,
                    canonical_batch.schema,
                    compression="snappy",
                )
            writer.write_batch(canonical_batch)
        if writer is None:
            fields = {
                str(field.name): field.type
                for field in parquet_file.schema_arrow
            }
            writer = pq.ParquetWriter(
                out_path,
                pa.schema([
                    pa.field(canonical, fields[raw])
                    for canonical, raw in zip(canonical_names, raw_names, strict=True)
                ]),
                compression="snappy",
            )
    finally:
        if writer is not None:
            writer.close()


def _arrow_scalar_value(value: float, arrow_type):
    import pyarrow as pa

    if pa.types.is_integer(arrow_type):
        return int(value)
    if pa.types.is_decimal(arrow_type):
        return Decimal(str(value))
    if pa.types.is_floating(arrow_type):
        return float(value)
    return value


def _detected_values(rows: list[list[float]]) -> list[float]:
    return sorted({float(row[0]) for row in rows if row})


def _normalized_numeric_values(values) -> list[float]:
    if not isinstance(values, (list, tuple)):
        values = [values]
    normalized = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("special values must be finite")
        normalized.append(number)
    return sorted(set(normalized))


def _fingerprint(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _unique_strings(values) -> list[str]:
    return list(dict.fromkeys(
        str(value).strip()
        for value in values
        if str(value).strip()
    ))


__all__ = [
    "SPECIAL_VALUE_POLICY_VERSION",
    "special_value_decision_fingerprint",
    "tool_resolve_special_values",
]
