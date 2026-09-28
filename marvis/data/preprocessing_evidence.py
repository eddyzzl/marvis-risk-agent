"""Dataset-owned receipts for fitting membership and scoring-time parameters.

The assurance concerns recorded transforms only. It says nothing about label
maturity, point-in-time availability or external authenticity of the source.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.feature.errors import FeatureError
from marvis.feature.fit_scope import fit_membership
from marvis.feature.preprocessing import read_preprocessing_chain
from marvis.files import sha256_file
from marvis.repositories.task_artifacts import TaskArtifactRepository


KIND = "feature_preprocessing"
ORIGIN = "feature.fitted_preprocessing.v1"
VERSION = "fitted-preprocessing.v1"


@dataclass(frozen=True)
class PreprocessingState:
    steps: list[dict]
    assurance: str
    artifact_id: str | None = None
    content_hash: str | None = None


def _digest(value):
    # Existing fitted maps represent missing/empty-bin parameters with NaN.
    # Preserve their established serialization; this digest is not a metric.
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def fitting_evidence(frame, inputs, *, tool, dataset_id):
    mask, scope = fit_membership(frame, inputs, tool=tool, dataset_id=dataset_id)
    packed = np.packbits(mask, bitorder="little").tobytes()
    return {
        "tool": tool,
        "scope": scope,
        "row_count": len(frame),
        "fit_rows": int(mask.sum()),
        "membership": base64.b64encode(packed).decode("ascii"),
        "membership_sha256": hashlib.sha256(packed).hexdigest(),
        "selection": {
            key: inputs[key]
            for key in (
                "split_col",
                "train_values",
                "holdout_values",
                "allow_full_fit",
                "target_col",
                "drop_nan_labels",
            )
            if key in inputs
        },
    }


def verify_source_on_connection(registry, conn, source):
    current = registry._repo.get_dataset_on_connection(conn, source.id)
    if current is None or any(
        getattr(current, key) != getattr(source, key)
        for key in (
            "task_id",
            "content_hash",
            "row_count",
            "has_target",
            "target_col",
        )
    ):
        raise FeatureError("preprocessing source identity changed")
    path = registry.datasets_root / current.source_path
    if path.resolve(strict=True) != path or sha256_file(path) != source.content_hash:
        raise FeatureError("preprocessing source bytes changed")


def register_preprocessing_evidence(
    registry,
    conn,
    *,
    dataset,
    source,
    path,
    steps,
    source_state,
    fit,
):
    """Register the parameter receipt in the dataset's existing transaction."""
    verify_source_on_connection(registry, conn, source)
    if dataset.row_count != source.row_count:
        raise FeatureError("fitted transform changed row membership")
    if _digest(read_preprocessing_chain(path)) != _digest(steps):
        raise FeatureError("preprocessing parameters changed before registration")
    record = TaskArtifactRepository(registry._repo.db_path).register_on_connection(
        conn,
        task_id=dataset.task_id,
        kind=KIND,
        path=Path(path).relative_to(registry.datasets_root.parent).as_posix(),
        content_hash=sha256_file(path),
        origin_tool=ORIGIN,
        provenance={
            "schema_version": VERSION,
            "output_dataset_id": dataset.id,
            "output_content_hash": dataset.content_hash,
            "source_dataset_id": source.id,
            "source_content_hash": source.content_hash,
            "source_target": [source.has_target, source.target_col],
            "source_row_count": source.row_count,
            "steps_sha256": _digest(steps),
            "parent_artifact_id": source_state.artifact_id,
            "parent_content_hash": source_state.content_hash,
            "parent_assurance": source_state.assurance,
            "fit": fit,
        },
    )
    verify_source_on_connection(registry, conn, source)
    if sha256_file(path) != record["content_hash"]:
        raise FeatureError("preprocessing evidence changed before commit")
    return record


def load_preprocessing_state(registry, dataset_id, *, _seen=()) -> PreprocessingState:
    if dataset_id in _seen or len(_seen) >= 64:
        raise FeatureError("preprocessing lineage is cyclic or exceeds 64 transforms")
    dataset = registry.get(dataset_id)
    records = [
        record
        for record in TaskArtifactRepository(registry._repo.db_path).list_for_task(
            dataset.task_id
        )
        if record["kind"] == KIND
        and record["provenance"].get("output_dataset_id") == dataset_id
    ]
    if not records:
        return PreprocessingState(
            read_preprocessing_chain(registry.resolve_path(dataset_id)), "unknown"
        )
    if len(records) != 1:
        raise FeatureError("ambiguous preprocessing evidence")
    record = records[0]
    proof = record["provenance"]
    if record["origin_tool"] != ORIGIN or proof.get("schema_version") != VERSION:
        raise FeatureError("unsupported preprocessing evidence origin")
    path = registry.datasets_root.parent / record["path"]
    resolved = path.resolve(strict=True)
    resolved.relative_to(registry.datasets_root)
    if resolved != path or resolved.stat().st_size > 20_000_000:
        raise FeatureError("invalid preprocessing evidence path or size")
    if sha256_file(path) != record["content_hash"]:
        raise FeatureError("preprocessing evidence content changed")
    steps = read_preprocessing_chain(path)
    if (
        sha256_file(path) != record["content_hash"]
        or _digest(steps) != proof["steps_sha256"]
    ):
        raise FeatureError("preprocessing parameters changed")
    if dataset.content_hash != proof["output_content_hash"]:
        raise FeatureError("preprocessing dataset binding changed")
    registry.resolve_verified_path(dataset_id)
    source = registry.get(proof["source_dataset_id"])
    if (
        source.task_id != dataset.task_id
        or source.content_hash != proof["source_content_hash"]
        or [source.has_target, source.target_col] != proof["source_target"]
        or source.row_count != proof["source_row_count"]
        or source.row_count != dataset.row_count
    ):
        raise FeatureError("preprocessing source binding changed")
    registry.resolve_verified_path(source.id)
    parent = load_preprocessing_state(registry, source.id, _seen=(*_seen, dataset_id))
    if (
        parent.artifact_id != proof["parent_artifact_id"]
        or parent.content_hash != proof["parent_content_hash"]
        or parent.assurance != proof["parent_assurance"]
    ):
        raise FeatureError("preprocessing parent evidence changed")
    if _digest(steps[: len(parent.steps)]) != _digest(parent.steps):
        raise FeatureError("preprocessing parent parameters changed")
    fit = proof["fit"]
    for component in _fit_components(fit):
        column = component["selection"].get("split_col")
        frame = (
            registry.read_authenticated_parquet_snapshot(source.id, columns=[column])
            if column
            else pd.DataFrame(index=range(source.row_count))
        )
        if (
            fitting_evidence(
                frame,
                component["selection"],
                tool=component["tool"],
                dataset_id=source.id,
            )
            != component
        ):
            raise FeatureError("preprocessing fitting membership changed")
    components = _fit_components(fit)
    assurance = (
        "exploration"
        if any(item["scope"] == "full" for item in components)
        else "training_only"
        if components
        else "row_local"
        if fit == [] and not parent.steps
        else parent.assurance
    )
    if parent.steps and parent.assurance not in {"training_only", "row_local"}:
        assurance = parent.assurance
    return PreprocessingState(steps, assurance, record["id"], record["content_hash"])


def training_preprocessing_state(registry, dataset_id, *, split_col, train_values):
    """A new split cannot turn previous fitting members into evaluation rows."""
    state = load_preprocessing_state(registry, dataset_id)
    if not state.artifact_id:
        return state
    frame = registry.read_authenticated_parquet_snapshot(
        dataset_id, columns=[split_col]
    )
    values = train_values if isinstance(train_values, (list, tuple)) else [train_values]
    train = frame[split_col].isin(values).to_numpy(dtype=bool)
    return _check_training_membership(registry, dataset_id, state, train)


def training_preprocessing_state_for_membership(registry, dataset_id, *, train_mask):
    """Apply the same fit boundary to native full-dataset membership masks.

    Governed training owns its partition in an authenticated sample artifact,
    not in the source's optional split column or the later private risk frame.
    """
    train = np.asarray(train_mask)
    dataset = registry.get(dataset_id)
    if train.dtype != np.bool_ or train.shape != (dataset.row_count,):
        raise FeatureError("training membership must be a full-dataset boolean mask")
    state = load_preprocessing_state(registry, dataset_id)
    return _check_training_membership(registry, dataset_id, state, train)


def _check_training_membership(registry, dataset_id, state, train):
    if not state.artifact_id:
        return state
    dataset = registry.get(dataset_id)
    records = {
        record["id"]: record
        for record in TaskArtifactRepository(registry._repo.db_path).list_for_task(
            dataset.task_id
        )
    }
    identity = state.artifact_id
    while identity:
        proof = records[identity]["provenance"]
        fit = proof["fit"]
        for component in _fit_components(fit):
            mask = np.unpackbits(
                np.frombuffer(
                    base64.b64decode(component["membership"]), dtype=np.uint8
                ),
                bitorder="little",
                count=component["row_count"],
            ).astype(bool)
            if len(mask) != len(train) or np.any(mask & ~train):
                raise FeatureError(
                    "fitted preprocessing includes current evaluation rows; refit on the training partition"
                )
        identity = proof["parent_artifact_id"]
    return state


def _fit_components(fit):
    # Existing single-fit receipts stay readable; a derivation recipe may fit
    # several independent transforms, each retaining its own exact selection.
    return [] if fit is None else fit if isinstance(fit, list) else [fit]
