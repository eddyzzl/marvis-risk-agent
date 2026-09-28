"""Replay label receipts and match actual sample members to observed records.

This establishes agreement with registered DPD/status data, not external truth,
feature point-in-time safety, or production effectiveness.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.data.label_construction import check_cohort_maturity, construct_label
from marvis.packs.labeling.contracts import LabelingRequest, frame_at_as_of
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.validation.vintage import _cohort_key


def _registered_frame(runtime, task_id, identity):
    dataset = runtime.registry.get(identity["dataset_id"])
    if (
        dataset.task_id != task_id
        or dataset.content_hash != identity["content_hash"]
        or dataset.row_count != identity["row_count"]
    ):
        raise ValueError("label evidence dataset identity drifted")
    return dataset, runtime.registry.read_authenticated_parquet_snapshot(dataset.id)


def _replay_labeling_evidence(runtime, task_id, reference):
    if not isinstance(reference, dict) or set(reference) != {
        "artifact_id",
        "content_hash",
    }:
        raise ValueError("invalid labeling_evidence_ref")
    record = TaskArtifactRepository(runtime.settings.db_path).get_for_task(
        task_id, reference["artifact_id"]
    )
    if (
        not record
        or record["kind"] != "labeling_quality_evidence_json"
        or record["origin_tool"] != "labeling.define_label"
        or record["content_hash"] != reference["content_hash"]
    ):
        raise ValueError("label evidence is not an authenticated task-owned receipt")
    root = Path(runtime.settings.tasks_dir) / task_id / "labeling"
    path = Path(record["path"])
    if (
        path.parent != root
        or not path.name.startswith("labeling_evidence_")
        or path.suffix != ".json"
        or any(item.is_symlink() for item in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > 5_000_000
    ):
        raise ValueError("invalid labeling evidence artifact path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["content_hash"]:
        raise ValueError("label evidence artifact drifted")
    payload = json.loads(raw)
    request = LabelingRequest(**payload["contract"])
    source, result = payload["source"], payload["result"]
    if (
        payload["schema_version"] != "labeling-quality-evidence.v1"
        or payload["producer_version"] != "labeling.define_label.v1"
        or payload["proposal_hash"] != request.contract_hash
        or source["dataset_id"] != request.dataset_id
        or source["content_hash"] != request.expected_content_hash
        or source["date_col"] != request.date_col
        or source["as_of_date"] != request.as_of_date
        or result["target_col"] != request.target_col
        or record["provenance"]
        != {
            "schema_version": payload["schema_version"],
            "producer_version": payload["producer_version"],
            "proposal_hash": request.contract_hash,
            "source_dataset_id": source["dataset_id"],
            "source_content_hash": source["content_hash"],
            "result_dataset_id": result["dataset_id"],
            "result_content_hash": result["content_hash"],
        }
        or payload["lineage"]
        != {
            "parent_dataset_id": source["dataset_id"],
            "child_dataset_id": result["dataset_id"],
            "relation_kind": "label_construction",
            "edge_order": 0,
        }
    ):
        raise ValueError("label evidence contract/provenance drifted")
    _source_dataset, source_frame = _registered_frame(runtime, task_id, source)
    result_dataset, result_frame = _registered_frame(runtime, task_id, result)
    if result_dataset.target_col != request.target_col:
        raise ValueError("label result target definition drifted")
    cutoff, excluded = frame_at_as_of(
        source_frame, date_col=request.date_col, as_of_date=request.as_of_date
    )
    if (
        len(cutoff) != source["rows_at_as_of"]
        or excluded != source["rows_excluded_after_as_of"]
    ):
        raise ValueError("label evidence cutoff counts drifted")
    replay = construct_label(
        cutoff,
        **{
            key: getattr(request, key)
            for key in (
                "id_col",
                "mob_col",
                "observation_window",
                "performance_window",
                "dpd_col",
                "threshold_dpd",
                "status_col",
                "threshold_status",
                "states",
                "at_mob",
                "cohort_col",
                "target_col",
            )
        },
    )
    maturity = check_cohort_maturity(
        cutoff,
        id_col=request.id_col,
        mob_col=request.mob_col,
        cohort_col=request.cohort_col,
        required_mob=request.at_mob,
    )
    mature_count = replay.n_bad + replay.n_good
    quality = {
        "n_loans": replay.n_loans,
        "n_bad": replay.n_bad,
        "n_good": replay.n_good,
        "n_unmatured": replay.n_unmatured,
        "label_coverage": mature_count / replay.n_loans if replay.n_loans else None,
        "bad_rate": replay.n_bad / mature_count if mature_count else None,
    }
    if (
        payload["bad_definition"] != replay.definition.to_dict()
        or payload["quality"] != quality
        or payload["maturity"]
        != {
            "required_mob": maturity.required_mob,
            "cohorts": [asdict(item) for item in maturity.cohorts],
            "immature_cohorts": list(maturity.immature_cohorts),
            "all_matured": maturity.all_matured,
        }
        or not result_frame.equals(replay.frame)
    ):
        raise ValueError("label evidence differs from deterministic source replay")
    return request, cutoff, replay.frame, payload


def _entity_key(value):
    # No string coercion: numeric 12 and textual "12" are distinct identities.
    if isinstance(value, str) and value.strip():
        return ("text", value)
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return ("integer", int(value))
    raise ValueError(
        "label matching requires non-null unambiguous text/integer entity IDs"
    )


def verified_member_labels(runtime, *, task_id, reference, sample, frame, mask):
    """Certify the complete measured denominator, never candidate/goal labels."""
    request, cutoff, labels, payload = _replay_labeling_evidence(
        runtime, task_id, reference
    )
    source = sample.source_binding
    risk = next(item for item in sample.bundle["populations"] if item["role"] == "risk")
    maturity = risk["maturity_evidence"]
    if (
        maturity["status"] != "confirmed_matured"
        or not maturity["cutoff_date"]
        or request.as_of_date > maturity["cutoff_date"]
    ):
        raise ValueError(
            "label receipt cutoff exceeds the certified sample maturity date"
        )
    fields = sample.bundle["sample_design"]["sample_semantics"]["field_bindings"]
    entity, cohort = fields["entity_field"], fields["month_field"]
    if (
        not entity
        or not cohort
        or source.target_col != request.target_col
        or source.target_bad_value != 1
    ):
        raise ValueError(
            "native sample entity/cohort/target definition differs from label producer"
        )
    selected = frame.loc[mask].copy()
    selected_keys = selected[entity].map(_entity_key)
    label_keys = labels[request.id_col].map(_entity_key)
    if (
        selected_keys.duplicated().any()
        or label_keys.duplicated().any()
        or selected.empty
    ):
        raise ValueError("label matching has duplicate or empty sample identities")
    source_keys = cutoff[request.id_col].map(_entity_key)
    source_cohorts = cutoff[request.cohort_col].map(_cohort_key)
    history = cutoff.copy()
    history["__entity"] = source_keys
    history["__cohort"] = source_cohorts
    if history.groupby("__entity")["__cohort"].nunique().gt(1).any():
        raise ValueError("label source entity has ambiguous cohorts")
    label_by_key = {key: index for index, key in enumerate(label_keys)}
    histories = {key: rows for key, rows in history.groupby("__entity", sort=False)}
    expected_mobs = set(range(request.observation_window + 1, request.at_mob + 1))
    for key, (_, row) in zip(selected_keys, selected.iterrows(), strict=True):
        if key not in label_by_key or key not in histories:
            raise ValueError(
                "label receipt does not cover every measured sample member"
            )
        observed = labels.iloc[label_by_key[key]]
        if _cohort_key(row[cohort]) != _cohort_key(observed[request.cohort_col]):
            raise ValueError("label receipt and sample member cohorts differ")
        if (
            pd.isna(row[source.target_col])
            or row[source.target_col] not in (0, 1)
            or row[source.target_col] != observed[request.target_col]
        ):
            raise ValueError(
                "sample member target differs from observed label producer"
            )
        rows = histories[key]
        mobs = pd.to_numeric(rows[request.mob_col], errors="raise")
        if (
            mobs.isna().any()
            or not np.isfinite(mobs).all()
            or (mobs % 1 != 0).any()
            or mobs.duplicated().any()
        ):
            raise ValueError(
                "label observations have missing or ambiguous MOB identities"
            )
        window = rows.loc[mobs.isin(expected_mobs)]
        if set(mobs) & expected_mobs != expected_mobs:
            raise ValueError(
                "measured sample member label performance window is not mature"
            )
        if request.rule_kind == "dpd":
            values = pd.to_numeric(window[request.value_col], errors="raise")
            complete = (
                values.notna().all()
                and np.isfinite(values).all()
                and values.ge(0).all()
            )
        else:
            complete = (
                window[request.value_col].notna().all()
                and window[request.value_col].isin(request.states).all()
            )
        if not complete:
            raise ValueError("measured sample member has unobserved performance values")
    return {
        "label_origin": "observed",
        "labeling_evidence_ref": dict(reference),
        "producer_version": payload["producer_version"],
        "proposal_hash": request.contract_hash,
        "source": {
            key: payload["source"][key] for key in ("dataset_id", "content_hash")
        },
        "result": {
            key: payload["result"][key] for key in ("dataset_id", "content_hash")
        },
        "bad_definition": payload["bad_definition"],
        "as_of_date": request.as_of_date,
        "sample_maturity_cutoff_date": maturity["cutoff_date"],
        "sample_dataset_id": source.dataset_id,
        "sample_dataset_content_hash": source.dataset_content_hash,
        "membership_content_hash": sample.membership["header"]["content_hash"],
        "covered_member_count": len(selected),
        "required_member_count": len(selected),
        "coverage": 1.0,
        "scope": "agreement_with_registered_label_producer",
    }
