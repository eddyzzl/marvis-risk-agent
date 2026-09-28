"""Independent business declarations bound to native, authenticated sample evidence.

Declarations describe meaning. They cannot certify measured membership, dates,
label provenance, or model metrics; those are recomputed from platform sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from marvis.artifacts import ArtifactUnitOfWork
from marvis.files import sha256_file
from marvis.orchestrator.evidence import payload_hash
from marvis.packs.strategy.sample_design_v2_native_tools import (
    load_native_strategy_sample_design_v2_artifacts,
    load_historical_native_strategy_sample_design_v2_artifacts,
    require_native_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository

CONTEXT_VERSION = "business-sample-context.v1"
CONTEXT_KIND = "business_sample_context_json"
CONTEXT_TOOL = "strategy.bind_business_context"


class SampleBusinessDeclaration(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, str_strip_whitespace=True
    )
    business_line: str = Field(min_length=1, max_length=200)
    decision_node: str = Field(min_length=1, max_length=200)
    population: str = Field(min_length=1, max_length=500)
    responsibility_source: str = Field(min_length=1, max_length=500)
    partition: Literal["development", "validation", "oot"]
    currency: str | None = Field(default=None, min_length=1, max_length=20)
    declared_label_origin: Literal[
        "observed", "reject_inferred", "assumed", "synthetic", "unknown"
    ] = "unknown"


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def bind_business_context(inputs, ctx, runtime):
    required = {"sample_design_ref", "declaration"}
    if not required <= set(inputs) or set(inputs) - required - {
        "labeling_evidence_ref"
    }:
        raise ValueError(
            "business context requires sample_design_ref and independent declaration"
        )
    declaration = SampleBusinessDeclaration.model_validate(
        inputs["declaration"]
    ).model_dump()
    sample = load_native_strategy_sample_design_v2_artifacts(
        runtime, task_id=ctx.task_id, **inputs["sample_design_ref"]
    )
    payload = {
        "schema_version": CONTEXT_VERSION,
        "task_id": ctx.task_id,
        "sample_design_ref": inputs["sample_design_ref"],
        "declaration": declaration,
    }
    if "labeling_evidence_ref" in inputs:
        payload["labeling_evidence_ref"] = inputs["labeling_evidence_ref"]
        _sample_measurement(runtime, payload, sample)
    encoded = _canonical(payload)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    repo = TaskArtifactRepository(runtime.settings.db_path)
    root = Path(runtime.settings.tasks_dir) / ctx.task_id / "business_context"
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(root, f"{digest}.json")
    try:
        staged.path.write_text(encoded, encoding="utf-8")
        with repo.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_native_strategy_sample_design_v2_artifact_binding_on_connection(
                conn, sample
            )
            if "labeling_evidence_ref" in payload:
                _sample_measurement(runtime, payload, sample)
            uow.promote_all()
            record = repo.register_on_connection(
                conn,
                task_id=ctx.task_id,
                kind=CONTEXT_KIND,
                path=str(staged.final_path),
                content_hash=digest,
                origin_tool=CONTEXT_TOOL,
                provenance={
                    "schema_version": CONTEXT_VERSION,
                    "sample_design_ref": inputs["sample_design_ref"],
                    "declaration_hash": payload_hash(declaration),
                    **(
                        {"labeling_evidence_ref": payload["labeling_evidence_ref"]}
                        if "labeling_evidence_ref" in payload
                        else {}
                    ),
                },
            )
        uow.commit()
    except Exception:
        uow.rollback()
        raise
    return {
        "business_context_ref": {"artifact_id": record["id"], "content_hash": digest},
        "declaration": declaration,
        "notice": "业务语义由用户独立声明；该声明不证明真实观察标签、因果效果或生产结果。",
    }


def load_business_context(runtime, task_id, reference):
    if not isinstance(reference, dict) or set(reference) != {
        "artifact_id",
        "content_hash",
    }:
        raise ValueError("invalid business_context_ref")
    repo = TaskArtifactRepository(runtime.settings.db_path)
    record = repo.get_for_task(task_id, reference["artifact_id"])
    if (
        not record
        or record["kind"] != CONTEXT_KIND
        or record["origin_tool"] != CONTEXT_TOOL
        or record["content_hash"] != reference["content_hash"]
    ):
        raise ValueError(
            "business context is not an authenticated task-owned declaration"
        )
    root = Path(runtime.settings.tasks_dir) / task_id / "business_context"
    path = Path(record["path"])
    expected = root / f"{record['content_hash']}.json"
    if (
        path != expected
        or any(item.is_symlink() for item in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > 100_000
    ):
        raise ValueError("invalid business context artifact path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["content_hash"]:
        raise ValueError("business context artifact drifted")
    payload = json.loads(raw)
    if (
        set(payload)
        not in (
            {"schema_version", "task_id", "sample_design_ref", "declaration"},
            {
                "schema_version",
                "task_id",
                "sample_design_ref",
                "declaration",
                "labeling_evidence_ref",
            },
        )
        or payload["schema_version"] != CONTEXT_VERSION
        or payload["task_id"] != task_id
    ):
        raise ValueError("business context identity drifted")
    declaration = SampleBusinessDeclaration.model_validate(
        payload["declaration"]
    ).model_dump()
    if _canonical(payload).encode() != raw or record["provenance"] != {
        "schema_version": CONTEXT_VERSION,
        "sample_design_ref": payload["sample_design_ref"],
        "declaration_hash": payload_hash(declaration),
        **(
            {"labeling_evidence_ref": payload["labeling_evidence_ref"]}
            if "labeling_evidence_ref" in payload
            else {}
        ),
    }:
        raise ValueError("business context provenance drifted")
    sample = load_historical_native_strategy_sample_design_v2_artifacts(
        runtime, task_id=task_id, **payload["sample_design_ref"]
    )
    return payload, sample


def _sample_measurement(runtime, payload, sample, *, exclude_unlabeled=True):
    declaration = payload["declaration"]
    partition = declaration["partition"]
    mask = np.asarray(sample.membership["masks"][f"risk/{partition}"], dtype=bool)
    source = sample.source_binding
    if sha256_file(source.dataset_path) != source.dataset_content_hash:
        raise ValueError("sample dataset changed before business measurement")
    if (
        runtime.registry.get(source.dataset_id).content_hash
        != source.dataset_content_hash
    ):
        raise ValueError("business sample registry hash changed")
    frame = runtime.registry.read_authenticated_parquet_snapshot(source.dataset_id)
    if len(frame) != len(mask) or not mask.any():
        raise ValueError("empty or inconsistent business measurement membership")
    if source.drop_nan_labels and exclude_unlabeled:
        mask &= frame[source.target_col].notna().to_numpy()
    if not mask.any():
        raise ValueError("business measurement has no labeled members")
    design = sample.bundle["sample_design"]
    time_col = design["sample_semantics"]["field_bindings"]["time_field"]
    if not time_col:
        raise ValueError("sample has no authenticated event date")
    times = pd.to_datetime(frame.loc[mask, time_col], errors="raise", utc=True)
    if times.isna().any():
        raise ValueError("business measurement dates are incomplete")
    risk = next(item for item in sample.bundle["populations"] if item["role"] == "risk")
    count = int(mask.sum())
    labeled = int(frame.loc[mask, source.target_col].notna().sum())
    label_provenance = None
    if "labeling_evidence_ref" in payload:
        from marvis.packs.labeling.evidence import verified_member_labels

        label_provenance = verified_member_labels(
            runtime,
            task_id=payload["task_id"],
            reference=payload["labeling_evidence_ref"],
            sample=sample,
            frame=frame,
            mask=mask,
        )
    if sha256_file(source.dataset_path) != source.dataset_content_hash:
        raise ValueError("sample dataset changed while measuring business context")
    return {
        "business_line": declaration["business_line"],
        "decision_node": declaration["decision_node"],
        "population": declaration["population"],
        "currency": declaration["currency"],
        "period_start": times.min().date().isoformat(),
        "period_end": times.max().date().isoformat(),
        "labels_mature": risk["maturity_evidence"]["status"] == "confirmed_matured"
        and count == labeled,
        "label_origin": label_provenance["label_origin"]
        if label_provenance
        else "unknown",
        **({"label_provenance": label_provenance} if label_provenance else {}),
        "effect_stage": "oot_validated" if partition == "oot" else "backtested",
        "measurement_membership": {
            "mask_name": f"risk/{partition}",
            "row_count": count,
            "labeled_count": labeled,
            "membership_content_hash": sample.membership["header"]["content_hash"],
        },
    }


def model_business_measurement(
    runtime, task_id, *, context_ref, training_ref, experiment_id, artifact_id
):
    from marvis.packs.modeling.evidence_tools import (
        load_historical_modeling_training_evidence_artifacts,
    )

    payload, sample = load_business_context(runtime, task_id, context_ref)
    if training_ref["sample_design_ref"] != payload["sample_design_ref"]:
        raise ValueError("business context and training evidence use different samples")
    training = load_historical_modeling_training_evidence_artifacts(
        runtime, task_id=task_id, **training_ref
    )
    if (
        training.experiment.id != experiment_id
        or training.model_artifact.id != artifact_id
    ):
        raise ValueError("business evidence does not belong to the adopted model")
    context = _sample_measurement(runtime, payload, sample)
    partition = payload["declaration"]["partition"]
    prefix = {"development": "train", "validation": "test", "oot": "oot"}[partition]
    values = training.evidence["metrics_snapshot"]["values"]
    metrics = {
        key: values[key]
        for key in (f"{prefix}_ks", f"{prefix}_auc")
        if values.get(key) is not None
    }
    return {
        "schema_version": "business-model-measurement.v1",
        "business_context_ref": context_ref,
        "training_evidence_ref": training_ref,
        "target_id": experiment_id,
        "target_version": artifact_id,
        "metrics": metrics,
        "metric_units": dict.fromkeys(metrics, "ratio"),
        "denominators": dict.fromkeys(metrics, f"risk/{partition}:labeled"),
        **context,
    }


def strategy_business_measurement(
    runtime,
    task_id,
    *,
    context_ref,
    sample_binding,
    strategy_id,
    version,
    backtest_id,
    metrics,
    population_count,
    labeled_count,
):
    payload, sample = load_business_context(runtime, task_id, context_ref)
    if (
        payload["declaration"]["partition"] != "development"
        or sample_binding is None
        or sample_binding.source_mode != "native_active_dataset"
    ):
        raise ValueError(
            "strategy adoption requires native risk/development business context"
        )
    ref = sample_binding.to_ref_dict()
    if (
        ref["artifact_id"] != sample.bundle_artifact_id
        or ref["artifact_content_hash"] != sample.bundle_artifact_content_hash
        or ref["sample_design_id"]
        != payload["sample_design_ref"]["expected_sample_design_id"]
        or ref["sample_design_content_hash"]
        != payload["sample_design_ref"]["expected_sample_design_content_hash"]
    ):
        raise ValueError(
            "business context differs from adopted strategy backtest membership"
        )
    # The typed strategy producer keeps all risk/development members. Missing
    # labels affect only bad-rate denominators, unlike model training metrics.
    measured = _sample_measurement(runtime, payload, sample, exclude_unlabeled=False)
    members = measured["measurement_membership"]
    if (
        population_count != members["row_count"]
        or labeled_count != members["labeled_count"]
    ):
        raise ValueError(
            "strategy backtest population differs from measured business membership"
        )
    denominators = {
        "approval_rate": "risk/development:rows",
        "approved_bad_rate": "risk/development:approved_labeled",
        "rejected_bad_rate": "risk/development:rejected_labeled",
    }
    measured_metrics = {
        key: metrics[key] for key in denominators if metrics.get(key) is not None
    }
    return {
        "schema_version": "business-strategy-measurement.v1",
        "business_context_ref": context_ref,
        "target_id": strategy_id,
        "target_version": str(version),
        "backtest_id": backtest_id,
        "metrics": measured_metrics,
        "metric_units": dict.fromkeys(measured_metrics, "ratio"),
        "denominators": {key: denominators[key] for key in measured_metrics},
        **measured,
    }


def authenticated_business_fields(plan_repository, task_id, output, target_kind):
    """Recompute producer facts; never trust a plugin's claimed measurement dict."""
    measurement = output.get("business_measurement")
    if not isinstance(measurement, dict):
        return {}
    from marvis.packs.modeling._runtime import _runtime

    workspace = Path(plan_repository.db_path).parent
    runtime = _runtime(
        SimpleNamespace(workspace=workspace, datasets_root=workspace / "datasets")
    )
    if runtime.settings.db_path != Path(plan_repository.db_path):
        return {}
    if target_kind == "model":
        expected = model_business_measurement(
            runtime,
            task_id,
            context_ref=measurement["business_context_ref"],
            training_ref=measurement["training_evidence_ref"],
            experiment_id=output["selected_experiment_id"],
            artifact_id=output["artifact_id"],
        )
    else:
        from marvis.packs.strategy.tools import (
            _runtime as strategy_runtime,
            _strategy_adoption_evidence,
        )

        runtime = strategy_runtime(
            SimpleNamespace(workspace=workspace, datasets_root=workspace / "datasets")
        )
        strategy = runtime.strategies.get_strategy(output["strategy_id"])
        backtest = runtime.strategies.get_backtest(output["backtest_id"])
        meta = runtime.strategies.get_strategy_meta(output["strategy_id"])
        if (
            strategy is None
            or meta is None
            or meta["task_id"] != task_id
            or str(meta["version"]) != str(output["version"])
            or backtest is None
            or backtest.strategy_id != strategy.id
        ):
            raise ValueError("adopted strategy source is unavailable")
        _evidence, approval_metrics, sample_binding, _requirements = (
            _strategy_adoption_evidence(
                runtime,
                strategy=strategy,
                backtest=backtest,
                backtest_id=output["backtest_id"],
                task_id=task_id,
            )
        )
        expected = strategy_business_measurement(
            runtime,
            task_id,
            context_ref=measurement["business_context_ref"],
            sample_binding=sample_binding,
            strategy_id=output["strategy_id"],
            version=output["version"],
            backtest_id=output["backtest_id"],
            metrics=dict(approval_metrics) if approval_metrics is not None else {},
            population_count=_evidence.get("population_count"),
            labeled_count=_evidence.get("labeled_count"),
        )
    if expected != measurement:
        raise ValueError(
            "business measurement differs from authenticated adopted evidence"
        )
    allowed = {
        "metrics",
        "metric_units",
        "denominators",
        "business_line",
        "decision_node",
        "population",
        "currency",
        "period_start",
        "period_end",
        "labels_mature",
        "label_origin",
        "label_provenance",
        "effect_stage",
    }
    return {key: value for key, value in expected.items() if key in allowed}
