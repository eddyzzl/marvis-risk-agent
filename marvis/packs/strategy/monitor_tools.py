"""S5: strategy monitoring closure.

``tool_run_strategy_monitoring`` reads a locally adopted strategy's monitoring plan and
runs one monitoring pass against a fresh dataset:

* if the plan carries an ``experiment_id`` (the strategy is driven by a scoring
  model), it delegates to the modeling ``monitor_run`` kernel unchanged (INV-1),
  passing the plan's threshold overrides through monitor_run's own
  ``monitoring_policy`` channel -> the same PSI/CSI/KS/AUC checks the model
  monitoring surface produces;
* approval/reject strategies keep their strategy-facing approval-rate and
  approved-bad-rate drift checks;
* limit, pricing, and segmentation strategies are applied through the canonical
  vectorized evaluator. Their fresh, directly observable metrics are judged by
  the type-specific threshold specs committed in the adoption plan. Limit and
  pricing economics are recomputed from immutable scalar/column bindings;
  only true legacy-v1 plans without those bindings stay explicitly ``n/a``;
* it composes an overall green/amber/red verdict and atomically appends an
  immutable monitoring-run receipt plus a ``strategy.monitor`` audit row. Plan
  artifacts are never rewritten; run time belongs to the run ledger.

A pure-rule strategy (no ``experiment_id``) skips PSI/CSI entirely and reports
only the strategy-facing checks. A strategy outside ``adopted_local`` raises a
typed ``StrategyNotAdoptedError``. Local adoption is not production deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, StrategyRepository
from marvis.db_schema import connect
from marvis.feature.metrics import compute_psi
from marvis.files import sha256_file
from marvis.packs.strategy.backtest import strategy_approval_mask
from marvis.packs.strategy.economics import limit_metrics, pricing_metrics
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.errors import StrategyError, StrategyNotAdoptedError
from marvis.strategy_lifecycle import (
    StrategyLifecycleError,
    is_locally_adopted,
    validate_lifecycle_pair,
)
from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    canonical_economics_bindings_hash,
    load_monitoring_plan,
    save_monitoring_plan,
)
from marvis.repositories.audit import _list_audit_rows, _write_audit_row
from marvis.repositories.strategy_monitoring import (
    MonitoringPlanRecord,
    MonitoringRunRecord,
    StrategyMonitoringRepository,
)
from marvis.settings import build_settings

#: Strategy-facing drift bands (percentage points, configurable). A metric that
#: has moved more than AMBER but at most RED off its adoption baseline is amber;
#: beyond RED is red. Symmetric so both a rising bad rate and a falling approval
#: rate (or the reverse) trip the same bands -- the spec's "approval ±5pp=amber
#: ±10pp=red" made a shared constant for both strategy-facing metrics.
STRATEGY_DRIFT_AMBER_PP = 0.05
STRATEGY_DRIFT_RED_PP = 0.10
_STRATEGY_ARTIFACT_PROVENANCE_SCHEMA_VERSION = "strategy-artifact-provenance.v1"
_MONITORING_PLAN_ARTIFACT_PRODUCER_VERSION = (
    "strategy.monitoring.adjust_plan.v1"
)


@dataclass(frozen=True)
class _MonitoringCalculation:
    overall_level: str
    checks: list[dict]
    top_drifted_features: list[dict]
    red_flags: list[dict]
    metrics: dict
    economics: dict
    monitoring_inputs: dict


def tool_run_strategy_monitoring(inputs: dict, ctx) -> dict:
    runtime = _Runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    dataset_id = str(inputs["dataset_id"])

    meta = _strategy_meta_for_task(runtime, strategy_id, str(ctx.task_id))
    _require_locally_adopted(meta, strategy_id=strategy_id)

    strategy = runtime.strategies.get_strategy(strategy_id)
    if strategy is None:
        raise StrategyError(f"strategy not found: {strategy_id}")
    strategy_effect_hash = runtime.strategies.get_strategy_spec_hash(strategy_id)
    if strategy_effect_hash is None:
        raise StrategyError(f"strategy not found: {strategy_id}")
    resolved_plan = _resolve_monitoring_plan(
        runtime,
        strategy_id,
        strategy_effect_hash=strategy_effect_hash,
    )
    plan = resolved_plan.plan
    if plan.version != int(meta.get("version", 1)):
        raise StrategyError(
            "monitoring plan strategy version does not match adopted strategy"
        )

    snapshot = _dataset_snapshot(runtime, dataset_id, task_id=str(ctx.task_id))
    calculation = _calculate_strategy_monitoring(
        inputs=inputs,
        ctx=ctx,
        strategy=strategy,
        plan=plan,
        snapshot=snapshot,
    )

    run_at = datetime.now(UTC).isoformat()
    economics_binding_hash = canonical_economics_bindings_hash(
        plan.economics_bindings
    )
    with connect(runtime.settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_unchanged_dataset(snapshot)
        _bind_dataset_hash_on_connection(conn, snapshot)
        plan_record = resolved_plan.record
        if plan_record is None:
            plan_record = runtime.monitoring.create_plan_on_connection(
                conn,
                plan,
                expected_revision=0,
                plan_id=plan.monitoring_plan_id,
                created_at=run_at,
            )
        run_payload = _monitoring_run_payload(
            strategy_id=strategy_id,
            dataset_id=dataset_id,
            plan=plan,
            plan_record=plan_record,
            plan_source=resolved_plan.source,
            snapshot=snapshot,
            strategy_effect_hash=strategy_effect_hash,
            economics_binding_hash=economics_binding_hash,
            calculation=calculation,
            run_at=run_at,
        )
        monitoring_run = runtime.monitoring.create_run_on_connection(
            conn,
            strategy_id=strategy_id,
            monitoring_plan_id=plan_record.id,
            expected_plan_revision=plan_record.revision,
            expected_plan_payload_hash=plan_record.payload_hash,
            dataset_id=dataset_id,
            dataset_content_hash=snapshot.content_hash,
            strategy_effect_hash=strategy_effect_hash,
            economics_binding_hash=economics_binding_hash,
            result=run_payload,
            overall_level=calculation.overall_level,
            created_at=run_at,
        )
        _write_strategy_monitor_audit(
            conn,
            task_id=str(ctx.task_id),
            strategy_id=strategy_id,
            dataset_id=dataset_id,
            snapshot=snapshot,
            plan=plan,
            plan_record=plan_record,
            monitoring_run=monitoring_run,
            calculation=calculation,
            plan_source=resolved_plan.source,
            run_at=run_at,
        )
        _require_unchanged_dataset(snapshot)

    return _monitoring_output(
        strategy_id=strategy_id,
        dataset_id=dataset_id,
        plan=plan,
        plan_record=plan_record,
        plan_source=resolved_plan.source,
        snapshot=snapshot,
        strategy_effect_hash=strategy_effect_hash,
        economics_binding_hash=economics_binding_hash,
        calculation=calculation,
        monitoring_run=monitoring_run,
        run_at=run_at,
    )


def _calculate_strategy_monitoring(
    *,
    inputs: Mapping[str, Any],
    ctx,
    strategy,
    plan: MonitoringPlan,
    snapshot: "_DatasetSnapshot",
) -> _MonitoringCalculation:
    monitoring_inputs = _monitoring_input_receipt(inputs)
    target_col = monitoring_inputs["target_col"]
    model_checks, top_drifted, model_level = _run_model_monitoring(
        dict(inputs), ctx, plan
    )
    metrics: dict = {}
    economics: dict = {}
    if strategy.strategy_type in {"approval", "reject"}:
        strategy_checks, strategy_level = _strategy_drift_checks(
            snapshot.frame,
            strategy,
            plan,
            target_col=target_col,
        )
    else:
        strategy_checks, strategy_level, metrics, economics = (
            _typed_strategy_threshold_checks(
                snapshot.frame,
                strategy,
                plan,
                target_col=target_col,
            )
        )
    checks = [*model_checks, *strategy_checks]
    overall_level = _overall_level([model_level, strategy_level])
    red_flags = [
        {
            "id": check["id"],
            "label": check.get("label"),
            "message": check.get("message"),
        }
        for check in checks
        if check.get("level") == "red"
    ]
    return _MonitoringCalculation(
        overall_level=overall_level,
        checks=checks,
        top_drifted_features=top_drifted,
        red_flags=red_flags,
        metrics=metrics,
        economics=economics,
        monitoring_inputs=monitoring_inputs,
    )


def _monitoring_input_receipt(inputs: Mapping[str, Any]) -> dict:
    score_col = _optional_str(inputs.get("score_col"))
    return {
        "target_col": _optional_str(inputs.get("target_col")),
        "score_col": score_col,
        "dataset_mode": "scored" if score_col is not None else "raw",
    }


def _monitoring_run_payload(
    *,
    strategy_id: str,
    dataset_id: str,
    plan: MonitoringPlan,
    plan_record: MonitoringPlanRecord,
    plan_source: str,
    snapshot: "_DatasetSnapshot",
    strategy_effect_hash: str,
    economics_binding_hash: str,
    calculation: _MonitoringCalculation,
    run_at: str,
    source_monitoring_run_id: str | None = None,
) -> dict:
    payload = {
        "strategy_id": strategy_id,
        "dataset_id": dataset_id,
        "experiment_id": plan.experiment_id,
        "overall_level": calculation.overall_level,
        "checks": calculation.checks,
        "top_drifted_features": calculation.top_drifted_features,
        "red_flags": calculation.red_flags,
        "row_count": int(len(snapshot.frame)),
        "metrics": calculation.metrics,
        "economics": calculation.economics,
        "monitoring_inputs": calculation.monitoring_inputs,
        "adjustable_threshold_ids": sorted(plan.thresholds),
        "plan_source": plan_source,
        "monitoring_plan_id": plan_record.id,
        "monitoring_plan_revision": plan_record.revision,
        "monitoring_plan_hash": plan_record.payload_hash,
        "dataset_content_hash": snapshot.content_hash,
        "strategy_effect_hash": strategy_effect_hash,
        "economics_binding_hash": economics_binding_hash,
        "run_at": run_at,
    }
    if source_monitoring_run_id is not None:
        payload["source_monitoring_run_id"] = source_monitoring_run_id
    return payload


def _write_strategy_monitor_audit(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    strategy_id: str,
    dataset_id: str,
    snapshot: "_DatasetSnapshot",
    plan: MonitoringPlan,
    plan_record: MonitoringPlanRecord,
    monitoring_run: MonitoringRunRecord,
    calculation: _MonitoringCalculation,
    plan_source: str,
    run_at: str,
    source_monitoring_run_id: str | None = None,
) -> None:
    detail = {
        "task_id": task_id,
        "strategy_id": strategy_id,
        "dataset_id": dataset_id,
        "dataset_content_hash": snapshot.content_hash,
        "experiment_id": plan.experiment_id,
        "overall_level": calculation.overall_level,
        "row_count": int(len(snapshot.frame)),
        "last_run_at": run_at,
        "monitoring_plan_id": plan_record.id,
        "monitoring_plan_revision": plan_record.revision,
        "monitoring_plan_hash": plan_record.payload_hash,
        "monitoring_run_id": monitoring_run.id,
        "monitoring_result_hash": monitoring_run.result_hash,
        "plan_source": plan_source,
    }
    if source_monitoring_run_id is not None:
        detail["source_monitoring_run_id"] = source_monitoring_run_id
    _write_audit_row(
        conn,
        kind="strategy.monitor",
        target_ref=strategy_id,
        inputs_hash=monitoring_run.result_hash,
        outcome="succeeded",
        detail=detail,
    )


def _monitoring_output(
    *,
    strategy_id: str,
    dataset_id: str,
    plan: MonitoringPlan,
    plan_record: MonitoringPlanRecord,
    plan_source: str,
    snapshot: "_DatasetSnapshot",
    strategy_effect_hash: str,
    economics_binding_hash: str,
    calculation: _MonitoringCalculation,
    monitoring_run: MonitoringRunRecord,
    run_at: str,
) -> dict:
    evidence = {
        "plan_source": plan_source,
        "monitoring_plan_id": plan_record.id,
        "monitoring_plan_revision": plan_record.revision,
        "monitoring_plan_hash": plan_record.payload_hash,
        "monitoring_run_id": monitoring_run.id,
        "dataset_content_hash": snapshot.content_hash,
        "strategy_effect_hash": strategy_effect_hash,
        "economics_binding_hash": economics_binding_hash,
        "result_hash": monitoring_run.result_hash,
    }
    return {
        "strategy_id": strategy_id,
        "dataset_id": dataset_id,
        "experiment_id": plan.experiment_id,
        "overall_level": calculation.overall_level,
        "checks": calculation.checks,
        "top_drifted_features": calculation.top_drifted_features,
        "red_flags": calculation.red_flags,
        "metrics": calculation.metrics,
        "economics": calculation.economics,
        "adjustable_threshold_ids": sorted(plan.thresholds),
        "plan_source": plan_source,
        "monitoring_plan_id": plan_record.id,
        "monitoring_plan_revision": plan_record.revision,
        "monitoring_plan_hash": plan_record.payload_hash,
        "monitoring_run_id": monitoring_run.id,
        "monitoring_evidence": evidence,
        "plan_updated": False,
        "last_run_at": run_at,
        "row_count": int(len(snapshot.frame)),
    }


def rerun_strategy_monitoring_with_candidate_plan(
    *,
    ctx,
    strategy_id: str,
    source_monitoring_run_id: str,
    expected_latest_plan_id: str,
    expected_latest_plan_revision: int,
    expected_latest_plan_hash: str,
    candidate_plan: MonitoringPlan,
    reason: str,
    threshold_patch: Mapping[str, Any],
) -> dict:
    """Atomically append an adjusted plan, rerun, audits, and plan artifact.

    The source run owns the immutable dataset evidence. Callers may supply only
    governance metadata and an already-validated candidate plan; metrics and
    prior results are always recomputed here. Model checks use the modeling
    kernel's pure calculation boundary, so no standalone modeling audit can leak
    before this strategy plan/run, disposition, audit and artifact transaction.
    """

    runtime = _Runtime(ctx)
    normalized_strategy_id = str(strategy_id).strip()
    normalized_source_run_id = str(source_monitoring_run_id).strip()
    normalized_reason = _validated_adjust_reason(reason)
    normalized_patch = _validated_threshold_patch(threshold_patch)
    meta = _strategy_meta_for_task(
        runtime,
        normalized_strategy_id,
        str(ctx.task_id),
    )
    _require_locally_adopted(meta, strategy_id=normalized_strategy_id)
    strategy = runtime.strategies.get_strategy(normalized_strategy_id)
    if strategy is None:
        raise StrategyError(f"strategy not found: {normalized_strategy_id}")
    source_run = runtime.monitoring.get_run(normalized_source_run_id)
    if source_run is None or source_run.strategy_id != normalized_strategy_id:
        raise StrategyError(f"monitoring run not found: {normalized_source_run_id}")
    expected_plan = runtime.monitoring.get_plan(str(expected_latest_plan_id))
    if expected_plan is None or expected_plan.strategy_id != normalized_strategy_id:
        raise StrategyError("monitoring plan CAS receipt does not match latest plan")
    _validate_adjust_candidate(
        expected_plan=expected_plan,
        expected_plan_revision=expected_latest_plan_revision,
        expected_plan_hash=expected_latest_plan_hash,
        source_run=source_run,
        candidate_plan=candidate_plan,
        threshold_patch=normalized_patch,
    )
    with connect(runtime.settings.db_path) as conn:
        _validate_adjust_source_on_connection(
            conn,
            task_id=str(ctx.task_id),
            strategy_id=normalized_strategy_id,
            source_run=source_run,
            expected_plan=expected_plan,
        )

    snapshot = _dataset_snapshot(
        runtime,
        source_run.dataset_id,
        task_id=str(ctx.task_id),
    )
    if snapshot.content_hash != source_run.dataset_content_hash:
        raise StrategyError("source monitoring dataset content hash no longer matches")
    strategy_effect_hash = runtime.strategies.get_strategy_spec_hash(
        normalized_strategy_id
    )
    if strategy_effect_hash != source_run.strategy_effect_hash:
        raise StrategyError("source monitoring run strategy effect is no longer current")
    economics_binding_hash = canonical_economics_bindings_hash(
        candidate_plan.economics_bindings
    )
    if economics_binding_hash != source_run.economics_binding_hash:
        raise StrategyError("candidate monitoring economics binding does not match source run")
    rerun_inputs = _source_run_monitoring_inputs(
        runtime,
        source_run=source_run,
        plan=candidate_plan,
        snapshot=snapshot,
    )
    calculation = _calculate_strategy_monitoring(
        inputs=rerun_inputs,
        ctx=ctx,
        strategy=strategy,
        plan=candidate_plan,
        snapshot=snapshot,
    )

    run_at = datetime.now(UTC).isoformat()
    strategy_dir = (
        Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    )
    artifact_name = (
        f"monitoring_plan_{_artifact_token(normalized_strategy_id)}"
        f"_v{candidate_plan.version}_r{candidate_plan.revision}_"
        f"{_artifact_token(candidate_plan.monitoring_plan_id or '')}.json"
    )
    uow = ArtifactUnitOfWork()
    staged_plan = uow.stage_file(strategy_dir, artifact_name)
    try:
        save_monitoring_plan(staged_plan.path, candidate_plan)

        def commit_adjustment(conn: sqlite3.Connection):
            conn.execute("BEGIN IMMEDIATE")
            _validate_adjust_source_on_connection(
                conn,
                task_id=str(ctx.task_id),
                strategy_id=normalized_strategy_id,
                source_run=source_run,
                expected_plan=expected_plan,
            )
            _require_unchanged_dataset(snapshot)
            _bind_dataset_hash_on_connection(conn, snapshot)
            plan_record = runtime.monitoring.create_plan_on_connection(
                conn,
                candidate_plan,
                expected_revision=expected_plan.revision,
                expected_payload_hash=expected_plan.payload_hash,
                plan_id=candidate_plan.monitoring_plan_id,
                created_at=run_at,
            )
            run_payload = _monitoring_run_payload(
                strategy_id=normalized_strategy_id,
                dataset_id=source_run.dataset_id,
                plan=candidate_plan,
                plan_record=plan_record,
                plan_source="ledger",
                snapshot=snapshot,
                strategy_effect_hash=strategy_effect_hash,
                economics_binding_hash=economics_binding_hash,
                calculation=calculation,
                run_at=run_at,
                source_monitoring_run_id=source_run.id,
            )
            monitoring_run = runtime.monitoring.create_run_on_connection(
                conn,
                strategy_id=normalized_strategy_id,
                monitoring_plan_id=plan_record.id,
                expected_plan_revision=plan_record.revision,
                expected_plan_payload_hash=plan_record.payload_hash,
                dataset_id=source_run.dataset_id,
                dataset_content_hash=snapshot.content_hash,
                strategy_effect_hash=strategy_effect_hash,
                economics_binding_hash=economics_binding_hash,
                result=run_payload,
                overall_level=calculation.overall_level,
                created_at=run_at,
            )
            _write_strategy_monitor_audit(
                conn,
                task_id=str(ctx.task_id),
                strategy_id=normalized_strategy_id,
                dataset_id=source_run.dataset_id,
                snapshot=snapshot,
                plan=candidate_plan,
                plan_record=plan_record,
                monitoring_run=monitoring_run,
                calculation=calculation,
                plan_source="ledger",
                run_at=run_at,
                source_monitoring_run_id=source_run.id,
            )
            artifact_content_hash = sha256_file(staged_plan.final_path)
            artifact_content_size = staged_plan.final_path.stat().st_size
            artifact_provenance = _adjusted_monitoring_plan_artifact_provenance(
                task_id=str(ctx.task_id),
                strategy_id=normalized_strategy_id,
                source_run=source_run,
                expected_plan=expected_plan,
                plan_record=plan_record,
                monitoring_run=monitoring_run,
                threshold_patch=normalized_patch,
            )
            artifact_record = runtime.strategies.register_verified_strategy_artifact_with_audit_on_connection(
                conn,
                normalized_strategy_id,
                kind="monitoring_plan_json",
                path=str(staged_plan.final_path),
                content_hash=artifact_content_hash,
                content_size=artifact_content_size,
                provenance=artifact_provenance,
                created_at=run_at,
                audit={
                    "kind": "strategy.artifact",
                    "target_ref": normalized_strategy_id,
                    "outcome": "succeeded",
                    "detail": {
                        "task_id": str(ctx.task_id),
                        "kind": "monitoring_plan_json",
                        "path": str(staged_plan.final_path),
                        "content_hash": artifact_content_hash,
                        "content_size": artifact_content_size,
                        "producer_version": _MONITORING_PLAN_ARTIFACT_PRODUCER_VERSION,
                        "monitoring_plan_id": plan_record.id,
                        "monitoring_plan_revision": plan_record.revision,
                        "monitoring_plan_hash": plan_record.payload_hash,
                    },
                },
            )
            disposition_detail = {
                "receipt_schema_version": "strategy.monitoring-disposition.v1",
                "task_id": str(ctx.task_id),
                "strategy_id": normalized_strategy_id,
                "disposition": "adjust_threshold",
                "status": "threshold_adjusted",
                "source_monitoring_run_id": source_run.id,
                "source_monitoring_run_hash": source_run.result_hash,
                "old_monitoring_plan_id": expected_plan.id,
                "old_monitoring_plan_revision": expected_plan.revision,
                "old_monitoring_plan_hash": expected_plan.payload_hash,
                "new_monitoring_plan_id": plan_record.id,
                "new_monitoring_plan_revision": plan_record.revision,
                "new_monitoring_plan_hash": plan_record.payload_hash,
                "new_monitoring_run_id": monitoring_run.id,
                "new_monitoring_run_hash": monitoring_run.result_hash,
                "reason": normalized_reason,
                "threshold_patch": normalized_patch,
            }
            _write_audit_row(
                conn,
                kind="strategy.monitoring.disposition",
                target_ref=source_run.id,
                inputs_hash=_canonical_receipt_hash(disposition_detail),
                outcome="succeeded",
                detail=disposition_detail,
            )
            _require_unchanged_dataset(snapshot)
            return plan_record, monitoring_run, artifact_record

        plan_record, monitoring_run, _artifact_record = uow.finalize_with_connection(
            runtime.strategies.transaction,
            commit_adjustment,
        )
    except Exception:
        uow.rollback()
        raise

    output = _monitoring_output(
        strategy_id=normalized_strategy_id,
        dataset_id=source_run.dataset_id,
        plan=candidate_plan,
        plan_record=plan_record,
        plan_source="ledger",
        snapshot=snapshot,
        strategy_effect_hash=strategy_effect_hash,
        economics_binding_hash=economics_binding_hash,
        calculation=calculation,
        monitoring_run=monitoring_run,
        run_at=run_at,
    )
    output["plan_artifact_path"] = str(staged_plan.final_path)
    return output


def _adjusted_monitoring_plan_artifact_provenance(
    *,
    task_id: str,
    strategy_id: str,
    source_run: MonitoringRunRecord,
    expected_plan: MonitoringPlanRecord,
    plan_record: MonitoringPlanRecord,
    monitoring_run: MonitoringRunRecord,
    threshold_patch: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": _STRATEGY_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "producer_version": _MONITORING_PLAN_ARTIFACT_PRODUCER_VERSION,
        "task_id": task_id,
        "strategy_id": strategy_id,
        "kind": "monitoring_plan_json",
        "evidence": {
            "operation": "strategy.monitoring.adjust_threshold",
            "source_monitoring_run_id": source_run.id,
            "source_monitoring_run_hash": source_run.result_hash,
            "dataset_id": source_run.dataset_id,
            "dataset_content_hash": source_run.dataset_content_hash,
            "strategy_effect_hash": source_run.strategy_effect_hash,
            "economics_binding_hash": source_run.economics_binding_hash,
            "old_monitoring_plan_id": expected_plan.id,
            "old_monitoring_plan_revision": expected_plan.revision,
            "old_monitoring_plan_hash": expected_plan.payload_hash,
            "monitoring_plan_id": plan_record.id,
            "monitoring_plan_revision": plan_record.revision,
            "monitoring_plan_hash": plan_record.payload_hash,
            "monitoring_run_id": monitoring_run.id,
            "monitoring_run_hash": monitoring_run.result_hash,
            "threshold_patch": dict(threshold_patch),
        },
    }


def _validate_adjust_candidate(
    *,
    expected_plan: MonitoringPlanRecord,
    expected_plan_revision: int,
    expected_plan_hash: str,
    source_run: MonitoringRunRecord,
    candidate_plan: MonitoringPlan,
    threshold_patch: dict,
) -> None:
    if not isinstance(candidate_plan, MonitoringPlan):
        raise StrategyError("candidate monitoring plan must be a MonitoringPlan")
    if (
        expected_plan.revision != expected_plan_revision
        or expected_plan.payload_hash != str(expected_plan_hash).lower()
    ):
        raise StrategyError("monitoring plan CAS receipt does not match latest plan")
    if source_run.monitoring_plan_id != expected_plan.id:
        raise StrategyError("source monitoring run is not bound to the expected plan")
    if source_run.overall_level != "red":
        raise StrategyError("adjust_threshold requires the latest red monitoring run")
    if (
        candidate_plan.monitoring_plan_id is None
        or candidate_plan.monitoring_plan_id == expected_plan.id
        or candidate_plan.revision != expected_plan.revision + 1
        or candidate_plan.supersedes_plan_id != expected_plan.id
    ):
        raise StrategyError("candidate monitoring plan revision identity is invalid")
    expected_payload = expected_plan.plan.to_dict()
    candidate_payload = candidate_plan.to_dict()
    for key in (
        "monitoring_plan_id",
        "revision",
        "supersedes_plan_id",
        "thresholds",
    ):
        expected_payload.pop(key, None)
        candidate_payload.pop(key, None)
    if candidate_payload != expected_payload:
        raise StrategyError("candidate monitoring plan may only change thresholds")
    reconstructed = {
        name: dict(spec) for name, spec in expected_plan.plan.thresholds.items()
    }
    for check_id, patch in threshold_patch.items():
        if check_id not in reconstructed:
            raise StrategyError(f"unknown monitoring check: {check_id}")
        reconstructed[check_id].update(patch)
    if reconstructed != candidate_plan.thresholds:
        raise StrategyError("threshold_patch does not describe the candidate plan")


def _validate_adjust_source_on_connection(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    strategy_id: str,
    source_run: MonitoringRunRecord,
    expected_plan: MonitoringPlanRecord,
) -> None:
    replay = conn.execute(
        "SELECT 1 FROM audit WHERE kind = ? AND target_ref = ? LIMIT 1",
        ("strategy.monitoring.disposition", source_run.id),
    ).fetchone()
    if replay is not None:
        raise StrategyError("monitoring run already has a disposition")
    strategy_row = conn.execute(
        """
        SELECT id, task_id, status, asset_status, version, strategy_type, rules_json,
               score_col, default_decision_json, description,
               dsl_json, dsl_schema_version
          FROM strategies
         WHERE id = ?
        """,
        (strategy_id,),
    ).fetchone()
    if strategy_row is None or str(strategy_row["task_id"]) != task_id:
        raise StrategyError(f"strategy not found: {strategy_id}")
    _require_locally_adopted(strategy_row, strategy_id=strategy_id)
    if int(strategy_row["version"]) != expected_plan.strategy_version:
        raise StrategyError("strategy version changed after monitoring")
    from marvis.repositories.strategy import _strategy_spec_hash_from_row

    if _strategy_spec_hash_from_row(strategy_row) != source_run.strategy_effect_hash:
        raise StrategyError("source monitoring run strategy effect is no longer current")
    latest_run = conn.execute(
        """
        SELECT id, monitoring_plan_id, dataset_id, dataset_content_hash,
               strategy_effect_hash, economics_binding_hash, result_hash,
               overall_level
          FROM strategy_monitoring_runs
         WHERE strategy_id = ?
         ORDER BY created_at DESC, id DESC
         LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    if latest_run is None or str(latest_run["id"]) != source_run.id:
        raise StrategyError("source monitoring run is not the latest monitoring run")
    if str(latest_run["overall_level"]) != "red":
        raise StrategyError("adjust_threshold requires the latest red monitoring run")
    for column, expected in (
        ("monitoring_plan_id", source_run.monitoring_plan_id),
        ("dataset_id", source_run.dataset_id),
        ("dataset_content_hash", source_run.dataset_content_hash),
        ("strategy_effect_hash", source_run.strategy_effect_hash),
        ("economics_binding_hash", source_run.economics_binding_hash),
        ("result_hash", source_run.result_hash),
    ):
        if str(latest_run[column]) != expected:
            raise StrategyError("source monitoring run receipt changed")
    latest_plan = conn.execute(
        """
        SELECT id, revision, payload_hash
          FROM strategy_monitoring_plans
         WHERE strategy_id = ?
         ORDER BY revision DESC, created_at DESC, id DESC
         LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    if (
        latest_plan is None
        or str(latest_plan["id"]) != expected_plan.id
        or int(latest_plan["revision"]) != expected_plan.revision
        or str(latest_plan["payload_hash"]) != expected_plan.payload_hash
    ):
        raise StrategyError("monitoring plan CAS receipt does not match latest plan")


def _validated_adjust_reason(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError("adjust_threshold reason must be non-empty")
    return value.strip()


def _validated_threshold_patch(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping) or not value:
        raise StrategyError("threshold_patch must be a non-empty object")
    normalized: dict[str, dict[str, float]] = {}
    for raw_check_id, raw_patch in value.items():
        check_id = str(raw_check_id).strip()
        if not check_id or not isinstance(raw_patch, Mapping) or not raw_patch:
            raise StrategyError("threshold_patch entries must be non-empty objects")
        if not set(raw_patch) <= {"warn", "fail"}:
            raise StrategyError("threshold_patch may only change warn/fail")
        patch: dict[str, float] = {}
        for field, raw_threshold in raw_patch.items():
            if (
                isinstance(raw_threshold, bool)
                or not isinstance(raw_threshold, (int, float))
                or not math.isfinite(float(raw_threshold))
            ):
                raise StrategyError("threshold_patch values must be finite numbers")
            patch[str(field)] = float(raw_threshold)
        normalized[check_id] = patch
    return normalized


def _source_run_monitoring_inputs(
    runtime: "_Runtime",
    *,
    source_run: MonitoringRunRecord,
    plan: MonitoringPlan,
    snapshot: "_DatasetSnapshot",
) -> dict:
    raw_receipt = source_run.result.get("monitoring_inputs")
    if isinstance(raw_receipt, Mapping):
        target_col = _optional_str(raw_receipt.get("target_col"))
        score_col = _optional_str(raw_receipt.get("score_col"))
        mode = str(raw_receipt.get("dataset_mode") or "raw")
        if mode not in {"raw", "scored"} or (mode == "scored") != (
            score_col is not None
        ):
            raise StrategyError("source monitoring input receipt is invalid")
    else:
        if plan.experiment_id is not None:
            raise StrategyError(
                "model-backed source monitoring run lacks immutable input provenance"
            )
        with connect(runtime.settings.db_path) as conn:
            task = conn.execute(
                "SELECT target_col FROM tasks WHERE id = ?",
                (snapshot.task_id,),
            ).fetchone()
        target_col = (
            str(task["target_col"])
            if task is not None and str(task["target_col"]) in snapshot.frame.columns
            else None
        )
        score_col = None
        mode = "raw"
    inputs: dict[str, Any] = {"dataset_id": source_run.dataset_id}
    if target_col is not None:
        inputs["target_col"] = target_col
    if mode == "scored" and score_col is not None:
        inputs["score_col"] = score_col
    return inputs


def _artifact_token(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    ).strip("_")
    return normalized[:48] or uuid.uuid4().hex


def _run_model_monitoring(inputs: dict, ctx, plan: MonitoringPlan):
    """Delegate to the modeling monitor_run kernel when the plan is model-backed.

    Returns (checks, top_drifted_features, level). A pure-rule strategy (no
    experiment_id) skips PSI/CSI and returns ([], [], None)."""
    if not plan.experiment_id:
        return [], [], None
    from marvis.packs.modeling.monitor_tools import _calculate_monitor_run

    monitor_inputs = {
        "experiment_id": plan.experiment_id,
        "dataset_id": inputs["dataset_id"],
    }
    if inputs.get("score_col"):
        monitor_inputs["score_col"] = inputs["score_col"]
        monitor_inputs["scored_dataset_id"] = inputs["dataset_id"]
        monitor_inputs.pop("dataset_id", None)
    if inputs.get("target_col"):
        monitor_inputs["target_col"] = inputs["target_col"]
    # Plan thresholds override monitor_run's defaults through its own
    # monitoring_policy channel (INV-1: same kernel, plan-supplied thresholds).
    model_thresholds = _model_monitoring_thresholds(plan)
    if model_thresholds:
        monitor_inputs["monitoring_policy"] = {"thresholds": model_thresholds}

    result = _calculate_monitor_run(monitor_inputs, ctx)
    raw_checks = result.get("checks")
    raw_top_drifted = result.get("top_drifted_features")
    if (
        not isinstance(raw_checks, list)
        or not raw_checks
        or any(not isinstance(check, dict) for check in raw_checks)
        or any(
            check.get("level") not in {"green", "amber", "red", "n/a"}
            for check in raw_checks
        )
    ):
        raise StrategyError("model monitoring returned invalid check evidence")
    if not isinstance(raw_top_drifted, list) or any(
        not isinstance(row, dict) for row in raw_top_drifted
    ):
        raise StrategyError("model monitoring returned invalid drift evidence")
    checks = [dict(check) for check in raw_checks]
    top_drifted = [dict(row) for row in raw_top_drifted]
    level = result.get("overall_level")
    if level not in {"green", "amber", "red"}:
        raise StrategyError(
            "model monitoring returned a missing or unsupported overall level"
        )
    if _overall_level(check["level"] for check in checks) != level:
        raise StrategyError("model monitoring overall level conflicts with its checks")
    return checks, top_drifted, str(level)


_MONITORING_PLAN_ORIGIN_KEY = "monitoring_plan_origin"
_LIMIT_ECONOMICS_KEYS = frozenset({"pd", "lgd", "utilization"})
_PRICING_ECONOMICS_KEYS = frozenset(
    {
        "ead",
        "pd",
        "lgd",
        "funding_rate",
        "term_months",
        "operating_cost_per_loan",
    }
)


def _model_monitoring_thresholds(plan: MonitoringPlan) -> dict[str, dict]:
    if not plan.experiment_id:
        return {}
    from marvis.packs.modeling.monitor_tools import MONITOR_RUN_THRESHOLDS

    return {
        str(check_id): dict(plan.thresholds[check_id])
        for check_id in MONITOR_RUN_THRESHOLDS
        if check_id in plan.thresholds
    }


def _strategy_monitoring_threshold_items(
    plan: MonitoringPlan,
) -> list[tuple[str, dict]]:
    model_ids = set(_model_monitoring_thresholds(plan))
    return [
        (str(check_id), dict(spec))
        for check_id, spec in plan.thresholds.items()
        if check_id not in model_ids
    ]


def _materialize_imported_model_thresholds(plan: MonitoringPlan) -> MonitoringPlan:
    """Freeze legacy model-kernel defaults when an artifact first enters V2.

    Before the immutable ledger existed, model-backed strategy plans could rely
    on ``monitor_run`` defaults without storing those ids.  Importing them as-is
    would preserve the verdict but make PSI/CSI thresholds impossible to patch.
    Materialization keeps the old artifact untouched while writing the exact
    effective model policy into the first ledger revision.
    """

    if not plan.experiment_id:
        return plan
    from marvis.packs.modeling.monitor_tools import MONITOR_RUN_THRESHOLDS

    combined = {str(check_id): dict(spec) for check_id, spec in plan.thresholds.items()}
    for check_id, default_spec in MONITOR_RUN_THRESHOLDS.items():
        raw_spec = plan.thresholds.get(check_id)
        if raw_spec is not None and not isinstance(raw_spec, Mapping):
            raise StrategyError(
                f"model monitoring threshold {check_id} must be an object"
            )
        effective = dict(default_spec)
        if isinstance(raw_spec, Mapping):
            effective.update(
                {
                    field: raw_spec[field]
                    for field in ("label", "metric", "direction", "warn", "fail")
                    if field in raw_spec
                }
            )
        combined[str(check_id)] = effective
    return replace(plan, thresholds=combined)


@dataclass(frozen=True)
class _ResolvedMonitoringPlan:
    plan: MonitoringPlan
    record: MonitoringPlanRecord | None
    source: str


@dataclass(frozen=True)
class _DatasetSnapshot:
    dataset_id: str
    task_id: str
    path: Path
    content_hash: str
    frame: pd.DataFrame


def _resolve_monitoring_plan(
    runtime: "_Runtime",
    strategy_id: str,
    *,
    strategy_effect_hash: str,
) -> _ResolvedMonitoringPlan:
    record = runtime.monitoring.latest_plan(strategy_id)
    if record is not None:
        _require_plan_effect_binding(
            record.plan,
            strategy_effect_hash=strategy_effect_hash,
            strict=record.plan.plan_version >= 2,
        )
        return _ResolvedMonitoringPlan(
            plan=record.plan,
            record=record,
            source="ledger",
        )

    plan_path = _latest_plan_path(runtime, strategy_id)
    artifact_plan = load_monitoring_plan(plan_path)
    if artifact_plan.monitoring_plan_id is not None:
        raise StrategyError(
            "监控计划声明了 immutable monitoring_plan_id，但账本中不存在对应记录。"
        )
    if artifact_plan.revision != 1 or artifact_plan.supersedes_plan_id is not None:
        raise StrategyError("未登记的监控计划不能恢复多 revision 历史。")
    if artifact_plan.plan_version == 1:
        source = "legacy_v1"
    elif artifact_plan.plan_version == 2:
        # Transitional artifacts written before adoption was wired directly to
        # the ledger. They are imported once after a successful calculation and
        # remain V2. They must satisfy the same fail-closed economics binding
        # contract as a plan that was born in the ledger.
        source = "compatibility_import"
    else:
        raise StrategyError(
            f"unsupported unledgered monitoring plan version: {artifact_plan.plan_version}"
        )

    artifact_plan = _materialize_imported_model_thresholds(artifact_plan)

    _require_plan_effect_binding(
        artifact_plan,
        strategy_effect_hash=strategy_effect_hash,
        strict=False,
    )
    baseline = dict(artifact_plan.expectation_baseline)
    baseline["strategy_effect_hash"] = strategy_effect_hash
    baseline[_MONITORING_PLAN_ORIGIN_KEY] = source
    imported = replace(
        artifact_plan,
        monitoring_plan_id=uuid.uuid4().hex,
        revision=1,
        supersedes_plan_id=None,
        last_run_at=None,
        expectation_baseline=baseline,
    )
    return _ResolvedMonitoringPlan(plan=imported, record=None, source=source)


def _require_plan_effect_binding(
    plan: MonitoringPlan,
    *,
    strategy_effect_hash: str,
    strict: bool,
) -> None:
    bound = plan.expectation_baseline.get("strategy_effect_hash")
    if bound in (None, ""):
        if strict:
            raise StrategyError(
                "V2 monitoring plan is missing its strategy_effect_hash binding"
            )
        return
    if str(bound) != strategy_effect_hash:
        raise StrategyError("monitoring plan strategy_effect_hash does not match strategy")


def _dataset_snapshot(
    runtime: "_Runtime",
    dataset_id: str,
    *,
    task_id: str,
) -> _DatasetSnapshot:
    try:
        dataset = runtime.registry.get(dataset_id)
    except KeyError:
        raise StrategyError(f"dataset not found: {dataset_id}") from None
    if str(dataset.task_id) != str(task_id):
        raise StrategyError(f"dataset not found: {dataset_id}")
    registered_hash = str(dataset.content_hash or "").lower()
    if registered_hash and len(registered_hash) != 64:
        raise StrategyError(f"dataset {dataset_id} has an invalid content hash")
    path = runtime.registry.resolve_path(dataset.id)
    before_hash = sha256_file(path)
    if registered_hash and before_hash != registered_hash:
        raise StrategyError(f"dataset content changed after registration: {dataset_id}")
    frame = runtime.backend.read_frame(path)
    after_hash = sha256_file(path)
    if after_hash != before_hash:
        raise StrategyError(f"dataset content changed while reading: {dataset_id}")
    return _DatasetSnapshot(
        dataset_id=dataset.id,
        task_id=str(dataset.task_id),
        path=path,
        content_hash=before_hash,
        frame=frame,
    )


def _require_unchanged_dataset(snapshot: _DatasetSnapshot) -> None:
    if sha256_file(snapshot.path) != snapshot.content_hash:
        raise StrategyError(
            f"dataset content changed during monitoring: {snapshot.dataset_id}"
        )


def _bind_dataset_hash_on_connection(conn, snapshot: _DatasetSnapshot) -> None:
    row = conn.execute(
        "SELECT task_id, content_hash FROM datasets WHERE id = ?",
        (snapshot.dataset_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != snapshot.task_id:
        raise StrategyError(f"dataset not found: {snapshot.dataset_id}")
    stored_hash = str(row["content_hash"] or "").lower()
    if stored_hash:
        if stored_hash != snapshot.content_hash:
            raise StrategyError(
                f"dataset content hash changed during monitoring: {snapshot.dataset_id}"
            )
        return
    cursor = conn.execute(
        "UPDATE datasets SET content_hash = ? WHERE id = ? AND content_hash IS NULL",
        (snapshot.content_hash, snapshot.dataset_id),
    )
    if cursor.rowcount != 1:
        raise StrategyError(
            f"dataset content hash changed during monitoring: {snapshot.dataset_id}"
        )


def _resolve_economics_inputs(
    frame: pd.DataFrame,
    plan: MonitoringPlan,
    *,
    strategy_type: str,
) -> dict:
    expected = (
        _LIMIT_ECONOMICS_KEYS
        if strategy_type == "limit"
        else _PRICING_ECONOMICS_KEYS
    )
    bindings = dict(plan.economics_bindings)
    if not bindings:
        if _allows_missing_economics_bindings(plan):
            return {}
        raise StrategyError(
            f"V2 {strategy_type} monitoring plan requires economics_bindings"
        )
    actual = set(bindings)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected: {', '.join(unexpected)}")
        raise StrategyError(
            f"{strategy_type} monitoring economics bindings must be complete; "
            + "; ".join(detail)
        )

    resolved: dict = {}
    for name in sorted(expected):
        binding = bindings[name]
        if binding["kind"] == "scalar":
            resolved[name] = float(binding["value"])
            continue
        column = str(binding["column"])
        if column not in frame.columns:
            raise StrategyError(f"missing economics column: {column}")
        values = frame[column]
        if not isinstance(values, pd.Series):
            raise StrategyError(
                f"economics column must identify exactly one column: {column}"
            )
        resolved[name] = values
    return resolved


def _allows_missing_economics_bindings(plan: MonitoringPlan) -> bool:
    return plan.plan_version == 1


def _economics_target(
    frame: pd.DataFrame,
    *,
    target_col: str | None,
    index: pd.Index,
) -> pd.Series:
    target = _fresh_binary_target(frame, target_col=target_col)
    if target is None:
        return pd.Series(np.nan, index=index, dtype=float, name="target")
    if not target.index.equals(index):
        raise StrategyError("monitoring target index does not match strategy decisions")
    return target.astype(float)


def _aggregate_economics(raw: object) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise StrategyError("strategy economics kernel returned a non-object result")
    return {
        str(key): value
        for key, value in raw.items()
        if str(key) != "by_row"
    }


def _typed_strategy_threshold_checks(
    frame: pd.DataFrame,
    strategy,
    plan: MonitoringPlan,
    *,
    target_col: str | None,
) -> tuple[list[dict], str, dict, dict]:
    """Evaluate fresh typed-strategy metrics against the adopted plan.

    Limit/pricing economics are recomputed from fresh data using the immutable
    plan's scalar/column bindings. Every V2 plan fails closed when that binding
    contract is absent or incomplete; only true V1 plans retain the historical
    explicit n/a behaviour.
    """

    evaluation = evaluate_strategy_frame(
        frame,
        strategy.spec or legacy_strategy_to_spec(strategy),
    )
    strategy_type = str(strategy.strategy_type)
    metrics: dict[str, float | None]
    economics: dict = {}
    if strategy_type == "limit":
        inputs = _resolve_economics_inputs(frame, plan, strategy_type="limit")
        assigned = _assigned_numeric_values(
            evaluation.decisions, metric="mean_limit"
        )
        calculated = limit_metrics(
            assigned,
            _economics_target(frame, target_col=target_col, index=assigned.index),
            **inputs,
        )
        economics = _aggregate_economics(calculated.get("economics"))
        metrics = {
            "mean_limit": calculated["mean_limit"],
            **economics,
        }
    elif strategy_type == "pricing":
        inputs = _resolve_economics_inputs(frame, plan, strategy_type="pricing")
        assigned = _assigned_numeric_values(
            evaluation.decisions, metric="mean_rate"
        )
        calculated = pricing_metrics(
            assigned,
            _economics_target(frame, target_col=target_col, index=assigned.index),
            **inputs,
        )
        economics = _aggregate_economics(calculated.get("economics"))
        metrics = {
            "mean_rate": calculated["mean_rate"],
            **economics,
        }
    elif strategy_type == "segmentation":
        metrics = {
            "overall_bad_rate": _fresh_overall_bad_rate(
                frame,
                target_col=target_col,
            ),
            "segment_share_psi": _segment_share_psi(
                evaluation.decisions,
                baseline=plan.expectation_baseline,
            ),
        }
    else:
        raise StrategyError(
            f"monitoring does not support strategy type: {strategy_type}"
        )

    checks = [
        _plan_threshold_check(check_id, spec, metrics)
        for check_id, spec in _strategy_monitoring_threshold_items(plan)
    ]
    return (
        checks,
        _overall_level(check["level"] for check in checks),
        metrics,
        economics,
    )


def _assigned_numeric_values(decisions: pd.Series, *, metric: str) -> pd.Series:
    numeric = pd.to_numeric(decisions, errors="coerce")
    if bool(numeric.isna().any()):
        raise StrategyError(
            f"typed strategy produced a non-numeric assigned value for {metric}"
        )
    numeric = numeric.astype(float)
    if not bool(numeric.map(math.isfinite).all()):
        raise StrategyError(
            f"typed strategy produced a non-finite assigned value for {metric}"
        )
    return numeric


def _fresh_overall_bad_rate(
    frame: pd.DataFrame,
    *,
    target_col: str | None,
) -> float | None:
    numeric = _fresh_binary_target(frame, target_col=target_col)
    if numeric is None:
        return None
    labeled = numeric.dropna()
    if labeled.empty:
        return None
    return float(labeled.eq(1).mean())


def _fresh_binary_target(
    frame: pd.DataFrame,
    *,
    target_col: str | None,
) -> pd.Series | None:
    if target_col is None or target_col not in frame.columns:
        return None
    raw = frame[target_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & (numeric.isna() | ~numeric.isin([0, 1]))
    if bool(invalid.any()):
        raise StrategyError("target must contain only 0, 1, or missing")
    return numeric


def _segment_share_psi(
    decisions: pd.Series,
    *,
    baseline: dict,
) -> float | None:
    """PSI of fresh segment shares versus the adoption breakdown.

    The union of baseline and fresh segment ids is used, so a newly appearing or
    disappearing segment is visible through the shared PSI smoothing kernel.
    """

    if decisions.empty:
        return None
    breakdown = baseline.get("breakdown")
    if not isinstance(breakdown, list) or not breakdown:
        return None

    population_count = _finite_float(baseline.get("population_count"))
    baseline_shares: dict[str, float] = {}
    for row in breakdown:
        if not isinstance(row, dict) or "segment" not in row:
            return None
        share = _finite_float(row.get("share"))
        if share is None and population_count not in {None, 0.0}:
            count = _finite_float(row.get("count"))
            if count is not None:
                share = count / float(population_count)
        if share is None or share < 0:
            return None
        token = _segment_token(row["segment"])
        baseline_shares[token] = baseline_shares.get(token, 0.0) + share
    if not baseline_shares or sum(baseline_shares.values()) <= 0:
        return None

    fresh_counts: dict[str, int] = {}
    for value in decisions.tolist():
        token = _segment_token(value)
        fresh_counts[token] = fresh_counts.get(token, 0) + 1
    tokens = sorted(set(baseline_shares) | set(fresh_counts))
    expected = np.asarray([baseline_shares.get(token, 0.0) for token in tokens])
    actual = np.asarray(
        [fresh_counts.get(token, 0) / len(decisions) for token in tokens]
    )
    return float(compute_psi(expected, actual))


def _segment_token(value) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "segmentation decisions must be finite scalar segment ids"
        ) from exc


def _plan_threshold_check(
    check_id,
    raw_spec,
    metrics: dict[str, float | None],
) -> dict:
    spec = raw_spec if isinstance(raw_spec, dict) else {}
    check_id = str(check_id)
    metric = str(spec.get("metric") or check_id)
    label = str(spec.get("label") or check_id)
    direction = str(spec.get("direction") or "max")
    warn = _finite_float(spec.get("warn"))
    fail = _finite_float(spec.get("fail"))
    actual = _finite_float(metrics.get(metric))
    check = {
        "id": check_id,
        "label": label,
        "metric": metric,
        "value": actual,
        "actual": actual,
        "direction": direction,
        "warn": warn,
        "fail": fail,
    }
    if actual is None:
        return {
            **check,
            "level": "n/a",
            "status": "missing",
            "message": (
                f"本次新鲜样本未提供计算 {metric} 所需的确定性输入；"
                "该指标标记为 n/a，未填充或推断数值。"
            ),
        }
    if direction not in {"min", "max"} or (warn is None and fail is None):
        return {
            **check,
            "level": "n/a",
            "status": "needs_policy",
            "message": "监控计划缺少可执行的 direction/warn/fail 阈值，无法自动判级。",
        }

    if direction == "min":
        if fail is not None and actual < fail - _DRIFT_EPS:
            level, status = "red", "fail"
        elif warn is not None and actual < warn - _DRIFT_EPS:
            level, status = "amber", "warn"
        else:
            level, status = "green", "pass"
    elif fail is not None and actual > fail + _DRIFT_EPS:
        level, status = "red", "fail"
    elif warn is not None and actual > warn + _DRIFT_EPS:
        level, status = "amber", "warn"
    else:
        level, status = "green", "pass"
    return {
        **check,
        "level": level,
        "status": status,
        "message": _threshold_message(
            actual,
            direction=direction,
            warn=warn,
            fail=fail,
            level=level,
        ),
    }


def _threshold_message(
    actual: float,
    *,
    direction: str,
    warn: float | None,
    fail: float | None,
    level: str,
) -> str:
    operator = "低于" if direction == "min" else "高于"
    if level == "red" and fail is not None:
        return f"实际 {actual:.6g} {operator} fail 阈值 {fail:.6g}。"
    if level == "amber" and warn is not None:
        return f"实际 {actual:.6g} {operator} warn 阈值 {warn:.6g}。"
    return (
        f"实际 {actual:.6g} 在监控阈值内"
        f"（warn={warn if warn is not None else 'n/a'}, "
        f"fail={fail if fail is not None else 'n/a'}）。"
    )


def _finite_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strategy_drift_checks(
    frame: pd.DataFrame,
    strategy,
    plan: MonitoringPlan,
    *,
    target_col: str | None,
):
    """Evaluate approval/reject metrics using the versioned plan thresholds.

    Approval-facing ids retain the historical ``*_drift`` projection for API
    compatibility, but their level now comes from the absolute warn/fail values
    committed in the monitoring plan.  Reject plans can therefore execute their
    bad-capture and good-reject contracts instead of silently ignoring them.
    """
    baseline = plan.expectation_baseline or {}
    approved = strategy_approval_mask(frame, strategy)
    row_count = int(len(frame))
    metrics: dict[str, float | None] = {
        "approval_rate": (
            float(approved.sum() / row_count) if row_count else None
        ),
        "approved_bad_rate": None,
        "bad_capture_rate": None,
        "good_reject_rate": None,
    }
    target = _fresh_binary_target(frame, target_col=target_col)
    if target is not None:
        approved_labeled = target.loc[approved & target.notna()]
        if not approved_labeled.empty:
            metrics["approved_bad_rate"] = float(approved_labeled.eq(1).mean())
        rejected = ~approved
        bad = target.eq(1)
        good = target.eq(0)
        bad_count = int(bad.sum())
        good_count = int(good.sum())
        if bad_count:
            metrics["bad_capture_rate"] = float((rejected & bad).sum() / bad_count)
        if good_count:
            metrics["good_reject_rate"] = float((rejected & good).sum() / good_count)

    checks = [
        _approval_plan_check(check_id, spec, metrics, baseline=baseline)
        for check_id, spec in _strategy_monitoring_threshold_items(plan)
    ]
    if not checks:
        # Historical plans without threshold specs retain their old drift-only
        # presentation rather than becoming silently healthy.
        for metric in ("approval_rate", "approved_bad_rate"):
            actual = metrics[metric]
            baseline_value = _optional_float(baseline.get(metric))
            if actual is None:
                checks.append(
                    {
                        "id": f"{metric}_drift",
                        "label": metric,
                        "metric": metric,
                        "value": None,
                        "level": "n/a",
                        "baseline": baseline_value,
                        "actual": None,
                        "message": "监控计划或新鲜样本缺少该指标所需证据。",
                    }
                )
            else:
                checks.append(
                    _drift_check(
                        check_id=f"{metric}_drift",
                        label=metric,
                        actual=actual,
                        baseline=baseline_value,
                        metric=metric,
                    )
                )
    level = _overall_level(check["level"] for check in checks)
    return checks, level


def _approval_plan_check(
    check_id,
    spec,
    metrics: dict[str, float | None],
    *,
    baseline: dict,
) -> dict:
    check = _plan_threshold_check(check_id, spec, metrics)
    metric = str(check["metric"])
    if metric not in {"approval_rate", "approved_bad_rate"}:
        return check
    baseline_value = _optional_float(baseline.get(metric))
    actual = check.get("actual")
    drift = (
        None
        if actual is None or baseline_value is None
        else float(actual) - baseline_value
    )
    return {
        **check,
        "id": f"{metric}_drift",
        "value": drift,
        "baseline": baseline_value,
        "actual": actual,
    }


def _drift_check(
    *,
    check_id: str,
    label: str,
    actual: float,
    baseline: float | None,
    metric: str | None = None,
) -> dict:
    if baseline is None:
        return {
            "id": check_id,
            "label": label,
            "metric": metric or check_id,
            "value": None,
            "level": "n/a",
            "baseline": None,
            "actual": float(actual),
            "message": "监控计划缺少该指标的采纳基线，无法比较漂移。",
        }
    drift = float(actual) - float(baseline)
    level = _drift_level(drift)
    return {
        "id": check_id,
        "label": label,
        "metric": metric or check_id,
        "value": drift,
        "level": level,
        "baseline": float(baseline),
        "actual": float(actual),
        "message": (
            f"实际 {actual:.4f} vs 采纳基线 {baseline:.4f}，漂移 {drift:+.4f}（{_drift_gloss(level)}）。"
        ),
    }


#: Float tolerance so a drift that sits exactly on a band boundary (e.g. an
#: approval rate that moved by precisely 10pp, where 0.7 - 0.8 evaluates to
#: -0.1000000000000001 in IEEE-754) grades to the lower/less-severe tier
#: deterministically instead of flipping on binary-float noise.
_DRIFT_EPS = 1e-9


def _drift_level(drift: float) -> str:
    magnitude = abs(float(drift))
    if magnitude > STRATEGY_DRIFT_RED_PP + _DRIFT_EPS:
        return "red"
    if magnitude > STRATEGY_DRIFT_AMBER_PP + _DRIFT_EPS:
        return "amber"
    return "green"


def _drift_gloss(level: str) -> str:
    return {"red": "红灯", "amber": "黄灯", "green": "绿灯"}.get(level, level)


def _overall_level(levels) -> str:
    values = {str(level) for level in levels if level is not None}
    if "red" in values:
        return "red"
    if "amber" in values:
        return "amber"
    if "green" in values:
        return "green"
    return "n/a"


def _latest_plan_path(runtime: "_Runtime", strategy_id: str) -> Path:
    artifacts = [
        artifact
        for artifact in runtime.strategies.list_strategy_artifacts(strategy_id)
        if artifact.get("kind") == "monitoring_plan_json"
    ]
    if not artifacts:
        raise StrategyError(
            f"策略 {strategy_id} 没有登记的监控计划（monitoring_plan_json）；请先采纳该策略。"
        )
    return Path(artifacts[-1]["path"])


#: Recent immutable monitoring-ledger runs summarised in the report timeline. N
#: is bounded so a long-lived strategy's report shows the recent trend, while the
#: complete run ledger remains the source of record.
_REPORT_TIMELINE_LIMIT = 20

_LEVEL_LABEL = {"green": "绿", "amber": "黄", "red": "红", "n/a": "n/a"}


def tool_render_monitoring_report(inputs: dict, ctx) -> dict:
    """Render only from an integrity-checked persisted disposition receipt.

    The caller may identify the strategy and the source monitoring run whose
    disposition was executed. Verdicts, checks, handoff ids and lifecycle claims
    are reloaded from the immutable run ledger and hash-bound disposition audit;
    they are never accepted as report inputs.
    """
    allowed = {"strategy_id", "source_monitoring_run_id"}
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyError(
            "monitoring report accepts only persisted receipt identifiers; "
            "unexpected fields: " + ", ".join(unexpected)
        )
    runtime = _Runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    task_id = str(ctx.task_id)
    meta = _strategy_meta_for_task(runtime, strategy_id, task_id)
    source_run_id = _required_report_text(
        inputs.get("source_monitoring_run_id"),
        field="source_monitoring_run_id",
    )
    receipt = _load_verified_disposition_receipt(
        runtime,
        task_id=task_id,
        strategy_id=strategy_id,
        source_run_id=source_run_id,
    )
    resolved_run = receipt["resolved_run"]
    run_result = resolved_run.result
    checks = [dict(check) for check in run_result["checks"]]
    overall_level = resolved_run.overall_level
    next_action = _executed_disposition_next_action(
        receipt["action_receipt"],
        strategy_id=strategy_id,
    )

    timeline_runs = _monitoring_timeline_runs(
        runtime,
        strategy_id,
        through_run=resolved_run,
    )
    timeline = [_monitoring_timeline_row(run) for run in timeline_runs]
    markdown = _render_report_markdown(
        strategy_id=strategy_id,
        version=int(meta.get("version", 1)),
        overall_level=overall_level,
        checks=checks,
        timeline=timeline,
        next_action=next_action,
    )

    # Lazy import avoids a module cycle: tools.py re-exports this entrypoint.
    from marvis.packs.strategy.tools import _persist_verified_strategy_markdown

    artifact = _persist_verified_strategy_markdown(
        runtime,
        ctx,
        strategy_id=strategy_id,
        kind="monitoring_report_md",
        filename_prefix=(
            f"monitoring_report_{strategy_id}_v{int(meta.get('version', 1))}_"
            f"{source_run_id}"
        ),
        markdown=markdown,
        evidence={
            "source_monitoring_run_id": source_run_id,
            "source_monitoring_run_hash": receipt["source_run"].result_hash,
            "resolved_monitoring_run_id": resolved_run.id,
            "resolved_monitoring_run_hash": resolved_run.result_hash,
            "disposition_audit_id": receipt["audit_id"],
            "disposition_receipt_hash": receipt["receipt_hash"],
            "timeline_runs": [
                {
                    "monitoring_run_id": run.id,
                    "monitoring_run_hash": run.result_hash,
                }
                for run in timeline_runs
            ],
        },
    )

    result = {
        "strategy_id": strategy_id,
        "report_path": artifact["path"],
        "overall_level": overall_level,
        "timeline": timeline,
        "artifact_id": artifact["artifact_id"],
    }
    result["next_action"] = next_action
    return result


def _monitoring_timeline_runs(
    runtime: "_Runtime",
    strategy_id: str,
    *,
    through_run,
) -> list:
    # The run repository verifies canonical result JSON and its hash. Audit rows
    # remain useful for operations but are not a report evidence source. Freeze
    # this report at the resolved receipt boundary: future runs must not mutate
    # the bytes or identity of an earlier evidence report.
    boundary = (through_run.created_at, through_run.id)
    eligible = [
        run
        for run in runtime.monitoring.list_runs(strategy_id)
        if (run.created_at, run.id) <= boundary
    ]
    if not any(run.id == through_run.id for run in eligible):
        raise StrategyError("resolved monitoring run is absent from report timeline")
    return eligible[-_REPORT_TIMELINE_LIMIT:]


def _monitoring_timeline_row(run) -> dict:
    return {
        "at": run.created_at,
        "overall_level": run.overall_level,
        "dataset_id": run.dataset_id,
        "row_count": run.result.get("row_count"),
        "monitoring_run_id": run.id,
    }


def _load_verified_disposition_receipt(
    runtime: "_Runtime",
    *,
    task_id: str,
    strategy_id: str,
    source_run_id: str,
) -> dict[str, Any]:
    rows = _list_audit_rows(
        runtime.settings.db_path,
        kind="strategy.monitoring.disposition",
        target_ref=source_run_id,
    )
    if len(rows) != 1 or rows[0].get("outcome") != "succeeded":
        raise StrategyError(
            f"verified monitoring disposition receipt not found: {source_run_id}"
        )
    audit = rows[0]
    detail = dict(audit.get("detail") or {})
    if detail.get("receipt_schema_version") != "strategy.monitoring-disposition.v1":
        raise StrategyError("monitoring disposition receipt schema is not verified")
    receipt_hash = _canonical_receipt_hash(detail)
    stored_hash = _required_report_sha256(
        audit.get("inputs_hash"), field="disposition receipt hash"
    )
    if not hmac.compare_digest(receipt_hash, stored_hash):
        raise StrategyError("monitoring disposition receipt hash does not match")
    if (
        str(detail.get("task_id") or "") != task_id
        or str(detail.get("strategy_id") or "") != strategy_id
        or str(detail.get("source_monitoring_run_id") or "") != source_run_id
    ):
        raise StrategyError("monitoring disposition receipt ownership does not match")

    source_run = runtime.monitoring.get_run(source_run_id)
    if source_run is None or source_run.strategy_id != strategy_id:
        raise StrategyError(f"monitoring run not found: {source_run_id}")
    source_hash = detail.get("source_monitoring_run_hash") or detail.get(
        "monitoring_run_result_hash"
    )
    if not isinstance(source_hash, str) or not hmac.compare_digest(
        _required_report_sha256(source_hash, field="source monitoring run hash"),
        source_run.result_hash,
    ):
        raise StrategyError("monitoring disposition source run hash does not match")
    _validate_report_run_contract(source_run)
    source_plan = runtime.monitoring.get_plan(source_run.monitoring_plan_id)
    if source_plan is None or source_plan.strategy_id != strategy_id:
        raise StrategyError("monitoring disposition source plan is missing")

    disposition, status = _validated_report_disposition(
        detail,
        source_level=source_run.overall_level,
    )
    _required_report_text(detail.get("reason"), field="reason")
    if disposition == DISPOSITION_ADJUST_THRESHOLD:
        old_plan_id = _required_report_text(
            detail.get("old_monitoring_plan_id"), field="old_monitoring_plan_id"
        )
        old_plan_revision = detail.get("old_monitoring_plan_revision")
        old_plan_hash = _required_report_sha256(
            detail.get("old_monitoring_plan_hash"),
            field="old_monitoring_plan_hash",
        )
        if (
            source_run.monitoring_plan_id != old_plan_id
            or source_plan.id != old_plan_id
            or not isinstance(old_plan_revision, int)
            or source_plan.revision != old_plan_revision
            or not hmac.compare_digest(source_plan.payload_hash, old_plan_hash)
        ):
            raise StrategyError(
                "threshold disposition source-plan receipt does not match ledger"
            )
        resolved_run_id = _required_report_text(
            detail.get("new_monitoring_run_id"), field="new_monitoring_run_id"
        )
        resolved_hash = _required_report_sha256(
            detail.get("new_monitoring_run_hash"), field="new monitoring run hash"
        )
        plan_id = _required_report_text(
            detail.get("new_monitoring_plan_id"), field="new_monitoring_plan_id"
        )
        plan_revision = detail.get("new_monitoring_plan_revision")
        plan_hash = _required_report_sha256(
            detail.get("new_monitoring_plan_hash"), field="new monitoring plan hash"
        )
    else:
        resolved_run_id = _required_report_text(
            detail.get("resolved_monitoring_run_id"),
            field="resolved_monitoring_run_id",
        )
        if resolved_run_id != source_run_id:
            raise StrategyError("monitoring disposition resolved run does not match source")
        resolved_hash = source_run.result_hash
        plan_id = _required_report_text(
            detail.get("monitoring_plan_id"), field="monitoring_plan_id"
        )
        plan_revision = detail.get("monitoring_plan_revision")
        plan_hash = _required_report_sha256(
            detail.get("monitoring_plan_hash"), field="monitoring_plan_hash"
        )

    resolved_run = runtime.monitoring.get_run(resolved_run_id)
    if resolved_run is None or resolved_run.strategy_id != strategy_id:
        raise StrategyError(f"monitoring run not found: {resolved_run_id}")
    if not hmac.compare_digest(resolved_run.result_hash, resolved_hash):
        raise StrategyError("monitoring disposition resolved run hash does not match")
    if resolved_run.monitoring_plan_id != plan_id:
        raise StrategyError("monitoring disposition plan does not match resolved run")
    plan = runtime.monitoring.get_plan(plan_id)
    if (
        plan is None
        or plan.strategy_id != strategy_id
        or not isinstance(plan_revision, int)
        or plan.revision != plan_revision
        or not hmac.compare_digest(plan.payload_hash, plan_hash)
    ):
        raise StrategyError("monitoring disposition plan receipt does not match ledger")
    _validate_report_run_contract(resolved_run)
    if disposition == DISPOSITION_ADJUST_THRESHOLD and (
        plan.supersedes_plan_id != source_plan.id
        or plan.revision != source_plan.revision + 1
        or resolved_run.result.get("source_monitoring_run_id") != source_run.id
    ):
        raise StrategyError(
            "threshold disposition plan/run lineage does not match source receipt"
        )
    if disposition == DISPOSITION_ADJUST_THRESHOLD:
        _validate_report_threshold_patch(
            source_plan,
            plan,
            detail.get("threshold_patch"),
        )
    if disposition == DISPOSITION_NEW_VERSION:
        _validate_report_handoff(
            runtime,
            detail,
            task_id=task_id,
            strategy_id=strategy_id,
            source_run=source_run,
            source_plan=source_plan,
        )

    action_receipt = {
        **detail,
        "status": status,
        "source_overall_level": source_run.overall_level,
        "overall_level": resolved_run.overall_level,
        "resolved_monitoring_run_id": resolved_run.id,
        "monitoring_plan_id": plan.id,
        "monitoring_plan_revision": plan.revision,
        "monitoring_plan_hash": plan.payload_hash,
    }
    return {
        "audit_id": str(audit["id"]),
        "receipt_hash": receipt_hash,
        "source_run": source_run,
        "resolved_run": resolved_run,
        "action_receipt": action_receipt,
    }


def _validate_report_handoff(
    runtime: "_Runtime",
    detail: Mapping[str, Any],
    *,
    task_id: str,
    strategy_id: str,
    source_run: MonitoringRunRecord,
    source_plan: MonitoringPlanRecord,
) -> None:
    new_task_id = _required_report_text(detail.get("new_task_id"), field="new_task_id")
    new_strategy_id = _required_report_text(
        detail.get("new_strategy_id"), field="new_strategy_id"
    )
    new_dataset_id = _required_report_text(
        detail.get("new_dataset_id"), field="new_dataset_id"
    )
    handoff_rows = _list_audit_rows(
        runtime.settings.db_path,
        kind="strategy.monitoring.new_version_handoff",
        target_ref=source_run.id,
    )
    if len(handoff_rows) != 1 or handoff_rows[0].get("outcome") != "succeeded":
        raise StrategyError("verified monitoring handoff receipt not found")
    handoff_audit = handoff_rows[0]
    handoff = dict(handoff_audit.get("detail") or {})
    expected_handoff_fields = {
        "source_task_id": task_id,
        "new_task_id": new_task_id,
        "parent_strategy_id": strategy_id,
        "new_strategy_id": new_strategy_id,
        "monitoring_run_id": source_run.id,
        "monitoring_plan_id": source_plan.id,
        "source_dataset_id": source_run.dataset_id,
        "new_dataset_id": new_dataset_id,
        "dataset_content_hash": source_run.dataset_content_hash,
        "monitoring_plan_revision": source_plan.revision,
        "monitoring_plan_payload_hash": source_plan.payload_hash,
        "monitoring_run_result_hash": source_run.result_hash,
        "parent_strategy_version": source_plan.strategy_version,
        "new_strategy_version": source_plan.strategy_version + 1,
    }
    if any(handoff.get(key) != value for key, value in expected_handoff_fields.items()):
        raise StrategyError("monitoring handoff receipt does not match disposition")
    handoff_inputs = {
        "source_task_id": task_id,
        "parent_strategy_id": strategy_id,
        "monitoring_run_id": source_run.id,
        "monitoring_run_result_hash": source_run.result_hash,
        "monitoring_plan_id": source_plan.id,
        "monitoring_plan_payload_hash": source_plan.payload_hash,
        "dataset_content_hash": source_run.dataset_content_hash,
    }
    handoff_hash = _required_report_sha256(
        handoff_audit.get("inputs_hash"), field="handoff receipt hash"
    )
    if not hmac.compare_digest(_canonical_receipt_hash(handoff_inputs), handoff_hash):
        raise StrategyError("monitoring handoff receipt hash does not match")

    with connect(runtime.settings.db_path) as conn:
        task = conn.execute(
            "SELECT id, task_type, strategy_input_json FROM tasks WHERE id = ?",
            (new_task_id,),
        ).fetchone()
        strategy = conn.execute(
            """
            SELECT task_id, parent_strategy_id, version, status, asset_status
              FROM strategies
             WHERE id = ?
            """,
            (new_strategy_id,),
        ).fetchone()
        dataset = conn.execute(
            """
            SELECT task_id, role, source_path, content_hash
              FROM datasets
             WHERE id = ?
            """,
            (new_dataset_id,),
        ).fetchone()
        source_dataset = conn.execute(
            "SELECT source_path, content_hash FROM datasets WHERE id = ?",
            (source_run.dataset_id,),
        ).fetchone()
    try:
        strategy_input = (
            None
            if task is None or task["strategy_input_json"] is None
            else json.loads(str(task["strategy_input_json"]))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyError("monitoring handoff task contract is invalid") from exc
    try:
        if strategy is not None:
            validate_lifecycle_pair(
                str(strategy["status"]),
                str(strategy["asset_status"]),
            )
    except StrategyLifecycleError as exc:
        raise StrategyError("monitoring handoff strategy lifecycle is invalid") from exc
    if (
        task is None
        or strategy is None
        or dataset is None
        or source_dataset is None
        or str(task["task_type"]) != "strategy"
        or not isinstance(strategy_input, dict)
        or strategy_input.get("baseline_strategy_id") != strategy_id
        or str(strategy["task_id"]) != new_task_id
        or str(dataset["task_id"]) != new_task_id
        or str(strategy["parent_strategy_id"] or "") != strategy_id
        or int(strategy["version"]) != source_plan.strategy_version + 1
        or str(dataset["role"]) != "strategy.new_version_source"
        or str(dataset["source_path"]) != str(source_dataset["source_path"])
        or str(dataset["content_hash"] or "") != source_run.dataset_content_hash
        or str(source_dataset["content_hash"] or "")
        != source_run.dataset_content_hash
        or str(handoff.get("dataset_source_path") or "")
        != str(source_dataset["source_path"])
        or new_task_id == task_id
    ):
        raise StrategyError("monitoring new-version handoff receipt does not match ledger")


def _validated_report_disposition(
    detail: Mapping[str, Any],
    *,
    source_level: str,
) -> tuple[str | None, str]:
    if "disposition" not in detail or "status" not in detail:
        raise StrategyError("monitoring disposition receipt is incomplete")
    disposition = detail.get("disposition")
    expected = {
        None: "acknowledged",
        DISPOSITION_OBSERVE: "observed",
        DISPOSITION_ADJUST_THRESHOLD: "threshold_adjusted",
        DISPOSITION_NEW_VERSION: "new_version_created",
    }
    if disposition not in expected:
        raise StrategyError("monitoring disposition receipt has unsupported disposition")
    status = _required_report_text(detail.get("status"), field="status")
    if status != expected[disposition]:
        raise StrategyError("monitoring disposition receipt status is inconsistent")
    if disposition is None:
        if source_level == "red":
            raise StrategyError("red monitoring receipt requires an explicit disposition")
    elif source_level != "red":
        raise StrategyError("only a red monitoring receipt may carry a disposition")
    return disposition, status


def _validate_report_run_contract(run: MonitoringRunRecord) -> None:
    checks = run.result.get("checks")
    if (
        run.overall_level not in {"green", "amber", "red", "n/a"}
        or run.result.get("overall_level") != run.overall_level
        or not isinstance(checks, list)
        or any(not isinstance(check, dict) for check in checks)
    ):
        raise StrategyError("monitoring run result violates report contract")


def _validate_report_threshold_patch(
    source_plan: MonitoringPlanRecord,
    resolved_plan: MonitoringPlanRecord,
    reported_patch: object,
) -> None:
    if not isinstance(reported_patch, Mapping) or not reported_patch:
        raise StrategyError("threshold disposition receipt lacks a threshold patch")
    source_payload = source_plan.plan.to_dict()
    resolved_payload = resolved_plan.plan.to_dict()
    for key in ("monitoring_plan_id", "revision", "supersedes_plan_id", "thresholds"):
        source_payload.pop(key, None)
        resolved_payload.pop(key, None)
    if source_payload != resolved_payload:
        raise StrategyError("threshold disposition changed non-threshold plan fields")

    source_thresholds = source_plan.plan.thresholds
    resolved_thresholds = resolved_plan.plan.thresholds
    if set(source_thresholds) != set(resolved_thresholds):
        raise StrategyError("threshold disposition changed the monitoring check set")
    actual_patch: dict[str, dict[str, Any]] = {}
    for check_id in sorted(source_thresholds):
        before = dict(source_thresholds[check_id])
        after = dict(resolved_thresholds[check_id])
        before_static = {key: value for key, value in before.items() if key not in {"warn", "fail"}}
        after_static = {key: value for key, value in after.items() if key not in {"warn", "fail"}}
        if before_static != after_static:
            raise StrategyError("threshold disposition changed check semantics")
        changes = {
            field: after.get(field)
            for field in ("warn", "fail")
            if before.get(field) != after.get(field)
        }
        if changes:
            actual_patch[str(check_id)] = changes
    if not actual_patch or dict(reported_patch) != actual_patch:
        raise StrategyError("threshold disposition patch does not match plan diff")


def _canonical_receipt_hash(detail: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(
            dict(detail),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError("monitoring disposition receipt is not canonical JSON") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _required_report_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"{field} must be a non-empty string")
    return value.strip()


def _required_report_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise StrategyError(f"{field} must be a SHA256 digest")
    return value.lower()


def _markdown_table_cell(value: object) -> str:
    """Keep evidence text inside one Markdown table cell."""

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _markdown_inline_code(value: object) -> str:
    """Render an untrusted receipt value as a single safe CommonMark code span."""

    text = " ".join(str(value).replace("\r", "\n").splitlines())
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{padding}{text}{padding}{fence}"


def _render_report_markdown(
    *,
    strategy_id: str,
    version: int,
    overall_level: str | None,
    checks: list[dict],
    timeline: list[dict],
    next_action: dict | None = None,
) -> str:
    lines = [
        f"# 策略监控报告 — {strategy_id} v{version}",
        "",
    ]
    if overall_level:
        lines.append(f"- 最近一次总体判级：**{_LEVEL_LABEL.get(overall_level, overall_level)}**")
    lines.append(f"- 历史监控次数：{len(timeline)}")
    lines.append("")
    if checks:
        lines.append("## 最近一次监控明细")
        lines.append("")
        lines.append("| 检查项 | 判级 | 值 | 说明 |")
        lines.append("| --- | --- | --- | --- |")
        for check in checks:
            value = check.get("value")
            value_text = "n/a" if value is None else f"{float(value):+.4f}" if isinstance(value, (int, float)) else str(value)
            lines.append(
                f"| {_markdown_table_cell(check.get('label') or check.get('id'))} "
                f"| {_markdown_table_cell(_LEVEL_LABEL.get(str(check.get('level')), check.get('level')))} "
                f"| {_markdown_table_cell(value_text)} "
                f"| {_markdown_table_cell(check.get('message') or '')} |"
            )
        lines.append("")
    if timeline:
        lines.append("## 监控判级时间线")
        lines.append("")
        lines.append("| 时间 | 总体判级 | 样本量 |")
        lines.append("| --- | --- | --- |")
        for entry in timeline:
            lines.append(
                f"| {_markdown_table_cell(entry.get('at') or '')} "
                f"| {_markdown_table_cell(_LEVEL_LABEL.get(str(entry.get('overall_level')), entry.get('overall_level')))} "
                f"| {_markdown_table_cell(entry.get('row_count') if entry.get('row_count') is not None else '')} |"
            )
        lines.append("")
    if next_action is not None:
        lines.extend(_render_executed_disposition_markdown(next_action))
    return "\n".join(lines)


def _render_executed_disposition_markdown(next_action: dict) -> list[str]:
    lines = ["## 处置结果", "", str(next_action.get("prompt") or "处置已记录。")]
    fields = (
        ("action", "处置动作"),
        ("status", "执行状态"),
        ("reason", "处置理由"),
        ("source_monitoring_run_id", "来源监控运行"),
        ("source_overall_level", "来源判级"),
        ("new_task_id", "新任务"),
        ("new_strategy_id", "新策略"),
        ("new_dataset_id", "新数据集"),
        ("old_monitoring_plan_id", "原监控计划"),
        ("old_monitoring_plan_revision", "原计划 revision"),
        ("monitoring_plan_id", "新监控计划"),
        ("monitoring_plan_revision", "监控计划 revision"),
        ("monitoring_run_id", "处置后监控运行"),
        ("overall_level", "处置后判级"),
        ("resolved_monitoring_run_id", "已处置监控运行"),
    )
    for key, label in fields:
        value = next_action.get(key)
        if value not in (None, ""):
            lines.append(f"- {label}：{_markdown_inline_code(value)}")
    threshold_patch = next_action.get("threshold_patch")
    if isinstance(threshold_patch, Mapping) and threshold_patch:
        lines.extend(
            [
                "- 阈值修改：",
                "",
                "```json",
                json.dumps(
                    dict(threshold_patch),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ),
                "```",
            ]
        )
    lines.append("")
    return lines


#: Governed monitoring disposition codes. Every non-null action is executed by
#: ``apply_monitoring_disposition`` before a report projects its receipt.
DISPOSITION_OBSERVE = "observe"
DISPOSITION_ADJUST_THRESHOLD = "adjust_threshold"
DISPOSITION_NEW_VERSION = "new_version"


def _executed_disposition_next_action(
    disposition_result: Mapping[str, Any],
    *,
    strategy_id: str,
) -> dict:
    disposition = _optional_str(disposition_result.get("disposition"))
    status = _optional_str(disposition_result.get("status")) or "completed"
    common = {
        "status": status,
        "reason": _optional_str(disposition_result.get("reason")),
        "source_monitoring_run_id": _optional_str(
            disposition_result.get("source_monitoring_run_id")
        ),
        "source_overall_level": _optional_str(
            disposition_result.get("source_overall_level")
        ),
    }
    if disposition == DISPOSITION_NEW_VERSION:
        new_task_id = _optional_str(disposition_result.get("new_task_id"))
        new_strategy_id = _optional_str(disposition_result.get("new_strategy_id"))
        new_dataset_id = _optional_str(disposition_result.get("new_dataset_id"))
        return {
            "kind": "completed",
            "action": DISPOSITION_NEW_VERSION,
            **common,
            "parent_strategy_id": strategy_id,
            "new_task_id": new_task_id,
            "new_strategy_id": new_strategy_id,
            "new_dataset_id": new_dataset_id,
            "prompt": (
                f"新版本已创建：任务 `{new_task_id or 'n/a'}`，策略 "
                f"`{new_strategy_id or 'n/a'}`，数据集 `{new_dataset_id or 'n/a'}`。"
            ),
        }
    if disposition == DISPOSITION_ADJUST_THRESHOLD:
        monitoring_run_id = _optional_str(
            disposition_result.get("resolved_monitoring_run_id")
            or disposition_result.get("monitoring_run_id")
        )
        plan_id = _optional_str(disposition_result.get("monitoring_plan_id"))
        revision = disposition_result.get("monitoring_plan_revision")
        overall_level = _optional_str(disposition_result.get("overall_level"))
        return {
            "kind": "completed",
            "action": DISPOSITION_ADJUST_THRESHOLD,
            **common,
            "old_monitoring_plan_id": _optional_str(
                disposition_result.get("old_monitoring_plan_id")
            ),
            "old_monitoring_plan_revision": disposition_result.get(
                "old_monitoring_plan_revision"
            ),
            "threshold_patch": disposition_result.get("threshold_patch"),
            "monitoring_plan_id": plan_id,
            "monitoring_plan_revision": revision,
            "monitoring_plan_hash": _optional_str(
                disposition_result.get("monitoring_plan_hash")
            ),
            "monitoring_run_id": monitoring_run_id,
            "overall_level": overall_level,
            "prompt": (
                f"阈值调整已执行：计划 `{plan_id or 'n/a'}` revision "
                f"{revision if revision is not None else 'n/a'}，重跑 "
                f"`{monitoring_run_id or 'n/a'}`，判级 `{overall_level or 'n/a'}`。"
            ),
        }
    if disposition == DISPOSITION_OBSERVE:
        resolved_run_id = _optional_str(
            disposition_result.get("resolved_monitoring_run_id")
        )
        return {
            "kind": "recorded",
            "action": DISPOSITION_OBSERVE,
            **common,
            "resolved_monitoring_run_id": resolved_run_id,
            "overall_level": _optional_str(
                disposition_result.get("overall_level")
            ),
            "prompt": (
                f"维持并观察已记录：监控运行 `{resolved_run_id or 'n/a'}`，"
                "当前策略与计划保持不变。"
            ),
        }
    resolved_run_id = _optional_str(
        disposition_result.get("resolved_monitoring_run_id")
    )
    return {
        "kind": "recorded",
        "action": "acknowledge",
        **common,
        "resolved_monitoring_run_id": resolved_run_id,
        "overall_level": _optional_str(disposition_result.get("overall_level")),
        "prompt": f"监控运行 `{resolved_run_id or 'n/a'}` 已确认并记录。",
}
class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.strategies = StrategyRepository(self.settings.db_path)
        self.monitoring = StrategyMonitoringRepository(self.settings.db_path)

def _strategy_meta_for_task(
    runtime: _Runtime, strategy_id: str, task_id: str
) -> dict:
    metadata = runtime.strategies.get_strategy_meta(strategy_id)
    if metadata is None or str(metadata["task_id"]) != str(task_id):
        raise StrategyError(f"strategy not found: {strategy_id}")
    return metadata


def _require_locally_adopted(record, *, strategy_id: str) -> None:
    status = record["status"] if "status" in record.keys() else None
    asset_status = (
        record["asset_status"] if "asset_status" in record.keys() else None
    )
    try:
        adopted = is_locally_adopted(status, asset_status)
    except StrategyLifecycleError as exc:
        raise StrategyError(
            f"strategy {strategy_id} lifecycle state is invalid"
        ) from exc
    if not adopted:
        raise StrategyNotAdoptedError(
            strategy_id=strategy_id,
            status=status,
            asset_status=asset_status,
        )


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DISPOSITION_ADJUST_THRESHOLD",
    "DISPOSITION_NEW_VERSION",
    "DISPOSITION_OBSERVE",
    "STRATEGY_DRIFT_AMBER_PP",
    "STRATEGY_DRIFT_RED_PP",
    "rerun_strategy_monitoring_with_candidate_plan",
    "tool_render_monitoring_report",
    "tool_run_strategy_monitoring",
]
