"""Trusted scheduled bindings to the existing deterministic monitoring tools."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pandas as pd


from marvis.operations.diagnostics import diagnostic
from marvis.data.backend import DataBackend
from marvis.data.errors import DataLayerError
from marvis.plugins.errors import PluginError
from marvis.data.registry import DatasetRegistry
from marvis.operations.contracts import HumanEscalationSuggestion, MonitoringOutcome
from marvis.files import write_json_atomic
from marvis.plugins.manifest import ToolRef
from marvis.repositories.datasets import DatasetRepository

BUILTIN_MONITORS = ("modeling.monitor_run", "strategy.run_strategy_monitoring")


class MonitoringExecutionError(RuntimeError):
    def __init__(self, code: str):
        self.error_code = code
        super().__init__(code)


class BuiltinMonitoringExecutor:
    def __init__(self, settings, store, tool_runner, *, clock=None, stop_event=None):
        self.stop_event = stop_event
        self.settings = settings
        self.store = store
        self.runner = tool_runner
        self.clock = clock or (lambda: datetime.now(UTC))
        self.registry = DatasetRegistry(
            DatasetRepository(settings.db_path),
            DataBackend(settings.datasets_dir),
            settings.datasets_dir,
        )

    def validate(self, contract):
        binding = contract.monitoring_binding
        if binding is None:
            raise ValueError("built-in monitoring requires monitoring_binding")
        try:
            self.registry.authenticate_dataset_binding(
                binding.dataset_id,
                expected_task_id=binding.task_id,
                expected_content_hash=binding.dataset_content_hash,
            )
            # The trusted registry, not user-controlled imports, selects code.
            plugin, tool = contract.monitoring_ref.split(".", 1)
            self.runner.prepare_invocation(ToolRef(plugin, tool))
        except (DataLayerError, KeyError, OSError, PluginError) as exc:
            raise ValueError(
                "monitoring source or tool binding could not be authenticated"
            ) from exc

    def __call__(self, request):
        schedule = self.store.get_schedule(
            request.schedule_id, revision=request.schedule_revision
        )
        if (
            schedule is None
            or schedule.contract.monitoring_ref != request.monitoring_ref
        ):
            raise MonitoringExecutionError("monitoring_binding_missing")
        binding = schedule.contract.monitoring_binding
        if binding is None:
            raise MonitoringExecutionError("monitoring_binding_missing")
        evidence = {
            "schema_version": "operations.builtin_monitor.v1",
            "schedule_id": request.schedule_id,
            "schedule_revision": request.schedule_revision,
            "schedule_hash": schedule.contract_hash,
            "period_key": request.period.key,
            "run_id": request.run_id,
            "monitoring_ref": request.monitoring_ref,
            "source_dataset_id": binding.dataset_id,
            "source_content_hash": binding.dataset_content_hash,
            "coverage_assurance": "publisher_declared",
            "declared_complete_through": binding.complete_through.isoformat(),
            "label_mode": binding.label_mode,
            "label_maturity_seconds": binding.label_maturity_seconds,
        }
        try:
            authenticated = self.registry.authenticate_dataset_binding(
                binding.dataset_id,
                expected_task_id=binding.task_id,
                expected_content_hash=binding.dataset_content_hash,
            )
            if authenticated.row_count > binding.max_source_rows:
                raise ValueError("source row budget exceeded")
            frame = self.registry.read_authenticated_binding_snapshot(authenticated)
            if binding.time_col not in frame:
                raise ValueError("missing time column")
            times = pd.to_datetime(
                frame[binding.time_col], errors="raise", format="mixed"
            )
            if times.isna().any():
                raise ValueError("null event time")
            if times.dt.tz is None:
                times = times.dt.tz_localize(
                    binding.timezone, ambiguous="raise", nonexistent="raise"
                )
            times = times.dt.tz_convert("UTC")
        except Exception as exc:
            raise MonitoringExecutionError("monitoring_source_error") from exc
        evidence["source_max_event_at"] = (
            None if times.empty else times.max().isoformat()
        )
        window = frame.loc[
            (times >= request.period.starts_at) & (times < request.period.ends_at)
        ].copy()
        evidence["window_row_count"] = len(window)
        if window.empty:
            return self._outcome(evidence, "not_available", "no_new_data")
        if binding.complete_through < request.period.ends_at:
            return self._outcome(evidence, "not_available", "window_incomplete")
        if binding.complete_through > self.clock():
            return self._outcome(evidence, "not_available", "future_source_watermark")
        if binding.label_mode == "required":
            mature_at = request.period.ends_at + pd.Timedelta(
                seconds=binding.label_maturity_seconds
            )
            if mature_at > self.clock():
                return self._outcome(evidence, "not_available", "labels_immature")
            if (
                binding.target_col not in window
                or not window[binding.target_col].isin([0, 1]).all()
            ):
                return self._outcome(evidence, "not_available", "labels_incomplete")
        self._check_cancelled()
        # Existing tools receive only this calendar window, never the full stale
        # source. Registered outputs remain task-bound and content-addressed.
        directory = self.settings.datasets_dir / binding.task_id / "operations"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{request.run_id}.parquet"
        window.to_parquet(path, index=False)
        dataset = self.registry.register_existing(
            path,
            task_id=binding.task_id,
            role="monitoring_window",
            anchor_target=binding.dataset_id,
        )
        pinned = self.registry.pin_authenticated_snapshot(
            dataset.id,
            expected_task_id=binding.task_id,
            expected_content_hash=dataset.content_hash,
        )
        inputs = {"dataset_id": pinned.id}
        inputs[
            "experiment_id"
            if request.monitoring_ref == "modeling.monitor_run"
            else "strategy_id"
        ] = binding.target_id
        if binding.target_col:
            inputs["target_col"] = binding.target_col
        if binding.score_col:
            inputs["score_col"] = binding.score_col
        plugin, tool = request.monitoring_ref.split(".", 1)
        result = self.runner.invoke(
            ToolRef(plugin, tool),
            inputs,
            task_id=binding.task_id,
            invocation_id=f"operations:{request.run_id}",
            cancellation_check=self._check_cancelled,
        )
        if not result.ok or not isinstance(result.output, dict):
            raise MonitoringExecutionError("monitoring_tool_failed")
        self.registry.verify_dataset_binding(authenticated)
        level = result.output.get("overall_level")
        if level not in {"green", "amber", "red", "not_available"}:
            raise MonitoringExecutionError("monitoring_result_invalid")
        evidence.update(
            {
                "window_dataset_id": pinned.id,
                "window_content_hash": pinned.content_hash,
                "tool_output": result.output,
                "tool_version": result.tool_version,
                "manifest_hash": result.manifest_hash,
            }
        )
        return self._outcome(evidence, level, f"monitoring_{level}")

    def _check_cancelled(self):
        if self.stop_event is not None and self.stop_event.is_set():
            raise MonitoringExecutionError("operations_stopped")

    def _outcome(self, evidence, level, reason):
        evidence.update(
            {"level": level, "reason_code": reason, "diagnostic": diagnostic(reason)}
        )
        encoded = json.dumps(
            evidence,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.settings.workspace / "operations" / "evidence" / f"{digest}.json"
        # Hash addresses canonical JSON independently of the readable file layout.
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != evidence:
                raise MonitoringExecutionError("monitoring_evidence_conflict")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, evidence)
        return MonitoringOutcome(
            level=level,
            evidence_ref=f"operations-evidence:{digest}",
            evidence_hash=digest,
            escalation=None
            if level == "green"
            else HumanEscalationSuggestion(reason_code=reason),
        )


def read_monitoring_evidence(workspace: Path, outcome: MonitoringOutcome) -> dict:
    if outcome.evidence_ref != f"operations-evidence:{outcome.evidence_hash}":
        raise ValueError("not a built-in monitoring evidence reference")
    root = workspace / "operations" / "evidence"
    path = root / f"{outcome.evidence_hash}.json"
    if (
        not root.resolve().is_relative_to(workspace.resolve())
        or path.is_symlink()
        or path.resolve().parent != root.resolve()
    ):
        raise ValueError("monitoring evidence escaped workspace")
    payload = json.loads(path.read_text())
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != outcome.evidence_hash:
        raise ValueError("monitoring evidence content changed")
    return payload
