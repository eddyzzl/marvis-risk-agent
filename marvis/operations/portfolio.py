"""Periodic portfolio adapter over installed deterministic analysis Tools."""

from marvis.plugins.manifest import ToolRef
from marvis.operations.monitoring import (
    BuiltinMonitoringExecutor,
    MonitoringExecutionError,
)
from marvis.operations.windows import source_window
from marvis.operations.schedule_proposals import sha

PORTFOLIO_REF = "analysis.portfolio_run"
_TOOLS = (
    "flow_rate",
    "bucket_migration",
    "segment_profile",
    "expected_loss_estimate",
    "portfolio_report",
)


class BuiltinPortfolioExecutor(BuiltinMonitoringExecutor):
    def validate(self, contract):
        binding = contract.portfolio_binding
        if binding is None or contract.monitoring_ref != PORTFOLIO_REF:
            raise ValueError("periodic portfolio requires its exact installed binding")
        self.registry.authenticate_dataset_binding(
            binding.source.dataset_id,
            expected_task_id=binding.source.task_id,
            expected_content_hash=binding.source.dataset_content_hash,
        )
        for name in _TOOLS:
            self.runner.prepare_invocation(ToolRef("analysis", name))
        if binding.experiment_id:
            self.runner.prepare_invocation(ToolRef("analysis", "score_stability_trend"))

    def __call__(self, request):
        record = self.store.get_schedule(
            request.schedule_id, revision=request.schedule_revision
        )
        if record is None or record.contract.monitoring_ref != PORTFOLIO_REF:
            raise MonitoringExecutionError("monitoring_binding_missing")
        binding = record.contract.portfolio_binding
        authenticated, frame, reason, source = source_window(
            self,
            binding.source,
            request.period,
            lookback_seconds=binding.lookback_seconds,
        )
        evidence = {
            "schema_version": "operations.periodic_portfolio.v1",
            "schedule_id": request.schedule_id,
            "schedule_revision": request.schedule_revision,
            "schedule_hash": record.contract_hash,
            "period_key": request.period.key,
            "run_id": request.run_id,
            "monitoring_ref": PORTFOLIO_REF,
            **source,
            "business_acceptance": "not_established",
        }
        if reason:
            return self._outcome(evidence, "not_available", reason)
        required = [
            binding.id_col,
            binding.bucket_col,
            binding.balance_col,
            binding.segment_col,
            binding.source.time_col,
        ]
        if binding.score_col:
            required.append(binding.score_col)
        if any(name not in frame for name in required):
            raise MonitoringExecutionError("monitoring_source_error")
        if frame[binding.source.time_col].nunique() < 2:
            return self._outcome(
                evidence, "not_available", "portfolio_previous_snapshot_missing"
            )
        self._check_cancelled()
        directory = self.settings.datasets_dir / binding.source.task_id / "operations"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{request.run_id}.parquet"
        frame.to_parquet(path, index=False)
        dataset = self.registry.register_existing(
            path,
            task_id=binding.source.task_id,
            role="portfolio_window",
            anchor_target=binding.source.dataset_id,
        )
        pinned = self.registry.pin_authenticated_snapshot(
            dataset.id,
            expected_task_id=binding.source.task_id,
            expected_content_hash=dataset.content_hash,
        )
        base = {
            "dataset_id": pinned.id,
            "expected_content_hash": pinned.content_hash,
            "id_col": binding.id_col,
            "snapshot_col": binding.source.time_col,
            "bucket_col": binding.bucket_col,
            "states": binding.states,
            "balance_col": binding.balance_col,
        }
        outputs = {}

        def invoke(name, inputs):
            ref = ToolRef("analysis", name)
            from marvis.operations.native_invocations import invoke_scheduled

            result, invocation = invoke_scheduled(
                self, request, ref, inputs, suffix=name
            )
            if not result.ok or not isinstance(result.output, dict):
                raise MonitoringExecutionError("monitoring_tool_failed")
            outputs[name] = {
                "invocation": invocation,
                "output": result.output,
                "output_hash": sha(result.output),
                "raw_output_hash": result.raw_output_hash or None,
            }
            return result.output

        flow = invoke("flow_rate", base)
        migration = invoke("bucket_migration", base)
        segment = invoke(
            "segment_profile",
            {
                "dataset_id": pinned.id,
                "expected_content_hash": pinned.content_hash,
                "segment_col": binding.segment_col,
                "ead_col": binding.balance_col,
            },
        )
        loss = invoke(
            "expected_loss_estimate",
            {
                **base,
                "loss_state": binding.loss_state,
                "lgd": binding.lgd,
                "horizon_months": binding.horizon_months,
            },
        )
        report_inputs = {
            "flow": flow,
            "migration": migration,
            "segment": segment,
            "expected_loss": loss,
        }
        if binding.experiment_id:
            report_inputs["trend"] = invoke(
                "score_stability_trend",
                {
                    "dataset_id": pinned.id,
                    "expected_content_hash": pinned.content_hash,
                    "experiment_id": binding.experiment_id,
                    "month_col": binding.source.time_col,
                    "score_col": binding.score_col,
                },
            )
        if binding.generate_report:
            invoke("portfolio_report", report_inputs)
        self.registry.verify_dataset_binding(authenticated)
        evidence.update(
            window_dataset_id=pinned.id,
            window_content_hash=pinned.content_hash,
            tool_results=outputs,
            assumptions={
                "ordered_states": binding.states,
                "loss_state": binding.loss_state,
                "lgd": binding.lgd,
                "horizon_months": binding.horizon_months,
            },
            human_business_decision="not_taken",
            automatic_action_permitted=False,
        )
        flags = [
            flag
            for value in report_inputs.values()
            for flag in value.get("red_flags", [])
            if isinstance(value, dict)
        ]
        evidence["red_flag_count"] = len(flags)
        return self._outcome(
            evidence,
            "red" if flags else "green",
            "monitoring_red" if flags else "monitoring_green",
        )
