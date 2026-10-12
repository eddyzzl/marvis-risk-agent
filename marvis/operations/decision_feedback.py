"""Authenticated outcome declarations linked to immutable native decisions.

No original feature payload is retained or inferred. This monitors the recorded
decision cohort and publisher-declared outcomes, not retrospective rule replay.
"""

from datetime import UTC, datetime, timedelta
import json
from zoneinfo import ZoneInfo

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.operations.extension_bindings import FeedbackImport, DecisionFeedbackBinding
from marvis.operations.monitoring import (
    BuiltinMonitoringExecutor,
    MonitoringExecutionError,
)
from marvis.operations.schedule_proposals import sha
from marvis.plugins.manifest import ToolRef
from marvis.reference_decision.packages import PackageStore
from marvis.repositories.datasets import DatasetRepository

FEEDBACK_REF = "analysis.decision_ledger_feedback"


def package_for_task(settings, package_hash, task_id):
    # Workspace capability remains local, never in prompts/requests/evidence.
    secret = settings.plugin_admin_token_path.read_text().strip().encode()
    manifest = PackageStore(settings, secret).get(package_hash, verify_files=False)
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT task_id FROM strategies WHERE id=?", (manifest["strategy"]["id"],)
        ).fetchone()
    if row is None or row["task_id"] != task_id:
        raise ValueError("decision package task owner mismatch")
    return manifest


def authenticated_decision(
    conn, environment, request_id, *, output_hash=None, package_hash=None
):
    row = conn.execute(
        "SELECT * FROM reference_decisions WHERE environment=? AND request_id=? AND status='completed'",
        (environment, request_id),
    ).fetchone()
    if row is None:
        raise ValueError("completed native decision required")
    response = json.loads(row["response_json"])
    declared = {
        key: response[key]
        for key in (
            "action",
            "matched_rule_id",
            "score",
            "score_product",
            "package_hash",
            "versions",
            "status",
        )
    }
    if (
        response.get("output_hash") != sha(declared)
        or response.get("package_hash") != row["package_hash"]
        or response.get("request_id") != request_id
        or response.get("input_hash") != row["input_hash"]
        or response.get("execution_identity") != "local_reference_worker.v1"
        or output_hash is not None
        and response["output_hash"] != output_hash
        or package_hash is not None
        and row["package_hash"] != package_hash
    ):
        raise ValueError("native decision receipt identity differs")
    return dict(row), response


def _time(value, zone):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("known event and availability time required")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(zone, ambiguous="raise", nonexistent="raise")
    return parsed.tz_convert("UTC").to_pydatetime()


def import_feedback(settings, binding, *, actor_id, now=None):
    if not isinstance(binding, FeedbackImport):
        raise ValueError("installed feedback source contract required")
    now = (now or datetime.now(UTC)).astimezone(UTC)
    zone = ZoneInfo(binding.timezone)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    source = registry.authenticate_dataset_binding(
        binding.dataset_id,
        expected_task_id=binding.task_id,
        expected_content_hash=binding.dataset_content_hash,
    )
    if source.row_count > 200_000:
        raise ValueError("feedback source row budget exceeded")
    frame = registry.read_authenticated_binding_snapshot(source)
    columns = [
        binding.request_id_col,
        binding.output_hash_col,
        binding.package_hash_col,
        binding.label_col,
        binding.event_at_col,
        binding.available_at_col,
    ]
    if len(set(columns)) != 6 or any(name not in frame for name in columns):
        raise ValueError("complete distinct feedback columns required")
    packages = set(frame[binding.package_hash_col])
    if len(packages) > 100:
        raise ValueError("feedback package count budget exceeded")
    for package in packages:
        package_for_task(settings, package, binding.task_id)
    inserted = replayed = 0
    with connect(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        from marvis.institution.policy import authorize_on_connection

        authorize_on_connection(
            conn, "task", binding.task_id, "execute", actor_id=actor_id
        )
        for index, row in enumerate(frame.to_dict("records")):
            request_id = row[binding.request_id_col]
            if (
                not isinstance(request_id, str)
                or not request_id.strip()
                or len(request_id) > 200
            ):
                raise ValueError("bounded decision request identity required")
            decision, response = authenticated_decision(
                conn,
                binding.environment,
                request_id,
                output_hash=row[binding.output_hash_col],
                package_hash=row[binding.package_hash_col],
            )
            label = row[binding.label_col]
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError("explicit binary outcome required")
            event, available = (
                _time(row[binding.event_at_col], zone),
                _time(row[binding.available_at_col], zone),
            )
            if (
                not datetime.fromisoformat(response["created_at"])
                <= event
                <= available
                <= now
            ):
                raise ValueError(
                    "feedback event/availability chronology cannot be certified"
                )
            value = {
                "environment": binding.environment,
                "request_id": request_id,
                "task_id": binding.task_id,
                "decision_output_hash": response["output_hash"],
                "package_hash": decision["package_hash"],
                "label": int(label),
                "event_at": event.isoformat(),
                "available_at": available.isoformat(),
                "source_dataset_id": binding.dataset_id,
                "source_content_hash": binding.dataset_content_hash,
                "source_row": index,
                "actor_id": actor_id,
            }
            digest = sha(value)
            old = conn.execute(
                "SELECT feedback_hash FROM operations_decision_feedback WHERE environment=? AND request_id=?",
                (binding.environment, request_id),
            ).fetchone()
            if old:
                if old["feedback_hash"] != digest:
                    raise ValueError("immutable feedback declaration conflicts")
                replayed += 1
            else:
                conn.execute(
                    "INSERT INTO operations_decision_feedback VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*value.values(), digest, now.isoformat()),
                )
                inserted += 1
        registry.verify_dataset_binding_on_connection(conn, source)
    return {
        "inserted": inserted,
        "replayed": replayed,
        "source_dataset_id": binding.dataset_id,
        "source_content_hash": binding.dataset_content_hash,
        "outcome_assurance": "publisher_declared",
        "automatic_action_permitted": False,
    }


def feedback_window(settings, binding, starts_at, ends_at, evaluated_at):
    package_for_task(settings, binding.package_hash, binding.task_id)
    if binding.complete_through < ends_at:
        return {"overall_level": "not_available", "reason_code": "window_incomplete"}
    if binding.complete_through > evaluated_at:
        return {
            "overall_level": "not_available",
            "reason_code": "future_source_watermark",
        }
    with connect(settings.db_path) as conn:
        decisions = conn.execute(
            "SELECT request_id FROM reference_decisions WHERE environment=? AND package_hash=? AND status='completed' AND created_at>=? AND created_at<? ORDER BY created_at,request_id LIMIT ?",
            (
                binding.environment,
                binding.package_hash,
                starts_at.isoformat(),
                ends_at.isoformat(),
                binding.max_source_rows + 1,
            ),
        ).fetchall()
        if len(decisions) > binding.max_source_rows:
            raise ValueError("feedback decision row budget exceeded")
        registry = DatasetRegistry(
            DatasetRepository(settings.db_path),
            DataBackend(settings.datasets_dir),
            settings.datasets_dir,
        )
        sources = {}
        values = []
        ids = []
        for row in decisions:
            decision, response = authenticated_decision(
                conn,
                binding.environment,
                row["request_id"],
                package_hash=binding.package_hash,
            )
            feedback = conn.execute(
                "SELECT * FROM operations_decision_feedback WHERE environment=? AND request_id=? AND task_id=?",
                (binding.environment, row["request_id"], binding.task_id),
            ).fetchone()
            label = None
            if (
                feedback
                and datetime.fromisoformat(feedback["available_at"]) <= evaluated_at
            ):
                if feedback["decision_output_hash"] != response["output_hash"]:
                    raise ValueError("feedback no longer binds native decision")
                if datetime.fromisoformat(
                    feedback["event_at"]
                ) >= datetime.fromisoformat(response["created_at"]) + timedelta(
                    seconds=binding.label_maturity_seconds
                ):
                    label = feedback["label"]
                identity = (
                    feedback["source_dataset_id"],
                    feedback["source_content_hash"],
                )
                if identity not in sources:
                    source = registry.authenticate_dataset_binding(
                        identity[0],
                        expected_task_id=binding.task_id,
                        expected_content_hash=identity[1],
                    )
                    registry.verify_dataset_binding(source)
                    sources[identity] = source
                ids.append(feedback["feedback_hash"])
            values.append((response, label))
    if not values:
        return {"overall_level": "not_available", "reason_code": "no_new_data"}
    labeled = [(r, y) for r, y in values if y is not None]
    fallback = sum(r["status"] == "fallback" for r, _ in values)
    approved = [(r, y) for r, y in labeled if r["action"]["type"] == "approval"]
    bad_rate = sum(y for _, y in labeled) / len(labeled) if labeled else None
    fallback_rate = fallback / len(values)
    enough = len(labeled) >= binding.min_labeled_count
    level = (
        "not_available"
        if not enough
        else "red"
        if bad_rate > binding.max_bad_rate or fallback_rate > binding.max_fallback_rate
        else "green"
    )
    return {
        "overall_level": level,
        "reason_code": "labels_incomplete" if not enough else "monitoring_" + level,
        "population_count": len(values),
        "labeled_count": len(labeled),
        "label_coverage": len(labeled) / len(values),
        "bad_count": sum(y for _, y in labeled),
        "bad_rate": bad_rate,
        "approval_labeled_count": len(approved),
        "approval_bad_rate": sum(y for _, y in approved) / len(approved)
        if approved
        else None,
        "fallback_count": fallback,
        "fallback_rate": fallback_rate,
        "package_hash": binding.package_hash,
        "feedback_refs_sha256": sha(sorted(ids)),
        "source_refs": [
            {"dataset_id": a, "content_hash": b} for a, b in sorted(sources)
        ],
        "decision_assurance": "local_reference_native_receipt",
        "outcome_assurance": "publisher_declared",
        "label_maturity_assurance": "publisher_declared_elapsed_time_checked",
        "rule_replay": "unavailable_original_features_not_retained",
        "score_psi": None,
        "economic_outcome": None,
        "automatic_action_permitted": False,
    }


def tool_decision_ledger_feedback(inputs, ctx):
    from marvis.settings import build_settings

    binding = DecisionFeedbackBinding.model_validate(inputs["binding"])
    if ctx.task_id != binding.task_id:
        raise ValueError("feedback task binding differs")
    starts, ends, evaluated = [
        datetime.fromisoformat(inputs[key].replace("Z", "+00:00"))
        for key in ("starts_at", "ends_at", "evaluated_at")
    ]
    if (
        any(value.tzinfo is None for value in (starts, ends, evaluated))
        or not starts < ends <= evaluated
    ):
        raise ValueError("complete known feedback calendar window required")
    return feedback_window(
        build_settings(ctx.workspace), binding, starts, ends, evaluated
    )


class BuiltinDecisionFeedbackExecutor(BuiltinMonitoringExecutor):
    def validate(self, contract):
        binding = contract.feedback_binding
        if binding is None or contract.monitoring_ref != FEEDBACK_REF:
            raise ValueError("native feedback binding required")
        package_for_task(self.settings, binding.package_hash, binding.task_id)
        self.runner.prepare_invocation(ToolRef("analysis", "decision_ledger_feedback"))

    def __call__(self, request):
        record = self.store.get_schedule(
            request.schedule_id, revision=request.schedule_revision
        )
        if record is None or record.contract.monitoring_ref != FEEDBACK_REF:
            raise MonitoringExecutionError("monitoring_binding_missing")
        binding = record.contract.feedback_binding
        ref = ToolRef("analysis", "decision_ledger_feedback")
        from marvis.operations.native_invocations import invoke_scheduled

        result, invocation = invoke_scheduled(
            self,
            request,
            ref,
            {
                "binding": binding.model_dump(mode="json"),
                "starts_at": request.period.starts_at.isoformat(),
                "ends_at": request.period.ends_at.isoformat(),
                "evaluated_at": self.clock().isoformat(),
            },
            suffix="decision_feedback",
        )
        if not result.ok or not isinstance(result.output, dict):
            raise MonitoringExecutionError("monitoring_tool_failed")
        evidence = {
            "schema_version": "operations.decision_feedback_monitor.v1",
            "schedule_id": request.schedule_id,
            "schedule_revision": request.schedule_revision,
            "schedule_hash": record.contract_hash,
            "period_key": request.period.key,
            "run_id": request.run_id,
            "monitoring_ref": FEEDBACK_REF,
            "window_starts_at": request.period.starts_at.isoformat(),
            "window_ends_at": request.period.ends_at.isoformat(),
            "tool_invocation": invocation,
            "tool_output": result.output,
            "tool_output_hash": sha(result.output),
            "raw_output_hash": result.raw_output_hash or None,
            "binding": binding.model_dump(mode="json"),
        }
        return self._outcome(
            evidence, result.output["overall_level"], result.output["reason_code"]
        )
