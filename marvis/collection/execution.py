"""Local reference queue effects, committed with the existing Approval ledger.

No customer destination, provider client, money movement or autonomous contact
exists here. The source/reference clock and actual platform timestamps differ.
"""

from collections import Counter
from datetime import UTC, datetime
import json

from marvis.artifacts import ArtifactUnitOfWork
from marvis.collection.batches import CollectionBatchRequest, CollectionBatchStore
from marvis.collection.execution_state import (
    OPERATIONS,
    batch_on_connection,
    ensure_execution_schema,
    final_for,
    queue_item,
    sign,
    target_for_batch,
    verify_output,
)
from marvis.collection.ledger import CollectionEvidenceError
from marvis.collection.planning import _contact_constraint, _history_state, _subject
from marvis.db_schema import connect
from marvis.decision_twin._canonical import (
    canonical_json,
    content_hash,
    iso_z,
    parse_datetime,
)
from marvis.governance.contracts import PRODUCER_RECEIPT_SCHEMA_VERSION
from marvis.governance.repository import (
    _binding_from_row,
    _producer_binding_on_connection,
    _require_bound_invocation,
)
from marvis.plugins.invocation import load_invocation_contract
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.risk_context.source_repository import require_actor

MAX_RESOURCE_ITEMS = 50_000


def now():
    return iso_z(datetime.now(UTC))


def _at(value):
    return parse_datetime(value, "collection execution time")


class CollectionExecutor:
    def __init__(self, settings):
        self.settings = settings
        self.batches = CollectionBatchStore(settings)
        self.secret = self.batches.ledger.secret
        self.artifacts = TaskArtifactRepository(settings.db_path)
        ensure_execution_schema(settings.db_path)

    def _authorization(self, conn, task_id, inputs, operation, ctx):
        effect = conn.execute(
            "SELECT * FROM effect_executions WHERE id=?", (ctx.effect_execution_id,)
        ).fetchone()
        if effect is None or not ctx.runtime_generation:
            raise CollectionEvidenceError("collection_governed_effect_required")
        approval = conn.execute(
            "SELECT * FROM approval_records WHERE id=?", (effect["approval_id"],)
        ).fetchone()
        if (
            approval is None
            or effect["status"] != "dispatched"
            or effect["released_at"] is not None
            or approval["status"] != "reserved"
            or effect["runtime_generation"] != ctx.runtime_generation
            or _at(approval["expires_at"]) <= _at(now())
        ):
            raise CollectionEvidenceError("collection_effect_not_committable")
        binding = _binding_from_row(approval)
        if binding.task_id != task_id or not binding.tool_ref.startswith(
            "collection." + operation + "@"
        ):
            raise CollectionEvidenceError("collection_effect_tool_or_task_mismatch")
        bindings = _producer_binding_on_connection(conn, effect, approval)
        run = conn.execute(
            "SELECT * FROM plan_step_runs WHERE id=?", (bindings["invocation_id"],)
        ).fetchone()
        _require_bound_invocation(
            conn,
            bindings["invocation_id"],
            binding,
            expected_contract=load_invocation_contract(run["invocation_contract_json"]),
        )
        if json.loads(run["input_json"]) != inputs:
            raise CollectionEvidenceError("collection_worker_inputs_changed")
        approver_role = require_actor(
            conn, approval["principal_id"], {"maker", "checker", "admin"}
        )
        batch = batch_on_connection(conn, task_id, inputs["batch_id"], self.secret)
        if approver_role == "maker" and approval["principal_id"] != batch["actor_id"]:
            raise CollectionEvidenceError("collection_approver_scope_forbidden")
        statuses, result_status = OPERATIONS[operation]
        if (
            batch["status"] not in statuses
            or target_for_batch(batch, result_status) != binding.effect_target
            or inputs["request_hash"] != batch["request_hash"]
            or inputs["preview_hash"] != batch["preview_hash"]
        ):
            raise CollectionEvidenceError("collection_approved_batch_changed")
        if operation != "cancel_batch":
            require_actor(conn, batch["actor_id"], {"maker"})
        return batch, effect, approval, binding, bindings

    def _resource_items(self, conn, request):
        subjects = sorted({_subject(case) for case in request.cases})
        rows = conn.execute(
            """SELECT * FROM collection_queue_items WHERE policy_id=? OR
            (subject_namespace,subject_token) IN
            (SELECT json_extract(value,'$[0]'),json_extract(value,'$[1]') FROM json_each(?))
            ORDER BY id LIMIT ?""",
            (
                request.policy.policy_id,
                canonical_json(subjects),
                MAX_RESOURCE_ITEMS + 1,
            ),
        ).fetchall()
        if len(rows) > MAX_RESOURCE_ITEMS:
            raise CollectionEvidenceError("collection_resource_budget_exceeded")
        return [
            (item, final_for(conn, item, self.secret))
            for item in (queue_item(row, self.secret) for row in rows)
        ]

    def _limits(self, request, resources):
        policy = request.policy
        queues = {q.queue_id: q for q in policy.queues}
        if policy.max_estimated_active_cost_minor is None or any(
            q.max_active_actions is None for q in policy.queues
        ):
            raise CollectionEvidenceError(
                "collection_execution_capacity_or_budget_unknown"
            )
        at = _at(request.as_of)
        if not _at(policy.valid_from) <= at < _at(policy.valid_until):
            raise CollectionEvidenceError("collection_reference_policy_expired")
        capacity, active_cost = Counter(), 0
        subjects = {_subject(case) for case in request.cases}
        for item, final in resources:
            if (
                (item["subject_namespace"], item["subject_token"]) in subjects
                or item["policy_id"] == policy.policy_id
            ) and _at(item["reference_at"]) > at:
                raise CollectionEvidenceError("collection_reference_clock_regression")
            if final is None and item["policy_id"] == policy.policy_id:
                if item["unit"] != policy.unit.model_dump():
                    raise CollectionEvidenceError(
                        "collection_active_resource_unit_conflict"
                    )
                capacity[item["queue_id"]] += 1
                active_cost += item["estimated_cost_minor"] or 0
        return queues, capacity, active_cost

    def _native_attempts(self, request, resources, *, exclude_ids=()):
        by_id = {item["id"]: (item, final) for item, final in resources}
        mapped = set()
        for history in request.histories:
            for attempt in history.attempts:
                identity = attempt.reference_action_id
                if identity is None:
                    continue
                if identity in mapped or identity not in by_id:
                    raise CollectionEvidenceError(
                        "collection_reference_attempt_identity_conflict"
                    )
                item, final = by_id[identity]
                state = (
                    "reserved"
                    if final is None
                    else "completed"
                    if final["state"] == "completed"
                    else "cancelled_before_dispatch"
                )
                if (
                    item["kind"] != "contact"
                    or item["case_id"] != attempt.case_id
                    or (item["subject_namespace"], item["subject_token"])
                    != _subject(history)
                    or _at(item["reference_at"]) != _at(attempt.attempted_at)
                    or attempt.state != state
                ):
                    raise CollectionEvidenceError(
                        "collection_reference_attempt_source_conflict"
                    )
                mapped.add(identity)
        attempts = {}
        for item, final in resources:
            if (
                item["id"] in mapped
                or item["id"] in exclude_ids
                or item["kind"] != "contact"
                or final is not None
                and final["state"] == "cancelled"
            ):
                continue
            subject = (item["subject_namespace"], item["subject_token"])
            attempts.setdefault(subject, []).append(_at(item["reference_at"]))
        return attempts

    def _queue(self, conn, batch, request, effect_id, stamp):
        resources = self._resource_items(conn, request)
        queues, active, active_cost = self._limits(request, resources)
        attempts = self._native_attempts(request, resources)
        at, cutoff = _at(request.as_of), _at(request.knowledge_cutoff)
        histories = {
            _subject(h): _history_state(h, request.policy, at, cutoff)
            for h in request.histories
        }
        cases = {case.case_id: case for case in request.cases}
        action_ids, held, batch_used, spent = [], [], Counter(), 0
        for decision in batch["preview"]["results"]:
            if decision["status"] not in {"eligible_for_approval", "manual_review"}:
                held.append(
                    {"case_id": decision["case_id"], "reason": decision["reason"]}
                )
                continue
            case, action = cases[decision["case_id"]], decision["action"]["value"]
            subject, queue = _subject(case), queues[action["queue_id"]]
            reason = None
            if action["kind"] == "contact":
                constraint = _contact_constraint(
                    case,
                    request.policy,
                    histories.get(subject),
                    attempts.get(subject, []),
                    at,
                )
                if constraint:
                    reason = constraint[1]
            cost = action["estimated_cost_minor"] or 0
            if reason is None and (
                active[queue.queue_id] >= queue.max_active_actions
                or batch_used[queue.queue_id] >= queue.max_batch_actions
            ):
                reason = "live_queue_capacity"
            if reason is None and (
                active_cost + cost > request.policy.max_estimated_active_cost_minor
                or spent + cost > request.policy.max_estimated_batch_cost_minor
            ):
                reason = "live_estimated_cost_budget"
            if reason:
                held.append({"case_id": case.case_id, "reason": reason})
                continue
            action_id = content_hash(
                [batch["task_id"], batch["batch_id"], case.case_id]
            )
            item = {
                "id": action_id,
                "task_id": batch["task_id"],
                "batch_id": batch["batch_id"],
                "case_id": case.case_id,
                "policy_id": request.policy.policy_id,
                "policy_hash": request.policy.content_hash,
                "queue_id": queue.queue_id,
                "subject_namespace": case.subject_namespace,
                "subject_token": case.subject_token,
                "kind": action["kind"],
                "channel": action["channel"],
                "estimated_cost_minor": action["estimated_cost_minor"],
                "estimate_scope": "contact_reservation_only",
                "review_cost_estimated": False,
                "unit": request.policy.unit.model_dump(),
                "reference_at": request.as_of,
                "platform_reserved_at": stamp,
                "assigned_to": batch["actor_id"],
                "assignment_scope": "local_reference_only",
                "effect_id": effect_id,
            }
            conn.execute(
                "INSERT INTO collection_queue_items VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    action_id,
                    batch["task_id"],
                    batch["batch_id"],
                    case.case_id,
                    request.policy.policy_id,
                    queue.queue_id,
                    case.subject_namespace,
                    case.subject_token,
                    canonical_json(item),
                    sign(self.secret, item),
                ),
            )
            action_ids.append(action_id)
            active[queue.queue_id] += 1
            batch_used[queue.queue_id] += 1
            active_cost += cost
            spent += cost
            if action["kind"] == "contact":
                attempts.setdefault(subject, []).append(at)
        return action_ids, held, spent

    def _finalize_actions(self, conn, batch, request, operation, effect_id, stamp):
        rows = conn.execute(
            "SELECT * FROM collection_queue_items WHERE task_id=? AND batch_id=? ORDER BY id",
            (batch["task_id"], batch["batch_id"]),
        ).fetchall()
        if len(rows) > 10000:
            raise CollectionEvidenceError("collection_batch_action_budget_exceeded")
        items = [queue_item(row, self.secret) for row in rows]
        if any(final_for(conn, item, self.secret) is not None for item in items):
            raise CollectionEvidenceError("collection_queue_action_already_final")
        if operation == "execute_reference":
            resources = self._resource_items(conn, request)
            queues, capacity, cost = self._limits(request, resources)
            if cost > request.policy.max_estimated_active_cost_minor or any(
                capacity[q.queue_id] > q.max_active_actions
                for q in request.policy.queues
            ):
                raise CollectionEvidenceError(
                    "collection_live_resource_contract_changed"
                )
            attempts = self._native_attempts(
                request, resources, exclude_ids={item["id"] for item in items}
            )
            at, cutoff = _at(request.as_of), _at(request.knowledge_cutoff)
            histories = {
                _subject(h): _history_state(h, request.policy, at, cutoff)
                for h in request.histories
            }
            cases = {case.case_id: case for case in request.cases}
            for item in sorted(items, key=lambda item: item["case_id"]):
                if item["kind"] != "contact":
                    continue
                case = cases[item["case_id"]]
                subject = _subject(case)
                constraint = _contact_constraint(
                    case,
                    request.policy,
                    histories.get(subject),
                    attempts.get(subject, []),
                    at,
                )
                if constraint:
                    raise CollectionEvidenceError(
                        "collection_live_contact_constraint:" + constraint[1]
                    )
                attempts.setdefault(subject, []).append(at)
        action_ids = []
        for item in items:
            final = {
                "action_id": item["id"],
                "task_id": item["task_id"],
                "batch_id": item["batch_id"],
                "state": "completed"
                if operation == "execute_reference"
                else "cancelled",
                "local_result": "reference_ticket_recorded"
                if operation == "execute_reference"
                else "cancelled_before_dispatch",
                "reference_at": request.as_of,
                "platform_recorded_at": stamp,
                "effect_id": effect_id,
                "actual_cost_minor": None,
                "actual_cost_state": "not_reported",
                "external_action_executed": False,
                "customer_contacted": False,
            }
            conn.execute(
                "INSERT INTO collection_action_finals VALUES(?,?,?,?)",
                (
                    item["id"],
                    item["task_id"],
                    canonical_json(final),
                    sign(self.secret, final),
                ),
            )
            action_ids.append(item["id"])
        return action_ids, [], sum(item["estimated_cost_minor"] or 0 for item in items)

    def apply(self, task_id, inputs, operation, ctx):
        if operation not in OPERATIONS:
            raise CollectionEvidenceError("collection_operation_unsupported")
        uow = ArtifactUnitOfWork()
        try:
            with connect(self.settings.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                batch, effect, approval, binding, bindings = self._authorization(
                    conn, task_id, inputs, operation, ctx
                )
                request = CollectionBatchRequest.model_validate(batch["request"])
                if operation != "cancel_batch":
                    self.batches._sources(conn, task_id, request)
                stamp = now()
                action_ids, held, estimated = (
                    self._queue(conn, batch, request, effect["id"], stamp)
                    if operation == "queue_batch"
                    else self._finalize_actions(
                        conn, batch, request, operation, effect["id"], stamp
                    )
                )
                unestimated_reviews = conn.execute(
                    "SELECT count(*) FROM collection_queue_items WHERE task_id=? AND batch_id=? AND json_extract(payload_json,'$.kind')='review'",
                    (task_id, batch["batch_id"]),
                ).fetchone()[0]
                next_status = OPERATIONS[operation][1]
                cursor = conn.execute(
                    "UPDATE collection_batches SET status=?,revision=revision+1 WHERE task_id=? AND batch_id=? AND status=? AND revision=?",
                    (
                        next_status,
                        task_id,
                        batch["batch_id"],
                        batch["status"],
                        batch["revision"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise CollectionEvidenceError("collection_batch_commit_conflict")
                facts = {
                    "schema_version": "collection.reference_effect.v1",
                    "bindings": bindings,
                    "target": binding.effect_target,
                    "operation": operation,
                    "batch_status_at_commit": next_status,
                    "batch_revision_at_commit": batch["revision"] + 1,
                    "action_ids": sorted(action_ids),
                    "held": held,
                    "estimated_cost_minor": estimated,
                    "estimated_cost_scope": "contact_actions_only",
                    "unestimated_review_count": unestimated_reviews,
                    "estimate_scope": "contact_reservation_only",
                    "actual_cost_minor": None,
                    "actual_cost_state": "not_reported",
                    "reference_at": request.as_of,
                    "knowledge_cutoff": request.knowledge_cutoff,
                    "clock_scope": "declared_reference_simulation",
                    "platform_committed_at": stamp,
                    "source_truth_verified": False,
                    "external_action_executed": False,
                    "customer_contacted": False,
                }
                path = (
                    self.settings.tasks_dir
                    / task_id
                    / "collection"
                    / f"execution-{content_hash(bindings)}.json"
                )
                if path.parent.resolve() != path.parent.absolute():
                    raise CollectionEvidenceError("collection_execution_path_invalid")
                staged = uow.stage_file(path.parent, path.name)
                staged.path.write_text(canonical_json(facts), encoding="utf-8")
                uow.promote_all()
                artifact = self.artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind="collection_reference_execution",
                    path=str(path.relative_to(self.settings.workspace)),
                    content_hash=content_hash(facts),
                    origin_tool="collection." + operation,
                    provenance={
                        "effect_id": effect["id"],
                        "invocation_id": bindings["invocation_id"],
                        "batch_id": batch["batch_id"],
                    },
                )
                output = {
                    "schema_version": "collection.reference_result.v1",
                    "task_id": task_id,
                    "batch_id": batch["batch_id"],
                    "operation": operation,
                    "status": next_status,
                    "revision": batch["revision"] + 1,
                    "artifact_id": artifact["id"],
                    "receipt_hash": content_hash(facts),
                    "action_count": len(action_ids),
                    "held_count": len(held),
                    "estimated_cost_minor": estimated,
                    "estimated_cost_scope": "contact_actions_only",
                    "unestimated_review_count": unestimated_reviews,
                    "actual_cost_minor": None,
                    "execution_mode": "local_reference",
                    "external_action_executed": False,
                    "evidence_url": f"/api/tasks/{task_id}/collection/batches/{batch['batch_id']}/effects/{effect['id']}",
                }
                native = {"facts": facts, "output": output}
                conn.execute(
                    "INSERT INTO collection_native_effects VALUES(?,?,?,?,?,?)",
                    (
                        effect["id"],
                        bindings["invocation_id"],
                        task_id,
                        batch["batch_id"],
                        canonical_json(native),
                        sign(self.secret, native),
                    ),
                )
                verify_output(
                    conn,
                    self.settings.db_path,
                    binding=binding,
                    invocation_id=bindings["invocation_id"],
                    output=output,
                )
                receipt = {
                    "schema_version": PRODUCER_RECEIPT_SCHEMA_VERSION,
                    "receipt_id": "producer:" + effect["id"],
                    "bindings": bindings,
                    "output": output,
                    "output_hash": content_hash(output),
                }
                receipt["receipt_hash"] = content_hash(receipt)
                detail = json.loads(effect["detail_json"] or "{}")
                detail.update(
                    domain_receipt={
                        "kind": "collection." + operation,
                        "receipt_hash": content_hash(facts),
                    },
                    producer_receipt=receipt,
                )
                cursor = conn.execute(
                    "UPDATE effect_executions SET status='committed',committed_at=?,result_hash=?,detail_json=? WHERE id=? AND status='dispatched' AND released_at IS NULL AND reservation_id=? AND runtime_generation=?",
                    (
                        stamp,
                        receipt["receipt_hash"],
                        canonical_json(detail),
                        effect["id"],
                        effect["reservation_id"],
                        ctx.runtime_generation,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CollectionEvidenceError("collection_effect_commit_conflict")
                cursor = conn.execute(
                    "UPDATE approval_records SET status='consumed',consumed_at=? WHERE id=? AND status='reserved' AND reservation_id=?",
                    (stamp, approval["id"], effect["reservation_id"]),
                )
                if cursor.rowcount != 1:
                    raise CollectionEvidenceError("collection_approval_commit_conflict")
            uow.commit()
        except Exception:
            uow.rollback()
            raise
        return output
