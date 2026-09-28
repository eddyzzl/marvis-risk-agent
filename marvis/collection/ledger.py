"""Append-only case cashflows; exact minor units, reversals and explicit support.

This ledger binds imported declarations, not causal collection effectiveness or
permission to contact a person. Action authorization is owned by the workflow.
"""

from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
from pathlib import Path

from marvis.artifacts.transactional import ArtifactUnitOfWork
from marvis.collection.contracts import (
    CashflowEvent,
    CollectionCase,
    InstallmentSchedule,
    ReconciliationRequest,
)
from marvis.db_schema import connect
from marvis.decision_twin._canonical import (
    canonical_json,
    content_hash,
    iso_z,
    parse_datetime,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository


class CollectionEvidenceError(ValueError):
    pass


def _at(value):
    return parse_datetime(value, "collection timestamp")


class CollectionLedger:
    MAX_EVENTS = 10000

    def __init__(self, settings):
        self.settings = settings
        self.secret = settings.plugin_admin_token_path.read_text().strip().encode()
        if not self.secret:
            raise CollectionEvidenceError("collection_authentication_unavailable")
        with connect(settings.db_path) as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS collection_evidence (
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                kind TEXT NOT NULL CHECK(kind IN ('case','schedule','flow','reconciliation')),
                identity TEXT NOT NULL, case_id TEXT NOT NULL,
                body_json TEXT NOT NULL, body_hash TEXT NOT NULL, signature TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(task_id,kind,identity)
            );
            CREATE INDEX IF NOT EXISTS idx_collection_case ON collection_evidence(task_id,case_id,kind);
            CREATE TRIGGER IF NOT EXISTS collection_evidence_no_update
            BEFORE UPDATE ON collection_evidence BEGIN SELECT RAISE(ABORT,'collection evidence is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS collection_evidence_no_delete
            BEFORE DELETE ON collection_evidence
            WHEN EXISTS(SELECT 1 FROM tasks WHERE id=OLD.task_id)
            BEGIN SELECT RAISE(ABORT,'collection evidence is immutable'); END;
            """)

    def _signature(self, task_id, kind, identity, body_hash):
        return hmac.new(
            self.secret,
            canonical_json(
                ["collection.ledger.v1", task_id, kind, identity, body_hash]
            ).encode(),
            "sha256",
        ).hexdigest()

    def _decode(self, row):
        if row is None:
            raise CollectionEvidenceError("collection_evidence_not_found")
        body = json.loads(row["body_json"])
        if content_hash(body) != row["body_hash"] or not hmac.compare_digest(
            row["signature"],
            self._signature(
                row["task_id"], row["kind"], row["identity"], row["body_hash"]
            ),
        ):
            raise CollectionEvidenceError("collection_evidence_integrity_failed")
        if body["case_id"] != row["case_id"]:
            raise CollectionEvidenceError("collection_case_binding_drifted")
        return body

    def _get(self, conn, task_id, kind, identity):
        return self._decode(
            conn.execute(
                "SELECT * FROM collection_evidence WHERE task_id=? AND kind=? AND identity=?",
                (task_id, kind, identity),
            ).fetchone()
        )

    def _put(self, conn, task_id, kind, identity, body):
        body_hash = content_hash(body)
        previous = conn.execute(
            "SELECT * FROM collection_evidence WHERE task_id=? AND kind=? AND identity=?",
            (task_id, kind, identity),
        ).fetchone()
        if previous is not None:
            if self._decode(previous) != body:
                raise CollectionEvidenceError("collection_idempotency_conflict")
            return body
        if not conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise CollectionEvidenceError("collection_task_not_found")
        conn.execute(
            "INSERT INTO collection_evidence VALUES (?,?,?,?,?,?,?,?)",
            (
                task_id,
                kind,
                identity,
                body["case_id"],
                canonical_json(body),
                body_hash,
                self._signature(task_id, kind, identity, body_hash),
                iso_z(datetime.now(UTC)),
            ),
        )
        return body

    def _source(self, conn, task_id, artifact_id, expected_hash):
        row = conn.execute(
            "SELECT * FROM task_artifacts WHERE id=? AND task_id=?",
            (artifact_id, task_id),
        ).fetchone()
        if row is None or row["content_hash"] != expected_hash:
            raise CollectionEvidenceError("collection_source_binding_invalid")
        root = (self.settings.tasks_dir / task_id).resolve()
        raw = Path(row["path"])
        path = raw if raw.is_absolute() else self.settings.workspace / raw
        if (
            not path.resolve().is_relative_to(root)
            or path.resolve() != path.absolute()
            or not path.is_file()
        ):
            raise CollectionEvidenceError("collection_source_path_invalid")
        if (
            path.stat().st_size > 64_000_000
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise CollectionEvidenceError("collection_source_integrity_failed")

    def create_case(self, task_id, case: CollectionCase, *, writer_guard=None):
        case = CollectionCase.model_validate(case.model_dump())
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if writer_guard is not None:
                writer_guard(conn)
            self._source(
                conn, task_id, case.source_artifact_id, case.source_artifact_hash
            )
            return self._put(conn, task_id, "case", case.case_id, case.model_dump())

    def register_schedule(
        self, task_id, schedule: InstallmentSchedule, *, writer_guard=None
    ):
        schedule = InstallmentSchedule.model_validate(schedule.model_dump())
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if writer_guard is not None:
                writer_guard(conn)
            case = CollectionCase.model_validate(
                self._get(conn, task_id, "case", schedule.case_id)
            )
            self._source(
                conn, task_id, schedule.terms_artifact_id, schedule.terms_artifact_hash
            )
            if (
                schedule.unit != case.unit
                or sum(p.amount_minor for p in schedule.installments)
                != case.opening_balance_minor
            ):
                raise CollectionEvidenceError("installment_balance_or_unit_mismatch")
            if any(_at(p.due_at) < _at(case.opened_at) for p in schedule.installments):
                raise CollectionEvidenceError("installment_precedes_case")
            return self._put(
                conn, task_id, "schedule", schedule.schedule_id, schedule.model_dump()
            )

    def _flows(self, conn, task_id, case_id):
        rows = conn.execute(
            "SELECT * FROM collection_evidence WHERE task_id=? AND case_id=? AND kind='flow' ORDER BY identity LIMIT ?",
            (task_id, case_id, self.MAX_EVENTS + 1),
        ).fetchall()
        if len(rows) > self.MAX_EVENTS:
            raise CollectionEvidenceError("collection_event_budget_exceeded")
        return [CashflowEvent.model_validate(self._decode(row)) for row in rows]

    @staticmethod
    def _flow_key(event):
        return canonical_json([event.source_id, event.event_id])

    def append(self, task_id, events: list[CashflowEvent], *, writer_guard=None):
        if not 0 < len(events) <= self.MAX_EVENTS:
            raise CollectionEvidenceError("collection_event_budget_exceeded")
        pending = [CashflowEvent.model_validate(e.model_dump()) for e in events]
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if writer_guard is not None:
                writer_guard(conn)
            cases, existing = {}, {}
            checked_sources = set()
            for event in pending:
                if event.case_id not in cases:
                    cases[event.case_id] = CollectionCase.model_validate(
                        self._get(conn, task_id, "case", event.case_id)
                    )
                    existing[event.case_id] = self._flows(conn, task_id, event.case_id)
                source = (event.source_artifact_id, event.source_artifact_hash)
                if source not in checked_sources:
                    self._source(conn, task_id, *source)
                    checked_sources.add(source)
            result = []
            # Dependency order, independent of import ordering. If no original can
            # be resolved, reject the whole batch instead of publishing half of it.
            while pending:
                deferred, progressed = [], False
                for event in pending:
                    case, flows = cases[event.case_id], existing[event.case_id]
                    if event.unit != case.unit or _at(event.event_at) < _at(
                        case.opened_at
                    ):
                        raise CollectionEvidenceError(
                            "collection_event_case_or_unit_mismatch"
                        )
                    key = self._flow_key(event)
                    prior = next((e for e in flows if self._flow_key(e) == key), None)
                    if prior is not None:
                        result.append(
                            self._put(conn, task_id, "flow", key, event.model_dump())
                        )
                        progressed = True
                        continue
                    if len(flows) >= self.MAX_EVENTS:
                        raise CollectionEvidenceError(
                            "collection_event_budget_exceeded"
                        )
                    if event.reverses:
                        parent = next(
                            (
                                e
                                for e in flows
                                if (e.source_id, e.event_id)
                                == (event.reverses.source_id, event.reverses.event_id)
                            ),
                            None,
                        )
                        if parent is None:
                            deferred.append(event)
                            continue
                        if parent.kind != event.kind.removesuffix("_reversal") or _at(
                            event.event_at
                        ) < _at(parent.event_at):
                            raise CollectionEvidenceError(
                                "collection_reversal_reference_invalid"
                            )
                        reversed_total = sum(
                            e.amount_minor
                            for e in flows
                            if e.reverses == event.reverses
                        )
                        if reversed_total + event.amount_minor > parent.amount_minor:
                            raise CollectionEvidenceError(
                                "collection_reversal_exceeds_original"
                            )
                    if event.installment:
                        schedule = InstallmentSchedule.model_validate(
                            self._get(
                                conn, task_id, "schedule", event.installment.schedule_id
                            )
                        )
                        if (
                            schedule.case_id != event.case_id
                            or event.installment.installment_id
                            not in {i.installment_id for i in schedule.installments}
                        ):
                            raise CollectionEvidenceError(
                                "collection_installment_binding_invalid"
                            )
                    result.append(
                        self._put(conn, task_id, "flow", key, event.model_dump())
                    )
                    flows.append(event)
                    progressed = True
                if not progressed:
                    raise CollectionEvidenceError(
                        "collection_reversal_original_missing"
                    )
                pending = deferred
            return result

    def reconcile(self, task_id, request: ReconciliationRequest, *, writer_guard=None):
        request = ReconciliationRequest.model_validate(request.model_dump())
        as_of, cutoff = _at(request.as_of), _at(request.knowledge_cutoff)
        if cutoff > datetime.now(UTC):
            raise CollectionEvidenceError("collection_future_knowledge_cutoff")
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if writer_guard is not None:
                writer_guard(conn)
            case = CollectionCase.model_validate(
                self._get(conn, task_id, "case", request.case_id)
            )
            if as_of < _at(case.opened_at):
                raise CollectionEvidenceError("collection_report_precedes_case")
            flows = self._flows(conn, task_id, request.case_id)
            reasons = []
            coverage = {c.source_id: c for c in request.coverage}
            for source in request.expected_sources:
                declared = coverage.get(source)
                if declared is None:
                    reasons.append("missing_source_coverage:" + source)
                    continue
                self._source(
                    conn,
                    task_id,
                    declared.source_artifact_id,
                    declared.source_artifact_hash,
                )
                if (
                    declared.status != "complete"
                    or _at(declared.from_at) > _at(case.opened_at)
                    or _at(declared.through_at) < as_of
                    or _at(declared.available_at) > cutoff
                ):
                    reasons.append("incomplete_source_coverage:" + source)
            visible = []
            for event in flows:
                if _at(event.event_at) > as_of:
                    continue
                if event.source_id not in request.expected_sources:
                    reasons.append("undeclared_event_source:" + event.source_id)
                if event.available_at is None:
                    reasons.append("cashflow_availability_unknown")
                    continue
                if _at(event.available_at) <= cutoff:
                    visible.append(event)
            known = {self._flow_key(e): e for e in visible}
            for event in visible:
                if (
                    event.reverses
                    and canonical_json(
                        [event.reverses.source_id, event.reverses.event_id]
                    )
                    not in known
                ):
                    reasons.append("reversal_original_unavailable_at_cutoff")
            totals = {
                kind: sum(e.amount_minor for e in visible if e.kind == kind)
                for kind in ("payment", "payment_reversal", "cost", "cost_reversal")
            }
            net_payments = totals["payment"] - totals["payment_reversal"]
            net_cost = totals["cost"] - totals["cost_reversal"]
            amounts = {
                **{k + "_minor": v for k, v in totals.items()},
                "net_payments_minor": net_payments,
                "net_cost_minor": net_cost,
                "net_recovery_minor": net_payments - net_cost,
                "remaining_balance_minor": max(
                    case.opening_balance_minor - net_payments, 0
                ),
                "overpayment_minor": max(net_payments - case.opening_balance_minor, 0),
            }
            installments = []
            if request.schedule_id:
                schedule = InstallmentSchedule.model_validate(
                    self._get(conn, task_id, "schedule", request.schedule_id)
                )
                if schedule.case_id != request.case_id:
                    raise CollectionEvidenceError(
                        "collection_installment_binding_invalid"
                    )
                for part in schedule.installments:
                    payments = [
                        e
                        for e in visible
                        if e.installment
                        and e.installment.schedule_id == request.schedule_id
                        and e.installment.installment_id == part.installment_id
                    ]
                    keys = {(e.source_id, e.event_id) for e in payments}
                    paid = sum(e.amount_minor for e in payments) - sum(
                        e.amount_minor
                        for e in visible
                        if e.reverses
                        and (e.reverses.source_id, e.reverses.event_id) in keys
                    )
                    installments.append(
                        {
                            **part.model_dump(),
                            "observed_paid_minor": paid,
                            "status": "unknown"
                            if reasons
                            else "paid"
                            if paid >= part.amount_minor
                            else "overdue"
                            if _at(part.due_at) <= as_of
                            else "not_due",
                            "remaining_minor": None
                            if reasons
                            else max(part.amount_minor - paid, 0),
                        }
                    )
            maturity = {
                "status": "not_declared",
                "mature_at": None,
                "mature_outcome_verified": False,
            }
            if request.maturity is not None:
                self._source(
                    conn,
                    task_id,
                    request.maturity.policy_artifact_id,
                    request.maturity.policy_artifact_hash,
                )
                mature_at = _at(case.opened_at) + timedelta(
                    days=request.maturity.observation_days
                )
                maturity = {
                    "status": "insufficient_evidence"
                    if reasons
                    else "mature_under_declared_policy"
                    if as_of >= mature_at
                    else "immature",
                    "mature_at": iso_z(mature_at),
                    "policy": request.maturity.model_dump(),
                    "mature_outcome_verified": not reasons and as_of >= mature_at,
                    "scope": "declared_observation_window_not_independent_source_truth",
                }
            body = {
                "schema_version": "collection.reconciliation.v1",
                "case_id": request.case_id,
                "contract": request.model_dump(),
                "case_hash": case.content_hash,
                "members_hash": content_hash([e.content_hash for e in visible]),
                "status": "insufficient_evidence"
                if reasons
                else "reconciled_under_declared_coverage",
                "reasons": sorted(set(reasons)),
                "unit": case.unit.model_dump(),
                "amounts": None if reasons else amounts,
                "observed_subtotals": amounts,
                "visible_event_count": len(visible),
                "stored_event_count": len(flows),
                "installments": installments,
                "unallocated_payment_count": sum(
                    e.kind == "payment" and e.installment is None for e in visible
                ),
                "interpretation": "retrospective_source_declared_cashflows",
                "allocation_scope": "case_total_balance_without_principal_interest_split",
                "maturity": maturity,
                "source_truth_verified": False,
                "incremental_recovery_identified": False,
                "external_action_executed": False,
            }
            # New late evidence yields a new receipt; previous reports stay immutable.
            identity = content_hash(body)
            self._put(conn, task_id, "reconciliation", identity, body)
            artifacts = TaskArtifactRepository(self.settings.db_path)
            relative = self._report_path(task_id, identity)
            prior = conn.execute(
                "SELECT id FROM task_artifacts WHERE task_id=? AND kind='collection_cashflow_reconciliation' AND path=?",
                (task_id, relative),
            ).fetchone()
            if prior is not None:
                self._source(conn, task_id, prior["id"], identity)
                return {"receipt_hash": identity, **body}
            unit = ArtifactUnitOfWork()
            try:
                staged = unit.stage_file(
                    self.settings.tasks_dir / task_id / "collection",
                    f"reconciliation-{identity}.json",
                )
                staged.path.write_text(canonical_json(body), encoding="utf-8")
                unit.promote_all()
                artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind="collection_cashflow_reconciliation",
                    path=relative,
                    content_hash=identity,
                    origin_tool="collection.reconcile.v1",
                    provenance={
                        "contract_hash": request.content_hash,
                        "case_hash": case.content_hash,
                        "receipt_hash": identity,
                        "source_truth_verified": False,
                    },
                )
                conn.commit()
                unit.commit()
            finally:
                unit.rollback()
            return {"receipt_hash": identity, **body}

    def read_report(self, task_id, receipt_hash):
        with connect(self.settings.db_path) as conn:
            body = self._get(conn, task_id, "reconciliation", receipt_hash)
            row = conn.execute(
                "SELECT id FROM task_artifacts WHERE task_id=? AND kind='collection_cashflow_reconciliation' AND path=?",
                (task_id, self._report_path(task_id, receipt_hash)),
            ).fetchone()
            if row is None:
                raise CollectionEvidenceError("collection_report_artifact_missing")
            self._source(conn, task_id, row["id"], receipt_hash)
            return {"receipt_hash": receipt_hash, **body}

    def _report_path(self, task_id, receipt_hash):
        return str(
            (
                self.settings.tasks_dir
                / task_id
                / "collection"
                / f"reconciliation-{receipt_hash}.json"
            ).relative_to(self.settings.workspace)
        )
