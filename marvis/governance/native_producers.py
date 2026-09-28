"""Atomic native Tool receipts in the existing immutable artifact registry.

These records bind one host-issued invocation to the exact output committed by
its producer. Reading a record never runs a Tool, performs a source query, or
reconstructs an output from today's domain state.
"""

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path

from marvis.decision_twin._canonical import canonical_json
from marvis.files import sha256_file
from marvis.governance.outcome_verifiers import OutcomeProof
from marvis.orchestrator.evidence import payload_hash
from marvis.plugins.invocation import valid_invocation_contract
from marvis.repositories.task_artifacts import TaskArtifactRepository

KIND = "native_tool_producer_receipt"
ORIGIN = "platform.native_tool_producer.v1"
SCHEMA = "native_tool_producer.v1"
PRODUCERS = {
    "risk_context.replay_events": (
        "risk_event_features",
        "risk_context.replay_events.v1",
    ),
    "decision_twin.replay_history": (
        "decision_twin_batch_replay",
        "decision_twin.historical_replay.v2",
    ),
    "decision_twin.reconcile_history": (
        "decision_twin_batch_reconciliation",
        "decision_twin.historical_replay.v2",
    ),
}
MAX_RECEIPT_BYTES = 16 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _sign(settings, body):
    secret = settings.plugin_admin_token_path.read_text().strip().encode()
    if not secret:
        raise ValueError("native_producer_secret_unavailable")
    return hmac.new(
        secret, (SCHEMA + ":" + canonical_json(body)).encode(), hashlib.sha256
    ).hexdigest()


def _path(settings, task_id, run_id):
    return (
        settings.workspace
        / "native_tool_receipts"
        / _digest(task_id)
        / (_digest(run_id) + ".json")
    )


def _safe_path(settings, value):
    path = Path(value)
    root = settings.workspace.resolve()
    if not path.is_absolute():
        path = settings.workspace / path
    relative = path.absolute().relative_to(root)
    current = root
    for part in relative.parts:
        if part in {".", ".."}:
            raise ValueError("native_producer_path_invalid")
        current = current / part
        if current.is_symlink():
            raise ValueError("native_producer_path_invalid")
    return current


def _receipt_bytes(path):
    with path.open("rb") as handle:
        raw = handle.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("native_producer_receipt_budget_exceeded")
    return raw


def _authenticated_body(settings, record, path):
    raw = _receipt_bytes(_safe_path(settings, path))
    envelope = json.loads(raw)
    body = envelope["body"]
    if (
        hashlib.sha256(raw).hexdigest() != record["content_hash"]
        or body["schema_version"] != SCHEMA
        or not hmac.compare_digest(envelope["signature"], _sign(settings, body))
    ):
        raise ValueError("native_producer_receipt_authentication_failed")
    return body


def _artifact_binding(record):
    provenance = record.get("provenance")
    if provenance is None:
        provenance = json.loads(record["provenance_json"])
    return {
        **{
            k: record[k]
            for k in ("id", "task_id", "kind", "path", "content_hash", "origin_tool")
        },
        "provenance_hash": _digest(provenance),
    }


def _run_binding(conn, run_id):
    row = conn.execute(
        """SELECT r.*, p.task_id, p.status AS plan_status, s.status AS step_status
        FROM plan_step_runs r JOIN plans p ON p.id=r.plan_id
        JOIN plan_steps s ON s.id=r.step_id AND s.plan_id=p.id WHERE r.id=?""",
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError("native_producer_invocation_missing")
    contract = json.loads(row["invocation_contract_json"] or "null")
    if (
        not valid_invocation_contract(contract)
        or not row["dispatch_started_at"]
        or contract["tool_ref"].split("@", 1)[0] != row["tool_ref"].split("@", 1)[0]
    ):
        raise ValueError("native_producer_invocation_invalid")
    return row, {
        "invocation_id": row["id"],
        "plan_id": row["plan_id"],
        "step_id": row["step_id"],
        "task_id": row["task_id"],
        "tool_ref": row["tool_ref"],
        "invocation_contract": contract,
        "input_hash": payload_hash(json.loads(row["input_json"])),
        "dispatch_started_at": row["dispatch_started_at"],
    }


@dataclass(frozen=True)
class NativeInvocation:
    settings: object
    task_id: str
    run_id: str
    tool_ref: str
    input_hash: str

    @classmethod
    def from_context(cls, ctx, tool_ref, inputs):
        from marvis.settings import build_settings

        # Direct library/legacy Tool calls remain usable but cannot establish
        # proof about an invocation that the host never froze.
        if not getattr(ctx, "invocation_id", None):
            return None
        return cls(
            build_settings(ctx.workspace),
            ctx.task_id,
            ctx.invocation_id,
            tool_ref,
            payload_hash(inputs),
        )

    def record(self, conn, uow, *, artifact, output, authority=None):
        """Join the producer's writer transaction and its file unit of work."""
        row, binding = _run_binding(conn, self.run_id)
        if (
            binding["task_id"] != self.task_id
            or binding["tool_ref"].split("@", 1)[0] != self.tool_ref
            or binding["input_hash"] != self.input_hash
            or row["status"] != "running"
            or row["plan_status"] != "running"
            or row["step_status"] != "running"
            or (artifact["kind"], artifact["origin_tool"]) != PRODUCERS[self.tool_ref]
            or artifact["task_id"] != self.task_id
        ):
            raise ValueError("native_producer_binding_failed")
        body = {
            "schema_version": SCHEMA,
            "binding": binding,
            "artifact": _artifact_binding(artifact),
            "output": output,
            "output_hash": payload_hash(output),
            "authority": authority,
        }
        envelope = {"body": body, "signature": _sign(self.settings, body)}
        raw = canonical_json(envelope).encode()
        if len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError("native_producer_receipt_budget_exceeded")
        path = _safe_path(
            self.settings, _path(self.settings, self.task_id, self.run_id)
        )
        if path.exists():
            if _receipt_bytes(path) != raw:
                raise ValueError("native_producer_receipt_conflict")
        else:
            stage = uow.stage_file(path.parent, path.name)
            stage.path.write_bytes(raw)
            stage.path.chmod(0o600)
            uow.promote_all()
        return TaskArtifactRepository(self.settings.db_path).register_on_connection(
            conn,
            task_id=self.task_id,
            kind=KIND,
            path=str(path),
            content_hash=hashlib.sha256(raw).hexdigest(),
            origin_tool=ORIGIN,
            provenance={"invocation_id": self.run_id, "binding_hash": _digest(binding)},
        )


def verify_native_outcome(settings, event_repository, target, conn):
    """Only SELECT and authenticated file reads on the caller's snapshot."""

    def unknown(reason):
        return OutcomeProof("unknown", target.id, reason=reason)

    try:
        requested = target.binding
        _, binding = _run_binding(conn, requested["run_id"])
        if (
            target.producer not in PRODUCERS
            or binding["tool_ref"] != target.producer
            or any(
                binding[k] != requested[k]
                for k in (
                    "task_id",
                    "plan_id",
                    "step_id",
                    "input_hash",
                    "invocation_contract",
                    "dispatch_started_at",
                )
            )
        ):
            return unknown("原始调用绑定不匹配，不能核对成功。")
        path = _safe_path(
            settings, _path(settings, binding["task_id"], binding["invocation_id"])
        )
        row = conn.execute(
            "SELECT * FROM task_artifacts WHERE task_id=? AND kind=? AND origin_tool=? AND path=?",
            (binding["task_id"], KIND, ORIGIN, str(path)),
        ).fetchone()
        if row is None:
            return unknown("没有绑定本次调用的原子生产者回执；缺失不能证明未发生。")
        body = _authenticated_body(settings, row, path)
        if (
            body["binding"] != binding
            or not isinstance(body["output"], dict)
            or payload_hash(body["output"]) != body["output_hash"]
            or json.loads(row["provenance_json"])
            != {
                "invocation_id": binding["invocation_id"],
                "binding_hash": _digest(binding),
            }
        ):
            raise ValueError("receipt authentication failed")
        artifact = body["artifact"]
        original = conn.execute(
            "SELECT * FROM task_artifacts WHERE id=? AND task_id=?",
            (artifact["id"], binding["task_id"]),
        ).fetchone()
        if (
            original is None
            or _artifact_binding(dict(original)) != artifact
            or (artifact["kind"], artifact["origin_tool"]) != PRODUCERS[target.producer]
            or sha256_file(_safe_path(settings, artifact["path"]))
            != artifact["content_hash"]
        ):
            raise ValueError("original artifact changed")
        authority = body["authority"]
        if authority:
            for scope in authority["scopes"]:
                source = event_repository._access(
                    conn,
                    scope["task_id"],
                    scope["source_id"],
                    scope["grant_id"],
                    authority["actor_id"],
                    "read",
                )
                if source.contract_hash != scope["source_contract_hash"]:
                    raise ValueError("source scope changed")
        return OutcomeProof(
            "applied",
            target.id,
            row["id"],
            row["content_hash"],
            "原生生产者已在同一事务保存本次调用与完整输出；仅恢复完成流程。",
            canonical_json(body["output"]),
        )
    except (ValueError, KeyError, TypeError, OSError):
        return unknown("原生回执、产物完整性或当前来源授权未通过核对。")


def authorize_native_read(settings, event_repository, run_id, conn, actor_id):
    row = conn.execute(
        "SELECT p.task_id FROM plan_step_runs r JOIN plans p ON p.id=r.plan_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        return
    path = _path(settings, row["task_id"], run_id)
    record = conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id=? AND kind=? AND origin_tool=? AND path=?",
        (row["task_id"], KIND, ORIGIN, str(path)),
    ).fetchone()
    if record is None:
        # No receipt is not success, but neither is there a source result to
        # disclose from this producer. The verifier will retain unknown.
        return
    try:
        body = _authenticated_body(settings, record, path)
        if (
            body["binding"]["invocation_id"] != run_id
            or body["binding"]["task_id"] != row["task_id"]
        ):
            raise ValueError("receipt authentication failed")
        authority = body["authority"]
        if authority:
            if not actor_id:
                raise ValueError("current reader required")
            for scope in authority["scopes"]:
                source = event_repository._access(
                    conn,
                    scope["task_id"],
                    scope["source_id"],
                    scope["grant_id"],
                    actor_id,
                    "read",
                )
                if source.contract_hash != scope["source_contract_hash"]:
                    raise ValueError("source scope changed")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise PermissionError(
            "当前用户的来源读取授权或原生凭据未通过校验，不能恢复敏感结果。"
        ) from exc


def register_native_outcome_verifiers(registry, settings, event_repository):
    def verify(target, conn):
        return verify_native_outcome(settings, event_repository, target, conn)

    def read_guard(run_id, conn, actor_id):
        authorize_native_read(settings, event_repository, run_id, conn, actor_id)

    for producer in PRODUCERS:
        registry.register("tool", producer, ORIGIN, verify, read_guard=read_guard)
