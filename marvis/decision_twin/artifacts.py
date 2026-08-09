from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Iterator, Mapping
import uuid

from marvis.decision_twin._canonical import (
    canonical_json,
    content_hash,
    iso_z,
    required_text,
    utc_datetime,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_KIND_RE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_GENESIS_HASH = "0" * 64


class TamperEvidenceError(RuntimeError):
    """Stored bytes no longer verify against the content/audit hashes."""


class IdempotencyConflict(RuntimeError):
    """One idempotency key was reused with different content or kind."""


class AuditStoreBusy(RuntimeError):
    """Another process currently owns the append lock."""


@dataclass(frozen=True)
class ArtifactReceipt:
    kind: str
    idempotency_key: str
    artifact_hash: str
    artifact_uri: str
    sequence: int
    previous_event_hash: str
    event_hash: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "decision_twin.artifact_receipt.v1",
            "kind": self.kind,
            "idempotency_key": self.idempotency_key,
            "artifact_hash": self.artifact_hash,
            "artifact_uri": self.artifact_uri,
            "sequence": self.sequence,
            "previous_event_hash": self.previous_event_hash,
            "event_hash": self.event_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactReceipt:
        if payload.get("schema_version") != "decision_twin.artifact_receipt.v1":
            raise TamperEvidenceError("artifact receipt schema drifted")
        try:
            return cls(
                kind=str(payload["kind"]),
                idempotency_key=str(payload["idempotency_key"]),
                artifact_hash=str(payload["artifact_hash"]),
                artifact_uri=str(payload["artifact_uri"]),
                sequence=int(payload["sequence"]),
                previous_event_hash=str(payload["previous_event_hash"]),
                event_hash=str(payload["event_hash"]),
                created_at=str(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TamperEvidenceError("artifact receipt is malformed") from exc


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    event_count: int
    artifact_count: int
    chain_head: str


class ContentAddressedAuditStore:
    """Append-only local artifacts with idempotency and a SHA-256 event chain.

    The chain is tamper-evident, not a substitute for an externally anchored or
    access-controlled audit service. A stale process lock fails closed and must
    be reviewed rather than silently stolen.
    """

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = Path(root)
        self._artifacts = self.root / "artifacts" / "sha256"
        self._events = self.root / "audit_events"
        self._idempotency = self.root / "idempotency"
        self._head = self.root / "HEAD.json"
        self._process_lock = self.root / ".append.lock"
        self._thread_lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._artifacts.mkdir(parents=True, exist_ok=True)
        self._events.mkdir(parents=True, exist_ok=True)
        self._idempotency.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> ArtifactReceipt:
        normalized_kind = required_text(kind, "kind", max_length=80)
        if not _KIND_RE.fullmatch(normalized_kind):
            raise ValueError("kind must be a lower_snake_case identifier")
        normalized_key = required_text(
            idempotency_key, "idempotency_key", max_length=500
        )
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a JSON object")
        try:
            artifact_json = canonical_json(payload)
            normalized_payload = json.loads(artifact_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("payload must be finite canonical JSON") from exc
        if not isinstance(normalized_payload, dict):
            raise ValueError("payload must be a JSON object")
        artifact_bytes = artifact_json.encode("utf-8")
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        with self._exclusive_append():
            verification = self._verify_unlocked()
            receipt_path = self._receipt_path(normalized_key)
            if receipt_path.exists():
                receipt = ArtifactReceipt.from_dict(
                    self._read_json(receipt_path, context="audit receipt")
                )
                if (
                    receipt.kind != normalized_kind
                    or receipt.artifact_hash != artifact_hash
                ):
                    raise IdempotencyConflict(
                        "idempotency key was already used for different content or kind"
                    )
                return receipt
            artifact_path = self.artifact_path(artifact_hash)
            if artifact_path.exists():
                if artifact_path.read_bytes() != artifact_bytes:
                    raise TamperEvidenceError("artifact hash drifted")
            else:
                self._write_once(artifact_path, artifact_bytes)
            sequence = verification.event_count + 1
            created_at = iso_z(utc_datetime(self._clock(), "clock"))
            previous_hash = verification.chain_head
            artifact_uri = f"sha256://{artifact_hash}"
            event_without_hash = {
                "schema_version": "decision_twin.audit_event.v1",
                "sequence": sequence,
                "created_at": created_at,
                "kind": normalized_kind,
                "idempotency_key": normalized_key,
                "artifact_hash": artifact_hash,
                "artifact_uri": artifact_uri,
                "previous_event_hash": previous_hash,
            }
            event_hash = content_hash(event_without_hash)
            event = {**event_without_hash, "event_hash": event_hash}
            event_path = self._events / self._event_filename(sequence, event_hash)
            receipt = ArtifactReceipt(
                kind=normalized_kind,
                idempotency_key=normalized_key,
                artifact_hash=artifact_hash,
                artifact_uri=artifact_uri,
                sequence=sequence,
                previous_event_hash=previous_hash,
                event_hash=event_hash,
                created_at=created_at,
            )
            self._write_once(event_path, canonical_json(event).encode("utf-8"))
            self._write_once(
                receipt_path,
                canonical_json(receipt.to_dict()).encode("utf-8"),
            )
            self._replace_file(
                self._head,
                canonical_json({"sequence": sequence, "event_hash": event_hash}).encode(
                    "utf-8"
                ),
            )
            return receipt

    def get(self, artifact_hash: str) -> dict[str, Any]:
        with self._exclusive_append():
            self._verify_unlocked()
            path = self.artifact_path(artifact_hash)
            if not path.exists():
                raise KeyError(f"artifact not found: {artifact_hash}")
            payload = self._read_json(path, context="artifact")
            if not isinstance(payload, dict):
                raise TamperEvidenceError("artifact payload is not an object")
            return payload

    def get_typed(
        self,
        artifact_hash: str,
        *,
        expected_kind: str,
    ) -> dict[str, Any]:
        """Load one audit-referenced artifact with an exact immutable kind.

        ``get`` remains the compatibility read API. Production consumers should
        use this method so an orphan file, or content registered under a
        different semantic kind, cannot become executable input merely because
        its bytes happen to exist in the content-addressed directory.
        """

        normalized_kind = required_text(
            expected_kind,
            "expected_kind",
            max_length=80,
        )
        if not _KIND_RE.fullmatch(normalized_kind):
            raise ValueError("expected_kind must be a lower_snake_case identifier")
        with self._exclusive_append():
            self._verify_unlocked()
            path = self.artifact_path(artifact_hash)
            if not path.exists():
                raise KeyError(f"artifact not found: {artifact_hash}")
            registered_kinds: set[str] = set()
            for event_path in sorted(self._events.glob("*.json")):
                event = self._read_json(event_path, context="audit event")
                if event.get("artifact_hash") == artifact_hash:
                    kind = event.get("kind")
                    if isinstance(kind, str):
                        registered_kinds.add(kind)
            if normalized_kind not in registered_kinds:
                if registered_kinds:
                    actual = ", ".join(sorted(registered_kinds))
                    raise TamperEvidenceError(
                        "artifact kind mismatch: "
                        f"expected {normalized_kind}, registered as {actual}"
                    )
                raise TamperEvidenceError(
                    "artifact exists without an audit-referenced receipt"
                )
            return self._read_json(path, context="artifact")

    def verify(self) -> AuditVerification:
        with self._exclusive_append():
            return self._verify_unlocked()

    def artifact_path(self, artifact_hash: str) -> Path:
        if not isinstance(artifact_hash, str) or not _SHA256_RE.fullmatch(
            artifact_hash
        ):
            raise ValueError("artifact_hash must be a lowercase SHA-256 hex digest")
        return self._artifacts / f"{artifact_hash}.json"

    def audit_event_path(self, sequence: int) -> Path:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("sequence must be a positive integer")
        matches = tuple(self._events.glob(f"{sequence:020d}-*.json"))
        if not matches:
            raise FileNotFoundError(f"audit event not found: {sequence}")
        if len(matches) != 1:
            raise TamperEvidenceError("duplicate audit event sequence")
        return matches[0]

    def _verify_unlocked(self) -> AuditVerification:
        previous_hash = _GENESIS_HASH
        referenced_artifacts: set[str] = set()
        seen_idempotency_keys: set[str] = set()
        event_paths = sorted(self._events.glob("*.json"))
        for expected_sequence, path in enumerate(event_paths, start=1):
            event = self._read_json(path, context="audit event")
            if event.get("schema_version") != "decision_twin.audit_event.v1":
                raise TamperEvidenceError("audit event schema drifted")
            event_hash = event.get("event_hash")
            if not isinstance(event_hash, str) or not _SHA256_RE.fullmatch(event_hash):
                raise TamperEvidenceError("audit event hash is malformed")
            try:
                sequence = int(event["sequence"])
                predecessor = str(event["previous_event_hash"])
                kind = str(event["kind"])
                idempotency_key = str(event["idempotency_key"])
                artifact_hash = str(event["artifact_hash"])
                artifact_uri = str(event["artifact_uri"])
                created_at = str(event["created_at"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TamperEvidenceError("audit event is malformed") from exc
            if sequence != expected_sequence:
                raise TamperEvidenceError("audit sequence is not contiguous")
            if predecessor != previous_hash:
                raise TamperEvidenceError("audit predecessor hash drifted")
            expected_filename = self._event_filename(sequence, event_hash)
            if path.name != expected_filename:
                raise TamperEvidenceError("audit event filename hash drifted")
            event_without_hash = {
                key: value for key, value in event.items() if key != "event_hash"
            }
            if content_hash(event_without_hash) != event_hash:
                raise TamperEvidenceError("audit event hash drifted")
            if not _KIND_RE.fullmatch(kind):
                raise TamperEvidenceError("audit event kind is malformed")
            if idempotency_key in seen_idempotency_keys:
                raise TamperEvidenceError("audit idempotency key was appended twice")
            seen_idempotency_keys.add(idempotency_key)
            if not _SHA256_RE.fullmatch(artifact_hash):
                raise TamperEvidenceError("audit artifact hash is malformed")
            if artifact_uri != f"sha256://{artifact_hash}":
                raise TamperEvidenceError("audit artifact URI drifted")
            artifact_path = self.artifact_path(artifact_hash)
            if not artifact_path.exists():
                raise TamperEvidenceError("audit references a missing artifact")
            artifact_bytes = artifact_path.read_bytes()
            if hashlib.sha256(artifact_bytes).hexdigest() != artifact_hash:
                raise TamperEvidenceError("artifact hash drifted")
            try:
                artifact_payload = json.loads(artifact_bytes)
                canonical_artifact = canonical_json(artifact_payload).encode("utf-8")
            except (
                TypeError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise TamperEvidenceError("artifact JSON is malformed") from exc
            if canonical_artifact != artifact_bytes:
                raise TamperEvidenceError("artifact canonical bytes drifted")
            receipt_path = self._receipt_path(idempotency_key)
            if not receipt_path.exists():
                raise TamperEvidenceError("audit receipt is missing")
            receipt = ArtifactReceipt.from_dict(
                self._read_json(receipt_path, context="audit receipt")
            )
            expected_receipt = ArtifactReceipt(
                kind=kind,
                idempotency_key=idempotency_key,
                artifact_hash=artifact_hash,
                artifact_uri=artifact_uri,
                sequence=sequence,
                previous_event_hash=predecessor,
                event_hash=event_hash,
                created_at=created_at,
            )
            if receipt != expected_receipt:
                raise TamperEvidenceError("audit receipt binding drifted")
            referenced_artifacts.add(artifact_hash)
            previous_hash = event_hash
        receipt_paths = tuple(self._idempotency.glob("*.json"))
        if len(receipt_paths) != len(seen_idempotency_keys):
            raise TamperEvidenceError("orphan or duplicate audit receipt detected")
        if event_paths:
            if not self._head.exists():
                raise TamperEvidenceError("audit chain head is missing")
            head = self._read_json(self._head, context="audit chain head")
            if head != {"sequence": len(event_paths), "event_hash": previous_hash}:
                raise TamperEvidenceError("audit chain head drifted")
        elif self._head.exists():
            raise TamperEvidenceError("audit chain head exists without events")
        return AuditVerification(
            valid=True,
            event_count=len(event_paths),
            artifact_count=len(referenced_artifacts),
            chain_head=previous_hash,
        )

    def _receipt_path(self, idempotency_key: str) -> Path:
        token = content_hash({"idempotency_key": idempotency_key})
        return self._idempotency / f"{token}.json"

    @staticmethod
    def _event_filename(sequence: int, event_hash: str) -> str:
        return f"{sequence:020d}-{event_hash}.json"

    @staticmethod
    def _read_json(
        path: Path,
        *,
        context: str = "stored JSON",
    ) -> dict[str, Any]:
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            canonical = canonical_json(value).encode("utf-8")
        except (
            OSError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise TamperEvidenceError(f"{context} is unreadable: {path.name}") from exc
        if not isinstance(value, dict):
            raise TamperEvidenceError(f"{context} is not an object: {path.name}")
        if raw != canonical:
            raise TamperEvidenceError(f"{context} canonical bytes drifted: {path.name}")
        return value

    @staticmethod
    def _write_once(path: Path, data: bytes) -> None:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise TamperEvidenceError(
                f"immutable file already exists: {path.name}"
            ) from exc
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            raise

    @classmethod
    def _replace_file(cls, path: Path, data: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        cls._write_once(temporary, data)
        os.replace(temporary, path)

    @contextmanager
    def _exclusive_append(self) -> Iterator[None]:
        with self._thread_lock:
            try:
                self._process_lock.mkdir()
            except FileExistsError as exc:
                raise AuditStoreBusy(
                    "decision-twin audit append lock already exists; review before recovery"
                ) from exc
            try:
                yield
            finally:
                try:
                    self._process_lock.rmdir()
                except FileNotFoundError:
                    pass
