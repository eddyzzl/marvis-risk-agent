"""Bounded private extension archive plumbing, without credential restoration."""
from contextlib import ExitStack, closing
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime_archive_domain import equal
from .runtime_archive_reader import _Unsupported, _authenticated_snapshot, _file, _json, _original_path, _public_source_identity
from .runtime_contracts import RuntimeCase, digest
from .runtime_custody import verify_case_archive
from .runtime_extensions_contracts import ExtensionBusinessCase

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")]


class FrozenExtensionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    case: RuntimeCase | ExtensionBusinessCase
    run_id: Identity
    task_id: Identity
    cases_sha256: Hash
    expected_sha256: Hash
    source: dict
    model_connection_sha256: Hash

    @model_validator(mode="after")
    def source_identity(self):
        clean = _public_source_identity(self.source)
        if clean.get("source_sha256") != self.source.get("source_sha256") or not clean.get("source_sha256"):
            raise ValueError("externally frozen source identity required")
        return self


def declared_requests(case, kind):
    """Common RuntimeCase and the dedicated DTO share the same typed input."""
    result = []
    for action in case.actions:
        item = getattr(action, "extension_request", None) or action
        if getattr(item, "kind", None) == kind:
            result.append(item)
    return result


def measured_values(actual, expected):
    """Preserve exact integer money/counts and never equate bool with one."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) - set(actual):
            raise ValueError("independent measured object differs")
        for key, value in expected.items():
            measured_values(actual[key], value)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("independent measured members differ")
        for left, right in zip(actual, expected, strict=True):
            measured_values(left, right)
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError("independent exact integer differs")
    elif type(expected) is float:
        if type(actual) not in {int, float} or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11):
            raise ValueError("independent measured number differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("independent measured literal differs")


class ExtensionArchive:
    def __init__(self, archive, sha, binding, family, checks):
        self.original = Path(archive).absolute()
        self.archive = self.original
        self.sha, self.binding = sha, binding
        self.stack = ExitStack()
        self.current = "archive_integrity"
        self.result = {
            "schema": f"marvis.runtime-{family}-revalidation.v1",
            "case_id": binding.case.id,
            "algorithm": f"independent_{family}_reference.v1",
            "source_authentication": "not_established",
            "acceptance_claim": "not_established",
            "original_source_binding": _public_source_identity(binding.source),
            "original_source_binding_sha256": digest(binding.source),
            "recomputation_isolation": "private_authenticated_copy; excludes_adversarial_same_uid_or_root_access",
            "checks": {name: {"status": "unverified"} for name in ("archive_integrity", "recomputation_snapshot", "external_bindings", "native_records", *checks)},
            "unsupported": {
                "native_signature_authentication": "original HMAC authority is excluded from archives; caller-pinned bytes are consistency evidence only",
                "external_source_truth_and_institutional_authorization": "requires independently held source and institution authority",
                "original_runtime_cost_timing_and_concurrency": "requires separate runtime adjudication",
            },
        }

    def __enter__(self):
        return self

    def mark(self, name, **details):
        self.current = name
        self.result["checks"][name] = {"status": "verified", **details}

    def open(self):
        verify_case_archive(self.original, expected_manifest_sha256=self.sha)
        self.mark("archive_integrity", scope="caller_pinned_manifest_bytes")
        self.current = "recomputation_snapshot"
        self.archive = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="marvis-extension-originals-"))).resolve()
        _authenticated_snapshot(self.original, self.archive, self.sha)
        self.mark("recomputation_snapshot", scope="bounded_authenticated_parser_inputs")
        self.current = "external_bindings"
        self.manifest = _json(self.archive, "manifest.json")
        bindings = _json(self.archive, "bindings.json")
        for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"):
            equal(bindings.get(key), getattr(self.binding, key))
        equal(_json(self.archive, "case.json"), self.binding.case.model_dump())
        if (self.manifest["case_id"] != self.binding.case.id
                or bindings["execution_before_scoring"]["execution"]["task_id"] != self.binding.task_id):
            raise ValueError("extension task or case binding differs")
        self.conn = self.stack.enter_context(closing(sqlite3.connect(_file(self.archive, "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)))
        self.conn.row_factory = sqlite3.Row
        task = self.conn.execute("SELECT * FROM tasks WHERE id=?", (self.binding.task_id,)).fetchone()
        if task is None or task["task_type"] != self.binding.case.task.task_type:
            raise ValueError("extension native task differs")
        self.mark("external_bindings", scope="consistency_not_signature_authority")
        self.current = "native_records"
        from .runtime_native_identity import resolve_native_case
        self.business_case = resolve_native_case(self)
        return self

    def row(self, query, parameters, expected_sha256):
        rows = self.conn.execute(query, parameters).fetchmany(2)
        if len(rows) != 1 or digest(dict(rows[0])) != expected_sha256:
            raise ValueError("externally pinned native record differs")
        return dict(rows[0])

    def artifact(self, artifact_id, expected_sha256, *, kind=None, origin=None):
        row = self.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?", (artifact_id, self.binding.task_id)).fetchone()
        if row is None or row["content_hash"] != expected_sha256 or (kind and row["kind"] != kind) or (origin and row["origin_tool"] != origin):
            raise ValueError("extension registered artifact identity differs")
        raw = Path(row["path"])
        path = (_original_path(self.archive, self.manifest["original_workspace"], raw)
                if raw.is_absolute() else _file(self.archive, Path("workspace") / raw))
        if digest(path.read_bytes()) != expected_sha256:
            raise ValueError("extension registered artifact bytes differ")
        return json.loads(path.read_bytes())

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.result["checks"][self.current] = {"status": "unsupported" if isinstance(exc, _Unsupported) else "mismatch", "error_type": type(exc).__name__}
        if self.result["checks"]["archive_integrity"]["status"] == "verified":
            try:
                verify_case_archive(self.original, expected_manifest_sha256=self.sha)
                verify_case_archive(self.archive, expected_manifest_sha256=self.sha)
                self.result["archive_unchanged_during_revalidation"] = True
            except Exception:
                self.result["checks"]["archive_integrity"] = {"status": "mismatch", "reason": "archive_changed"}
                self.result["archive_unchanged_during_revalidation"] = False
        self.stack.close()
        self.result["supported_checks_verified"] = all(item["status"] == "verified" for item in self.result["checks"].values())
        return exc is not None
