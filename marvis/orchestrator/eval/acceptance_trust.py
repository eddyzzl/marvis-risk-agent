"""Read-only, externally pinned audit acceptance composition.

Only the two installed adapters below can run. No key, signature, trust profile
or original receipt is created here. Public keys express the accepting operator's
external authority policy; a valid signature authenticates that authority's
statement, not an independently discovered institution or real-world fact.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from xml.etree import ElementTree
from xml.parsers import expat

from .runtime_contracts import digest


SCHEMA = "marvis.audit-trust.v1"
ATTESTATION_SCHEMA = "marvis.audit-attestation.v1"
SOURCE_SCOPE = "actual_checkout_except_vcs_and_generated_caches.v1"
_SOURCE_CACHE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
MAX_JSON_BYTES = 4 * 1024**2
MAX_RESULT_BYTES = 16 * 1024**2
MAX_SOURCE_BYTES = 512 * 1024**2
MAX_FILES = 30_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ADAPTERS = {"code_checks.v1": {"C"}, "runtime_validation_v2.v1": {"R", "A", "H"}}


class TrustInputError(ValueError):
    """Fixed reason codes only; never include submitted documents or credentials."""


def _require(condition, reason):
    if not condition:
        raise TrustInputError(reason)


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _keys(value, expected):
    _require(
        isinstance(value, dict) and set(value) == set(expected), "invalid_trust_schema"
    )


def _strings(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and 0 < len(item) <= 500 for item in value)
        and len(set(value)) == len(value)
    )


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise TrustInputError("nonfinite_json_value")

    try:
        result = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=invalid_constant
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TrustInputError("invalid_json") from exc
    _require(isinstance(result, dict), "json_object_required")
    return result


def _path(root, name, *, directory=False):
    _require(isinstance(name, str) and bool(name), "unsafe_trust_path")
    relative = Path(name)
    _require(
        not relative.is_absolute()
        and all(p not in {".", ".."} for p in relative.parts),
        "unsafe_trust_path",
    )
    path = root / relative
    _require(path.resolve() == path and path.is_relative_to(root), "unsafe_trust_path")
    info = path.lstat()
    _require(
        stat.S_ISDIR(info.st_mode)
        if directory
        else stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
        "unsafe_trust_path",
    )
    return path


def _read(path, maximum=MAX_JSON_BYTES):
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and path.resolve() == path,
        "unsafe_trust_path",
    )
    _require(info.st_size <= maximum, "evidence_size_limit")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    _require(
        len(raw) == info.st_size and len(raw) <= maximum,
        "evidence_changed_or_oversized",
    )
    return raw


def _bound(root, reference, *, maximum=MAX_JSON_BYTES):
    _keys(reference, {"path", "sha256"})
    _require(_hash(reference["sha256"]), "invalid_digest")
    raw = _read(_path(root, reference["path"]), maximum)
    _require(digest(raw) == reference["sha256"], "external_pin_mismatch")
    return raw


def source_inventory(source_root: Path) -> dict[str, str]:
    """Hash the actual checkout, independent of tracked state or .gitignore.

    Use a dedicated source checkout with evidence stored outside it. Only Git
    metadata and fixed generated cache directories are omitted. Execution
    authorities must attest isolated empty caches; neither this inventory nor
    signatures discover the original process or its environment independently.
    """
    root = Path(source_root).resolve()
    try:
        tracked = (
            subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
            .decode()
            .split("\0")
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise TrustInputError("source_inventory_unavailable") from exc
    names = {name for name in tracked if name}
    for current, directories, files in os.walk(root, followlinks=False):
        current = Path(current)
        retained = []
        for name in directories:
            if (current == root and name == ".git") or name in _SOURCE_CACHE_DIRS:
                continue
            _require(not (current / name).is_symlink(), "source_alias_rejected")
            retained.append(name)
        directories[:] = retained
        for name in files:
            if current == root and name == ".git":
                continue  # A managed Git worktree uses a metadata pointer file.
            path = current / name
            _require(not path.is_symlink(), "source_alias_rejected")
            names.add(path.relative_to(root).as_posix())
            _require(len(names) <= MAX_FILES, "source_inventory_size_limit")
    _require(0 < len(names) <= MAX_FILES, "source_inventory_size_limit")
    result, total = {}, 0
    for name in sorted(names):
        path = _path(root, name)
        total += path.stat().st_size
        _require(total <= MAX_SOURCE_BYTES, "source_inventory_size_limit")
        result[name] = digest(_read(path, MAX_SOURCE_BYTES))
    return result


def verify_junit(raw: bytes, *, expected_test_ids: list[str]) -> dict:
    """Recount the entire frozen check set from bounded, non-executable XML."""
    _require(
        isinstance(raw, bytes) and len(raw) <= MAX_RESULT_BYTES, "junit_size_limit"
    )
    _require(_strings(expected_test_ids), "invalid_frozen_check_set")

    def forbidden(*_):
        raise TrustInputError("junit_dtd_or_entity_forbidden")

    depth, elements = 0, 0

    def start(*_):
        nonlocal depth, elements
        depth += 1
        elements += 1
        _require(depth <= 64 and elements <= MAX_FILES * 8, "junit_structure_limit")

    def end(*_):
        nonlocal depth
        depth -= 1

    try:
        parser = expat.ParserCreate()
        parser.StartElementHandler = start
        parser.EndElementHandler = end
        parser.StartDoctypeDeclHandler = forbidden
        parser.EntityDeclHandler = forbidden
        parser.ExternalEntityRefHandler = forbidden
        parser.Parse(raw, True)
        root = ElementTree.fromstring(raw)
    except (expat.ExpatError, ElementTree.ParseError) as exc:
        raise TrustInputError("invalid_junit_xml") from exc
    _require(root.tag in {"testsuite", "testsuites"}, "invalid_junit_root")
    cases = []

    def ancillary(node):
        if node.tag == "properties":
            _require(
                all(child.tag == "property" and len(child) == 0 for child in node),
                "invalid_junit_structure",
            )
        else:
            _require(len(node) == 0, "invalid_junit_structure")

    def walk(node):
        counts = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
        _require(node.tag in {"testsuite", "testsuites"}, "invalid_junit_structure")
        for child in node:
            if child.tag in {"testsuite", "testsuites"}:
                nested = walk(child)
                for key in counts:
                    counts[key] += nested[key]
            elif child.tag == "testcase":
                classname, name = child.get("classname"), child.get("name")
                _require(
                    isinstance(classname, str)
                    and bool(classname.strip())
                    and isinstance(name, str)
                    and bool(name.strip()),
                    "junit_case_identity_missing",
                )
                cases.append(f"{classname}::{name}")
                _require(len(cases) <= MAX_FILES, "junit_case_limit")
                counts["tests"] += 1
                outcomes = [
                    c.tag for c in child if c.tag in {"failure", "error", "skipped"}
                ]
                _require(len(outcomes) <= 1, "ambiguous_junit_outcome")
                if outcomes:
                    counts[
                        {
                            "failure": "failures",
                            "error": "errors",
                            "skipped": "skipped",
                        }[outcomes[0]]
                    ] += 1
                _require(
                    all(
                        c.tag
                        in {
                            "failure",
                            "error",
                            "skipped",
                            "system-out",
                            "system-err",
                            "properties",
                        }
                        for c in child
                    ),
                    "invalid_junit_structure",
                )
                for detail in child:
                    ancillary(detail)
            else:
                _require(
                    child.tag in {"properties", "system-out", "system-err"},
                    "invalid_junit_structure",
                )
                ancillary(child)
        for key, count in counts.items():
            if key in node.attrib:
                declared = node.attrib[key]
                _require(
                    len(declared) <= 10
                    and re.fullmatch(r"[0-9]+", declared) is not None
                    and int(declared) == count,
                    "junit_count_mismatch",
                )
        return counts

    counts = walk(root)
    _require(
        len(cases) == len(set(cases)) and set(cases) == set(expected_test_ids),
        "frozen_check_set_mismatch",
    )
    return {
        **counts,
        "passed": counts["tests"]
        - sum(counts[k] for k in ("failures", "errors", "skipped")),
        "case_ids": sorted(cases),
    }


def _public_key(value):
    _require(isinstance(value, str), "invalid_authority_key")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise TrustInputError("invalid_authority_key") from exc
    _require(len(raw) == 32, "invalid_authority_key")
    return raw


def _verify_signature(key, body, signature):
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise TrustInputError("signature_dependency_unavailable") from exc
    try:
        raw = base64.b64decode(signature, validate=True)
        _require(len(raw) == 64, "invalid_attestation_signature")
        payload = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        Ed25519PublicKey.from_public_bytes(key).verify(
            raw, ATTESTATION_SCHEMA.encode() + b"\0" + payload
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise TrustInputError("attestation_signature_not_verified") from exc


def _current_commit(source_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrustInputError("source_commit_unavailable") from exc


def load_trusted_environment(
    config_path: Path,
    *,
    expected_sha256: str,
    source_root: Path,
    evidence_root: Path,
    expected_commit: str,
    spec_sha256: str,
):
    """Read an operator-pinned configuration; never discover one in a bundle."""
    path = Path(config_path).absolute()
    source_root, evidence_root = (
        Path(source_root).resolve(),
        Path(evidence_root).resolve(),
    )
    _require(
        not path.resolve().is_relative_to(source_root)
        and not path.resolve().is_relative_to(evidence_root),
        "trust_configuration_must_be_external",
    )
    config = _json(_bound(path.parent, {"path": path.name, "sha256": expected_sha256}))
    _keys(
        config,
        {
            "schema",
            "commit",
            "spec_sha256",
            "source_scope",
            "source_inventory",
            "authorities",
            "records",
        },
    )
    _require(
        config["schema"] == SCHEMA and config["source_scope"] == SOURCE_SCOPE,
        "unsupported_trust_configuration",
    )
    _require(
        isinstance(expected_commit, str)
        and _COMMIT.fullmatch(expected_commit) is not None
        and config["commit"] == expected_commit
        and _hash(spec_sha256)
        and config["spec_sha256"] == spec_sha256,
        "trust_commit_or_spec_mismatch",
    )
    _require(
        _current_commit(source_root) == expected_commit,
        "current_source_commit_mismatch",
    )
    inventory = source_inventory(source_root)
    _require(
        config["source_inventory"] == inventory, "current_source_inventory_mismatch"
    )
    authorities = config["authorities"]
    _require(
        isinstance(authorities, dict) and 2 <= len(authorities) <= 100,
        "invalid_authority_registry",
    )
    for identity, authority in authorities.items():
        _keys(
            authority,
            {
                "principal_id",
                "organization_id",
                "role",
                "public_key",
                "tiers",
                "environments",
                "sources",
            },
        )
        _require(
            all(
                isinstance(v, str) and 0 < len(v) <= 160
                for v in (
                    identity,
                    authority["principal_id"],
                    authority["organization_id"],
                )
            ),
            "invalid_authority_identity",
        )
        _require(
            authority["role"] in {"executor", "reviewer"}
            and all(
                _strings(authority[k]) for k in ("tiers", "environments", "sources")
            ),
            "invalid_authority_policy",
        )
        _public_key(authority["public_key"])
    records = config["records"]
    _require(
        isinstance(records, list) and 0 < len(records) <= 1000,
        "invalid_record_inventory",
    )
    seen, seen_hashes = set(), set()
    for entry in records:
        _keys(entry, {"path", "sha256", "adapter", "inputs", "execution", "review"})
        _require(
            isinstance(entry["path"], str)
            and entry["path"] not in seen
            and _hash(entry["sha256"])
            and entry["sha256"] not in seen_hashes,
            "invalid_record_inventory",
        )
        seen.add(entry["path"])
        seen_hashes.add(entry["sha256"])
        _require(
            isinstance(entry["adapter"], str)
            and entry["adapter"] in _ADAPTERS
            and isinstance(entry["inputs"], dict),
            "unsupported_acceptance_adapter",
        )
        _path(evidence_root, entry["path"])
        for role in ("execution", "review"):
            _keys(entry[role], {"path", "sha256"})
            attestation_path = _path(path.parent, entry[role]["path"])
            _require(
                not attestation_path.is_relative_to(evidence_root)
                and not attestation_path.is_relative_to(source_root),
                "attestation_must_be_external",
            )
    return TrustedAcceptanceEnvironment(
        deepcopy(config), path.parent, source_root, evidence_root, expected_sha256
    )


@dataclass
class TrustedAcceptanceEnvironment:
    config: dict
    trust_root: Path
    source_root: Path
    evidence_root: Path
    config_sha256: str
    results: dict = field(default_factory=dict)

    @property
    def inventory_sha256(self):
        return digest(
            [
                {key: entry[key] for key in ("path", "sha256", "adapter", "inputs")}
                for entry in self.config["records"]
            ]
        )

    def begin(self, ledger):
        # Memoization serves the receipt/review pair within one check only.
        # Reusing an environment must authenticate and recompute current bytes.
        self.results.clear()
        return self.inventory_errors(ledger)

    def inventory_errors(self, ledger):
        try:
            actual, coverage = {}, {}
            for finding in ledger["findings"]:
                finding_paths = set()
                for item in finding["evidence"]:
                    _keys(item, {"path", "sha256"})
                    _require(
                        isinstance(item["path"], str) and _hash(item["sha256"]),
                        "invalid_record_reference",
                    )
                    _require(
                        item["path"] not in actual
                        or actual[item["path"]] == item["sha256"],
                        "conflicting_record_references",
                    )
                    _require(
                        item["path"] not in finding_paths, "duplicate_finding_record"
                    )
                    finding_paths.add(item["path"])
                    actual[item["path"]] = item["sha256"]
                    coverage.setdefault(item["path"], set()).add(finding["id"])
            expected = {
                entry["path"]: entry["sha256"] for entry in self.config["records"]
            }
            _require(actual == expected, "complete_record_inventory_mismatch")
            for entry in self.config["records"]:
                record = _json(
                    _bound(
                        self.evidence_root, {k: entry[k] for k in ("path", "sha256")}
                    )
                )
                _require(
                    _strings(record.get("finding_ids"))
                    and coverage[entry["path"]] == set(record["finding_ids"]),
                    "complete_finding_record_coverage_mismatch",
                )
            _require(
                _current_commit(self.source_root) == self.config["commit"],
                "current_source_commit_mismatch",
            )
            _require(
                source_inventory(self.source_root) == self.config["source_inventory"],
                "current_source_inventory_mismatch",
            )
            return []
        except (TrustInputError, KeyError, TypeError, OSError):
            return ["trusted complete record or current source inventory mismatch"]

    def verifiers_for(self, reference):
        entry = next(
            (
                item
                for item in self.config["records"]
                if item["path"] == reference.get("path")
                and item["sha256"] == reference.get("sha256")
            ),
            None,
        )
        if entry is None:
            return {}, None

        def receipt(record, root):
            return self._verify(entry, record, root).get("execution_verified") is True

        def review(record, root):
            return self._verify(entry, record, root).get("review_verified") is True

        return {tier: receipt for tier in _ADAPTERS[entry["adapter"]]}, review

    def _attestation(self, entry, record, role):
        envelope = _json(_bound(self.trust_root, entry[role]))
        _keys(envelope, {"body", "signature"})
        body = envelope["body"]
        _keys(
            body,
            {
                "schema",
                "kind",
                "authority_id",
                "record_sha256",
                "spec_sha256",
                "commit",
                "source_inventory_sha256",
                "record_inventory_sha256",
                "adapter",
                "inputs_sha256",
                "run_id",
                "target_id",
                "evidence_tier",
                "environment_kind",
                "source_kind",
                "facts",
            },
        )
        _require(
            body["schema"] == ATTESTATION_SCHEMA and body["kind"] == role,
            "attestation_schema_mismatch",
        )
        authority = self.config["authorities"].get(body["authority_id"])
        _require(
            isinstance(authority, dict)
            and authority["role"]
            == ("executor" if role == "execution" else "reviewer"),
            "authority_not_permitted",
        )
        for name, policy in (
            ("evidence_tier", "tiers"),
            ("environment_kind", "environments"),
            ("source_kind", "sources"),
        ):
            _require(
                record.get(name) == body[name] and body[name] in authority[policy],
                "authority_scope_mismatch",
            )
        expected = {
            "record_sha256": entry["sha256"],
            "spec_sha256": self.config["spec_sha256"],
            "commit": self.config["commit"],
            "source_inventory_sha256": digest(self.config["source_inventory"]),
            "record_inventory_sha256": self.inventory_sha256,
            "adapter": entry["adapter"],
            "inputs_sha256": digest(entry["inputs"]),
            "run_id": record.get("run_id"),
            "target_id": record.get("target_id"),
        }
        _require(
            all(body[key] == value for key, value in expected.items()),
            "attestation_binding_mismatch",
        )
        _verify_signature(
            _public_key(authority["public_key"]), body, envelope["signature"]
        )
        _require(isinstance(body["facts"], dict), "attestation_facts_unavailable")
        return authority, body["facts"]

    def _verify(self, entry, record, root):
        identity = entry["sha256"]
        if identity in self.results:
            return self.results[identity]
        result = {
            "record_sha256": identity,
            "adapter": entry["adapter"],
            "execution_verified": False,
            "review_verified": False,
            "status": "unavailable",
        }
        self.results[identity] = result
        try:
            _require(
                Path(root).resolve() == self.evidence_root, "evidence_root_mismatch"
            )
            original = _json(
                _bound(self.evidence_root, {"path": entry["path"], "sha256": identity})
            )
            _require(digest(original) == digest(record), "record_binding_mismatch")
            _require(
                record["evidence_tier"] in _ADAPTERS[entry["adapter"]],
                "unsupported_acceptance_tier",
            )
            executor, execution = self._attestation(entry, record, "execution")
            reviewer, review = self._attestation(entry, record, "review")
            _require(
                executor["principal_id"] != reviewer["principal_id"]
                and _public_key(executor["public_key"])
                != _public_key(reviewer["public_key"]),
                "independent_reviewer_required",
            )
            _require(
                record["independent_review"]["reviewer_id"] == reviewer["principal_id"],
                "reviewer_identity_mismatch",
            )
            if record["evidence_tier"] != "C":
                _require(
                    record["real_executor"]["id"] == executor["principal_id"],
                    "executor_identity_mismatch",
                )
            if entry["adapter"] == "code_checks.v1":
                facts = _code_checks(
                    entry["inputs"], execution, self.evidence_root, record
                )
            else:
                facts = _runtime_validation(
                    entry["inputs"], execution, self.evidence_root, record, self.config
                )
            result["observed"] = facts
            result["execution_verified"] = facts["successful"]
            _keys(
                review,
                {
                    "decision",
                    "review_artifact_sha256",
                    "reviewed_findings",
                    "coverage",
                    "unresolved_findings",
                },
            )
            review_path = record["independent_review"]["artifact_path"]
            _require(
                review["review_artifact_sha256"]
                == record["artifact_hashes"][review_path],
                "review_artifact_binding_mismatch",
            )
            _bound(
                self.evidence_root,
                {"path": review_path, "sha256": review["review_artifact_sha256"]},
            )
            _require(
                review["reviewed_findings"] == record["finding_ids"]
                and isinstance(review["unresolved_findings"], list),
                "review_scope_mismatch",
            )
            required = {"frozen_criteria", "complete_run_denominator"}
            if entry["adapter"] == "runtime_validation_v2.v1":
                required.add("narrative_semantics")
            if record["evidence_tier"] == "A":
                required.add("blind_real_model_execution")
            if record["evidence_tier"] == "H":
                required.update({"historical_data_origin", "business_label_validity"})
            _require(
                _strings(review["coverage"]) and required <= set(review["coverage"]),
                "review_facts_unavailable",
            )
            result["review_verified"] = (
                review["decision"] == "accepted" and review["unresolved_findings"] == []
            )
            result["status"] = (
                "verified"
                if result["execution_verified"] and result["review_verified"]
                else "not_verified"
            )
            if not facts["successful"] and isinstance(facts.get("reason"), str):
                result["reason"] = facts["reason"]
                if facts["reason"].endswith("unavailable"):
                    result["status"] = "unavailable"
        except TrustInputError as exc:
            result.update(
                status="unavailable"
                if str(exc).endswith("unavailable")
                else "not_verified",
                reason=str(exc),
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            result.update(status="not_verified", reason="trust_evidence_rejected")
        return result

    def report(self):
        return {
            "config_sha256": self.config_sha256,
            "source_scope": SOURCE_SCOPE,
            "record_inventory_sha256": self.inventory_sha256,
            "authority_scope": "externally_pinned_authority_statements_not_institution_discovery",
            "configured_record_count": len(self.config["records"]),
            "records": [
                deepcopy(
                    self.results.get(
                        entry["sha256"],
                        {
                            "record_sha256": entry["sha256"],
                            "adapter": entry["adapter"],
                            "status": "unavailable",
                            "reason": "record_not_referenced_or_precheck_failed",
                            "execution_verified": False,
                            "review_verified": False,
                        },
                    )
                )
                for entry in self.config["records"]
            ],
        }


def _verify_invocation(frozen, observed):
    """Bind external process observations, without exposing argv/env values.

    Digests cover canonical complete argv, runtime identity (interpreter bytes,
    implementation/version/platform), installed dependency inventory (including
    loaded pytest plugins), and the complete process environment descriptor.
    Their private originals are held by the external execution authority. This
    verifier checks its signed observation against the operator's frozen values;
    it never claims to recover or independently inspect that past environment.
    """
    fields = {
        "entrypoint",
        "cwd",
        "argv_sha256",
        "runtime_identity_sha256",
        "dependency_inventory_sha256",
        "environment_sha256",
        "isolation",
    }
    _keys(frozen, fields)
    _keys(observed, fields)
    _require(
        frozen["entrypoint"] == "python_module:pytest"
        and frozen["cwd"] == "source_root"
        and frozen["isolation"] == "fresh_process_isolated_empty_caches",
        "unsupported_check_invocation",
    )
    _require(
        all(_hash(frozen[key]) for key in fields if key.endswith("_sha256")),
        "check_invocation_facts_unavailable",
    )
    _require(frozen == observed, "observed_check_invocation_mismatch")


def _code_checks(inputs, execution, root, record):
    _keys(inputs, {"checks"})
    checks = inputs["checks"]
    _require(
        isinstance(checks, list) and 0 < len(checks) <= 100, "invalid_frozen_checks"
    )
    _keys(execution, {"checks"})
    _require(isinstance(execution["checks"], dict), "execution_facts_unavailable")
    rows, ids = [], set()
    for check in checks:
        _require(
            isinstance(check, dict) and "invocation" in check,
            "check_invocation_facts_unavailable",
        )
        _keys(check, {"id", "junit", "test_ids", "invocation"})
        check_id = check["id"]
        _require(
            isinstance(check_id, str)
            and re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", check_id) is not None
            and check_id not in ids,
            "invalid_frozen_checks",
        )
        ids.add(check_id)
        observed = execution["checks"].get(check_id)
        _require(
            isinstance(observed, dict) and "invocation" in observed,
            "check_invocation_facts_unavailable",
        )
        _keys(observed, {"exit_code", "junit_sha256", "invocation"})
        _verify_invocation(check["invocation"], observed["invocation"])
        _require(
            type(observed["exit_code"]) is int
            and observed["junit_sha256"] == check["junit"]["sha256"],
            "execution_check_binding_mismatch",
        )
        _require(
            record["artifact_hashes"].get(check["junit"]["path"])
            == check["junit"]["sha256"],
            "junit_not_record_bound",
        )
        facts = verify_junit(
            _bound(root, check["junit"], maximum=MAX_RESULT_BYTES),
            expected_test_ids=check["test_ids"],
        )
        # Parameterized test IDs can contain private values; public diagnostics
        # retain their set digest, not original JUnit text or identities.
        case_ids = facts.pop("case_ids")
        rows.append(
            {
                "id": check_id,
                "exit_code": observed["exit_code"],
                "check_set_sha256": digest(case_ids),
                **facts,
            }
        )
    _require(set(execution["checks"]) == ids, "complete_check_inventory_mismatch")
    denominator = sum(row["tests"] for row in rows)
    passed = sum(row["passed"] for row in rows)
    return {
        "checks": rows,
        "denominator": denominator,
        "passed": passed,
        "successful": passed == denominator
        and all(row["exit_code"] == 0 for row in rows),
    }


def _runtime_validation(inputs, execution, root, record, config):
    from .runtime_run_manifest import verify_run_manifest

    _keys(inputs, {"run_dir", "final_manifest_sha256", "cases"})
    run = verify_run_manifest(
        _path(root, inputs["run_dir"], directory=True),
        expected_manifest_sha256=inputs["final_manifest_sha256"],
    )
    # Once the final carrier is authenticated, every original case remains in
    # the output even if a required private archive/binding is unavailable.
    failed = {
        "denominator": run["manifest"]["denominator"],
        "passed": 0,
        "successful": False,
        "cases": [
            {
                "case_id": case_id,
                "runtime_status": run["executions"][case_id]["runtime_status"],
                "recorded_passed": run["scores"][case_id]["passed"],
                "successful": False,
                "domain_status": "unverified",
            }
            for case_id in run["manifest"]["case_ids"]
        ],
    }
    if record["evidence_tier"] == "A":
        # The V2 validation carrier has no formal benchmark threshold policy.
        # Do not invent all-case or any-case semantics for formal acceptance.
        return {**failed, "reason": "formal_acceptance_aggregate_unavailable"}
    try:
        return _runtime_validation_checks(inputs, execution, root, record, config, run)
    except TrustInputError as exc:
        return {**failed, "reason": str(exc)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {**failed, "reason": "runtime_domain_evidence_unavailable"}


def _runtime_validation_checks(inputs, execution, root, record, config, run):
    # Imported from installed code, never from the candidate's paths or config.
    from .runtime_archive_validation import (
        FrozenValidationBinding,
        revalidate_validation_archive,
    )

    _keys(inputs, {"run_dir", "final_manifest_sha256", "cases"})
    _keys(execution, {"case_ids", "environment_id", "model", "historical_data"})
    _require(
        isinstance(execution["environment_id"], str)
        and bool(execution["environment_id"]),
        "execution_facts_unavailable",
    )
    manifest, frozen = run["manifest"], run["manifest"]["frozen"]
    manifest_path = str(Path(inputs["run_dir"]) / "final-manifest.json")
    _require(
        record["artifact_hashes"].get(manifest_path) == inputs["final_manifest_sha256"],
        "final_manifest_not_record_bound",
    )
    _require(
        manifest["run_id"] == record["run_id"]
        and execution["case_ids"] == manifest["case_ids"],
        "run_identity_or_denominator_mismatch",
    )
    package_files = {
        path: sha
        for path, sha in config["source_inventory"].items()
        if path.startswith("marvis/")
        and "__pycache__" not in Path(path).parts
        and Path(path).suffix != ".pyc"
    }
    _require(
        frozen["source"].get("commit") == config["commit"]
        and frozen["source"].get("source_sha256") == digest(package_files),
        "runtime_current_source_mismatch",
    )
    _require(
        isinstance(inputs["cases"], list)
        and [item.get("case_id") for item in inputs["cases"]] == manifest["case_ids"],
        "complete_case_inventory_mismatch",
    )
    tier = record["evidence_tier"]
    if tier == "H":
        historical = execution["historical_data"]
        _keys(historical, {"source_kind", "dataset_hashes", "label_contract"})
        _require(
            historical["source_kind"] == "deidentified_historical"
            and historical["dataset_hashes"] == record["dataset_hashes"]
            and bool(record["dataset_hashes"]),
            "historical_facts_unavailable",
        )
        _require(
            record["input_hashes"].get(historical["label_contract"]["path"])
            == historical["label_contract"]["sha256"],
            "historical_label_contract_unbound",
        )
        _bound(root, historical["label_contract"])
    cases, data_hashes = [], set()
    for item in inputs["cases"]:
        _keys(item, {"case_id", "archive", "manifest_sha256", "binding"})
        binding_path = _path(root, item["binding"]["path"])
        archive = _path(root, item["archive"], directory=True)
        _require(
            not binding_path.is_relative_to(archive),
            "binding_must_be_external_to_archive",
        )
        binding = FrozenValidationBinding.model_validate_json(
            _bound(root, item["binding"])
        )
        _require(
            record["input_hashes"].get(item["binding"]["path"])
            == item["binding"]["sha256"],
            "case_input_not_record_bound",
        )
        case_id = item["case_id"]
        _require(
            binding.case.id == case_id and binding.run_id == record["run_id"],
            "case_binding_mismatch",
        )
        _require(
            all(
                getattr(binding, key) == frozen[key]
                for key in (
                    "cases_sha256",
                    "expected_sha256",
                    "source",
                    "model_connection_sha256",
                )
            ),
            "frozen_runtime_binding_mismatch",
        )
        observed = run["executions"][case_id]
        row_materials = [
            m
            for m in binding.case.materials
            if m.role in {"sample", "feature", "unknown"}
        ]
        data_hashes.update(m.sha256 for m in row_materials)
        if tier == "H":
            _require(
                row_materials
                and all(
                    m.source_kind == "deidentified_historical" for m in row_materials
                ),
                "historical_material_source_mismatch",
            )
        _require(
            observed.get("materials")
            == [
                {"sha256": m.sha256, "source_kind": m.source_kind, "role": m.role}
                for m in binding.case.materials
            ],
            "final_material_binding_mismatch",
        )
        _require(
            observed.get("case_sha256") == digest(binding.case.model_dump())
            and observed.get("budget") == binding.case.budget.model_dump(),
            "final_case_binding_mismatch",
        )
        custody = observed.get("evidence_custody", {})
        _require(
            custody.get("status") == "retained"
            and custody.get("manifest_sha256") == item["manifest_sha256"],
            "final_custody_binding_mismatch",
        )
        pipeline = observed.get("execution", {}).get("validation_pipeline", {})
        _require(
            observed.get("execution", {}).get("task_id") == binding.task_id
            and all(
                pipeline.get(key) == getattr(binding, key)
                for key in (
                    "input_contract_sha256",
                    "confirmed_draft_sha256",
                    "report_revision",
                )
            ),
            "final_task_binding_mismatch",
        )
        reports = pipeline.get("report_files", [])
        _require(
            isinstance(reports, list)
            and len({r["kind"] for r in reports}) == len(reports)
            and {r["kind"]: r["sha256"] for r in reports} == binding.report_sha256,
            "final_report_binding_mismatch",
        )
        recomputed = revalidate_validation_archive(
            archive,
            expected_manifest_sha256=item["manifest_sha256"],
            frozen_binding=binding,
        )
        original_success = (
            observed["runtime_status"] == "completed"
            and run["scores"][case_id]["passed"] is True
        )
        cases.append(
            {
                "case_id": case_id,
                "runtime_status": observed["runtime_status"],
                "recorded_passed": run["scores"][case_id]["passed"],
                "domain_checks": recomputed["checks"],
                "unsupported": recomputed["unsupported"],
                "successful": original_success
                and recomputed["supported_checks_verified"],
            }
        )
    if tier == "H":
        _require(
            set(record["dataset_hashes"].values()) == data_hashes,
            "historical_datasets_differ_from_runtime_materials",
        )
    return {
        "cases": cases,
        "denominator": manifest["denominator"],
        "passed": sum(c["successful"] for c in cases),
        "regression_gate_passed": run["report"]["regression_gate_passed"],
        "successful": all(c["successful"] for c in cases)
        and run["report"]["regression_gate_passed"] is True,
    }
