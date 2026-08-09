from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import re
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    MappingProxyType,
    ModuleType,
)
from typing import Any, Callable, Iterable, Mapping

from marvis.decision_twin._canonical import parse_datetime
from marvis.decision_twin.artifacts import (
    ContentAddressedAuditStore,
    TamperEvidenceError,
)
from marvis.decision_twin.contracts import ReplayManifest
from marvis.decision_twin.replay import (
    AdapterDecision,
    ObservedField,
    ProtectedGroupObservation,
    ReplayEngine,
    ReplayFacts,
    ReplayRecord,
    ReplayedDecision,
    TrustedAdapterIdentity,
    TrustedReplayAdapter,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRUSTED_PYTHON_GLOBAL_TYPES = frozenset({
    AdapterDecision,
    ReplayFacts,
    ReplayManifest,
})


class AuthenticatedReplayError(RuntimeError):
    """An immutable replay input was missing, malformed, or untrusted."""


class ReplayArtifactNotFound(AuthenticatedReplayError):
    """A required content-addressed replay artifact does not exist."""


class ReplayMaterialRejected(AuthenticatedReplayError):
    """Stored replay material failed schema, provenance, or trust checks."""


@dataclass(frozen=True)
class _AuthenticatedManifest:
    manifest: ReplayManifest
    record_artifact_id: str
    facts_artifact_id: str


@dataclass(frozen=True)
class _PinnedAdapter:
    """Execute only the exact implementation admitted at trusted startup."""

    identity: TrustedAdapterIdentity
    adapter_type: type[TrustedReplayAdapter]
    instance: TrustedReplayAdapter
    replay_implementation: Callable[..., AdapterDecision]
    execution_implementation: Callable[..., AdapterDecision]
    implementation_sha256: str
    referenced_globals: tuple[tuple[str, object], ...]

    def replay(
        self,
        *,
        manifest: ReplayManifest,
        facts: ReplayFacts,
    ) -> AdapterDecision:
        self._verify_implementation()
        output = self.execution_implementation(
            self.instance,
            manifest=manifest,
            facts=facts,
        )
        self._verify_implementation()
        return output

    def _verify_implementation(self) -> None:
        if type(self.instance) is not self.adapter_type:
            raise ReplayMaterialRejected("trusted adapter instance type drifted")
        if vars(self.adapter_type).get("identity") != self.identity:
            raise ReplayMaterialRejected("trusted adapter type identity drifted")
        if vars(self.adapter_type).get("replay") is not self.replay_implementation:
            raise ReplayMaterialRejected("trusted adapter implementation drifted")
        bound_replay = getattr(self.instance, "replay", None)
        if getattr(bound_replay, "__func__", None) is not self.replay_implementation:
            raise ReplayMaterialRejected("trusted adapter instance was substituted")
        if getattr(self.instance, "identity", None) != self.identity:
            raise ReplayMaterialRejected("trusted adapter instance identity drifted")
        if not hmac.compare_digest(
            _callable_implementation_sha256(self.replay_implementation),
            self.implementation_sha256,
        ):
            raise ReplayMaterialRejected("trusted adapter implementation digest drifted")
        for name, expected in self.referenced_globals:
            if self.replay_implementation.__globals__.get(name) is not expected:
                raise ReplayMaterialRejected(
                    f"trusted adapter global binding drifted: {name}"
                )


class ContentAddressedReplayRepository:
    """Fail-closed typed reads over the decision-twin audit store."""

    def __init__(self, store: ContentAddressedAuditStore) -> None:
        if not isinstance(store, ContentAddressedAuditStore):
            raise ValueError("store must be ContentAddressedAuditStore")
        self._store = store

    def load(self, artifact_id: str, *, expected_kind: str) -> Mapping[str, Any]:
        normalized_id = _artifact_id(artifact_id, "artifact_id")
        try:
            return MappingProxyType(
                self._store.get_typed(
                    normalized_id,
                    expected_kind=expected_kind,
                )
            )
        except KeyError as exc:
            raise ReplayArtifactNotFound(
                f"missing {expected_kind} artifact: {normalized_id}"
            ) from exc
        except (TamperEvidenceError, ValueError) as exc:
            raise ReplayMaterialRejected(
                f"rejected {expected_kind} artifact {normalized_id}: {exc}"
            ) from exc


class TrustedAdapterRegistry:
    """Startup-time allowlist of exact adapter types and deployment artifacts.

    Each registration is ``(adapter_type, binary_bytes, package_bytes)``. The
    exact zero-argument type is the executable trust root and is instantiated
    internally; callers cannot pair a look-alike object with authenticated
    bytes. Binary/package digests are measured here rather than accepted as
    caller claims. Registration is frozen after construction.
    """

    def __init__(
        self,
        registrations: Iterable[tuple[type[TrustedReplayAdapter], bytes, bytes]],
    ) -> None:
        entries: dict[
            tuple[str, str, str, str, str, str],
            _PinnedAdapter,
        ] = {}
        for registration in registrations:
            if not isinstance(registration, tuple) or len(registration) != 3:
                raise ValueError(
                    "adapter registration must be "
                    "(trusted adapter type, binary_bytes, package_bytes)"
                )
            adapter_type, raw_binary, raw_package = registration
            if not isinstance(adapter_type, type):
                raise ValueError(
                    "adapter registration must use a trusted adapter type, not an object"
                )
            identity = vars(adapter_type).get("identity")
            replay_implementation = vars(adapter_type).get("replay")
            if not isinstance(identity, TrustedAdapterIdentity) or not callable(
                replay_implementation
            ):
                raise ValueError(
                    "trusted adapter type must declare TrustedAdapterIdentity and replay()"
                )
            try:
                adapter = adapter_type()
            except TypeError as exc:
                raise ValueError(
                    "trusted adapter type must support zero-argument construction"
                ) from exc
            if type(adapter) is not adapter_type:
                raise ValueError("trusted adapter factory returned a substituted type")
            bound_replay = getattr(adapter, "replay", None)
            if (
                getattr(adapter, "identity", None) != identity
                or getattr(bound_replay, "__func__", None) is not replay_implementation
            ):
                raise ValueError(
                    "trusted adapter instance does not match its registered exact type"
                )
            binary = _immutable_bytes(raw_binary, "adapter binary")
            package = _immutable_bytes(raw_package, "adapter package")
            binary_sha256 = hashlib.sha256(binary).hexdigest()
            package_sha256 = hashlib.sha256(package).hexdigest()
            implementation_sha256 = _callable_implementation_sha256(
                replay_implementation
            )
            deployment_sha256 = trusted_adapter_deployment_sha256(
                identity,
                binary_sha256=binary_sha256,
                package_sha256=package_sha256,
                implementation_sha256=implementation_sha256,
            )
            if identity.sha256 != binary_sha256:
                raise ValueError(
                    "registered adapter identity does not match measured binary digest"
                )
            key = (
                identity.adapter_id,
                identity.version,
                binary_sha256,
                package_sha256,
                implementation_sha256,
                deployment_sha256,
            )
            if key in entries:
                raise ValueError("duplicate trusted adapter digest registration")
            entries[key] = _PinnedAdapter(
                identity=identity,
                adapter_type=adapter_type,
                instance=adapter,
                replay_implementation=replay_implementation,
                execution_implementation=_clone_function(replay_implementation),
                implementation_sha256=implementation_sha256,
                referenced_globals=_referenced_globals(replay_implementation),
            )
        self._entries = MappingProxyType(entries)

    def resolve(
        self,
        identity: TrustedAdapterIdentity,
        *,
        binary_sha256: str,
        package_sha256: str,
        implementation_sha256: str,
        deployment_sha256: str,
    ) -> TrustedReplayAdapter:
        key = (
            identity.adapter_id,
            identity.version,
            _artifact_id(binary_sha256, "binary_sha256"),
            _artifact_id(package_sha256, "package_sha256"),
            _artifact_id(implementation_sha256, "implementation_sha256"),
            _artifact_id(deployment_sha256, "deployment_sha256"),
        )
        pinned_adapter = self._entries.get(key)
        if pinned_adapter is None:
            raise ReplayMaterialRejected(
                "adapter binary/package digest is not in the trusted allowlist"
            )
        if pinned_adapter.identity != identity:
            raise ReplayMaterialRejected(
                "resolved adapter identity drifted from trusted material"
            )
        pinned_adapter._verify_implementation()
        return pinned_adapter


class AuthenticatedReplayEntryPoint:
    """Production-facing replay boundary accepting immutable artifact IDs only.

    ``ReplayEngine`` remains the deterministic in-memory primitive. Production
    integrations use this boundary so callers cannot supply their own manifest,
    facts, protected observations, or adapter object.
    """

    def __init__(
        self,
        repository: ContentAddressedReplayRepository,
        adapters: TrustedAdapterRegistry,
    ) -> None:
        if not isinstance(repository, ContentAddressedReplayRepository):
            raise ValueError("repository must be ContentAddressedReplayRepository")
        if not isinstance(adapters, TrustedAdapterRegistry):
            raise ValueError("adapters must be TrustedAdapterRegistry")
        self._repository = repository
        self._adapters = adapters

    def replay(
        self,
        *,
        manifest_artifact_id: str,
        record_artifact_id: str,
        adapter_artifact_id: str,
    ) -> ReplayedDecision:
        authenticated_manifest = self._load_manifest(manifest_artifact_id)
        manifest = authenticated_manifest.manifest
        record = self._load_record(
            record_artifact_id,
            manifest=manifest,
            expected_record_artifact_id=authenticated_manifest.record_artifact_id,
            expected_facts_artifact_id=authenticated_manifest.facts_artifact_id,
        )
        adapter = self._load_adapter(adapter_artifact_id)
        return ReplayEngine(adapter).replay(manifest, record)

    def _load_manifest(self, artifact_id: str) -> _AuthenticatedManifest:
        material = self._repository.load(
            artifact_id,
            expected_kind="replay_manifest",
        )
        _exact_keys(
            material,
            {"schema_version", "manifest"},
            "replay manifest material",
        )
        _schema(
            material,
            "decision_twin.replay_manifest_material.v1",
            "replay manifest material",
        )
        raw_manifest = _mapping(material.get("manifest"), "manifest")
        _exact_keys(
            raw_manifest,
            {"schema_version", "as_of", "issued_at", "bindings"},
            "manifest",
        )
        raw_bindings = _mapping(raw_manifest.get("bindings"), "manifest bindings")
        if set(raw_bindings) != set(ReplayManifest.REQUIRED_BINDINGS):
            raise ReplayMaterialRejected(
                "manifest bindings must contain exactly the required replay bindings"
            )
        for name in ReplayManifest.REQUIRED_BINDINGS:
            raw_binding = _mapping(raw_bindings.get(name), f"{name} binding")
            _exact_keys(
                raw_binding,
                {"version", "sha256", "visible_at"},
                f"{name} binding",
            )
        try:
            manifest = ReplayManifest.from_dict(raw_manifest)
        except (TypeError, ValueError) as exc:
            raise ReplayMaterialRejected(f"invalid replay manifest: {exc}") from exc
        data_record_artifact_id = ""
        data_facts_artifact_id = ""
        for name, binding in manifest.bindings.items():
            is_data_binding = name == "data"
            source = self._repository.load(
                binding.sha256,
                expected_kind=(
                    "replay_data_binding" if is_data_binding else "replay_binding"
                ),
            )
            _exact_keys(
                source,
                (
                    {
                        "schema_version",
                        "name",
                        "version",
                        "visible_at",
                        "record_artifact_id",
                        "facts_artifact_id",
                    }
                    if is_data_binding
                    else {"schema_version", "name", "version", "visible_at"}
                ),
                f"{name} binding source",
            )
            _schema(
                source,
                (
                    "decision_twin.replay_data_binding.v1"
                    if is_data_binding
                    else "decision_twin.replay_binding.v1"
                ),
                f"{name} binding source",
            )
            if is_data_binding:
                data_record_artifact_id = _artifact_id(
                    source.get("record_artifact_id"),
                    "data binding record_artifact_id",
                )
                data_facts_artifact_id = _artifact_id(
                    source.get("facts_artifact_id"),
                    "data binding facts_artifact_id",
                )
            try:
                source_visible_at = parse_datetime(
                    source.get("visible_at"),
                    f"{name} binding source visible_at",
                )
            except ValueError as exc:
                raise ReplayMaterialRejected(str(exc)) from exc
            if (
                source.get("name") != name
                or source.get("version") != binding.version
                or source_visible_at != binding.visible_at
            ):
                raise ReplayMaterialRejected(
                    f"{name} binding does not match its authenticated source artifact"
                )
        return _AuthenticatedManifest(
            manifest=manifest,
            record_artifact_id=data_record_artifact_id,
            facts_artifact_id=data_facts_artifact_id,
        )

    def _load_record(
        self,
        artifact_id: str,
        *,
        manifest: ReplayManifest,
        expected_record_artifact_id: str,
        expected_facts_artifact_id: str,
    ) -> ReplayRecord:
        normalized_artifact_id = _artifact_id(artifact_id, "record_artifact_id")
        if normalized_artifact_id != expected_record_artifact_id:
            raise ReplayMaterialRejected(
                "record artifact does not match manifest data binding"
            )
        material = self._repository.load(
            normalized_artifact_id,
            expected_kind="replay_record",
        )
        _exact_keys(
            material,
            {
                "schema_version",
                "facts_artifact_id",
                "protected_group_artifact_id",
            },
            "replay record",
        )
        _schema(
            material,
            "decision_twin.replay_record.v1",
            "replay record",
        )
        facts_artifact_id = _artifact_id(
            material.get("facts_artifact_id"),
            "facts_artifact_id",
        )
        if facts_artifact_id != expected_facts_artifact_id:
            raise ReplayMaterialRejected(
                "facts artifact does not match manifest data binding"
            )
        protected_artifact_id = _artifact_id(
            material.get("protected_group_artifact_id"),
            "protected_group_artifact_id",
        )
        facts = self._load_facts(facts_artifact_id)
        protected_group, protected_visible_at, protected_record_id = (
            self._load_protected_group(protected_artifact_id)
        )
        if protected_record_id != facts.record_id:
            raise ReplayMaterialRejected(
                "protected group source is bound to a different record_id"
            )
        if protected_visible_at > facts.decision_at:
            raise ReplayMaterialRejected(
                "protected group source is visible after decision_at"
            )
        if protected_visible_at > manifest.as_of:
            raise ReplayMaterialRejected(
                "protected group source is visible after replay as_of"
            )
        return ReplayRecord(facts=facts, protected_group=protected_group)

    def _load_facts(self, artifact_id: str) -> ReplayFacts:
        material = self._repository.load(
            artifact_id,
            expected_kind="replay_facts",
        )
        _exact_keys(
            material,
            {
                "schema_version",
                "record_id",
                "decision_at",
                "field_artifact_ids",
            },
            "replay facts",
        )
        _schema(material, "decision_twin.replay_facts.v1", "replay facts")
        record_id = material.get("record_id")
        raw_field_ids = material.get("field_artifact_ids")
        if not isinstance(raw_field_ids, list) or not raw_field_ids:
            raise ReplayMaterialRejected(
                "field_artifact_ids must be a non-empty JSON array"
            )
        field_ids = tuple(
            _artifact_id(value, "field_artifact_id") for value in raw_field_ids
        )
        if len(set(field_ids)) != len(field_ids):
            raise ReplayMaterialRejected("field_artifact_ids must be unique")
        fields = tuple(
            self._load_field_source(field_id, expected_record_id=record_id)
            for field_id in field_ids
        )
        try:
            return ReplayFacts(
                record_id=record_id,  # type: ignore[arg-type]
                decision_at=parse_datetime(
                    material.get("decision_at"),
                    "decision_at",
                ),
                source_sha256=artifact_id,
                fields=fields,
            )
        except (TypeError, ValueError) as exc:
            raise ReplayMaterialRejected(f"invalid replay facts: {exc}") from exc

    def _load_field_source(
        self,
        artifact_id: str,
        *,
        expected_record_id: object,
    ) -> ObservedField:
        material = self._repository.load(
            artifact_id,
            expected_kind="observed_field_source",
        )
        _exact_keys(
            material,
            {
                "schema_version",
                "record_id",
                "name",
                "value",
                "visible_at",
            },
            "observed field source",
        )
        _schema(
            material,
            "decision_twin.observed_field_source.v1",
            "observed field source",
        )
        if material.get("record_id") != expected_record_id:
            raise ReplayMaterialRejected(
                "observed field source is bound to a different record_id"
            )
        try:
            return ObservedField(
                name=material.get("name"),  # type: ignore[arg-type]
                value=material.get("value"),  # type: ignore[arg-type]
                visible_at=parse_datetime(
                    material.get("visible_at"),
                    "observed field visible_at",
                ),
                source_sha256=artifact_id,
            )
        except (TypeError, ValueError) as exc:
            raise ReplayMaterialRejected(
                f"invalid observed field source: {exc}"
            ) from exc

    def _load_protected_group(
        self,
        artifact_id: str,
    ) -> tuple[ProtectedGroupObservation, datetime, object]:
        material = self._repository.load(
            artifact_id,
            expected_kind="protected_group_source",
        )
        _exact_keys(
            material,
            {
                "schema_version",
                "record_id",
                "attribute",
                "group",
                "governance_ref",
                "visible_at",
            },
            "protected group source",
        )
        _schema(
            material,
            "decision_twin.protected_group_source.v1",
            "protected group source",
        )
        try:
            visible_at = parse_datetime(
                material.get("visible_at"),
                "protected group visible_at",
            )
            protected_group = ProtectedGroupObservation(
                attribute=material.get("attribute"),  # type: ignore[arg-type]
                group=material.get("group"),  # type: ignore[arg-type]
                governance_ref=material.get("governance_ref"),  # type: ignore[arg-type]
                source_sha256=artifact_id,
            )
        except (TypeError, ValueError) as exc:
            raise ReplayMaterialRejected(
                f"invalid protected group source: {exc}"
            ) from exc
        return protected_group, visible_at, material.get("record_id")

    def _load_adapter(self, artifact_id: str) -> TrustedReplayAdapter:
        material = self._repository.load(
            artifact_id,
            expected_kind="trusted_adapter_material",
        )
        _exact_keys(
            material,
            {
                "schema_version",
                "identity",
                "binary_artifact_id",
                "binary_sha256",
                "package_artifact_id",
                "package_sha256",
                "implementation_sha256",
                "deployment_sha256",
            },
            "trusted adapter material",
        )
        _schema(
            material,
            "decision_twin.trusted_adapter_material.v2",
            "trusted adapter material",
        )
        raw_identity = _mapping(material.get("identity"), "adapter identity")
        _exact_keys(
            raw_identity,
            {"adapter_id", "version", "sha256", "execution_mode"},
            "adapter identity",
        )
        try:
            identity = TrustedAdapterIdentity(
                adapter_id=raw_identity.get("adapter_id"),  # type: ignore[arg-type]
                version=raw_identity.get("version"),  # type: ignore[arg-type]
                sha256=raw_identity.get("sha256"),  # type: ignore[arg-type]
                execution_mode=raw_identity.get("execution_mode"),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ReplayMaterialRejected(f"invalid adapter identity: {exc}") from exc
        binary_id = _artifact_id(
            material.get("binary_artifact_id"),
            "binary_artifact_id",
        )
        package_id = _artifact_id(
            material.get("package_artifact_id"),
            "package_artifact_id",
        )
        claimed_binary_sha256 = _artifact_id(
            material.get("binary_sha256"),
            "binary_sha256",
        )
        claimed_package_sha256 = _artifact_id(
            material.get("package_sha256"),
            "package_sha256",
        )
        claimed_implementation_sha256 = _artifact_id(
            material.get("implementation_sha256"),
            "implementation_sha256",
        )
        claimed_deployment_sha256 = _artifact_id(
            material.get("deployment_sha256"),
            "deployment_sha256",
        )
        binary = self._load_encoded_material(
            binary_id,
            expected_kind="trusted_adapter_binary",
            expected_schema="decision_twin.trusted_adapter_binary.v1",
        )
        package = self._load_encoded_material(
            package_id,
            expected_kind="trusted_adapter_package",
            expected_schema="decision_twin.trusted_adapter_package.v1",
        )
        measured_binary_sha256 = hashlib.sha256(binary).hexdigest()
        measured_package_sha256 = hashlib.sha256(package).hexdigest()
        if (
            claimed_binary_sha256 != measured_binary_sha256
            or identity.sha256 != measured_binary_sha256
        ):
            raise ReplayMaterialRejected(
                "adapter binary digest does not match authenticated material"
            )
        if claimed_package_sha256 != measured_package_sha256:
            raise ReplayMaterialRejected(
                "adapter package digest does not match authenticated material"
            )
        measured_deployment_sha256 = trusted_adapter_deployment_sha256(
            identity,
            binary_sha256=measured_binary_sha256,
            package_sha256=measured_package_sha256,
            implementation_sha256=claimed_implementation_sha256,
        )
        if not hmac.compare_digest(
            claimed_deployment_sha256,
            measured_deployment_sha256,
        ):
            raise ReplayMaterialRejected(
                "adapter deployment digest does not bind authenticated material"
            )
        return self._adapters.resolve(
            identity,
            binary_sha256=measured_binary_sha256,
            package_sha256=measured_package_sha256,
            implementation_sha256=claimed_implementation_sha256,
            deployment_sha256=measured_deployment_sha256,
        )

    def _load_encoded_material(
        self,
        artifact_id: str,
        *,
        expected_kind: str,
        expected_schema: str,
    ) -> bytes:
        material = self._repository.load(
            artifact_id,
            expected_kind=expected_kind,
        )
        _exact_keys(
            material,
            {"schema_version", "encoding", "content"},
            expected_kind,
        )
        _schema(material, expected_schema, expected_kind)
        if material.get("encoding") != "base64":
            raise ReplayMaterialRejected(f"{expected_kind} encoding must be base64")
        content = material.get("content")
        if not isinstance(content, str) or not content:
            raise ReplayMaterialRejected(
                f"{expected_kind} content must be non-empty base64 text"
            )
        try:
            decoded = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ReplayMaterialRejected(
                f"{expected_kind} content is not valid base64"
            ) from exc
        if not decoded:
            raise ReplayMaterialRejected(f"{expected_kind} content is empty")
        return decoded


def trusted_adapter_implementation_sha256(
    adapter_type: type[TrustedReplayAdapter],
) -> str:
    """Measure the exact Python replay implementation admitted for execution."""

    if not isinstance(adapter_type, type):
        raise ValueError("adapter_type must be a type")
    implementation = vars(adapter_type).get("replay")
    if not callable(implementation):
        raise ValueError("adapter_type must declare replay()")
    return _callable_implementation_sha256(implementation)


def trusted_adapter_deployment_sha256(
    identity: TrustedAdapterIdentity,
    *,
    binary_sha256: str,
    package_sha256: str,
    implementation_sha256: str,
) -> str:
    """Bind deployment bytes and measured executable code into one trust root."""

    if not isinstance(identity, TrustedAdapterIdentity):
        raise ValueError("identity must be TrustedAdapterIdentity")
    components = (
        "decision_twin.trusted_adapter_deployment.v1",
        identity.adapter_id,
        identity.version,
        identity.execution_mode,
        _artifact_id(binary_sha256, "binary_sha256"),
        _artifact_id(package_sha256, "package_sha256"),
        _artifact_id(implementation_sha256, "implementation_sha256"),
    )
    return hashlib.sha256("\x00".join(components).encode("utf-8")).hexdigest()


def _callable_implementation_sha256(implementation: object) -> str:
    if not isinstance(implementation, FunctionType):
        raise ValueError("trusted adapter replay must be a Python function")
    return hashlib.sha256(
        _callable_implementation_bytes(implementation, active=set())
    ).hexdigest()


def _callable_implementation_bytes(
    implementation: FunctionType,
    *,
    active: set[int],
) -> bytes:
    if implementation.__closure__:
        raise ValueError(
            "trusted adapter replay/helper closures are unsupported; "
            "use measured module-level functions and immutable arguments"
        )
    marker = id(implementation)
    identity = (
        f"{implementation.__module__}\x00{implementation.__qualname__}"
    ).encode("utf-8")
    if marker in active:
        return b"decision_twin.python_function.recursion.v1\x00" + identity
    active.add(marker)
    try:
        digest = hashlib.sha256()
        digest.update(b"decision_twin.python_replay_implementation.v2\x00")
        _update_digest_frame(digest, b"identity", identity)
        _update_digest_frame(
            digest,
            b"code",
            _stable_code_bytes(implementation.__code__),
        )
        _update_digest_frame(
            digest,
            b"defaults",
            _stable_implementation_value(
                implementation.__defaults__,
                active_functions=active,
            ),
        )
        _update_digest_frame(
            digest,
            b"kwdefaults",
            _stable_implementation_value(
                implementation.__kwdefaults__,
                active_functions=active,
            ),
        )
        global_frames: list[bytes] = []
        for name in sorted(set(implementation.__code__.co_names)):
            if name not in implementation.__globals__:
                continue
            encoded_name = name.encode("utf-8")
            encoded_value = _stable_global_binding(
                implementation.__globals__[name],
                active_functions=active,
            )
            global_frames.append(
                len(encoded_name).to_bytes(4, "big")
                + encoded_name
                + len(encoded_value).to_bytes(8, "big")
                + encoded_value
            )
        _update_digest_frame(
            digest,
            b"referenced_globals",
            b"".join(global_frames),
        )
        return digest.digest()
    finally:
        active.remove(marker)


def _clone_function(
    implementation: object,
    *,
    memo: dict[int, FunctionType] | None = None,
) -> Callable[..., AdapterDecision]:
    if not isinstance(implementation, FunctionType):
        raise ValueError("trusted adapter replay must be a Python function")
    if implementation.__closure__:
        raise ValueError(
            "trusted adapter replay/helper closures are unsupported; "
            "use measured module-level functions and immutable arguments"
        )
    clones = {} if memo is None else memo
    existing = clones.get(id(implementation))
    if existing is not None:
        return existing
    raw_builtins = implementation.__globals__.get("__builtins__", {})
    if isinstance(raw_builtins, ModuleType):
        frozen_builtins = dict(vars(raw_builtins))
    elif isinstance(raw_builtins, Mapping):
        frozen_builtins = dict(raw_builtins)
    else:
        raise ValueError("trusted adapter globals contain invalid __builtins__")
    frozen_globals: dict[str, object] = {"__builtins__": frozen_builtins}
    frozen_defaults = _freeze_runtime_value(
        implementation.__defaults__,
        memo=clones,
    )
    clone = FunctionType(
        implementation.__code__,
        frozen_globals,
        name=implementation.__name__,
        argdefs=frozen_defaults,
        closure=None,
    )
    clones[id(implementation)] = clone
    clone.__kwdefaults__ = (
        None
        if implementation.__kwdefaults__ is None
        else _freeze_runtime_value(implementation.__kwdefaults__, memo=clones)
    )
    for name in sorted(set(implementation.__code__.co_names)):
        if name in implementation.__globals__:
            frozen_globals[name] = _freeze_global_binding(
                implementation.__globals__[name],
                memo=clones,
            )
    return clone


def _referenced_globals(
    implementation: object,
) -> tuple[tuple[str, object], ...]:
    if not isinstance(implementation, FunctionType):
        raise ValueError("trusted adapter replay must be a Python function")
    return tuple(
        (name, implementation.__globals__[name])
        for name in sorted(set(implementation.__code__.co_names))
        if name in implementation.__globals__
    )


def _stable_global_binding(
    value: object,
    *,
    active_functions: set[int],
) -> bytes:
    """Fingerprint behavior reachable through one function global.

    A function object is mutable in place: checking only its identity leaves a
    registered adapter vulnerable to a later ``helper.__code__`` swap.  Recurse
    through Python helpers and measure their defaults and referenced globals as
    part of the adapter trust root.  Modules and arbitrary Python classes are
    deliberately rejected because they expose a large mutable namespace that
    cannot be pinned by this in-process boundary.
    """

    if isinstance(value, FunctionType):
        return b"P" + _callable_implementation_bytes(value, active=active_functions)
    if isinstance(value, BuiltinFunctionType):
        return (
            b"U"
            + str(getattr(value, "__module__", "builtins")).encode("utf-8")
            + b"\x00"
            + str(getattr(value, "__qualname__", value.__name__)).encode("utf-8")
        )
    if isinstance(value, ModuleType):
        raise ValueError(
            "trusted adapter replay/helpers must import the exact function or "
            "immutable value, not a mutable module namespace"
        )
    if isinstance(value, type):
        return _stable_trusted_type(value)
    return b"V" + _stable_implementation_value(
        value,
        active_functions=active_functions,
    )


def _stable_trusted_type(value: type) -> bytes:
    if value.__module__ != "builtins" and value not in _TRUSTED_PYTHON_GLOBAL_TYPES:
        raise ValueError(
            "trusted adapter replay/helpers reference an unsupported mutable "
            f"Python type: {value.__module__}.{value.__qualname__}"
        )
    return (
        b"T"
        + value.__module__.encode("utf-8")
        + b"\x00"
        + value.__qualname__.encode("utf-8")
    )


def _freeze_global_binding(
    value: object,
    *,
    memo: dict[int, FunctionType],
) -> object:
    """Detach a registered function global from later caller-side mutation."""

    if isinstance(value, FunctionType):
        return _clone_function(value, memo=memo)
    if isinstance(value, BuiltinFunctionType):
        return value
    if isinstance(value, ModuleType):
        raise ValueError(
            "trusted adapter replay/helpers must import the exact function or "
            "immutable value, not a mutable module namespace"
        )
    if isinstance(value, type):
        _stable_trusted_type(value)
        return value
    return _freeze_runtime_value(value, memo=memo)


def _freeze_runtime_value(
    value: object,
    *,
    memo: dict[int, FunctionType],
    active_values: set[int] | None = None,
) -> object:
    """Copy admitted defaults/constants so the execution clone owns its state."""

    if value is None or isinstance(value, (bool, int, float, str, bytes, CodeType)):
        return value
    if isinstance(value, FunctionType):
        return _clone_function(value, memo=memo)
    if isinstance(value, BuiltinFunctionType):
        return value
    if isinstance(value, ModuleType):
        raise ValueError("trusted adapter runtime values cannot contain modules")
    if isinstance(value, type):
        _stable_trusted_type(value)
        return value

    active = set() if active_values is None else active_values
    marker = id(value)
    if marker in active:
        raise ValueError("trusted adapter runtime values cannot contain cycles")
    active.add(marker)
    try:
        if isinstance(value, tuple):
            return tuple(
                _freeze_runtime_value(item, memo=memo, active_values=active)
                for item in value
            )
        if isinstance(value, list):
            return [
                _freeze_runtime_value(item, memo=memo, active_values=active)
                for item in value
            ]
        if isinstance(value, set):
            return {
                _freeze_runtime_value(item, memo=memo, active_values=active)
                for item in value
            }
        if isinstance(value, frozenset):
            return frozenset(
                _freeze_runtime_value(item, memo=memo, active_values=active)
                for item in value
            )
        if isinstance(value, Mapping):
            return {
                _freeze_runtime_value(key, memo=memo, active_values=active):
                _freeze_runtime_value(item, memo=memo, active_values=active)
                for key, item in value.items()
            }
    finally:
        active.remove(marker)
    raise ValueError(
        "trusted adapter runtime values contain unsupported mutable value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _update_digest_frame(digest, label: bytes, payload: bytes) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _stable_implementation_value(
    value: object,
    *,
    active_functions: set[int] | None = None,
) -> bytes:
    active = set() if active_functions is None else active_functions
    if value is None:
        return b"N"
    if value is True:
        return b"B1"
    if value is False:
        return b"B0"
    if isinstance(value, int):
        return b"I" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"F" + value.hex().encode("ascii")
    if isinstance(value, str):
        return b"S" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"Y" + value
    if isinstance(value, CodeType):
        return b"C" + _stable_code_bytes(value)
    if isinstance(value, type):
        return _stable_trusted_type(value)
    if isinstance(value, FunctionType):
        return b"P" + _callable_implementation_bytes(value, active=active)
    if isinstance(value, tuple):
        return _framed_values(b"Q", value, active_functions=active)
    if isinstance(value, list):
        return _framed_values(b"L", tuple(value), active_functions=active)
    if isinstance(value, (set, frozenset)):
        items = sorted(
            _stable_implementation_value(item, active_functions=active)
            for item in value
        )
        return _framed_values(
            b"E",
            tuple(items),
            preencoded=True,
            active_functions=active,
        )
    if isinstance(value, Mapping):
        items = sorted(
            (
                _stable_implementation_value(key, active_functions=active),
                _stable_implementation_value(item, active_functions=active),
            )
            for key, item in value.items()
        )
        flattened = tuple(part for pair in items for part in pair)
        return _framed_values(
            b"D",
            flattened,
            preencoded=True,
            active_functions=active,
        )
    raise ValueError(
        "trusted adapter replay defaults/closure contain unsupported mutable value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _stable_code_bytes(code: CodeType) -> bytes:
    """Serialize behavior-bearing code fields without runtime quickening state."""

    fields: tuple[tuple[bytes, object], ...] = (
        (b"argcount", code.co_argcount),
        (b"posonlyargcount", code.co_posonlyargcount),
        (b"kwonlyargcount", code.co_kwonlyargcount),
        (b"nlocals", code.co_nlocals),
        (b"stacksize", code.co_stacksize),
        (b"flags", code.co_flags),
        (b"code", code.co_code),
        (b"consts", code.co_consts),
        (b"names", code.co_names),
        (b"varnames", code.co_varnames),
        (b"freevars", code.co_freevars),
        (b"cellvars", code.co_cellvars),
        (b"exceptiontable", getattr(code, "co_exceptiontable", b"")),
    )
    payload = bytearray(b"decision_twin.python_code.v1")
    for label, value in fields:
        encoded = _stable_implementation_value(value)
        payload.extend(len(label).to_bytes(4, "big"))
        payload.extend(label)
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return bytes(payload)


def _framed_values(
    prefix: bytes,
    values: tuple[object, ...],
    *,
    preencoded: bool = False,
    active_functions: set[int] | None = None,
) -> bytes:
    payload = bytearray(prefix)
    payload.extend(len(values).to_bytes(8, "big"))
    for value in values:
        encoded = (
            value
            if preencoded
            else _stable_implementation_value(
                value,
                active_functions=active_functions,
            )
        )
        if not isinstance(encoded, bytes):
            raise ValueError("implementation fingerprint frame must be bytes")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return bytes(payload)


def _artifact_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ReplayMaterialRejected(
            f"{field} must be an immutable lowercase SHA-256 artifact id"
        )
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayMaterialRejected(f"{context} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing={','.join(missing)}")
        if extra:
            detail.append(f"extra={','.join(extra)}")
        raise ReplayMaterialRejected(
            f"{context} fields do not match schema ({'; '.join(detail)})"
        )


def _schema(
    value: Mapping[str, Any],
    expected: str,
    context: str,
) -> None:
    if value.get("schema_version") != expected:
        raise ReplayMaterialRejected(f"{context} schema version is unsupported")


def _immutable_bytes(value: object, context: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{context} must be non-empty immutable bytes")
    return value
