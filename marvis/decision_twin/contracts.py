from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from marvis.decision_twin._canonical import (
    content_hash,
    iso_z,
    parse_datetime,
    required_text,
    utc_datetime,
)


REPLAY_MANIFEST_SCHEMA_VERSION = "decision_twin.replay_manifest.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VersionedArtifact:
    """One immutable, content-addressed input visible at replay time."""

    version: str
    sha256: str
    visible_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", required_text(self.version, "version"))
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")
        object.__setattr__(
            self,
            "visible_at",
            utc_datetime(self.visible_at, "visible_at"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "sha256": self.sha256,
            "visible_at": iso_z(self.visible_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VersionedArtifact:
        if not isinstance(payload, Mapping):
            raise ValueError("versioned artifact must be an object")
        version = payload.get("version")
        if not isinstance(version, str):
            raise ValueError("version must be a non-empty string")
        return cls(
            version=version,
            sha256=str(payload.get("sha256", "")),
            visible_at=parse_datetime(payload.get("visible_at"), "visible_at"),
        )


@dataclass(frozen=True)
class ReplayManifest:
    """Complete point-in-time binding required for a trustworthy replay."""

    REQUIRED_BINDINGS: ClassVar[tuple[str, ...]] = (
        "data",
        "dictionary",
        "label",
        "preprocessing",
        "model",
        "strategy",
        "limit_pricing",
        "manual_override",
    )

    as_of: datetime
    issued_at: datetime
    data: VersionedArtifact
    dictionary: VersionedArtifact
    label: VersionedArtifact
    preprocessing: VersionedArtifact
    model: VersionedArtifact
    strategy: VersionedArtifact
    limit_pricing: VersionedArtifact
    manual_override: VersionedArtifact
    schema_version: str = REPLAY_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay manifest schema: {self.schema_version}"
            )
        as_of = utc_datetime(self.as_of, "as_of")
        issued_at = utc_datetime(self.issued_at, "issued_at")
        if as_of > issued_at:
            raise ValueError("as_of must not be after issued_at")
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "issued_at", issued_at)
        for name, binding in self.bindings.items():
            if not isinstance(binding, VersionedArtifact):
                raise ValueError(f"{name} must be a VersionedArtifact")
            if binding.visible_at > as_of:
                raise ValueError(f"{name} is visible after replay as_of")

    @property
    def bindings(self) -> Mapping[str, VersionedArtifact]:
        return MappingProxyType(
            {name: getattr(self, name) for name in self.REQUIRED_BINDINGS}
        )

    @property
    def manifest_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of": iso_z(self.as_of),
            "issued_at": iso_z(self.issued_at),
            "bindings": {
                name: binding.to_dict() for name, binding in self.bindings.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayManifest:
        if not isinstance(payload, Mapping):
            raise ValueError("replay manifest must be an object")
        raw_bindings = payload.get("bindings")
        if not isinstance(raw_bindings, Mapping):
            raise ValueError("bindings must be an object")
        missing = [name for name in cls.REQUIRED_BINDINGS if name not in raw_bindings]
        if missing:
            raise ValueError(f"missing required binding versions: {', '.join(missing)}")
        bindings = {
            name: VersionedArtifact.from_dict(raw_bindings[name])
            for name in cls.REQUIRED_BINDINGS
        }
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            as_of=parse_datetime(payload.get("as_of"), "as_of"),
            issued_at=parse_datetime(payload.get("issued_at"), "issued_at"),
            **bindings,
        )
