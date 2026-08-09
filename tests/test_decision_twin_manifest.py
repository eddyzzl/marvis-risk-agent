from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
import hashlib

import pytest

from marvis.decision_twin import ReplayManifest, VersionedArtifact


AS_OF = datetime(2026, 1, 31, tzinfo=UTC)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding(name: str, *, visible_at: datetime = AS_OF) -> VersionedArtifact:
    return VersionedArtifact(
        version=f"{name}.v1",
        sha256=_sha(f"{name}.v1"),
        visible_at=visible_at,
    )


def _manifest(**overrides: object) -> ReplayManifest:
    values: dict[str, object] = {
        "as_of": AS_OF,
        "issued_at": AS_OF + timedelta(days=1),
        "data": _binding("data"),
        "dictionary": _binding("dictionary"),
        "label": _binding("label"),
        "preprocessing": _binding("preprocessing"),
        "model": _binding("model"),
        "strategy": _binding("strategy"),
        "limit_pricing": _binding("limit_pricing"),
        "manual_override": _binding("manual_override"),
    }
    values.update(overrides)
    return ReplayManifest(**values)  # type: ignore[arg-type]


def test_manifest_immutably_binds_every_replay_version_and_hash() -> None:
    manifest = _manifest()

    assert tuple(manifest.bindings) == (
        "data",
        "dictionary",
        "label",
        "preprocessing",
        "model",
        "strategy",
        "limit_pricing",
        "manual_override",
    )
    assert len(manifest.manifest_hash) == 64
    assert manifest == ReplayManifest.from_dict(manifest.to_dict())
    with pytest.raises(FrozenInstanceError):
        manifest.as_of = AS_OF + timedelta(days=2)  # type: ignore[misc]


@pytest.mark.parametrize("field", tuple(ReplayManifest.REQUIRED_BINDINGS))
def test_manifest_fails_closed_when_a_required_version_or_hash_is_missing(
    field: str,
) -> None:
    with pytest.raises(ValueError, match="version"):
        _manifest(**{field: VersionedArtifact("", _sha(field), AS_OF)})
    with pytest.raises(ValueError, match="SHA-256"):
        _manifest(**{field: VersionedArtifact("v1", "", AS_OF)})


def test_manifest_rejects_time_travel_and_future_visible_bindings() -> None:
    with pytest.raises(ValueError, match="as_of must not be after issued_at"):
        _manifest(issued_at=AS_OF - timedelta(seconds=1))
    with pytest.raises(ValueError, match="visible after replay as_of"):
        _manifest(data=_binding("data", visible_at=AS_OF + timedelta(seconds=1)))
    with pytest.raises(ValueError, match="timezone-aware"):
        _manifest(as_of=AS_OF.replace(tzinfo=None))


def test_manifest_hash_changes_when_any_binding_changes() -> None:
    baseline = _manifest()
    changed = _manifest(model=VersionedArtifact("model.v2", _sha("model.v2"), AS_OF))

    assert changed.manifest_hash != baseline.manifest_hash


def test_manifest_deserialization_does_not_coerce_null_versions() -> None:
    payload = _manifest().to_dict()
    payload["bindings"]["data"]["version"] = None

    with pytest.raises(ValueError, match="version"):
        ReplayManifest.from_dict(payload)
