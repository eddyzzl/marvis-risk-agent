"""Canonical strategy-asset lifecycle with legacy wire compatibility.

``strategies.status`` remains the V1/V2 compatibility field.  New code owns
the more precise ``asset_status`` field and must keep both values consistent.
There is deliberately no ``deployed`` state: local adoption is not evidence of
production deployment.
"""

from __future__ import annotations


LEGACY_STATUS_DRAFT = "draft"
LEGACY_STATUS_ADOPTED = "adopted"
LEGACY_STATUS_RETIRED = "retired"

ASSET_STATUS_DRAFT = "draft"
ASSET_STATUS_VALIDATED = "validated"
ASSET_STATUS_ADOPTED_LOCAL = "adopted_local"
ASSET_STATUS_RETIRED = "retired"

LEGACY_STATUSES = frozenset(
    {LEGACY_STATUS_DRAFT, LEGACY_STATUS_ADOPTED, LEGACY_STATUS_RETIRED}
)
ASSET_STATUSES = frozenset(
    {
        ASSET_STATUS_DRAFT,
        ASSET_STATUS_VALIDATED,
        ASSET_STATUS_ADOPTED_LOCAL,
        ASSET_STATUS_RETIRED,
    }
)

_ASSET_FROM_LEGACY = {
    LEGACY_STATUS_DRAFT: ASSET_STATUS_DRAFT,
    LEGACY_STATUS_ADOPTED: ASSET_STATUS_ADOPTED_LOCAL,
    LEGACY_STATUS_RETIRED: ASSET_STATUS_RETIRED,
}
_LEGACY_FROM_ASSET = {
    ASSET_STATUS_DRAFT: LEGACY_STATUS_DRAFT,
    ASSET_STATUS_VALIDATED: LEGACY_STATUS_DRAFT,
    ASSET_STATUS_ADOPTED_LOCAL: LEGACY_STATUS_ADOPTED,
    ASSET_STATUS_RETIRED: LEGACY_STATUS_RETIRED,
}


class StrategyLifecycleError(ValueError):
    """Stored or requested lifecycle state is unknown or internally drifting."""


def asset_status_from_legacy(legacy_status: str | None) -> str:
    """Map a legacy status when reading a row/target created before migration 9."""

    status = _strict_status(legacy_status)
    try:
        return _ASSET_FROM_LEGACY[status]
    except KeyError as exc:
        raise StrategyLifecycleError(
            f"unknown legacy strategy status: {status!r}"
        ) from exc


def legacy_status_from_asset(asset_status: str | None) -> str:
    """Return the compatibility status that must be dual-written with an asset."""

    status = _strict_status(asset_status)
    try:
        return _LEGACY_FROM_ASSET[status]
    except KeyError as exc:
        raise StrategyLifecycleError(
            f"unknown canonical strategy asset_status: {status!r}"
        ) from exc


def validate_lifecycle_pair(legacy_status: str | None, asset_status: str | None) -> None:
    """Reject unknown or inconsistent dual-written lifecycle values."""

    legacy = _strict_status(legacy_status)
    asset = _strict_status(asset_status)
    # Validate each side independently so unknown values never look like drift.
    asset_status_from_legacy(legacy)
    expected_legacy = legacy_status_from_asset(asset)
    if legacy != expected_legacy:
        raise StrategyLifecycleError(
            "strategy lifecycle drift: "
            f"legacy status {legacy!r} is incompatible with asset_status {asset!r}"
        )


def resolve_asset_status(
    legacy_status: str | None,
    asset_status: str | None,
) -> str:
    """Resolve an old missing canonical value, otherwise validate the new pair."""

    if asset_status is None or not str(asset_status).strip():
        return asset_status_from_legacy(legacy_status)
    validate_lifecycle_pair(legacy_status, asset_status)
    return str(asset_status)


def is_locally_adopted(
    legacy_status: str | None,
    asset_status: str | None,
) -> bool:
    """Return whether the asset is the locally adopted strategy champion."""

    return (
        resolve_asset_status(legacy_status, asset_status)
        == ASSET_STATUS_ADOPTED_LOCAL
    )


def _strict_status(value: str | None) -> str:
    if not isinstance(value, str) or not value:
        raise StrategyLifecycleError(f"strategy lifecycle status is required: {value!r}")
    return value


__all__ = [
    "ASSET_STATUSES",
    "ASSET_STATUS_ADOPTED_LOCAL",
    "ASSET_STATUS_DRAFT",
    "ASSET_STATUS_RETIRED",
    "ASSET_STATUS_VALIDATED",
    "LEGACY_STATUSES",
    "LEGACY_STATUS_ADOPTED",
    "LEGACY_STATUS_DRAFT",
    "LEGACY_STATUS_RETIRED",
    "StrategyLifecycleError",
    "asset_status_from_legacy",
    "is_locally_adopted",
    "legacy_status_from_asset",
    "resolve_asset_status",
    "validate_lifecycle_pair",
]
