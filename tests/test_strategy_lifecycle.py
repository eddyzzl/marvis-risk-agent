import pytest

from marvis.strategy_lifecycle import (
    ASSET_STATUS_ADOPTED_LOCAL,
    ASSET_STATUS_DRAFT,
    ASSET_STATUS_RETIRED,
    ASSET_STATUS_VALIDATED,
    StrategyLifecycleError,
    asset_status_from_legacy,
    is_locally_adopted,
    legacy_status_from_asset,
    resolve_asset_status,
    validate_lifecycle_pair,
)


@pytest.mark.parametrize(
    ("legacy_status", "asset_status"),
    [
        ("draft", ASSET_STATUS_DRAFT),
        ("adopted", ASSET_STATUS_ADOPTED_LOCAL),
        ("retired", ASSET_STATUS_RETIRED),
    ],
)
def test_legacy_status_maps_to_canonical_asset_status(
    legacy_status: str,
    asset_status: str,
):
    assert asset_status_from_legacy(legacy_status) == asset_status


@pytest.mark.parametrize(
    ("asset_status", "legacy_status"),
    [
        (ASSET_STATUS_DRAFT, "draft"),
        (ASSET_STATUS_VALIDATED, "draft"),
        (ASSET_STATUS_ADOPTED_LOCAL, "adopted"),
        (ASSET_STATUS_RETIRED, "retired"),
    ],
)
def test_canonical_asset_status_maps_to_compatible_legacy_status(
    asset_status: str,
    legacy_status: str,
):
    assert legacy_status_from_asset(asset_status) == legacy_status


@pytest.mark.parametrize(
    ("legacy_status", "asset_status"),
    [
        ("draft", ASSET_STATUS_DRAFT),
        ("draft", ASSET_STATUS_VALIDATED),
        ("adopted", ASSET_STATUS_ADOPTED_LOCAL),
        ("retired", ASSET_STATUS_RETIRED),
    ],
)
def test_valid_lifecycle_pairs_are_accepted(
    legacy_status: str,
    asset_status: str,
):
    assert validate_lifecycle_pair(legacy_status, asset_status) is None


@pytest.mark.parametrize(
    ("legacy_status", "asset_status"),
    [
        ("draft", ASSET_STATUS_ADOPTED_LOCAL),
        ("adopted", ASSET_STATUS_VALIDATED),
        ("retired", ASSET_STATUS_DRAFT),
        ("production", ASSET_STATUS_ADOPTED_LOCAL),
        ("adopted", "deployed"),
    ],
)
def test_unknown_or_drifting_lifecycle_pairs_fail_closed(
    legacy_status: str,
    asset_status: str,
):
    with pytest.raises(StrategyLifecycleError):
        validate_lifecycle_pair(legacy_status, asset_status)


def test_missing_canonical_status_is_derived_only_for_legacy_rows():
    assert resolve_asset_status("adopted", None) == ASSET_STATUS_ADOPTED_LOCAL
    assert resolve_asset_status("draft", "") == ASSET_STATUS_DRAFT


def test_only_adopted_local_is_monitorable():
    assert is_locally_adopted("adopted", ASSET_STATUS_ADOPTED_LOCAL) is True
    assert is_locally_adopted("draft", ASSET_STATUS_VALIDATED) is False
    assert is_locally_adopted("draft", ASSET_STATUS_DRAFT) is False
    assert is_locally_adopted("retired", ASSET_STATUS_RETIRED) is False


@pytest.mark.parametrize("value", [None, "", "unknown", "deployed"])
def test_unknown_single_status_values_fail_closed(value: str | None):
    with pytest.raises(StrategyLifecycleError):
        asset_status_from_legacy(value)
    with pytest.raises(StrategyLifecycleError):
        legacy_status_from_asset(value)
