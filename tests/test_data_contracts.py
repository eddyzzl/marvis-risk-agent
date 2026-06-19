from dataclasses import asdict

import marvis.data as data_contracts
from marvis.data import (
    DATE_FORMATS,
    HASH_ALGO_CANDIDATES,
    HASH_HEX_LENGTHS,
    LARGE_ROW_THRESHOLD,
    MIN_KEY_MATCH_RATE,
    SHRINK_WARN_THRESHOLD,
    SMALL_SAMPLE_N,
    ColumnFingerprint,
    ColumnProfile,
    Dataset,
    JoinDiagnostics,
    JoinPlan,
    JoinSpec,
    KeyPair,
)


def test_dataset_contract_round_trips_to_structured_dict():
    fingerprint = ColumnFingerprint(
        value_kind="hash",
        length_mode=64,
        regex_pattern=r"^[0-9a-f]{64}$",
        is_hashed=True,
        hash_type="sha256",
        hex_case="lower",
        date_format=None,
    )
    profile = ColumnProfile(
        name="customer_id_hash",
        dtype="string",
        semantic_role="id",
        fingerprint=fingerprint,
        null_rate=0.0,
        cardinality=2,
        sample_values=("a***", "b***"),
    )
    dataset = Dataset(
        id="ds-1",
        task_id="task-1",
        role="sample",
        source_path="datasets/sample.parquet",
        format="parquet",
        sheet=None,
        row_count=2,
        columns=(profile,),
        has_target=True,
        target_col="bad",
        created_at="2026-06-19T00:00:00Z",
    )

    payload = asdict(dataset)

    assert payload["id"] == "ds-1"
    assert payload["columns"][0]["fingerprint"]["hash_type"] == "sha256"
    assert payload["columns"][0]["sample_values"] == ("a***", "b***")


def test_join_plan_defaults_and_nested_contracts():
    key_pair = KeyPair(
        anchor_col="customer_id",
        feature_col="customer_id_hash",
        match_method="hash:md5",
        transform_side="anchor",
        match_rate=0.91,
        resolved_by="empirical",
    )
    diagnostics = JoinDiagnostics(
        anchor_rows=100,
        feature_rows=90,
        feature_key_unique=True,
        matched_rows=91,
        match_rate=0.91,
        joined_rows_preview=100,
        fan_out_detected=False,
        shrink_detected=False,
        new_columns=3,
        new_columns_null_rate=0.05,
    )
    join_spec = JoinSpec(
        feature_dataset_id="feature-1",
        key_pairs=[key_pair],
        diagnostics=diagnostics,
        dedup_strategy=None,
    )
    plan = JoinPlan(
        id="join-1",
        task_id="task-1",
        anchor_dataset_id="anchor-1",
        joins=[join_spec],
        status="draft",
    )

    assert join_spec.confirmed is False
    assert plan.result_dataset_id is None
    assert asdict(plan)["joins"][0]["key_pairs"][0]["match_method"] == "hash:md5"


def test_hash_constants_cover_supported_sha_family_and_priority():
    assert HASH_HEX_LENGTHS[32] == "md5"
    assert HASH_HEX_LENGTHS[40] == "sha1"
    assert HASH_HEX_LENGTHS[56] == "sha224"
    assert HASH_HEX_LENGTHS[64] == "sha256"
    assert HASH_HEX_LENGTHS[96] == "sha384"
    assert HASH_HEX_LENGTHS[128] == "sha512"
    assert HASH_ALGO_CANDIDATES[:2] == ("md5", "sha256")
    assert set(HASH_ALGO_CANDIDATES) == {"md5", "sha1", "sha256", "sha512"}


def test_data_layer_thresholds_and_date_formats_are_stable():
    assert SHRINK_WARN_THRESHOLD == 0.5
    assert MIN_KEY_MATCH_RATE == 0.5
    assert SMALL_SAMPLE_N == 5000
    assert LARGE_ROW_THRESHOLD == 200_000
    assert "%Y%m%d" in DATE_FORMATS
    assert "%Y-%m-%d %H:%M:%S" in DATE_FORMATS


def test_package_exports_contract_surface():
    assert data_contracts.Dataset is Dataset
    assert data_contracts.JoinPlan is JoinPlan
    assert "HASH_HEX_LENGTHS" in data_contracts.__all__
