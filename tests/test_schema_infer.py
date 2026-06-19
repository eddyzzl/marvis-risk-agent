import hashlib

import pandas as pd

from marvis.data.contracts import ColumnFingerprint, ColumnProfile
from marvis.data.schema_infer import (
    detect_semantic_role,
    detect_target_column,
    infer_column_profile,
    infer_dataset_schema,
)


def test_infer_dataset_schema_roles_and_desensitized_samples():
    frame = pd.DataFrame({
        "mobile": ["13800138000", "13900139000", None],
        "cert_no": ["11010119900101001X", "110101199001010028", None],
        "apply_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "bad_flag": [0, 1, 0],
        "model_score": [0.1, 0.8, 0.3],
        "loan_amount": [1000, 2000, 3000],
    })

    profiles = {profile.name: profile for profile in infer_dataset_schema(frame)}

    assert profiles["mobile"].semantic_role == "phone"
    assert profiles["cert_no"].semantic_role == "idcard"
    assert profiles["apply_date"].semantic_role == "date"
    assert profiles["bad_flag"].semantic_role == "target"
    assert profiles["model_score"].semantic_role == "score"
    assert profiles["loan_amount"].semantic_role == "amount"
    assert "13800138000" not in profiles["mobile"].sample_values
    assert profiles["mobile"].sample_values[0] == "138******00"
    assert "11010119900101001X" not in profiles["cert_no"].sample_values
    assert profiles["mobile"].null_rate == 1 / 3
    assert profiles["mobile"].cardinality == 2


def test_hash_columns_use_column_name_to_resolve_semantic_role():
    phone_hash = pd.Series([
        hashlib.md5(value.encode()).hexdigest()
        for value in ["13800138000", "13900139000"]
    ])
    id_hash = pd.Series([
        hashlib.sha256(value.encode()).hexdigest()
        for value in ["11010119900101001X", "110101199001010028"]
    ])

    phone_profile = infer_column_profile(phone_hash, "phone_md5")
    id_profile = infer_column_profile(id_hash, "idcard_sha256")
    anonymous_profile = infer_column_profile(phone_hash, "join_key")

    assert phone_profile.semantic_role == "phone"
    assert id_profile.semantic_role == "idcard"
    assert anonymous_profile.semantic_role == "id"
    assert phone_profile.sample_values[0] != phone_hash.iloc[0]
    assert anonymous_profile.sample_values[0].startswith(phone_hash.iloc[0][:4])


def test_detect_target_column_uses_role_then_binary_name_fallback():
    fp = ColumnFingerprint("categorical", None, None, False, None, None, None)
    role_target = ColumnProfile(
        name="approved_target",
        dtype="int64",
        semantic_role="target",
        fingerprint=fp,
        null_rate=0.0,
        cardinality=2,
        sample_values=(0, 1),
    )
    fallback = ColumnProfile(
        name="bad_outcome",
        dtype="int64",
        semantic_role="categorical",
        fingerprint=fp,
        null_rate=0.0,
        cardinality=2,
        sample_values=(0, 1),
    )
    frame = pd.DataFrame({"approved_target": [1, 0], "bad_outcome": [0, 1]})

    assert detect_target_column([role_target, fallback], frame) == "approved_target"
    assert detect_target_column([fallback], frame) == "bad_outcome"


def test_detect_semantic_role_avoids_y_substring_false_positive():
    fp = ColumnFingerprint("date", None, None, False, None, None, "%Y-%m-%d")
    numeric_fp = ColumnFingerprint("numeric", None, None, False, None, None, None)

    assert detect_semantic_role("day", fp) == "date"
    assert detect_semantic_role("yearly_income", numeric_fp) == "amount"
