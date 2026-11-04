import hashlib

import pandas as pd

from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset
from marvis.data.schema_infer import infer_dataset_schema


def _dataset(dataset_id: str, frame: pd.DataFrame, source_path: str) -> Dataset:
    profiles = tuple(infer_dataset_schema(frame))
    return Dataset(
        id=dataset_id,
        task_id="task-1",
        role="sample",
        source_path=source_path,
        format="csv",
        sheet=None,
        row_count=len(frame),
        columns=profiles,
        has_target=False,
        target_col=None,
        created_at="2026-06-19T00:00:00Z",
    )


def _write_csv(tmp_path, name: str, frame: pd.DataFrame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path, _dataset(name, frame, name)


def test_align_resolves_raw_phone_to_md5_hash_by_empirical_match(tmp_path):
    anchor_frame = pd.DataFrame({"mobile": ["13800138000", "13900139000"]})
    feature_frame = pd.DataFrame({
        "phone_md5": [
            hashlib.md5(value.encode()).hexdigest()
            for value in anchor_frame["mobile"]
        ],
    })
    anchor_path, anchor = _write_csv(tmp_path, "anchor.csv", anchor_frame)
    feature_path, feature = _write_csv(tmp_path, "feature.csv", feature_frame)

    pairs = ColumnAligner(DataBackend(tmp_path)).align(anchor, anchor_path, feature, feature_path)

    assert len(pairs) == 1
    assert pairs[0].anchor_col == "mobile"
    assert pairs[0].feature_col == "phone_md5"
    assert pairs[0].match_method == "hash:md5"
    assert pairs[0].transform_side == "anchor"
    assert pairs[0].match_rate == 1.0
    assert pairs[0].resolved_by == "empirical"


def test_align_selects_sha256_when_md5_candidate_does_not_match(tmp_path):
    anchor_frame = pd.DataFrame({"mobile": ["13800138000", "13900139000"]})
    feature_frame = pd.DataFrame({
        "phone_sha256": [
            hashlib.sha256(value.encode()).hexdigest()
            for value in anchor_frame["mobile"]
        ],
    })
    anchor_path, anchor = _write_csv(tmp_path, "anchor.csv", anchor_frame)
    feature_path, feature = _write_csv(tmp_path, "feature.csv", feature_frame)

    pairs = ColumnAligner(DataBackend(tmp_path)).align(anchor, anchor_path, feature, feature_path)

    assert len(pairs) == 1
    assert pairs[0].match_method == "hash:sha256"
    assert pairs[0].transform_side == "anchor"
    assert pairs[0].match_rate == 1.0


def test_align_normalizes_hash_case_and_date_formats(tmp_path):
    raw_values = ["13800138000", "13900139000"]
    anchor_hash = pd.DataFrame({
        "phone_md5": [hashlib.md5(value.encode()).hexdigest() for value in raw_values],
    })
    feature_hash = pd.DataFrame({
        "phone_md5": [
            hashlib.md5(value.encode()).hexdigest().upper()
            for value in raw_values
        ],
    })
    anchor_hash_path, anchor_hash_ds = _write_csv(tmp_path, "anchor_hash.csv", anchor_hash)
    feature_hash_path, feature_hash_ds = _write_csv(tmp_path, "feature_hash.csv", feature_hash)

    hash_pairs = ColumnAligner(DataBackend(tmp_path)).align(
        anchor_hash_ds,
        anchor_hash_path,
        feature_hash_ds,
        feature_hash_path,
    )

    assert hash_pairs[0].match_method == "exact_lower"
    assert hash_pairs[0].transform_side == "both"
    assert hash_pairs[0].match_rate == 1.0

    anchor_dates = pd.DataFrame({"applydate": ["20260101", "20260102"]})
    feature_dates = pd.DataFrame({"huisudate": ["2026-01-01", "2026-01-02"]})
    anchor_date_path, anchor_date_ds = _write_csv(tmp_path, "anchor_date.csv", anchor_dates)
    feature_date_path, feature_date_ds = _write_csv(tmp_path, "feature_date.csv", feature_dates)

    date_pairs = ColumnAligner(DataBackend(tmp_path)).align(
        anchor_date_ds,
        anchor_date_path,
        feature_date_ds,
        feature_date_path,
    )

    assert date_pairs[0].match_method == "date"
    assert date_pairs[0].transform_side == "both"
    assert date_pairs[0].match_rate == 1.0


def test_align_rejects_name_match_when_data_does_not_match(tmp_path):
    anchor_frame = pd.DataFrame({"mobile": ["13800138000", "13900139000"]})
    feature_frame = pd.DataFrame({
        "phone_md5": [
            hashlib.md5(value.encode()).hexdigest()
            for value in ["13600136000", "13700137000"]
        ],
    })
    anchor_path, anchor = _write_csv(tmp_path, "anchor.csv", anchor_frame)
    feature_path, feature = _write_csv(tmp_path, "feature.csv", feature_frame)

    pairs = ColumnAligner(DataBackend(tmp_path)).align(anchor, anchor_path, feature, feature_path)

    assert pairs == []


def test_align_uses_fuzzy_fallback_only_after_dictionary_miss(tmp_path):
    anchor_frame = pd.DataFrame({"customer_key": ["A1", "B2", "C3"]})
    feature_frame = pd.DataFrame({"customerkey": ["A1", "B2", "C3"]})
    anchor_path, anchor = _write_csv(tmp_path, "anchor.csv", anchor_frame)
    feature_path, feature = _write_csv(tmp_path, "feature.csv", feature_frame)

    pairs = ColumnAligner(DataBackend(tmp_path)).align(anchor, anchor_path, feature, feature_path)

    assert len(pairs) == 1
    assert pairs[0].anchor_col == "customer_key"
    assert pairs[0].feature_col == "customerkey"
    assert pairs[0].match_method == "exact"
    assert pairs[0].resolved_by == "fuzzy"


# --- customer-id key precedence (eval S6: 申请表 + 征信特征表按客户关联) -------------------


def _md5(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def _s6_frames(n: int = 40):
    """Application table keyed by cust_id + raw mobile; bureau table keyed by the same cust_id
    + md5(mobile) — both identity keys match 100%, the customer id is the raw=raw one."""
    cust_ids = [f"C{i:06d}" for i in range(1, n + 1)]
    phones = [f"138{i:08d}" for i in range(n)]
    applications = pd.DataFrame({
        "cust_id": cust_ids,
        "mobile": phones,
        "bad_flag": [i % 2 for i in range(n)],
    })
    bureau = pd.DataFrame({
        "cust_id": cust_ids,
        "mobile_md5": [_md5(phone) for phone in phones],
        "score": list(range(n)),
    })
    return applications, bureau


def _propose(tmp_path, anchor_frame, feature_frame):
    anchor_path, anchor = _write_csv(tmp_path, "anchor.csv", anchor_frame)
    feature_path, feature = _write_csv(tmp_path, "feature.csv", feature_frame)
    return ColumnAligner(DataBackend(tmp_path)).propose_keys(anchor, anchor_path, feature, feature_path)


def _cols(pairs):
    return [(pair.anchor_col, pair.feature_col) for pair in pairs]


def test_align_prefers_exact_unique_customer_id_over_hashed_phone(tmp_path):
    applications, bureau = _s6_frames()

    proposal = _propose(tmp_path, applications, bureau)

    # the raw=raw unique customer id is the SOLE key — not composed with the hashed phone
    assert _cols(proposal.key_pairs) == [("cust_id", "cust_id")]
    assert proposal.key_pairs[0].match_method == "exact"
    assert proposal.key_pairs[0].match_rate == 1.0
    assert proposal.choice == "customer_id"
    # the hashed phone stays a candidate the user can switch to
    assert ("mobile", "mobile_md5") in _cols(proposal.candidates)
    assert [_cols(key) for key in proposal.alternatives] == [[("mobile", "mobile_md5")]]


def test_align_keeps_hashed_phone_key_when_there_is_no_customer_id(tmp_path):
    applications, bureau = _s6_frames()

    proposal = _propose(tmp_path, applications.drop(columns=["cust_id"]), bureau.drop(columns=["cust_id"]))

    assert _cols(proposal.key_pairs) == [("mobile", "mobile_md5")]
    assert proposal.key_pairs[0].match_method == "hash:md5"
    assert proposal.choice == "dictionary"
    assert proposal.alternatives == ()


def test_align_keeps_phone_plus_date_composite_without_customer_id(tmp_path):
    phones = ["13800138000", "13900139000"]
    anchor_frame = pd.DataFrame({"mobile": phones, "apply_date": ["20260101", "20260102"]})
    feature_frame = pd.DataFrame({
        "phone_md5": [_md5(phone) for phone in phones],
        "create_date": ["2026-01-01", "2026-01-02"],
    })

    proposal = _propose(tmp_path, anchor_frame, feature_frame)

    assert _cols(proposal.key_pairs) == [("apply_date", "create_date"), ("mobile", "phone_md5")]
    assert proposal.choice == "dictionary"
    assert proposal.alternatives == ()


def test_align_keeps_identity_key_when_customer_id_is_not_unique_on_feature_side(tmp_path):
    applications, bureau = _s6_frames()
    # one extra bureau row reuses C000001 for a different phone: cust_id is no longer unique
    extra = pd.DataFrame({"cust_id": ["C000001"], "mobile_md5": [_md5("13700000000")], "score": [-1]})

    proposal = _propose(tmp_path, applications, pd.concat([bureau, extra], ignore_index=True))

    assert _cols(proposal.key_pairs) == [("mobile", "mobile_md5")]
    assert proposal.choice == "customer_id_not_unique"
    assert [("cust_id", "cust_id")] in [_cols(key) for key in proposal.alternatives]


def test_align_keeps_identity_key_when_customer_id_matches_worse(tmp_path):
    applications, bureau = _s6_frames()
    # 30% of bureau customer numbers come from another system → cust_id matches only 70%
    bureau["cust_id"] = [f"X{i:06d}" if i < 12 else value for i, value in enumerate(bureau["cust_id"])]

    proposal = _propose(tmp_path, applications, bureau)

    assert _cols(proposal.key_pairs) == [("mobile", "mobile_md5")]
    assert proposal.choice == "customer_id_lower_match"
    assert [("cust_id", "cust_id")] in [_cols(key) for key in proposal.alternatives]


def test_align_composes_non_unique_customer_id_with_date_when_no_other_identity_key(tmp_path):
    # a monthly snapshot table: one bureau row per customer per date
    anchor_frame = pd.DataFrame({
        "cust_id": ["C1", "C2", "C3"],
        "apply_date": ["20260101", "20260102", "20260103"],
    })
    feature_frame = pd.DataFrame({
        "cust_id": ["C1", "C1", "C2", "C2", "C3", "C3"],
        "apply_date": ["20251201", "20260101", "20251201", "20260102", "20251201", "20260103"],
        "score": [1, 2, 3, 4, 5, 6],
    })

    proposal = _propose(tmp_path, anchor_frame, feature_frame)

    assert _cols(proposal.key_pairs) == [("apply_date", "apply_date"), ("cust_id", "cust_id")]
    assert proposal.choice == "customer_id_composite"


def test_customer_id_family_matches_whole_name_tokens_only():
    from marvis.data.align import is_customer_id_column

    for name in ("cust_id", "APP_CUST_ID", "CustID", "customer_no", "user_id_md5", "member-id"):
        assert is_customer_id_column(name), name
    for name in ("cust_idcard", "user_identity", "uid", "id", "cust_name", "customer_key", "acct_id"):
        assert not is_customer_id_column(name), name


def test_low_cardinality_fuzzy_column_never_outranks_a_unique_customer_id(tmp_path):
    # Review H2: a 3-value `channel` overlaps 100% by value; it outranked a raw,
    # unique cust_id that covers 90% of customers and was composed into the key.
    n = 40
    anchor_frame = pd.DataFrame({
        "cust_id": [f"C{i:06d}" for i in range(n)],
        "channel": [("app", "web", "partner")[i % 3] for i in range(n)],
        "bad_flag": [i % 2 for i in range(n)],
    })
    feature_frame = pd.DataFrame({
        "cust_id": [f"C{i:06d}" if i < 36 else f"X{i:06d}" for i in range(n)],
        "channel": [("app", "web", "partner")[i % 3] for i in range(n)],
        "score": list(range(n)),
    })

    proposal = _propose(tmp_path, anchor_frame, feature_frame)

    assert _cols(proposal.key_pairs) == [("cust_id", "cust_id")]
    assert proposal.choice == "customer_id"


def test_customer_id_is_compared_with_the_key_that_would_run(tmp_path):
    # Review N1: the phone alone matches 100%, but the key that would run is
    # phone + apply_date at 70%, worse than the unique 80% customer id.
    applications, bureau = _s6_frames(40)
    applications["apply_date"] = [f"2026-01-{(i % 28) + 1:02d}" for i in range(40)]
    bureau["apply_date"] = [
        f"2026-02-{(i % 28) + 1:02d}" if i % 10 < 3 else value
        for i, value in enumerate(applications["apply_date"])
    ]
    bureau["cust_id"] = [f"X{i:06d}" if i % 5 == 0 else value for i, value in enumerate(bureau["cust_id"])]

    proposal = _propose(tmp_path, applications, bureau)

    assert _cols(proposal.key_pairs) == [("cust_id", "cust_id")]
    assert proposal.choice == "customer_id"
    alternatives = [_cols(key) for key in proposal.alternatives]
    assert [("mobile", "mobile_md5")] in alternatives
