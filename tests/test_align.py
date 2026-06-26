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
