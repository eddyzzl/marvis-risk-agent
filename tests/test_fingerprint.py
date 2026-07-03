import hashlib

import pandas as pd

from marvis.data.contracts import ColumnFingerprint
from marvis.data.fingerprint import candidate_match_methods, fingerprint_column


def _hash_values(algorithm: str, *, upper: bool = False) -> list[str]:
    values = [
        hashlib.new(algorithm, value.encode("utf-8")).hexdigest()
        for value in ["A1", "B2", "C3"]
    ]
    return [value.upper() for value in values] if upper else values


def test_fingerprint_detects_hash_families_and_hex_case():
    assert fingerprint_column(pd.Series(_hash_values("md5"))).hash_type == "md5"
    assert fingerprint_column(pd.Series(_hash_values("sha1"))).hash_type == "sha1"
    assert fingerprint_column(pd.Series(_hash_values("sha256"))).hash_type == "sha256"
    sha512_fp = fingerprint_column(pd.Series(_hash_values("sha512", upper=True)))

    assert sha512_fp.value_kind == "hash"
    assert sha512_fp.hash_type == "sha512"
    assert sha512_fp.hex_case == "upper"
    assert sha512_fp.is_hashed is True


def test_fingerprint_detects_raw_identifiers_dates_and_numeric_values():
    phone_fp = fingerprint_column(pd.Series(["13800138000", "13900139000"]))
    idcard_fp = fingerprint_column(pd.Series(["11010119900101001X", "110101199001010028"]))
    date_compact = fingerprint_column(pd.Series(["20260101", "20260102"]))
    date_dash = fingerprint_column(pd.Series(["2026-01-01", "2026-01-02"]))
    date_slash = fingerprint_column(pd.Series(["2026/01/01", "2026/01/02"]))
    datetime_fp = fingerprint_column(pd.to_datetime(pd.Series(["2026-01-01", "2026-01-02"])))
    numeric_fp = fingerprint_column(pd.Series([1.2, 3.4, 5.6]))

    assert phone_fp.value_kind == "raw_phone"
    assert phone_fp.is_hashed is False
    assert idcard_fp.value_kind == "raw_idcard"
    assert date_compact.date_format == "%Y%m%d"
    assert date_dash.date_format == "%Y-%m-%d"
    assert date_slash.date_format == "%Y/%m/%d"
    assert datetime_fp.date_format == "datetime"
    assert numeric_fp.value_kind == "numeric"


def test_candidate_match_methods_prioritize_known_hash_and_dates():
    raw_phone = ColumnFingerprint("raw_phone", 11, None, False, None, None, None)
    md5_hash = ColumnFingerprint("hash", 32, None, True, "md5", "lower", None)
    sha256_hash = ColumnFingerprint("hash", 64, None, True, "sha256", "lower", None)
    date_a = ColumnFingerprint("date", None, None, False, None, None, "%Y%m%d")
    date_b = ColumnFingerprint("date", None, None, False, None, None, "%Y-%m-%d")

    assert candidate_match_methods(raw_phone, md5_hash) == [
        "hash:md5",
        "hash:sha256",
        "hash:sha1",
        "hash:sha512",
    ]
    assert candidate_match_methods(md5_hash, sha256_hash) == []
    assert candidate_match_methods(date_a, date_b) == ["date"]
    assert candidate_match_methods(raw_phone, raw_phone) == ["exact", "exact_lower"]
