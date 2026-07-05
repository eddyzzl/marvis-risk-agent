from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from rapidfuzz import fuzz

from marvis.data.backend import DataBackend
from marvis.data.contracts import (
    MIN_KEY_MATCH_RATE,
    SMALL_SAMPLE_N,
    ColumnProfile,
    Dataset,
    KeyPair,
)
from marvis.data.fingerprint import candidate_match_methods


FUZZY_NAME_THRESHOLD = 75

KEY_DICTIONARY = {
    "phone": [
        "phone",
        "mobile",
        "tel",
        "phone_no",
        "phone_md5",
        "mobile_md5",
        "tel_md5",
    ],
    "idcard": [
        "idcard",
        "idnumber",
        "id_no",
        "cert_no",
        "id_md5",
        "idcard_md5",
    ],
    "date": [
        "date",
        "applydate",
        "apply_date",
        "huisudate",
        "data_date",
        "dt",
        "create_date",
    ],
    # Person-name identity element (§4/§5): a COMPOSABLE key (phone+name, id_no+name+date),
    # never a sole key — names collide across people, so a name-only join fans out and is
    # caught by the 1:1 anchor assertion. Conservative compound keywords (no bare "name").
    # Chinese 姓名 columns are matched via their semantic_role == "name" (set by schema_infer),
    # NOT a keyword here — _normalized strips non-ASCII so a "姓名" keyword would match everything.
    "name": [
        "cust_name",
        "customer_name",
        "real_name",
        "full_name",
        "fullname",
        "applicant_name",
        "true_name",
    ],
}


class ColumnAligner:
    def __init__(self, backend: DataBackend):
        self._backend = backend

    def align(
        self,
        anchor: Dataset,
        anchor_path: Path,
        feature: Dataset,
        feature_path: Path,
        *,
        seed: int = 0,
    ) -> list[KeyPair]:
        pairs = []
        for family in ("phone", "idcard", "date", "name"):
            anchor_columns = self._family_columns(anchor.columns, family)
            feature_columns = self._family_columns(feature.columns, family)
            for anchor_column in anchor_columns:
                best = self._resolve_by_data(
                    anchor_column,
                    feature_columns,
                    anchor_path,
                    feature_path,
                    seed,
                    resolved_by="empirical",
                )
                if best is not None:
                    pairs.append(best)
        if not pairs:
            pairs.extend(self._fuzzy_resolve(anchor, anchor_path, feature, feature_path, seed))
        return _dedupe_keypairs(pairs)

    def _resolve_by_data(
        self,
        anchor_col: ColumnProfile,
        feature_cols: list[ColumnProfile],
        anchor_path: Path,
        feature_path: Path,
        seed: int,
        *,
        resolved_by: str,
    ) -> KeyPair | None:
        best: KeyPair | None = None
        for feature_col in feature_cols:
            methods = candidate_match_methods(anchor_col.fingerprint, feature_col.fingerprint)
            if not methods:
                continue
            # PERF-4: try every candidate method for this column pair in ONE batched
            # DuckDB call (one feature-table scan shared across methods) instead of one
            # match_rate_for_method call -- and therefore one feature scan -- per method.
            fingerprint = _pair_fp(anchor_col, feature_col)
            rates = self._backend.match_rates_for_methods(
                anchor_path,
                anchor_col.name,
                feature_path,
                feature_col.name,
                methods=methods,
                key_fingerprints=[fingerprint] * len(methods),
                sample_n=SMALL_SAMPLE_N,
                seed=seed,
            )
            # T1-B8: record each side's stored dtype-family and whether they diverge across
            # files, so the diagnostics/gate can surface a precision-loss / silent-miss risk.
            anchor_family = _dtype_family(anchor_col)
            feature_family = _dtype_family(feature_col)
            dtype_divergent = anchor_family != feature_family
            for method, (matched, sampled) in zip(methods, rates):
                rate = matched / sampled if sampled else 0.0
                if rate < MIN_KEY_MATCH_RATE:
                    continue
                candidate = KeyPair(
                    anchor_col=anchor_col.name,
                    feature_col=feature_col.name,
                    match_method=method,
                    transform_side=_raw_side(anchor_col, feature_col, method),
                    match_rate=round(rate, 4),
                    resolved_by=resolved_by,
                    anchor_dtype=str(anchor_col.dtype),
                    feature_dtype=str(feature_col.dtype),
                    dtype_divergent=dtype_divergent,
                )
                if best is None or candidate.match_rate > best.match_rate:
                    best = candidate
        return best

    def _fuzzy_resolve(
        self,
        anchor: Dataset,
        anchor_path: Path,
        feature: Dataset,
        feature_path: Path,
        seed: int,
    ) -> list[KeyPair]:
        pairs = []
        for anchor_col in anchor.columns:
            candidates = [
                feature_col
                for feature_col in feature.columns
                if fuzz.ratio(_normalized(anchor_col.name), _normalized(feature_col.name))
                >= FUZZY_NAME_THRESHOLD
            ]
            best = self._resolve_by_data(
                anchor_col,
                candidates,
                anchor_path,
                feature_path,
                seed,
                resolved_by="empirical",
            )
            if best is not None:
                pairs.append(replace(best, resolved_by="fuzzy"))
        return pairs

    def _family_columns(
        self,
        columns: tuple[ColumnProfile, ...],
        family: str,
    ) -> list[ColumnProfile]:
        keywords = KEY_DICTIONARY[family]
        return [
            column
            for column in columns
            if column.semantic_role == family or _matches_dictionary(column.name, keywords)
        ]


def _dtype_family(profile: ColumnProfile) -> str:
    """T1-B8: coarse dtype family for a key column -- 'text' | 'float' | 'int' | 'date' |
    'other'. Divergence between families across the two files is what silently mis-matches a
    join key (a float64-stored id vs a string-stored id)."""
    return _dtype_family_from_str(profile.dtype)


def _dtype_family_from_str(dtype: str) -> str:
    dt = str(dtype).lower()
    if dt in {"object", "string", "str"} or dt.startswith("string"):
        return "text"
    if "float" in dt:
        return "float"
    if "datetime" in dt or dt.startswith("date"):
        return "date"
    if "int" in dt:
        return "int"
    return "other"


def _divergence_level(anchor_dtype: str, feature_dtype: str) -> str:
    """T1-B8: 'red' only for text<->float, the precision-loss / leading-zero-loss case that
    silently drops rows -- it forces confirmation. Every other family mismatch is lossless
    under the VARCHAR-cast join, so it is 'warn' (surfaced, not blocked)."""
    families = {_dtype_family_from_str(anchor_dtype), _dtype_family_from_str(feature_dtype)}
    if families == {"text", "float"}:
        return "red"
    return "warn"


def _raw_side(anchor_col: ColumnProfile, feature_col: ColumnProfile, method: str) -> str:
    if not method.startswith("hash:"):
        return "both"
    if anchor_col.fingerprint.is_hashed and not feature_col.fingerprint.is_hashed:
        return "feature"
    if feature_col.fingerprint.is_hashed and not anchor_col.fingerprint.is_hashed:
        return "anchor"
    return "both"


def _pair_fp(anchor_col: ColumnProfile, feature_col: ColumnProfile):
    return (anchor_col.fingerprint, feature_col.fingerprint)


def _dedupe_keypairs(pairs: list[KeyPair]) -> list[KeyPair]:
    selected: dict[tuple[str, str], KeyPair] = {}
    for pair in pairs:
        key = (pair.anchor_col, pair.feature_col)
        current = selected.get(key)
        if current is None or pair.match_rate > current.match_rate:
            selected[key] = pair
    return sorted(selected.values(), key=lambda item: (item.anchor_col, item.feature_col))


def _matches_dictionary(name: str, keywords: list[str]) -> bool:
    normalized = _normalized(name)
    return any(_normalized(keyword) in normalized for keyword in keywords)


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


__all__ = [
    "FUZZY_NAME_THRESHOLD",
    "KEY_DICTIONARY",
    "ColumnAligner",
    "_divergence_level",
    "_dtype_family",
]
