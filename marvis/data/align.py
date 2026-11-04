from __future__ import annotations

import re
from dataclasses import dataclass, replace
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
# One anchor column can match several feature columns (mobile = phone and mobile =
# phone_md5). The best one composes the proposed key as before; the next ones stay
# candidates, each diagnosed as its own switchable key. Capped to bound the diagnosis cost.
MAX_FEATURE_CANDIDATES_PER_ANCHOR = 3

KEY_DICTIONARY = {
    # Customer-id family: a business customer number shared by both systems. Matched on whole
    # underscore-delimited tokens (see _TOKEN_MATCH_FAMILIES) so "cust_id" never claims
    # "cust_idcard" and "user_id" never claims "user_identity"; no bare "id"/"uid"/"no".
    "customer_id": [
        "cust_id",
        "custid",
        "customer_id",
        "customerid",
        "cust_no",
        "custno",
        "customer_no",
        "customerno",
        "client_id",
        "clientid",
        "client_no",
        "member_id",
        "memberid",
        "member_no",
        "user_id",
        "userid",
    ],
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

# Families whose keywords must match whole name tokens rather than any substring.
_TOKEN_MATCH_FAMILIES = frozenset({"customer_id"})
# Families composed into the identity composite key (phone+date, idcard+name, ...). date and
# name are composable elements only: they never identify a customer on their own.
_COMPOSITE_FAMILIES = ("phone", "idcard", "date", "name")
_COMPOSABLE_ONLY_FAMILIES = frozenset({"date", "name"})


@dataclass(frozen=True)
class KeyProposal:
    """Key discovery result for one feature table (spec §4/§5 择键).

    ``key_pairs`` is the proposed key — the only key that can execute, and only after the
    user confirms it at C2. ``candidates`` holds every data-resolved pair; ``alternatives``
    are other whole keys (built from candidates) the user may switch to; ``choice`` records
    why ``key_pairs`` won (see JoinDiagnostics.key_choice)."""

    key_pairs: list[KeyPair]
    candidates: tuple[KeyPair, ...]
    alternatives: tuple[tuple[KeyPair, ...], ...]
    choice: str


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
        return self.propose_keys(anchor, anchor_path, feature, feature_path, seed=seed).key_pairs

    def propose_keys(
        self,
        anchor: Dataset,
        anchor_path: Path,
        feature: Dataset,
        feature_path: Path,
        *,
        seed: int = 0,
    ) -> KeyProposal:
        """Resolve key candidates by data and pick the proposed key.

        Without a customer-id pair this is the historical behaviour: every dictionary pair
        (phone/idcard/date/name) composes one key, else the fuzzy-name fallback. With one,
        a raw=raw customer-id pair that is unique on the feature side and matches at least
        as well as every other identity candidate becomes the SOLE key — it is never
        silently composed with a hashed phone into a stricter key. Otherwise the historical
        composite stands (the customer id is composed with it only when that composite holds
        no identity element). Every other identity key stays a switchable alternative.

        Only the best feature column of each anchor column takes part in that choice. The
        other feature columns that also match (up to MAX_FEATURE_CANDIDATES_PER_ANCHOR per
        anchor column) are kept as candidates and offered as whole keys of their own: the
        proposed key with that anchor column's pair swapped, plus the standalone key for an
        identity column."""
        composite: list[KeyPair] = []
        extras: list[KeyPair] = []
        composable: set[tuple[str, str]] = set()
        identity: set[tuple[str, str]] = set()
        for family in _COMPOSITE_FAMILIES:
            resolved, more = self._resolve_family(anchor, anchor_path, feature, feature_path, family, seed)
            composite.extend(resolved)
            extras.extend(more)
            target = composable if family in _COMPOSABLE_ONLY_FAMILIES else identity
            target.update(_pair_cols(pair) for pair in (*resolved, *more))
        choice = "dictionary"
        if not composite:
            composite = self._fuzzy_resolve(anchor, anchor_path, feature, feature_path, seed)
            choice = "fuzzy"
        composite = _dedupe_keypairs(composite)
        if not composite:
            choice = ""
        customer_best, customer_more = self._resolve_family(
            anchor, anchor_path, feature, feature_path, "customer_id", seed
        )
        customer = _dedupe_keypairs(customer_best)
        extras = _dedupe_keypairs(customer_more + extras)
        if not customer:
            return KeyProposal(
                composite,
                tuple(_dedupe_keypairs(composite + extras)),
                _without_key(_swap_variants(composite, extras), composite),
                choice,
            )

        customer_cols = {_pair_cols(pair) for pair in (*customer, *customer_more)}
        # Customer pairs first so a fuzzy duplicate of the same columns never displaces them.
        candidates = tuple(_dedupe_keypairs(customer + composite + extras))
        unique = {
            _pair_cols(pair): self._backend.is_key_unique(
                feature_path, [pair.feature_col], key_pairs=[pair]
            )
            for pair in customer
        }
        # Only other identity keys (phone / idcard) compete with the customer id. A fuzzy
        # name match such as a 3-value `channel` overlaps 100% by value and must never
        # outrank a raw, unique customer id (review H2).
        rival_rate = max(
            (
                pair.match_rate
                for pair in candidates
                if _pair_cols(pair) in identity
            ),
            default=0.0,
        )
        ranked = sorted(customer, key=lambda pair: (
            not unique[_pair_cols(pair)],
            -pair.match_rate,
            pair.match_method != "exact",
            pair.anchor_col,
            pair.feature_col,
        ))
        best = ranked[0]
        # Single-key alternatives: each standalone identity key (customer id / phone / idcard).
        identity_alternatives = tuple(
            (pair,)
            for pair in candidates
            if _pair_cols(pair) in customer_cols | identity
        )
        # Compare with the key that would run instead, at key level: a phone+date
        # composite matching 70% must not beat an 80% unique customer id just
        # because the phone alone matches 100% (review N1).
        if rival_rate > best.match_rate and any(_pair_cols(pair) in identity for pair in composite):
            rival_rate = self._key_match_rate(anchor, anchor_path, feature, feature_path, composite, seed)
        if (
            unique[_pair_cols(best)]
            and _same_representation(best)
            and best.match_rate >= rival_rate
        ):
            return KeyProposal(
                [best],
                candidates,
                _without_key(
                    (tuple(composite), *identity_alternatives, *_swap_variants([best], extras)),
                    [best],
                ),
                "customer_id",
            )
        if not unique[_pair_cols(best)]:
            choice = "customer_id_not_unique"
        elif best.match_rate < rival_rate:
            choice = "customer_id_lower_match"
        else:
            choice = "customer_id_hashed"
        if any(_pair_cols(pair) not in composable for pair in composite):
            key_pairs = composite
        else:
            # The historical key holds only dates/names (or nothing): the customer id is the
            # identity element, composed with them as phone+date used to be.
            key_pairs = _dedupe_keypairs([best, *composite])
            if composite:
                choice = "customer_id_composite"
        return KeyProposal(
            key_pairs,
            candidates,
            _without_key(
                (tuple(composite), *identity_alternatives, *_swap_variants(key_pairs, extras)),
                key_pairs,
            ),
            choice,
        )

    def _key_match_rate(
        self,
        anchor: Dataset,
        anchor_path: Path,
        feature: Dataset,
        feature_path: Path,
        key_pairs: list[KeyPair],
        seed: int,
    ) -> float:
        """Sampled match rate of a whole key, as the join diagnosis measures it."""
        if len(key_pairs) == 1:
            return key_pairs[0].match_rate
        from marvis.data.join_engine import _key_fps  # join_engine imports this module

        matched, sampled = self._backend.match_rate_for_method(
            anchor_path,
            [pair.anchor_col for pair in key_pairs],
            feature_path,
            [pair.feature_col for pair in key_pairs],
            method=[pair.match_method for pair in key_pairs],
            key_fingerprints=_key_fps(anchor, feature, key_pairs),
            sample_n=SMALL_SAMPLE_N,
            seed=seed,
        )
        return matched / sampled if sampled else 0.0

    def _resolve_family(
        self,
        anchor: Dataset,
        anchor_path: Path,
        feature: Dataset,
        feature_path: Path,
        family: str,
        seed: int,
    ) -> tuple[list[KeyPair], list[KeyPair]]:
        """(best pair per anchor column, the other matching feature columns of each anchor
        column — at most MAX_FEATURE_CANDIDATES_PER_ANCHOR pairs per anchor column in all)."""
        best_pairs: list[KeyPair] = []
        extras: list[KeyPair] = []
        feature_columns = self._family_columns(feature.columns, family)
        for anchor_column in self._family_columns(anchor.columns, family):
            ranked = self._rank_by_data(
                anchor_column,
                feature_columns,
                anchor_path,
                feature_path,
                seed,
                resolved_by="empirical",
            )
            if ranked:
                best_pairs.append(ranked[0])
                extras.extend(ranked[1:MAX_FEATURE_CANDIDATES_PER_ANCHOR])
        return best_pairs, extras

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
        ranked = self._rank_by_data(
            anchor_col, feature_cols, anchor_path, feature_path, seed, resolved_by=resolved_by,
        )
        return ranked[0] if ranked else None

    def _rank_by_data(
        self,
        anchor_col: ColumnProfile,
        feature_cols: list[ColumnProfile],
        anchor_path: Path,
        feature_path: Path,
        seed: int,
        *,
        resolved_by: str,
    ) -> list[KeyPair]:
        """Every feature column that matches ``anchor_col`` by data, best match first.

        Each feature column keeps its best method (the first on a tie). Ties between feature
        columns keep their column order, so ``ranked[0]`` is the single pair this resolver
        has always returned."""
        per_column: list[KeyPair] = []
        for feature_col in feature_cols:
            best: KeyPair | None = None
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
            if best is not None:
                per_column.append(best)
        # sorted() is stable: equal rates keep feature-column order.
        return sorted(per_column, key=lambda pair: -pair.match_rate)

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
        matches = _matches_tokens if family in _TOKEN_MATCH_FAMILIES else _matches_dictionary
        return [
            column
            for column in columns
            if column.semantic_role == family or matches(column.name, keywords)
        ]


def is_customer_id_column(name: str) -> bool:
    """True when ``name`` reads as a customer-id column (the customer_id key family)."""
    return _matches_tokens(name, KEY_DICTIONARY["customer_id"])


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


def _pair_cols(pair: KeyPair) -> tuple[str, str]:
    return (pair.anchor_col, pair.feature_col)


def _same_representation(pair: KeyPair) -> bool:
    """Both sides compared as stored (raw = raw, or the same hash on both sides) — no hash
    transform of one side and no date normalisation."""
    return pair.match_method in {"exact", "exact_lower"} and pair.transform_side == "both"


def _without_key(
    key_sets: tuple[tuple[KeyPair, ...], ...],
    key_pairs: list[KeyPair],
) -> tuple[tuple[KeyPair, ...], ...]:
    """Distinct, non-empty key sets other than ``key_pairs`` (order kept)."""
    seen = {frozenset(_pair_cols(pair) for pair in key_pairs)}
    kept = []
    for key_set in key_sets:
        signature = frozenset(_pair_cols(pair) for pair in key_set)
        if key_set and signature not in seen:
            seen.add(signature)
            kept.append(key_set)
    return tuple(kept)


def _swap_variants(
    key_pairs: list[KeyPair],
    extras: list[KeyPair],
) -> tuple[tuple[KeyPair, ...], ...]:
    """One whole key per extra candidate whose anchor column is in ``key_pairs``: the same
    key with that anchor column's pair swapped for the extra feature column."""
    variants = []
    for extra in extras:
        index = next(
            (i for i, pair in enumerate(key_pairs) if pair.anchor_col == extra.anchor_col),
            None,
        )
        if index is None or _pair_cols(key_pairs[index]) == _pair_cols(extra):
            continue
        variants.append(tuple(extra if i == index else pair for i, pair in enumerate(key_pairs)))
    return tuple(variants)


def _matches_dictionary(name: str, keywords: list[str]) -> bool:
    normalized = _normalized(name)
    return any(_normalized(keyword) in normalized for keyword in keywords)


def _matches_tokens(name: str, keywords: list[str]) -> bool:
    """Keyword match on whole underscore-delimited tokens: "app_cust_id" and "cust_id_md5"
    match "cust_id"; "cust_idcard" does not."""
    normalized = _normalized(name)
    return any(
        re.search(rf"(?:^|_){re.escape(_normalized(keyword))}(?:_|$)", normalized)
        for keyword in keywords
    )


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


__all__ = [
    "FUZZY_NAME_THRESHOLD",
    "KEY_DICTIONARY",
    "MAX_FEATURE_CANDIDATES_PER_ANCHOR",
    "ColumnAligner",
    "KeyProposal",
    "_divergence_level",
    "_dtype_family",
    "is_customer_id_column",
]
