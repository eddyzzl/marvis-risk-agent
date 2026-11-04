from __future__ import annotations

from dataclasses import dataclass


SHRINK_WARN_THRESHOLD = 0.5
SMALL_SAMPLE_N = 5000
LARGE_ROW_THRESHOLD = 200_000
MIN_KEY_MATCH_RATE = 0.5

HASH_HEX_LENGTHS = {
    32: "md5",
    40: "sha1",
    56: "sha224",
    64: "sha256",
    96: "sha384",
    128: "sha512",
}

HASH_ALGO_CANDIDATES = ("md5", "sha256", "sha1", "sha512")

DATE_FORMATS = (
    "%Y%m%d",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)


@dataclass(frozen=True)
class ColumnFingerprint:
    value_kind: str
    length_mode: int | None
    regex_pattern: str | None
    is_hashed: bool
    hash_type: str | None
    hex_case: str | None
    date_format: str | None


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    semantic_role: str
    fingerprint: ColumnFingerprint
    null_rate: float
    cardinality: int
    sample_values: tuple[object, ...]


@dataclass(frozen=True)
class Dataset:
    id: str
    task_id: str
    role: str
    source_path: str
    format: str
    sheet: str | None
    row_count: int
    columns: tuple[ColumnProfile, ...]
    has_target: bool
    target_col: str | None
    created_at: str
    # GAP-7: sha256 of the registered parquet's file bytes, used to detect when a
    # new upload is byte-identical to an already-registered dataset (possibly
    # owned by a different task) so the parquet + profiling work can be reused
    # instead of duplicated. None for datasets written before this field existed.
    content_hash: str | None = None


@dataclass(frozen=True)
class KeyPair:
    anchor_col: str
    feature_col: str
    match_method: str
    transform_side: str
    match_rate: float
    resolved_by: str
    # T1-B8: the stored dtype of each side of this key, and whether they diverge across files.
    # Defaulted so existing constructors / stored plans keep working. Divergence coupling
    # text<->float is the dangerous precision-loss / silent-miss case (see _divergence_level).
    anchor_dtype: str = ""
    feature_dtype: str = ""
    dtype_divergent: bool = False


# A C2 join-key choice names an anchor column, or — when that anchor column has several
# feature-column candidates — one candidate pair written "anchor_col=feature_col".
KEY_SELECTION_SEPARATOR = "="


def key_selection_token(anchor_col: str, feature_col: str) -> str:
    return f"{anchor_col}{KEY_SELECTION_SEPARATOR}{feature_col}"


@dataclass(frozen=True)
class ConflictReport:
    """Result of a two-level dedup (spec §6). Level-1 removes whole-row duplicates
    (key + ALL values identical) losslessly. Level-2 detects rows that share a key but
    DISAGREE on some value (同人同天特征不一致) — a data-quality red flag that is
    REPORTED, never silently dropped."""

    key_columns: tuple[str, ...]
    conflict_columns: tuple[str, ...]   # non-key columns whose values disagree within a key
    n_conflict_keys: int
    n_conflict_rows: int
    safe_dropped: int                   # level-1 whole-row duplicates removed (lossless)
    sample_keys: tuple[tuple, ...]      # capped sample of conflicting key-value tuples
    # UX-6: for each key in sample_keys (same order, same cap), the row-level values the
    # conflict_columns actually took for that key -- e.g. {"amount": [500, 800]} -- so a
    # dedup gate can show "k=123 时 amount 两行分别为 500/800" instead of a bare count.
    sample_conflicts: tuple[dict, ...] = ()

    @property
    def has_conflicts(self) -> bool:
        return self.n_conflict_keys > 0


@dataclass(frozen=True)
class KeyDtypeDivergence:
    """T1-B8: a proposed join key whose two sides are stored under different dtype families
    across files (e.g. text one side, float the other). ``level`` == "red" for text<->float
    (precision loss / leading-zero loss -> silent miss; forces confirmation); "warn" for the
    lossless-at-VARCHAR-cast cases (int<->float, text<->int) which are surfaced but not blocked.
    REPORTED, never silently swapped or dropped."""

    anchor_col: str
    feature_col: str
    anchor_dtype: str
    feature_dtype: str
    level: str


@dataclass(frozen=True)
class KeyAlternative:
    """A join-key candidate the user may pick at C2 (spec §4/§5 动态择键). ``kind`` is
    "relaxation" (drop one identity element of a poorly matching composite key) or "switch"
    (a different data-resolved identity key, e.g. mobile=mobile_md5 instead of the preferred
    cust_id). SURFACED as a proposal only — the engine never silently swaps the key; the user
    picks it at C2, and match / uniqueness / fan-out are re-checked for that key."""

    key_pairs: tuple[tuple[str, str], ...]   # (anchor_col, feature_col) of the alternative key
    dropped: str                             # anchor_col(s) of the current key it leaves out
    match_rate: float
    feature_key_unique: bool
    fan_out_detected: bool
    kind: str = "relaxation"


@dataclass
class JoinDiagnostics:
    anchor_rows: int
    feature_rows: int
    feature_key_unique: bool
    matched_rows: int
    match_rate: float
    joined_rows_preview: int
    fan_out_detected: bool
    shrink_detected: bool
    new_columns: int
    new_columns_null_rate: float
    # Two-level dedup breakdown (spec §6), present when the feature key is not unique:
    # how many duplicates are safe (whole-row identical) vs genuine same-key conflicts.
    conflict_report: "ConflictReport | None" = None
    # Relaxed-key proposals (spec §4/§5), present when the full key matches poorly: each
    # drops one identity element to raise the match rate (with its re-checked fan-out).
    key_alternatives: tuple["KeyAlternative", ...] = ()
    # T1-A6: key columns stored as float64 whose ids exceed ~15 significant digits, where the
    # true id may already be corrupted by float precision loss (join can silently mis-match).
    # Reported so the C2 gate can warn the user to re-import the column as string (see B8).
    precision_loss_columns: tuple[str, ...] = ()
    # T1-B8: proposed keys whose two sides have divergent stored dtypes across files. A "red"
    # (text<->float) divergence forces confirmation before execute; "warn" is surfaced only.
    key_dtype_divergences: tuple["KeyDtypeDivergence", ...] = ()
    # Every data-resolved key pair for this feature table: the proposed key plus the identity
    # keys the user can switch to at C2. Proposal evidence only — execution always uses
    # JoinSpec.key_pairs, and a switched key is re-diagnosed before it can be confirmed.
    key_candidates: tuple["KeyPair", ...] = ()
    # Why the proposed key was chosen (see ColumnAligner.propose_keys): "customer_id",
    # "customer_id_composite", "customer_id_not_unique", "customer_id_lower_match",
    # "customer_id_hashed", "dictionary", "fuzzy", "user" (picked at C2), or "" (no key /
    # legacy plan).
    key_choice: str = ""


@dataclass
class JoinSpec:
    feature_dataset_id: str
    key_pairs: list[KeyPair]
    diagnostics: JoinDiagnostics
    dedup_strategy: str | None
    confirmed: bool = False


@dataclass
class JoinPlan:
    id: str
    task_id: str
    anchor_dataset_id: str
    joins: list[JoinSpec]
    status: str
    result_dataset_id: str | None = None


__all__ = [
    "DATE_FORMATS",
    "HASH_ALGO_CANDIDATES",
    "HASH_HEX_LENGTHS",
    "KEY_SELECTION_SEPARATOR",
    "LARGE_ROW_THRESHOLD",
    "MIN_KEY_MATCH_RATE",
    "SHRINK_WARN_THRESHOLD",
    "SMALL_SAMPLE_N",
    "ColumnFingerprint",
    "ColumnProfile",
    "ConflictReport",
    "Dataset",
    "JoinDiagnostics",
    "JoinPlan",
    "JoinSpec",
    "KeyAlternative",
    "KeyDtypeDivergence",
    "KeyPair",
    "key_selection_token",
]
