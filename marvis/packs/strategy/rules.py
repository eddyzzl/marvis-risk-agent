"""Deterministic rule mining + rule-set evaluation core (S4).

Two pure functions:

* :func:`mine_rules` proposes candidate reject rules from a labeled sample via
  two deterministic channels -- shallow decision-tree paths and single-variable
  optimal cutpoints -- and returns them ranked by lift.
* :func:`evaluate_rule_set` scores a user-chosen, ordered subset of rules with a
  first-match-wins waterfall, an overlap matrix, and residual/combined stats.

Both resolve a rule's per-row hits through
:func:`marvis.packs.strategy.strategy.evaluate_condition_mask` -- the *same*
validated evaluator ``apply_strategy``/``build_strategy`` use. There is no second
evaluator: a condition string produced here lands on the identical hit set when
fed to ``build_strategy`` (round-trip guarantee).

Determinism (INV-1): the decision tree is fixed to ``random_state=seed``; tree
paths and univariate cutpoints are extracted in a stable, sorted order; two runs
with the same inputs return byte-identical dicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.strategy import evaluate_condition_mask

# Default mining seed (INV-1). Chosen once; every mine_rules run pins it unless a
# caller overrides, so the decision tree -- the only stochastic component -- is
# fully reproducible.
DEFAULT_MINE_SEED = 20260701

# Two candidate rules are treated as the same rule when they compare the same
# feature with operators of the same direction-family and thresholds within this
# absolute tolerance (dedup across the tree/univariate channels).
_THRESHOLD_DEDUP_TOL = 1e-9

# A clause's operator is bucketed into one of two direction families for dedup:
# "ge" (>=, >) selects the high side, "lt" (<=, <) the low side.
_OP_FAMILY = {">=": "ge", ">": "ge", "<=": "lt", "<": "lt"}


@dataclass(frozen=True)
class CandidateRule:
    rule_id: str
    clauses: tuple[dict, ...]
    condition: str
    support: float
    hit_count: int
    hit_bad_rate: float
    lift: float
    source: str

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "clauses": [dict(clause) for clause in self.clauses],
            "condition": self.condition,
            "support": self.support,
            "hit_count": self.hit_count,
            "hit_bad_rate": self.hit_bad_rate,
            "lift": self.lift,
            "source": self.source,
        }


def mine_rules(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    max_depth: int = 3,
    min_support: float = 0.02,
    min_lift: float = 1.5,
    top_k: int = 20,
    seed: int = DEFAULT_MINE_SEED,
) -> list[CandidateRule]:
    """Propose candidate reject rules from a labeled sample (deterministic).

    Channel A (tree): a depth-limited ``DecisionTreeClassifier`` fixed to
    ``random_state=seed``; every root->leaf path whose leaf bad rate beats the
    base rate becomes one conjunctive rule.
    Channel B (univariate): for each feature, scan a deterministic quantile grid
    of thresholds and keep the single best (highest-lift) ``>=``/``<`` cut.

    Candidates from both channels are merged (equivalent conditions deduped, tree
    kept over univariate), filtered by ``min_support``/``min_lift``, and returned
    ranked by lift desc (ties broken by support desc then condition text) -- a
    stable sort, so two identical calls return identical lists (INV-1).
    """
    frame, target = _prepare(df, feature_cols, target_col)
    n = int(len(frame))
    if n == 0:
        return []
    base_rate = float(target.mean())
    numeric = _numeric_features(frame, feature_cols)
    if not numeric:
        raise StrategyError("mine_rules requires at least one numeric feature column")

    min_leaf = max(1, int(round(min_support * n)))
    tree_rules = _mine_tree(frame, target, numeric, base_rate, max_depth, min_leaf, seed)
    univariate_rules = _mine_univariate(frame, target, numeric, base_rate)

    merged = _merge_candidates(tree_rules, univariate_rules)
    kept = [
        rule
        for rule in merged
        if rule["support"] >= min_support and rule["lift"] >= min_lift
    ]
    ranked = sorted(
        kept,
        key=lambda rule: (-rule["lift"], -rule["support"], rule["condition"]),
    )
    top = ranked[: max(0, int(top_k))]
    return [_finalize(rule, index) for index, rule in enumerate(top)]


def evaluate_rule_set(
    df: pd.DataFrame,
    rules_ordered: list[dict],
    *,
    target_col: str,
    decision: str = "reject",
) -> dict:
    """Score an ordered rule subset with first-match-wins waterfall + overlap.

    ``rules_ordered`` is a list of ``{condition, ...}`` dicts (the user-selected
    subset). Semantics match ``apply_strategy`` exactly: rule i "owns" only the
    rows it hits that no earlier rule already claimed. Returns the waterfall,
    an NxN pairwise-overlap matrix, residual (approved) stats, and combined
    reject/approve stats.
    """
    frame, target = _prepare_eval(df, target_col)
    n = int(len(frame))
    masks = [evaluate_condition_mask(frame, str(rule["condition"])).to_numpy(dtype=bool) for rule in rules_ordered]
    target_arr = target.to_numpy(dtype=float)

    assigned = np.zeros(n, dtype=bool)
    waterfall: list[dict] = []
    combined_reject = np.zeros(n, dtype=bool)
    for index, mask in enumerate(masks):
        incremental = mask & ~assigned
        assigned = assigned | mask
        combined_reject = combined_reject | mask
        inc_count = int(incremental.sum())
        cum_reject = int(assigned.sum())
        waterfall.append(
            {
                "rule_id": _rule_id(index),
                "incremental_hits": inc_count,
                "incremental_bad_rate": _mean_or_zero(target_arr[incremental]),
                "cum_reject_rate": _ratio(cum_reject, n),
                "cum_reject_bad_rate": _mean_or_zero(target_arr[assigned]),
            }
        )

    overlap_matrix = _overlap_matrix(masks)
    approved = ~combined_reject
    residual = {
        "approval_rate": _ratio(int(approved.sum()), n),
        "bad_rate": _mean_or_zero(target_arr[approved]),
    }
    combined = {
        "reject_rate": _ratio(int(combined_reject.sum()), n),
        "rejected_bad_rate": _mean_or_zero(target_arr[combined_reject]),
        "approved_bad_rate": _mean_or_zero(target_arr[approved]),
    }
    return {
        "decision": str(decision),
        "waterfall": waterfall,
        "overlap_matrix": overlap_matrix,
        "residual": residual,
        "combined": combined,
    }


# ---------------------------------------------------------------------------
# tree channel
# ---------------------------------------------------------------------------
def _mine_tree(
    frame: pd.DataFrame,
    target: pd.Series,
    numeric: list[str],
    base_rate: float,
    max_depth: int,
    min_leaf: int,
    seed: int,
) -> list[dict]:
    x = frame[numeric].to_numpy(dtype=float)
    y = target.to_numpy(dtype=int)
    if len(set(y.tolist())) < 2:
        return []
    clf = DecisionTreeClassifier(
        max_depth=max(1, int(max_depth)),
        min_samples_leaf=max(1, int(min_leaf)),
        random_state=int(seed),
    )
    clf.fit(x, y)
    tree = clf.tree_
    rules: list[dict] = []
    # Deterministic pre-order walk; each leaf yields its accumulated clause path.
    stack: list[tuple[int, tuple[dict, ...]]] = [(0, ())]
    while stack:
        node, path = stack.pop()
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == -1 and right == -1:
            rule = _rule_from_path(frame, target, numeric, path, base_rate, "tree")
            if rule is not None:
                rules.append(rule)
            continue
        feature = numeric[int(tree.feature[node])]
        threshold = float(tree.threshold[node])
        # sklearn: left child is feature <= threshold, right child is >.
        # Push right first so left is processed first after pop (stable order).
        stack.append((right, (*path, {"feature": feature, "op": ">", "value": threshold})))
        stack.append((left, (*path, {"feature": feature, "op": "<=", "value": threshold})))
    return rules


def _rule_from_path(
    frame: pd.DataFrame,
    target: pd.Series,
    numeric: list[str],
    path: tuple[dict, ...],
    base_rate: float,
    source: str,
) -> dict | None:
    clauses = _collapse_clauses(path)
    if not clauses:
        return None
    condition = _clauses_to_condition(clauses)
    return _stats_for_condition(frame, target, clauses, condition, base_rate, source)


def _collapse_clauses(path: tuple[dict, ...]) -> tuple[dict, ...]:
    """Collapse repeated feature/op pairs along a tree path into the tightest
    bound (max threshold for ``>``, min for ``<=``) so a path that splits the
    same feature twice yields one clause per (feature, op), not two redundant
    ones -- keeps conditions short and comparable for dedup."""
    tightest: dict[tuple[str, str], float] = {}
    order: list[tuple[str, str]] = []
    for clause in path:
        key = (clause["feature"], clause["op"])
        value = float(clause["value"])
        if key not in tightest:
            tightest[key] = value
            order.append(key)
        elif clause["op"] == ">":
            tightest[key] = max(tightest[key], value)
        else:
            tightest[key] = min(tightest[key], value)
    return tuple(
        {"feature": feature, "op": op, "value": tightest[(feature, op)]}
        for feature, op in order
    )


# ---------------------------------------------------------------------------
# univariate channel
# ---------------------------------------------------------------------------
_QUANTILE_GRID = tuple(round(q, 2) for q in np.linspace(0.05, 0.95, 19))


def _mine_univariate(
    frame: pd.DataFrame,
    target: pd.Series,
    numeric: list[str],
    base_rate: float,
) -> list[dict]:
    rules: list[dict] = []
    for feature in numeric:
        series = frame[feature]
        best: dict | None = None
        thresholds = _quantile_thresholds(series)
        for threshold in thresholds:
            for op in (">=", "<"):
                clauses = ({"feature": feature, "op": op, "value": float(threshold)},)
                condition = _clauses_to_condition(clauses)
                stats = _stats_for_condition(frame, target, clauses, condition, base_rate, "univariate")
                if stats is None:
                    continue
                if best is None or _univariate_better(stats, best):
                    best = stats
        if best is not None:
            rules.append(best)
    return rules


def _quantile_thresholds(series: pd.Series) -> list[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return []
    values = sorted({round(float(numeric.quantile(q)), 12) for q in _QUANTILE_GRID})
    return values


def _univariate_better(candidate: dict, current: dict) -> bool:
    # Higher lift wins; ties broken by higher support, then lexicographic
    # condition (deterministic).
    key_c = (candidate["lift"], candidate["support"], _neg_text(candidate["condition"]))
    key_b = (current["lift"], current["support"], _neg_text(current["condition"]))
    return key_c > key_b


def _neg_text(text: str) -> tuple[int, ...]:
    # A total order on strings that ranks lexicographically-smaller text as
    # "greater" (so ties pick the alphabetically-first condition), without
    # comparing strings against floats in the tuple key above.
    return tuple(-ord(ch) for ch in text)


# ---------------------------------------------------------------------------
# shared stats + merge
# ---------------------------------------------------------------------------
def _stats_for_condition(
    frame: pd.DataFrame,
    target: pd.Series,
    clauses: tuple[dict, ...],
    condition: str,
    base_rate: float,
    source: str,
) -> dict | None:
    mask = evaluate_condition_mask(frame, condition).to_numpy(dtype=bool)
    hit_count = int(mask.sum())
    if hit_count == 0:
        return None
    n = int(len(frame))
    target_arr = target.to_numpy(dtype=float)
    hit_bad_rate = float(target_arr[mask].mean())
    lift = float(hit_bad_rate / base_rate) if base_rate > 0 else 0.0
    return {
        "clauses": tuple(dict(clause) for clause in clauses),
        "condition": condition,
        "support": _ratio(hit_count, n),
        "hit_count": hit_count,
        "hit_bad_rate": hit_bad_rate,
        "lift": lift,
        "source": source,
    }


def _merge_candidates(tree_rules: list[dict], univariate_rules: list[dict]) -> list[dict]:
    """Merge tree + univariate candidates, dropping later duplicates. Tree rules
    are added first so an equivalent univariate rule (same feature set + same
    direction-family thresholds within tolerance) is dropped in favor of the
    tree one, matching the spec's 'tree kept' dedup preference."""
    merged: list[dict] = []
    signatures: list[tuple] = []
    for rule in [*tree_rules, *univariate_rules]:
        signature = _rule_signature(rule["clauses"])
        if any(_signatures_equivalent(signature, existing) for existing in signatures):
            continue
        signatures.append(signature)
        merged.append(rule)
    return merged


def _rule_signature(clauses: tuple[dict, ...]) -> tuple:
    return tuple(
        sorted(
            (clause["feature"], _OP_FAMILY.get(clause["op"], clause["op"]), round(float(clause["value"]), 12))
            for clause in clauses
        )
    )


def _signatures_equivalent(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    for (fa, opa, va), (fb, opb, vb) in zip(a, b, strict=True):
        if fa != fb or opa != opb or abs(va - vb) > _THRESHOLD_DEDUP_TOL:
            return False
    return True


def _finalize(rule: dict, index: int) -> CandidateRule:
    return CandidateRule(
        rule_id=_rule_id(index),
        clauses=tuple(dict(clause) for clause in rule["clauses"]),
        condition=rule["condition"],
        support=rule["support"],
        hit_count=rule["hit_count"],
        hit_bad_rate=rule["hit_bad_rate"],
        lift=rule["lift"],
        source=rule["source"],
    )


# ---------------------------------------------------------------------------
# condition string generation
# ---------------------------------------------------------------------------
def _clauses_to_condition(clauses: tuple[dict, ...]) -> str:
    """Render clauses to a condition string ``build_strategy`` parses verbatim
    (round-trip): ``feature op literal`` joined by ``and``. Numbers use the same
    integer-vs-6-significant-figures literal format the setup module uses so
    build_strategy's AST parser accepts them unchanged."""
    parts = [f"{clause['feature']} {clause['op']} {_number_literal(float(clause['value']))}" for clause in clauses]
    return " and ".join(parts)


def _number_literal(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


# ---------------------------------------------------------------------------
# overlap + helpers
# ---------------------------------------------------------------------------
def _overlap_matrix(masks: list[np.ndarray]) -> list[list[float]]:
    """Symmetric NxN matrix of pairwise co-hit share: cell (i, j) = |i∩j| / |i∪j|
    (Jaccard), 1.0 on the diagonal, 0.0 when either rule hits nothing."""
    size = len(masks)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = 1.0 if bool(masks[i].any()) else 0.0
                continue
            inter = int((masks[i] & masks[j]).sum())
            union = int((masks[i] | masks[j]).sum())
            matrix[i][j] = float(inter / union) if union > 0 else 0.0
    return matrix


def _prepare(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if not feature_cols:
        raise StrategyError("mine_rules requires at least one feature column")
    _assert_columns(df, [*feature_cols, target_col])
    frame = df[[*feature_cols, target_col]].copy()
    target = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.loc[target.notna()].reset_index(drop=True)
    target = target.loc[target.notna()].reset_index(drop=True).astype(int)
    return frame, target


def _prepare_eval(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    _assert_columns(df, [target_col])
    frame = df.copy().reset_index(drop=True)
    target = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.loc[target.notna()].reset_index(drop=True)
    target = target.loc[target.notna()].reset_index(drop=True).astype(int)
    return frame, target


def _numeric_features(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    numeric: list[str] = []
    for column in feature_cols:
        coerced = pd.to_numeric(frame[column], errors="coerce")
        if coerced.notna().any():
            numeric.append(column)
    return numeric


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise StrategyError(f"missing columns: {missing}")


def _rule_id(index: int) -> str:
    return f"rule_{index + 1}"


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


__all__ = [
    "DEFAULT_MINE_SEED",
    "CandidateRule",
    "evaluate_rule_set",
    "mine_rules",
]
