"""T2-1: label-conversion convergence tests + anti-drift guard (INV-1 / INV-2).

Two concerns live here:

1. **Regression** — the strategy *setup preview* (``marvis/agent/strategy_setup.py``)
   used to silently ``coerce``-drop NaN labels and report a ``bad_rate`` that
   disagreed with the canonical ``resolve_labeled_frame`` gate applied later at
   backtest time. The preview cannot hard-stop (it is a best-effort pre-gate step),
   but it must (a) compute the bad-rate over the SAME finite-label set the gate uses,
   so the numbers match after a confirmed drop, and (b) surface the NaN-label count
   so nothing is hidden.

2. **Anti-drift guard** — a lint-style scan asserting that no NEW bare
   ``pd.to_numeric(df[target_col], errors="raise").astype(int)`` label conversion is
   added to ``marvis/`` outside the reviewed allowlist. Every allowlisted site was
   audited (T2-1) to be either the canonical gate itself, an already-gated
   defense-in-depth conversion, or an intentionally-different exploratory path. A new
   unlisted site fails the test so the divergence class cannot silently regrow.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import marvis.agent.strategy_setup as strategy_setup

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MARVIS = _REPO_ROOT / "marvis"


class _FakeBackend:
    """Minimal backend stub exposing the two methods strategy_setup calls."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def read_frame(self, path, columns=None):  # noqa: ANN001 - stub signature
        if columns is not None:
            return self._frame[columns].copy()
        return self._frame.copy()


def _nan_frame() -> pd.DataFrame:
    # 6 finite labels (2 bad / 4 good) + 2 NaN labels -> finite bad-rate = 2/6.
    return pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 0, 0, np.nan, np.nan],
            "score": [0.9, 0.8, 0.2, 0.3, 0.1, 0.4, 0.5, 0.6],
        }
    )


# --------------------------------------------------------------------------- #
# 1. Regression: setup preview converges with the canonical gate
# --------------------------------------------------------------------------- #
def test_target_bad_rate_matches_gate_and_surfaces_nan_count() -> None:
    from marvis.data.labels import resolve_labeled_frame

    frame = _nan_frame()[["y"]]
    bad_rate, n_nan = strategy_setup._target_bad_rate(_FakeBackend(frame), "path", "y")

    gated, dropped = resolve_labeled_frame(frame, "y", drop_nan_labels=True)
    gate_rate = float((pd.to_numeric(gated["y"], errors="raise") == 1).mean())

    assert n_nan == dropped == 2
    assert bad_rate == pytest.approx(gate_rate)
    assert bad_rate == pytest.approx(2 / 6)


def test_target_bad_rate_best_effort_none_on_unreadable() -> None:
    class _BoomBackend:
        def read_frame(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    assert strategy_setup._target_bad_rate(_BoomBackend(), "path", "y") == (None, 0)


def test_score_profile_bad_rate_excludes_nan_labels_and_notes_them() -> None:
    profile = strategy_setup._score_profile(_nan_frame(), target_col="y", score_col="score")
    # bad-rate over finite labels only (2 bad / 6 finite), NOT 2/8.
    assert profile["bad_rate"] == pytest.approx(2 / 6)
    assert any("2 行标签" in note for note in profile["notes"])


def test_score_profile_non_numeric_labels_raise_like_the_gate() -> None:
    frame = pd.DataFrame({"y": ["Y", "N", 1, 0], "score": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(strategy_setup.StrategySetupError):
        strategy_setup._score_profile(frame, target_col="y", score_col="score")


def test_score_profile_clean_labels_unchanged() -> None:
    frame = pd.DataFrame({"y": [1, 0, 1, 0], "score": [0.9, 0.1, 0.8, 0.2]})
    profile = strategy_setup._score_profile(frame, target_col="y", score_col="score")
    assert profile["bad_rate"] == pytest.approx(0.5)
    # no NaN labels -> no NaN-exclusion note appended
    assert not any("标签为空" in note for note in profile["notes"])


# --------------------------------------------------------------------------- #
# 2. Anti-drift guard: no NEW bare label->int coercion outside the allowlist
# --------------------------------------------------------------------------- #
#
# Every entry below was audited in T2-1. Each is one of:
#   - the canonical gate/helper itself (labels.py / checks.py), or
#   - an already-gated defense-in-depth conversion (strategy cores behind a tool
#     wrapper that calls resolve_labeled_frame; modeling paths downstream of
#     resolve_modeling_splits / require_labels_confirmed), or
#   - synthetic data generation (sample_data.py), or
#   - an intentionally-different exploratory/optional-label path.
# A NEW `pd.to_numeric(<target>, errors="raise").astype(int)` site anywhere in
# marvis/ that is not listed here fails this test on purpose: it must be reviewed
# and either routed through the canonical gate or explicitly allowlisted.
_ALLOWLISTED_RAISE_ASTYPE_INT = frozenset(
    {
        # canonical surfaces
        "validation/checks.py",
        # strategy cores: behind tool wrappers that gate via resolve_labeled_frame
        "packs/strategy/tradeoff.py",
        "packs/strategy/bands.py",
        "packs/strategy/compare.py",
        "packs/strategy/backtest.py",
    }
)

_LABEL_ARG_HINTS = ("target", "label", "y", "bad")


def _rel(path: Path) -> str:
    return path.relative_to(_MARVIS).as_posix()


def _is_to_numeric_raise(call: ast.AST) -> bool:
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "to_numeric"):
        return False
    for kw in call.keywords:
        if kw.arg == "errors" and isinstance(kw.value, ast.Constant) and kw.value.value == "raise":
            return True
    return False


def _looks_like_label_subscript(node: ast.AST) -> bool:
    """True when the first arg looks like frame[<target-ish>] / df[target_col]."""
    if not (isinstance(node, ast.Call) and node.args):
        return False
    arg = node.args[0]
    if not isinstance(arg, ast.Subscript):
        return False
    key = arg.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return any(hint in key.value.lower() for hint in _LABEL_ARG_HINTS)
    if isinstance(key, ast.Name):
        return any(hint in key.id.lower() for hint in _LABEL_ARG_HINTS)
    return False


def _find_bare_label_raise_astype_int(tree: ast.AST) -> list[int]:
    """Line numbers of `pd.to_numeric(df[target], errors='raise').astype(int)`."""
    hits: list[int] = []
    for node in ast.walk(tree):
        # match `<expr>.astype(int)`
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "astype":
            continue
        if not (node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "int"):
            # also accept astype(np.int...) / astype("int")
            first = node.args[0] if node.args else None
            is_int_str = isinstance(first, ast.Constant) and str(first.value).startswith("int")
            is_np_int = isinstance(first, ast.Attribute) and str(first.attr).startswith("int")
            if not (is_int_str or is_np_int):
                continue
        inner = node.func.value
        if _is_to_numeric_raise(inner) and _looks_like_label_subscript(inner):
            hits.append(node.lineno)
    return hits


def test_no_new_bare_label_raise_astype_int_outside_allowlist() -> None:
    offenders: dict[str, list[int]] = {}
    for path in sorted(_MARVIS.rglob("*.py")):
        rel = _rel(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        hits = _find_bare_label_raise_astype_int(tree)
        if hits and rel not in _ALLOWLISTED_RAISE_ASTYPE_INT:
            offenders[rel] = hits

    assert not offenders, (
        "New bare `pd.to_numeric(<target>, errors='raise').astype(int)` on a label "
        "column found outside the T2-1 reviewed allowlist. Route it through "
        "marvis.data.labels (resolve_labeled_frame / nan_label_mask) or "
        "marvis.validation.checks.binary_target_series, or add it to "
        f"_ALLOWLISTED_RAISE_ASTYPE_INT with a review note. Offenders: {offenders}"
    )


def test_allowlist_entries_still_exist_and_still_match() -> None:
    """Keep the allowlist honest: every listed file must still contain the pattern.

    If a converged/removed conversion leaves a stale allowlist entry, this fails so
    the entry gets pruned (prevents the guard from silently rotting into a no-op).
    """
    stale: list[str] = []
    for rel in _ALLOWLISTED_RAISE_ASTYPE_INT:
        path = _MARVIS / rel
        if not path.exists():
            stale.append(f"{rel} (missing file)")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _find_bare_label_raise_astype_int(tree):
            stale.append(f"{rel} (pattern gone)")
    assert not stale, f"Prune stale allowlist entries: {stale}"
