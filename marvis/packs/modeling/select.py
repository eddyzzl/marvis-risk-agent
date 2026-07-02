from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from marvis.data.labels import require_labels_confirmed
from marvis.feature.binning import chimerge_edges, monotonic_direction, monotonic_edges
from marvis.feature.correlation import correlation_matrix, find_collinear_pairs, vif
from marvis.feature.encode import woe_encode
from marvis.feature.errors import FeatureError, FitRequiresSplitError
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning
from marvis.feature.metrics import DEFAULT_IV_BINS, feature_ks, feature_metrics
from marvis.packs.modeling.prepare import SPLIT_COLUMN


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]
    scores: dict[str, dict[str, Any]]
    nan_labels_dropped: int = 0
    warnings: tuple[str, ...] = ()
    fit_rows: int = 0
    fit_split: str = "train"


def select_features(
    backend,
    dataset_path: Path,
    *,
    features: list[str],
    target_col: str,
    target_type: str = "binary",
    iv_min: float = 0.02,
    corr_max: float = 0.8,
    vif_max: float = 10.0,
    top_k: int | None = None,
    seed: int = 0,
    drop_nan_labels: bool = False,
    space: str = "raw",
    split_col: str | None = None,
    split_value: Any = None,
    holdout_values: tuple[str, ...] = ("test", "oot"),
    allow_full_fit: bool = False,
    scorecard_max_bins: int = 6,
    enforce_monotonic: bool = True,
    monotonic_direction_request: str = "auto",
    sign_check: bool = True,
) -> SelectionResult:
    del seed  # Selection is deterministic; seed is reserved for API symmetry.
    dataset_columns = set(backend.column_names(dataset_path))
    resolved_split_col = split_col or (SPLIT_COLUMN if SPLIT_COLUMN in dataset_columns else None)
    columns = _unique([*features, target_col, resolved_split_col])
    frame = backend.read_frame(dataset_path, columns=columns)
    if split_col and split_value is not None:
        # Legacy exact-match filter: caller picked a single split value explicitly
        # (e.g. split_value="train"). This already excludes every holdout row.
        frame = frame[frame[str(split_col)] == split_value].copy()
        if frame.empty:
            raise FeatureError(f"feature selection split has no rows: {split_col}={split_value}")
        fit_rows = int(len(frame))
        fit_split = "train"
    else:
        fit_mask, fit_split = _selection_fit_mask(
            frame,
            split_col=resolved_split_col,
            holdout_values=holdout_values,
            allow_full_fit=allow_full_fit,
            dataset_path=dataset_path,
        )
        frame = frame.loc[fit_mask].copy()
        fit_rows = int(len(frame))
    nan_labels_dropped = require_labels_confirmed(
        frame, target_col, drop_nan_labels=drop_nan_labels,
    )
    if str(target_type or "binary") != "binary":
        # IV/KS (and therefore the WOE space) are binary-only statistics (the FS-1 funnel
        # is framed around binary KS/IV throughout). For continuous/multiclass targets,
        # skip the multivariate refinement and pass every candidate through top_k only —
        # mirrors screen_features' non-binary bypass (tools.py::_screen_features_non_binary).
        return _select_features_passthrough(
            features, top_k=top_k, nan_labels_dropped=nan_labels_dropped,
            fit_rows=fit_rows, fit_split=fit_split,
        )
    target = frame[target_col].to_numpy(dtype=float)
    normalized_space = str(space or "raw").strip().lower()
    if normalized_space == "woe":
        return _select_features_woe(
            frame,
            features,
            target_col=target_col,
            target=target,
            iv_min=iv_min,
            corr_max=corr_max,
            vif_max=vif_max,
            top_k=top_k,
            nan_labels_dropped=nan_labels_dropped,
            max_bins=scorecard_max_bins,
            enforce_monotonic=enforce_monotonic,
            monotonic_direction_request=monotonic_direction_request,
            sign_check=sign_check,
            fit_rows=fit_rows,
            fit_split=fit_split,
        )
    if normalized_space != "raw":
        raise FeatureError("select_features space must be 'raw' or 'woe'")
    return _select_features_raw(
        frame,
        features,
        target=target,
        iv_min=iv_min,
        corr_max=corr_max,
        vif_max=vif_max,
        top_k=top_k,
        nan_labels_dropped=nan_labels_dropped,
        fit_rows=fit_rows,
        fit_split=fit_split,
    )


def _selection_fit_mask(
    frame: pd.DataFrame,
    *,
    split_col: str | None,
    holdout_values: tuple[str, ...],
    allow_full_fit: bool,
    dataset_path: Path,
) -> tuple[Any, str]:
    """Rows used to fit selection statistics (IV/corr/VIF/WOE) — excludes holdout
    (default test+OOT) so selection never peeks at evaluation labels (FS-2).

    ``split_col`` is already resolved by the caller (explicit input, else the
    platform-standard ``marvis.packs.modeling.prepare.SPLIT_COLUMN`` when present in
    the dataset). No split column at all is a typed-error stop unless the caller
    explicitly confirms a full-pool fit via ``allow_full_fit``.
    """
    if not split_col:
        if allow_full_fit:
            return frame.index.notna(), "full"
        raise FitRequiresSplitError(tool="select_features", dataset_id=str(dataset_path))
    holdout = tuple(str(value) for value in (holdout_values or ("test", "oot")))
    mask = ~frame[str(split_col)].astype(str).isin(holdout)
    if not mask.any():
        raise FeatureError("select_features fit frame is empty after excluding holdout rows")
    return mask, "train"


def _select_features_raw(
    frame: pd.DataFrame,
    features: list[str],
    *,
    target,
    iv_min: float,
    corr_max: float,
    vif_max: float,
    top_k: int | None,
    nan_labels_dropped: int,
    fit_rows: int = 0,
    fit_split: str = "train",
) -> SelectionResult:
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    scores: dict[str, dict[str, Any]] = {}
    for feature in features:
        metrics = feature_metrics(
            frame[feature].to_numpy(dtype=float),
            target,
            feature=feature,
            bins=DEFAULT_IV_BINS,
        )
        # FS-9: record the binning convention (equal-frequency, DEFAULT_IV_BINS bins) so
        # this IV is comparable against the WOE-space chimerge convention (see below).
        scores[feature] = {
            "iv": float(metrics.iv),
            "ks": float(metrics.ks),
            "space": "raw",
            "iv_binning": f"equal_frequency_{DEFAULT_IV_BINS}",
        }
        if metrics.iv < iv_min:
            dropped.append((feature, f"low IV {metrics.iv:.3f}"))
        else:
            kept.append(feature)

    warnings: list[str] = []
    kept, dropped = _drop_collinear(frame, features, kept, dropped, scores, corr_max, label="")
    kept, dropped = _drop_high_vif(frame, features, kept, dropped, scores, vif_max, label="", warnings=warnings)
    kept, dropped = _apply_top_k(features, kept, dropped, scores, top_k)
    return SelectionResult(
        tuple(kept), tuple(dropped), scores, nan_labels_dropped, tuple(warnings),
        fit_rows=fit_rows, fit_split=fit_split,
    )


def _select_features_woe(
    frame: pd.DataFrame,
    features: list[str],
    *,
    target_col: str,
    target,
    iv_min: float,
    corr_max: float,
    vif_max: float,
    top_k: int | None,
    nan_labels_dropped: int,
    max_bins: int,
    enforce_monotonic: bool,
    monotonic_direction_request: str,
    sign_check: bool,
    fit_rows: int = 0,
    fit_split: str = "train",
) -> SelectionResult:
    target_arr = frame[target_col].to_numpy(dtype=float)
    encoded = pd.DataFrame(index=frame.index)
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    scores: dict[str, dict[str, Any]] = {}
    directions: dict[str, str] = {}
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        # PREP-9: minimum bin share (5%) keeps WOE/IV estimates stable, consistent
        # with the scorecard training path (recipes/scorecard.py::_fit_woe_maps).
        edges = chimerge_edges(values, target_arr, max_bins=max_bins, min_bin_pct=0.05)
        resolved_direction = "not_enforced"
        if enforce_monotonic:
            resolved_direction = monotonic_direction(
                values,
                target_arr,
                edges,
                direction=monotonic_direction_request,
            )
            edges = monotonic_edges(values, target_arr, edges, direction=resolved_direction)
        binning = compute_woe_iv(values, target_arr, edges, feature=feature)
        woe = woe_result_from_binning(binning)
        encoded[feature] = woe_encode(frame, feature, woe).to_numpy(dtype=float)
        directions[feature] = resolved_direction
        # FS-9: record the binning convention — this path always uses chimerge with
        # max_bins bins (a different convention than the raw-space equal-frequency default;
        # the 0.02 iv_min threshold means different things under each, so record which was
        # used rather than silently mixing them).
        scores[feature] = {
            "iv": float(binning.total_iv),
            "ks": float(feature_ks(encoded[feature].to_numpy(dtype=float), target)),
            "space": "woe",
            "monotonic_direction": resolved_direction,
            "bin_count": len(binning.bins),
            "iv_binning": f"chimerge_{max_bins}",
        }
        if binning.total_iv < iv_min:
            dropped.append((feature, f"low WOE IV {binning.total_iv:.3f}"))
        else:
            kept.append(feature)

    warnings: list[str] = []
    kept, dropped = _drop_collinear(encoded, features, kept, dropped, scores, corr_max, label="WOE ")
    kept, dropped = _drop_high_vif(encoded, features, kept, dropped, scores, vif_max, label="WOE ", warnings=warnings)
    kept, dropped = _apply_top_k(features, kept, dropped, scores, top_k)
    if sign_check:
        warnings.extend(_woe_sign_warnings(encoded, kept, target_arr, scores, directions))
    return SelectionResult(
        tuple(kept), tuple(dropped), scores, nan_labels_dropped, tuple(warnings),
        fit_rows=fit_rows, fit_split=fit_split,
    )


def _drop_collinear(
    frame: pd.DataFrame,
    features: list[str],
    kept: list[str],
    dropped: list[tuple[str, str]],
    scores: dict[str, dict[str, Any]],
    corr_max: float,
    *,
    label: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    for left, right, corr in find_collinear_pairs(
        correlation_matrix(frame, kept),
        kept,
        threshold=corr_max,
    ):
        if left not in kept or right not in kept:
            continue
        loser, winner = (left, right) if scores[left]["iv"] < scores[right]["iv"] else (right, left)
        kept.remove(loser)
        dropped.append((loser, f"{label}collinear with {winner} ({corr:.2f})"))
    return kept, dropped


def _drop_high_vif(
    frame: pd.DataFrame,
    features: list[str],
    kept: list[str],
    dropped: list[tuple[str, str]],
    scores: dict[str, dict[str, Any]],
    vif_max: float,
    *,
    label: str,
    warnings: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    vifs = vif(frame, kept)
    for feature in features:
        if feature in vifs:
            # FS-8: keep None (VIF unavailable) visible instead of coercing to a number.
            value = vifs[feature]
            scores.setdefault(feature, {})["vif"] = None if value is None else float(value)
    skipped = 0
    for feature, value in vifs.items():
        if feature not in kept:
            continue
        # FS-8: VIF unavailable (insufficient complete-row sample) -> do NOT drop and do
        # NOT compare None; record that the gate was skipped for this feature.
        if value is None:
            skipped += 1
            continue
        if value > vif_max:
            kept.remove(feature)
            dropped.append((feature, f"high {label}VIF {value:.1f}"))
    if skipped:
        warnings.append(
            f"{label}VIF 样本不足（完整行数不足 max(30, 2×特征数)），"
            f"已跳过 {skipped} 个特征的 VIF 门（其 vif 记为 None）"
        )
    return kept, dropped


def _apply_top_k(
    features: list[str],
    kept: list[str],
    dropped: list[tuple[str, str]],
    scores: dict[str, dict[str, Any]],
    top_k: int | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    if top_k is not None and top_k > 0 and len(kept) > top_k:
        ranked = sorted(kept, key=lambda feature: (-float(scores[feature]["iv"]), features.index(feature)))
        selected = ranked[:top_k]
        for feature in ranked[top_k:]:
            dropped.append((feature, f"outside top_k {top_k}"))
        kept = selected
    return kept, dropped


def _select_features_passthrough(
    features: list[str],
    *,
    top_k: int | None,
    nan_labels_dropped: int,
    fit_rows: int,
    fit_split: str,
) -> SelectionResult:
    """Non-binary bypass: no IV/corr/VIF statistic applies, so every candidate is kept
    (column order preserved) and only the top_k guardrail is enforced."""
    unique_features = list(dict.fromkeys(features))
    kept = unique_features
    dropped: list[tuple[str, str]] = []
    if top_k is not None and top_k > 0 and len(kept) > top_k:
        dropped = [(feature, f"outside top_k {top_k}") for feature in kept[top_k:]]
        kept = kept[:top_k]
    scores = {feature: {"space": "n/a (non-binary target)"} for feature in unique_features}
    return SelectionResult(
        tuple(kept), tuple(dropped), scores, nan_labels_dropped,
        fit_rows=fit_rows, fit_split=fit_split,
    )


def _woe_sign_warnings(
    encoded: pd.DataFrame,
    kept: list[str],
    target,
    scores: dict[str, dict[str, Any]],
    directions: dict[str, str],
) -> list[str]:
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=300, solver="lbfgs")
        model.fit(encoded[kept], pd.Series(target).astype(int))
    except Exception as exc:
        return [f"WOE sign check skipped: {exc}"]
    warnings = []
    for feature, coefficient in zip(kept, model.coef_[0], strict=True):
        coef = float(coefficient)
        scores[feature]["woe_coef"] = coef
        # WOE is ln(good_dist / bad_dist), so higher WOE should lower bad=1 risk.
        if coef > 0:
            scores[feature]["sign_warning"] = True
            message = (
                f"{feature} WOE coefficient is positive ({coef:.4f}); "
                "expected non-positive for bad=1 target"
            )
            if directions.get(feature):
                message += f", monotonic_direction={directions[feature]}"
            warnings.append(message)
        else:
            scores[feature]["sign_warning"] = False
    return warnings


def _unique(values: list[str | None]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


__all__ = ["SelectionResult", "select_features"]
