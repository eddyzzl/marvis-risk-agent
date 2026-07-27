from __future__ import annotations

from datetime import date
import hashlib

import numpy as np
import pandas as pd
import pytest

from marvis.packs.modeling.score_evidence import (
    MAX_GOVERNED_SCORE_MONTHS,
    ModelScoreEvidenceError,
    build_single_model_score_evidence,
    normalize_governed_months,
)
from marvis.packs.strategy.model_evidence import build_artifact_ref
from marvis.packs.strategy.sample_design_v2 import (
    build_strategy_sample_design_v2,
    build_strategy_sample_design_v2_bundle,
)
from tests.test_strategy_sample_design_v2 import (
    _bundle,
    _components,
    _decoded_membership,
    _design_kwargs,
    _diagnostic_statistics,
    _metric_observations,
    _policy,
    _source_ref,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(label: str, kind: str) -> dict[str, str]:
    return build_artifact_ref(
        kind=kind,
        ref_id=label,
        content_hash=_hash(label),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 1, 0],
            "apply_month": [
                "202601",
                202601,
                "2026-01",
                "202602",
                "202603",
                "2026-02",
                "202603",
                "202604",
            ],
            "x": np.arange(8, dtype=float),
        }
    )


def _build(
    *,
    bundle=None,
    masks=None,
    frame=None,
    scores=None,
) -> dict:
    sample_bundle = bundle or _bundle()
    membership = masks or _decoded_membership()["masks"]
    return build_single_model_score_evidence(
        sample_design_bundle=sample_bundle,
        membership_masks=membership,
        frame=_frame() if frame is None else frame,
        scores=np.asarray(
            [0.1, 0.9, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4] if scores is None else scores,
            dtype=np.float64,
        ),
        training_evidence_ref=_artifact(
            "training-evidence",
            "modeling_training_evidence_json",
        ),
        model_ref=_artifact("model-binary", "modeling_model_binary"),
        score_ref=_artifact("score-vector", "model_score_vector_parquet"),
        features=["x"],
    )


def _observation(
    evidence: dict,
    metric: str,
    population: str,
    partition: str,
    *,
    bin_id: str | None = None,
    period: str | None = None,
) -> dict:
    matches = [
        item
        for item in evidence["observations"]
        if item["metric_key"] == metric
        and item["sample_ref"]["population"] == population
        and item["sample_ref"]["partition"] == partition
        and item["bin_id"] == bin_id
        and item["period"] == period
    ]
    assert len(matches) == 1
    return matches[0]


def test_score_evidence_uses_common_development_bins_and_conserves_zero_bins() -> None:
    evidence = _build()

    assert len(evidence["score_bins"]) == 2
    low, high = evidence["score_bins"]
    assert (low["lower_bound"], low["upper_bound"]) == (None, 0.5)
    assert (high["lower_bound"], high["upper_bound"]) == (0.5, None)
    assert (
        _observation(
            evidence,
            "score_bin_count",
            "approval",
            "validation",
            bin_id=high["bin_id"],
        )["value"]
        == 0
    )

    for population in ("approval", "risk"):
        for partition in ("development", "validation", "oot"):
            counts = [
                _observation(
                    evidence,
                    "score_bin_count",
                    population,
                    partition,
                    bin_id=item["bin_id"],
                )["value"]
                for item in evidence["score_bins"]
            ]
            shares = [
                _observation(
                    evidence,
                    "score_bin_share",
                    population,
                    partition,
                    bin_id=item["bin_id"],
                )["value"]
                for item in evidence["score_bins"]
            ]
            expected = int(
                np.count_nonzero(
                    _decoded_membership()["masks"][f"{population}/{partition}"]
                )
            )
            assert sum(counts) == expected
            assert sum(shares) == pytest.approx(1.0)

    assert _observation(evidence, "auc", "risk", "development")["value"] == 1.0
    assert _observation(evidence, "score_psi", "risk", "development")["value"] == 0.0


def test_score_evidence_constant_score_is_one_unbounded_bin() -> None:
    evidence = _build(scores=np.full(8, 0.42, dtype=np.float64))

    assert len(evidence["score_bins"]) == 1
    assert evidence["score_bins"][0]["lower_bound"] is None
    assert evidence["score_bins"][0]["upper_bound"] is None


def test_score_evidence_never_flips_higher_risk_score_direction() -> None:
    evidence = _build(scores=np.asarray([0.9, 0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4]))

    assert _observation(evidence, "auc", "risk", "development")["value"] == 0.0


def test_empty_unlabeled_and_single_class_oot_are_explicitly_unavailable() -> None:
    empty_bundle = _bundle(empty_oot=True)
    empty_masks = _decoded_membership(empty_oot=True)["masks"]
    empty = _build(bundle=empty_bundle, masks=empty_masks)
    assert (
        _observation(
            empty,
            "score_bin_count",
            "risk",
            "oot",
            bin_id=empty["score_bins"][0]["bin_id"],
        )["status"]
        == "unavailable"
    )
    assert _observation(empty, "auc", "risk", "oot")["reason"] == (
        "empty_sample_partition"
    )

    unlabeled_frame = _frame()
    unlabeled_frame.loc[[4, 6], "target"] = np.nan
    unlabeled = _build(frame=unlabeled_frame)
    assert _observation(unlabeled, "auc", "risk", "oot")["reason"] == (
        "no_labeled_rows"
    )

    single_class_frame = _frame()
    single_class_frame.loc[[4, 6], "target"] = 1
    single_class = _build(frame=single_class_frame)
    assert _observation(single_class, "ks", "risk", "oot")["reason"] == (
        "requires_two_labeled_classes"
    )
    assert _observation(
        single_class,
        "score_bin_bad_rate",
        "risk",
        "oot",
        bin_id=single_class["score_bins"][0]["bin_id"],
    )["status"] in {"present", "unavailable"}


def _bundle_without_month() -> dict:
    decoded = _decoded_membership()
    approval, risk, target, historical = _components(decoded)
    kwargs = _design_kwargs()
    kwargs["field_bindings"] = {
        **kwargs["field_bindings"],
        "month_field": None,
    }
    design = build_strategy_sample_design_v2(
        task_id="task-v2",
        membership_header=decoded["header"],
        relationship="nested_same_cohort",
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_policy(),
        source_refs=[_source_ref("design-a"), _source_ref("design-b")],
        **kwargs,
    )
    return build_strategy_sample_design_v2_bundle(
        task_id="task-v2",
        membership_header=decoded["header"],
        membership_masks=decoded["masks"],
        relationship="nested_same_cohort",
        target_selector=target,
        approval_population=approval,
        risk_population=risk,
        historical_score=historical,
        policy=_policy(),
        diagnostic_statistics=_diagnostic_statistics(decoded),
        metric_observations=_metric_observations(
            decoded,
            design,
            maturity_status="confirmed_matured",
        ),
        source_refs=[_source_ref("design-a"), _source_ref("design-b")],
        **kwargs,
    )


def test_month_metrics_are_absent_without_binding_and_yyyymm_is_canonicalized() -> None:
    absent = _build(bundle=_bundle_without_month())
    assert all(item["period"] is None for item in absent["observations"])

    assert normalize_governed_months([202601.0]) == ("2026-01",)
    evidence = _build()
    periods = {item["period"] for item in evidence["observations"] if item["period"]}
    assert periods == {"2026-01", "2026-02", "2026-03", "2026-04"}


def test_month_values_fail_closed_on_invalid_value_or_distinct_month_budget() -> None:
    invalid = _frame()
    invalid.loc[3, "apply_month"] = "202613"
    with pytest.raises(ModelScoreEvidenceError, match="month"):
        _build(frame=invalid)

    months = []
    year, month = 2000, 1
    for _ in range(MAX_GOVERNED_SCORE_MONTHS + 1):
        months.append(date(year, month, 1).strftime("%Y%m"))
        month += 1
        if month == 13:
            year += 1
            month = 1
    with pytest.raises(ModelScoreEvidenceError, match="month budget"):
        normalize_governed_months(months)
