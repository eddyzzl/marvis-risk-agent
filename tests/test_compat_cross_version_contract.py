"""Cross-version compatibility contract for legacy V1 vs native V2 sampling.

This file pins the invariants declared in ``docs/roadmap.md`` as focused unit
tests so a regression in either the legacy ``strategy_sample_design +
partition=development`` path or the native ``strategy_sample_design_v2 +
risk/development`` path is caught in one place:

1. ``target_bad_value=0`` normalizes to the internal ``1=bad`` semantics on
   both paths.
2. Native membership never implicitly intersects the approval and risk
   populations; the persisted ``risk/development`` mask is consumed verbatim.
3. Empty labels are excluded from the risk denominator only on explicit user
   consent, and the overall population is always retained.
4. The legacy ``strategy_sample_design + development`` canonical token bytes
   stay stable (golden contract).  The matching sample-context hash and bundle
   id/hash golden values are already pinned in
   ``tests/test_strategy_candidate_fragment.py`` and
   ``tests/test_strategy_sample_design_v2.py``.
5. Corrupted latest native evidence fails typed and never silently falls back
   to an older V1 sample.  That invariant is exercised end-to-end in
   ``tests/test_strategy_v2_agent_workflows.py::test_candidate_workflow_never_falls_back_to_older_v1_after_native_sample``
   and
   ``tests/test_strategy_project_context_native_v2.py::test_project_context_fails_closed_when_latest_native_evidence_is_corrupt``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design import build_strategy_sample_design_bundle
from marvis.packs.strategy.sample_design_execution import StrategyRiskDevelopmentRef
from marvis.packs.strategy.sample_design_v2 import build_target_selector_v2
from marvis.packs.strategy.sample_design_v2_tools import (
    _bad_label_mask,
    _binary_label_mask,
)
from marvis.packs.strategy.sample_membership import (
    decode_sample_membership,
    encode_sample_membership,
)


def _legacy_bundle(
    *,
    target_bad_value: int,
    drop_nan_labels: bool,
    frame: pd.DataFrame | None = None,
) -> dict:
    if frame is None:
        frame = pd.DataFrame(
            {"bad": [0, 1, 1, 0, 0], "sample_split": ["dev"] * 5}
        )
    return build_strategy_sample_design_bundle(
        frame=frame,
        task_id="task-compat",
        dataset_id="dataset-compat",
        dataset_content_hash="a" * 64,
        workspace_revision=1,
        workspace_generation=2,
        semantic_mapping_hash="b" * 64,
        target_col="bad",
        target_bad_value=target_bad_value,
        drop_nan_labels=drop_nan_labels,
        performance_window={"status": "provided", "days": 30},
        observation_window={
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-03-31",
        },
        split_definition={
            "status": "available",
            "column": "sample_split",
            "development_values": ["dev"],
            "validation_values": [],
            "oot_values": [],
        },
        maturity="confirmed_matured",
    )


def _overall_metrics(bundle: dict) -> dict[str, dict]:
    key_by_id = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    return {
        key_by_id[item["metric_definition_ref"]["metric_definition_id"]]: item
        for item in bundle["metric_observations"]
        if item["dimension"] == {"kind": "overall", "value": "overall"}
    }


def test_legacy_target_bad_value_zero_normalizes_to_internal_bad_semantics():
    # Raw polarity is preserved explicitly on the receipt while the deterministic
    # metric kernel counts raw-0 rows as bad (internal 1=bad semantics).
    bundle = _legacy_bundle(target_bad_value=0, drop_nan_labels=True)

    assert bundle["sample_design"]["target_definition"] == {
        "column": "bad",
        "good_value": 1,
        "bad_value": 0,
        "drop_nan_labels": True,
    }
    overall = _overall_metrics(bundle)
    # frame bad = [0, 1, 1, 0, 0] -> the three raw 0s are the bad rows.
    assert overall["bad_count"]["value"] == 3
    assert overall["good_count"]["value"] == 2
    assert overall["bad_rate"]["value"] == pytest.approx(3 / 5)


def test_native_bad_label_kernel_normalizes_target_bad_value_zero_to_internal_bad():
    # The native statistics kernel marks rows equal to ``target_bad_value`` as
    # bad; with target_bad_value=0 the raw 0s become the internal bad rows.
    series = pd.Series([0.0, 1.0, 0.0, 1.0, float("nan")])

    assert _bad_label_mask(series, bad_value=0).tolist() == [
        True,
        False,
        True,
        False,
        False,
    ]
    assert _binary_label_mask(series).tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]


def test_native_membership_does_not_implicitly_intersect_approval_and_risk():
    # A risk/development row that lives entirely outside approval must survive
    # the codec untouched; the codec never ANDs risk masks with approval masks.
    masks = {
        "approval/development": np.array([True, False, False, False, False, False]),
        "approval/validation": np.array([False, True, False, False, False, False]),
        "approval/oot": np.array([False, False, True, False, False, False]),
        "risk/development": np.array([False, False, False, True, False, False]),
        "risk/validation": np.array([False, True, False, False, False, False]),
        "risk/oot": np.array([False, False, False, False, False, False]),
    }

    decoded = decode_sample_membership(
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash="a" * 64,
            masks=masks,
        )
    )

    # The risk/development row (index 3) is not in any approval mask.
    np.testing.assert_array_equal(
        decoded["masks"]["risk/development"], masks["risk/development"]
    )
    assert not np.any(
        decoded["masks"]["risk/development"]
        & decoded["masks"]["approval/development"]
    )
    assert decoded["header"]["counts"]["relationship"] == {
        "risk_within_approval": {
            "development": 0,
            "validation": 1,
            "oot": 0,
            "total": 1,
        },
        "risk_outside_approval": {
            "development": 1,
            "validation": 0,
            "oot": 0,
            "total": 1,
        },
    }


def test_empty_labels_excluded_only_with_explicit_consent_and_population_retained():
    frame = pd.DataFrame(
        {
            "bad": [0.0, 1.0, np.nan, 0.0, 1.0],
            "sample_split": ["dev"] * 5,
        }
    )

    with pytest.raises(StrategyError, match="drop_nan_labels=true"):
        _legacy_bundle(
            target_bad_value=1,
            drop_nan_labels=False,
            frame=frame,
        )

    bundle = _legacy_bundle(
        target_bad_value=1,
        drop_nan_labels=True,
        frame=frame,
    )
    overall = _overall_metrics(bundle)
    # The overall population is retained even though the NaN label is excluded
    # from the risk denominator.
    assert overall["population_count"]["value"] == 5
    assert overall["labeled_count"]["value"] == 4
    assert overall["bad_count"]["value"] == 2
    assert overall["bad_rate"]["value"] == pytest.approx(2 / 4)


def test_native_drop_missing_requires_explicit_boolean_and_empty_labels_are_not_bad():
    # The native target selector only accepts an explicit boolean consent flag.
    with pytest.raises(StrategyError, match="drop_missing must be a boolean"):
        build_target_selector_v2(
            status="resolved",
            column="bad",
            good_value=1,
            bad_value=0,
            drop_missing="yes",
            source_refs=[{"kind": "test", "ref_id": "x", "content_hash": "c" * 64}],
        )

    # Empty labels are neither labeled nor bad: they never inflate the risk
    # numerator and are only omitted from the labeled denominator.
    series = pd.Series([0.0, 1.0, float("nan")])
    assert _binary_label_mask(series).tolist() == [True, True, False]
    assert _bad_label_mask(series, bad_value=1).tolist() == [False, True, False]


def test_legacy_strategy_sample_design_development_token_is_golden():
    # The canonical source token for a legacy ``strategy_sample_design +
    # development`` reference must stay byte-identical across versions.
    reference = StrategyRiskDevelopmentRef.from_value(
        {
            "artifact_id": "d" * 64,
            "artifact_content_hash": "e" * 64,
            "sample_design_id": "strategy-sample-design-test",
            "sample_design_content_hash": "f" * 64,
            "partition": "development",
        }
    )
    token = "strategy-sample-design:" + json.dumps(
        {"kind": "strategy_sample_design", **reference.to_ref_dict()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    assert token == (
        "strategy-sample-design:"
        '{"artifact_content_hash":"'
        + "e" * 64
        + '","artifact_id":"'
        + "d" * 64
        + '","kind":"strategy_sample_design","partition":"development",'
        '"sample_design_content_hash":"'
        + "f" * 64
        + '","sample_design_id":"strategy-sample-design-test"}'
    )
