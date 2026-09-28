"""Preflight the inputs of the shared preprocessing chain, without evaluating it."""

import pandas as pd

from marvis.feature.derived_preprocessing import replay_derived_step
from marvis.feature.errors import FeatureError
from marvis.reference_decision.contracts import DecisionError


def feature_outputs(raw_names, steps):
    available = set(raw_names)
    generated = set()
    for step in steps:
        kind = step.get("kind")
        columns = step.get("columns") or []
        params = step.get("params") or {}
        if not columns or not set(columns) <= available:
            raise DecisionError("preprocessing_inputs_missing")
        outputs = set()
        if kind in {"sentinel", "impute", "cap", "normalize"}:
            pass
        elif kind == "missing_indicator":
            outputs = {str(params[c]) for c in columns}
        elif kind in {"woe", "categorical_woe"}:
            outputs = {f"{c}_woe" for c in columns}
        elif kind == "onehot":
            # pandas.get_dummies used by the existing encoder uses this convention.
            outputs = {f"{c}_{v}" for c in columns for v in params.get(c, [])}
            available -= set(columns)
        elif kind == "derive":
            # Ask the existing recipe implementation for its output names; no
            # second arithmetic/date naming or evaluation dialect is introduced.
            recipe = params.get("recipe") or []
            if any(
                r.get("kind") == "transform" and r.get("ops") != ["log1p"]
                for r in recipe
            ):
                raise DecisionError("unsupported_online_transform")
            try:
                frame = replay_derived_step(pd.DataFrame(columns=sorted(available)), step)
            except FeatureError as exc:
                raise DecisionError("derived_feature_contract_invalid") from exc
            outputs = set(frame.columns) - available
        elif kind == "group_aggregate":
            outputs = {
                f"{params['value']}_by_{params['group']}_{agg}"
                for agg in params["aggs"]
            }
        elif kind == "fitted_rank":
            outputs = {f"{params['column']}__rank"}
        else:
            raise DecisionError("unsupported_preprocessing_kind")
        if outputs & set(raw_names):
            raise DecisionError("derived_features_cannot_be_raw_inputs")
        generated |= outputs
        available |= outputs
    return available, generated
