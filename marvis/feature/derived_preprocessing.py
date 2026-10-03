"""Compile explicit derivation recipes into the existing preprocessing chain."""

from __future__ import annotations

from marvis.feature.derive import (
    cross_arithmetic,
    derive_date_features,
    transform_feature,
)
from marvis.feature.derived_parameters import (
    apply_aggregate_parameters,
    apply_rank_parameters,
    fit_aggregate_parameters,
    fit_rank_parameters,
)
from marvis.feature.errors import FeatureError
from marvis.feature.fit_scope import fit_membership


def derive_with_parameters(frame, recipe, *, dataset_id, target_col=None, before_fit=None):
    out, columns, steps, fits = frame.copy(), [], [], []
    for item in recipe:
        kind = item.get("kind")
        inputs = recipe_inputs(item, out.columns)
        if target_col and target_col in inputs:
            raise FeatureError(
                "derived features cannot use the registered target column"
            )
        if kind == "agg":
            mask, _ = fit_membership(
                out, item, tool="aggregate_feature", dataset_id=dataset_id
            )
            if before_fit is not None:
                before_fit(out, mask, inputs, steps, fits)
            params = fit_aggregate_parameters(
                out,
                str(item["group"]),
                str(item["value"]),
                list(item["aggs"]),
                fit_mask=mask,
                min_group_size=item.get("min_group_size", 30),
                target_col=target_col or item.get("target_col"),
            )
            out, added = apply_aggregate_parameters(out, params)
            steps.append(
                {
                    "kind": "group_aggregate",
                    "columns": [params["group"]],
                    "params": params,
                }
            )
            fits.append(("aggregate_feature", item))
        elif kind == "transform":
            added = []
            for op in item["ops"]:
                if op == "rank":
                    mask, _ = fit_membership(
                        out, item, tool="rank", dataset_id=dataset_id
                    )
                    if before_fit is not None:
                        before_fit(out, mask, [str(item["col"])], steps, fits)
                    params = fit_rank_parameters(out, str(item["col"]), mask)
                    out, produced = apply_rank_parameters(out, params)
                    steps.append(
                        {
                            "kind": "fitted_rank",
                            "columns": [str(item["col"])],
                            "params": params,
                        }
                    )
                    fits.append(("rank", item))
                else:
                    local = {"kind": "transform", "col": item["col"], "ops": [op]}
                    out, produced = _row_local(out, local)
                    steps.append(
                        {
                            "kind": "derive",
                            "columns": inputs,
                            "params": {"recipe": [local]},
                        }
                    )
                added.extend(produced)
        else:
            out, added = _row_local(out, item)
            steps.append(
                {
                    "kind": "derive",
                    "columns": inputs,
                    "params": {"recipe": [dict(item)]},
                }
            )
        columns.extend(added)
    return out, columns, steps, fits


def recipe_inputs(item, columns):
    kind = item.get("kind")
    fields = {
        "cross": ("a", "b"),
        "ratio": ("num", "den"),
        "agg": ("group", "value"),
        "transform": ("col",),
        "month": ("col",),
        "datediff": ("col",),
        "tenure_months": ("col",),
    }
    if kind not in fields:
        raise FeatureError(f"unsupported derive recipe kind: {kind}")
    result = [str(item[key]) for key in fields[kind]]
    if kind in {"datediff", "tenure_months"} and item.get("anchor") in columns:
        result.append(str(item["anchor"]))
    return list(dict.fromkeys(result))


def _row_local(frame, item):
    kind = item.get("kind")
    if kind == "cross":
        return cross_arithmetic(
            frame, str(item["a"]), str(item["b"]), list(item["ops"])
        )
    if kind == "ratio":
        return cross_arithmetic(frame, str(item["num"]), str(item["den"]), ["ratio"])
    if kind == "transform" and set(item["ops"]) <= {"log1p"}:
        return transform_feature(frame, str(item["col"]), list(item["ops"]))
    if kind in {"month", "datediff", "tenure_months"}:
        return derive_date_features(frame, [item])
    raise FeatureError(
        "population-dependent derivation requires frozen fitted parameters"
    )


def replay_derived_step(frame, step):
    kind, params = step["kind"], step["params"]
    if kind == "group_aggregate":
        return apply_aggregate_parameters(frame, params)[0]
    if kind == "fitted_rank":
        return apply_rank_parameters(frame, params)[0]
    out = frame
    for item in params["recipe"]:
        out, _ = _row_local(out, item)
    return out
