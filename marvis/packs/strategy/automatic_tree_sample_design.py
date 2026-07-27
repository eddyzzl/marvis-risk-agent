"""Exact governed sample-design reference carried by automatic-tree assets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hmac
import json

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentRef,
)


_SOURCE_REF_PREFIX = "strategy-sample-design:"


def sample_design_ref_from_automatic_tree_source_refs(
    source_refs: object,
) -> dict[str, str]:
    """Recover the one exact canonical sample-design reference from an asset."""

    if isinstance(source_refs, (str, bytes, bytearray)) or not isinstance(
        source_refs, Sequence
    ):
        raise StrategyError("automatic-tree source_refs must be an array")
    tokens = [
        value
        for value in source_refs
        if isinstance(value, str) and value.startswith(_SOURCE_REF_PREFIX)
    ]
    if len(tokens) != 1:
        raise StrategyError(
            "automatic-tree lineage must contain exactly one governed "
            "sample-design reference"
        )
    token = tokens[0]
    try:
        payload = json.loads(token.removeprefix(_SOURCE_REF_PREFIX))
    except json.JSONDecodeError as exc:
        raise StrategyError(
            "automatic-tree sample-design source reference is invalid"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StrategyError("automatic-tree sample-design source reference is invalid")
    kind = payload.get("kind")
    if kind not in {"strategy_sample_design", "strategy_sample_design_v2"}:
        raise StrategyError(
            "automatic-tree sample-design source reference is invalid"
        )
    reference = StrategyRiskDevelopmentRef.from_value(
        {key: value for key, value in payload.items() if key != "kind"}
    )
    expected_partition = (
        "development"
        if kind == "strategy_sample_design"
        else "risk/development"
    )
    if reference.partition != expected_partition:
        raise StrategyError(
            "automatic-tree sample-design source kind and partition "
            "are inconsistent"
        )
    canonical_payload = {"kind": kind, **reference.to_ref_dict()}
    canonical_token = _SOURCE_REF_PREFIX + json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if not hmac.compare_digest(token, canonical_token):
        raise StrategyError(
            "automatic-tree sample-design source reference is not canonical"
        )
    return reference.to_ref_dict()


__all__ = ["sample_design_ref_from_automatic_tree_source_refs"]
