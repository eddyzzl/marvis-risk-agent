"""Independent bounded reference calculations for native governance carriers."""
import json
import math

from .runtime_contracts import digest


def governance_input_reference(value):
    """Recompute the installed type-preserving input commitment independently."""
    remaining = 100000

    def typed(item, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            raise ValueError("safety_input_reference_budget")
        if item is None:
            return ["null"]
        if type(item) is bool:
            return ["bool", item]
        if type(item) is int:
            return ["int", item]
        if type(item) is float:
            tag = "nan" if math.isnan(item) else "+inf" if item == math.inf else "-inf" if item == -math.inf else item
            return ["float", tag]
        if type(item) is str:
            return ["str", item]
        if type(item) in {list, tuple}:
            return ["list", [typed(child, depth + 1) for child in item]]
        if type(item) is dict:
            entries = [[typed(key, depth + 1), typed(child, depth + 1)] for key, child in item.items()]
            entries.sort(key=lambda pair: json.dumps(pair[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
            return ["dict", entries]
        raise ValueError("safety_input_reference_type")

    return digest(["marvis-governance-canonical.v1", typed(value)])
