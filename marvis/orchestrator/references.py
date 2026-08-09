from __future__ import annotations


_REFERENCE_PREFIX = "$ref:"
_OUTPUT_MARKER = ".output"


def parse_step_output_ref(value: str) -> tuple[str, str]:
    """Parse ``$ref:<step-id>.output[.<path>]`` without owning caller errors."""

    if not isinstance(value, str) or not value.startswith(_REFERENCE_PREFIX):
        raise ValueError(f"invalid ref {value}")
    raw = value[len(_REFERENCE_PREFIX) :]
    if _OUTPUT_MARKER not in raw:
        raise ValueError(f"invalid ref {value}")
    step_id, tail = raw.split(_OUTPUT_MARKER, 1)
    if not step_id:
        raise ValueError(f"invalid ref {value}")
    if not tail:
        return step_id, ""
    if not tail.startswith(".") or tail == ".":
        raise ValueError(f"invalid ref {value}")
    return step_id, tail[1:]


__all__ = ["parse_step_output_ref"]
