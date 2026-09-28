"""Compare complete dataset identities while preserving unknown NaN profiles."""
from dataclasses import fields, is_dataclass
import math

from marvis.data.contracts import Dataset


def dataset_identity_equal(left: Dataset | None, right: Dataset | None) -> bool:
    """Only corresponding NaNs gain equality; every other field must still match.

    Empty-table profiles have an undefined null rate. A second database decode
    creates a different NaN object, so dataclass equality alone spuriously treats
    an unchanged dataset as drift. This does not rewrite or serialize evidence.
    """
    return isinstance(left, Dataset) and isinstance(right, Dataset) and _equal(left, right)


def _equal(left, right):
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    if is_dataclass(left) and is_dataclass(right):
        return type(left) is type(right) and all(
            _equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left)
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    return left == right
