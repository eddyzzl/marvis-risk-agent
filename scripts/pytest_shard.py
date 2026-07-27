"""Deterministically shard collected pytest items for parallel CI jobs.

Load with ``-p scripts.pytest_shard`` and set both ``PYTEST_SHARD_INDEX``
(zero-based) and ``PYTEST_SHARD_TOTAL``. Every collected node id is assigned to
exactly one shard using a stable hash; no test is skipped from the aggregate
matrix.
"""

from __future__ import annotations

import hashlib
import os

import pytest


def shard_for_nodeid(nodeid: str, total: int) -> int:
    if total < 1:
        raise ValueError("total must be at least 1")
    digest = hashlib.blake2b(
        nodeid.encode("utf-8"),
        digest_size=8,
        person=b"marvis-ci",
    ).digest()
    return int.from_bytes(digest, "big") % total


def shard_config() -> tuple[int, int] | None:
    raw_index = os.getenv("PYTEST_SHARD_INDEX")
    raw_total = os.getenv("PYTEST_SHARD_TOTAL")
    if raw_index is None and raw_total is None:
        return None
    if raw_index is None or raw_total is None:
        raise pytest.UsageError(
            "PYTEST_SHARD_INDEX and PYTEST_SHARD_TOTAL must be set together"
        )
    try:
        index = int(raw_index)
        total = int(raw_total)
    except ValueError as exc:
        raise pytest.UsageError("pytest shard values must be integers") from exc
    if total < 1 or index < 0 or index >= total:
        raise pytest.UsageError(
            "pytest shard requires total >= 1 and 0 <= index < total"
        )
    return index, total


def pytest_collection_modifyitems(config, items) -> None:
    resolved = shard_config()
    if resolved is None:
        return
    index, total = resolved
    selected = [
        item
        for item in items
        if shard_for_nodeid(item.nodeid, total) == index
    ]
    deselected = [
        item
        for item in items
        if shard_for_nodeid(item.nodeid, total) != index
    ]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


__all__ = ["shard_config", "shard_for_nodeid"]
