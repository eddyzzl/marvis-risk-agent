from __future__ import annotations

import pytest

from scripts.pytest_shard import shard_config, shard_for_nodeid


def test_pytest_shard_assigns_every_node_once_and_deterministically():
    nodeids = [f"tests/test_example.py::test_case[{index}]" for index in range(400)]
    first = [shard_for_nodeid(nodeid, 4) for nodeid in nodeids]
    second = [shard_for_nodeid(nodeid, 4) for nodeid in nodeids]

    assert first == second
    assert set(first) == {0, 1, 2, 3}
    assert all(0 <= shard < 4 for shard in first)


def test_pytest_shard_is_disabled_without_environment(monkeypatch):
    monkeypatch.delenv("PYTEST_SHARD_INDEX", raising=False)
    monkeypatch.delenv("PYTEST_SHARD_TOTAL", raising=False)
    assert shard_config() is None


@pytest.mark.parametrize(
    ("index", "total"),
    [
        ("0", None),
        (None, "4"),
        ("x", "4"),
        ("4", "4"),
        ("-1", "4"),
        ("0", "0"),
    ],
)
def test_pytest_shard_rejects_invalid_environment(monkeypatch, index, total):
    if index is None:
        monkeypatch.delenv("PYTEST_SHARD_INDEX", raising=False)
    else:
        monkeypatch.setenv("PYTEST_SHARD_INDEX", index)
    if total is None:
        monkeypatch.delenv("PYTEST_SHARD_TOTAL", raising=False)
    else:
        monkeypatch.setenv("PYTEST_SHARD_TOTAL", total)

    with pytest.raises(pytest.UsageError):
        shard_config()
