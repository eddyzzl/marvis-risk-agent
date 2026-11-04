from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from scripts.pytest_shard import pytest_collection_modifyitems, shard_config, shard_for_nodeid


@pytest.mark.parametrize("total", [4, 8])
def test_pytest_shard_assigns_every_node_once_and_deterministically(total):
    nodeids = [f"tests/test_example.py::test_case[{index}]" for index in range(400)]
    first = [shard_for_nodeid(nodeid, total) for nodeid in nodeids]
    second = [shard_for_nodeid(nodeid, total) for nodeid in nodeids]

    assert first == second
    assert set(first) == set(range(total))
    assert all(0 <= shard < total for shard in first)


@pytest.mark.parametrize("total", [4, 8])
def test_collection_hook_partitions_full_and_filtered_nodeids_exactly_once(monkeypatch, total):
    all_items = [
        SimpleNamespace(nodeid=f"tests/test_示例.py::test_case[参数-{index}]", fast=index % 5 != 0)
        for index in range(400)
    ]
    fast_items = [item for item in all_items if item.fast]
    selections = {}
    monkeypatch.setenv("PYTEST_SHARD_TOTAL", str(total))
    for name, original in (("full", all_items), ("fast", fast_items)):
        original_ids = [item.nodeid for item in original]
        selected_across_shards = Counter()
        selections[name] = []
        for index in range(total):
            monkeypatch.setenv("PYTEST_SHARD_INDEX", str(index))
            deselected = []
            config = SimpleNamespace(hook=SimpleNamespace(
                pytest_deselected=lambda *, items: deselected.extend(items),
            ))
            items = list(original)
            pytest_collection_modifyitems(config, items)
            selected_ids = [item.nodeid for item in items]
            selected_set = set(selected_ids)
            # Selection and deselection preserve the collection's order.
            assert selected_ids == [nodeid for nodeid in original_ids if nodeid in selected_set]
            assert [item.nodeid for item in deselected] == [
                nodeid for nodeid in original_ids if nodeid not in selected_set
            ]
            assert not selected_across_shards.keys() & selected_set
            selected_across_shards.update(selected_ids)
            selections[name].append(selected_ids)
        assert selected_across_shards == Counter(original_ids)

    # Marker filtering is a set intersection, so it must commute with sharding.
    fast_ids = {item.nodeid for item in fast_items}
    assert selections["fast"] == [
        [nodeid for nodeid in shard if nodeid in fast_ids]
        for shard in selections["full"]
    ]


@pytest.mark.parametrize("job_name", ["fast_tests", "full_checks"])
def test_ci_shard_matrix_covers_every_configured_index_once(job_name):
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    job = workflow.split(f"\n  {job_name}:\n", maxsplit=1)[1]
    job = re.split(r"\n  [a-z_]+:\n", job, maxsplit=1)[0]
    total = int(re.search(r"PYTEST_SHARD_TOTAL: (\d+)", job).group(1))
    if job_name == "fast_tests":
        indices = [int(value) for value in re.findall(r"^\s+index: (\d+)\s*$", job, re.MULTILINE)]
        labels = [int(value) for value in re.findall(r"- shard: (\d+)", job)]
        assert labels == [index + 1 for index in indices]
    else:
        values = re.search(r"index: \[([^\]]+)\]", job).group(1)
        indices = [int(value.strip()) for value in values.split(",")]
    assert total == 8
    assert Counter(indices) == Counter(range(total))
    assert "PYTEST_SHARD_INDEX: ${{ matrix.index }}" in job
    assert "-p scripts.pytest_shard" in job
    assert "--durations=25" in job


def test_pytest_shard_is_disabled_without_environment(monkeypatch):
    monkeypatch.delenv("PYTEST_SHARD_INDEX", raising=False)
    monkeypatch.delenv("PYTEST_SHARD_TOTAL", raising=False)
    assert shard_config() is None


def test_pytest_shard_accepts_last_of_eight_indices(monkeypatch):
    monkeypatch.setenv("PYTEST_SHARD_INDEX", "7")
    monkeypatch.setenv("PYTEST_SHARD_TOTAL", "8")
    assert shard_config() == (7, 8)


@pytest.mark.parametrize(
    ("index", "total"),
    [
        ("0", None),
        (None, "4"),
        ("x", "4"),
        ("4", "4"),
        ("8", "8"),
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
