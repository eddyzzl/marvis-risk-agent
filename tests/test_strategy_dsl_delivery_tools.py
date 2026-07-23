from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from marvis.files import sha256_file
from marvis.packs.strategy.dsl_delivery import (
    validate_strategy_delivery_equivalence,
)
from marvis.packs.strategy.dsl_delivery_tools import (
    DELIVERY_ARTIFACT_KINDS,
    DELIVERY_AUDIT_KIND,
    StrategyDeliveryToolError,
    run_export_strategy_delivery,
    validate_export_strategy_delivery_tool_output,
)
from marvis.packs.strategy.tools import _runtime
from test_strategy_apply_tool import _runtime_fixture


def _inputs(fixture: tuple) -> dict:
    settings, _task, _registry, dataset, strategy, _ctx = fixture
    strategies = _runtime(fixture[-1]).strategies
    meta = strategies.get_strategy_meta(strategy.id)
    assert meta is not None
    spec_hash = strategies.get_strategy_spec_hash(strategy.id)
    assert spec_hash is not None
    return {
        "strategy_ref": {
            "strategy_id": strategy.id,
            "expected_strategy_type": strategy.strategy_type,
            "expected_version": meta["version"],
            "expected_spec_hash": spec_hash,
        },
        "dataset_ref": {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
        },
        "maximum_equivalence_rows": 4096,
    }


def _run(fixture: tuple, inputs: dict | None = None) -> tuple[dict, object]:
    runtime = _runtime(fixture[-1])
    request = _inputs(fixture) if inputs is None else inputs
    return (
        run_export_strategy_delivery(request, fixture[-1], runtime),
        runtime,
    )


def _artifact_rows(runtime, task_id: str) -> list[dict]:
    return [
        row
        for row in runtime.task_artifacts.list_for_task(task_id)
        if row["kind"] in set(DELIVERY_ARTIFACT_KINDS.values())
    ]


def _audit_count(settings, *, delivery_id: str) -> int:
    with sqlite3.connect(settings.db_path) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind = ? AND target_ref = ?",
                (DELIVERY_AUDIT_KIND, delivery_id),
            ).fetchone()[0]
        )


@pytest.mark.parametrize(
    "strategy_type",
    ["approval", "reject", "limit", "pricing", "segmentation"],
)
def test_export_strategy_delivery_publishes_authenticated_code_and_equivalence(
    tmp_path: Path,
    strategy_type: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, strategy_type)
    request = _inputs(fixture)

    output, runtime = _run(fixture, request)

    assert validate_export_strategy_delivery_tool_output(
        output,
        expected_task_id=fixture[1].id,
        expected_strategy_ref=request["strategy_ref"],
        expected_dataset_ref=request["dataset_ref"],
    ) == output
    assert output["strategy_type"] == strategy_type
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert {item["kind"] for item in output["artifacts"]} == set(
        DELIVERY_ARTIFACT_KINDS.values()
    )
    rows = _artifact_rows(runtime, fixture[1].id)
    assert len(rows) == 4
    for artifact in output["artifacts"]:
        row = next(item for item in rows if item["id"] == artifact["artifact_id"])
        path = Path(row["path"])
        assert path.is_file()
        assert sha256_file(path) == artifact["content_hash"]
        assert row["provenance"]["delivery_id"] == output["delivery_id"]
        assert row["provenance"]["strategy_ref"] == request["strategy_ref"]
        assert row["provenance"]["dataset_ref"] == request["dataset_ref"]
    python_path = Path(
        next(
            row["path"]
            for row in rows
            if row["kind"] == DELIVERY_ARTIFACT_KINDS["python"]
        )
    )
    python_source = python_path.read_text(encoding="utf-8").lower()
    assert "from marvis" not in python_source
    assert "import marvis" not in python_source
    equivalence = output["equivalence"]
    assert validate_strategy_delivery_equivalence(
        equivalence,
        expected_strategy_spec_hash=request["strategy_ref"][
            "expected_spec_hash"
        ],
        expected_sample_hash=equivalence["sample_hash"],
        expected_content_hash=equivalence["content_hash"],
    ) == equivalence
    assert _audit_count(
        fixture[0],
        delivery_id=output["delivery_id"],
    ) == 1


def test_export_strategy_delivery_exact_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)

    first, runtime = _run(fixture, request)
    second, _runtime_again = _run(fixture, request)

    assert second == first
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 1


def test_export_strategy_delivery_rejects_strategy_or_dataset_drift(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    wrong_strategy = deepcopy(request)
    wrong_strategy["strategy_ref"]["expected_spec_hash"] = "f" * 64
    with pytest.raises(StrategyDeliveryToolError, match="strategy.*exact"):
        _run(fixture, wrong_strategy)

    wrong_dataset = deepcopy(request)
    wrong_dataset["dataset_ref"]["expected_content_hash"] = "e" * 64
    with pytest.raises(StrategyDeliveryToolError, match="dataset.*exact"):
        _run(fixture, wrong_dataset)


def test_export_strategy_delivery_rolls_back_all_files_and_rows_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    original = runtime.task_artifacts.register_on_connection
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected registry failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        runtime.task_artifacts,
        "register_on_connection",
        fail_second,
    )

    with pytest.raises(RuntimeError, match="injected registry failure"):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert _artifact_rows(runtime, fixture[1].id) == []
    delivery_root = (
        Path(fixture[0].tasks_dir)
        / fixture[1].id
        / "strategy_delivery"
    )
    assert not list(delivery_root.rglob("*.py"))
    assert not list(delivery_root.rglob("*.sql"))
    assert not list(delivery_root.rglob("*.json"))
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind = ?",
                (DELIVERY_AUDIT_KIND,),
            ).fetchone()[0]
            == 0
        )


def test_delivery_output_validator_requires_external_exact_refs(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    forged["strategy_ref"]["expected_spec_hash"] = "0" * 64
    body = {
        key: value
        for key, value in forged["equivalence"].items()
        if key not in {"equivalence_id", "content_hash"}
    }
    body["strategy_spec_hash"] = "0" * 64
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    forged["equivalence"] = {
        **body,
        "equivalence_id": "strategy-dsl-equivalence-" + digest[:24],
    }
    forged["equivalence"]["content_hash"] = hashlib.sha256(
        json.dumps(
            forged["equivalence"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(StrategyDeliveryToolError, match="strategy_ref"):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
        )
