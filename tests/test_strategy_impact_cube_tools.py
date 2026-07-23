from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path

import pandas as pd
import pytest

from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube import (
    canonical_strategy_impact_cube_json,
)
from marvis.packs.strategy.impact_cube_tools import (
    IMPACT_CUBE_ARTIFACT_KIND,
    IMPACT_CUBE_TOOL_SCHEMA_VERSION,
    run_measure_strategy_impact_cube,
    validate_measure_strategy_impact_cube_tool_output,
)
import marvis.packs.strategy.impact_cube_tools as impact_tools
from marvis.packs.strategy.pool import compile_strategy_pool
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_pool_validation_tools import _setup as _pool_setup


def _setup(tmp_path: Path) -> dict:
    fx = _pool_setup(tmp_path)
    request = {
        "strategy_type": "approval",
        "pool_ref": copy.deepcopy(
            fx["validation_request"]["pool_ref"]
        ),
        "sample_design_ref": copy.deepcopy(fx["sample_ref"]),
        "partitions": ["development", "validation"],
        "population": "risk",
        "dimension_bindings": {
            "month_col": "apply_month",
            "group_col": "channel",
            "segment_col": "sample_split",
        },
        "current_strategy_ref": None,
        "economics_inputs": None,
    }
    return {**fx, "impact_request": request}


def _artifacts(fx: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["kind"] == IMPACT_CUBE_ARTIFACT_KIND
    ]


def test_measure_impact_cube_publishes_exact_aggregate_only_evidence(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    output = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert output["schema_version"] == IMPACT_CUBE_TOOL_SCHEMA_VERSION
    assert output["strategy_type"] == "approval"
    assert output["partitions"] == ["development", "validation"]
    assert output["slice_count"] == len(output["cube"]["slices"])
    assert output["cube"]["slice_families"]["group"]["availability"] == (
        "present"
    )
    assert output["cube"]["slice_families"]["segment"]["availability"] == (
        "present"
    )
    assert output["cube"]["slice_families"]["group_month"][
        "availability"
    ] == "present"
    overall = next(
        row
        for row in output["cube"]["slices"]
        if row["family"] == "overall"
        and row["dimensions"]["partition"]["value"] == "development"
    )
    assert overall["current"]["availability"] == "unavailable"
    assert overall["transition"]["availability"] == "unavailable"
    assert overall["economics"] == {
        "availability": "unavailable",
        "reason": "economics_inputs_not_provided",
        "value": None,
    }
    assert validate_measure_strategy_impact_cube_tool_output(output) == output
    assert output["not_mutated_pool"] is True
    assert output["not_created_strategy"] is True
    assert output["not_adopted"] is True
    assert output["not_promoted"] is True
    assert output["not_deployed"] is True

    records = _artifacts(fx)
    assert len(records) == 1
    record = records[0]
    path = Path(record["path"])
    assert path.parent.name == "strategy_impact_cubes"
    assert path.read_text("utf-8") == canonical_strategy_impact_cube_json(
        output["cube"]
    )
    assert sha256_file(path) == output["artifact"]["content_hash"]
    assert output["artifact"]["download_url"].endswith(
        "?expected_content_hash=" + output["artifact"]["content_hash"]
    )
    assert record["origin_tool"] == "strategy.measure_strategy_impact_cube"
    assert fx["runtime"].strategies.list_for_task(fx["task"].id) == []


def test_measure_impact_cube_is_idempotent_and_cached_scalars_fail_closed(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    first = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )
    second = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert first == second
    assert len(_artifacts(fx)) == 1
    tampered = copy.deepcopy(first)
    tampered["slice_count"] += 1
    with pytest.raises(StrategyError, match="slice_count drifted"):
        validate_measure_strategy_impact_cube_tool_output(tampered)


def test_measure_impact_cube_recovers_exact_orphan_file_idempotently(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )
    artifact_id = first["artifact"]["artifact_id"]
    artifact_path = Path(_artifacts(fx)[0]["path"])
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute(
            "DELETE FROM task_artifacts WHERE id = ?",
            (artifact_id,),
        )
        conn.commit()

    second = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert second == first
    assert artifact_path.exists()
    assert len(_artifacts(fx)) == 1


def test_measure_impact_cube_reads_authenticated_private_dataset_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    read_calls = 0
    real_read_frame = fx["runtime"].backend.read_frame

    def hash_bypass_read(path, *, columns=None):
        nonlocal read_calls
        read_calls += 1
        if read_calls == 1:
            return real_read_frame(path, columns=columns)
        source = Path(path)
        original = source.read_bytes()
        forged = pd.read_parquet(source)
        numeric = next(
            column
            for column in forged.columns
            if pd.api.types.is_numeric_dtype(forged[column])
        )
        forged[numeric] = 9_999_999
        forged.to_parquet(source, index=False)
        try:
            return real_read_frame(source, columns=columns)
        finally:
            source.write_bytes(original)

    monkeypatch.setattr(
        fx["runtime"].backend,
        "read_frame",
        hash_bypass_read,
    )

    output = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert read_calls == 1
    assert output["cube"]["source_bindings"]["dataset"][
        "dataset_content_hash"
    ] == fx["dataset"].content_hash


def test_measure_impact_cube_binds_exact_current_strategy_transition(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    spec = compile_strategy_pool(fx["pool"]["pool"])["strategy_spec"]
    current = build_strategy_from_spec(spec)
    fx["runtime"].strategies.create_strategy(fx["task"].id, current)
    request = copy.deepcopy(fx["impact_request"])
    request["current_strategy_ref"] = {
        "strategy_id": current.id,
        "expected_strategy_spec_hash": strategy_spec_hash(current.spec),
    }

    output = run_measure_strategy_impact_cube(
        request,
        fx["ctx"],
        fx["runtime"],
    )

    overall = next(
        row
        for row in output["cube"]["slices"]
        if row["family"] == "overall"
        and row["dimensions"]["partition"]["value"] == "development"
    )
    assert overall["current"]["availability"] == "present"
    assert overall["current"]["value"]["strategy_id"] == current.id
    assert overall["transition"]["availability"] == "present"
    assert {
        row["direction"]
        for row in overall["transition"]["value"]["rows"]
    } == {"unchanged"}


def test_measure_impact_cube_rejects_stale_pool_and_unknown_dimensions(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    stale = copy.deepcopy(fx["impact_request"])
    stale["pool_ref"]["expected_snapshot_hash"] = "f" * 64
    with pytest.raises(StrategyError, match="snapshot|current"):
        run_measure_strategy_impact_cube(stale, fx["ctx"], fx["runtime"])

    unknown = copy.deepcopy(fx["impact_request"])
    unknown["dimension_bindings"]["group_col"] = "not_a_column"
    with pytest.raises(StrategyError, match="missing.*not_a_column"):
        run_measure_strategy_impact_cube(unknown, fx["ctx"], fx["runtime"])
    assert _artifacts(fx) == []


def test_measure_impact_cube_rejects_duplicate_or_empty_partitions(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    duplicate = copy.deepcopy(fx["impact_request"])
    duplicate["partitions"] = ["validation", "validation"]
    with pytest.raises(StrategyError, match="partitions"):
        run_measure_strategy_impact_cube(
            duplicate,
            fx["ctx"],
            fx["runtime"],
        )

    empty = copy.deepcopy(fx["impact_request"])
    empty["partitions"] = []
    with pytest.raises(StrategyError, match="partitions"):
        run_measure_strategy_impact_cube(empty, fx["ctx"], fx["runtime"])
    assert _artifacts(fx) == []


def test_measure_impact_cube_rejects_partial_economics_as_typed_unavailable(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    request = copy.deepcopy(fx["impact_request"])
    request["economics_inputs"] = {
        "ead": {"kind": "column", "column": "loan_amount"},
    }

    output = run_measure_strategy_impact_cube(
        request,
        fx["ctx"],
        fx["runtime"],
    )

    overall = next(
        row
        for row in output["cube"]["slices"]
        if row["family"] == "overall"
        and row["dimensions"]["partition"]["value"] == "development"
    )
    assert overall["economics"]["availability"] == "unavailable"
    assert overall["economics"]["value"] is None
    assert overall["economics"]["reason"].startswith(
        "missing_economics_inputs:"
    )


def test_measure_impact_cube_registration_failure_rolls_back_file_and_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)

    def fail_register(*args, **kwargs):
        raise RuntimeError("forced registry failure")

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        fail_register,
    )
    with pytest.raises(RuntimeError, match="forced registry failure"):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    out_dir = (
        Path(fx["settings"].tasks_dir)
        / fx["task"].id
        / "strategy_impact_cubes"
    )
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []
    assert _artifacts(fx) == []


def test_measure_impact_cube_concurrent_identical_calls_publish_once(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_measure_strategy_impact_cube,
                fx["impact_request"],
                fx["ctx"],
                fx["runtime"],
            )
            for _ in range(2)
        ]
        outputs = [future.result(timeout=20) for future in futures]

    assert outputs[0] == outputs[1]
    assert len(_artifacts(fx)) == 1


def test_measure_impact_cube_detects_dataset_drift_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    real_build = impact_tools.build_strategy_impact_cube

    def mutate_after_build(**kwargs):
        cube = real_build(**kwargs)
        path = (
            Path(fx["settings"].datasets_dir)
            / fx["dataset"].source_path
        )
        path.write_bytes(path.read_bytes() + b"drift")
        return cube

    monkeypatch.setattr(
        impact_tools,
        "build_strategy_impact_cube",
        mutate_after_build,
    )
    with pytest.raises(StrategyError, match="dataset changed"):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert _artifacts(fx) == []
