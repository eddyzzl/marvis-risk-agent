from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.db_schema import connect
from marvis.files import sha256_file
from marvis.packs.strategy.errors import StrategyError
import marvis.packs.strategy.pool_apply_tools as pool_apply_tools
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_apply_tools import (
    EVIDENCE_ARTIFACT_KIND,
    RESULT_DATASET_ROLE,
    run_apply_strategy_pool,
    validate_apply_strategy_pool_tool_output,
)
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.tools import tool_apply_strategy_pool
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_pool_tools import _add_inputs, _setup


def _prepared_pool(tmp_path: Path) -> tuple[dict, dict]:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    return fixture, added


def _inputs(added: dict, *, output_prefix: str | None = None) -> dict:
    value = {
        "strategy_type": "approval",
        "expected_pool_revision": added["revision"],
        "expected_pool_snapshot_hash": added["snapshot_hash"],
    }
    if output_prefix is not None:
        value["output_prefix"] = output_prefix
    return value


def _count(conn, table: str, *, where: str = "", values: tuple = ()) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} {where}",  # noqa: S608 - test-only fixed names
        values,
    ).fetchone()
    return int(row[0])


def test_apply_current_pool_creates_one_non_active_governed_dataset(
    tmp_path: Path,
) -> None:
    fixture, added = _prepared_pool(tmp_path)
    runtime = fixture["runtime"]
    source_dataset = fixture["dataset"]
    before_workspace = runtime.data_workspaces.get_or_default(fixture["task"].id)
    before_pool = StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_current(fixture["task"].id, "approval")

    result = run_apply_strategy_pool(
        _inputs(added),
        fixture["ctx"],
        runtime,
    )

    assert validate_apply_strategy_pool_tool_output(result) == result
    assert result["schema_version"] == "strategy.apply-strategy-pool-tool.v1"
    assert result["cached"] is False
    assert result["activated"] is False
    assert result["adopted"] is False
    assert result["deployed"] is False
    assert result["source"]["pool_id"] == added["pool_id"]
    assert result["source"]["revision"] == added["revision"]
    assert result["source"]["snapshot_hash"] == added["snapshot_hash"]
    assert result["source"]["dataset_id"] == source_dataset.id
    assert result["source"]["dataset_content_hash"] == source_dataset.content_hash
    assert result["source"]["row_count"] == source_dataset.row_count
    assert result["requirements"]["virtual_fields"] == []
    assert result["workspace"]["active_dataset_id"] == source_dataset.id
    assert result["workspace"]["result_revision"] is None
    assert result["workspace"]["result_analysis_generation"] is None

    derived_dataset = runtime.registry.get(result["result"]["dataset_id"])
    assert derived_dataset.task_id == fixture["task"].id
    assert derived_dataset.role == RESULT_DATASET_ROLE
    assert derived_dataset.row_count == source_dataset.row_count
    assert derived_dataset.content_hash == result["result"]["dataset_content_hash"]
    derived_path = runtime.registry.resolve_verified_path(derived_dataset.id)
    assert sha256_file(derived_path) == result["result"]["dataset_content_hash"]
    derived = pd.read_parquet(derived_path)
    source = pd.read_parquet(runtime.registry.resolve_verified_path(source_dataset.id))
    assert derived.iloc[:, : len(source.columns)].equals(source)
    assert result["columns"] == {
        "action": "strategy_pool_action",
        "value": "strategy_pool_value",
        "value_type": "strategy_pool_value_type",
        "rule_id": "strategy_pool_rule_id",
        "entry_id": "strategy_pool_entry_id",
        "reason_code": "strategy_pool_reason_code",
    }
    assert derived.columns.tolist() == [
        *source.columns.tolist(),
        result["columns"]["action"],
        result["columns"]["value"],
        result["columns"]["value_type"],
        result["columns"]["rule_id"],
        result["columns"]["entry_id"],
        result["columns"]["reason_code"],
    ]
    assert sum(result["action_counts"].values()) == source_dataset.row_count
    assert (
        sum(result["rule_counts"].values()) + result["default_count"]
        == source_dataset.row_count
    )
    assert sorted(result["rule_counts"].values()) == sorted(
        result["entry_counts"].values()
    )

    after_workspace = runtime.data_workspaces.get_or_default(fixture["task"].id)
    assert after_workspace == before_workspace
    assert runtime.strategies.list_for_task(fixture["task"].id) == []
    assert (
        StrategyCandidatePoolRepository(
            fixture["settings"].db_path
        ).get_current(fixture["task"].id, "approval")
        == before_pool
    )
    record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(fixture["task"].id, result["evidence"]["artifact_id"])
    assert record is not None
    assert record["kind"] == EVIDENCE_ARTIFACT_KIND
    assert record["content_hash"] == result["evidence"]["content_hash"]
    assert record["provenance"]["input_hash"] == result["input_hash"]
    assert record["provenance"]["input_identity"]["pool"]["snapshot_hash"] == added[
        "snapshot_hash"
    ]
    assert record["provenance"]["input_identity"]["dataset"]["dataset_id"] == (
        source_dataset.id
    )
    assert record["provenance"]["input_identity"]["requirements"] == result[
        "requirements"
    ]
    evidence = json.loads(Path(record["path"]).read_text("utf-8"))
    assert evidence["run_id"] == result["run_id"]
    assert evidence["result"] == result["result"]


def test_apply_current_pool_exact_retry_is_side_effect_free(
    tmp_path: Path,
) -> None:
    fixture, added = _prepared_pool(tmp_path)
    runtime = fixture["runtime"]
    inputs = _inputs(added, output_prefix="decision_")

    first = run_apply_strategy_pool(inputs, fixture["ctx"], runtime)
    with connect(fixture["settings"].db_path) as conn:
        before = {
            "datasets": _count(conn, "datasets"),
            "artifacts": _count(
                conn,
                "task_artifacts",
                where="WHERE kind = ?",
                values=(EVIDENCE_ARTIFACT_KIND,),
            ),
            "audits": _count(
                conn,
                "audit",
                where="WHERE kind = ?",
                values=("strategy.pool.apply",),
            ),
        }

    replay = run_apply_strategy_pool(inputs, fixture["ctx"], runtime)

    assert replay == {**first, "cached": True}
    with connect(fixture["settings"].db_path) as conn:
        after = {
            "datasets": _count(conn, "datasets"),
            "artifacts": _count(
                conn,
                "task_artifacts",
                where="WHERE kind = ?",
                values=(EVIDENCE_ARTIFACT_KIND,),
            ),
            "audits": _count(
                conn,
                "audit",
                where="WHERE kind = ?",
                values=("strategy.pool.apply",),
            ),
        }
    assert after == before


def test_apply_current_pool_is_registered_with_strict_manifest_contract(
    tmp_path: Path,
) -> None:
    fixture, added = _prepared_pool(tmp_path)

    result = tool_apply_strategy_pool(_inputs(added), fixture["ctx"])

    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tool = next(item for item in manifest.tools if item.name == "apply_strategy_pool")
    validate_against_schema(
        _inputs(added),
        tool.input_schema,
        label="Strategy Pool apply inputs",
    )
    validate_against_schema(
        result,
        tool.output_schema,
        label="Strategy Pool apply output",
    )
    assert tool.entrypoint == "tool_apply_strategy_pool"
    assert set(tool.side_effects) == {
        "read:task",
        "read:dataset",
        "read:artifacts",
        "write:artifact",
        "write:dataset",
    }


def test_apply_current_pool_rejects_drift_and_invalid_surface_without_writes(
    tmp_path: Path,
) -> None:
    fixture, added = _prepared_pool(tmp_path)
    runtime = fixture["runtime"]
    with connect(fixture["settings"].db_path) as conn:
        before_datasets = _count(conn, "datasets")
        before_artifacts = _count(
            conn,
            "task_artifacts",
            where="WHERE kind = ?",
            values=(EVIDENCE_ARTIFACT_KIND,),
        )

    with pytest.raises(StrategyError, match="unsupported fields"):
        run_apply_strategy_pool(
            {**_inputs(added), "dataset_id": fixture["dataset"].id},
            fixture["ctx"],
            runtime,
        )

    source_path = runtime.registry.resolve_verified_path(fixture["dataset"].id)
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(StrategyError, match="drift|changed|hash"):
        run_apply_strategy_pool(
            _inputs(added),
            fixture["ctx"],
            runtime,
        )

    with connect(fixture["settings"].db_path) as conn:
        assert _count(conn, "datasets") == before_datasets
        assert (
            _count(
                conn,
                "task_artifacts",
                where="WHERE kind = ?",
                values=(EVIDENCE_ARTIFACT_KIND,),
            )
            == before_artifacts
        )


def test_apply_current_pool_rechecks_source_under_writer_lock_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, added = _prepared_pool(tmp_path)
    runtime = fixture["runtime"]
    source_path = runtime.registry.resolve_verified_path(fixture["dataset"].id)
    source_bytes = source_path.read_bytes()
    original_apply = pool_apply_tools.apply_strategy_pool

    def drift_after_compute(*args, **kwargs):
        result = original_apply(*args, **kwargs)
        source_path.write_bytes(source_bytes + b"\n")
        return result

    monkeypatch.setattr(
        pool_apply_tools,
        "apply_strategy_pool",
        drift_after_compute,
    )
    with connect(fixture["settings"].db_path) as conn:
        before_datasets = _count(conn, "datasets")
        before_audits = _count(
            conn,
            "audit",
            where="WHERE kind = ?",
            values=("strategy.pool.apply",),
        )

    with pytest.raises(StrategyError, match="drift|changed|hash"):
        run_apply_strategy_pool(_inputs(added), fixture["ctx"], runtime)

    source_path.write_bytes(source_bytes)
    with connect(fixture["settings"].db_path) as conn:
        assert _count(conn, "datasets") == before_datasets
        assert (
            _count(
                conn,
                "task_artifacts",
                where="WHERE kind = ?",
                values=(EVIDENCE_ARTIFACT_KIND,),
            )
            == 0
        )
        assert (
            _count(
                conn,
                "audit",
                where="WHERE kind = ?",
                values=("strategy.pool.apply",),
            )
            == before_audits
        )


def test_apply_current_pool_cached_evidence_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    fixture, added = _prepared_pool(tmp_path)
    runtime = fixture["runtime"]
    inputs = _inputs(added)
    first = run_apply_strategy_pool(inputs, fixture["ctx"], runtime)
    record = runtime.task_artifacts.get_for_task(
        fixture["task"].id,
        first["evidence"]["artifact_id"],
    )
    assert record is not None
    evidence_path = Path(record["path"])
    evidence_path.write_bytes(evidence_path.read_bytes() + b"\n")
    with connect(fixture["settings"].db_path) as conn:
        before_datasets = _count(conn, "datasets")
        before_audits = _count(
            conn,
            "audit",
            where="WHERE kind = ?",
            values=("strategy.pool.apply",),
        )

    with pytest.raises(StrategyError, match="evidence.*hash|hash.*changed"):
        run_apply_strategy_pool(inputs, fixture["ctx"], runtime)

    with connect(fixture["settings"].db_path) as conn:
        assert _count(conn, "datasets") == before_datasets
        assert (
            _count(
                conn,
                "audit",
                where="WHERE kind = ?",
                values=("strategy.pool.apply",),
            )
            == before_audits
        )


def test_apply_current_pool_rolls_back_files_and_rows_when_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, added = _prepared_pool(tmp_path)
    runtime = fixture["runtime"]
    with connect(fixture["settings"].db_path) as conn:
        before_datasets = _count(conn, "datasets")

    def reject_audit(*args, **kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(
        runtime.repo,
        "write_audit_on_connection",
        reject_audit,
    )
    with pytest.raises(RuntimeError, match="audit unavailable"):
        run_apply_strategy_pool(_inputs(added), fixture["ctx"], runtime)

    with connect(fixture["settings"].db_path) as conn:
        assert _count(conn, "datasets") == before_datasets
        assert (
            _count(
                conn,
                "task_artifacts",
                where="WHERE kind = ?",
                values=(EVIDENCE_ARTIFACT_KIND,),
            )
            == 0
        )
        assert (
            _count(
                conn,
                "audit",
                where="WHERE kind = ?",
                values=("strategy.pool.apply",),
            )
            == 0
        )
    result_dir = (
        fixture["settings"].datasets_dir
        / fixture["task"].id
        / "strategy_pool_applies"
    )
    evidence_dir = (
        fixture["settings"].tasks_dir
        / fixture["task"].id
        / "strategy_pool_applies"
    )
    assert list(result_dir.glob("*.parquet")) == []
    assert list(evidence_dir.glob("*.json")) == []
