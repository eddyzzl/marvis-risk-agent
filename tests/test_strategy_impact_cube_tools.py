from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
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
    IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
    IMPACT_CUBE_TOOL_SCHEMA_VERSION,
    run_measure_strategy_impact_cube,
    validate_measure_strategy_impact_cube_tool_output,
)
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
import marvis.packs.strategy.impact_cube_tools as impact_tools
from marvis.packs.strategy.pool import compile_strategy_pool
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_pool_validation_tools import (
    _controlled_score_requirement,
    _setup as _pool_setup,
)


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


def _measurement_audits(fx: dict) -> list:
    with fx["runtime"].task_artifacts.transaction() as conn:
        return conn.execute(
            "SELECT * FROM audit WHERE kind = ? ORDER BY at, id",
            (IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,),
        ).fetchall()


def _validate_output(fx: dict, output: dict) -> dict:
    record = _artifacts(fx)[0]
    producer_run = record["provenance"]["producer_run"]
    return validate_measure_strategy_impact_cube_tool_output(
        output,
        trusted_task_id=fx["task"].id,
        trusted_artifact_id=record["id"],
        trusted_artifact_content_hash=record["content_hash"],
        trusted_producer_run_id=producer_run["run_id"],
        trusted_producer_run_content_hash=producer_run["content_hash"],
    )


def test_measure_impact_cube_hydrates_score_requirement_before_all_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    original_load = impact_tools._load_pool_binding
    base_binding = original_load(
        fx["runtime"],
        task_id=fx["task"].id,
        request=fx["impact_request"],
    )
    base_development = impact_tools.bind_strategy_pool_development_execution(
        fx["runtime"],
        base_binding,
    )
    controlled_binding, resolved = _controlled_score_requirement(
        pool_binding=base_binding,
    )
    controlled_development = replace(
        base_development,
        pool=controlled_binding,
    )
    virtual_field = resolved.virtual_fields[0]
    original_read = impact_tools.pd.read_parquet
    original_build = impact_tools.build_strategy_impact_cube
    calls = {"hydrate": 0, "cas": 0}

    def load_with_requirement(*args, **kwargs):
        return controlled_binding

    def bind_development(runtime, pool):
        assert runtime is fx["runtime"]
        assert pool is controlled_binding
        return controlled_development

    def hydrate(frame: pd.DataFrame, *, resolved: object) -> pd.DataFrame:
        assert resolved is score_requirements
        assert len(frame) == fx["dataset"].row_count
        assert isinstance(frame.index, pd.RangeIndex)
        assert virtual_field not in frame.columns
        calls["hydrate"] += 1
        hydrated = frame.copy(deep=True)
        hydrated[virtual_field] = [
            index / max(1, len(frame) - 1) for index in range(len(frame))
        ]
        return hydrated

    def read_snapshot(source, *args, **kwargs):
        assert virtual_field not in kwargs["columns"]
        frame = original_read(source, *args, **kwargs)
        frame.index = pd.Index(
            range(1000, 1000 + len(frame)),
            name="persisted_source_index",
        )
        return frame

    def build(**kwargs):
        frames = [
            *kwargs["partition_frames"].values(),
            *kwargs["approval_partition_frames"].values(),
        ]
        assert frames
        assert all(virtual_field in frame.columns for frame in frames)
        assert all(len(frame) < fx["dataset"].row_count for frame in frames)
        return original_build(**kwargs)

    def require_on_connection(conn, received):
        assert received is resolved
        assert conn.in_transaction
        calls["cas"] += 1

    def requirement_provenance(received):
        assert received is resolved
        return {
            "requirements_hash": resolved.requirements_hash,
            "requirements": list(resolved.requirements),
            "virtual_fields": list(resolved.virtual_fields),
        }

    score_requirements = resolved
    monkeypatch.setattr(impact_tools, "_load_pool_binding", load_with_requirement)
    monkeypatch.setattr(
        impact_tools,
        "bind_strategy_pool_development_execution",
        bind_development,
    )
    monkeypatch.setattr(
        impact_tools,
        "resolve_pool_requirements",
        lambda *args, **kwargs: resolved,
        raising=False,
    )
    monkeypatch.setattr(
        impact_tools,
        "hydrate_requirement_fields",
        hydrate,
        raising=False,
    )
    monkeypatch.setattr(impact_tools.pd, "read_parquet", read_snapshot)
    monkeypatch.setattr(impact_tools, "build_strategy_impact_cube", build)
    monkeypatch.setattr(
        impact_tools,
        "require_resolved_pool_requirements_on_connection",
        require_on_connection,
        raising=False,
    )
    monkeypatch.setattr(
        impact_tools,
        "pool_requirement_bindings_provenance",
        requirement_provenance,
    )
    monkeypatch.setattr(
        impact_tools,
        "require_strategy_candidate_pool_artifact_binding_on_connection",
        lambda conn, binding: None,
    )

    output = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert output["slice_count"] > 0
    validation_risk = next(
        row
        for row in output["cube"]["slices"]
        if row["family"] == "overall"
        and row["population_role"] == "risk"
        and row["dimensions"]["partition"]["value"] == "validation"
    )
    assert validation_risk["new"]["value"]["metrics"]["reject_count"] == 1
    assert calls["hydrate"] == 1
    assert calls["cas"] >= 4
    provenance = _artifacts(fx)[0]["provenance"]
    assert provenance["requirement_bindings"]["requirements_hash"] == (
        resolved.requirements_hash
    )
    assert provenance["requirement_bindings"]["virtual_fields"] == [
        virtual_field
    ]


@pytest.mark.parametrize(
    "message",
    [
        "virtual score field conflicts with physical dataset column",
        "model score evidence and SampleDesign V2 differ",
    ],
)
def test_measure_impact_cube_fails_closed_on_requirement_resolution_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    fx = _setup(tmp_path)

    def reject_requirement(*args, **kwargs):
        raise StrategyError(message)

    monkeypatch.setattr(
        impact_tools,
        "resolve_pool_requirements",
        reject_requirement,
    )

    with pytest.raises(StrategyError, match=message):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert _artifacts(fx) == []


def test_measure_impact_cube_requirement_cas_drift_rolls_back_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    real_require = impact_tools.require_resolved_pool_requirements_on_connection
    calls = 0

    def drift_on_final_write(conn, resolved):
        nonlocal calls
        calls += 1
        real_require(conn, resolved)
        if calls == 2:
            raise StrategyError(
                "model score vector disappeared before ImpactCube commit"
            )

    monkeypatch.setattr(
        impact_tools,
        "require_resolved_pool_requirements_on_connection",
        drift_on_final_write,
    )

    with pytest.raises(StrategyError, match="disappeared"):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert calls == 2
    assert _artifacts(fx) == []
    assert _measurement_audits(fx) == []


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
    assert _validate_output(fx, output) == output
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
    run_ref = output["producer_run_ref"]
    assert run_ref["kind"] == "tool_run"
    assert run_ref["ref_id"].startswith("strategy-impact-cube-run-")
    assert len(run_ref["ref_id"]) == len("strategy-impact-cube-run-") + 24
    assert len(run_ref["content_hash"]) == 64
    provenance_run = record["provenance"]["producer_run"]
    assert provenance_run["run_id"] == run_ref["ref_id"]
    assert provenance_run["content_hash"] == run_ref["content_hash"]
    assert provenance_run["artifact_ref"] == {
        "artifact_id": record["id"],
        "kind": IMPACT_CUBE_ARTIFACT_KIND,
        "filename": path.name,
        "content_hash": record["content_hash"],
        "origin_tool": "strategy.measure_strategy_impact_cube",
    }
    assert provenance_run["cube_ref"] == {
        "cube_id": output["cube_id"],
        "content_hash": output["content_hash"],
    }

    tool = next(
        item
        for item in load_manifest(
            Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
            builtin=True,
        ).tools
        if item.name == "measure_strategy_impact_cube"
    )
    validate_against_schema(
        output,
        tool.output_schema,
        label="ImpactCube v3 output",
    )
    missing_run_ref = copy.deepcopy(output)
    missing_run_ref.pop("producer_run_ref")
    with pytest.raises(SchemaValidationError, match="required property"):
        validate_against_schema(
            missing_run_ref,
            tool.output_schema,
            label="ImpactCube output without producer run",
        )
    audits = _measurement_audits(fx)
    assert len(audits) == 1
    assert audits[0]["target_ref"] == run_ref["ref_id"]
    assert audits[0]["inputs_hash"] == provenance_run["input_hash"]
    assert audits[0]["outcome"] == "succeeded"
    assert json.loads(str(audits[0]["detail_json"])) == {
        "producer_run": provenance_run,
    }
    assert fx["runtime"].strategies.list_for_task(fx["task"].id) == []


def test_measure_impact_cube_normalizes_omitted_optional_inputs(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    request = copy.deepcopy(fx["impact_request"])
    request.pop("current_strategy_ref")
    request.pop("economics_inputs")

    output = run_measure_strategy_impact_cube(
        request,
        fx["ctx"],
        fx["runtime"],
    )

    assert _validate_output(fx, output) == output
    overall = next(
        row
        for row in output["cube"]["slices"]
        if row["population_role"] == "approval"
        and row["family"] == "overall"
        and row["dimensions"]["partition"]["value"] == "development"
    )
    assert overall["current"]["availability"] == "unavailable"
    assert overall["economics"]["availability"] == "unavailable"


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
    assert len(_measurement_audits(fx)) == 1
    tampered = copy.deepcopy(first)
    tampered["slice_count"] += 1
    with pytest.raises(StrategyError, match="slice_count drifted"):
        _validate_output(fx, tampered)

    with pytest.raises(StrategyError, match="trusted artifact_id"):
        validate_measure_strategy_impact_cube_tool_output(
            first,
            trusted_task_id=fx["task"].id,
            trusted_artifact_id="f" * 64,
            trusted_artifact_content_hash=_artifacts(fx)[0][
                "content_hash"
            ],
            trusted_producer_run_id=first["producer_run_ref"]["ref_id"],
            trusted_producer_run_content_hash=first[
                "producer_run_ref"
            ]["content_hash"],
        )
    tampered_run = copy.deepcopy(first)
    tampered_run["producer_run_ref"]["content_hash"] = "0" * 64
    with pytest.raises(StrategyError, match="producer_run_ref drifted"):
        _validate_output(fx, tampered_run)


def test_measure_impact_cube_replay_requires_exact_unique_measurement_audit(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )
    run_id = first["producer_run_ref"]["ref_id"]
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute(
            "DELETE FROM audit WHERE kind = ? AND target_ref = ?",
            (IMPACT_CUBE_MEASUREMENT_AUDIT_KIND, run_id),
        )
        conn.commit()

    with pytest.raises(StrategyError, match="measurement audit is missing"):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    record = _artifacts(fx)[0]
    producer_run = record["provenance"]["producer_run"]
    with fx["runtime"].task_artifacts.transaction() as conn:
        fx["runtime"].repo.write_audit_on_connection(
            conn,
            kind=IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
            target_ref=run_id,
            inputs_hash=producer_run["input_hash"],
            outcome="succeeded",
            detail={"producer_run": producer_run},
        )
        fx["runtime"].repo.write_audit_on_connection(
            conn,
            kind=IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
            target_ref=run_id,
            inputs_hash=producer_run["input_hash"],
            outcome="succeeded",
            detail={"producer_run": producer_run},
        )
        conn.commit()

    with pytest.raises(
        StrategyError,
        match="measurement audit is duplicated",
    ):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )


def test_measure_impact_cube_replay_rejects_tampered_measurement_audit(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute(
            """
            UPDATE audit
               SET inputs_hash = ?
             WHERE kind = ? AND target_ref = ?
            """,
            (
                "0" * 64,
                IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
                output["producer_run_ref"]["ref_id"],
            ),
        )
        conn.commit()

    with pytest.raises(StrategyError, match="audit binding changed"):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )


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
        conn.execute(
            "DELETE FROM audit WHERE kind = ?",
            (IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,),
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
    monkeypatch: pytest.MonkeyPatch,
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
    for method_name in (
        "get_strategy",
        "get_strategy_meta",
        "get_strategy_spec_hash",
    ):
        monkeypatch.setattr(
            fx["runtime"].strategies,
            method_name,
            lambda *_args, **_kwargs: pytest.fail(
                "ImpactCube must use one atomic strategy snapshot"
            ),
        )

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


def test_measure_impact_cube_rejects_identifier_dimension_role(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    request = copy.deepcopy(fx["impact_request"])
    request["dimension_bindings"]["group_col"] = "customer_id"

    with pytest.raises(
        StrategyError,
        match="identifier|sensitive|personal",
    ):
        run_measure_strategy_impact_cube(
            request,
            fx["ctx"],
            fx["runtime"],
        )
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
    assert _measurement_audits(fx) == []


def test_measure_impact_cube_audit_failure_rolls_back_file_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("measurement audit unavailable")

    monkeypatch.setattr(
        fx["runtime"].repo,
        "write_audit_on_connection",
        fail_audit,
    )
    with pytest.raises(RuntimeError, match="measurement audit unavailable"):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert _artifacts(fx) == []
    assert _measurement_audits(fx) == []
    out_dir = (
        Path(fx["settings"].tasks_dir)
        / fx["task"].id
        / "strategy_impact_cubes"
    )
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []


def test_measure_impact_cube_detects_artifact_swap_during_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    real_register = fx["runtime"].task_artifacts.register_on_connection

    def mutate_after_register(conn, **kwargs):
        record = real_register(conn, **kwargs)
        Path(kwargs["path"]).write_bytes(b'{"tampered":true}')
        return record

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        mutate_after_register,
    )
    with pytest.raises(
        StrategyError,
        match="changed during registration",
    ):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert _artifacts(fx) == []


def test_measure_impact_cube_compensates_swap_after_precommit_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    real_require = impact_tools._require_retained_exact_file
    call_count = 0

    def mutate_after_precommit_check(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        real_require(*args, **kwargs)
        if call_count == 2:
            Path(kwargs["path"]).write_bytes(b'{"tampered":true}')

    monkeypatch.setattr(
        impact_tools,
        "_require_retained_exact_file",
        mutate_after_precommit_check,
    )
    with pytest.raises(
        StrategyError,
        match="changed after registration commit",
    ):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert call_count == 3
    assert _artifacts(fx) == []
    assert _measurement_audits(fx) == []


def test_measure_impact_cube_compensation_cas_preserves_changed_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    real_require = (
        impact_tools.require_impact_cube_measurement_audit_on_connection
    )
    call_count = 0

    def tamper_after_precommit_check(conn, producer_run):
        nonlocal call_count
        call_count += 1
        real_require(conn, producer_run)
        if call_count == 1:
            conn.execute(
                """
                UPDATE audit
                   SET outcome = 'failed'
                 WHERE kind = ? AND target_ref = ?
                """,
                (
                    IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
                    producer_run["run_id"],
                ),
            )

    monkeypatch.setattr(
        impact_tools,
        "require_impact_cube_measurement_audit_on_connection",
        tamper_after_precommit_check,
    )
    with pytest.raises(
        StrategyError,
        match="compensation CAS failed",
    ):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert call_count == 2
    assert len(_artifacts(fx)) == 1
    audits = _measurement_audits(fx)
    assert len(audits) == 1
    assert audits[0]["outcome"] == "failed"


def test_measure_impact_cube_replay_transient_postcommit_failure_retains_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )
    artifacts_before = copy.deepcopy(_artifacts(fx))
    audits_before = [
        dict(row) for row in _measurement_audits(fx)
    ]
    real_require = (
        impact_tools.require_impact_cube_measurement_audit_on_connection
    )
    call_count = 0

    def fail_only_postcommit(conn, producer_run):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise StrategyError("transient postcommit verification failure")
        real_require(conn, producer_run)

    monkeypatch.setattr(
        impact_tools,
        "require_impact_cube_measurement_audit_on_connection",
        fail_only_postcommit,
    )
    with pytest.raises(
        StrategyError,
        match="pre-existing registry and audit entries were retained",
    ):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert call_count == 2
    assert _artifacts(fx) == artifacts_before
    assert [
        dict(row) for row in _measurement_audits(fx)
    ] == audits_before


def test_measure_impact_cube_replay_file_drift_retains_existing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    run_measure_strategy_impact_cube(
        fx["impact_request"],
        fx["ctx"],
        fx["runtime"],
    )
    artifacts_before = copy.deepcopy(_artifacts(fx))
    audits_before = [
        dict(row) for row in _measurement_audits(fx)
    ]
    real_require = impact_tools._require_retained_exact_file
    call_count = 0

    def drift_after_precommit_check(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        real_require(*args, **kwargs)
        if call_count == 2:
            Path(kwargs["path"]).write_bytes(b'{"tampered":true}')

    monkeypatch.setattr(
        impact_tools,
        "_require_retained_exact_file",
        drift_after_precommit_check,
    )
    with pytest.raises(
        StrategyError,
        match="pre-existing registry and audit entries were retained",
    ):
        run_measure_strategy_impact_cube(
            fx["impact_request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert call_count == 3
    assert _artifacts(fx) == artifacts_before
    assert [
        dict(row) for row in _measurement_audits(fx)
    ] == audits_before


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
    assert len(_measurement_audits(fx)) == 1


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
