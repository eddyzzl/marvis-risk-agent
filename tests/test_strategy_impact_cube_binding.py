from __future__ import annotations

import hashlib

import pytest

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_binding import (
    StrategyImpactCubeArtifactBinding,
    load_strategy_impact_cube_artifact,
    require_strategy_impact_cube_artifact_binding_on_connection,
    validate_strategy_impact_cube_artifact_binding,
)
from marvis.packs.strategy.impact_cube_tools import (
    IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
)
from marvis.packs.strategy.report_bundle_adapters import (
    StrategyImpactCubeArtifactBinding as ReportImpactCubeArtifactBinding,
)
from marvis.packs.strategy.report_bundle_tools import (
    load_strategy_impact_cube_artifact as load_report_impact_cube_artifact,
)
from test_strategy_report_bundle_tools import _setup_impact_cube_report


def _load_kwargs(fixture: dict) -> dict:
    return {
        "task_id": fixture["task"].id,
        **fixture["request"]["impact_cube_ref"],
    }


def test_shared_binding_preserves_report_loader_contract(tmp_path) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    kwargs = _load_kwargs(fixture)

    shared = load_strategy_impact_cube_artifact(
        fixture["runtime"],
        **kwargs,
    )
    report = load_report_impact_cube_artifact(
        fixture["runtime"],
        **kwargs,
    )

    assert shared == report
    assert isinstance(shared, StrategyImpactCubeArtifactBinding)
    assert isinstance(report, ReportImpactCubeArtifactBinding)
    assert validate_strategy_impact_cube_artifact_binding(shared) == shared.cube
    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_impact_cube_artifact_binding_on_connection(
            conn,
            shared,
        )
        conn.rollback()


def test_shared_binding_recheck_rejects_missing_measurement_audit(
    tmp_path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    binding = load_strategy_impact_cube_artifact(
        fixture["runtime"],
        **_load_kwargs(fixture),
    )
    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute(
            "DELETE FROM audit WHERE kind = ?",
            (IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,),
        )
        conn.commit()

    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(
            StrategyError,
            match="measurement audit is missing",
        ):
            require_strategy_impact_cube_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()


@pytest.mark.parametrize("missing_main_evidence", ["artifact", "audit"])
def test_shared_binding_recheck_ignores_temp_schema_shadows(
    tmp_path,
    missing_main_evidence: str,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    binding = load_strategy_impact_cube_artifact(
        fixture["runtime"],
        **_load_kwargs(fixture),
    )

    with fixture["runtime"].task_artifacts.transaction() as conn:
        main_path = next(
            str(row["file"])
            for row in conn.execute("PRAGMA database_list").fetchall()
            if str(row["name"]) == "main"
        )
        conn.execute(
            "CREATE TEMP TABLE pragma_database_list(name TEXT, file TEXT)"
        )
        conn.execute(
            "INSERT INTO temp.pragma_database_list(name, file) VALUES (?, ?)",
            ("main", main_path),
        )
        conn.execute(
            "CREATE TEMP TABLE task_artifacts "
            "AS SELECT * FROM main.task_artifacts"
        )
        conn.execute("CREATE TEMP TABLE audit AS SELECT * FROM main.audit")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        if missing_main_evidence == "artifact":
            conn.execute(
                "DELETE FROM main.task_artifacts WHERE id = ?",
                (binding.artifact_id,),
            )
        else:
            conn.execute(
                """
                DELETE FROM main.audit
                 WHERE kind = ? AND target_ref = ?
                """,
                (
                    IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
                    binding.artifact_provenance["producer_run"]["run_id"],
                ),
            )

        with pytest.raises(StrategyError):
            require_strategy_impact_cube_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()


@pytest.mark.parametrize(
    "prefix",
    [
        b'{"schema_version":"duplicate",',
        b'{"nonfinite":NaN,',
    ],
)
def test_shared_loader_rejects_duplicate_and_nonfinite_json(
    tmp_path,
    prefix: bytes,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    binding = load_strategy_impact_cube_artifact(
        fixture["runtime"],
        **_load_kwargs(fixture),
    )
    invalid = prefix + binding.artifact_path.read_bytes()[1:]
    invalid_hash = hashlib.sha256(invalid).hexdigest()
    binding.artifact_path.write_bytes(invalid)
    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET content_hash = ? WHERE id = ?",
            (invalid_hash, binding.artifact_id),
        )
        conn.commit()
    kwargs = {
        **_load_kwargs(fixture),
        "expected_artifact_content_hash": invalid_hash,
    }

    with pytest.raises(StrategyError, match="artifact JSON is invalid"):
        load_strategy_impact_cube_artifact(
            fixture["runtime"],
            **kwargs,
        )
