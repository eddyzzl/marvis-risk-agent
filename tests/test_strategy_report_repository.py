from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sqlite3

import pytest

import marvis.db_schema as db_schema_module
from marvis.db_schema import connect, init_db
from marvis.output.strategy_report_bundle import render_strategy_report_bundle
from marvis.packs.strategy.project_context import (
    build_report_field,
    build_source_ref,
)
from marvis.packs.strategy.report_bundle import (
    REPORT_SECTION_KEYS,
    build_strategy_report_bundle,
    build_strategy_report_section,
)
from marvis.repositories.strategy_reports import (
    STRATEGY_REPORT_HEAD_SCHEMA_VERSION,
    STRATEGY_REPORT_ORIGIN_TOOL,
    STRATEGY_REPORT_OUTPUT_KINDS,
    StrategyReportConflictError,
    StrategyReportDataError,
    StrategyReportRepository,
    build_strategy_report_output_artifact_provenance,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository


_CREATED_AT = "2026-07-23T08:00:00+00:00"
_SOURCE_HASH = "a" * 64


def _seed_strategy_task(
    db_path: Path,
    *,
    task_id: str = "strategy-task-1",
    strategy_id: str = "strategy-1",
) -> tuple[str, str]:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, task_type, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, 'strategy', 'report fixture', 'v1', 'qa', ?, 'draft',
                      '', ?, ?)
            """,
            (task_id, f"/tmp/{task_id}", _CREATED_AT, _CREATED_AT),
        )
        conn.execute(
            """
            INSERT INTO strategies(
                id, task_id, strategy_type, rules_json, score_col,
                default_decision_json, description, created_at, version,
                status, asset_status
            ) VALUES (?, ?, 'approval', '[]', 'score', '"approve"',
                      'report fixture', ?, 1, 'draft', 'draft')
            """,
            (strategy_id, task_id, _CREATED_AT),
        )
    return task_id, strategy_id


def _bundle(
    *,
    task_id: str = "strategy-task-1",
    strategy_id: str = "strategy-1",
    report_revision: int = 1,
    previous_report_id: str | None = None,
    title: str = "策略迭代报告",
) -> dict:
    source_ref = build_source_ref(
        kind="task_artifact",
        ref_id="trusted-context",
        content_hash=_SOURCE_HASH,
    )
    title_field = build_report_field(
        value=title,
        availability="present",
        origin="tool_output",
        source_refs=[source_ref],
    )
    sections = [
        build_strategy_report_section(
            key=key,
            title=key,
            availability="unavailable",
        )
        for key in REPORT_SECTION_KEYS
    ]
    return build_strategy_report_bundle(
        task_id=task_id,
        report_revision=report_revision,
        strategy_id=strategy_id,
        strategy_version="1",
        strategy_type="approval",
        title=title_field,
        status="partial",
        sections=sections,
        generated_at=_CREATED_AT,
        previous_report_id=previous_report_id,
    )


def _register_outputs(
    db_path: Path,
    bundle: dict,
) -> dict[str, dict]:
    rendered = render_strategy_report_bundle(bundle)
    artifact_repo = TaskArtifactRepository(db_path)
    records = {}
    output_dir = (
        db_path.parent
        / "tasks"
        / bundle["task_id"]
        / "strategy_reports"
        / bundle["report_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_format, payload in rendered.items():
        suffix = "md" if output_format == "markdown" else output_format
        path = output_dir / f"report.{suffix}"
        path.write_bytes(payload)
        records[output_format] = artifact_repo.register(
            task_id=bundle["task_id"],
            kind=STRATEGY_REPORT_OUTPUT_KINDS[output_format],
            path=str(path),
            content_hash=hashlib.sha256(payload).hexdigest(),
            origin_tool=STRATEGY_REPORT_ORIGIN_TOOL,
            provenance=build_strategy_report_output_artifact_provenance(
                bundle,
                output_format=output_format,
            ),
            created_at=_CREATED_AT,
        )
    return records


def test_migration_020_is_registered_and_builds_guarded_report_ledger(tmp_path):
    db_path = tmp_path / "migration.sqlite"
    _seed_strategy_task(db_path)
    init_db(db_path)

    with connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        triggers = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' AND name LIKE 'trg_strategy_report_%'"
            )
        }

    assert version == db_schema_module.SCHEMA_VERSION == 20
    assert db_schema_module._MIGRATIONS[-1] == (
        20,
        db_schema_module._migration_020_strategy_report_revisions,
    )
    assert {"strategy_report_heads", "strategy_report_revisions"} <= tables
    assert {
        "trg_strategy_report_revisions_parent",
        "trg_strategy_report_revisions_head_parent",
        "trg_strategy_report_revisions_artifacts",
        "trg_strategy_report_heads_target_update",
        "trg_strategy_report_heads_append_only",
        "trg_strategy_report_revisions_immutable_update",
        "trg_strategy_report_revisions_immutable_delete",
        "trg_strategy_report_output_artifacts_immutable_delete",
    } <= triggers


def test_publish_roundtrips_and_exact_current_retry_is_idempotent(tmp_path):
    db_path = tmp_path / "publish.sqlite"
    task_id, strategy_id = _seed_strategy_task(db_path)
    bundle = _bundle(task_id=task_id, strategy_id=strategy_id)
    artifacts = _register_outputs(db_path, bundle)
    repo = StrategyReportRepository(db_path)

    first = repo.publish(
        bundle=bundle,
        artifacts=artifacts,
        expected_revision=0,
        expected_report_id=None,
        expected_content_hash=None,
        created_at=_CREATED_AT,
    )
    replay = repo.publish(
        bundle=bundle,
        artifacts=artifacts,
        expected_revision=0,
        expected_report_id=None,
        expected_content_hash=None,
        created_at="2026-07-23T09:00:00+00:00",
    )

    assert first == replay
    assert repo.get_current(task_id=task_id, strategy_id=strategy_id) == first
    assert repo.get_revision(
        task_id=task_id,
        strategy_id=strategy_id,
        report_revision=1,
    ) == first
    assert repo.get_by_id(task_id=task_id, report_id=bundle["report_id"]) == first
    head = repo.get_head(task_id=task_id, strategy_id=strategy_id)
    assert head == {
        "task_id": task_id,
        "strategy_id": strategy_id,
        "strategy_scope": f"strategy:{strategy_id}",
        "current_revision": 1,
        "current_report_id": bundle["report_id"],
        "current_content_hash": bundle["content_sha256"],
    }
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM strategy_report_heads WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    assert row["schema_version"] == STRATEGY_REPORT_HEAD_SCHEMA_VERSION


def test_revision_chain_requires_exact_parent_and_rejects_stale_retry(tmp_path):
    db_path = tmp_path / "chain.sqlite"
    task_id, strategy_id = _seed_strategy_task(db_path)
    repo = StrategyReportRepository(db_path)
    first_bundle = _bundle(task_id=task_id, strategy_id=strategy_id)
    first_artifacts = _register_outputs(db_path, first_bundle)
    repo.publish(
        bundle=first_bundle,
        artifacts=first_artifacts,
        expected_revision=0,
        expected_report_id=None,
        expected_content_hash=None,
    )
    second_bundle = _bundle(
        task_id=task_id,
        strategy_id=strategy_id,
        report_revision=2,
        previous_report_id=first_bundle["report_id"],
        title="策略迭代报告（修订）",
    )
    second_artifacts = _register_outputs(db_path, second_bundle)
    second = repo.publish(
        bundle=second_bundle,
        artifacts=second_artifacts,
        expected_revision=1,
        expected_report_id=first_bundle["report_id"],
        expected_content_hash=first_bundle["content_sha256"],
    )

    assert second["bundle"]["report_revision"] == 2
    with pytest.raises(
        StrategyReportConflictError,
        match="original parent head triple",
    ):
        repo.publish(
            bundle=second_bundle,
            artifacts=second_artifacts,
            expected_revision=1,
            expected_report_id=first_bundle["report_id"],
            expected_content_hash="f" * 64,
        )
    with pytest.raises(
        StrategyReportConflictError,
        match="no longer the current head",
    ):
        repo.publish(
            bundle=first_bundle,
            artifacts=first_artifacts,
            expected_revision=0,
            expected_report_id=None,
            expected_content_hash=None,
        )

    wrong_parent = deepcopy(second_bundle)
    wrong_parent["previous_report_id"] = "strategy-report-" + "f" * 24
    with pytest.raises(StrategyReportDataError, match="invalid StrategyReportBundle"):
        repo.publish(
            bundle=wrong_parent,
            artifacts=second_artifacts,
            expected_revision=2,
            expected_report_id=second_bundle["report_id"],
            expected_content_hash=second_bundle["content_sha256"],
        )


def test_publish_reloads_registry_and_rejects_forged_or_cross_task_artifacts(tmp_path):
    db_path = tmp_path / "artifact-trust.sqlite"
    task_id, strategy_id = _seed_strategy_task(db_path)
    _seed_strategy_task(
        db_path,
        task_id="strategy-task-2",
        strategy_id="strategy-2",
    )
    bundle = _bundle(task_id=task_id, strategy_id=strategy_id)
    artifacts = _register_outputs(db_path, bundle)
    repo = StrategyReportRepository(db_path)

    forged = deepcopy(artifacts)
    forged["xlsx"]["content_hash"] = "f" * 64
    with pytest.raises(StrategyReportDataError, match="does not match registry"):
        repo.publish(
            bundle=bundle,
            artifacts=forged,
            expected_revision=0,
            expected_report_id=None,
            expected_content_hash=None,
        )

    other_bundle = _bundle(
        task_id="strategy-task-2",
        strategy_id="strategy-2",
    )
    other_artifacts = _register_outputs(db_path, other_bundle)
    mixed = {**artifacts, "xlsx": other_artifacts["xlsx"]}
    with pytest.raises(StrategyReportDataError, match="another task"):
        repo.publish(
            bundle=bundle,
            artifacts=mixed,
            expected_revision=0,
            expected_report_id=None,
            expected_content_hash=None,
        )


@pytest.mark.parametrize("output_format", ["json", "markdown", "xlsx"])
def test_publish_reproduces_and_reads_exact_physical_output_bytes(
    tmp_path: Path,
    output_format: str,
) -> None:
    db_path = tmp_path / f"physical-{output_format}.sqlite"
    task_id, strategy_id = _seed_strategy_task(db_path)
    bundle = _bundle(task_id=task_id, strategy_id=strategy_id)
    artifacts = _register_outputs(db_path, bundle)
    path = Path(artifacts[output_format]["path"])
    if output_format == "markdown":
        path.unlink()
        match = "unavailable"
    else:
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        match = "bytes do not match|hash does not match"

    with pytest.raises(StrategyReportDataError, match=match):
        StrategyReportRepository(db_path).publish(
            bundle=bundle,
            artifacts=artifacts,
            expected_revision=0,
            expected_report_id=None,
            expected_content_hash=None,
        )


def test_database_guards_report_rows_heads_and_published_artifacts(tmp_path):
    db_path = tmp_path / "guards.sqlite"
    task_id, strategy_id = _seed_strategy_task(db_path)
    bundle = _bundle(task_id=task_id, strategy_id=strategy_id)
    artifacts = _register_outputs(db_path, bundle)
    StrategyReportRepository(db_path).publish(
        bundle=bundle,
        artifacts=artifacts,
        expected_revision=0,
        expected_report_id=None,
        expected_content_hash=None,
    )

    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE strategy_report_revisions SET created_at = ? "
                "WHERE report_id = ?",
                ("changed", bundle["report_id"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE strategy_report_heads SET updated_at = ? "
                "WHERE task_id = ?",
                ("changed", task_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "DELETE FROM task_artifacts WHERE id = ?",
                (artifacts["json"]["id"],),
            )


def test_task_deletion_cascades_report_ledger_and_output_artifacts(tmp_path):
    db_path = tmp_path / "cascade.sqlite"
    task_id, strategy_id = _seed_strategy_task(db_path)
    bundle = _bundle(task_id=task_id, strategy_id=strategy_id)
    artifacts = _register_outputs(db_path, bundle)
    StrategyReportRepository(db_path).publish(
        bundle=bundle,
        artifacts=artifacts,
        expected_revision=0,
        expected_report_id=None,
        expected_content_hash=None,
    )

    with connect(db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_report_revisions"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_report_heads"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_artifacts WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0
