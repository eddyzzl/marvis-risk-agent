from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

import pytest

import marvis.packs.strategy.pool_stability_tools as stability_tools
from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_tools import (
    IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
    run_measure_strategy_impact_cube,
)
from marvis.packs.strategy.pool_stability_tools import (
    POOL_STABILITY_ARTIFACT_KIND,
    POOL_STABILITY_MEASUREMENT_AUDIT_KIND,
    POOL_STABILITY_TOOL_SCHEMA_VERSION,
    load_strategy_pool_stability_artifact,
    require_strategy_pool_stability_artifact_binding_on_connection,
    run_measure_strategy_pool_stability,
    validate_measure_strategy_pool_stability_tool_output,
)
from marvis.packs.strategy.pool_stability import (
    build_strategy_pool_stability,
    canonical_strategy_pool_stability_json,
)
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.manifest import ToolRef
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.task_artifacts import (
    TaskArtifactRepository,
    stable_task_artifact_id,
)
from tests.test_strategy_impact_cube_tools import _setup as _impact_setup


def _setup(tmp_path: Path) -> dict:
    fixture = _impact_setup(tmp_path)
    impact_output = run_measure_strategy_impact_cube(
        fixture["impact_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    request = {
        "artifact_id": impact_output["artifact"]["artifact_id"],
        "expected_artifact_content_hash": impact_output["artifact"][
            "content_hash"
        ],
        "expected_cube_id": impact_output["cube_id"],
        "expected_cube_content_hash": impact_output["content_hash"],
    }
    return {
        **fixture,
        "impact_output": impact_output,
        "stability_request": request,
    }


def _stability_artifacts(fixture: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(
            fixture["settings"].db_path
        ).list_for_task(fixture["task"].id)
        if item["kind"] == POOL_STABILITY_ARTIFACT_KIND
    ]


def _stability_audits(fixture: dict) -> list:
    with fixture["runtime"].task_artifacts.transaction() as conn:
        return conn.execute(
            "SELECT * FROM audit WHERE kind = ? ORDER BY at, id",
            (POOL_STABILITY_MEASUREMENT_AUDIT_KIND,),
        ).fetchall()


def test_measure_pool_stability_publishes_one_read_only_exact_source_artifact(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    output = run_measure_strategy_pool_stability(
        copy.deepcopy(fixture["stability_request"]),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert output["schema_version"] == POOL_STABILITY_TOOL_SCHEMA_VERSION
    assert output["baseline_partition"] == "development"
    assert output["comparison_partitions"] == ["validation"]
    assert output["read_only"] is True
    assert output["effect_validation"] is False
    assert output["not_mutated_pool"] is True
    assert output["not_created_strategy"] is True
    assert output["not_adopted"] is True
    assert output["not_promoted"] is True
    assert output["not_deployed"] is True
    assert output["stability"]["source_bindings"]["impact_cube"] == (
        fixture["stability_request"]
    )
    records = _stability_artifacts(fixture)
    assert len(records) == 1
    record = records[0]
    path = Path(record["path"])
    assert path == (
        Path(fixture["settings"].tasks_dir).absolute()
        / fixture["task"].id
        / "strategy_pool_stabilities"
        / f"{output['stability_id']}.json"
    )
    assert record["origin_tool"] == (
        "strategy.measure_strategy_pool_stability"
    )
    assert record["provenance"]["impact_cube_ref"] == (
        fixture["stability_request"]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        output["artifact"]["content_hash"]
    )
    assert len(_stability_audits(fixture)) == 1
    assert fixture["runtime"].strategies.list_for_task(
        fixture["task"].id
    ) == []


def test_pool_stability_manifest_exposes_only_exact_impact_cube_ref(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    tool = next(
        item
        for item in load_manifest(
            Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
            builtin=True,
        ).tools
        if item.name == "measure_strategy_pool_stability"
    )

    assert tool.entrypoint == "tool_measure_strategy_pool_stability"
    assert tool.side_effects == (
        "read:artifacts",
        "read:task",
        "write:artifact",
    )
    validate_against_schema(
        fixture["stability_request"],
        tool.input_schema,
        label="Pool stability input",
    )
    validate_against_schema(
        output,
        tool.output_schema,
        label="Pool stability output",
    )
    for invalid in (
        {
            **fixture["stability_request"],
            "raw_cube": fixture["impact_output"]["cube"],
        },
        {
            key: value
            for key, value in fixture["stability_request"].items()
            if key != "artifact_id"
        },
    ):
        with pytest.raises(SchemaValidationError):
            validate_against_schema(
                invalid,
                tool.input_schema,
                label="invalid Pool stability input",
            )


def test_pool_stability_exact_retry_reuses_registry_file_and_audit(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    first = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    replay = run_measure_strategy_pool_stability(
        copy.deepcopy(fixture["stability_request"]),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert replay == first
    records = _stability_artifacts(fixture)
    assert len(records) == 1
    assert len(_stability_audits(fixture)) == 1
    binding = load_strategy_pool_stability_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=first["artifact"]["artifact_id"],
        expected_artifact_content_hash=first["artifact"]["content_hash"],
        expected_stability_id=first["stability_id"],
        expected_stability_content_hash=first["content_hash"],
    )
    assert binding.stability == first["stability"]
    assert binding.impact_cube.artifact_id == (
        fixture["stability_request"]["artifact_id"]
    )
    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_pool_stability_artifact_binding_on_connection(
            conn,
            binding,
        )
        conn.commit()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_pool_stability_rejects_non_exact_tool_inputs(mutation: str) -> None:
    request = {
        "artifact_id": "1" * 64,
        "expected_artifact_content_hash": "2" * 64,
        "expected_cube_id": "strategy-impact-cube-" + "3" * 24,
        "expected_cube_content_hash": "4" * 64,
    }
    if mutation == "missing":
        request.pop("artifact_id")
    else:
        request["latest"] = True

    with pytest.raises(StrategyError, match="missing|unsupported"):
        run_measure_strategy_pool_stability(
            request,
            SimpleNamespace(task_id="task-1"),
            None,
        )


def test_pool_stability_output_validator_rejects_cached_or_binding_tamper(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    record = _stability_artifacts(fixture)[0]
    producer_run = record["provenance"]["producer_run"]
    mutations = [
        lambda value: value.__setitem__("max_psi", value["max_psi"] + 1),
        lambda value: value["artifact"].__setitem__(
            "download_url", "/forged"
        ),
        lambda value: value["producer_run_ref"].__setitem__(
            "content_hash", "0" * 64
        ),
        lambda value: value.__setitem__("effect_validation", True),
        lambda value: value.__setitem__("caller_metric", 0.99),
    ]
    for mutate in mutations:
        forged = copy.deepcopy(output)
        mutate(forged)
        with pytest.raises(StrategyError):
            validate_measure_strategy_pool_stability_tool_output(
                forged,
                trusted_task_id=fixture["task"].id,
                trusted_artifact_id=record["id"],
                trusted_artifact_content_hash=record["content_hash"],
                trusted_producer_run_id=producer_run["run_id"],
                trusted_producer_run_content_hash=producer_run["content_hash"],
            )


def test_pool_stability_source_tamper_before_commit_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    original = (
        stability_tools
        .require_strategy_impact_cube_artifact_binding_on_connection
    )
    calls = 0

    def tamper_before_second_recheck(conn, binding):
        nonlocal calls
        calls += 1
        if calls == 2:
            binding.artifact_path.write_bytes(b"tampered")
        return original(conn, binding)

    monkeypatch.setattr(
        stability_tools,
        "require_strategy_impact_cube_artifact_binding_on_connection",
        tamper_before_second_recheck,
    )

    with pytest.raises(StrategyError, match="ImpactCube"):
        run_measure_strategy_pool_stability(
            fixture["stability_request"],
            fixture["ctx"],
            fixture["runtime"],
        )

    assert calls == 2
    assert _stability_artifacts(fixture) == []
    assert _stability_audits(fixture) == []
    output_dir = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_pool_stabilities"
    )
    assert list(output_dir.glob("*.json")) == []


def test_pool_stability_rechecks_publication_before_commit_and_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    for failure_point in (
        "registry_recheck",
        "audit_recheck",
        "artifact_toctou",
    ):
        with monkeypatch.context() as patch:
            if failure_point in {"registry_recheck", "artifact_toctou"}:
                original_register = (
                    fixture["runtime"].task_artifacts.register_on_connection
                )

                def tampering_register(conn, **kwargs):
                    record = original_register(conn, **kwargs)
                    if failure_point == "registry_recheck":
                        conn.execute(
                            """
                            UPDATE task_artifacts
                               SET origin_tool = ?
                             WHERE id = ?
                            """,
                            ("forged.tool", record["id"]),
                        )
                    else:
                        path = Path(kwargs["path"])
                        path.unlink()
                        path.write_bytes(b"replacement")
                    return record

                patch.setattr(
                    fixture["runtime"].task_artifacts,
                    "register_on_connection",
                    tampering_register,
                )
            else:
                original_audit = (
                    fixture["runtime"].repo.write_audit_on_connection
                )

                def tampering_audit(conn, **kwargs):
                    original_audit(conn, **kwargs)
                    conn.execute(
                        """
                        UPDATE audit
                           SET detail_json = ?
                         WHERE kind = ? AND target_ref = ?
                        """,
                        ("{}", kwargs["kind"], kwargs["target_ref"]),
                    )

                patch.setattr(
                    fixture["runtime"].repo,
                    "write_audit_on_connection",
                    tampering_audit,
                )

            with pytest.raises((StrategyError, sqlite3.IntegrityError)):
                run_measure_strategy_pool_stability(
                    fixture["stability_request"],
                    fixture["ctx"],
                    fixture["runtime"],
                )

        assert _stability_artifacts(fixture) == []
        assert _stability_audits(fixture) == []
        output_dir = (
            Path(fixture["settings"].tasks_dir)
            / fixture["task"].id
            / "strategy_pool_stabilities"
        )
        assert list(output_dir.glob("*.json")) == []


def test_pool_stability_compensates_post_commit_file_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    stability = build_strategy_pool_stability(
        impact_cube=fixture["impact_output"]["cube"],
        impact_cube_ref=fixture["stability_request"],
    )
    final_path = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_pool_stabilities"
        / f"{stability['stability_id']}.json"
    )
    original_transaction = fixture["runtime"].task_artifacts.transaction
    tampered = False

    class CommitTamperingConnection:
        def __init__(self, transaction):
            self._transaction = transaction
            self._conn = None

        def __enter__(self):
            self._conn = self._transaction.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._transaction.__exit__(
                exc_type,
                exc_value,
                traceback,
            )

        def __getattr__(self, name):
            if self._conn is None:
                raise AttributeError(name)
            return getattr(self._conn, name)

        def commit(self):
            nonlocal tampered
            self._conn.commit()
            if not tampered and final_path.is_file():
                tampered = True
                final_path.unlink()
                final_path.write_bytes(b"post-commit replacement")

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "transaction",
        lambda: CommitTamperingConnection(original_transaction()),
    )

    with pytest.raises(StrategyError):
        run_measure_strategy_pool_stability(
            fixture["stability_request"],
            fixture["ctx"],
            fixture["runtime"],
        )

    assert tampered is True
    assert _stability_artifacts(fixture) == []
    assert _stability_audits(fixture) == []
    assert not final_path.exists()

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "transaction",
        original_transaction,
    )
    replay = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    assert replay["stability_id"] == stability["stability_id"]
    assert len(_stability_artifacts(fixture)) == 1
    assert len(_stability_audits(fixture)) == 1
    assert final_path.is_file()


def test_pool_stability_conflict_and_registration_failure_are_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    stability = build_strategy_pool_stability(
        impact_cube=fixture["impact_output"]["cube"],
        impact_cube_ref=fixture["stability_request"],
    )
    output_dir = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_pool_stabilities"
    )
    output_dir.mkdir()
    orphan = output_dir / f"{stability['stability_id']}.json"
    orphan.write_text(
        canonical_strategy_pool_stability_json(stability),
        encoding="utf-8",
    )

    with pytest.raises(StrategyError, match="without a registry row"):
        run_measure_strategy_pool_stability(
            fixture["stability_request"],
            fixture["ctx"],
            fixture["runtime"],
        )
    assert _stability_artifacts(fixture) == []
    assert _stability_audits(fixture) == []
    assert orphan.is_file()
    orphan.unlink()

    def fail_registration(*_args, **_kwargs):
        raise RuntimeError("injected registration failure")

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "register_on_connection",
        fail_registration,
    )

    with pytest.raises(RuntimeError, match="injected registration failure"):
        run_measure_strategy_pool_stability(
            fixture["stability_request"],
            fixture["ctx"],
            fixture["runtime"],
        )

    assert _stability_artifacts(fixture) == []
    assert _stability_audits(fixture) == []
    assert list(output_dir.glob("*.json")) == []


def test_pool_stability_loader_fails_closed_on_path_bytes_and_json_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    record = _stability_artifacts(fixture)[0]
    path = Path(record["path"])
    canonical = path.read_bytes()
    original_get = fixture["runtime"].task_artifacts.get_for_task

    def load(
        *,
        expected_artifact_id: str = record["id"],
        expected_artifact_hash: str = record["content_hash"],
        expected_stability_id: str = output["stability_id"],
        expected_stability_hash: str = output["content_hash"],
    ):
        return load_strategy_pool_stability_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            artifact_id=expected_artifact_id,
            expected_artifact_content_hash=expected_artifact_hash,
            expected_stability_id=expected_stability_id,
            expected_stability_content_hash=expected_stability_hash,
        )

    forged_path_record = {**record, "path": str(path.with_name("forged.json"))}
    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "get_for_task",
        lambda *_args: forged_path_record,
    )
    with pytest.raises(StrategyError, match="registry binding"):
        load()
    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "get_for_task",
        original_get,
    )

    backup = path.with_suffix(".backup")
    path.rename(backup)
    path.symlink_to(backup)
    try:
        with pytest.raises(StrategyError, match="regular file|symlink"):
            load()
    finally:
        path.unlink()
        backup.rename(path)

    hostile_payloads = [
        json.dumps(
            json.loads(canonical),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8"),
        b'{"schema_version":"forged",' + canonical[1:],
        re.sub(
            rb'"psi":[0-9.eE+-]+',
            b'"psi":NaN',
            canonical,
            count=1,
        ),
    ]
    for payload in hostile_payloads:
        payload_hash = hashlib.sha256(payload).hexdigest()
        path.write_bytes(payload)
        fake_record = {**record, "content_hash": payload_hash}
        monkeypatch.setattr(
            fixture["runtime"].task_artifacts,
            "get_for_task",
            lambda *_args, current=fake_record: current,
        )
        with pytest.raises(StrategyError):
            load(expected_artifact_hash=payload_hash)
        path.write_bytes(canonical)
    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "get_for_task",
        original_get,
    )


def test_pool_stability_loader_rebinds_embedded_impact_cube_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    forged = copy.deepcopy(output["stability"])
    forged["source_bindings"]["impact_cube"]["artifact_id"] = "f" * 64
    body = {
        key: value
        for key, value in forged.items()
        if key not in {"stability_id", "content_hash"}
    }
    forged["stability_id"] = (
        "strategy-pool-stability-"
        + hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:24]
    )
    without_hash = {
        key: value for key, value in forged.items() if key != "content_hash"
    }
    forged["content_hash"] = hashlib.sha256(
        json.dumps(
            without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    raw = canonical_strategy_pool_stability_json(forged).encode("utf-8")
    artifact_hash = hashlib.sha256(raw).hexdigest()
    path = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_pool_stabilities"
        / f"{forged['stability_id']}.json"
    )
    path.write_bytes(raw)
    artifact_id = stable_task_artifact_id(
        task_id=fixture["task"].id,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        path=str(path),
    )
    original = _stability_artifacts(fixture)[0]
    fake_record = {
        **original,
        "id": artifact_id,
        "path": str(path),
        "content_hash": artifact_hash,
    }
    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "get_for_task",
        lambda *_args: fake_record,
    )

    with pytest.raises(StrategyError, match="ImpactCube|artifact"):
        load_strategy_pool_stability_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            artifact_id=artifact_id,
            expected_artifact_content_hash=artifact_hash,
            expected_stability_id=forged["stability_id"],
            expected_stability_content_hash=forged["content_hash"],
        )


def test_pool_stability_transaction_recheck_rejects_artifact_or_audit_drift(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    binding = load_strategy_pool_stability_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=output["artifact"]["artifact_id"],
        expected_artifact_content_hash=output["artifact"]["content_hash"],
        expected_stability_id=output["stability_id"],
        expected_stability_content_hash=output["content_hash"],
    )

    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM task_artifacts WHERE id = ?",
            (binding.artifact_id,),
        )
        with pytest.raises(StrategyError, match="registry binding"):
            require_strategy_pool_stability_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()

    run_id = binding.artifact_provenance["producer_run"]["run_id"]
    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE audit
               SET outcome = 'forged'
             WHERE kind = ? AND target_ref = ?
            """,
            (POOL_STABILITY_MEASUREMENT_AUDIT_KIND, run_id),
        )
        with pytest.raises(StrategyError, match="audit binding changed"):
            require_strategy_pool_stability_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()


def test_pool_stability_transaction_recheck_ignores_temp_schema_shadows(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    binding = load_strategy_pool_stability_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=output["artifact"]["artifact_id"],
        expected_artifact_content_hash=output["artifact"]["content_hash"],
        expected_stability_id=output["stability_id"],
        expected_stability_content_hash=output["content_hash"],
    )

    with fixture["runtime"].task_artifacts.transaction() as conn:
        database_rows = conn.execute("PRAGMA database_list").fetchall()
        main_path = next(
            str(row["file"])
            for row in database_rows
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
        conn.execute(
            "CREATE TEMP TABLE audit AS SELECT * FROM main.audit"
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM main.task_artifacts WHERE id IN (?, ?)",
            (
                binding.artifact_id,
                binding.impact_cube.artifact_id,
            ),
        )
        conn.execute(
            """
            DELETE FROM main.audit
             WHERE (kind = ? AND target_ref = ?)
                OR (kind = ? AND target_ref = ?)
            """,
            (
                POOL_STABILITY_MEASUREMENT_AUDIT_KIND,
                binding.artifact_provenance["producer_run"]["run_id"],
                IMPACT_CUBE_MEASUREMENT_AUDIT_KIND,
                binding.impact_cube.artifact_provenance["producer_run"][
                    "run_id"
                ],
            ),
        )

        with pytest.raises(StrategyError):
            require_strategy_pool_stability_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()

    attacker_path = tmp_path / "attacker.sqlite"
    with sqlite3.connect(attacker_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "ATTACH DATABASE ? AS governed",
            (str(fixture["settings"].db_path),),
        )
        conn.execute(
            "CREATE TEMP TABLE pragma_database_list(name TEXT, file TEXT)"
        )
        conn.execute(
            "INSERT INTO temp.pragma_database_list(name, file) VALUES (?, ?)",
            ("main", str(fixture["settings"].db_path)),
        )
        conn.execute(
            "CREATE TEMP TABLE task_artifacts "
            "AS SELECT * FROM governed.task_artifacts"
        )
        conn.execute(
            "CREATE TEMP TABLE audit AS SELECT * FROM governed.audit"
        )
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")

        with pytest.raises(StrategyError, match="database changed"):
            require_strategy_pool_stability_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()


def test_pool_stability_renderer_is_deterministic_and_names_read_only_scope(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    record = _stability_artifacts(fixture)[0]
    context = {
        "trusted_task_id": fixture["task"].id,
        "trusted_inputs": fixture["stability_request"],
        "trusted_artifacts": {
            "pool_stability": {
                "record": record,
                "tasks_root": str(
                    Path(fixture["settings"].tasks_dir).absolute()
                ),
                "db_path": str(
                    Path(fixture["settings"].db_path).absolute()
                ),
            }
        },
    }

    first = render_tool_output(
        "measure_strategy_pool_stability",
        output,
        **context,
    )
    second = render_tool_output(
        "measure_strategy_pool_stability",
        copy.deepcopy(output),
        **context,
    )

    assert first == second
    text, tables = first
    assert "跨分区稳定性" in text
    assert "development" in text
    assert "validation" in text
    assert "不是策略效果验证" in text
    assert "未修改 Pool、未晋级、未采纳、未部署" in text
    assert output["artifact"]["download_url"] in text
    assert tables[0]["title"] == "Pool 跨分区分布稳定性"
    assert {row[0] for row in tables[0]["rows"]} == {"approval", "risk"}


def test_pool_stability_renderer_fails_closed_on_output_or_context_drift(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    record = _stability_artifacts(fixture)[0]
    artifacts = {
        "pool_stability": {
            "record": record,
            "tasks_root": str(
                Path(fixture["settings"].tasks_dir).absolute()
            ),
            "db_path": str(
                Path(fixture["settings"].db_path).absolute()
            ),
        }
    }
    forged_output = copy.deepcopy(output)
    forged_output["artifact"]["download_url"] = "/forged"
    forged_inputs = {
        **fixture["stability_request"],
        "artifact_id": "f" * 64,
    }
    forged_record = copy.deepcopy(record)
    forged_record["provenance"]["producer_run"]["content_hash"] = "0" * 64

    for value, inputs, trusted_artifacts in (
        (forged_output, fixture["stability_request"], artifacts),
        (output, forged_inputs, artifacts),
        (
            output,
            fixture["stability_request"],
            {
                "pool_stability": {
                    "record": forged_record,
                    "tasks_root": artifacts["pool_stability"]["tasks_root"],
                    "db_path": artifacts["pool_stability"]["db_path"],
                }
            },
        ),
    ):
        text, tables = render_tool_output(
            "measure_strategy_pool_stability",
            value,
            trusted_task_id=fixture["task"].id,
            trusted_inputs=inputs,
            trusted_artifacts=trusted_artifacts,
        )
        assert "结果完整性校验失败" in text
        assert output["stability_id"] not in text
        assert output["artifact"]["download_url"] not in text
        assert tables == []


def test_pool_stability_renderer_reauthenticates_live_artifact_bytes(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    record = _stability_artifacts(fixture)[0]
    context = {
        "trusted_task_id": fixture["task"].id,
        "trusted_inputs": fixture["stability_request"],
        "trusted_artifacts": {
            "pool_stability": {
                "record": record,
                "tasks_root": str(
                    Path(fixture["settings"].tasks_dir).absolute()
                ),
                "db_path": str(
                    Path(fixture["settings"].db_path).absolute()
                ),
            }
        },
    }
    Path(record["path"]).unlink()

    text, tables = render_tool_output(
        "measure_strategy_pool_stability",
        output,
        **context,
    )

    assert "结果完整性校验失败" in text
    assert output["stability_id"] not in text
    assert tables == []


def test_plan_composer_loads_registered_pool_stability_artifact(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    record = _stability_artifacts(fixture)[0]
    calls: list[tuple[str, str]] = []
    cube_step = PlanStep(
        id="impact-cube-step",
        plan_id="plan-1",
        index=0,
        title="measure exact ImpactCube",
        tool_ref=ToolRef("strategy", "measure_strategy_impact_cube"),
        inputs=fixture["impact_request"],
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
        output_ref="artifact:impact-cube",
    )
    step = PlanStep(
        id="pool-stability-step",
        plan_id="plan-1",
        index=1,
        title="measure Pool stability",
        tool_ref=ToolRef("strategy", "measure_strategy_pool_stability"),
        inputs={
            "artifact_id": (
                "$ref:impact-cube-step.output.artifact.artifact_id"
            ),
            "expected_artifact_content_hash": (
                "$ref:impact-cube-step.output.artifact.content_hash"
            ),
            "expected_cube_id": (
                "$ref:impact-cube-step.output.cube_id"
            ),
            "expected_cube_content_hash": (
                "$ref:impact-cube-step.output.content_hash"
            ),
        },
        depends_on=[cube_step.id],
        post_checks=[],
        status=StepStatus.DONE,
        output_ref="artifact:pool-stability",
    )
    plan = Plan(
        id="plan-1",
        task_id=fixture["task"].id,
        goal="measure Pool stability",
        source="agent",
        template_id=None,
        steps=[cube_step, step],
        autonomy_level=1,
    )

    def load_task_artifact(task_id: str, artifact_id: str):
        calls.append((task_id, artifact_id))
        return record

    message = PlanMessageComposer(
        load_output=lambda step_id: (
            fixture["impact_output"]
            if step_id == cube_step.id
            else output
        ),
        load_task_artifact=load_task_artifact,
        tasks_root=fixture["settings"].tasks_dir,
        db_path=fixture["settings"].db_path,
    ).done_message(plan, run_seq=1)

    assert calls == [
        (fixture["task"].id, output["artifact"]["artifact_id"])
    ]
    assert "跨分区稳定性测量完成" in message.content
    assert output["artifact"]["download_url"] in message.content

    forged_plan = copy.deepcopy(plan)
    forged_plan.steps[1].inputs["expected_cube_id"] = (
        "$ref:unrelated-step.output.cube_id"
    )
    forged = PlanMessageComposer(
        load_output=lambda step_id: (
            fixture["impact_output"]
            if step_id == cube_step.id
            else output
        ),
        load_task_artifact=load_task_artifact,
        tasks_root=fixture["settings"].tasks_dir,
        db_path=fixture["settings"].db_path,
    ).done_message(forged_plan, run_seq=2)

    assert "结果完整性校验失败" in forged.content
    assert output["stability_id"] not in forged.content

    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            DELETE FROM main.audit
             WHERE kind = ? AND target_ref = ?
            """,
            (
                POOL_STABILITY_MEASUREMENT_AUDIT_KIND,
                record["provenance"]["producer_run"]["run_id"],
            ),
        )
        conn.commit()
    missing_audit = PlanMessageComposer(
        load_output=lambda step_id: (
            fixture["impact_output"]
            if step_id == cube_step.id
            else output
        ),
        load_task_artifact=load_task_artifact,
        tasks_root=fixture["settings"].tasks_dir,
        db_path=fixture["settings"].db_path,
    ).done_message(plan, run_seq=3)

    assert "结果完整性校验失败" in missing_audit.content
    assert output["stability_id"] not in missing_audit.content
    assert missing_audit.metadata["tables"] == []
