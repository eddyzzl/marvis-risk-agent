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
import marvis.packs.strategy.dsl_delivery_tools as delivery_tools
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


def _artifact_projections(output: dict) -> dict[str, dict[str, str]]:
    return {
        name: {
            "artifact_id": output["artifacts"][index]["artifact_id"],
            "content_hash": output["artifacts"][index]["content_hash"],
        }
        for index, name in enumerate(
            ("python", "sql", "strategy_json", "equivalence_json")
        )
    }


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
        expected_artifacts=_artifact_projections(output),
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


def test_export_strategy_delivery_rejects_tampered_registered_file(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, runtime = _run(fixture, request)
    row = next(
        item
        for item in _artifact_rows(runtime, fixture[1].id)
        if item["kind"] == DELIVERY_ARTIFACT_KINDS["python"]
    )
    path = Path(row["path"])
    path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        StrategyDeliveryToolError,
        match="existing.*bytes|artifact.*changed",
    ):
        _run(fixture, request)

    assert path.read_text(encoding="utf-8") == "tampered\n"
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 1


def test_export_strategy_delivery_recovers_exact_promoted_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    original = {
        str(row["path"]): Path(row["path"]).read_bytes()
        for row in rows
    }
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )

    replay, _runtime_again = _run(fixture, request)

    assert replay == first
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert all(Path(path).read_bytes() == raw for path, raw in original.items())


def test_export_strategy_delivery_recovers_exact_partial_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    original = {
        str(row["path"]): Path(row["path"]).read_bytes()
        for row in rows
    }
    retained_path = Path(rows[0]["path"])
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    for row in rows[1:]:
        Path(row["path"]).unlink()

    replay, _runtime_again = _run(fixture, request)

    assert replay == first
    assert retained_path.read_bytes() == original[str(retained_path)]
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert all(Path(path).read_bytes() == raw for path, raw in original.items())


def test_export_strategy_delivery_rejects_drifted_partial_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    _first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    retained_path = Path(rows[0]["path"])
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    for row in rows[1:]:
        Path(row["path"]).unlink()
    retained_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(StrategyDeliveryToolError, match="artifact bytes changed"):
        _run(fixture, request)

    assert retained_path.read_text(encoding="utf-8") == "tampered\n"
    assert _artifact_rows(runtime, fixture[1].id) == []


def test_export_strategy_delivery_rejects_symlinked_partial_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    _first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    retained_path = Path(rows[0]["path"])
    outside = tmp_path / "outside-delivery.py"
    outside.write_bytes(retained_path.read_bytes())
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    for row in rows[1:]:
        Path(row["path"]).unlink()
    retained_path.unlink()
    retained_path.symlink_to(outside)

    with pytest.raises(
        StrategyDeliveryToolError,
        match="regular file|unavailable|must stay under",
    ):
        _run(fixture, request)

    assert retained_path.is_symlink()
    assert _artifact_rows(runtime, fixture[1].id) == []


def test_export_strategy_delivery_rejects_drifted_audit_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE audit
               SET detail_json = '{}'
             WHERE kind = ? AND target_ref = ?
            """,
            (DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("actor", "operator"), ("outcome", "failed")),
)
def test_export_strategy_delivery_rejects_drifted_audit_outcome_on_retry(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            f"UPDATE audit SET {field} = ? "
            "WHERE kind = ? AND target_ref = ?",
            (replacement, DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)


def test_export_strategy_delivery_rejects_drifted_audit_inputs_hash_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE audit
               SET inputs_hash = ?
             WHERE kind = ? AND target_ref = ?
            """,
            ("0" * 64, DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 1


@pytest.mark.parametrize("field", ("kind", "target_ref"))
def test_export_strategy_delivery_rejects_drifted_audit_identity_on_retry(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    replacement = (
        f"{DELIVERY_AUDIT_KIND}.tampered"
        if field == "kind"
        else f"{first['delivery_id']}-tampered"
    )
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            f"UPDATE audit SET {field} = ? "
            "WHERE kind = ? AND target_ref = ?",
            (replacement, DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_export_strategy_delivery_rejects_joint_audit_identity_drift_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE audit
               SET kind = ?, target_ref = ?
             WHERE kind = ? AND target_ref = ?
            """,
            (
                f"{DELIVERY_AUDIT_KIND}.tampered",
                f"{first['delivery_id']}-tampered",
                DELIVERY_AUDIT_KIND,
                first["delivery_id"],
            ),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_export_strategy_delivery_rejects_duplicate_audit_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit(
                id, kind, actor, target_ref, inputs_hash, outcome,
                detail_json, at
            )
            SELECT lower(hex(randomblob(16))), kind, actor, target_ref,
                   inputs_hash, outcome, detail_json, at
              FROM audit
             WHERE kind = ? AND target_ref = ?
            """,
            (DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 2


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


def test_export_strategy_delivery_rejects_symlink_swap_after_promotion_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    outside = tmp_path / "outside-strategy.py"
    original = delivery_tools._require_exact_delivery_file
    swapped = False

    def swap_after_first_check(path, *, root, expected, expected_hash):
        nonlocal swapped
        original(
            path,
            root=root,
            expected=expected,
            expected_hash=expected_hash,
        )
        if not swapped and Path(path).name == "strategy.py":
            outside.write_bytes(expected)
            Path(path).unlink()
            Path(path).symlink_to(outside)
            swapped = True

    monkeypatch.setattr(
        delivery_tools,
        "_require_exact_delivery_file",
        swap_after_first_check,
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="regular file|unavailable|bytes changed",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert swapped is True
    assert _artifact_rows(runtime, fixture[1].id) == []


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
            expected_artifacts=_artifact_projections(output),
        )


def test_delivery_output_validator_rejects_sample_count_above_declared_budget(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    forged["maximum_equivalence_rows"] = 1
    forged["delivery_id"] = delivery_tools._delivery_id(
        strategy_ref=forged["strategy_ref"],
        dataset_ref=forged["dataset_ref"],
        maximum_equivalence_rows=1,
        equivalence=forged["equivalence"],
        content_hashes={
            name: forged["artifacts"][index]["content_hash"]
            for index, name in enumerate(
                ("python", "sql", "strategy_json", "equivalence_json")
            )
        },
    )

    with pytest.raises(StrategyDeliveryToolError, match="sample_count.*budget"):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_artifacts=_artifact_projections(output),
        )


@pytest.mark.parametrize(
    ("artifact_index", "artifact_name"),
    tuple(enumerate(("python", "sql", "strategy_json", "equivalence_json"))),
)
def test_delivery_output_validator_binds_all_published_artifact_content(
    tmp_path: Path,
    artifact_index: int,
    artifact_name: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    forged_hash = "0" * 64
    assert forged_hash != forged["artifacts"][artifact_index]["content_hash"]
    forged["artifacts"][artifact_index]["content_hash"] = forged_hash
    forged["artifacts"][artifact_index]["download_url"] = (
        forged["artifacts"][artifact_index]["download_url"].rsplit("=", 1)[0]
        + f"={forged_hash}"
    )
    forged["delivery_id"] = delivery_tools._delivery_id(
        strategy_ref=forged["strategy_ref"],
        dataset_ref=forged["dataset_ref"],
        maximum_equivalence_rows=forged["maximum_equivalence_rows"],
        equivalence=forged["equivalence"],
        content_hashes={
            name: forged["artifacts"][index]["content_hash"]
            for index, name in enumerate(
                ("python", "sql", "strategy_json", "equivalence_json")
            )
        },
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match=f"artifacts.*{artifact_name}|authenticated publication",
    ):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_artifacts=_artifact_projections(output),
        )


def test_delivery_output_validator_binds_equivalence_artifact_to_document_bytes(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    equivalence_artifact = forged["artifacts"][3]
    forged_hash = "0" * 64
    assert forged_hash != equivalence_artifact["content_hash"]
    equivalence_artifact["content_hash"] = forged_hash
    equivalence_artifact["download_url"] = (
        equivalence_artifact["download_url"].rsplit("=", 1)[0]
        + f"={forged_hash}"
    )
    forged["delivery_id"] = delivery_tools._delivery_id(
        strategy_ref=forged["strategy_ref"],
        dataset_ref=forged["dataset_ref"],
        maximum_equivalence_rows=forged["maximum_equivalence_rows"],
        equivalence=forged["equivalence"],
        content_hashes={
            name: forged["artifacts"][index]["content_hash"]
            for index, name in enumerate(
                ("python", "sql", "strategy_json", "equivalence_json")
            )
        },
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="equivalence.*artifact.*content",
    ):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_artifacts=_artifact_projections(forged),
        )


def test_export_strategy_delivery_authenticates_snapshot_and_artifact_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    private_reads: list[object] = []
    original_read_parquet = delivery_tools.pd.read_parquet

    def reject_live_path_read(*args, **kwargs):
        raise AssertionError("delivery must not reopen the live dataset path")

    def record_private_read(source, *args, **kwargs):
        assert not isinstance(source, (str, Path))
        private_reads.append(source)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(runtime.backend, "read_frame", reject_live_path_read)
    monkeypatch.setattr(
        delivery_tools.pd,
        "read_parquet",
        record_private_read,
    )

    output = run_export_strategy_delivery(
        request,
        fixture[-1],
        runtime,
    )
    assert len(private_reads) == 1
    for artifact in output["artifacts"]:
        assert artifact["download_url"].endswith(
            f"?expected_content_hash={artifact['content_hash']}"
        )

    forged = deepcopy(output)
    forged["artifacts"][0]["artifact_id"] = "0" * 64
    forged["artifacts"][0]["download_url"] = forged["artifacts"][0][
        "download_url"
    ].replace(
        output["artifacts"][0]["artifact_id"],
        "0" * 64,
    )
    with pytest.raises(
        StrategyDeliveryToolError,
        match="authenticated publication",
    ):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_artifacts=_artifact_projections(output),
        )


def test_export_strategy_delivery_revalidates_source_without_path_hash_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    dataset_path = Path(runtime.registry.resolve_path(fixture[3].id))
    original_hash = delivery_tools.sha256_file
    original_authenticate = delivery_tools._require_authenticated_file_hash
    authenticated: list[Path] = []

    def reject_dataset_path_hash(path):
        if Path(path) == dataset_path:
            raise AssertionError("delivery must not follow the live dataset path")
        return original_hash(path)

    def record_authenticated(path, *, root, expected_hash):
        authenticated.append(Path(path))
        return original_authenticate(
            path,
            root=root,
            expected_hash=expected_hash,
        )

    monkeypatch.setattr(delivery_tools, "sha256_file", reject_dataset_path_hash)
    monkeypatch.setattr(
        delivery_tools,
        "_require_authenticated_file_hash",
        record_authenticated,
    )

    output = run_export_strategy_delivery(request, fixture[-1], runtime)

    assert output["dataset_ref"] == request["dataset_ref"]
    assert authenticated == [dataset_path]
