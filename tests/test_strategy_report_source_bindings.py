from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import uuid

import pytest

from marvis.db import TaskRepository
from marvis.db_schema import connect, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy import model_evidence_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.model_evidence_tools import (
    StrategyModelEvidenceV2ArtifactBinding,
    load_strategy_model_evidence_v2_artifact,
    require_strategy_model_evidence_v2_artifact_binding_on_connection,
    run_materialize_model_evidence_v2,
)
from marvis.packs.strategy.project_context_tools import (
    StrategyProjectContextArtifactBinding,
    load_current_strategy_project_context,
    load_current_strategy_project_context_artifact,
    require_strategy_project_context_artifact_binding_on_connection,
    run_materialize_project_context,
)

from test_strategy_model_evidence_tool import (
    _fixture as _model_fixture,
    _registered_model_evidence,
)
from test_strategy_project_context_tool import (
    _request_bound_to_message,
    _setup as _project_fixture,
)


def _project_binding(tmp_path: Path) -> tuple[dict, dict, StrategyProjectContextArtifactBinding]:
    fx = _project_fixture(tmp_path)
    output = run_materialize_project_context(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    binding = load_current_strategy_project_context_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
    )
    assert isinstance(binding, StrategyProjectContextArtifactBinding)
    return fx, output, binding


def _model_binding(tmp_path: Path) -> tuple[dict, dict, StrategyModelEvidenceV2ArtifactBinding]:
    fx = _model_fixture(tmp_path)
    output = run_materialize_model_evidence_v2(
        fx["inputs"],
        fx["ctx"],
        fx["runtime"],
    )
    record = _registered_model_evidence(fx)
    binding = load_strategy_model_evidence_v2_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        artifact_id=record["id"],
        expected_artifact_content_hash=record["content_hash"],
        expected_bundle_id=output["bundle_id"],
        expected_bundle_content_hash=output["bundle_content_hash"],
        sample_design_ref=fx["inputs"]["sample_design_ref"],
    )
    return fx, output, binding


def _assert_transaction_remains_caller_owned(
    *,
    db_path: Path,
    task_id: str,
    require,
) -> None:
    marker = uuid.uuid4().hex
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO agent_messages(
                id, task_id, role, stage, content, created_at, metadata_json
            ) VALUES (?, ?, 'assistant', 'test', 'transaction marker',
                      '2026-07-23T00:00:00+00:00', '{}')
            """,
            (marker, task_id),
        )
        require(conn)
        assert conn.in_transaction
        conn.rollback()
    with connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM agent_messages WHERE id = ?",
                (marker,),
            ).fetchone()
            is None
        )


def test_project_context_binding_is_current_authenticated_and_caller_owned(
    tmp_path: Path,
) -> None:
    fx, output, binding = _project_binding(tmp_path)

    assert binding.revision == output["revision"]
    assert (
        load_current_strategy_project_context(
            fx["runtime"],
            task_id=fx["task"].id,
        )
        == binding.revision
    )
    _assert_transaction_remains_caller_owned(
        db_path=fx["settings"].db_path,
        task_id=fx["task"].id,
        require=lambda conn: (
            require_strategy_project_context_artifact_binding_on_connection(
                conn,
                binding,
            )
        ),
    )
    with connect(fx["settings"].db_path) as conn:
        with pytest.raises(StrategyError, match="caller-owned transaction"):
            require_strategy_project_context_artifact_binding_on_connection(
                conn,
                binding,
            )


def test_project_context_binding_rejects_stale_head(tmp_path: Path) -> None:
    fx, output, binding = _project_binding(tmp_path)
    request = _request_bound_to_message(
        fx,
        "补充：本次项目还覆盖复借客群。",
        expected_revision=output["revision"]["revision"],
        expected_revision_id=output["revision"]["revision_id"],
        expected_state_hash=output["revision"]["state_hash"],
        scope="贷前准入与复借策略",
    )
    updated = run_materialize_project_context(request, fx["ctx"], fx["runtime"])
    assert updated["revision"]["revision"] == binding.revision["revision"] + 1

    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="no longer current"):
            require_strategy_project_context_artifact_binding_on_connection(
                conn,
                binding,
            )


def test_project_context_binding_rejects_cross_task_and_registry_loss(
    tmp_path: Path,
) -> None:
    fx, _, binding = _project_binding(tmp_path)
    foreign = TaskRepository(fx["settings"].db_path).create_task(
        TaskCreate(
            model_name="foreign",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
        )
    )
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="current|ownership"):
            require_strategy_project_context_artifact_binding_on_connection(
                conn,
                replace(binding, task_id=foreign.id),
            )
        conn.rollback()
    with connect(fx["settings"].db_path) as conn:
        conn.execute(
            "DELETE FROM task_artifacts WHERE id = ?",
            (binding.artifact_id,),
        )
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="not registered"):
            require_strategy_project_context_artifact_binding_on_connection(
                conn,
                binding,
            )


@pytest.mark.parametrize("target", ["payload", "upstream"])
def test_project_context_binding_rejects_post_load_tampering(
    tmp_path: Path,
    target: str,
) -> None:
    fx, _, binding = _project_binding(tmp_path)
    if target == "payload":
        binding.artifact_path.write_bytes(
            binding.artifact_path.read_bytes() + b"tampered"
        )
    else:
        message_id = fx["request"]["user_message_ref"]["message_id"]
        with connect(fx["settings"].db_path) as conn:
            conn.execute(
                "UPDATE agent_messages SET content = content || 'tampered' WHERE id = ?",
                (message_id,),
            )

    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="hash|bytes|source changed"):
            require_strategy_project_context_artifact_binding_on_connection(
                conn,
                binding,
            )


def test_model_evidence_binding_is_authenticated_and_caller_owned(
    tmp_path: Path,
) -> None:
    fx, output, binding = _model_binding(tmp_path)

    assert binding.bundle == output["bundle"]
    assert isinstance(binding.sources, tuple)
    assert isinstance(binding.warnings, tuple)
    _assert_transaction_remains_caller_owned(
        db_path=fx["settings"].db_path,
        task_id=fx["task"].id,
        require=lambda conn: (
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                binding,
            )
        ),
    )
    with connect(fx["settings"].db_path) as conn:
        with pytest.raises(StrategyError, match="caller-owned transaction"):
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                binding,
            )


@pytest.mark.parametrize("target", ["output_registry", "output_file", "source_file"])
def test_model_evidence_binding_rejects_post_load_artifact_tampering(
    tmp_path: Path,
    target: str,
) -> None:
    fx, _, binding = _model_binding(tmp_path)
    if target == "output_registry":
        with connect(fx["settings"].db_path) as conn:
            conn.execute(
                "DELETE FROM task_artifacts WHERE id = ?",
                (binding.artifact_id,),
            )
    else:
        path = binding.path if target == "output_file" else binding.sources[0].path
        path.write_bytes(path.read_bytes() + b"tampered")

    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="disappeared|hash|bytes|canonical"):
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                binding,
            )


def test_model_evidence_binding_rejects_cross_task_and_upstream_toctou(
    tmp_path: Path,
) -> None:
    fx, _, binding = _model_binding(tmp_path)
    foreign = TaskRepository(fx["settings"].db_path).create_task(
        TaskCreate(
            model_name="foreign",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
        )
    )
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="another task|binding|disappeared"):
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                replace(binding, task_id=foreign.id),
            )
        conn.rollback()
    with connect(fx["settings"].db_path) as conn:
        conn.execute(
            "UPDATE data_workspaces SET revision = revision + 1 WHERE task_id = ?",
            (fx["task"].id,),
        )
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="workspace|binding|DataWorkspace"):
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                binding,
            )


def test_model_evidence_binding_requires_original_database_and_fixed_task_root(
    tmp_path: Path,
) -> None:
    fx, _, binding = _model_binding(tmp_path)
    clone_path = tmp_path / "other.sqlite"
    init_db(clone_path)
    with connect(clone_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="database changed"):
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                binding,
            )
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="task root changed"):
            require_strategy_model_evidence_v2_artifact_binding_on_connection(
                conn,
                replace(binding, tasks_root=tmp_path / "forged-root"),
            )


def test_model_evidence_reader_detects_same_inode_change_during_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, binding = _model_binding(tmp_path)
    original_read = model_evidence_tools.os.read
    changed = False

    def mutate_after_first_chunk(descriptor: int, count: int) -> bytes:
        nonlocal changed
        chunk = original_read(descriptor, count)
        if chunk and not changed:
            changed = True
            binding.path.write_bytes(binding.path.read_bytes() + b"tampered")
        return chunk

    monkeypatch.setattr(model_evidence_tools.os, "read", mutate_after_first_chunk)
    with pytest.raises(StrategyError, match="changed while being read"):
        model_evidence_tools._read_verified(
            binding.path,
            root=binding.tasks_root,
            expected_hash=binding.artifact_content_hash,
            maximum_bytes=10 * 1024 * 1024,
        )


def test_model_evidence_reader_detects_atomic_path_replacement_during_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, binding = _model_binding(tmp_path)
    original_read = model_evidence_tools.os.read
    replaced = False

    def replace_after_first_chunk(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, count)
        if chunk and not replaced:
            replaced = True
            backup = binding.path.with_name(binding.path.name + ".opened")
            binding.path.replace(backup)
            binding.path.write_bytes(b"tampered-at-registry-path")
        return chunk

    monkeypatch.setattr(model_evidence_tools.os, "read", replace_after_first_chunk)
    with pytest.raises(StrategyError, match="registry path changed"):
        model_evidence_tools._read_verified(
            binding.path,
            root=binding.tasks_root,
            expected_hash=binding.artifact_content_hash,
            maximum_bytes=10 * 1024 * 1024,
        )
