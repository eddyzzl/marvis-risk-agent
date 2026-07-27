from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.db import TaskRepository
from marvis.domain import TaskCreate
from marvis.packs.strategy import cross_matrix_cell_selection_tools as selection_tools
from marvis.packs.strategy.cross_matrix_candidate_tools import (
    run_build_cross_matrix_candidate,
)
from marvis.packs.strategy.cross_matrix_cell_selection import (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
    canonical_cross_matrix_cell_selection_json,
    cross_matrix_cell_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_cross_matrix_candidate_tool import (
    _replace_source_with_manual_evidence,
    _replace_source_with_native_parallel_evidence,
    _setup,
)


def _fixture(
    tmp_path: Path,
    *,
    manual: bool = False,
    native: bool = False,
    age_special: str | None = None,
) -> SimpleNamespace:
    if manual and native:
        raise ValueError("manual and native fixture modes are mutually exclusive")
    base = _setup(
        tmp_path,
        age_special=age_special,
        with_split=native,
        target_bad_value=(0 if native else 1),
    )
    if native:
        _replace_source_with_native_parallel_evidence(base)
    if manual:
        _replace_source_with_manual_evidence(
            base,
            manual_breakpoints={
                "age": [30.0, 50.0],
                "score": [200.0, 320.0],
            },
        )
    matrix_result = run_build_cross_matrix_candidate(
        base["inputs"],
        base["ctx"],
        base["runtime"],
    )
    matrix_artifact = matrix_result["artifacts"][0]
    matrix = matrix_result["cross_matrix_candidate"]
    populated = [
        cell for cell in matrix["matrix"]["cells"] if cell["effect"]["count"] > 0
    ]
    inputs = {
        "source_artifact_id": matrix_artifact["artifact_id"],
        "expected_artifact_content_hash": matrix_artifact["content_hash"],
        "expected_asset_id": matrix["asset_id"],
        "expected_asset_hash": matrix["asset_hash"],
        "expected_candidate_id": matrix["candidate_evidence"]["candidate_id"],
        "expected_evidence_hash": matrix["candidate_evidence"]["evidence_hash"],
        "cell_ids": [populated[0]["cell_id"]],
    }
    repository = TaskArtifactRepository(base["settings"].db_path)
    source_record = repository.get_for_task(
        base["task"].id,
        matrix_artifact["artifact_id"],
    )
    assert source_record is not None
    return SimpleNamespace(
        **{
            **base,
            "matrix_result": matrix_result,
            "matrix": matrix,
            "matrix_artifact": matrix_artifact,
            "populated": populated,
            "inputs": inputs,
            "repository": repository,
            "source_record": source_record,
        }
    )


def _selection_records(fx: SimpleNamespace) -> list[dict]:
    return [
        record
        for record in fx.repository.list_for_task(fx.task.id)
        if record["kind"] == CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND
    ]


def _drop_immutability_and_update(
    repository: TaskArtifactRepository,
    artifact_id: str,
    **changes: object,
) -> None:
    assignments = ", ".join(f"{field} = ?" for field in changes)
    with repository.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            f"UPDATE task_artifacts SET {assignments} WHERE id = ?",  # noqa: S608
            (*changes.values(), artifact_id),
        )
        conn.commit()


def test_materialize_single_and_multi_are_canonical_idempotent_and_replayable(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    first = selection_tools.run_materialize_cross_matrix_cell_selection(
        fx.inputs,
        fx.ctx,
        fx.runtime,
    )
    repeated = selection_tools.run_materialize_cross_matrix_cell_selection(
        fx.inputs,
        fx.ctx,
        fx.runtime,
    )
    requested = [fx.populated[2]["cell_id"], fx.populated[0]["cell_id"]]
    multi = selection_tools.run_materialize_cross_matrix_cell_selection(
        {
            **fx.inputs,
            "cell_ids": requested,
            "selection_reason": "  analyst\t review  ",
        },
        fx.ctx,
        fx.runtime,
    )

    assert repeated == first
    assert set(first) == {
        "schema_version",
        "selection_id",
        "selection_hash",
        "selection_reason",
        "group_id",
        "cell_ids",
        "source_asset_id",
        "source_asset_hash",
        "source_candidate_id",
        "source_evidence_hash",
        "fragment_id",
        "fragment_type",
        "rule_id",
        "effect_id",
        "candidate_stage",
        "observation_stage",
        "validation_status",
        "artifacts",
        "not_admitted",
        "not_applied",
        "not_adopted",
        "not_deployed",
    }
    assert first["schema_version"] == selection_tools.TOOL_SCHEMA_VERSION
    assert first["selection_reason"] is None
    assert first["fragment_id"] == first["group_id"]
    assert first["fragment_type"] == "cross_matrix_cell_group"
    assert first["candidate_stage"] == "development"
    assert first["observation_stage"] == "backtested"
    assert first["validation_status"] == "unvalidated"
    assert all(
        first[field] is True
        for field in ("not_admitted", "not_applied", "not_adopted", "not_deployed")
    )
    assert set(first["artifacts"][0]) == {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
    assert first["artifacts"][0]["format"] == "json"
    assert multi["selection_reason"] == "analyst review"
    expected_order = [
        cell["cell_id"]
        for cell in fx.matrix["matrix"]["cells"]
        if cell["cell_id"] in set(requested)
    ]
    assert multi["cell_ids"] == expected_order
    assert len(_selection_records(fx)) == 2

    for output in (first, multi):
        descriptor = output["artifacts"][0]
        verified_selection = (
            selection_tools.load_verified_cross_matrix_cell_selection_artifact(
                fx.runtime,
                task_id=fx.task.id,
                artifact_id=descriptor["artifact_id"],
                expected_content_hash=descriptor["content_hash"],
                expected_asset_id=fx.matrix["asset_id"],
                expected_asset_hash=fx.matrix["asset_hash"],
            )
        )
        verified_source = selection_tools.load_verified_cross_matrix_source_artifact(
            fx.runtime,
            task_id=fx.task.id,
            artifact_id=fx.matrix_artifact["artifact_id"],
            expected_content_hash=fx.matrix_artifact["content_hash"],
            expected_asset_id=fx.matrix["asset_id"],
            expected_asset_hash=fx.matrix["asset_hash"],
            expected_candidate_id=fx.matrix["candidate_evidence"]["candidate_id"],
            expected_evidence_hash=fx.matrix["candidate_evidence"]["evidence_hash"],
        )
        assert verified_selection.selection["cell_ids"] == output["cell_ids"]
        assert verified_source.asset == fx.matrix
        fragment = cross_matrix_cell_selection_to_verified_candidate_fragment(
            verified_selection.selection,
            verified_source.asset,
            selection_artifact_binding=verified_selection.replay_binding(),
            source_artifact_binding=verified_source.builder_binding(),
        )
        assert fragment["fragment"]["fragment_id"] == output["group_id"]


def test_native_matrix_cell_selection_preserves_exact_fragment_lineage(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path, native=True)

    output = selection_tools.run_materialize_cross_matrix_cell_selection(
        fx.inputs,
        fx.ctx,
        fx.runtime,
    )
    descriptor = output["artifacts"][0]
    verified_selection = (
        selection_tools.load_verified_cross_matrix_cell_selection_artifact(
            fx.runtime,
            task_id=fx.task.id,
            artifact_id=descriptor["artifact_id"],
            expected_content_hash=descriptor["content_hash"],
            expected_asset_id=fx.matrix["asset_id"],
            expected_asset_hash=fx.matrix["asset_hash"],
        )
    )
    verified_source = selection_tools.load_verified_cross_matrix_source_artifact(
        fx.runtime,
        task_id=fx.task.id,
        artifact_id=fx.matrix_artifact["artifact_id"],
        expected_content_hash=fx.matrix_artifact["content_hash"],
        expected_asset_id=fx.matrix["asset_id"],
        expected_asset_hash=fx.matrix["asset_hash"],
        expected_candidate_id=fx.matrix["candidate_evidence"]["candidate_id"],
        expected_evidence_hash=fx.matrix["candidate_evidence"]["evidence_hash"],
    )
    fragment = cross_matrix_cell_selection_to_verified_candidate_fragment(
        verified_selection.selection,
        verified_source.asset,
        selection_artifact_binding=verified_selection.replay_binding(),
        source_artifact_binding=verified_source.builder_binding(),
    )

    native_ref = fx.source["candidate_evidence"]["generation"]["parameters"][
        "sample_design_ref"
    ]
    assert native_ref == fx.sample_design_ref
    assert native_ref["partition"] == "risk/development"
    assert fx.matrix["parent"]["candidate_id"] == fx.source["candidate_id"]
    assert fx.matrix["parent"]["evidence_hash"] == fx.source["evidence_hash"]
    selected_candidate = verified_selection.selection["source_candidate"]
    assert selected_candidate["candidate_id"] == fx.matrix[
        "candidate_evidence"
    ]["candidate_id"]
    assert selected_candidate["evidence_hash"] == fx.matrix[
        "candidate_evidence"
    ]["evidence_hash"]
    assert selected_candidate["evidence_identity"] == {
        key: fx.matrix["sample_identity"][key]
        for key in (
            "dataset_id",
            "dataset_content_hash",
            "workspace_revision",
            "workspace_generation",
            "semantic_mapping_hash",
            "sample_context_hash",
        )
    }
    assert fragment["fragment"]["fragment_id"] == output["group_id"]
    assert fragment["evidence"]["evidence_id"] == fx.matrix[
        "candidate_evidence"
    ]["candidate_id"]
    assert fragment["evidence"]["evidence_hash"] == fx.matrix[
        "candidate_evidence"
    ]["evidence_hash"]
    assert fragment["evidence"]["identity"]["sample_context_hash"] == fx.matrix[
        "sample_identity"
    ]["sample_context_hash"]


def test_materialize_manual_v2_matrix_preserves_exact_source_versions(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path, manual=True)

    output = selection_tools.run_materialize_cross_matrix_cell_selection(
        fx.inputs,
        fx.ctx,
        fx.runtime,
    )

    assert fx.matrix["schema_version"] == "strategy.cross-matrix-candidate-asset.v2"
    assert fx.source_record["provenance"]["schema_version"] == (
        "strategy.cross-matrix-candidate-artifact.v2"
    )
    descriptor = output["artifacts"][0]
    verified = selection_tools.load_verified_cross_matrix_cell_selection_artifact(
        fx.runtime,
        task_id=fx.task.id,
        artifact_id=descriptor["artifact_id"],
        expected_content_hash=descriptor["content_hash"],
        expected_asset_id=fx.matrix["asset_id"],
        expected_asset_hash=fx.matrix["asset_hash"],
    )
    assert verified.selection["source_artifact"]["artifact_schema_version"] == (
        "strategy.cross-matrix-candidate-artifact.v2"
    )
    assert verified.selection["source_asset"]["schema_version"] == (
        "strategy.cross-matrix-candidate-asset.v2"
    )
    selection_record = fx.repository.get_for_task(
        fx.task.id,
        descriptor["artifact_id"],
    )
    assert selection_record is not None
    assert selection_record["provenance"]["source_artifact_schema_version"] == (
        "strategy.cross-matrix-candidate-artifact.v2"
    )
    assert selection_record["provenance"]["source_asset_schema_version"] == (
        "strategy.cross-matrix-candidate-asset.v2"
    )


@pytest.mark.parametrize(
    ("manual", "changes"),
    [
        (
            False,
            {"schema_version": "strategy.cross-matrix-candidate-artifact.v2"},
        ),
        (
            True,
            {"schema_version": "strategy.cross-matrix-candidate-artifact.v1"},
        ),
        (
            True,
            {"producer_version": "strategy.cross-matrix-candidate-asset/1"},
        ),
        (
            True,
            {"asset_schema_version": "strategy.cross-matrix-candidate-asset.v1"},
        ),
    ],
)
def test_source_loader_rejects_mixed_asset_and_provenance_versions(
    tmp_path: Path,
    manual: bool,
    changes: dict[str, str],
) -> None:
    fx = _fixture(tmp_path, manual=manual)
    provenance = {
        **fx.source_record["provenance"],
        **changes,
    }
    _drop_immutability_and_update(
        fx.repository,
        fx.matrix_artifact["artifact_id"],
        provenance_json=selection_tools._canonical_json(provenance),
    )

    with pytest.raises(StrategyError, match="provenance|source artifact"):
        selection_tools.load_verified_cross_matrix_source_artifact(
            fx.runtime,
            task_id=fx.task.id,
            artifact_id=fx.matrix_artifact["artifact_id"],
            expected_content_hash=fx.matrix_artifact["content_hash"],
            expected_asset_id=fx.matrix["asset_id"],
            expected_asset_hash=fx.matrix["asset_hash"],
            expected_candidate_id=fx.matrix["candidate_evidence"]["candidate_id"],
            expected_evidence_hash=fx.matrix["candidate_evidence"]["evidence_hash"],
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"cell_ids": []}, "non-empty"),
        ({"cell_ids": ["cross-cell-" + "0" * 32]}, "unknown"),
        ({"expected_asset_hash": "0" * 64}, "asset_hash"),
        ({"expected_evidence_hash": "0" * 64}, "evidence_hash"),
    ],
)
def test_materialize_rejects_empty_unknown_or_wrong_exact_binding(
    tmp_path: Path,
    override: dict,
    message: str,
) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(StrategyError, match=message):
        selection_tools.run_materialize_cross_matrix_cell_selection(
            {**fx.inputs, **override},
            fx.ctx,
            fx.runtime,
        )
    assert _selection_records(fx) == []


def test_source_loader_rejects_foreign_task_noncanonical_path_and_symlink(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    foreign = TaskRepository(fx.settings.db_path).create_task(
        TaskCreate(
            model_name="foreign-cross",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
            target_col="bad",
        )
    )
    kwargs = {
        "artifact_id": fx.matrix_artifact["artifact_id"],
        "expected_content_hash": fx.matrix_artifact["content_hash"],
        "expected_asset_id": fx.matrix["asset_id"],
        "expected_asset_hash": fx.matrix["asset_hash"],
        "expected_candidate_id": fx.matrix["candidate_evidence"]["candidate_id"],
        "expected_evidence_hash": fx.matrix["candidate_evidence"]["evidence_hash"],
    }
    with pytest.raises(StrategyError, match="not found"):
        selection_tools.load_verified_cross_matrix_source_artifact(
            fx.runtime,
            task_id=foreign.id,
            **kwargs,
        )

    _drop_immutability_and_update(
        fx.repository,
        fx.matrix_artifact["artifact_id"],
        path=str(tmp_path / "forged.json"),
    )
    with pytest.raises(StrategyError, match="path is not canonical"):
        selection_tools.load_verified_cross_matrix_source_artifact(
            fx.runtime,
            task_id=fx.task.id,
            **kwargs,
        )

    # A fresh fixture isolates the deliberate registry corruption above.
    symlink_fx = _fixture(tmp_path / "symlink-case")
    source_path = Path(symlink_fx.source_record["path"])
    target = source_path.with_suffix(".real.json")
    source_path.rename(target)
    source_path.symlink_to(target)
    with pytest.raises(StrategyError, match="symlink"):
        selection_tools.load_verified_cross_matrix_source_artifact(
            symlink_fx.runtime,
            task_id=symlink_fx.task.id,
            artifact_id=symlink_fx.matrix_artifact["artifact_id"],
            expected_content_hash=symlink_fx.matrix_artifact["content_hash"],
            expected_asset_id=symlink_fx.matrix["asset_id"],
            expected_asset_hash=symlink_fx.matrix["asset_hash"],
            expected_candidate_id=symlink_fx.matrix["candidate_evidence"][
                "candidate_id"
            ],
            expected_evidence_hash=symlink_fx.matrix["candidate_evidence"][
                "evidence_hash"
            ],
        )


def test_selection_loader_rejects_duplicate_key_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    for case in ("duplicate", "noncanonical"):
        fx = _fixture(tmp_path / case)
        output = selection_tools.run_materialize_cross_matrix_cell_selection(
            fx.inputs,
            fx.ctx,
            fx.runtime,
        )
        descriptor = output["artifacts"][0]
        record = fx.repository.get_for_task(fx.task.id, descriptor["artifact_id"])
        assert record is not None
        path = Path(record["path"])
        canonical = path.read_bytes()
        if case == "duplicate":
            tampered = (
                b'{"schema_version":"strategy.cross-matrix-cell-selection.v1",'
                + canonical[1:]
            )
        else:
            tampered = canonical + b"\n"
        path.write_bytes(tampered)
        tampered_hash = hashlib.sha256(tampered).hexdigest()
        _drop_immutability_and_update(
            fx.repository,
            descriptor["artifact_id"],
            content_hash=tampered_hash,
        )
        with pytest.raises(
            StrategyError, match="duplicate JSON key|not canonical JSON"
        ):
            selection_tools.load_verified_cross_matrix_cell_selection_artifact(
                fx.runtime,
                task_id=fx.task.id,
                artifact_id=descriptor["artifact_id"],
                expected_content_hash=tampered_hash,
                expected_asset_id=fx.matrix["asset_id"],
                expected_asset_hash=fx.matrix["asset_hash"],
            )


def test_verified_loaders_reject_wrong_expected_content_hash(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(StrategyError, match="content hash changed"):
        selection_tools.load_verified_cross_matrix_source_artifact(
            fx.runtime,
            task_id=fx.task.id,
            artifact_id=fx.matrix_artifact["artifact_id"],
            expected_content_hash="0" * 64,
            expected_asset_id=fx.matrix["asset_id"],
            expected_asset_hash=fx.matrix["asset_hash"],
            expected_candidate_id=fx.matrix["candidate_evidence"]["candidate_id"],
            expected_evidence_hash=fx.matrix["candidate_evidence"]["evidence_hash"],
        )

    output = selection_tools.run_materialize_cross_matrix_cell_selection(
        fx.inputs,
        fx.ctx,
        fx.runtime,
    )
    descriptor = output["artifacts"][0]
    with pytest.raises(StrategyError, match="content hash changed"):
        selection_tools.load_verified_cross_matrix_cell_selection_artifact(
            fx.runtime,
            task_id=fx.task.id,
            artifact_id=descriptor["artifact_id"],
            expected_content_hash="0" * 64,
            expected_asset_id=fx.matrix["asset_id"],
            expected_asset_hash=fx.matrix["asset_hash"],
        )


def test_source_and_dataset_drift_roll_back_staged_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for drift in ("source", "dataset"):
        fx = _fixture(tmp_path / drift)
        original = selection_tools._persist_selection

        def drift_then_persist(*args, **kwargs):
            if drift == "source":
                path = Path(fx.source_record["path"])
            else:
                path = Path(fx.runtime.registry.resolve_verified_path(fx.dataset.id))
            path.write_bytes(path.read_bytes() + b"drift")
            return original(*args, **kwargs)

        monkeypatch.setattr(selection_tools, "_persist_selection", drift_then_persist)
        with pytest.raises(StrategyError, match="drift|hash|binding"):
            selection_tools.run_materialize_cross_matrix_cell_selection(
                fx.inputs,
                fx.ctx,
                fx.runtime,
            )
        assert _selection_records(fx) == []
        out_dir = (
            Path(fx.settings.tasks_dir)
            / fx.task.id
            / "strategy_cross_matrix_cell_selections"
        )
        assert not out_dir.exists() or list(out_dir.glob("*.json")) == []
        monkeypatch.setattr(selection_tools, "_persist_selection", original)


def test_registration_failure_rolls_back_promoted_file_and_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)

    def fail_registration(*args, **kwargs):
        raise StrategyError("synthetic registration failure")

    monkeypatch.setattr(
        fx.runtime.task_artifacts,
        "register_on_connection",
        fail_registration,
    )
    with pytest.raises(StrategyError, match="synthetic registration failure"):
        selection_tools.run_materialize_cross_matrix_cell_selection(
            fx.inputs,
            fx.ctx,
            fx.runtime,
        )
    assert _selection_records(fx) == []
    out_dir = (
        Path(fx.settings.tasks_dir)
        / fx.task.id
        / "strategy_cross_matrix_cell_selections"
    )
    assert not out_dir.exists() or list(out_dir.glob("*.json")) == []


def test_selection_persistence_is_pointer_only(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    output = selection_tools.run_materialize_cross_matrix_cell_selection(
        fx.inputs,
        fx.ctx,
        fx.runtime,
    )
    record = fx.repository.get_for_task(
        fx.task.id,
        output["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    persisted = Path(record["path"]).read_bytes()
    selection = selection_tools._strict_cell_selection_from_bytes(persisted)
    assert persisted == canonical_cross_matrix_cell_selection_json(selection).encode(
        "utf-8"
    )
    assert record["provenance"] == (
        selection_tools.cross_matrix_cell_selection_provenance(selection)
    )
    forbidden = {
        "condition",
        "rule",
        "effect",
        "metrics",
        "action",
        "lifecycle",
        "candidate_stage",
        "observation_stage",
        "validation_status",
    }
    stack: list[object] = [selection]
    keys: set[str] = set()
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            keys.update(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    assert forbidden.isdisjoint(keys)


def test_selection_provenance_tamper_is_rejected(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    output = selection_tools.run_materialize_cross_matrix_cell_selection(
        fx.inputs,
        fx.ctx,
        fx.runtime,
    )
    descriptor = output["artifacts"][0]
    record = fx.repository.get_for_task(fx.task.id, descriptor["artifact_id"])
    assert record is not None
    provenance = deepcopy(record["provenance"])
    provenance["cell_ids"] = ["cross-cell-" + "0" * 32]
    _drop_immutability_and_update(
        fx.repository,
        descriptor["artifact_id"],
        provenance_json=selection_tools._canonical_json(provenance),
    )
    with pytest.raises(StrategyError, match="provenance"):
        selection_tools.load_verified_cross_matrix_cell_selection_artifact(
            fx.runtime,
            task_id=fx.task.id,
            artifact_id=descriptor["artifact_id"],
            expected_content_hash=descriptor["content_hash"],
            expected_asset_id=fx.matrix["asset_id"],
            expected_asset_hash=fx.matrix["asset_hash"],
        )
