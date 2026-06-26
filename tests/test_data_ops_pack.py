import hashlib
import sys
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, init_db
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    return runner, registry, data_repo


def _register_csv(registry, tmp_path, name: str, frame: pd.DataFrame, *, role: str):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return registry.register_from_upload("task-1", path, role=role)


def test_data_ops_ingest_excel_and_infer_schema_via_runner(tmp_path):
    runner, _registry, repo = _runtime(tmp_path)
    workbook_path = tmp_path / "features.xlsx"
    pd.DataFrame({
        "mobile": ["13800138000", "13900139000"],
        "bad_flag": [0, 1],
    }).to_excel(workbook_path, sheet_name="Sheet1", index=False)

    ingest = runner.invoke(
        ToolRef("data_ops", "ingest_excel"),
        {"path": str(workbook_path), "sheets": ["Sheet1"], "role": "sample"},
        task_id="task-1",
    )

    assert ingest.ok is True
    assert len(ingest.output["datasets"]) == 1
    dataset_id = ingest.output["datasets"][0]["id"]
    assert repo.get_dataset(dataset_id) is not None

    schema = runner.invoke(
        ToolRef("data_ops", "infer_schema"),
        {"dataset_id": dataset_id},
        task_id="task-1",
    )

    assert schema.ok is True
    assert schema.output["has_target"] is True
    assert schema.output["target_col"] == "bad_flag"


def test_data_ops_align_propose_and_execute_join_via_runner(tmp_path):
    runner, registry, repo = _runtime(tmp_path)
    anchor = _register_csv(
        registry,
        tmp_path,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"]}),
        role="sample",
    )
    feature = _register_csv(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "balance": [10, 20],
        }),
        role="feature",
    )

    align = runner.invoke(
        ToolRef("data_ops", "align_columns"),
        {"anchor_id": anchor.id, "feature_ids": [feature.id]},
        task_id="task-1",
    )
    propose = runner.invoke(
        ToolRef("data_ops", "propose_join"),
        {"anchor_id": anchor.id, "feature_ids": [feature.id]},
        task_id="task-1",
    )

    assert align.ok is True
    assert align.output["alignments"][0]["key_pairs"][0]["match_method"] == "hash:md5"
    assert propose.ok is True
    assert propose.output["anchor_dataset_id"] == anchor.id
    assert propose.output["joins"][0]["diagnostics"]["fan_out_detected"] is False

    unconfirmed = runner.invoke(
        ToolRef("data_ops", "execute_join"),
        {"join_plan_id": propose.output["join_plan_id"]},
        task_id="task-1",
    )

    assert unconfirmed.ok is False
    assert unconfirmed.error_kind == "execution"
    assert "confirmed" in unconfirmed.error

    plan = repo.load_join_plan(propose.output["join_plan_id"])
    spec = plan.joins[0]
    spec.confirmed = True
    repo.update_join_spec(plan.id, spec)
    executed = runner.invoke(
        ToolRef("data_ops", "execute_join"),
        {"join_plan_id": plan.id},
        task_id="task-1",
    )

    assert executed.ok is True
    assert executed.output["anchor_rows"] == 2
    assert executed.output["joined_rows"] == 2
    assert repo.get_dataset(executed.output["result_dataset_id"]) is not None


def test_data_ops_clean_format_and_dedup_rows_via_runner(tmp_path):
    runner, registry, repo = _runtime(tmp_path)
    dataset = _register_csv(
        registry,
        tmp_path,
        "dirty",
        pd.DataFrame({"acct": [" A1 ", "a1", "B2"], "value": ["1", "1", "2"]}),
        role="feature",
    )

    cleaned = runner.invoke(
        ToolRef("data_ops", "clean_format"),
        {
            "dataset_id": dataset.id,
            "ops": [{"col": "acct", "op": "strip"}, {"col": "acct", "op": "upper"}],
        },
        task_id="task-1",
    )

    assert cleaned.ok is True
    assert cleaned.output["changed_columns"] == ["acct", "acct"]

    deduped = runner.invoke(
        ToolRef("data_ops", "dedup_rows"),
        {
            "dataset_id": cleaned.output["dataset_id"],
            "keys": ["acct"],
            "strategy": "first",
        },
        task_id="task-1",
    )

    assert deduped.ok is True
    assert deduped.output["removed_rows"] == 1
    deduped_dataset = repo.get_dataset(deduped.output["dataset_id"])
    assert deduped_dataset is not None
    assert deduped_dataset.row_count == 2


def test_dedup_rows_reports_same_key_conflict_without_dropping(tmp_path):
    """spec §6: a same-key value conflict is reported, never silently dropped — only an
    explicit strategy resolves it."""
    runner, registry, repo = _runtime(tmp_path)
    dataset = _register_csv(
        registry,
        tmp_path,
        "conflicts",
        pd.DataFrame({"acct": ["A1", "A1", "B2"], "value": ["1", "2", "9"]}),  # A1 disagrees
        role="feature",
    )

    # No strategy → the conflict is surfaced for review, nothing removed.
    reported = runner.invoke(
        ToolRef("data_ops", "dedup_rows"),
        {"dataset_id": dataset.id, "keys": ["acct"]},
        task_id="task-1",
    )
    assert reported.ok is True, reported.error
    assert reported.output["needs_conflict_review"] is True
    assert reported.output["removed_rows"] == 0
    report = reported.output["conflict_report"]
    assert report["n_conflict_keys"] == 1
    assert report["conflict_columns"] == ["value"]
    assert repo.get_dataset(reported.output["dataset_id"]).row_count == 3  # conflict kept

    # An explicit strategy resolves it deterministically.
    resolved = runner.invoke(
        ToolRef("data_ops", "dedup_rows"),
        {"dataset_id": dataset.id, "keys": ["acct"], "strategy": "first"},
        task_id="task-1",
    )
    assert resolved.ok is True, resolved.error
    assert resolved.output["needs_conflict_review"] is False
    assert resolved.output["removed_rows"] == 1
    assert repo.get_dataset(resolved.output["dataset_id"]).row_count == 2


def test_data_ops_confirm_join_enables_execute(tmp_path):
    runner, registry, repo = _runtime(tmp_path)
    phones = ["13800138000", "13900139000"]
    anchor = _register_csv(registry, tmp_path, "anchor", pd.DataFrame({"mobile": phones}), role="sample")
    feature = _register_csv(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({
            "phone_md5": [hashlib.md5(value.encode()).hexdigest() for value in phones],
            "balance": [10, 20],
        }),
        role="feature",
    )
    propose = runner.invoke(
        ToolRef("data_ops", "propose_join"),
        {"anchor_id": anchor.id, "feature_ids": [feature.id]},
        task_id="task-1",
    )
    plan_id = propose.output["join_plan_id"]

    # execute is hard-blocked until the join is confirmed (join-safety invariant)
    blocked = runner.invoke(ToolRef("data_ops", "execute_join"), {"join_plan_id": plan_id}, task_id="task-1")
    assert blocked.ok is False
    assert "confirmed" in blocked.error

    # confirm_join confirms each feature spec via the engine (unique key => no dedup needed)
    confirm = runner.invoke(ToolRef("data_ops", "confirm_join"), {"join_plan_id": plan_id}, task_id="task-1")
    assert confirm.ok is True
    assert confirm.output["status"] == "confirmed"
    assert feature.id in confirm.output["confirmed"]

    # now execute succeeds and preserves the anchor 1:1
    executed = runner.invoke(ToolRef("data_ops", "execute_join"), {"join_plan_id": plan_id}, task_id="task-1")
    assert executed.ok is True
    assert executed.output["anchor_rows"] == executed.output["joined_rows"] == 2
