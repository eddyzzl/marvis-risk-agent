"""Recovered modeling gate can continue its actual declared full normal workflow."""
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_native_json import decode_native
from test_runtime_domain_archives import _model_binding
from test_runtime_modeling_portfolio_recovery import _run_case
from test_runtime_safety_archives import repin_private


@pytest.fixture(scope="module")
def complete_model_recovery(tmp_path_factory):
    root = tmp_path_factory.mktemp("complete-model-recovery")
    archive, sha, safety = _run_case(root, "modeling", keep_followups=True)
    business = _model_binding(archive, safety.case.model_dump())
    return archive, sha, safety, business, {"expected": root / "suite/private/expected.json"}


def test_actual_model_recovery_continues_selection_reports_delivery_and_independent_math(complete_model_recovery):
    from marvis.orchestrator.eval.runtime_archive_modeling import revalidate_modeling_archive
    archive, sha, safety, business, _ = complete_model_recovery
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=safety)
    assert result["status"] == "verified", result
    math = revalidate_modeling_archive(archive, expected_manifest_sha256=sha, frozen_binding=business)
    assert math["supported_checks_verified"], math


def test_complete_model_recovery_formal_composes_actual_math(complete_model_recovery, tmp_path):
    from test_runtime_stale_recovery import test_formal_stale_recovery_requires_actual_family_math
    test_formal_stale_recovery_requires_actual_family_math(complete_model_recovery, tmp_path)


@pytest.mark.parametrize("damage", ["missing", "target_changed", "revision_changed", "wrong_action", "timeline", "selection", "native_decision_time", "native_input", "native_refit", "native_policy"])
def test_repin_cannot_substitute_model_continuation(complete_model_recovery, tmp_path, damage):
    archive, _, binding, _, _ = complete_model_recovery
    target = tmp_path / "archive"
    shutil.copytree(archive, target)
    private = decode_native(json.loads((target / "private.json").read_bytes()))
    fault = private["runtime_faults"][0]
    if damage == "missing":
        fault.pop("declared_followups")
    elif damage == "target_changed":
        fault["declared_followups"][0]["after"]["target"]["input_sha256"] = "f" * 64
    elif damage == "revision_changed":
        fault["declared_followups"][0]["after"]["target"]["replan_count"] += 1
    elif damage == "wrong_action":
        fault["declared_followups"][0]["action_index"] += 1
    elif damage == "timeline":
        fault["declared_followups"][0]["before"]["started_elapsed_ns"] = 0
    elif damage == "selection":
        private["model_selections"][0]["request"]["adjust_params"]["selected_experiment_id"] = "wrong-experiment"
    else:
        step = private["model_selections"][0]["request"]["expected_step_id"]
        with sqlite3.connect(target / "workspace/marvis.sqlite") as conn:
            # Attack only this copied archive, including its immutable-table
            # guards, then re-pin all bytes to exercise the semantic reader.
            for name, in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='decision_records'").fetchall():
                conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
            if damage == "native_decision_time":
                conn.execute("UPDATE decision_records SET created_at='2000-01-01T00:00:00+00:00' WHERE step_id=?", (step,))
            elif damage == "native_refit":
                raw = json.loads(conn.execute("SELECT inputs_json FROM plan_steps WHERE id=?", (step,)).fetchone()[0])
                raw["refit_on_train_plus_test"] = True
                conn.execute("UPDATE plan_steps SET inputs_json=? WHERE id=?", (json.dumps(raw), step))
            else:
                field = "input_hash" if damage == "native_input" else "policy_hash"
                conn.execute(f"UPDATE decision_records SET {field}=? WHERE step_id=?", ("sha256:" + "f" * 64, step))
    from test_runtime_archive_validation import _rewrite_manifest
    repin_private(target, private)
    result = revalidate_recovery_archive(target, expected_manifest_sha256=_rewrite_manifest(target), frozen_binding=binding)
    assert result["checks"]["archive_integrity"]["status"] == "verified", result
    assert result["status"] in {"mismatch", "unavailable"}, result
