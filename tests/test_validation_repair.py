"""B-line: deterministic diagnose / repair / verify for V2 model validation.

Spec: docs/superpowers/specs/2026-10-16-validation-agent-self-repair-spec.md
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from marvis.agent import validation_repair as repair
from marvis.agent.service import is_agent_material_reselection_intent
from marvis.api_scan_helpers import perform_scan_task
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.settings import build_settings
from tests.validation_material_builders import write_validation_material_bundle


NOTEBOOK = (
    "RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\n"
    "RMC_TIME_COL='apply_month'\n"
    "RMC_PMML_OUTPUT_FIELD='probability_1'\nRMC_MODEL_PARAMS={}\n"
)


def _sample(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0] * 3,
            "x2": [0.0, 1.0, 0.0, 1.0] * 3,
            "y": [0, 1, 0, 1] * 3,
            "split": ["train"] * 4 + ["test"] * 4 + ["oot"] * 4,
            "apply_month": ["202601"] * 4 + ["202602"] * 4 + ["202603"] * 4,
            "apply_date": ["2026-01-05"] * 4 + ["2026-02-05"] * 4 + ["2026-03-05"] * 4,
        }
    )
    for key, value in overrides.items():
        if key == "rename":
            frame = frame.rename(columns=value)
        elif key == "bad_month":
            frame.loc[5, "apply_month"] = value
    return frame


def _task(tmp_path: Path, sample: pd.DataFrame, *, run_mode: str = "agent", tier: str = ""):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    bundle = write_validation_material_bundle(
        settings.workspace / "bundle", notebook_source=NOTEBOOK, sample=sample
    )
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="m",
            model_version="v2",
            validator="pytest",
            source_dir=str(bundle.root),
            notebook_path=bundle.notebook_path.name,
            sample_path=bundle.sample_path.name,
            pmml_path=bundle.pmml_path.name,
            dictionary_path=bundle.dictionary_path.name,
            run_mode=run_mode,
            capability_tier=tier,
        )
    )
    return settings, repo, task


def _requested(**changes) -> dict:
    from tests.validation_builders import make_validation_confirmation

    payload = asdict(replace(make_validation_confirmation(), metadata_sheet=None, **changes))
    payload["transformations"] = []
    return payload


# ---------- probes ----------


def test_match_missing_fields_needs_one_unambiguous_near_identical_column():
    assert repair.match_missing_fields(["x1"], ["X1", "x2"]) == {"x1": "X1"}
    assert repair.match_missing_fields(["apply_amt"], ["Apply-Amt"]) == {"apply_amt": "Apply-Amt"}
    # Two candidates, or none, give no mapping.
    assert repair.match_missing_fields(["x1"], ["X1", "x_1"]) == {"x1": None}
    assert repair.match_missing_fields(["x9"], ["X1"]) == {"x9": None}
    # A column that is itself a model input is never borrowed.
    assert repair.match_missing_fields(["X1"], ["x1"], reserved=["x1"]) == {"X1": None}
    # Two inputs cannot share one sample column.
    assert repair.match_missing_fields(["x1", "X_1"], ["X1"]) == {"x1": None, "X_1": None}


# ---------- field mismatch: auto tier ----------


def test_field_mismatch_is_diagnosed_repaired_verified_and_scores(tmp_path):
    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "X1"}))
    perform_scan_task(repo, task, settings)
    task = repo.get_task(task.id)
    assert task.status == TaskStatus.FAILED

    diagnostic = repair.diagnose_validation_failure(
        settings=settings, task=task, stage="scan", error=task.status_message
    )
    assert diagnostic["code"] == "validation_field_mismatch"
    evidence = {item["label"]: item["value"] for item in diagnostic["evidence"]}
    assert evidence["样本里的近似列"] == "X1 → x1"
    auto = diagnostic["repair_options"][0]
    assert auto["id"] == "map_fields" and auto["tier"] == repair.TIER_AUTO
    assert auto["params"] == {"field_aliases": {"x1": "X1"}}
    assert diagnostic["recovery_actions"][0]["consequence"].startswith("只在验证合同里")

    option = repair.plan_auto_repair(settings, task, diagnostic, [])
    assert option is not None
    report = repair.apply_repair_option(settings, task, option, finding=diagnostic["summary"])
    assert report["verified"] is True
    assert report["undo_command"] == repair.UNDO_COMMAND
    assert "X1 → x1" in report["action"]
    assert repair.load_repair_overrides(settings.tasks_dir, task.id)["field_aliases"] == {"x1": "X1"}

    scan = perform_scan_task(repo, repo.get_task(task.id), settings)
    contract = scan["validation_input_contract"]
    assert contract["status"] == "pending_confirmation"
    copies = [
        item for item in contract["contract"]["transformations"]
        if item["operation"] == "copy" and item["output_field"] == "x1"
    ]
    assert copies == [{"operation": "copy", "output_field": "x1", "input_fields": ["X1"], "params": {}}]

    pytest.importorskip("pypmml")
    from marvis.pipeline import PipelineSettings, run_pmml_scoring_stage
    from marvis.validation.input_confirmation import validate_confirmation_against_materials
    from marvis.validation.input_contracts import transformation_spec_from_dict
    from marvis.validation_materials import resolve_selected_validation_materials

    record = ValidationContractRepository(settings.db_path).get(task.id)
    paths = resolve_selected_validation_materials(repo.get_task(task.id))
    requested = repair.confirmation_from_dict(_requested())
    requested = replace(
        requested,
        transformations=tuple(transformation_spec_from_dict(item) for item in copies),
    )
    validated = validate_confirmation_against_materials(
        contract=record.contract,
        sample_path=paths.sample,
        dictionary_path=paths.dictionary,
        requested=requested,
    )
    ValidationContractRepository(settings.db_path).confirm(
        task.id,
        validated.values,
        expected_revision=record.revision,
        resolved_sample_schema=validated.sample_schema,
        resolved_feature_metadata=validated.feature_metadata,
    )
    run_pmml_scoring_stage(
        task_id=task.id,
        settings=PipelineSettings(
            workspace=settings.workspace,
            db_path=settings.db_path,
            report_template_path=settings.workspace / "unused.docx",
            pmml_scoring_chunk_size=4,
        ),
    )
    assert repo.get_task(task.id).status == TaskStatus.EXECUTED


def test_undo_removes_alias_and_auto_never_reapplies_it(tmp_path):
    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "X1"}))
    perform_scan_task(repo, task, settings)
    task = repo.get_task(task.id)
    diagnostic = repair.diagnose_validation_failure(
        settings=settings, task=task, stage="scan", error=task.status_message
    )
    report = repair.apply_repair_option(
        settings, task, repair.plan_auto_repair(settings, task, diagnostic, [])
    )
    messages = [{"id": "m1", "role": "assistant", "metadata": {"repair_report": report}}]
    assert repair.latest_undoable_report(messages)["repair_id"] == report["repair_id"]

    undo = repair.undo_repair(settings, task, report)
    overrides = repair.load_repair_overrides(settings.tasks_dir, task.id)
    assert overrides["field_aliases"] == {}
    assert overrides["rejected_fingerprints"] == [report["fingerprint"]]
    assert "X1 → x1" in undo["finding"]
    messages.append({"id": "m2", "role": "assistant", "metadata": {"repair_undo": undo}})
    assert repair.latest_undoable_report(messages) is None
    # Rejected by the user: offered as a button, never auto-applied again.
    assert repair.plan_auto_repair(settings, task, diagnostic, []) is None

    enriched = [dict(item) for item in messages]
    repair.enrich_repair_reports(enriched)
    assert enriched[0]["metadata"]["repair_report"]["undone"] is True


def test_unverified_alias_is_rolled_back_and_reported_honestly(tmp_path):
    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "X1"}))
    perform_scan_task(repo, task, settings)
    task = repo.get_task(task.id)
    option = repair._option(
        "map_fields",
        tier=repair.TIER_AUTO,
        label="l",
        command="c",
        consequence="x",
        params={"field_aliases": {"x1": "missing_column"}},
        reversible=True,
    )
    report = repair.apply_repair_option(settings, task, option)
    assert report["verified"] is False
    assert report["undo_command"] == ""
    assert report["verification"].startswith("核对没通过")
    assert "已还原" in report["action"]
    assert repair.load_repair_overrides(settings.tasks_dir, task.id)["field_aliases"] == {}
    assert "处理没成功，已还原" in repair.repair_report_content(report)


def test_auto_repair_budget_follows_capability_tier_and_skips_manual(tmp_path):
    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "X1"}), tier="conservative")
    perform_scan_task(repo, task, settings)
    task = repo.get_task(task.id)
    diagnostic = repair.diagnose_validation_failure(
        settings=settings, task=task, stage="scan", error=task.status_message
    )
    used = [
        {"metadata": {"repair_report": {"tier": "auto", "fingerprint": f"f{index}"}}}
        for index in range(2)
    ]
    assert repair.repair_budget(settings, used, task) == (2, 2)
    assert repair.plan_auto_repair(settings, task, diagnostic, used) is None
    assert repair.budget_exhausted_note(settings, used, task)["value"].endswith("这次需要你点一下")
    assert repair.plan_auto_repair(settings, task, diagnostic, used[:1]) is not None
    manual = replace(task, run_mode="manual")
    assert repair.plan_auto_repair(settings, manual, diagnostic, []) is None


def test_agent_scan_stage_repairs_then_rescans_into_field_confirmation(tmp_path, monkeypatch):
    from marvis.agent.validation_app_service import run_agent_scan_stage

    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "X1"}))
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.summarize_stage",
        lambda **_kwargs: ("材料扫描完成。", {"fallback": True}),
    )
    assert run_agent_scan_stage(repo, settings, task.id, {"model_id": "m1"}) is True
    messages = repo.list_agent_messages(task.id)
    reports = [m for m in messages if (m.get("metadata") or {}).get("repair_report")]
    assert len(reports) == 1 and reports[0]["metadata"]["repair_report"]["verified"] is True
    assert reports[0]["content"].startswith("**遇到问题，已处理好**")
    # No failure card: the repair report carries the finding.
    assert not any((m.get("metadata") or {}).get("error_diagnostic") for m in messages)
    assert repo.get_task(task.id).status == TaskStatus.SCANNED
    assert messages[-1]["stage"] == "input_confirmation"


# ---------- date parse: one-click ----------


def test_date_parse_offers_an_alternative_time_column_and_confirms_it(tmp_path):
    settings, repo, task = _task(tmp_path, _sample(bad_month="bad"))
    perform_scan_task(repo, task, settings)
    record = ValidationContractRepository(settings.db_path).get(task.id)
    error = "time column 'apply_month' contains unparseable date values: 'bad'"
    diagnostic = repair.diagnose_validation_failure(
        settings=settings,
        task=repo.get_task(task.id),
        stage="input_confirmation",
        error=error,
        requested_confirmation=_requested(),
    )
    assert diagnostic["code"] == "validation_time_parse_failed"
    evidence = {item["label"]: item["value"] for item in diagnostic["evidence"]}
    assert evidence["无法解析"] == "1 / 12 行"
    assert evidence["样例"] == "'bad'"
    assert evidence["能完整解析的其他日期列"] == "apply_date"
    [option] = diagnostic["repair_options"]
    assert option["tier"] == repair.TIER_ONE_CLICK
    assert option["command"] == "改用 apply_date 作为时间列并确认"
    assert "OOT 月份" in option["consequence"]
    assert option["params"]["revision"] == record.revision

    report = repair.apply_repair_option(settings, repo.get_task(task.id), option)
    assert report["verified"] is True
    confirmed = ValidationContractRepository(settings.db_path).get(task.id)
    assert confirmed.status == "ready"
    assert confirmed.contract.confirmed["time_col"] == "apply_date"
    # The contract moved on: the same option is now stale.
    with pytest.raises(repair.RepairStale):
        repair.apply_repair_option(settings, repo.get_task(task.id), option)


def test_confirmation_failure_posts_a_card_in_agent_mode_only(tmp_path):
    from fastapi.testclient import TestClient

    from marvis.app import create_app

    for run_mode in ("agent", "manual"):
        root = tmp_path / run_mode
        settings, repo, task = _task(root, _sample(bad_month="bad"), run_mode=run_mode)
        perform_scan_task(repo, task, settings)
        record = ValidationContractRepository(settings.db_path).get(task.id)
        client = TestClient(create_app(settings))
        body = {**_requested(), "revision": record.revision}
        body.pop("transformations")
        body["transformations"] = []
        response = client.put(f"/api/tasks/{task.id}/validation-input-contract", json=body)
        assert response.status_code == 422
        cards = [
            m for m in repo.list_agent_messages(task.id)
            if (m.get("metadata") or {}).get("error_diagnostic")
        ]
        assert len(cards) == (1 if run_mode == "agent" else 0)
        if run_mode == "agent":
            # Submitting the same failing confirmation again does not stack.
            client.put(f"/api/tasks/{task.id}/validation-input-contract", json=body)
            assert len([
                m for m in repo.list_agent_messages(task.id)
                if (m.get("metadata") or {}).get("error_diagnostic")
            ]) == 1


# ---------- material / dependency ----------


def test_material_failure_offers_reselection_with_inventory(tmp_path):
    settings, repo, task = _task(tmp_path, _sample())
    diagnostic = repair.diagnose_validation_failure(
        settings=settings,
        task=task,
        stage="scan",
        error="ValueError: selected PMML file does not exist",
    )
    assert diagnostic["code"] == "validation_material_unrecognized"
    facts = {item["label"]: item["value"] for item in diagnostic["evidence"]}
    assert facts["目录里的PMML"] == "model.pmml"
    [option] = diagnostic["repair_options"]
    assert option["id"] == "reselect_materials"
    assert is_agent_material_reselection_intent(option["command"])


@pytest.mark.parametrize(
    "error",
    [
        "FileNotFoundError: [Errno 2] No such file or directory: 'java'",
        "ModuleNotFoundError: No module named 'pypmml'",
    ],
)
def test_missing_dependency_is_explained_never_installed(tmp_path, error):
    settings, _repo, task = _task(tmp_path, _sample())
    diagnostic = repair.diagnose_validation_failure(
        settings=settings, task=task, stage="reproducibility", error=error
    )
    assert diagnostic["code"] == "validation_dependency_missing"
    tiers = {option["id"]: option["tier"] for option in diagnostic["repair_options"]}
    assert tiers == {"install_dependency": "confirm", "rerun_pmml_scoring": "one_click"}
    # The install option has no command: the platform does not execute it.
    assert [action["repair_option_id"] for action in diagnostic["recovery_actions"]] == [
        "rerun_pmml_scoring"
    ]
    assert any("平台不会替你安装" in action for action in diagnostic["actions"])


def test_unclassified_and_v1_failures_keep_the_llm_explanation(tmp_path):
    settings, _repo, task = _task(tmp_path, _sample())
    assert repair.diagnose_validation_failure(
        settings=settings, task=task, stage="metrics", error="RuntimeError: boom"
    ) is None
    legacy = replace(task, validation_workflow_version=1)
    assert repair.diagnose_validation_failure(
        settings=settings,
        task=legacy,
        stage="scan",
        error="sample missing required PMML inputs: x1",
    ) is None


# ---------- commands ----------


def test_repair_commands_match_only_the_offered_words_until_superseded():
    option = {"id": "switch_time_column", "command": "改用 apply_date 作为时间列并确认"}
    failure = {
        "id": "f1",
        "role": "assistant",
        "stage": "failure",
        "metadata": {
            "error": True,
            "error_diagnostic": {"workflow": "validation", "repair_options": [option]},
        },
    }
    messages = [failure]
    kind, matched, source = repair.match_repair_command(messages, "改用 apply_date 作为时间列并确认")
    assert (kind, matched["id"], source["id"]) == ("apply", "switch_time_column", "f1")
    assert repair.match_repair_command(messages, "改用 apply_date 作为时间列吧") is None
    assert repair.match_repair_command(messages, "") is None
    # A later stage message supersedes the card.
    later = [*messages, {"id": "s1", "role": "assistant", "stage": "reproducibility", "metadata": {}}]
    assert repair.match_repair_command(later, "改用 apply_date 作为时间列并确认") == ("stale", None)
    # The reselection command belongs to the existing material branch.
    reselect = {"id": "reselect_materials", "command": repair.RESELECT_MATERIALS_COMMAND}
    failure["metadata"]["error_diagnostic"]["repair_options"] = [reselect]
    assert repair.match_repair_command(messages, repair.RESELECT_MATERIALS_COMMAND) is None
    # Undo targets the newest report that can still be undone.
    reports = [{"id": "r1", "role": "assistant", "metadata": {"repair_report": {
        "repair_id": "a", "undo_command": repair.UNDO_COMMAND}}}]
    kind, report = repair.match_repair_command(reports, repair.UNDO_COMMAND)
    assert (kind, report["repair_id"]) == ("undo", "a")


# ---------- router ----------


def test_router_undo_and_one_click_dispatch_the_right_stage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from marvis.agent.validation_app_service import run_agent_scan_stage
    from marvis.routers import validation_agent as router

    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "X1"}))
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.summarize_stage",
        lambda **_kwargs: ("材料扫描完成。", {"fallback": True}),
    )
    run_agent_scan_stage(repo, settings, task.id, {"model_id": "m1"})
    dispatched = []
    monkeypatch.setattr(router, "resolve_agent_model", lambda *_args: {"model_id": "m1"})
    monkeypatch.setattr(router, "capture_user_preference_memory", lambda *_args: None)
    monkeypatch.setattr(
        router,
        "dispatch_agent_validation_job",
        lambda **kwargs: dispatched.append(kwargs) or {"status": "dispatched"},
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    payload = SimpleNamespace(model_id=None, effort=None, acceptance_mode=None)

    def send(text):
        return router._handle_validation_repair_command(
            request, repo, repo.get_task(task.id), content=text,
            payload=payload, background_tasks=None,
        )

    assert send("随便聊聊") is None
    assert send(repair.UNDO_COMMAND) == {"status": "dispatched"}
    assert dispatched[-1]["forced_stage"] == "scan"
    messages = repo.list_agent_messages(task.id)
    assert messages[-1]["metadata"]["repair_undo"]["repair_id"]
    assert repair.load_repair_overrides(settings.tasks_dir, task.id)["field_aliases"] == {}
    # Nothing left to undo: a deterministic reply, never the chat fallthrough.
    assert send(repair.UNDO_COMMAND)["status"] == "message_saved"
    assert repo.list_agent_messages(task.id)[-1]["content"] == repair.UNDO_UNAVAILABLE_REPLY

    # The rescan fails again; this time the auto option is rejected, so the
    # failure card offers it as a button and the click repairs + rescans.
    run_agent_scan_stage(repo, settings, task.id, {"model_id": "m1"})
    card = repo.list_agent_messages(task.id)[-1]
    commands = [a["command"] for a in card["metadata"]["error_diagnostic"]["recovery_actions"]]
    assert repair.MAP_FIELDS_COMMAND in commands
    assert send(repair.MAP_FIELDS_COMMAND) == {"status": "dispatched"}
    assert dispatched[-1]["forced_stage"] == "scan"
    assert repo.list_agent_messages(task.id)[-1]["metadata"]["repair_report"]["verified"] is True


def test_router_rejects_stale_commands_and_stale_contract_revisions(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from marvis.routers import validation_agent as router

    settings, repo, task = _task(tmp_path, _sample(bad_month="bad"))
    perform_scan_task(repo, task, settings)
    record = ValidationContractRepository(settings.db_path).get(task.id)
    diagnostic = repair.diagnose_validation_failure(
        settings=settings, task=repo.get_task(task.id), stage="input_confirmation",
        error="time column 'apply_month' contains unparseable date values: 'bad'",
        requested_confirmation=_requested(),
    )
    repo.add_agent_message(task.id, role="assistant", stage="failure", content="x",
                           metadata={"error": True, "error_diagnostic": diagnostic})
    dispatched = []
    monkeypatch.setattr(router, "resolve_agent_model", lambda *_args: {"model_id": "m1"})
    monkeypatch.setattr(router, "capture_user_preference_memory", lambda *_args: None)
    monkeypatch.setattr(router, "dispatch_agent_validation_job",
                        lambda **kwargs: dispatched.append(kwargs) or {"status": "dispatched"})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    payload = SimpleNamespace(model_id=None, effort=None, acceptance_mode=None)
    command = diagnostic["repair_options"][0]["command"]

    def send(text):
        return router._handle_validation_repair_command(
            request, repo, repo.get_task(task.id), content=text,
            payload=payload, background_tasks=None,
        )

    # The contract moved on behind the card: nothing is executed.
    ValidationContractRepository(settings.db_path).replace_candidates(task.id, record.contract)
    assert send(command)["status"] == "message_saved"
    assert "没有做任何改动" in repo.list_agent_messages(task.id)[-1]["content"]
    assert dispatched == []

    # A later stage supersedes the card: its command answers "expired".
    repo.add_agent_message(task.id, role="assistant", stage="reproducibility", content="打分完成")
    assert send(command)["status"] == "message_saved"
    assert repo.list_agent_messages(task.id)[-1]["content"] == repair.STALE_COMMAND_REPLY
    assert dispatched == []


def test_undo_is_unavailable_once_a_later_stage_started():
    report = {"repair_id": "a", "undo_command": repair.UNDO_COMMAND, "verified": True}
    messages = [{"id": "r", "role": "assistant", "stage": "scan", "metadata": {"repair_report": report}}]
    assert repair.latest_undoable_report(messages)["repair_id"] == "a"
    messages.append({"id": "c", "role": "assistant", "stage": "input_confirmation", "metadata": {}})
    assert repair.latest_undoable_report(messages)["repair_id"] == "a"
    messages.append({"id": "p", "role": "assistant", "stage": "reproducibility", "metadata": {}})
    assert repair.latest_undoable_report(messages) is None
    assert repair.match_repair_command(messages, repair.UNDO_COMMAND) == ("stale", "undo")


def test_aliases_apply_only_to_the_materials_they_were_made_for(tmp_path):
    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "X1"}))
    perform_scan_task(repo, task, settings)
    task = repo.get_task(task.id)
    diagnostic = repair.diagnose_validation_failure(
        settings=settings, task=task, stage="scan", error=task.status_message
    )
    report = repair.apply_repair_option(
        settings, task, repair.plan_auto_repair(settings, task, diagnostic, []), trigger="auto"
    )
    assert report["verified"] and report["trigger"] == "auto"
    hashes = repair._material_hashes(task)
    assert repair.active_field_aliases(
        settings.tasks_dir, task.id, sample_sha256=hashes["sample"], pmml_sha256=hashes["pmml"]
    ) == {"x1": "X1"}
    # The user replaces the sample: the old mapping no longer applies.
    sample_path = Path(task.source_dir) / "sample.parquet"
    _sample(rename={"x1": "X1"}).assign(extra=1).to_parquet(sample_path, index=False)
    scan = perform_scan_task(repo, task, settings)
    assert scan["validation_input_contract"]["status"] == "blocked"


def test_label_column_is_never_borrowed_as_a_model_input(tmp_path):
    # PMML needs x1; the sample has no x1 but its label is called X1-like.
    settings, repo, task = _task(tmp_path, _sample(rename={"x1": "dropped", "y": "x_1"}))
    from tests.validation_material_builders import write_validation_material_bundle as _write

    _write(Path(task.source_dir), notebook_source=NOTEBOOK.replace("'y'", "'x_1'"),
           sample=_sample(rename={"x1": "dropped", "y": "x_1"}))
    perform_scan_task(repo, task, settings)
    task = repo.get_task(task.id)
    diagnostic = repair.diagnose_validation_failure(
        settings=settings, task=task, stage="scan", error=task.status_message
    )
    assert diagnostic is not None
    assert all(option["id"] != "map_fields" for option in diagnostic["repair_options"])


def test_user_clicked_repairs_do_not_use_the_automatic_budget(tmp_path):
    settings, _repo, task = _task(tmp_path, _sample(), tier="conservative")
    clicked = [{"metadata": {"repair_report": {"tier": "auto", "trigger": "user"}}}] * 3
    assert repair.repair_budget(settings, clicked, task) == (0, 2)
    legacy = [{"metadata": {"repair_report": {"tier": "auto"}}}]
    assert repair.repair_budget(settings, legacy, task) == (1, 2)


def test_numeric_features_with_time_like_names_are_not_probed(tmp_path):
    sample = _sample(bad_month="bad").assign(
        max_overdue_days=range(12), loan_months=range(12)
    )
    settings, repo, task = _task(tmp_path, sample)
    perform_scan_task(repo, task, settings)
    record = ValidationContractRepository(settings.db_path).get(task.id)
    schema = record.contract.sample_schema
    assert not repair._could_hold_dates("max_overdue_days", schema)
    assert not repair._could_hold_dates("loan_months", schema)
    assert repair._could_hold_dates("apply_date", schema)
    assert repair.alternative_time_columns(
        repo.get_task(task.id), schema.columns, exclude="apply_month", schema=schema
    ) == ["apply_date"]


def test_exception_path_classifies_cheaply_and_falls_back(tmp_path, monkeypatch):
    from marvis.agent import validation_app_service as service

    settings, repo, task = _task(tmp_path, _sample())
    service.add_agent_job_exception_summary(
        repo, task_id=task.id, task=task, model_profile={"model_id": "m1"},
        error=FileNotFoundError(2, "No such file or directory", "java"),
    )
    card = repo.list_agent_messages(task.id)[-1]
    assert card["metadata"]["error_diagnostic"]["code"] == "validation_dependency_missing"
    # A field-mismatch text is not probed on this path; the LLM text is used.
    calls = []
    monkeypatch.setattr(service, "failure_summary", lambda **kwargs: calls.append(kwargs) or ("解释", {}))
    service.add_agent_job_exception_summary(
        repo, task_id=task.id, task=task, model_profile={"model_id": "m1"},
        error=ValueError("sample missing required PMML inputs: x1"),
    )
    assert calls and "error_diagnostic" not in (repo.list_agent_messages(task.id)[-1]["metadata"] or {})


def test_batch_children_get_the_confirmation_card_and_auto_repair(tmp_path, monkeypatch):
    # 2026-10-23 extension spec §5.4: the batch workbench sends the selected
    # child's commands to that child, so children repair like single models.
    from types import SimpleNamespace

    from marvis.routers import validation_contracts as contracts_router

    settings, repo, task = _task(tmp_path, _sample(bad_month="bad"))
    perform_scan_task(repo, task, settings)
    monkeypatch.setattr("marvis.api_task_helpers.validation_batch_parent_id", lambda *_args: "parent")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    contracts_router._add_confirmation_failure_card(
        request, repo, repo.get_task(task.id),
        error="time column 'apply_month' contains unparseable date values: 'bad'",
        requested=repair.confirmation_from_dict(_requested()),
    )
    assert any((m.get("metadata") or {}).get("error_diagnostic") for m in repo.list_agent_messages(task.id))

    child_settings, child_repo, child = _task(tmp_path / "child", _sample(rename={"x1": "X1"}))
    perform_scan_task(child_repo, child, child_settings)
    child = child_repo.get_task(child.id)
    diagnostic = repair.diagnose_validation_failure(
        settings=child_settings, task=child, stage="scan", error=child.status_message
    )
    assert repair.plan_auto_repair(child_settings, child, diagnostic, [])["id"] == "map_fields"


def test_rerun_pmml_phrase_keeps_its_own_handler_after_a_dependency_card():
    from marvis.agent.service import agent_rerun_stage

    card = {"id": "d", "role": "assistant", "stage": "failure", "metadata": {"error": True, "error_diagnostic": {
        "workflow": "validation",
        "repair_options": [{"id": "rerun_pmml_scoring", "command": repair.RERUN_PMML_COMMAND}],
    }}}
    later = [card, {"id": "p", "role": "assistant", "stage": "metrics", "metadata": {}}]
    assert repair.match_repair_command(later, repair.RERUN_PMML_COMMAND) is None
    assert agent_rerun_stage(repair.RERUN_PMML_COMMAND) == "reproducibility"


def test_time_profile_handles_nullable_dtypes_and_one_read(tmp_path, monkeypatch):
    sample = _sample(bad_month="bad")
    sample["apply_month"] = sample["apply_month"].astype("string")
    sample.loc[7, "apply_month"] = pd.NA
    settings, repo, task = _task(tmp_path, sample)
    perform_scan_task(repo, task, settings)
    schema = ValidationContractRepository(settings.db_path).get(task.id).contract.sample_schema
    reads = []
    original = repair._read_columns
    monkeypatch.setattr(repair, "_read_columns", lambda *a, **k: reads.append(a[1]) or original(*a, **k))
    profile, alternatives = repair.date_parse_probes(
        repo.get_task(task.id), "apply_month", schema.columns, schema=schema
    )
    assert len(reads) == 1
    assert profile["bad_rows"] == 2 and profile["truncated"] is False
    assert profile["examples"] == ["'bad'", "空值"] or set(profile["examples"]) == {"'bad'", "空值"}
    assert alternatives == ["apply_date"]


def test_job_exception_material_card_needs_a_missing_material():
    task = type("T", (), {"validation_workflow_version": 2, "id": "t", "source_dir": "/nonexistent"})()
    settings = type("S", (), {"db_path": Path("/nonexistent.sqlite")})()
    contract_error = "ValueError: feature metadata: confirmed selection is not an atomic scanned candidate"
    assert repair.diagnose_validation_failure(
        settings=settings, task=task, stage="exception", error=contract_error,
        kinds=frozenset({"dependency", "material"}),
    ) is None
    missing = repair.diagnose_validation_failure(
        settings=settings, task=task, stage="exception",
        error="ValueError: selected PMML file does not exist",
        kinds=frozenset({"dependency", "material"}),
    )
    assert missing["code"] == "validation_material_unrecognized"
