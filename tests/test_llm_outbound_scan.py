from copy import deepcopy
import json
from pathlib import Path

import pytest

from marvis.agent.service import summarize_stage
from marvis.api_scan_helpers import perform_scan_task
from marvis.llm_outbound_scan import scan_context
from tests.test_api_scan_helpers import _as_historical_validation_task
from tests.test_llm_outbound import Wire
from tests.validation_material_builders import create_repository_validation_task

PRIVATE = "PRIVATE_SCAN_SOURCE_973"
SOURCE = (
    "import pandas as pd\nRMC_SAMPLE_DF=pd.DataFrame({'y':[0]})\n"
    "RMC_TARGET_COL='y'\nRMC_ALGORITHM='lr'\nRMC_SCORE_FN=lambda frame: [0.5]*len(frame)\n"
    "RMC_SPLIT_COL='split'\nRMC_TIME_COL='apply_month'\nRMC_PMML_OUTPUT_FIELD='probability_1'\n"
    f"# {PRIVATE}\n"
)


@pytest.mark.parametrize("version", [1, 2])
def test_native_scan_reaches_wire_with_correct_version_contract_and_no_source(tmp_path, monkeypatch, version):
    task, repo, settings = create_repository_validation_task(tmp_path, notebook_source=SOURCE)
    if version == 1:
        task = _as_historical_validation_task(repo, task)
    native = perform_scan_task(repo, task, settings)
    assert native["status"] == "scanned"
    before = deepcopy(native)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "材料检查状态已读取。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    deltas = []
    _, meta = summarize_stage(task=task, stage="scan", evidence={"scan": native},
                              memory_context={"summary": PRIVATE}, model_profile={}, fallback="本地回退",
                              on_delta=deltas.append)
    assert meta == {"fallback": False, "history_comparison_available": False}
    assert deltas == [wire.reply] and native == before
    assert len(wire.requests) == 1 and wire.requests[0]["stream"] is False
    body = json.dumps(wire.requests)
    assert PRIVATE not in body and str(tmp_path) not in body
    assert "model.ipynb" not in body and "source_previews" not in body
    context = wire.last_context()["evidence"]
    assert context["scan_interpretation"]["required_materials_complete"] is True
    assert context["scan_interpretation"]["missing_or_unsatisfied_checks"] == []
    expected_contract = "validation_input_contract" if version == 2 else "notebook_contract"
    assert context["scan_interpretation"]["required_checks"][-1] == expected_contract
    assert [item["status"] for item in context["scan"]["checks"]] == [item["status"] for item in native["checks"]]
    if version == 2:
        assert context["scan"]["validation_input_contract"] == {
            "status": "pending_confirmation", "needs_confirmation": True, "read_only": True, "revision": 1,
        }
    else:
        assert context["scan"]["notebook_contract"]["target_col"] == "y"
        assert context["scan"]["notebook_contract"]["sample_df_defined"] is True


def test_native_v2_missing_importance_stays_failed_in_scan_context(tmp_path):
    import pandas as pd

    task, repo, settings = create_repository_validation_task(tmp_path, notebook_source=SOURCE)
    path = Path(task.source_dir) / task.dictionary_path
    pd.read_csv(path).drop(columns=["importance"]).to_csv(path, index=False)
    native = perform_scan_task(repo, task, settings)
    assert native["status"] == "failed"
    context = scan_context({"scan": native}, workflow_version=2)
    assert context["scan"]["status"] == "failed"
    assert context["scan_interpretation"]["required_materials_complete"] is False
    assert "validation_input_contract" in context["scan_interpretation"]["missing_or_unsatisfied_checks"]
    assert context["scan_interpretation"]["failed_check_count"] >= 1


def test_v2_scan_guard_precedes_ui_emission_and_never_requires_v1_contract(tmp_path, monkeypatch):
    task, repo, settings = create_repository_validation_task(tmp_path, notebook_source=SOURCE)
    native = perform_scan_task(repo, task, settings)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "缺少 Notebook RMC 契约、PMML 或 pickle。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    deltas = []
    content, meta = summarize_stage(task=task, stage="scan", evidence=native, model_profile={},
                                    fallback="本地回退", on_delta=deltas.append)
    assert meta["guarded_scan_summary"] is True
    assert "当前未发现必需材料缺失" in content
    assert "Notebook RMC 契约" not in content
    assert deltas == [content]


def test_failure_status_and_tail_checks_cannot_be_overridden_by_messages_or_labels():
    checks = [{"id": name, "status": "success"} for name in (
        "material_notebook", "material_sample", "material_model_pmml", "material_data_dictionary", "notebook_contract",
    )]
    checks.extend({"id": f"private-extension-{i}", "label": PRIVATE, "status": "success"} for i in range(210))
    checks.append({"id": "notebook_contract", "status": "error", "message": "已定义 " + PRIVATE})
    context = scan_context({"checks": checks}, workflow_version=1)
    assert len(context["scan"]["checks"]) == 216
    assert context["scan_interpretation"]["failed_check_count"] == 1
    assert context["scan_interpretation"]["required_materials_complete"] is False
    assert PRIVATE not in json.dumps(context)
    impersonation = scan_context({"checks": [{"id": PRIVATE, "label": "Notebook 文件", "status": "success"}]}, workflow_version=1)
    assert impersonation["scan_interpretation"]["satisfied_checks"] == []
