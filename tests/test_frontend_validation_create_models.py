import json
import subprocess
from pathlib import Path

import pytest

from marvis.validation_report_copy import narrative_report_values


STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _run_create_dialog_module(script_body: str) -> None:
    module_url = (STATIC_DIR / "js" / "create-task-dialog.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
const createDialog = await import({json.dumps(module_url)});
{script_body}
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout or "node test failed")


def test_extra_validation_models_reuse_the_primary_form_layout():
    index_html = _read_static("index.html")
    create_js = _read_static("js/create-task-dialog.js")
    css = _read_static("css/dialogs-governance.css")

    extra_section = index_html.split('id="validationCreateExtraModelsSection"', 1)[1].split(
        'id="createTaskAlgorithmField"',
        1,
    )[0]
    assert "每个模型单独上传 Notebook / 样本 / PMML / 数据字典" not in extra_section
    assert "validation-batch-create-row" not in extra_section
    assert "添加模型" in extra_section

    assert 'id="createTaskPrimaryModelHeading"' in index_html

    assert "export function validationExtraModelCardMarkup" in create_js
    assert "validation-batch-create-row" not in create_js
    assert "data-extra-file-role" not in create_js
    assert "添加多个模型时请用文件上传" not in create_js
    assert ".validation-create-model-card" in css
    assert ".validation-create-model-head" in css


def test_validation_extra_model_card_markup_matches_primary_fields():
    _run_create_dialog_module(
        r"""
const html = createDialog.validationExtraModelCardMarkup("validation-extra-1", 2);
assert.match(html, /class="task-form-section validation-create-model-card"/);
assert.match(html, /data-validation-extra-row-id="validation-extra-1"/);
assert.match(html, />模型 2</);
assert.match(html, />模型名称</);
assert.match(html, /data-extra-model-field="name"/);
assert.match(html, />模型概述</);
assert.match(html, /data-extra-model-field="overview"/);
assert.match(html, />适用范围</);
assert.match(html, /data-extra-model-field="scope"/);
assert.match(html, />坏样本定义</);
assert.match(html, /data-extra-model-field="bad-sample"/);
assert.match(html, />好样本定义</);
assert.match(html, /data-extra-model-field="good-sample"/);
assert.match(html, />文件路径</);
assert.match(html, />文件上传</);
assert.match(html, />材料目录</);
assert.match(html, /data-extra-model-field="source-dir"/);
assert.match(html, /点击或拖拽上传/);
assert.match(html, />移除</);
assert.doesNotMatch(html, /data-extra-file-role/);
assert.doesNotMatch(html, /validation-batch-create-row/);
assert.doesNotMatch(html, /四份材料分别上传/);
assert.doesNotMatch(html, />验证人员</);
"""
    )


def test_create_agent_batch_opens_workbench_instead_of_parent_messages():
    app_js = _read_static("app.js")
    start = app_js.index("async function createTaskAndScan")
    end = app_js.index("async function pollValidationProgress", start)
    body = app_js[start:end]
    assert "if (usesAgentValidationWorkbench(task))" in body
    assert "await requestTaskSelection(task);" in body
    workbench_branch = body.split("if (usesAgentValidationWorkbench(task))", 1)[1].split(
        'if (task.run_mode === "agent")',
        1,
    )[0]
    assert "loadAgentMessages" not in workbench_branch
    assert "await continueAgentValidationBatch();" in workbench_branch
    assert "正在按自动审查逐个执行模型验证" in workbench_branch
    assert "async function continueAgentValidationBatch" in app_js
    assert "dispatchAgentValidation(childId)" in app_js
    assert 'agentAcceptanceMode = "auto_accept"' in app_js


def test_create_multi_model_returns_parent_task_identity():
    create_js = _read_static("js/create-task-dialog.js")
    assert "created?.parent_task" in create_js
    assert "created?.batch?.parent_task_id" in create_js
    helper = create_js.split("async function createMultiModelValidationTask", 1)[1].split(
        "function bindMaterialSourceControls",
        1,
    )[0]
    assert "parent_task_id" in helper
    assert "task_type: \"validation_batch\"" in helper or "task_type: 'validation_batch'" in helper
    assert "rows.length < 2" in helper


def test_single_model_validation_create_uses_ordinary_task_not_batch():
    create_js = _read_static("js/create-task-dialog.js")
    body = create_js.split("async function createTask()", 1)[1].split(
        "function resetValidationExtraModels",
        1,
    )[0]
    assert "if (extraRows.length > 0)" in body
    extra_branch = body.split("if (extraRows.length > 0)", 1)[1].split(
        "if (materialSourceController.mode()",
        1,
    )[0]
    assert "createMultiModelValidationTask" in extra_branch
    assert 'api("api/tasks"' not in extra_branch
    assert 'return await api("api/tasks"' in body
    assert body.index("if (extraRows.length > 0)") < body.index('return await api("api/tasks"')


@pytest.mark.parametrize("model_name", [
    "", "本模型", "渠道甲T卡MOB6", "渠道乙A卡MOB3模型", "自营T卡MOB3",
    "自营通用a卡mob6", "渠道丙模型", "A卡T卡MOB6",
])
def test_create_narrative_defaults_match_backend(model_name):
    expected = narrative_report_values(model_name)
    _run_create_dialog_module(f"""
assert.equal(typeof createDialog.validationNarrativeDefaults, "function");
assert.deepEqual(
  createDialog.validationNarrativeDefaults({json.dumps(model_name)}),
  {json.dumps(expected, ensure_ascii=False)}
);
""")


def test_changing_model_name_refreshes_only_untouched_report_defaults():
    _run_create_dialog_module(r"""
assert.equal(typeof createDialog.updateAutoReportValue, "function");
const field = {value: "", dataset: {}};
createDialog.updateAutoReportValue(field, "本模型默认概述");
assert.equal(field.value, "本模型默认概述");
createDialog.updateAutoReportValue(field, "T卡支用概述");
assert.equal(field.value, "T卡支用概述");
field.value = "用户编辑的概述";
createDialog.updateAutoReportValue(field, "A卡授信概述");
assert.equal(field.value, "用户编辑的概述");
field.value = "";
createDialog.updateAutoReportValue(field, "重新打开后的默认概述");
assert.equal(field.value, "重新打开后的默认概述");
""")
