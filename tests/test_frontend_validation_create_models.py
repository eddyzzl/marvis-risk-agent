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
    index_html = _read_static('index.html')
    css = _read_static('css/dialogs-governance.css')
    extra_section = index_html.split('id="validationCreateExtraModelsSection"', 1)[1].split('id="createTaskAlgorithmField"', 1)[0]
    assert '每个模型单独上传 Notebook / 样本 / PMML / 数据字典' not in extra_section
    assert 'validation-batch-create-row' not in extra_section
    assert '添加模型' in extra_section
    assert 'id="createTaskPrimaryModelHeading"' in index_html
    assert '.validation-create-model-card' in css
    assert '.validation-create-model-head' in css


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
