import json
import subprocess
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _run_create_module(script_body: str) -> None:
    module_url = (STATIC_DIR / "js" / "validation-batch-create.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
const batchCreate = await import({json.dumps(module_url)});
{script_body}
"""
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_batch_create_draft_validates_row_count_required_fields_and_file_types():
    _run_create_module(
        r"""
const file = (name) => ({ name, size: 12, lastModified: 7 });
const completeRow = (id = "row-1") => ({
  id,
  modelName: "贷前评分卡",
  modelVersion: "v1",
  files: {
    notebook: file("validation.ipynb"),
    sample: file("sample.parquet"),
    pmml: file("model.pmml"),
    dictionary: file("dictionary.xlsx"),
  },
});

const empty = batchCreate.validateValidationBatchDraft({
  batchName: "",
  validator: "",
  rows: [],
});
assert.equal(empty.valid, false);
assert.match(empty.formErrors.join(" "), /批次名称/);
assert.match(empty.formErrors.join(" "), /验证人员/);
assert.match(empty.formErrors.join(" "), /2 至 10/);

const tooMany = batchCreate.validateValidationBatchDraft({
  batchName: "批次 A",
  validator: "Eddy",
  rows: Array.from({ length: 11 }, (_, index) => completeRow(`row-${index}`)),
});
assert.equal(tooMany.valid, false);
assert.match(tooMany.formErrors.join(" "), /2 至 10/);

const oneModel = batchCreate.validateValidationBatchDraft({
  batchName: "批次 A",
  validator: "Eddy",
  rows: [completeRow("row-only")],
});
assert.equal(oneModel.valid, false);
assert.match(oneModel.formErrors.join(" "), /2 至 10/);

const invalid = batchCreate.validateValidationBatchDraft({
  batchName: "批次 A",
  validator: "Eddy",
  rows: [{
    id: "row-bad",
    modelName: "",
    modelVersion: "",
    files: {
      notebook: file("validation.py"),
      sample: file("sample.txt"),
      pmml: file("model.xml"),
      dictionary: file("dictionary.feather"),
    },
  }],
});
assert.equal(invalid.valid, false);
assert.match(invalid.rowErrors["row-bad"].join(" "), /模型名称/);
assert.match(invalid.rowErrors["row-bad"].join(" "), /\.ipynb/);
assert.match(invalid.rowErrors["row-bad"].join(" "), /样本/);
assert.match(invalid.rowErrors["row-bad"].join(" "), /\.pmml/);
assert.match(invalid.rowErrors["row-bad"].join(" "), /数据字典/);

const valid = batchCreate.validateValidationBatchDraft({
  batchName: "批次 A",
  validator: "Eddy",
  rows: [completeRow("row-1"), completeRow("row-2")],
});
assert.deepEqual(valid, { valid: true, formErrors: [], rowErrors: {} });
"""
    )


def test_batch_create_uses_role_subdirectories_and_one_isolated_upload_per_model():
    _run_create_module(
        r"""
const file = (name, marker) => ({ name, size: marker, lastModified: marker });
const makeRow = (id, modelName, marker) => ({
  id,
  modelName,
  modelVersion: `v${marker}`,
  files: {
    notebook: file("shared.ipynb", marker),
    sample: file("shared.csv", marker),
    pmml: file("shared.pmml", marker),
    dictionary: file("shared.xlsx", marker),
  },
});
const rows = [makeRow("row-1", "模型 A", 1), makeRow("row-2", "模型 B", 2)];
const uploadCalls = [];
let createPayload = null;
const statuses = [];

const result = await batchCreate.submitValidationBatchDraft(
  { batchName: "季度批次", validator: "Eddy", rows },
  {
    uploadMaterials: async ({ row, relativePaths }) => {
      uploadCalls.push({ rowId: row.id, relativePaths });
      return { source_dir: `/uploads/${row.id}`, upload_token: `token-${row.id}` };
    },
    createBatch: async (payload) => {
      createPayload = payload;
      return { batch: { parent_task_id: "parent-1" }, items: [] };
    },
    onRowStatus: (rowId, state, message) => statuses.push([rowId, state, message]),
  },
);

assert.equal(result.batch.parent_task_id, "parent-1");
assert.equal(uploadCalls.length, 2);
assert.deepEqual(uploadCalls[0].relativePaths, {
  notebook: "notebook/shared.ipynb",
  sample: "sample/shared.csv",
  pmml: "pmml/shared.pmml",
  dictionary: "dictionary/shared.xlsx",
});
assert.equal(uploadCalls[1].rowId, "row-2");
assert.deepEqual(createPayload, {
  batch_name: "季度批次",
  validator: "Eddy",
  run_mode: "agent",
  items: [
    {
      model_name: "模型 A",
      model_version: "v1",
      source_dir: "/uploads/row-1",
      upload_token: "token-row-1",
      notebook_path: "notebook/shared.ipynb",
      sample_path: "sample/shared.csv",
      pmml_path: "pmml/shared.pmml",
      dictionary_path: "dictionary/shared.xlsx",
    },
    {
      model_name: "模型 B",
      model_version: "v2",
      source_dir: "/uploads/row-2",
      upload_token: "token-row-2",
      notebook_path: "notebook/shared.ipynb",
      sample_path: "sample/shared.csv",
      pmml_path: "pmml/shared.pmml",
      dictionary_path: "dictionary/shared.xlsx",
    },
  ],
});
assert.deepEqual(statuses.map((entry) => entry.slice(0, 2)), [
  ["row-1", "uploading"],
  ["row-1", "uploaded"],
  ["row-2", "uploading"],
  ["row-2", "uploaded"],
]);
"""
    )


def test_batch_create_stops_before_batch_post_when_a_model_upload_fails():
    _run_create_module(
        r"""
const file = (name) => ({ name, size: 1, lastModified: 1 });
const row = (id) => ({
  id,
  modelName: id,
  modelVersion: "v1",
  files: {
    notebook: file("validation.ipynb"),
    sample: file("sample.csv"),
    pmml: file("model.pmml"),
    dictionary: file("dictionary.csv"),
  },
});
let createCalls = 0;
const statuses = [];
const cleanedTokens = [];
await assert.rejects(
  batchCreate.submitValidationBatchDraft(
    { batchName: "批次", validator: "Eddy", rows: [row("模型 A"), row("模型 B")] },
    {
      uploadMaterials: async ({ row: current }) => {
        if (current.id === "模型 B") throw new Error("网络中断");
        return { source_dir: "/uploads/model-a", upload_token: "token-model-a" };
      },
      createBatch: async () => { createCalls += 1; },
      cleanupMaterials: async (token) => { cleanedTokens.push(token); },
      onRowStatus: (rowId, state, message) => statuses.push([rowId, state, message]),
    },
  ),
  /网络中断/,
);
assert.equal(createCalls, 0);
assert.deepEqual(cleanedTokens, ["token-model-a"]);
assert.deepEqual(statuses.at(-1).slice(0, 2), ["模型 B", "error"]);
assert.match(statuses.at(-1)[2], /网络中断/);
"""
    )


def test_batch_create_cleans_every_owned_upload_when_batch_post_fails():
    _run_create_module(
        r"""
const file = (name, marker) => ({ name, size: marker, lastModified: marker });
const makeRow = (id, marker) => ({
  id,
  modelName: id,
  modelVersion: "v1",
  files: {
    notebook: file("validation.ipynb", marker),
    sample: file("sample.csv", marker),
    pmml: file("model.pmml", marker),
    dictionary: file("dictionary.csv", marker),
  },
});
const rows = [makeRow("模型 A", 1), makeRow("模型 B", 2)];
const cleanedTokens = [];

await assert.rejects(
  batchCreate.submitValidationBatchDraft(
    { batchName: "批次", validator: "Eddy", rows },
    {
      uploadMaterials: async ({ row }) => ({
        source_dir: `/uploads/${row.id}`,
        upload_token: `token-${row.id}`,
      }),
      createBatch: async () => { throw new Error("批次校验失败"); },
      cleanupMaterials: async (token) => { cleanedTokens.push(token); },
    },
  ),
  /批次校验失败/,
);
assert.deepEqual(cleanedTokens.sort(), ["token-模型 A", "token-模型 B"]);
assert.equal(rows[0].uploadCache, null);
assert.equal(rows[1].uploadCache, null);
"""
    )


def test_batch_create_dialog_and_welcome_card_are_wired_as_an_independent_flow():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    task_types = _read_static("js/task-types.js")
    create_js = _read_static("js/validation-batch-create.js")

    assert 'id="welcomeValidationBatchCard"' not in index_html
    assert 'data-task-kind="validation_batch"' not in index_html
    assert '<strong>批量模型验证</strong>' not in index_html
    assert 'id="addValidationModelRowButton"' in index_html
    assert 'id="validationBatchCreateDialog"' in index_html
    assert 'id="validationBatchCreateForm"' in index_html
    assert 'id="validationBatchRows"' in index_html
    assert 'id="addValidationBatchRowButton"' in index_html
    assert 'static/css/validation-batch-create.css' in index_html

    assert 'validation_batch: {' in task_types
    batch_definition = task_types.split('validation_batch: {', 1)[1].split('},', 1)[0]
    assert 'available: true' in batch_definition

    assert 'from "./js/validation-batch-create.js"' in app_js
    assert 'createValidationBatchCreateController({' in app_js
    assert 'validationBatchCreateController.open()' not in app_js
    assert 'validationBatchCreateController.bind()' in app_js
    assert 'validationBatchCreateController.close()' in app_js
    assert 'payload.batch.parent_task_id' in app_js
    assert 'await refreshTasks()' in app_js
    assert 'selectTask(parentTask)' in app_js

    assert '"/api/validation-batches/material-uploads"' in create_js
    assert 'upload_token' in create_js
    assert 'method: "DELETE"' in create_js
    assert '"/api/validation-batches"' in create_js
    assert "body: JSON.stringify(payload)" in create_js
    assert '"/start"' not in create_js
    assert 'api/validation-batches/${' not in create_js
    assert "/api/validation-batches" in _read_static("js/create-task-dialog.js")
    assert "function addValidationModelRow" in _read_static("js/create-task-dialog.js")


def test_classify_validation_batch_files_maps_roles_without_reusing_the_same_file():
    _run_create_module(
        r"""
const files = [
  { name: "model.ipynb", size: 10 },
  { name: "sample.parquet", size: 11 },
  { name: "score.pmml", size: 12 },
  { name: "数据字典.xlsx", size: 13 },
];
const classified = batchCreate.classifyValidationBatchFiles(files);
assert.equal(classified.notebook.name, "model.ipynb");
assert.equal(classified.sample.name, "sample.parquet");
assert.equal(classified.pmml.name, "score.pmml");
assert.equal(classified.dictionary.name, "数据字典.xlsx");
assert.equal(batchCreate.classifyValidationBatchFiles(files.slice(0, 3)), null);
"""
    )


def test_batch_draft_accepts_source_dir_instead_of_four_file_uploads():
    _run_create_module(
        r"""
const byPath = batchCreate.validateValidationBatchDraft({
  batchName: "批次 A",
  validator: "Eddy",
  rows: [
    { id: "row-1", modelName: "T卡", sourceDir: "/tmp/model-a" },
    { id: "row-2", modelName: "A卡", sourceDir: "/tmp/model-b" },
  ],
});
assert.equal(byPath.valid, true);
assert.deepEqual(byPath.rowErrors, {});
"""
    )


def test_multi_model_create_uses_classified_primary_files_not_first_extra_row():
    create_js = _read_static("js/create-task-dialog.js")
    assert "classifyValidationBatchFiles(" in create_js
    assert "files: rows[0].files" not in create_js
    assert "collectPrimaryValidationModel" in create_js
    assert "classifiedFilesFromSelection" in create_js


def test_batch_create_dialog_declares_four_explicit_file_roles_and_limits():
    create_js = _read_static("js/validation-batch-create.js")
    create_css = _read_static("css/validation-batch-create.css")

    for role in ("notebook", "sample", "pmml", "dictionary"):
        assert f'fileInputMarkup("{role}")' in create_js
    assert 'data-batch-file-role="${role}"' in create_js
    assert 'accept: ".ipynb"' in create_js
    assert 'accept: ".pmml"' in create_js
    assert ".csv,.parquet,.feather,.xlsx,.xls" in create_js
    assert ".csv,.parquet,.xlsx,.xls" in create_js
    assert "MAX_VALIDATION_BATCH_ROWS = 10" in create_js
    assert "MIN_VALIDATION_BATCH_ROWS = 2" in create_js
    assert ".validation-batch-create" in create_css
