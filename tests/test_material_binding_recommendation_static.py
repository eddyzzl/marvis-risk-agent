import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _binding_source() -> str:
    return (ROOT / "marvis/static/js/material-binding-dialog.js").read_text(
        encoding="utf-8"
    )


def test_material_binding_prefers_unique_backend_recommendation() -> None:
    source = _binding_source()
    function_start = source.index("function defaultSelectionForRole")
    function_end = source.index("\n}\n", function_start)
    body = source[function_start:function_end]

    assert "candidate.recommended" in body
    assert "selectedCandidate" in body
    assert body.index("if (selected && selectedCandidate) return selected") < body.index(
        "candidate.recommended"
    )
    assert body.index("candidate.recommended") < body.index("candidate.role === role.role")


def test_material_binding_does_not_fall_back_to_incomplete_dictionary() -> None:
    source = _binding_source()
    function_start = source.index("function defaultSelectionForRole")
    function_end = source.index("\n}\n", function_start)
    body = source[function_start:function_end]

    assert 'role.role === "data_dictionary"' in body
    assert "hasMetadataAssessment" in body
    assert body.index("hasMetadataAssessment") < body.index(
        "candidate.role === role.role"
    )


def test_material_binding_refreshes_recommendation_after_pmml_selection() -> None:
    source = _binding_source()

    assert "async function refreshMetadataRecommendation(pmmlPath)" in source
    assert "?pmml_path=${encodeURIComponent(pmmlPath)}" in source
    assert 'if (field === "pmml_path")' in source
    assert 'if (dictionarySelect) dictionarySelect.value = ""' in source
    assert "confirmButton.disabled = true" in source
    assert "confirmButton.disabled = false" in source
    assert "const currentSelection = collectSelection();" in source
    assert "...currentSelection" in source


def test_material_binding_explains_compatible_metadata_recommendation() -> None:
    source = _binding_source()
    function_start = source.index("function candidateText")
    function_end = source.index("\n}\n", function_start)
    body = source[function_start:function_end]

    assert "metadata_compatibility" in body
    assert "推荐：与 PMML 完整匹配" in body
    assert "与 PMML 匹配" in body
    assert "incompatibleMetadataText(candidate)" in body
    assert "元数据不完整" in source
    assert "需人工确认" in body


def test_incompatible_feature_metadata_is_disabled_with_recoverable_guidance() -> None:
    source = _binding_source()
    module_url = (
        ROOT / "marvis" / "static" / "js" / "material-binding-dialog.js"
    ).as_uri()
    script = "\n".join(
        [
            "import assert from 'node:assert/strict';",
            f"import {{ materialSelectionIssue, renderRoleRow }} from {json.dumps(module_url)};",
            "const role = { role: 'data_dictionary', field: 'dictionary_path', label: 'Metadata', caption: '数据字典 / 特征元数据' };",
            "const candidate = { relative_path: 'data_dictionary.csv', size_bytes: 79, recommended: false, metadata_compatibility: { status: 'incompatible', blocking_errors: ['CSV: missing importance column alias'] } };",
            "const payload = { candidates: { data_dictionary: [candidate] } };",
            "const html = renderRoleRow(role, [candidate], 'data_dictionary.csv');",
            "assert.ok(html.includes('data_dictionary.csv · 79 B · 元数据不完整：缺少 importance（特征重要性）列，请补充后重新上传 · 不可选'));",
            "assert.ok(html.includes('data_dictionary.csv：缺少 importance（特征重要性）列，请补充后重新上传。'));",
            "const sample = { role: 'sample', relative_path: 'sample.csv', size_bytes: 9506, metadata_compatibility: { status: 'incompatible', blocking_errors: ['metadata file: candidate is the sample data file'] } };",
            "const sampleHtml = renderRoleRow(role, [candidate, sample], '');",
            "assert.match(sampleHtml, /<option[^>]*disabled[^>]*>sample[.]csv · 9[.]3 KB · 样本数据，不能作数据字典 · 不可选/);",
            "assert.ok(materialSelectionIssue({ dictionary_path: 'data_dictionary.csv' }, payload).includes('缺少 importance（特征重要性）列'));",
            "assert.match(html, /<option[^>]*disabled[^>]*>data_dictionary[.]csv/);",
            "assert.ok(html.includes('请更换 PMML 或补齐 feature/category/importance 列后重新扫描'));",
            "assert.ok(materialSelectionIssue({ dictionary_path: 'data_dictionary.csv' }, payload).includes('当前特征元数据不可用'));",
            "assert.ok(materialSelectionIssue({ dictionary_path: '' }, payload).includes('没有可用的特征元数据'));",
        ]
    )
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "completeSelection(payload.selection) && !selectionIssue" in source
    # The dialog opens on the selection issue (or on why the automatic save
    # of a fully matched selection failed).
    assert "const opening = autoSaveError || selectionIssue;" in source
    assert "setStatus(opening, opening ? \"error\" : \"info\")" in source


def test_material_dialog_uploads_supplements_and_picks_one_role_per_file() -> None:
    # Eval round 1 (P0-5): the dialog had no upload, so a corrected dictionary
    # could not reach the task. A CSV is offered under every CSV role; only a
    # dictionary the platform found usable, or the role the file was
    # recognised as, gets the new file.
    module = (ROOT / "marvis/static/js/material-binding-dialog.js").as_uri()
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        "const dict = 'Supp/metadata_v2.csv';",
        "const sample = 'Supp/sample_v2.csv';",
        "const payload = { candidates: {",
        "  sample: [{ relative_path: dict, role: 'data_dictionary' }, { relative_path: sample, role: 'sample' }],",
        "  data_dictionary: [",
        "    { relative_path: dict, role: 'data_dictionary', metadata_compatibility: { status: 'compatible' } },",
        "    { relative_path: sample, role: 'sample', metadata_compatibility: { status: 'incompatible', blocking_errors: ['sample data file'] } },",
        "  ],",
        "  notebook: [], model_pmml: [],",
        "} };",
        "const both = m.uploadedMaterialSelection(payload, [dict, sample]);",
        "const broken = m.uploadedMaterialSelection({ candidates: { data_dictionary: [",
        "  { relative_path: dict, role: 'data_dictionary', metadata_compatibility: { status: 'incompatible', blocking_errors: ['missing importance column alias'] } },",
        "] } }, [dict]);",
        "process.stdout.write(JSON.stringify({ both, broken }));",
    ])
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    assert data["both"] == {
        "selection": {"dictionary_path": "Supp/metadata_v2.csv", "sample_path": "Supp/sample_v2.csv"},
        "chosenRoles": ["数据字典 / 特征元数据", "样本数据"],
    }
    assert data["broken"] == {"selection": {}, "chosenRoles": []}

    html = (ROOT / "marvis/static/index.html").read_text(encoding="utf-8")
    dialog = html[html.index('id="materialBindingDialog"'):html.index("</dialog>", html.index('id="materialBindingDialog"'))]
    for marker in ('id="materialBindingFileInput"', 'id="materialBindingRefreshButton"', "上传补充文件"):
        assert marker in dialog
    source = _binding_source()
    assert "/materials/files" in source
    assert "supplement_upload_supported" in source




def test_fully_matched_materials_are_saved_without_the_dialog() -> None:
    # Eval S2: four matched materials (dictionary marked 推荐) still opened
    # the dialog for one more click. Only the choices the dialog would itself
    # preselect are saved; anything ambiguous still asks.
    module = (ROOT / "marvis/static/js/material-binding-dialog.js").as_uri()
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        "const base = { selection: {}, candidates: {",
        "  notebook: [{ relative_path: 'model.ipynb', role: 'notebook' }],",
        "  sample: [{ relative_path: 'sample.csv', role: 'sample' }],",
        "  model_pmml: [{ relative_path: 'model.pmml', role: 'model_pmml' }],",
        "  data_dictionary: [",
        "    { relative_path: 'dict.csv', role: 'data_dictionary', recommended: true, metadata_compatibility: { status: 'compatible' } },",
        "    { relative_path: 'sample.csv', role: 'sample', metadata_compatibility: { status: 'incompatible', blocking_errors: ['sample data file'] } },",
        "  ],",
        "} };",
        "const ambiguous = JSON.parse(JSON.stringify(base));",
        "ambiguous.candidates.data_dictionary = [",
        "  { relative_path: 'a.csv', role: 'data_dictionary', metadata_compatibility: { status: 'compatible' } },",
        "  { relative_path: 'b.csv', role: 'data_dictionary', metadata_compatibility: { status: 'compatible' } },",
        "];",
        "const twoSamples = JSON.parse(JSON.stringify(base));",
        "twoSamples.candidates.sample.push({ relative_path: 'sample_v2.csv', role: 'sample' });",
        "process.stdout.write(JSON.stringify({",
        "  matched: m.autoMaterialSelection(base),",
        "  ambiguous: m.autoMaterialSelection(ambiguous),",
        "  twoSamples: m.autoMaterialSelection(twoSamples),",
        "}));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data["matched"] == {
        "notebook_path": "model.ipynb",
        "sample_path": "sample.csv",
        "pmml_path": "model.pmml",
        "dictionary_path": "dict.csv",
    }
    assert data["ambiguous"] is None
    assert data["twoSamples"] is None
    source = _binding_source()
    assert "const automatic = force ? null : autoMaterialSelection(payload);" in source
