import json
from pathlib import Path
import subprocess


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
    assert "元数据不完整" in body
    assert "需人工确认" in body


def test_material_binding_names_the_dictionary_role_as_feature_metadata_too() -> None:
    source = _binding_source()

    assert 'label: "Metadata", caption: "数据字典 / 特征元数据"' in source


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
            "assert.ok(html.includes('data_dictionary.csv · 79 B · 元数据不完整 · 不可选'));",
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
    assert "setStatus(selectionIssue, selectionIssue ? \"error\" : \"info\")" in source
