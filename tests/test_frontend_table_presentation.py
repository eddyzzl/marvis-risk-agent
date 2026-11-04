"""Readable platform tables (2026-10-19 user feedback, eval U13).

Enum codes read as Chinese tags, evidence strings as metric chips, long
decimals are trimmed for display only, and field/value tables read as a
field list. The platform's values are never recomputed or judged.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

MODULE = (Path(__file__).resolve().parents[1] / "marvis" / "static" / "js" / "table-presentation.js").as_uri()


def _run(body: str):
    script = f"const m = await import({json.dumps(MODULE)});\n{body}"
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_status_codes_and_evidence_strings_read_as_tags_and_chips():
    data = _run(
        "process.stdout.write(JSON.stringify({"
        "candidate: m.presentedCellHtml('candidate'),"
        "notRec: m.enumTag('not_recommended'),"
        "medium: m.enumTag('medium'),"
        "free: m.presentedCellHtml('有区分力；本次没有可用 PSI 结果'),"
        "pairs: m.metricPairs('iv=0.90438；ks=0.40096692502494047；missing_rate=0.0；mode_rate=0.0020833333333333333；unique_count=480'),"
        "mixed: m.metricPairs('iv=0.9 说明文字 ks=0.4'),"
        "}));"
    )
    assert data["candidate"]["cls"] == "cell-tag" and "候选" in data["candidate"]["html"]
    assert 'title="candidate"' in data["candidate"]["html"]
    assert data["notRec"] == ["不推荐", "muted"]
    assert data["medium"] == ["中", "muted"]
    assert data["free"] is None
    assert data["pairs"] == [
        {"label": "IV", "value": "0.9044"},
        {"label": "KS", "value": "0.401"},
        {"label": "缺失率", "value": "0%"},
        {"label": "最常见值占比", "value": "0.21%"},
        {"label": "唯一值数", "value": "480"},
    ]
    assert data["mixed"] is None


def test_long_decimals_trim_and_field_tables_become_lists():
    data = _run(
        "process.stdout.write(JSON.stringify({"
        "trim: [m.displayNumberText('0.40096692502494047'), m.displayNumberText('480'), m.displayNumberText('0.5'), m.displayNumberText('abc')],"
        "list: m.fieldListModel({ columns: ['验证字段', '候选值'], rows: ["
        "  ['algorithm', '\"lr\"'], ['algorithm', '\"lr\"'],"
        "  ['feature_metadata_selection', '{\"category_col\":\"类别\",\"feature_col\":\"feature\",\"importance_col\":\"importance\",\"metadata_sheet\":null}'],"
        "  ['model_params', '{\"algorithm\":\"lr\",\"intercept\":0.0}'],"
        "] }),"
        "notList: m.fieldListModel({ columns: ['特征', 'IV'], rows: [['sig1', '0.9'], ['sig2', '0.3']] }),"
        "}));"
    )
    assert data["trim"] == ["0.4010", "480", "0.5", "abc"]
    assert data["list"]["items"] == [
        {"key": "algorithm", "label": "算法", "value": "lr"},
        {"key": "feature_metadata_selection", "label": "特征元数据列", "value": "类别列 类别 · 特征列 feature · 重要性列 importance"},
        {"key": "model_params", "label": "模型参数", "value": "算法 lr · 截距 0"},
    ]
    assert data["notList"] is None




def test_structured_contract_candidates_read_as_business_words():
    # Eval S1/S2: 「模型参数来源」「特征元数据列」 options showed raw JSON.
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    module = (root / "marvis/static/js/table-presentation.js").as_uri()
    script = "\n".join([
        f"const m = await import({json.dumps(module)});",
        "process.stdout.write(JSON.stringify([",
        "  m.readableValueText({ category_col: 'category', feature_col: 'feature', importance_col: 'importance', metadata_sheet: null }),",
        "  m.readableValueText({}),",
        "]));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=root, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["类别列 category · 特征列 feature · 重要性列 importance", "无额外参数"]
    app_js = (root / "marvis/static/app.js").read_text(encoding="utf-8")
    assert 'readableValueText(value) : JSON.stringify(value))' in app_js
