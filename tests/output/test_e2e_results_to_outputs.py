import json
from pathlib import Path

import pandas as pd
from docx import Document
from openpyxl import load_workbook

from marvis.output.excel import write_validation_excel
from marvis.output.word import write_validation_word
from marvis.validation.config import ValidationConfig
from marvis.validation.engine import EngineInputs, run_validation


class _IdentityScorer:
    def score(self, df: pd.DataFrame) -> list[float]:
        return df["x1"].astype(float).tolist()


def _make_template(path: Path) -> Path:
    document = Document()
    document.add_paragraph("{{TEXT:report_title}}")
    document.add_paragraph("OOT KS：{{TEXT:oot_ks}}")
    document.add_paragraph("{{IMAGE:overall_model_effect}}")
    document.add_paragraph("{{IMAGE:pressure_ks_table}}")
    document.save(path)
    return path


def test_engine_output_round_trip(tmp_path: Path):
    rows = []
    for split in ("train", "test", "oot"):
        for i in range(20):
            x1 = (i + 1) / 21
            rows.append({
                "x1": x1, "sample_score": x1, "y": int(i >= 10),
                "split": split, "apply_month": "202503" if split == "train" else "202505",
            })
    sample = pd.DataFrame(rows)
    dictionary = pd.DataFrame({"特征名": ["x1"], "类别": ["征信"]})
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(json.dumps({
        "feature_importance": [{"feature": "x1", "importance": 1.0}],
        "hyperparameters": {"max_depth": 4, "learning_rate": 0.1},
    }), encoding="utf-8")

    results = run_validation(
        inputs=EngineInputs(
            model_name="A卡", model_version="v1",
            algorithm="lgb",
            sample=sample, data_dictionary=dictionary, model_meta_path=meta_path,
            input_scorer=_IdentityScorer(),
        ),
        config=ValidationConfig(
            target_col="y", score_col="sample_score", split_col="split",
            time_col="apply_month", feature_columns=["x1"],
            bin_count=5, random_sample_size=10,
        ),
    )

    excel_path = tmp_path / "out.xlsx"
    write_validation_excel(results, excel_path)
    wb = load_workbook(excel_path)
    assert "验证总览" in wb.sheetnames
    assert "压力测试_分箱_征信" in wb.sheetnames

    template = _make_template(tmp_path / "template.docx")
    word_path = tmp_path / "report.docx"
    word_result = write_validation_word(
        results, template_path=template,
        output_path=word_path, image_output_dir=tmp_path / "images",
    )
    assert word_result.unresolved_placeholders == []
    assert word_path.exists()
