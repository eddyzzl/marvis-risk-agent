"""Public, fixed synthetic Notebook/PMML compatibility Workflow scenario.

This is the existing manual/API Workflow with real model critiques, not the
separate V2 PMML-first validation Agent entry. No production acceptance is claimed.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd


_NOTEBOOK_SOURCE = """import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Public synthetic supplied model: logit(p_bad) = x1 - x2.
# This is a fixed mathematical validation fixture, not a trained business model.
RMC_SAMPLE_DF = pd.read_csv('sample.csv')
RMC_TARGET_COL = 'y'
RMC_ALGORITHM = 'lr'
RMC_SPLIT_COL = 'split'
RMC_TIME_COL = 'apply_month'
RMC_FEATURE_COLS = ['x1', 'x2']
RMC_MODEL = LogisticRegression()
RMC_MODEL.classes_ = np.array([0, 1])
RMC_MODEL.coef_ = np.array([[1.0, -1.0]])
RMC_MODEL.intercept_ = np.array([0.0])
RMC_MODEL.n_features_in_ = 2
RMC_MODEL.feature_names_in_ = np.array(['x1', 'x2'])
def RMC_SCORE_FN(df):
    return RMC_MODEL.predict_proba(df[['x1', 'x2']])[:, 1]
"""
_PMML = '''<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">
 <Header copyright="MARVIS public synthetic validation scenario"/>
 <DataDictionary numberOfFields="3">
  <DataField name="x1" optype="continuous" dataType="double"/>
  <DataField name="x2" optype="continuous" dataType="double"/>
  <DataField name="y" optype="categorical" dataType="integer"><Value value="0"/><Value value="1"/></DataField>
 </DataDictionary>
 <RegressionModel functionName="classification" normalizationMethod="logit" algorithmName="logisticRegression">
  <MiningSchema><MiningField name="x1"/><MiningField name="x2"/><MiningField name="y" usageType="target"/></MiningSchema>
  <Output><OutputField name="probability_1" feature="probability" value="1"/></Output>
  <RegressionTable intercept="0.0" targetCategory="1"><NumericPredictor name="x1" coefficient="1.0"/><NumericPredictor name="x2" coefficient="-1.0"/></RegressionTable>
  <RegressionTable intercept="0.0" targetCategory="0"/>
 </RegressionModel>
</PMML>
'''


def normal_validation_materials(data: Path):
    rows = []
    for split, month in (("train", "202601"), ("test", "202602"), ("oot", "202603")):
        for i in range(60):
            rows.append({
                "x1": (i - 30) / 15, "x2": (i % 7 - 3) / 10,
                "y": int((i >= 30) != (i % 11 == 0)), "split": split,
                "apply_month": month,
                # Deliberately unrelated supplied sample score. The required
                # consistency comparison must use actual notebook model scores.
                "pred": 0.01,
            })
    pd.DataFrame(rows).to_csv(data / "sample.csv", index=False, lineterminator="\n")
    (data / "dictionary.csv").write_text(
        "特征名,类别,importance\nx1,synthetic_internal,0.5\nx2,synthetic_external,0.5\n",
        encoding="utf-8",
    )
    (data / "model.pmml").write_text(_PMML, encoding="utf-8")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{
            "cell_type": "code", "id": "public-supplied-logistic-model", "execution_count": None,
            "metadata": {}, "outputs": [], "source": _NOTEBOOK_SOURCE.splitlines(keepends=True),
        }],
    }
    (data / "model.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return [{
        "path": name, "role": role, "sha256": hashlib.sha256((data / name).read_bytes()).hexdigest(),
        "source_kind": "synthetic",
    } for role, name in (
        ("notebook", "model.ipynb"), ("sample", "sample.csv"),
        ("pmml", "model.pmml"), ("dictionary", "dictionary.csv"),
    )]


def normal_validation_cases(materials):
    case = {
        "id": "synthetic_normal_validation_compatibility", "revision": "2",
        "family": "validation", "case_set": "development", "scenario": "normal",
        "task": {"task_type": "validation", "algorithm": "lr", "feature_columns": ["x1", "x2"]},
        "materials": materials,
        "business_constraints_source": (
            "Public synthetic 180-row sample and fixed supplied logistic model logit(p_bad)=x1-x2; "
            "60 rows each in train/test/OOT. Labels, months and dictionary importance 0.5/0.5 are "
            "explicit synthetic declarations. Sample pred=0.01 is deliberately unrelated: consistency "
            "must compare the executed notebook in-memory LogisticRegression with submitted PMML. "
            "User explicitly selects all four uploaded roles, starts the existing manual/API model_validation "
            "Workflow and reviews the report gate. Actual v1_compat subprocess tools are required. "
            "This is not the separate V2 PMML-first validation Agent P1 entry. No real maturity, source "
            "authenticity, financial evidence, production deployment or business acceptance is established."
        ),
        "actions": [
            {"kind": "start_validation_workflow", "content": "已人工选择四份公开合成材料，确认字典 importance=0.5/0.5，仅运行现有兼容模型验证工作流，比较 Notebook 内存模型与 PMML 分数。"},
            {"kind": "approve_step", "tool": "v1_compat.render_reports", "content": "已查看实际分数一致性与合成样本验证结果，批准生成本地 Word 和 Excel；这不代表真实业务或生产验收。"},
        ],
        "budget": {"wall_seconds": 420, "max_llm_attempts": 40, "max_http_requests": 260, "max_output_tokens_per_attempt": 2048},
    }
    assertions = [{"kind": "tool_succeeded", "tool": "v1_compat." + tool} for tool in (
        "scan_materials", "run_notebook", "compute_validation_metrics", "render_reports",
    )] + [
        {"kind": "http_status", "stage": "human_validation_material_selection", "value": 200},
        {"kind": "http_status", "stage": "human_validation_workflow_start", "value": 200},
        {"kind": "http_status", "stage": "human_approval", "value": 202},
        {"kind": "output_equals", "tool": "v1_compat.run_notebook", "path": ["status"], "value": "executed"},
        {"kind": "output_equals", "tool": "v1_compat.compute_validation_metrics", "path": ["score_consistency_passed"], "value": True},
        {"kind": "output_length", "tool": "v1_compat.render_reports", "path": ["artifacts"], "value": 2},
        {"kind": "validation_report_verified", "tool": "v1_compat.render_reports", "value": "excel"},
        {"kind": "validation_report_verified", "tool": "v1_compat.render_reports", "value": "word"},
    ]
    return [case], {case["id"]: {"result": "done", "assertions": assertions}}
