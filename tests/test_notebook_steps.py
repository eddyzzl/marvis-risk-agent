from pathlib import Path

import nbformat

from marvis.notebook_steps import notebook_step_plan, notebook_step_preview


def test_notebook_step_plan_groups_code_cells_under_markdown_headings():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("bootstrap = True"),
            nbformat.v4.new_markdown_cell("# 数据准备"),
            nbformat.v4.new_code_cell("load_data()"),
            nbformat.v4.new_code_cell("clean_data()"),
            nbformat.v4.new_markdown_cell("## 模型训练"),
            nbformat.v4.new_code_cell("fit_model()"),
            nbformat.v4.new_markdown_cell("ordinary paragraph"),
            nbformat.v4.new_code_cell("evaluate_model()"),
        ]
    )

    plan = notebook_step_plan(notebook)

    assert [step.title for step in plan.steps] == [
        "Notebook 初始化",
        "数据准备",
        "模型训练",
    ]
    assert plan.cell_to_step == {
        0: "notebook-init",
        2: "step-2",
        3: "step-2",
        5: "step-5",
        7: "step-5",
    }
    assert plan.steps[1].cell_indexes == [2, 3]
    assert plan.steps[2].source_previews == ["fit_model()", "evaluate_model()"]


def test_notebook_step_plan_marks_injected_cells_as_system_steps():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("RMC_SAMPLE_PATH = 'sample.csv'"),
            nbformat.v4.new_markdown_cell("# 模型训练"),
            nbformat.v4.new_code_cell("fit_model()"),
            nbformat.v4.new_code_cell("validate_contract()"),
        ]
    )
    notebook.cells[0].metadata["marvis"] = "head"
    notebook.cells[3].metadata["marvis"] = "tail"

    plan = notebook_step_plan(notebook)

    assert [step.title for step in plan.steps] == [
        "平台初始化",
        "模型训练",
        "平台契约检查",
    ]
    assert plan.cell_to_step[0] == "system-head"
    assert plan.cell_to_step[3] == "system-tail"


def test_notebook_step_plan_infers_titles_for_headingless_code_cells():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("import pandas as pd\nfrom xgboost import XGBClassifier"),
            nbformat.v4.new_code_cell("sample = pd.read_csv('sample.csv')"),
            nbformat.v4.new_code_cell("train, oot = train_test_split(sample, test_size=0.2)"),
            nbformat.v4.new_code_cell("model = XGBClassifier()\nmodel.fit(X_train, y_train)"),
            nbformat.v4.new_code_cell("score = model.predict_proba(X_oot)[:, 1]\nauc = roc_auc_score(y_oot, score)"),
            nbformat.v4.new_code_cell("importance = model.feature_importances_"),
            nbformat.v4.new_code_cell("result.to_csv('metrics.csv', index=False)"),
        ]
    )

    plan = notebook_step_plan(notebook)

    assert [step.title for step in plan.steps] == [
        "导入依赖",
        "读取数据",
        "样本切分",
        "模型训练",
        "模型打分与指标计算",
        "特征分析",
        "保存结果",
    ]
    assert [step.cell_indexes for step in plan.steps] == [[0], [1], [2], [3], [4], [5], [6]]
    assert plan.cell_to_step == {
        0: "cell-1",
        1: "cell-2",
        2: "cell-3",
        3: "cell-4",
        4: "cell-5",
        5: "cell-6",
        6: "cell-7",
    }


def test_notebook_step_plan_uses_function_name_when_headingless_code_is_unknown():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("build_custom_snapshot(user_df)"),
        ]
    )

    plan = notebook_step_plan(notebook)

    assert [step.title for step in plan.steps] == ["执行 build_custom_snapshot"]


def test_notebook_step_plan_does_not_label_pmml_import_or_pipeline_as_export():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import pandas as pd\n"
                "from sklearn2pmml import PMMLPipeline, sklearn2pmml\n"
                "df = pd.read_csv('sample.csv')\n"
            ),
            nbformat.v4.new_code_cell(
                "vars1 = df.columns.tolist()\n"
                "vars1.remove('target')\n"
            ),
            nbformat.v4.new_code_cell(
                "trainx = df[(df.split_tag == 'train')][vars1]\n"
                "testx = df[(df.split_tag == 'test')][vars1]\n"
            ),
            nbformat.v4.new_code_cell(
                "pipeline = PMMLPipeline([('classifier', bst)])\n"
                "pipeline.fit(trainx, trainy)\n"
                "train_pred = pipeline.predict_proba(trainx)[:, 1]\n"
            ),
            nbformat.v4.new_code_cell("sklearn2pmml(pipeline, 'model.pmml')"),
        ]
    )

    plan = notebook_step_plan(notebook)

    assert [step.title for step in plan.steps] == [
        "读取数据",
        "特征筛选",
        "样本切分",
        "模型训练与打分",
        "PMML 导出",
    ]


def test_notebook_step_plan_names_injected_validation_stages():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("score_pmml()"),
            nbformat.v4.new_code_cell("compare_scores()"),
            nbformat.v4.new_code_cell("compute_ks()"),
            nbformat.v4.new_code_cell("compute_psi()"),
            nbformat.v4.new_code_cell("compute_bins()"),
            nbformat.v4.new_code_cell("run_pressure_test()"),
        ]
    )
    notebook.cells[0].metadata["marvis"] = "repro-pmml"
    notebook.cells[1].metadata["marvis"] = "repro-compare"
    notebook.cells[2].metadata["marvis"] = "metrics-ks"
    notebook.cells[3].metadata["marvis"] = "metrics-psi"
    notebook.cells[4].metadata["marvis"] = "metrics-binning"
    notebook.cells[5].metadata["marvis"] = "metrics-stress"

    plan = notebook_step_plan(notebook)

    assert [step.title for step in plan.steps] == [
        "PMML 打分",
        "分数一致性对比",
        "KS 计算",
        "PSI 计算",
        "分箱计算",
        "压力测试",
    ]
    assert plan.cell_to_step[0] == "system-repro-pmml"
    assert plan.cell_to_step[1] == "system-repro-compare"
    assert plan.cell_to_step[2] == "system-metrics-ks"
    assert plan.cell_to_step[3] == "system-metrics-psi"
    assert plan.cell_to_step[4] == "system-metrics-binning"
    assert plan.cell_to_step[5] == "system-metrics-stress"


def test_notebook_step_plan_uses_latest_retried_system_cell():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("old_prepare()"),
            nbformat.v4.new_code_cell("old_score()"),
            nbformat.v4.new_code_cell("new_prepare()"),
            nbformat.v4.new_code_cell("new_score()"),
        ]
    )
    notebook.cells[0].metadata["marvis"] = "metrics-prepare"
    notebook.cells[1].metadata["marvis"] = "metrics-score"
    notebook.cells[2].metadata["marvis"] = "metrics-prepare"
    notebook.cells[3].metadata["marvis"] = "metrics-score"

    plan = notebook_step_plan(notebook)
    steps_by_id = {step.id: step for step in plan.steps}

    assert steps_by_id["system-metrics-prepare"].cell_indexes == [2]
    assert steps_by_id["system-metrics-prepare"].source_previews == ["new_prepare()"]
    assert steps_by_id["system-metrics-score"].cell_indexes == [3]
    assert steps_by_id["system-metrics-score"].source_previews == ["new_score()"]
    assert plan.cell_to_step == {
        2: "system-metrics-prepare",
        3: "system-metrics-score",
    }


def test_notebook_step_preview_reads_titles_without_execution(tmp_path: Path):
    notebook_path = tmp_path / "model.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("# 数据准备"),
                nbformat.v4.new_code_cell("load_data()"),
                nbformat.v4.new_markdown_cell("## 模型训练"),
                nbformat.v4.new_code_cell("fit_model()"),
            ]
        ),
        notebook_path,
    )

    preview = notebook_step_preview(notebook_path)

    assert preview == [
        {
            "id": "step-1",
            "step_order": 1,
            "title": "数据准备",
            "status": "pending",
            "cell_count": 1,
            "cell_indexes": [1],
            "source_previews": ["load_data()"],
            "system": False,
        },
        {
            "id": "step-3",
            "step_order": 2,
            "title": "模型训练",
            "status": "pending",
            "cell_count": 1,
            "cell_indexes": [3],
            "source_previews": ["fit_model()"],
            "system": False,
        },
    ]


def test_notebook_step_preview_accepts_non_utf8_notebook_bytes(tmp_path: Path):
    notebook_path = tmp_path / "model.ipynb"
    notebook_path.write_bytes(
        (
            '{"cells":[{"cell_type":"markdown","id":"markdown-1","metadata":{},'
            '"source":["score · label"]},{"cell_type":"code","id":"code-1","metadata":{},'
            '"source":["sample = pd.read_csv(\\"sample.csv\\")"]}],'
            '"metadata":{},"nbformat":4,"nbformat_minor":5}'
        ).encode("cp1252")
    )

    preview = notebook_step_preview(notebook_path)

    assert [step["title"] for step in preview] == ["读取数据"]


def test_notebook_step_preview_infers_titles_without_execution_for_headingless_notebook(
    tmp_path: Path,
):
    notebook_path = tmp_path / "model.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "RMC_SAMPLE_DF = sample_df\n"
                    "RMC_TARGET_COL = 'y'\n"
                    "RMC_ALGORITHM = 'lgb'\n"
                    "def RMC_SCORE_FN(df):\n"
                    "    return model.predict_proba(df)[:, 1]\n"
                ),
                nbformat.v4.new_code_cell("sample = pd.read_parquet('sample.parquet')"),
            ]
        ),
        notebook_path,
    )

    preview = notebook_step_preview(notebook_path)

    assert [step["title"] for step in preview] == ["Notebook 契约配置", "读取数据"]
    assert [step["id"] for step in preview] == ["cell-1", "cell-2"]
