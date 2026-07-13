import json
from hashlib import sha256
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from marvis.notebook_contract import (
    NotebookContractError,
    build_contract_head_cell_source,
    build_contract_tail_cell_source,
    inspect_notebook_contract,
    load_runtime_contract,
    precheck_notebook_contract,
)


def test_precheck_requires_score_function_and_target_col():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("RMC_SAMPLE_DF = sample_df\nRMC_TARGET_COL = 'y'"),
            nbformat.v4.new_code_cell("print('no scoring function')"),
        ]
    )

    with pytest.raises(NotebookContractError) as excinfo:
        precheck_notebook_contract(notebook)

    assert "missing RMC_SCORE_FN" in str(excinfo.value)


def test_precheck_path_uses_one_byte_snapshot_for_parse_and_revision(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "model.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("RMC_TARGET_COL = 'y'")]
        ),
        path,
    )
    expected_bytes = path.read_bytes()
    original_read_bytes = Path.read_bytes
    reads = {"count": 0}

    def counted_read_bytes(candidate):
        if candidate.resolve() == path.resolve():
            reads["count"] += 1
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    with pytest.raises(NotebookContractError) as excinfo:
        precheck_notebook_contract(path)

    assert reads["count"] == 1
    assert excinfo.value.notebook_sha256 == sha256(expected_bytes).hexdigest()


def test_precheck_requires_top_level_score_function_binding_not_string_reference():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "print('RMC_SCORE_FN must be implemented here')\n"
                "# RMC_SCORE_FN is mentioned but not bound\n"
            ),
        ]
    )

    with pytest.raises(NotebookContractError) as excinfo:
        precheck_notebook_contract(notebook)

    assert "missing RMC_SCORE_FN" in str(excinfo.value)


def test_precheck_does_not_require_feature_list():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = 'Lightgbm'\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            )
        ]
    )

    result = precheck_notebook_contract(notebook)

    assert result.missing_names == []
    assert result.algorithm == "lgb"


def test_precheck_requires_algorithm_contract_or_model_params_algorithm():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            )
        ]
    )

    with pytest.raises(NotebookContractError) as excinfo:
        precheck_notebook_contract(notebook)

    assert "missing RMC_ALGORITHM" in str(excinfo.value)


def test_precheck_accepts_model_params_algorithm_alias_for_transition():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_MODEL_PARAMS = {'algorithm': 'XGBClassifier'}\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            )
        ]
    )

    result = precheck_notebook_contract(notebook)

    assert result.algorithm == "xgb"
    assert result.algorithm_raw == "XGBClassifier"
    assert result.algorithm_source == "RMC_MODEL_PARAMS.algorithm"
    summary = inspect_notebook_contract(notebook)
    assert summary["read_only"] is True
    assert summary["algorithm_raw"] == "XGBClassifier"
    assert summary["algorithm"] == "xgb"
    assert summary["algorithm_valid"] is True
    assert summary["contract_cell_indexes"] == [0]
    assert (
        "RMC_MODEL_PARAMS = {'algorithm': 'XGBClassifier'}"
        in summary["source_previews"][0]
    )


def test_precheck_accepts_xgbm_algorithm_alias():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = 'xgbm'\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            )
        ]
    )

    result = precheck_notebook_contract(notebook)

    assert result.algorithm == "xgb"
    assert result.algorithm_raw == "xgbm"
    assert result.algorithm_source == "RMC_ALGORITHM"
    summary = inspect_notebook_contract(notebook)
    assert summary["read_only"] is True
    assert summary["algorithm_raw"] == "xgbm"
    assert summary["algorithm"] == "xgb"
    assert summary["algorithm_valid"] is True
    assert summary["contract_cell_indexes"] == [0]
    assert "RMC_ALGORITHM = 'xgbm'" in summary["source_previews"][0]


def test_precheck_prefers_explicit_algorithm_over_legacy_model_params():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_MODEL_PARAMS = {'algorithm': 'random forest'}\n"
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = 'score_card'\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            )
        ]
    )

    result = precheck_notebook_contract(notebook)

    assert result.algorithm == "scorecard"


def test_precheck_rejects_unknown_algorithm_contract_value():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = 'random forest'\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            )
        ]
    )

    with pytest.raises(NotebookContractError) as excinfo:
        precheck_notebook_contract(notebook)

    assert "RMC_ALGORITHM value 'random forest' unsupported model algorithm" in str(
        excinfo.value
    )
    summary = inspect_notebook_contract(notebook)
    assert summary["algorithm_raw"] == "random forest"
    assert summary["algorithm"] is None
    assert summary["algorithm_valid"] is False
    assert "unsupported model algorithm" in summary["algorithm_error"]


def test_precheck_rejects_blank_algorithm_contract_value():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = ''\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            )
        ]
    )

    with pytest.raises(NotebookContractError, match="model algorithm is required"):
        precheck_notebook_contract(notebook)


def test_precheck_tolerates_ipython_syntax_outside_contract_cells():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("%matplotlib inline\n!echo preparing notebook\nx = 1"),
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = 'lr'\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            ),
        ]
    )

    result = precheck_notebook_contract(notebook)

    assert result.missing_names == []


def test_precheck_can_scan_contract_cell_with_leading_cell_magic():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "%%time\n"
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = 'lgbm'\n"
                "def RMC_SCORE_FN(df):\n"
                "    return [0.1] * len(df)\n"
            ),
        ]
    )

    result = precheck_notebook_contract(notebook)

    assert result.missing_names == []


def test_precheck_accepts_ipython_shell_escape_inside_contract_function():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_SAMPLE_DF = sample_df\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_ALGORITHM = 'xgb'\n"
                "def RMC_SCORE_FN(df):\n"
                "    !echo scoring\n"
            ),
        ]
    )

    result = precheck_notebook_contract(notebook)

    assert result.missing_names == []


def test_contract_head_cell_defines_paths(tmp_path: Path):
    source = build_contract_head_cell_source(
        sample_path=tmp_path / "sample.csv",
        contract_meta_path=tmp_path / "contract.json",
        code_scores_path=tmp_path / "scores.csv",
        feature_importance_path=tmp_path / "importance.csv",
        model_params_path=tmp_path / "params.json",
    )

    namespace: dict[str, str] = {}
    exec(source, namespace)

    assert namespace["RMC_SAMPLE_PATH"].endswith("sample.csv")
    assert namespace["RMC_CONTRACT_META_PATH"].endswith("contract.json")
    assert namespace["RMC_CODE_SCORES_PATH"].endswith("scores.csv")


def test_contract_head_cell_read_csv_falls_back_only_when_encoding_unspecified(
    tmp_path: Path,
):
    csv_path = tmp_path / "feature_importance.csv"
    csv_path.write_bytes(
        "feature,importance,分类\nx1,1.0,征信\n".encode("gb18030")
    )
    source = build_contract_head_cell_source(
        sample_path=tmp_path / "sample.csv",
        contract_meta_path=tmp_path / "contract.json",
        code_scores_path=tmp_path / "scores.csv",
        feature_importance_path=tmp_path / "importance.csv",
        model_params_path=tmp_path / "params.json",
    )
    original_read_csv = pd.read_csv
    namespace: dict = {}
    try:
        exec(source, namespace)

        with pytest.raises(UnicodeDecodeError):
            pd.read_csv(csv_path, encoding="utf-8")
        frame = pd.read_csv(csv_path)
    finally:
        pd.read_csv = original_read_csv

    assert frame.to_dict(orient="records") == [
        {"feature": "x1", "importance": 1.0, "分类": "征信"}
    ]


def test_contract_tail_cell_scores_and_writes_metadata(tmp_path: Path):
    sample_path = tmp_path / "sample.csv"
    meta_path = tmp_path / "contract.json"
    scores_path = tmp_path / "scores.csv"
    importance_path = tmp_path / "importance.csv"
    params_path = tmp_path / "params.json"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(sample_path, index=False)
    head = build_contract_head_cell_source(
        sample_path=sample_path,
        contract_meta_path=meta_path,
        code_scores_path=scores_path,
        feature_importance_path=importance_path,
        model_params_path=params_path,
    )
    tail = build_contract_tail_cell_source()
    namespace: dict = {}
    exec(head, namespace)
    exec(
        "\n".join(
            [
                "import pandas as pd",
                "RMC_SAMPLE_DF = pd.DataFrame({'x': [1.0, 2.0], 'y': [0, 1]})",
                "RMC_TARGET_COL = 'y'",
                "RMC_ALGORITHM = 'Lightgbm'",
                "RMC_PMML_OUTPUT_FIELD = 'probability_good'",
                "RMC_SCORE_DECIMAL_PLACES = 5",
                "def RMC_SCORE_FN(df):",
                "    return df['x'] / 10",
                "RMC_FEATURE_IMPORTANCE = pd.DataFrame({'feature': ['x'], '类别': ['征信'], 'importance': [1.0]})",
                "RMC_MODEL_PARAMS = {'algorithm': 'lr', 'C': 0.5}",
            ]
        ),
        namespace,
    )

    exec(tail, namespace)

    meta = load_runtime_contract(meta_path)
    scores = pd.read_csv(scores_path)
    assert meta.algorithm == "lgb"
    assert meta.target_col == "y"
    assert meta.pmml_output_field == "probability_good"
    assert meta.score_decimal_places == 5
    assert meta.code_model_scores_path == scores_path
    assert scores["row_index"].tolist() == [0, 1]
    assert scores["code_model_score"].tolist() == [0.1, 0.2]
    importance = pd.read_csv(importance_path)
    assert importance["feature"].tolist() == ["x"]
    assert importance["category"].tolist() == ["征信"]
    assert json.loads(params_path.read_text(encoding="utf-8")) == {
        "algorithm": "lr",
        "C": 0.5,
    }


def test_contract_tail_and_loader_preserve_zero_score_decimal_places(tmp_path: Path):
    sample_path = tmp_path / "sample.csv"
    meta_path = tmp_path / "contract.json"
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame({"x": [1.0], "y": [0]}).to_csv(sample_path, index=False)
    namespace: dict = {}
    exec(
        build_contract_head_cell_source(
            sample_path=sample_path,
            contract_meta_path=meta_path,
            code_scores_path=scores_path,
            feature_importance_path=tmp_path / "importance.csv",
            model_params_path=tmp_path / "params.json",
        ),
        namespace,
    )
    exec(
        "\n".join(
            [
                "import pandas as pd",
                "RMC_SAMPLE_DF = pd.DataFrame({'x': [1.0], 'y': [0]})",
                "RMC_TARGET_COL = 'y'",
                "RMC_ALGORITHM = 'lr'",
                "RMC_SCORE_DECIMAL_PLACES = 0",
                "def RMC_SCORE_FN(df):",
                "    return df['x'] / 10",
            ]
        ),
        namespace,
    )

    exec(build_contract_tail_cell_source(), namespace)

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["score_decimal_places"] == 0
    assert load_runtime_contract(meta_path).score_decimal_places == 0


def test_contract_tail_cell_is_self_contained_for_legacy_kernels():
    source = build_contract_tail_cell_source()

    assert "from marvis" not in source
    assert "import marvis" not in source
    assert "def _rmc_normalize_algorithm" in source
    assert "_rmc_algorithm = _rmc_normalize_algorithm(_rmc_algorithm_raw)" in source


def test_contract_tail_cell_uses_positional_row_index_for_custom_sample_index(
    tmp_path: Path,
):
    sample_path = tmp_path / "sample.csv"
    meta_path = tmp_path / "contract.json"
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(sample_path, index=False)
    namespace: dict = {}
    exec(
        build_contract_head_cell_source(
            sample_path=sample_path,
            contract_meta_path=meta_path,
            code_scores_path=scores_path,
            feature_importance_path=tmp_path / "importance.csv",
            model_params_path=tmp_path / "params.json",
        ),
        namespace,
    )
    exec(
        "\n".join(
            [
                "import pandas as pd",
                "RMC_SAMPLE_DF = pd.DataFrame({'x': [1.0, 2.0], 'y': [0, 1]}, index=['a', 'b'])",
                "RMC_TARGET_COL = 'y'",
                "RMC_ALGORITHM = 'lgb'",
                "def RMC_SCORE_FN(df):",
                "    return df['x'] / 10",
            ]
        ),
        namespace,
    )

    exec(build_contract_tail_cell_source(), namespace)

    scores = pd.read_csv(scores_path)
    assert scores["row_index"].tolist() == [0, 1]


def test_contract_tail_cell_accepts_xgbm_algorithm_alias(tmp_path: Path):
    sample_path = tmp_path / "sample.csv"
    meta_path = tmp_path / "contract.json"
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(sample_path, index=False)
    namespace: dict = {}
    exec(
        build_contract_head_cell_source(
            sample_path=sample_path,
            contract_meta_path=meta_path,
            code_scores_path=scores_path,
            feature_importance_path=tmp_path / "importance.csv",
            model_params_path=tmp_path / "params.json",
        ),
        namespace,
    )
    exec(
        "\n".join(
            [
                "import pandas as pd",
                "RMC_SAMPLE_DF = pd.DataFrame({'x': [1.0, 2.0], 'y': [0, 1]})",
                "RMC_TARGET_COL = 'y'",
                "RMC_ALGORITHM = 'xgbm'",
                "def RMC_SCORE_FN(df):",
                "    return df['x'] / 10",
            ]
        ),
        namespace,
    )

    exec(build_contract_tail_cell_source(), namespace)

    assert json.loads(meta_path.read_text(encoding="utf-8"))["algorithm"] == "xgb"


def test_contract_tail_cell_rejects_blank_algorithm(tmp_path: Path):
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}).to_csv(sample_path, index=False)
    namespace: dict = {}
    exec(
        build_contract_head_cell_source(
            sample_path=sample_path,
            contract_meta_path=tmp_path / "contract.json",
            code_scores_path=tmp_path / "scores.csv",
            feature_importance_path=tmp_path / "importance.csv",
            model_params_path=tmp_path / "params.json",
        ),
        namespace,
    )
    exec(
        "\n".join(
            [
                "import pandas as pd",
                "RMC_SAMPLE_DF = pd.DataFrame({'x': [1.0, 2.0], 'y': [0, 1]})",
                "RMC_TARGET_COL = 'y'",
                "RMC_ALGORITHM = ''",
                "def RMC_SCORE_FN(df):",
                "    return df['x'] / 10",
            ]
        ),
        namespace,
    )

    with pytest.raises(ValueError, match="model algorithm is required"):
        exec(build_contract_tail_cell_source(), namespace)


@pytest.mark.parametrize("payload", [{"target_col": "y"}, {"target_col": "y", "algorithm": ""}])
def test_load_runtime_contract_rejects_missing_or_blank_algorithm(
    tmp_path: Path,
    payload: dict,
):
    contract_path = tmp_path / "runtime_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "code_model_scores_path": str(tmp_path / "scores.csv"),
                **payload,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model algorithm is required"):
        load_runtime_contract(contract_path)


def test_contract_tail_cell_rejects_bad_score_length(tmp_path: Path):
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame({"placeholder": [1, 2]}).to_csv(sample_path, index=False)
    namespace: dict = {}
    exec(
        build_contract_head_cell_source(
            sample_path=sample_path,
            contract_meta_path=tmp_path / "contract.json",
            code_scores_path=tmp_path / "scores.csv",
            feature_importance_path=tmp_path / "importance.csv",
            model_params_path=tmp_path / "params.json",
        ),
        namespace,
    )
    exec(
        "import pandas as pd\n"
        "RMC_SAMPLE_DF = pd.DataFrame({'x': [1.0, 2.0], 'y': [0, 1]})\n"
        "RMC_TARGET_COL = 'y'\n"
        "RMC_ALGORITHM = 'lgb'\n"
        "def RMC_SCORE_FN(df):\n"
        "    return [0.1]\n",
        namespace,
    )

    with pytest.raises(ValueError, match="RMC_SCORE_FN returned 1 scores for 2 rows"):
        exec(build_contract_tail_cell_source(), namespace)
