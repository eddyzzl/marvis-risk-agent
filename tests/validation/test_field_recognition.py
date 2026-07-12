from __future__ import annotations

import builtins
from hashlib import sha256
import math
from pathlib import Path

import nbformat
import pytest

from marvis.validation import field_recognition
from marvis.validation.field_recognition import recognize_notebook_fields


def _write_notebook(path: Path, *, cells: list[object], encoding: str = "utf-8") -> Path:
    notebook = nbformat.v4.new_notebook(cells=cells)
    path.write_bytes(nbformat.writes(notebook).encode(encoding))
    return path


def test_recognizer_extracts_rmc_literals_without_executing_code(
    tmp_path, monkeypatch
):
    marker = tmp_path / "executed"
    notebook = _write_notebook(
        tmp_path / "model.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                f"open({str(marker)!r}, 'w')\n"
                "__import__('pathlib').Path('also-executed').touch()\n"
                "get_ipython().system('touch shell-executed')\n"
                "RMC_TARGET_COL = 'y'\n"
                "RMC_SPLIT_COL = 'model_flag'\n"
                "RMC_TIME_COL = 'loan_month'\n"
                "RMC_PMML_OUTPUT_FIELD = 'probability(1)'\n"
                "RMC_MODEL_PARAMS = {'learning_rate': 0.05, 'n_estimators': 300}\n"
            )
        ],
    )
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Notebook code attempted to open a file")
        ),
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates["target_col"][0].value == "y"
    assert result.candidates["split_col"][0].value == "model_flag"
    assert result.candidates["time_col"][0].value == "loan_month"
    assert result.candidates["pmml_output_field"][0].value == "probability(1)"
    assert result.candidates["model_params"][0].value == {
        "learning_rate": 0.05,
        "n_estimators": 300,
    }
    assert not marker.exists()
    assert not (Path.cwd() / "also-executed").exists()
    assert not (Path.cwd() / "shell-executed").exists()


def test_gb18030_notebook_and_leading_cell_magic_keep_alias_assignments_visible(
    tmp_path,
):
    notebook = _write_notebook(
        tmp_path / "中文模型.ipynb",
        encoding="gb18030",
        cells=[
            nbformat.v4.new_code_cell(
                "%%time\n"
                "LABEL = '坏客户丂'\n"
                "SPLIT_COL = '样本类型'\n"
                "TIME_COL = '申请月份'\n"
                "RMC_PMML_OUTPUT_FIELD = 'probability_1'\n"
                "RMC_ALGORITHM = 'xgb'\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates["target_col"][0].value == "坏客户丂"
    assert result.candidates["split_col"][0].value == "样本类型"
    assert result.candidates["time_col"][0].value == "申请月份"
    assert result.candidates["pmml_output_field"][0].value == "probability_1"
    assert result.candidates["algorithm"][0].value == "xgb"
    assert result.conflicts == ()


def test_lowercase_target_aliases_and_one_hop_literal_history_are_preserved(tmp_path):
    notebook = _write_notebook(
        tmp_path / "legacy-aliases.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "label = 'bad_flag'\n"
                "candidate = 'first_target'\n"
                "candidate = 'second_target'\n"
                "target = candidate\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "bad_flag",
        "first_target",
        "second_target",
    ]
    assert result.conflicts == ()


def test_literal_lowercase_label_keyword_and_nested_estimator_are_evidence_only(
    tmp_path,
):
    notebook = _write_notebook(
        tmp_path / "legacy-helper.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def fit_model(params, train, test, label='y'):\n"
                "    model = LGBMClassifier(n_jobs=-1, **params)\n"
                "    return model\n"
                "result = fit_model(params, train, test, label='y')\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "y"
    ]
    assert [candidate.value for candidate in result.candidates["algorithm"]] == [
        "LGBMClassifier"
    ]
    assert "model_params" not in result.candidates
    assert result.conflicts == ()


def test_nested_explicit_assign_and_annassign_fields_are_ordered_evidence(tmp_path):
    notebook = _write_notebook(
        tmp_path / "nested-contract.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "RMC_TIME_COL = 'month'\n"
                "def config():\n"
                "    RMC_TARGET_COL = 'y'\n"
                "    RMC_SPLIT_COL: str = 'split'\n"
                "    return RMC_TARGET_COL\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates["time_col"][0].evidence[0].source_kind == "rmc_literal"
    assert result.candidates["target_col"][0].value == "y"
    assert result.candidates["split_col"][0].value == "split"
    assert result.candidates["target_col"][0].evidence[0].source_kind == (
        "nested_rmc_literal"
    )
    assert result.candidates["split_col"][0].evidence[0].source_kind == (
        "nested_rmc_literal"
    )
    assert result.conflicts == ()


def test_nested_literal_estimator_params_support_scope_local_dict_without_leaking(
    tmp_path,
):
    notebook = _write_notebook(
        tmp_path / "nested-estimator.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def build():\n"
                "    local_params = {'num_leaves': 31}\n"
                "    model = LGBMClassifier(**local_params, n_estimators=120)\n"
                "    return model\n"
                "RMC_MODEL_PARAMS = model.get_params()\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["algorithm"]] == [
        "LGBMClassifier"
    ]
    assert [candidate.value for candidate in result.candidates["model_params"]] == [
        {"num_leaves": 31, "n_estimators": 120}
    ]
    assert result.candidates["model_params"][0].evidence[0].source_kind == (
        "nested_estimator_constructor"
    )
    assert any("no complete static literal binding" in item for item in result.diagnostics)


def test_nested_dynamic_estimator_params_are_never_partially_emitted(tmp_path):
    notebook = _write_notebook(
        tmp_path / "nested-dynamic-estimator.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def build():\n"
                "    model = LGBMClassifier(num_leaves=load_num_leaves(), n_estimators=120)\n"
                "    return model\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates["algorithm"][0].value == "LGBMClassifier"
    assert "model_params" not in result.candidates
    assert any("requires confirmation" in item for item in result.diagnostics)


def test_multiple_equal_priority_rmc_values_remain_ordered_candidates_not_conflicts(
    tmp_path,
):
    notebook = _write_notebook(
        tmp_path / "ambiguous.ipynb",
        cells=[
            nbformat.v4.new_code_cell("RMC_TARGET_COL = 'old_target'"),
            nbformat.v4.new_code_cell("RMC_TARGET_COL = 'new_target'"),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "old_target",
        "new_target",
    ]
    assert [
        candidate.evidence[0].notebook_cell
        for candidate in result.candidates["target_col"]
    ] == [0, 1]
    assert result.conflicts == ()


def test_static_symbol_table_resolves_dict_references_estimator_kwargs_and_get_params(
    tmp_path,
):
    notebook = _write_notebook(
        tmp_path / "params.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "xgb_base = {'max_depth': 4, 'learning_rate': 0.05}\n"
                "xgb_model = xgb.XGBClassifier(**xgb_base, n_estimators=200)\n"
                "RMC_MODEL_PARAMS = xgb_model.get_params()\n"
                "lgb_base = {'num_leaves': 31, 'n_estimators': 150}\n"
                "lgb_model = LGBMRegressor(**lgb_base)\n"
                "MODEL_HYPERPARAMETERS = lgb_base\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)
    params = [candidate.value for candidate in result.candidates["model_params"]]

    assert {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200} in params
    assert {"num_leaves": 31, "n_estimators": 150} in params
    assert [candidate.value for candidate in result.candidates["algorithm"]] == [
        "XGBClassifier",
        "LGBMRegressor",
    ]


def test_direct_known_literal_dict_reference_is_a_model_params_candidate(tmp_path):
    notebook = _write_notebook(
        tmp_path / "dict-ref.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "params = {'objective': 'binary', 'max_depth': 3}\n"
                "RMC_MODEL_PARAMS = params\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["model_params"]] == [
        {"objective": "binary", "max_depth": 3}
    ]


def test_dynamic_estimator_parts_are_not_partially_merged(tmp_path):
    notebook = _write_notebook(
        tmp_path / "dynamic.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "base = {'max_depth': 4}\n"
                "model = XGBClassifier(**base, n_estimators=load_count())\n"
                "RMC_MODEL_PARAMS = model.get_params()\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "model_params" not in result.candidates
    assert result.candidates["algorithm"][0].value == "XGBClassifier"
    assert any("confirmation" in item for item in result.diagnostics)
    assert result.conflicts == ()


@pytest.mark.parametrize(
    "source",
    [
        "RMC_MODEL_PARAMS = ('depth', 4)",
        "RMC_MODEL_PARAMS = {1: 'not-a-json-object'}",
        "RMC_MODEL_PARAMS = {'bad': float('nan')}",
        "RMC_MODEL_PARAMS = {'bad': b'bytes'}",
        "RMC_MODEL_PARAMS = {'bad': 1 + 2j}",
    ],
)
def test_non_json_dynamic_and_nonfinite_model_literals_are_rejected(tmp_path, source):
    notebook = _write_notebook(
        tmp_path / "bad-literal.ipynb",
        cells=[nbformat.v4.new_code_cell(source)],
    )

    result = recognize_notebook_fields(notebook)

    assert "model_params" not in result.candidates
    assert result.diagnostics
    assert all(len(item) <= field_recognition.MAX_DIAGNOSTIC_CHARS for item in result.diagnostics)
    assert result.conflicts == ()


def test_markdown_comments_and_saved_outputs_use_only_anchored_allowlisted_evidence(
    tmp_path,
):
    output = nbformat.v4.new_output(
        "display_data",
        data={"application/json": {"RMC_TIME_COL": "output_month"}},
    )
    rejected_output = nbformat.v4.new_output(
        "display_data",
        data={
            "application/json": {
                "RMC_TIME_COL": "must_not_be_partial",
                "unknown": "value",
            },
            "text/html": "<script>bad()</script>",
        },
    )
    notebook = _write_notebook(
        tmp_path / "evidence.ipynb",
        cells=[
            nbformat.v4.new_markdown_cell(
                "The prose mentions RMC_TARGET_COL = 'ignored'.\n"
                "```rmc\nRMC_TARGET_COL: 'markdown_y'\n```"
            ),
            nbformat.v4.new_code_cell(
                "# RMC_SPLIT_COL = 'comment_split'\n"
                "# arbitrary RMC_TIME_COL = 'ignored'\n"
                "pass",
                outputs=[output, rejected_output],
            ),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "markdown_y"
    ]
    assert [candidate.value for candidate in result.candidates["split_col"]] == [
        "comment_split"
    ]
    assert [candidate.value for candidate in result.candidates["time_col"]] == [
        "output_month"
    ]
    source_kinds = {
        candidate.evidence[0].source_kind
        for candidates in result.candidates.values()
        for candidate in candidates
    }
    assert source_kinds == {"markdown", "comment", "saved_output"}


def test_saved_text_output_requires_anchored_literal_lines(tmp_path):
    notebook = _write_notebook(
        tmp_path / "text-output.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "pass",
                outputs=[
                    nbformat.v4.new_output(
                        "execute_result",
                        execution_count=1,
                        data={
                            "text/plain": [
                                "RMC_PMML_OUTPUT_FIELD = 'probability_1'\n",
                                "prose RMC_TARGET_COL = 'ignored'",
                            ]
                        },
                    )
                ],
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates["pmml_output_field"][0].value == "probability_1"
    assert "target_col" not in result.candidates


def test_malformed_cells_and_oversized_saved_outputs_are_bounded_diagnostics_not_conflicts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(field_recognition, "MAX_SAVED_OUTPUT_BYTES", 32)
    notebook = _write_notebook(
        tmp_path / "malformed.ipynb",
        cells=[
            nbformat.v4.new_code_cell("RMC_TARGET_COL = ["),
            nbformat.v4.new_code_cell(
                "pass",
                outputs=[
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            "application/json": {
                                "RMC_TARGET_COL": "x" * 100,
                            }
                        },
                    )
                ],
            ),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "target_col" not in result.candidates
    assert result.conflicts == ()
    assert any("parse" in item.casefold() for item in result.diagnostics)
    assert any("output" in item.casefold() for item in result.diagnostics)
    assert all(len(item) <= field_recognition.MAX_DIAGNOSTIC_CHARS for item in result.diagnostics)


def test_repeated_saved_outputs_report_each_limit_once(tmp_path, monkeypatch):
    monkeypatch.setattr(field_recognition, "MAX_SAVED_OUTPUT_BYTES", 16)
    outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"application/json": {"RMC_TARGET_COL": "x" * 20}},
        )
        for _ in range(10)
    ]
    notebook = _write_notebook(
        tmp_path / "many-outputs.ipynb",
        cells=[nbformat.v4.new_code_cell("pass", outputs=outputs)],
    )

    result = recognize_notebook_fields(notebook)

    assert sum("output exceeds" in item for item in result.diagnostics) == 1


def test_notebook_is_read_once_as_an_immutable_byte_snapshot(tmp_path, monkeypatch):
    path = _write_notebook(
        tmp_path / "snapshot.ipynb",
        cells=[nbformat.v4.new_code_cell("RMC_TARGET_COL = 'first'")],
    )
    original_raw = path.read_bytes()
    replacement = nbformat.writes(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("RMC_TARGET_COL = 'second'")]
        )
    ).encode()
    original_read_bytes = Path.read_bytes
    read_count = 0

    def read_then_replace(self):
        nonlocal read_count
        read_count += 1
        raw = original_read_bytes(self)
        self.write_bytes(replacement)
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)

    result = recognize_notebook_fields(path)

    assert read_count == 1
    assert result.notebook_sha256 == sha256(original_raw).hexdigest()
    assert result.candidates["target_col"][0].value == "first"


def test_real_corpus_limits_leave_headroom_and_structural_bounds_are_enforced(
    tmp_path, monkeypatch
):
    assert field_recognition.MAX_NOTEBOOK_BYTES >= 6_000_000
    assert field_recognition.MAX_NOTEBOOK_CELLS >= 206
    assert field_recognition.MAX_NOTEBOOK_CELL_CHARS >= 75_000

    path = _write_notebook(
        tmp_path / "bounded.ipynb",
        cells=[nbformat.v4.new_code_cell("RMC_TARGET_COL = 'y'")],
    )
    monkeypatch.setattr(field_recognition, "MAX_NOTEBOOK_BYTES", 8)
    with pytest.raises(ValueError, match="byte limit"):
        recognize_notebook_fields(path)


def test_oversized_cell_is_a_non_confirmable_structural_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(field_recognition, "MAX_NOTEBOOK_CELL_CHARS", 20)
    notebook = _write_notebook(
        tmp_path / "long-cell.ipynb",
        cells=[nbformat.v4.new_code_cell("RMC_TARGET_COL = 'a-long-name'")],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates == {}
    assert result.conflicts == ("cell 0 exceeds static inspection character limit",)


def test_literal_recursion_and_evidence_limits_do_not_emit_unbounded_payloads(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(field_recognition, "MAX_LITERAL_DEPTH", 3)
    monkeypatch.setattr(field_recognition, "MAX_EVIDENCE_COUNT", 2)
    deep = "[[[[['x']]]]]"
    notebook = _write_notebook(
        tmp_path / "bounded-evidence.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                f"RMC_MODEL_PARAMS = {{'deep': {deep}}}\n"
                "RMC_TARGET_COL = 'a'\n"
                "RMC_SPLIT_COL = 'b'\n"
                "RMC_TIME_COL = 'c'\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "model_params" not in result.candidates
    evidence_count = sum(
        len(candidate.evidence)
        for candidates in result.candidates.values()
        for candidate in candidates
    )
    assert evidence_count == 2
    assert any("evidence limit" in item for item in result.diagnostics)


def test_multi_root_source_labels_are_conservatively_removed_with_diagnostic(tmp_path):
    notebook = _write_notebook(
        tmp_path / "concat-sources.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "development['source'] = 'train'\n"
                "holdout['source'] = 'oot'\n"
                "sample = pd.concat([development, holdout])\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.transformations == ()
    assert any("multiple dataframe roots" in item for item in result.diagnostics)
    assert result.conflicts == ()


def test_nonfinite_saved_json_is_rejected_even_if_nbformat_allows_it(tmp_path):
    notebook = _write_notebook(
        tmp_path / "nonfinite-output.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "pass",
                outputs=[
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            "application/json": {
                                "RMC_MODEL_PARAMS": {"bad": math.nan}
                            }
                        },
                    )
                ],
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "model_params" not in result.candidates
    assert result.diagnostics


def test_model_params_candidates_must_be_json_objects_in_every_evidence_path(
    tmp_path,
):
    notebook = _write_notebook(
        tmp_path / "model-params-shapes.ipynb",
        cells=[
            nbformat.v4.new_markdown_cell("RMC_MODEL_PARAMS: [1, 2]"),
            nbformat.v4.new_code_cell(
                "RMC_MODEL_PARAMS = ['not', 'an', 'object']\n"
                "# MODEL_HYPERPARAMETERS = 42\n",
                outputs=[
                    nbformat.v4.new_output(
                        "display_data",
                        data={"application/json": {"RMC_MODEL_PARAMS": [3, 4]}},
                    )
                ],
            ),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "model_params" not in result.candidates
    assert sum("model_params candidate must be a JSON object" in item for item in result.diagnostics) == 4


def test_label_keyword_requires_a_statically_defined_simple_function(tmp_path):
    notebook = _write_notebook(
        tmp_path / "keyword-safety.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def lgbfit(data, label='y'):\n"
                "    return data\n"
                "lgbfit(frame, label='accepted')\n"
                "plt.plot(x, y, label='ROC must not be a target')\n"
                "unknown_fit(frame, label='also rejected')\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "accepted"
    ]
    assert result.candidates["target_col"][0].evidence[0].source_kind == (
        "alias_keyword"
    )


def test_repeated_weak_alias_calls_cannot_exhaust_later_strong_rmc_evidence(tmp_path):
    weak_calls = "\n".join(
        "lgbfit(frame, label='weak_y')" for _ in range(2_000)
    )
    notebook = _write_notebook(
        tmp_path / "weak-quota.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def lgbfit(data, label='y'):\n    return data"
            ),
            nbformat.v4.new_code_cell(weak_calls),
            nbformat.v4.new_code_cell("RMC_TARGET_COL = 'strong_y'"),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "strong_y",
        "weak_y",
    ]


def test_candidate_field_shapes_are_validated_for_every_source(tmp_path):
    notebook = _write_notebook(
        tmp_path / "field-shapes.ipynb",
        cells=[
            nbformat.v4.new_markdown_cell("RMC_TARGET_COL: ['not-a-column']"),
            nbformat.v4.new_code_cell(
                "RMC_ALGORITHM = 7\n"
                "RMC_POSITIVE_LABEL = {'not': 'a scalar'}\n"
                "RMC_SPLIT_VALUE_MAPPING = ['not', 'an', 'object']\n"
                "RMC_TIME_COL = 'valid_time'\n"
                "RMC_NEGATIVE_LABEL = 0\n"
                "# RMC_PMML_OUTPUT_FIELD = []\n",
                outputs=[
                    nbformat.v4.new_output(
                        "display_data",
                        data={"application/json": {"RMC_TIME_GRANULARITY": {}}},
                    )
                ],
            ),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert set(result.candidates) == {"time_col", "negative_label"}
    assert result.candidates["time_col"][0].value == "valid_time"
    assert result.candidates["negative_label"][0].value == 0
    assert len(result.diagnostics) >= 5
    assert all("invalid" in item or "must" in item for item in result.diagnostics)


def test_candidate_order_uses_priority_then_source_position(tmp_path):
    notebook = _write_notebook(
        tmp_path / "candidate-order.ipynb",
        cells=[
            nbformat.v4.new_markdown_cell("RMC_TARGET_COL: 'markdown_y'"),
            nbformat.v4.new_code_cell("LABEL = 'alias_y'"),
            nbformat.v4.new_code_cell(
                "RMC_TARGET_COL = 'first_rmc'\n"
                "RMC_TARGET_COL = 'second_rmc'\n"
            ),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "first_rmc",
        "second_rmc",
        "alias_y",
        "markdown_y",
    ]


def test_nested_rmc_priority_and_equal_priority_source_order_are_stable(tmp_path):
    notebook = _write_notebook(
        tmp_path / "nested-candidate-order.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def config():\n"
                "    RMC_TARGET_COL = 'nested_first'\n"
                "    RMC_TARGET_COL = 'nested_second'\n"
                "LABEL = 'top_alias'\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "nested_first",
        "nested_second",
        "top_alias",
    ]


def test_mutually_exclusive_branches_do_not_leak_bindings_and_invalidate_afterward(
    tmp_path,
):
    notebook = _write_notebook(
        tmp_path / "branch-scope.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def config(flag):\n"
                "    candidate = 'outer_y'\n"
                "    if flag:\n"
                "        candidate = 'then_y'\n"
                "    else:\n"
                "        RMC_TARGET_COL = candidate\n"
                "    RMC_SPLIT_COL = candidate\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "outer_y"
    ]
    assert "split_col" not in result.candidates


def test_dynamic_binding_targets_do_not_reuse_stale_outer_literals(tmp_path):
    notebook = _write_notebook(
        tmp_path / "dynamic-bindings.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def config(items, manager, subject):\n"
                "    stale = 'outer'\n"
                "    for stale in items:\n"
                "        RMC_TARGET_COL = stale\n"
                "    handle = 'outer'\n"
                "    with manager as handle:\n"
                "        RMC_SPLIT_COL = handle\n"
                "    problem = 'outer'\n"
                "    try:\n"
                "        run()\n"
                "    except Exception as problem:\n"
                "        RMC_TIME_COL = problem\n"
                "    captured = 'outer'\n"
                "    match subject:\n"
                "        case {'value': captured}:\n"
                "            RMC_PMML_OUTPUT_FIELD = captured\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert not {
        "target_col",
        "split_col",
        "time_col",
        "pmml_output_field",
    } & set(result.candidates)


def test_methods_do_not_inherit_bare_class_local_bindings(tmp_path):
    notebook = _write_notebook(
        tmp_path / "class-scope.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "class Config:\n"
                "    candidate = 'class_y'\n"
                "    def fields(self):\n"
                "        RMC_TARGET_COL = candidate\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "target_col" not in result.candidates
    assert any("unknown static symbol candidate" in item for item in result.diagnostics)


def test_weak_markdown_evidence_cannot_starve_later_explicit_rmc(tmp_path):
    markdown = "\n".join(
        f"RMC_TARGET_COL: 'weak_{index}'" for index in range(2_000)
    )
    notebook = _write_notebook(
        tmp_path / "markdown-starvation.ipynb",
        cells=[
            nbformat.v4.new_markdown_cell(markdown),
            nbformat.v4.new_code_cell("RMC_TARGET_COL = 'strong_y'"),
        ],
    )

    result = recognize_notebook_fields(notebook)

    values = [candidate.value for candidate in result.candidates["target_col"]]
    assert values[0] == "strong_y"
    assert len(values) <= 513
    assert any("secondary evidence limit" in item for item in result.diagnostics)


def test_transformations_across_cells_with_different_roots_are_cleared(tmp_path):
    notebook = _write_notebook(
        tmp_path / "cross-cell-roots.ipynb",
        cells=[
            nbformat.v4.new_code_cell("left['month'] = left['date'].str[:7]"),
            nbformat.v4.new_code_cell("right['split'] = 'oot'"),
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.transformations == ()
    assert any("multiple dataframe roots" in item for item in result.diagnostics)


def test_dynamic_reassignment_invalidates_top_level_literal_binding(tmp_path):
    notebook = _write_notebook(
        tmp_path / "stale-top-level.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "candidate = 'stale_y'\n"
                "candidate = load_target()\n"
                "RMC_TARGET_COL = candidate\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "target_col" not in result.candidates
    assert any(
        "candidate" in item and "requires confirmation" in item
        for item in result.diagnostics
    )


def test_dynamic_reassignment_invalidates_nested_literal_binding(tmp_path):
    notebook = _write_notebook(
        tmp_path / "stale-nested.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "def config():\n"
                "    candidate = 'stale_y'\n"
                "    candidate = load_target()\n"
                "    RMC_TARGET_COL = candidate\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert "target_col" not in result.candidates
    assert any(
        "candidate" in item and "requires confirmation" in item
        for item in result.diagnostics
    )


def test_static_history_restarts_after_dynamic_reassignment(tmp_path):
    notebook = _write_notebook(
        tmp_path / "history-reset.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "candidate = 'a'\n"
                "candidate = load_target()\n"
                "candidate = 'b'\n"
                "RMC_TARGET_COL = candidate\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [candidate.value for candidate in result.candidates["target_col"]] == [
        "b"
    ]


def test_dynamic_estimator_reassignment_is_not_reused_by_later_get_params(tmp_path):
    notebook = _write_notebook(
        tmp_path / "stale-estimator.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "model = XGBClassifier(max_depth=3)\n"
                "model = load_model()\n"
                "RMC_MODEL_PARAMS = model.get_params()\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert [
        candidate.evidence[0].source_kind
        for candidate in result.candidates["model_params"]
    ] == ["estimator_constructor"]
    assert any("no complete static literal binding" in item for item in result.diagnostics)


def test_rhs_resolves_before_lhs_binding_is_replaced(tmp_path):
    notebook = _write_notebook(
        tmp_path / "self-assignment.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "candidate = 'y'\n"
                "candidate = candidate\n"
                "model = XGBClassifier(max_depth=3)\n"
                "saved_params = model.get_params()\n"
                "RMC_TARGET_COL = candidate\n"
                "RMC_MODEL_PARAMS = saved_params\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates["target_col"][0].value == "y"
    assert {"max_depth": 3} in [
        candidate.value for candidate in result.candidates["model_params"]
    ]
    assert any(
        candidate.value == {"max_depth": 3}
        and candidate.evidence[0].source_kind == "rmc_literal"
        for candidate in result.candidates["model_params"]
    )


def test_ten_thousand_irrelevant_assignments_remain_bounded_and_keep_relevant_field(
    tmp_path,
):
    source = "\n".join(f"noise_{index} = {index}" for index in range(10_000))
    notebook = _write_notebook(
        tmp_path / "large-ast.ipynb",
        cells=[
            nbformat.v4.new_code_cell(source + "\nRMC_TARGET_COL = 'bounded_y'\n")
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates["target_col"][0].value == "bounded_y"
    assert not any("AST node limit" in item for item in result.conflicts)


def test_over_budget_ast_becomes_bounded_structural_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(field_recognition, "MAX_AST_NODES", 20, raising=False)
    notebook = _write_notebook(
        tmp_path / "over-budget-ast.ipynb",
        cells=[
            nbformat.v4.new_code_cell(
                "\n".join(f"noise_{index} = {index}" for index in range(20))
                + "\nRMC_TARGET_COL = 'must_not_be_collected'\n"
            )
        ],
    )

    result = recognize_notebook_fields(notebook)

    assert result.candidates == {}
    assert result.conflicts == ("cell 0 exceeds static AST node limit",)
    assert all(len(item) <= field_recognition.MAX_DIAGNOSTIC_CHARS for item in result.conflicts)
