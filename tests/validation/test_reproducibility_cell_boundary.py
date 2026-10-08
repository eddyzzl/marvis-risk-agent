"""Generated Notebook cells must share the deterministic score boundary."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.pipeline_cellgen import _build_reproducibility_cell_sources


@pytest.mark.parametrize("case", ["code_inf", "empty", "zero_sample", "pmml_inf", "healthy", "overflow", "large_equal"])
def test_generated_cells_reject_invalid_inputs_and_preserve_normal_results(tmp_path, monkeypatch, case):
    count = 0 if case == "empty" else 1
    code_value = float("inf") if case == "code_inf" else .5
    if case in {"overflow", "large_equal"}:
        code_value = 1e308
    score_path = tmp_path / "code_scores.csv"
    pd.DataFrame({"row_index": list(range(count)), "code_model_score": [code_value] * count}).to_csv(score_path, index=False)
    contract_path = tmp_path / "runtime_contract.json"
    contract_path.write_text(json.dumps({"target_col": "y", "code_model_scores_path": str(score_path),
                                         "score_decimal_places": 6, "pmml_output_field": "probability_1"}))
    output_path = tmp_path / "reproducibility.json"
    calls = []

    class Scorer:
        def score(self, frame):
            calls.append(len(frame))
            if case in {"overflow", "large_equal"}:
                return [-1e308 if case == "overflow" else 1e308] * len(frame)
            return [float("inf") if case in {"code_inf", "pmml_inf"} else .5] * len(frame)

    monkeypatch.setattr("marvis.validation.pmml_scoring.load_pmml_scorer", lambda *args, **kwargs: Scorer())
    sources = _build_reproducibility_cell_sources(
        package_root=Path(__file__).resolve().parents[2],
        task=SimpleNamespace(split_col="split", time_col="month"),
        settings=SimpleNamespace(random_sample_size=0 if case == "zero_sample" else 1,
                                 random_seed=42, bin_count=10),
        input_pmml_path=tmp_path / "unused.pmml", contract_meta_path=contract_path, output_path=output_path,
    )
    namespace = {"RMC_SAMPLE_DF": pd.DataFrame({"x": [1.] * count, "y": [0] * count})}

    def execute():
        for name, source in sources:
            exec(compile(source, f"<generated-{name}>", "exec"), namespace)

    if case in {"code_inf", "empty", "zero_sample"}:
        with pytest.raises(ValueError, match="non-finite|at least one sampled row"):
            execute()
        assert calls == [] and not output_path.exists()
    elif case == "overflow":
        with pytest.raises(ValueError, match="score difference is non-finite"):
            execute()
        assert calls == [1] and not output_path.exists()
    else:
        execute()
        result = json.loads(output_path.read_text())
        assert calls == [1]
        assert result["summary"]["status"] == ("fail" if case == "pmml_inf" else "pass")
        assert result["summary"]["mismatch_count"] == int(case == "pmml_inf")
        expected_score = None if case == "pmml_inf" else 1e308 if case == "large_equal" else .5
        assert result["rows"][0]["score_submitted_pmml"] == expected_score
        json.dumps(result, allow_nan=False)
