"""The validation adjudicator's recomputation is independent of production code."""
import ast
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from marvis.orchestrator.eval import runtime_validation_independent as ind

MODULE = Path(ind.__file__)
FORBIDDEN = ("marvis", "numpy", "pandas", "scipy", "sklearn")


def test_module_imports_nothing_from_production_or_numeric_stacks():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports would pull the eval package"
            names.append(node.module)
    assert names and not [n for n in names if n.split(".")[0] in FORBIDDEN]


def test_standalone_load_leaves_production_modules_unimported():
    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('independent', {str(MODULE)!r})\n"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "module.independent_pmml_regression_scores\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'marvis', 'numpy', 'pandas', 'scipy', 'sklearn'})))\n"
    )
    loaded = json.loads(subprocess.run([sys.executable, "-I", "-c", code], check=True,
                                       capture_output=True, text=True).stdout)
    assert loaded == []


def _sample(seed=7, n=300):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        split = ("train", "test", "oot")[i % 3]
        month = ("202601", "202602", "202604")[(i // 3) % 3]
        rows.append({"label": int(rng.random() < .3), "split": split, "date": month + "01"})
    # Ties and a negatively associated score exercise ordering and bin reversal.
    scores = [round(float(rng.random()), 2) - .4 * r["label"] for r in rows]
    return rows, scores


def _close(actual, expected, path=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _close(a, b, f"{path}[{index}]")
    elif isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        assert actual == expected or math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_independent_effectiveness_equals_installed_production_on_synthetic_rows(seed):
    from marvis.validation.config import ValidationConfig
    from marvis.validation.effectiveness import run_effectiveness

    rows, scores = _sample(seed)
    frame = pd.DataFrame({"__target__": [r["label"] for r in rows], "__split__": [r["split"] for r in rows],
                          "__time__": [r["date"] for r in rows], "__pmml_score__": scores})
    config = ValidationConfig(target_col="__target__", score_col="__pmml_score__", split_col="__split__",
                              time_col="__time__", bin_count=10)
    production = json.loads(json.dumps(asdict(run_effectiveness(sample=frame, config=config))))
    _close(json.loads(json.dumps(ind.independent_effectiveness(rows, scores, bin_count=10))), production)


def test_quantile_matches_numpy_linear_interpolation_bitwise():
    rng = np.random.default_rng(3)
    for size in (3, 7, 60, 181):
        values = sorted(float(v) for v in rng.normal(size=size))
        for q in np.linspace(0, 1, 11):
            assert ind._quantile(values, float(q)) == float(np.quantile(values, q))


def test_pmml_subset_scores_and_rejects_unsupported_structures():
    from marvis.orchestrator.eval.runtime_validation_case import _PMML

    scores = ind.independent_pmml_regression_scores(_PMML.encode(), [{"x1": "1.0", "x2": "0.5"}], "probability_1")
    assert scores == [pytest.approx(1 / (1 + math.exp(-0.5)), rel=1e-15)]
    with pytest.raises(ind.IndependentUnsupported, match="missing PMML input"):
        ind.independent_pmml_regression_scores(_PMML.encode(), [{"x1": "", "x2": "0"}], "probability_1")
    tree = _PMML.replace("<RegressionModel", "<TreeModel").replace("</RegressionModel>", "</TreeModel>")
    with pytest.raises(ind.IndependentUnsupported):
        ind.independent_pmml_regression_scores(tree.encode(), [{"x1": "1", "x2": "0"}], "probability_1")
    with pytest.raises(ind.IndependentUnsupported, match="link"):
        ind.independent_pmml_regression_scores(_PMML.replace('"logit"', '"softmax"').encode(), [], "probability_1")


def test_input_decoding_uses_frozen_choices_and_refuses_unknown_formats():
    confirmed = {"target_col": "y", "split_col": "s", "time_col": "t", "positive_label": 1, "negative_label": 0,
                 "split_value_mapping": {"train": "dev", "test": "val", "oot": "oot"}}
    raw = "y,s,t\n1,dev,202601\n0,val,2026-02-03\n0,oot,20260301\n".encode()
    rows = ind.read_validation_rows(raw, encoding="utf-8", confirmed=confirmed)
    assert [(r["label"], r["split"], r["date"]) for r in rows] == [
        (1, "train", "20260101"), (0, "test", "20260203"), (0, "oot", "20260301")]
    with pytest.raises(ValueError, match="outside confirmed labels"):
        ind.read_validation_rows(b"y,s,t\n2,dev,202601\n", encoding="utf-8", confirmed=confirmed)
    with pytest.raises(ind.IndependentUnsupported, match="time format"):
        ind.read_validation_rows(b"y,s,t\n1,dev,Jan 2026\n", encoding="utf-8", confirmed=confirmed)
    with pytest.raises(ind.IndependentUnsupported, match="encoding"):
        ind.read_validation_rows(raw, encoding="gbk", confirmed=confirmed)
