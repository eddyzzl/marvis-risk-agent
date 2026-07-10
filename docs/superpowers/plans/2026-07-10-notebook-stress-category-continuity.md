# Notebook Stress Category Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model-validation stress testing use the final feature/category semantics exposed by the executed Notebook, so transformed features such as `BH_A044_C0580` remain in the “睿智” pressure-test category and incomplete category coverage cannot be reported as completed.

**Architecture:** Add one deterministic resolver in `marvis.validation` whose small interface accepts final model feature/category rows plus the uploaded dictionary and returns one audited resolution. Both the Notebook-appended scenario producer and the platform metrics consumer use that resolution; the scenario artifact carries the resolution evidence, and the validation result exposes unclassified features so JSON, Web, Excel, and Word share the same coverage status.

**Tech Stack:** Python 3.13, pandas, dataclasses, pytest, openpyxl, generated Jupyter code cells.

## Global Constraints

- Preserve isolated Notebook execution and deterministic retry; do not restore a cross-stage persistent kernel.
- Preserve `RMC_SAMPLE_DF`, `RMC_TARGET_COL`, `RMC_ALGORITHM`, and `RMC_SCORE_FN` behavior.
- Keep `-9999` as the stress missing-value sentinel.
- Use exact feature-name matching only; do not add `_C0580`, `BH_`, prefix, suffix, or fuzzy matching heuristics.
- Notebook `RMC_FEATURE_IMPORTANCE.category/类别` wins over the uploaded dictionary for non-empty categories.
- The uploaded dictionary may only fill an empty Notebook category for an exact feature name.
- Keep the user's existing changes in `marvis/notebooks.py` and `tests/test_notebooks.py` out of all task commits.

## File Structure

- Create `marvis/validation/feature_categories.py`: the sole model-feature category resolution module and its structured result types.
- Create `tests/validation/test_feature_categories.py`: focused resolver behavior and conflict tests.
- Modify `marvis/pipeline_cellgen.py`: generated stress scenario cell consumes the Notebook importance DataFrame, calls the resolver, and emits schema v2 evidence.
- Modify `marvis/validation/platform_metrics.py`: validates and consumes the artifact mapping instead of rebuilding a competing mapping from disk.
- Modify `marvis/pipeline.py`: validates schema v2 stress artifacts and rejects incomplete/contradictory cached artifacts.
- Modify `marvis/validation/stress_test.py`: propagates unclassified features into overall pressure-test status.
- Modify `marvis/validation/results.py`: backward-compatible result serialization/deserialization for coverage evidence.
- Modify `marvis/report_texts.py`, `marvis/output/excel.py`, and `marvis/metric_tables.py`: expose unclassified feature counts in Word text, Excel, and Web table payloads.
- Modify `docs/notebook_contract.md` and `docs/对notebook的要求.md`: document category precedence and third-step retry semantics.
- Modify the matching tests under `tests/test_pipeline_v2.py`, `tests/validation/`, `tests/output/`, and `tests/test_metric_tables.py`.

---

### Task 1: Canonical Feature Category Resolution

**Files:**
- Create: `marvis/validation/feature_categories.py`
- Create: `tests/validation/test_feature_categories.py`

**Interfaces:**
- Consumes: `model_features: Sequence[tuple[str, str | None]]`, optional `dictionary: pandas.DataFrame`, and exact dictionary column names.
- Produces: `resolve_feature_categories(...) -> FeatureCategoryResolution` with `per_category`, `unclassified_features`, `conflicts`, and `source_counts`.

- [ ] **Step 1: Write failing resolver tests**

```python
import pandas as pd

from marvis.validation.feature_categories import resolve_feature_categories


def test_notebook_category_keeps_transformed_feature_name():
    dictionary = pd.DataFrame({"特征名": ["BH_A044"], "类别": ["睿智"]})

    result = resolve_feature_categories(
        model_features=[("BH_A044_C0580", "睿智")],
        dictionary=dictionary,
        feature_col="特征名",
        category_col="类别",
    )

    assert result.per_category == {"睿智": ["BH_A044_C0580"]}
    assert result.unclassified_features == []
    assert result.source_counts == {"notebook": 1, "dictionary": 0, "unresolved": 0}


def test_dictionary_only_fills_exact_feature_with_empty_notebook_category():
    dictionary = pd.DataFrame({"特征名": ["income"], "类别": ["内部特征"]})
    result = resolve_feature_categories(
        model_features=[("income", "")],
        dictionary=dictionary,
        feature_col="特征名",
        category_col="类别",
    )
    assert result.per_category == {"内部特征": ["income"]}
    assert result.source_counts["dictionary"] == 1


def test_dictionary_does_not_fuzzy_match_transformed_feature():
    dictionary = pd.DataFrame({"特征名": ["BH_A044"], "类别": ["睿智"]})
    result = resolve_feature_categories(
        model_features=[("BH_A044_C0580", "")],
        dictionary=dictionary,
        feature_col="特征名",
        category_col="类别",
    )
    assert result.per_category == {}
    assert result.unclassified_features == ["BH_A044_C0580"]


def test_conflicting_notebook_categories_are_reported():
    result = resolve_feature_categories(
        model_features=[("income", "内部特征"), ("income", "征信")],
        dictionary=None,
        feature_col="特征名",
        category_col="类别",
    )
    assert [(row.feature, row.categories, row.source) for row in result.conflicts] == [
        ("income", ("内部特征", "征信"), "notebook")
    ]
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/validation/test_feature_categories.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'marvis.validation.feature_categories'`.

- [ ] **Step 3: Implement the resolver**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureCategoryConflict:
    feature: str
    categories: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class FeatureCategoryResolution:
    per_category: dict[str, list[str]]
    unclassified_features: list[str]
    conflicts: list[FeatureCategoryConflict]
    source_counts: dict[str, int]


def resolve_feature_categories(
    *,
    model_features: Sequence[tuple[str, str | None]],
    dictionary: pd.DataFrame | None,
    feature_col: str,
    category_col: str,
) -> FeatureCategoryResolution:
    """Resolve final model features without guessing transformed names."""
```

Implementation requirements:

- Normalize feature/category values with `str(...).strip()` while treating `None` and pandas nulls as empty.
- Preserve first model-feature order and first category order.
- Deduplicate identical repeated feature/category rows.
- Record conflicting non-empty Notebook categories instead of picking one.
- Build an exact dictionary lookup only for features with empty Notebook categories.
- Record conflicting dictionary categories for an otherwise unresolved feature.
- Count final assignments by `notebook`, `dictionary`, and `unresolved`.

- [ ] **Step 4: Run resolver tests and existing stress tests**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/validation/test_feature_categories.py tests/validation/test_stress_test.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add marvis/validation/feature_categories.py tests/validation/test_feature_categories.py
git commit -m "Resolve stress categories from Notebook semantics" \
  -m "Constraint: exact feature names only" \
  -m "Rejected: suffix heuristics | fuzzy matching" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: pytest tests/validation/test_feature_categories.py tests/validation/test_stress_test.py" \
  -m "Not-tested: pipeline integration"
```

---

### Task 2: Use One Resolution in Scenario Production and Consumption

**Files:**
- Modify: `marvis/pipeline_cellgen.py:187-302`
- Modify: `marvis/validation/platform_metrics.py:140-270`
- Modify: `marvis/pipeline.py:1078-1110`
- Modify: `tests/test_pipeline_v2.py:145-200, 420-455`
- Create: `tests/validation/test_platform_metrics_stress_categories.py`

**Interfaces:**
- Consumes: `resolve_feature_categories` from Task 1 and `RMC_FEATURE_IMPORTANCE`/`FEATURE_IMPORTANCE` in the executed Notebook kernel.
- Produces: `marvis.validation_stress_scores.v2` containing `feature_categories`, `unclassified_features`, `source_counts`, `conflicts`, and category score rows.
- Produces: `stress_category_resolution_for_metrics(*, feature_importance, dictionary, feature_col, category_col, stress_scores_path) -> FeatureCategoryResolution`, which validates the scenario artifact against model metadata before metrics run.

- [ ] **Step 1: Write failing generated-cell regression test**

Add a test that builds the `stress-scores` source, executes it with:

```python
RMC_SAMPLE_DF = pd.DataFrame({
    "BH_A044_C0580": [1.0, 2.0],
    "y": [0, 1],
    "split": ["oot", "oot"],
})
RMC_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": ["BH_A044_C0580"],
    "类别": ["睿智"],
    "importance": [0.8],
})
RMC_SCORE_FN = lambda frame: pd.Series([0.1, 0.9], index=frame.index)
```

with an uploaded dictionary containing only `BH_A044`. Assert the generated JSON contains:

```python
assert payload["schema_version"] == "marvis.validation_stress_scores.v2"
assert payload["feature_categories"] == {"睿智": ["BH_A044_C0580"]}
assert payload["unclassified_features"] == []
assert payload["categories"][0]["category"] == "睿智"
assert payload["categories"][0]["dropped_features"] == ["BH_A044_C0580"]
```

- [ ] **Step 2: Write failing platform-consumer regression test**

Construct a schema v2 scenario artifact where `feature_categories` contains
`{"睿智": ["BH_A044_C0580"]}` while the uploaded dictionary contains only
`BH_A044`. Call `stress_category_resolution_for_metrics` with a
`FeatureImportanceRow(feature="BH_A044_C0580", category="睿智", ...)` and assert:

```python
resolution = stress_category_resolution_for_metrics(
    feature_importance=[
        FeatureImportanceRow(
            rank=1,
            feature="BH_A044_C0580",
            category="睿智",
            importance=0.8,
        )
    ],
    dictionary=pd.DataFrame({"特征名": ["BH_A044"], "类别": ["睿智"]}),
    feature_col="特征名",
    category_col="类别",
    stress_scores_path=artifact,
)
assert resolution.per_category == {"睿智": ["BH_A044_C0580"]}
assert resolution.unclassified_features == []
```

Add a second test whose artifact maps an extra feature not present in model metadata and
assert `ValueError("stress scenario artifact category mapping does not match model metadata")`.

- [ ] **Step 3: Verify both regression tests fail for the current reason**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest \
  tests/test_pipeline_v2.py -k 'stress_score_cell_prefers_notebook_categories or stress_score_reuse' \
  tests/validation/test_platform_metrics_stress_categories.py -q
```

Expected: the producer omits “睿智” and/or emits schema v1; the consumer rebuilds categories from the raw dictionary.

- [ ] **Step 4: Update the generated scenario producer**

In `_build_stress_scenario_score_cell_sources`:

- Import `resolve_feature_categories` from `marvis.validation.feature_categories`.
- Convert `RMC_FEATURE_IMPORTANCE` or legacy `FEATURE_IMPORTANCE` into ordered `(feature, category)` pairs; accept `category` first, then `类别`, otherwise empty category.
- Read the dictionary only as resolver fallback.
- Raise a clear `ValueError` when `resolution.conflicts` is non-empty.
- Score `resolution.per_category` in the current Notebook kernel using `RMC_SCORE_FN`.
- Emit schema v2 and serialize resolution evidence before atomically replacing the output path.

The output shape must be:

```python
{
    "schema_version": "marvis.validation_stress_scores.v2",
    "row_index": [...],
    "feature_categories": resolution.per_category,
    "unclassified_features": resolution.unclassified_features,
    "source_counts": resolution.source_counts,
    "conflicts": [asdict(row) for row in resolution.conflicts],
    "categories": [...],
}
```

- [ ] **Step 5: Update artifact validation and the platform consumer**

- `_stress_scores_artifact_valid` accepts v2 only for newly reusable evidence and validates all new fields and that each category row's `dropped_features` equals its mapping entry.
- Keep v1 readable only through an explicit compatibility path that forces regeneration when category coverage cannot be proven; cached empty v1 artifacts are invalid.
- Add `load_stress_category_resolution(path) -> FeatureCategoryResolution` in `platform_metrics.py` or the category module.
- `write_platform_validation_metrics` uses the artifact resolution for `run_stress_test` and passes the same mapping to `PrecomputedStressScenarioScorer`.
- Validate that artifact model features are present in `basic_info.feature_importance`; reject extra or mismatched features with `stress scenario artifact category mapping does not match model metadata`.

- [ ] **Step 6: Run producer/consumer tests and pipeline regressions**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest \
  tests/test_pipeline_v2.py \
  tests/validation/test_platform_metrics_stress_categories.py -q
```

Expected: all selected tests pass, including the transformed “睿智” regression.

- [ ] **Step 7: Commit Task 2**

```bash
git add marvis/pipeline_cellgen.py marvis/pipeline.py \
  marvis/validation/platform_metrics.py tests/test_pipeline_v2.py \
  tests/validation/test_platform_metrics_stress_categories.py
git commit -m "Keep stress categories aligned with Notebook state" \
  -m "Constraint: scenario producer and consumer share one exact mapping" \
  -m "Rejected: raw dictionary rebuild | persistent live kernel" \
  -m "Confidence: high" \
  -m "Scope-risk: medium" \
  -m "Tested: pytest tests/test_pipeline_v2.py tests/validation/test_platform_metrics_stress_categories.py" \
  -m "Not-tested: output rendering"
```

---

### Task 3: Propagate Coverage Status to JSON, Web, Excel, and Word

**Files:**
- Modify: `marvis/validation/results.py:215-235, 395-435`
- Modify: `marvis/validation/stress_test.py:63-178`
- Modify: `marvis/report_texts.py:290-318`
- Modify: `marvis/output/excel.py:298-335`
- Modify: `marvis/metric_tables.py:85-225`
- Modify: `tests/validation/test_stress_test.py`
- Modify: `tests/validation/test_results.py`
- Modify: `tests/output/test_excel.py`
- Modify: `tests/test_metric_tables.py`

**Interfaces:**
- Consumes: `FeatureCategoryResolution.unclassified_features` and `source_counts` from Tasks 1-2.
- Produces: backward-compatible `StressTestResult.unclassified_features`, `category_source_counts`, and status semantics shared by every output.

- [ ] **Step 1: Write failing result/status tests**

Add tests covering:

```python
result = run_stress_test(
    oot_sample=oot,
    config=_config(),
    feature_categories={"内部特征": ["x1"]},
    input_scorer=scorer,
    unclassified_features=["BH_A044_C0580"],
    category_source_counts={"notebook": 1, "dictionary": 0, "unresolved": 1},
)
assert result.status == "partial"
assert result.unclassified_features == ["BH_A044_C0580"]
```

and:

```python
result = run_stress_test(
    oot_sample=oot,
    config=_config(),
    feature_categories={},
    input_scorer=scorer,
    unclassified_features=["BH_A044_C0580"],
)
assert result.status == "failed"
```

Verify `validation_results_from_dict` defaults both new fields for legacy JSON.

- [ ] **Step 2: Verify status tests fail**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest \
  tests/validation/test_stress_test.py tests/validation/test_results.py -q
```

Expected: `run_stress_test` rejects the new keyword arguments and/or result fields are missing.

- [ ] **Step 3: Implement result fields and status semantics**

Add defaults:

```python
@dataclass(frozen=True)
class StressTestResult:
    baseline: StressBaseline
    per_category: list[StressCategoryResult]
    status: str = "completed"
    unclassified_features: list[str] = field(default_factory=list)
    category_source_counts: dict[str, int] = field(default_factory=dict)
```

Extend `run_stress_test` with optional lists/dicts, and calculate status as:

```python
category_status = _stress_test_status(per_category)
if unclassified_features and not per_category:
    overall_status = "failed"
elif unclassified_features and category_status == "completed":
    overall_status = "partial"
else:
    overall_status = category_status
```

Deserialize missing fields to empty values for old `validation_results.json` files.

- [ ] **Step 4: Write failing rendering tests**

- Word summary contains `未分类特征 1 个：BH_A044_C0580`.
- Excel `压力测试_汇总` contains a `分类覆盖` row whose error/evidence cell names the count and feature.
- Web metric table payload includes a `压力测试分类覆盖` table with overall status, unclassified count, and bounded feature-name text.

- [ ] **Step 5: Implement rendering changes**

- `_stress_text` prepends the unclassified count and names before category metrics.
- `_write_stress_summary` always writes one coverage row before per-category rows and adjusts conditional-format row offsets.
- `metric_table_sections_from_payload` adds a small table under the existing “压力测试” section:

```python
_table(
    "TEXT:stress_category_coverage",
    "压力测试分类覆盖",
    ["整体状态", "未分类特征数", "未分类特征"],
    [[status_label, len(unclassified), preview]],
)
```

Limit Web/Word previews to the first 20 feature names and append `等 N 个` when truncated; JSON retains the full list.

- [ ] **Step 6: Run result and rendering tests**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest \
  tests/validation/test_stress_test.py \
  tests/validation/test_results.py \
  tests/output/test_excel.py \
  tests/test_metric_tables.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add marvis/validation/results.py marvis/validation/stress_test.py \
  marvis/report_texts.py marvis/output/excel.py marvis/metric_tables.py \
  tests/validation/test_stress_test.py tests/validation/test_results.py \
  tests/output/test_excel.py tests/test_metric_tables.py
git commit -m "Expose incomplete stress category coverage" \
  -m "Constraint: unresolved model features cannot report completed" \
  -m "Rejected: Agent-only warning | hidden JSON-only evidence" \
  -m "Confidence: high" \
  -m "Scope-risk: medium" \
  -m "Tested: pytest stress result and output suites" \
  -m "Not-tested: full validation suite"
```

---

### Task 4: Update Contracts and Run End-to-End Verification

**Files:**
- Modify: `docs/notebook_contract.md:72-95, 177-184`
- Modify: `docs/对notebook的要求.md:254-300, 386-420`
- Modify: `tests/output/test_e2e_results_to_outputs.py`

**Interfaces:**
- Consumes: completed behavior from Tasks 1-3.
- Produces: documented Notebook/category precedence and one end-to-end regression from resolution through rendered outputs.

- [ ] **Step 1: Add an end-to-end transformed-feature regression**

Use a model importance row `BH_A044_C0580 / 睿智`, a dictionary row `BH_A044 / 睿智`, and a sample containing `BH_A044_C0580`. Assert the resulting validation object and Excel workbook include the “睿智” pressure category and do not classify the final feature as unresolved.

- [ ] **Step 2: Run the end-to-end test and verify its current status**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest tests/output/test_e2e_results_to_outputs.py -q
```

Expected after Tasks 1-3: PASS; if it fails, fix the production seam rather than weakening the assertion.

- [ ] **Step 3: Update Notebook documentation**

Document exactly:

- non-empty `RMC_FEATURE_IMPORTANCE.category/类别` is authoritative for its final feature name;
- the uploaded dictionary only fills empty categories by exact name;
- incomplete classification makes pressure-test status `partial` or `failed`;
- default third-step cells continue in the original Notebook execution;
- standalone third-step retries deterministically rerun the Notebook before appending metrics instead of reusing a stale kernel.

- [ ] **Step 4: Run full targeted verification**

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n py_313 python -m pytest \
  tests/validation \
  tests/test_pipeline_v2.py \
  tests/output \
  tests/test_metric_tables.py \
  tests/test_api_v2.py -q
CONDA_NO_PLUGINS=true conda run -n py_313 python -m ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 5: Reproduce the original category-resolution symptom without rerunning the user task**

Run a read-only resolver script against copies loaded from:

```text
workspace/material_uploads/73c42b2c089641e6898a19deb47f0ccd/特征字典.xlsx
workspace/tasks/afa268ab72c841babcf6a12ed5e34668/execution/feature_importance.csv
```

Expected:

```text
睿智 includes BH_A044_C0580
unclassified does not include BH_A044_C0580
```

Do not overwrite the existing task artifacts during this verification.

- [ ] **Step 6: Commit Task 4**

```bash
git add docs/notebook_contract.md docs/对notebook的要求.md \
  tests/output/test_e2e_results_to_outputs.py
git commit -m "Document Notebook stress category precedence" \
  -m "Constraint: Notebook final feature semantics are authoritative" \
  -m "Rejected: raw dictionary overwrite" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: targeted validation, pipeline, output, API, ruff, node check, diff check" \
  -m "Not-tested: destructive rerun of existing task"
```
