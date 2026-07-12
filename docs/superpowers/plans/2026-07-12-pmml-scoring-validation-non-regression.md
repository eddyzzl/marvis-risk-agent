# PMML-Only Model Validation Non-Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Notebook/code-model consistency validation with full-sample PMML scoring while preserving every other model-validation capability and mirroring the final Word report into Excel.

**Architecture:** New validation tasks build a confirmed, read-only `ValidationInputContract` from the four selected materials, score the complete sample through one chunked pypmml backend, and feed the persisted PMML scores into the existing deterministic metrics and mandatory OOT stress tests. Historical tasks retain the legacy result reader; Word, Excel, Web, and Agent render from one versioned presentation payload so only the old consistency stage changes.

**Tech Stack:** Python 3.13 (`py_313`), FastAPI, Pydantic, SQLite migrations, pandas, PyArrow/Parquet, pypmml/PMML4S, openpyxl, python-docx, vanilla JavaScript, pytest, ruff.

## Global Constraints

- Work from an isolated worktree created with `superpowers:using-git-worktrees`; branch name `codex/pmml-scoring-validation`, based on commit `b63ce09e`.
- Do not modify or discard the existing dirty changes in the main worktree: `marvis/notebook_contract.py` and `tests/test_notebook_contract.py`.
- New validation tasks require exactly four selected material roles: Notebook, sample, PMML, and data dictionary/feature metadata.
- Notebook is mandatory and read-only; no kernel, `exec`, `eval`, `%run`, shell, SQL, network, or Notebook-referenced file execution is permitted.
- `feature`, `category`, and finite numeric `importance` are mandatory for every PMML model feature; exact matching only.
- `PMML打分测试` must score 100% of rows with finite output; any failed row blocks all downstream stages.
- Default scoring backend is pypmml DataFrame chunk scoring; no per-row Python/JVM calls, container, or new default JPMML engine.
- `模型压力测试` is mandatory, uses `-9999`, and must rescore every non-empty category on the complete OOT with the same PMML backend.
- Preserve all current non-consistency Word placeholders/images, Excel sheets/charts/data, Agent tables/charts/conclusions, manual mode, Agent mode, confirmation cadence, and historical-task display.
- Final Word and Excel must consume the same confirmed `final_report_values`; Excel must contain searchable final report text, every Word image, and all underlying table data.
- Use `conda run -n py_313 python -m pytest -q`, `conda run -n py_313 python -m ruff check marvis tests --extend-exclude '*.ipynb'`, and the task-specific `conda run -n py_313 python <script>` commands for local Python work.
- Every task follows TDD, ends with its focused tests passing, and creates a narrow commit using the prescribed commit message.
- Do not expose the new flow on a stable release until every task in this plan and the final non-regression gate pass.

---

## Wave 1 — Read-Only Input Contract Foundation

### Task 1: Isolate the Work and Freeze the Non-Regression Baseline

**Files:**
- Create: `tests/fixtures/validation_non_regression_contract.json`
- Create: `tests/validation_output_contract.py`
- Create: `tests/test_validation_non_regression_contract.py`
- Read: `marvis/report_texts.py`
- Read: `marvis/output/image_render.py`
- Read: `marvis/output/excel.py`
- Read: `marvis/metric_tables.py`
- Read: `marvis/agent/validation_messages.py`

**Interfaces:**
- Consumes: commit `b63ce09e` and existing validation renderers.
- Produces: a machine-readable allowlist used by later tasks to prove that only consistency-related surfaces changed.

- [ ] **Step 1: Create the isolated worktree before touching implementation files**

Invoke `superpowers:using-git-worktrees`, then create or select:

```bash
git worktree add /Users/eddyz/zzl/projects/ai/agent/risk_manager-pmml-scoring \
  -b codex/pmml-scoring-validation b63ce09e
```

Expected: the new worktree is clean at `b63ce09e`; the original worktree still shows only the two pre-existing modified files.

- [ ] **Step 2: Write the failing baseline inventory test**

```python
# tests/test_validation_non_regression_contract.py
from dataclasses import asdict
from pathlib import Path

from openpyxl import load_workbook

from marvis.metric_tables import metric_table_sections_from_payload
from marvis.output.excel import write_validation_excel
from marvis.output.image_render import render_all_images
from marvis.report_texts import report_text_values_from_results
from marvis.template_reports import find_placeholders
from tests.output.test_excel import _make_results
from tests.validation_output_contract import load_validation_non_regression_contract


def test_validation_non_regression_contract_covers_current_outputs(tmp_path):
    contract = load_validation_non_regression_contract()
    results = _make_results()
    payload = {
        "basic_info": asdict(results.basic_info),
        "effectiveness": asdict(results.effectiveness),
        "stress_test": asdict(results.stress_test),
    }
    assert contract["report_text_keys"] == sorted(report_text_values_from_results(results))

    placeholders = [value[2:-2] for value in find_placeholders(
        Path("workspace/report_templates/default.docx")
    )]
    assert contract["template_text_keys"] == [
        value for value in placeholders if value.startswith("TEXT:")
    ]
    assert contract["template_image_keys"] == [
        value for value in placeholders if value.startswith("IMAGE:")
    ]

    images = render_all_images(results, tmp_path / "images")
    assert contract["rendered_image_keys"] == list(images)

    workbook_path = write_validation_excel(results, tmp_path / "validation.xlsx")
    workbook = load_workbook(workbook_path, read_only=True)
    assert contract["excel_sheets"] == workbook.sheetnames

    sections = metric_table_sections_from_payload(payload)
    normalized_sections = [
        {
            "title": section["title"],
            "table_keys": [table["key"] for table in section.get("tables", [])],
            "chart_keys": [chart["key"] for chart in section.get("charts", [])],
        }
        for section in sections
    ]
    assert contract["agent_sections"] == normalized_sections
```

- [ ] **Step 3: Run the test and verify the fixture is missing**

Run:

```bash
conda run -n py_313 python -m pytest tests/test_validation_non_regression_contract.py -q
```

Expected: FAIL with `FileNotFoundError` for `validation_non_regression_contract.json`.

- [ ] **Step 4: Add the explicit baseline contract**

```json
{
  "schema_version": "marvis.validation_non_regression.v1",
  "report_text_keys": [
    "TEXT:algorithm", "TEXT:data_source_summary", "TEXT:dataset_split_summary",
    "TEXT:model_name", "TEXT:model_training_description", "TEXT:model_version",
    "TEXT:oot_bad_rate", "TEXT:oot_count", "TEXT:oot_ks", "TEXT:oot_period",
    "TEXT:oot_psi", "TEXT:pressure_test_summary", "TEXT:report_title",
    "TEXT:reproducibility_summary", "TEXT:sample_end_month", "TEXT:sample_period",
    "TEXT:sample_start_month", "TEXT:stress_test_summary", "TEXT:test_bad_rate",
    "TEXT:test_count", "TEXT:train_bad_rate", "TEXT:train_count",
    "TEXT:train_test_period", "TEXT:train_test_ratio"
  ],
  "template_text_keys": [
    "TEXT:report_title", "TEXT:model_name", "TEXT:model_overview", "TEXT:model_scope",
    "TEXT:sample_start_month", "TEXT:sample_end_month", "TEXT:bad_sample_definition",
    "TEXT:good_sample_definition", "TEXT:train_test_period", "TEXT:train_test_ratio",
    "TEXT:oot_period", "TEXT:model_training_description", "TEXT:oot_psi", "TEXT:oot_ks",
    "TEXT:pressure_test_summary", "TEXT:pressure_impact_recommendation",
    "TEXT:final_validation_conclusion", "TEXT:drafter", "TEXT:draft_date",
    "TEXT:revision_version", "TEXT:revision_date", "TEXT:revision_author",
    "TEXT:revision_description"
  ],
  "template_image_keys": [
    "IMAGE:sample_overall_distribution", "IMAGE:sample_month_distribution",
    "IMAGE:model_parameters", "IMAGE:top20_feature_ranking", "IMAGE:psi_stability_table",
    "IMAGE:roc_ks_graph_train", "IMAGE:roc_ks_graph_test", "IMAGE:roc_ks_graph_oot",
    "IMAGE:ranking_table_train", "IMAGE:ranking_table_test", "IMAGE:ranking_table_oot",
    "IMAGE:overall_model_effect", "IMAGE:loan_month_effect", "IMAGE:pressure_ks_table",
    "IMAGE:pressure_psi_table", "IMAGE:pressure_score_shift"
  ],
  "rendered_image_keys": [
    "IMAGE:sample_overall_distribution", "IMAGE:sample_month_distribution",
    "IMAGE:top20_feature_ranking", "IMAGE:ranking_table", "IMAGE:roc_ks_graph_train",
    "IMAGE:ranking_table_train", "IMAGE:roc_ks_graph_test", "IMAGE:ranking_table_test",
    "IMAGE:roc_ks_graph_oot", "IMAGE:ranking_table_oot", "IMAGE:model_parameters",
    "IMAGE:overall_model_effect", "IMAGE:dataset_model_effect", "IMAGE:loan_month_effect",
    "IMAGE:psi_stability_table", "IMAGE:ks_discrimination_table",
    "IMAGE:pressure_ks_table", "IMAGE:pressure_psi_table",
    "IMAGE:pressure_score_shift_1", "IMAGE:pressure_score_shift",
    "IMAGE:pressure_score_shift_2", "IMAGE:pressure_score_shift_3",
    "IMAGE:pressure_score_shift_4", "IMAGE:pressure_score_shift_5",
    "IMAGE:pressure_score_shift_6", "IMAGE:pressure_score_shift_7"
  ],
  "excel_sheets": [
    "验证总览", "样本基本信息", "样本逐月分布", "模型超参", "特征重要性",
    "模型效果", "PSI稳定性", "ROC_KS曲线", "分箱_train", "分箱_test",
    "分箱_oot", "逐月效果", "压力测试_汇总", "压力测试_分箱_征信"
  ],
  "agent_sections": [
    {"title": "样本情况", "table_keys": ["IMAGE:sample_overall_distribution", "IMAGE:sample_month_distribution"], "chart_keys": []},
    {"title": "整体效果&稳定性", "table_keys": ["IMAGE:overall_model_effect"], "chart_keys": []},
    {"title": "分月效果&稳定性", "table_keys": ["IMAGE:loan_month_effect"], "chart_keys": []},
    {"title": "分箱排序性", "table_keys": ["IMAGE:ranking_table_train", "IMAGE:ranking_table_test", "IMAGE:ranking_table_oot"], "chart_keys": []},
    {"title": "特征重要性", "table_keys": ["IMAGE:top20_feature_ranking"], "chart_keys": []},
    {"title": "压力测试", "table_keys": ["IMAGE:pressure_ks_table", "TEXT:stress_category_coverage"], "chart_keys": []},
    {"title": "ROC&KS 曲线", "table_keys": ["ROC_KS_CURVES"], "chart_keys": []}
  ],
  "allowed_replacements": {
    "TEXT:reproducibility_summary": "TEXT:pmml_scoring_summary",
    "reproducibility": "pmml_scoring",
    "模型可复现性验证": "PMML打分测试",
    "分数一致性": "PMML打分测试"
  }
}
```

Add the shared test loader used by every later non-regression assertion:

```python
# tests/validation_output_contract.py
import json
from pathlib import Path


CONTRACT_PATH = Path(__file__).parent / "fixtures" / "validation_non_regression_contract.json"


def load_validation_non_regression_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
```

- [ ] **Step 5: Run the baseline test**

Run:

```bash
conda run -n py_313 python -m pytest tests/test_validation_non_regression_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the baseline**

```bash
git add tests/fixtures/validation_non_regression_contract.json \
  tests/validation_output_contract.py tests/test_validation_non_regression_contract.py
git commit -m "test: freeze validation output contract"
```

### Task 2: Add Versioned Input-Contract Types and Persistence

**Files:**
- Create: `marvis/validation/input_contracts.py`
- Create: `marvis/repositories/validation_contracts.py`
- Create: `tests/validation/test_input_contracts.py`
- Create: `tests/validation_builders.py`
- Create: `tests/test_validation_contract_repository.py`
- Modify: `tests/conftest.py`
- Modify: `marvis/db_schema.py`
- Modify: `marvis/domain.py`
- Modify: `marvis/db.py`

**Interfaces:**
- Consumes: selected material hashes and static evidence generated by Tasks 3–5.
- Produces: `ValidationInputContract`, JSON codecs, and `ValidationContractRepository` with optimistic revision control.

- [ ] **Step 1: Write failing round-trip and validation tests**

```python
# tests/validation/test_input_contracts.py
import pytest

from marvis.validation.input_contracts import (
    FieldCandidate,
    FieldEvidence,
    ValidationInputContract,
    input_contract_from_dict,
    input_contract_to_dict,
)


def test_input_contract_round_trip_preserves_evidence_and_confirmation():
    contract = ValidationInputContract.minimal_for_test(
        material_hashes={"notebook": "n", "sample": "s", "pmml": "p", "dictionary": "d"},
        target_col=FieldCandidate(
            value="y",
            evidence=(FieldEvidence("rmc_literal", 4, "RMC_TARGET_COL = 'y'", 1.0),),
        ),
    )
    restored = input_contract_from_dict(input_contract_to_dict(contract))
    assert restored == contract


def test_input_contract_rejects_unknown_schema_version():
    with pytest.raises(ValueError, match="unsupported validation input contract schema"):
        input_contract_from_dict({"schema_version": "unknown"})
```

- [ ] **Step 2: Run and verify missing module failure**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_input_contracts.py -q
```

Expected: collection FAIL with `ModuleNotFoundError: marvis.validation.input_contracts`.

- [ ] **Step 3: Implement focused immutable contract types and codecs**

```python
# marvis/validation/input_contracts.py
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
INPUT_CONTRACT_SCHEMA = "marvis.validation_input_contract.v1"
TRANSFORMATION_OPERATIONS = frozenset({
    "copy", "rename", "date_to_month", "constant_threshold",
    "constant_mapping", "constant_source_label",
})


@dataclass(frozen=True)
class FieldEvidence:
    source_kind: str
    notebook_cell: int | None
    source_excerpt: str
    confidence: float


@dataclass(frozen=True)
class FieldCandidate:
    value: JsonValue
    evidence: tuple[FieldEvidence, ...]


@dataclass(frozen=True)
class TransformationSpec:
    operation: Literal[
        "copy", "rename", "date_to_month", "constant_threshold",
        "constant_mapping", "constant_source_label",
    ]
    output_field: str
    input_fields: tuple[str, ...]
    params: dict[str, JsonValue]


@dataclass(frozen=True)
class StressUnit:
    model_feature: str
    raw_input_fields: tuple[str, ...]
    derivation_evidence: tuple[str, ...]


@dataclass(frozen=True)
class PmmlInputManifest:
    schema_version: str
    raw_required_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    model_features: tuple[str, ...]
    stress_units: tuple[StressUnit, ...]
    unsupported_derivations: tuple[str, ...]
    output_candidates: tuple[str, ...]
    algorithm: str


@dataclass(frozen=True)
class SampleSchema:
    path: str
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    row_count: int | None
    preview_row_count: int
    encoding: str | None
    sha256: str


@dataclass(frozen=True)
class FeatureMetadataRow:
    feature: str
    category: str
    importance: float
    source_sheet: str | None
    in_pmml: bool


@dataclass(frozen=True)
class MetadataCoverage:
    feature: float
    category: float
    importance: float
    stress_unit: float


@dataclass(frozen=True)
class FeatureMetadataResolution:
    schema_version: str
    rows: tuple[FeatureMetadataRow, ...]
    coverage: MetadataCoverage
    per_category_raw_fields: dict[str, tuple[str, ...]]
    extra_features: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class FieldRecognitionResult:
    schema_version: str
    notebook_sha256: str
    candidates: dict[str, tuple[FieldCandidate, ...]]
    transformations: tuple[TransformationSpec, ...]
    conflicts: tuple[str, ...]

    @classmethod
    def from_candidates(
        cls, *, notebook_sha256: str, candidates, transformations, conflicts
    ) -> "FieldRecognitionResult":
        return cls(
            schema_version="marvis.field_recognition.v1",
            notebook_sha256=notebook_sha256,
            candidates={key: tuple(value) for key, value in candidates.items()},
            transformations=tuple(transformations),
            conflicts=tuple(conflicts),
        )


@dataclass(frozen=True)
class ValidationInputContract:
    schema_version: str
    material_hashes: dict[str, str]
    status: Literal["ready", "pending_confirmation", "blocked"]
    candidates: dict[str, tuple[FieldCandidate, ...]]
    sample_schema: SampleSchema | None = None
    pmml_manifest: PmmlInputManifest | None = None
    feature_metadata: FeatureMetadataResolution | None = None
    confirmed: dict[str, Any] = field(default_factory=dict)
    transformations: tuple[TransformationSpec, ...] = ()
    conflicts: tuple[str, ...] = ()

    @classmethod
    def minimal_for_test(cls, *, material_hashes, target_col):
        return cls(
            schema_version=INPUT_CONTRACT_SCHEMA,
            material_hashes=material_hashes,
            status="pending_confirmation",
            candidates={"target_col": (target_col,)},
        )

    def require_pmml_manifest(self) -> PmmlInputManifest:
        if self.pmml_manifest is None:
            raise ValueError("validation input contract has no PMML manifest")
        return self.pmml_manifest

    def require_sample_schema(self) -> SampleSchema:
        if self.sample_schema is None:
            raise ValueError("validation input contract has no sample schema")
        return self.sample_schema

    def require_feature_metadata(self) -> FeatureMetadataResolution:
        if self.feature_metadata is None:
            raise ValueError("validation input contract has no resolved feature metadata")
        return self.feature_metadata

    def require_output_field(self) -> str:
        value = str(self.confirmed.get("pmml_output_field") or "")
        if not value:
            raise ValueError("validation input contract has no confirmed PMML output field")
        return value

    def require_algorithm(self) -> str:
        manifest = self.require_pmml_manifest()
        value = str(self.confirmed.get("algorithm") or manifest.algorithm or "")
        if not value:
            raise ValueError("validation input contract has no model algorithm")
        return value

    def require_model_params(self) -> dict[str, JsonValue]:
        value = self.confirmed.get("model_params")
        if not isinstance(value, dict):
            raise ValueError("validation input contract has no confirmed model parameters")
        return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class ValidationInputConfirmation:
    target_col: str
    positive_label: JsonScalar
    negative_label: JsonScalar
    split_col: str
    # Canonical group -> exact raw sample scalar; supports numeric splits.
    split_value_mapping: dict[str, JsonScalar]
    time_col: str
    time_granularity: str
    pmml_output_field: str
    model_params: dict[str, JsonValue]
    metadata_sheet: str | None
    feature_col: str
    category_col: str
    importance_col: str
    transformations: tuple[TransformationSpec, ...]


def input_contract_to_dict(value: ValidationInputContract) -> dict[str, Any]:
    return asdict(value)


def input_contract_from_dict(payload: dict[str, Any]) -> ValidationInputContract:
    if payload.get("schema_version") != INPUT_CONTRACT_SCHEMA:
        raise ValueError("unsupported validation input contract schema")
    return _decode_validation_input_contract(payload)


def transformation_spec_from_dict(payload: dict[str, Any]) -> TransformationSpec:
    expected = {"operation", "output_field", "input_fields", "params"}
    if set(payload) != expected:
        raise ValueError(
            f"invalid transformation keys: {sorted(set(payload) ^ expected)}"
        )
    operation = str(payload["operation"])
    if operation not in TRANSFORMATION_OPERATIONS:
        raise ValueError(f"unsupported transformation operation: {operation}")
    output_field = str(payload["output_field"] or "").strip()
    input_fields = tuple(str(value).strip() for value in payload["input_fields"])
    params = dict(payload["params"])
    if not output_field or any(not value for value in input_fields):
        raise ValueError("transformation fields must be non-empty")
    _require_json_value(params)
    return TransformationSpec(operation, output_field, input_fields, params)
```

Implement `_decode_validation_input_contract` through dedicated decoders for evidence, candidate, transformation, sample schema, PMML manifest/stress units, metadata rows/coverage, and the top-level contract. Each decoder rejects unknown/missing keys, validates recursive `JsonValue` via `_require_json_value`, restores tuples explicitly, and rejects non-finite floats; do not instantiate nested dataclasses through unchecked `**payload` calls. Add one malformed-payload test per nested type, not only a happy-path round trip.

- [ ] **Step 4: Add migration 3 and repository tests**

```python
# tests/validation_builders.py
from dataclasses import asdict, replace
from hashlib import sha256

from marvis.validation.input_contracts import (
    INPUT_CONTRACT_SCHEMA,
    FeatureMetadataResolution,
    FeatureMetadataRow,
    FieldCandidate,
    FieldEvidence,
    MetadataCoverage,
    PmmlInputManifest,
    SampleSchema,
    StressUnit,
    ValidationInputConfirmation,
    ValidationInputContract,
)


def make_candidate_contract(
    *, material_hashes: dict[str, str] | None = None
) -> ValidationInputContract:
    evidence = FieldEvidence("rmc_literal", 0, "RMC_TARGET_COL='y'", 1.0)
    manifest = PmmlInputManifest(
        schema_version="marvis.pmml_input_manifest.v1",
        raw_required_fields=("x1", "x2"),
        derived_fields=(),
        model_features=("x1", "x2"),
        stress_units=(
            StressUnit("x1", ("x1",), ()),
            StressUnit("x2", ("x2",), ()),
        ),
        unsupported_derivations=(),
        output_candidates=("probability_1",),
        algorithm="xgb",
    )
    metadata = FeatureMetadataResolution(
        schema_version="marvis.feature_metadata.v1",
        rows=(
            FeatureMetadataRow("x1", "内部", 0.6, "features", True),
            FeatureMetadataRow("x2", "征信", 0.4, "features", True),
        ),
        coverage=MetadataCoverage(1.0, 1.0, 1.0, 1.0),
        per_category_raw_fields={"内部": ("x1",), "征信": ("x2",)},
        extra_features=(),
        conflicts=(),
    )
    return ValidationInputContract(
        schema_version=INPUT_CONTRACT_SCHEMA,
        material_hashes=material_hashes or {
            key: sha256(key.encode("utf-8")).hexdigest()
            for key in ("notebook", "sample", "pmml", "dictionary")
        },
        status="pending_confirmation",
        candidates={"target_col": (FieldCandidate("y", (evidence,)),)},
        sample_schema=SampleSchema(
            path="sample.parquet",
            columns=("x1", "x2", "y", "split", "apply_month"),
            dtypes={}, row_count=4, preview_row_count=4, encoding=None,
            sha256="s" * 64,
        ),
        pmml_manifest=manifest,
        feature_metadata=metadata,
    )


def make_validation_confirmation() -> ValidationInputConfirmation:
    return ValidationInputConfirmation(
        target_col="y", positive_label=1, negative_label=0,
        split_col="split",
        split_value_mapping={"train": "train", "test": "test", "oot": "oot"},
        time_col="apply_month", time_granularity="month",
        pmml_output_field="probability_1", model_params={},
        metadata_sheet="features", feature_col="feature",
        category_col="category", importance_col="importance",
        transformations=(),
    )


def make_ready_contract() -> ValidationInputContract:
    contract = make_candidate_contract()
    confirmation = make_validation_confirmation()
    confirmed = asdict(confirmation)
    confirmed.pop("transformations")
    confirmed["algorithm"] = contract.require_pmml_manifest().algorithm
    return replace(
        contract,
        status="ready",
        confirmed=confirmed,
        transformations=confirmation.transformations,
    )
```

```python
# tests/test_validation_contract_repository.py
import sqlite3

import pytest

from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.repositories.validation_contracts import (
    ValidationContractRepository,
    ValidationContractRevisionConflict,
)
from tests.validation_builders import (
    make_candidate_contract,
    make_validation_confirmation,
)


def test_validation_contract_confirmation_uses_optimistic_revision(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    paths = {
        "notebook": tmp_path / "model.ipynb",
        "sample": tmp_path / "sample.parquet",
        "pmml": tmp_path / "model.pmml",
        "dictionary": tmp_path / "metadata.xlsx",
    }
    for role, path in paths.items():
        path.write_bytes(f"fixture-{role}".encode("utf-8"))
    task = TaskRepository(db_path).create_task(TaskCreate(
        model_name="A卡", model_version="v1", validator="qa", source_dir=str(tmp_path),
        notebook_path=str(paths["notebook"]), sample_path=str(paths["sample"]),
        pmml_path=str(paths["pmml"]), dictionary_path=str(paths["dictionary"]),
    ))
    assert task.validation_workflow_version == 2
    unrelated = TaskRepository(db_path).create_task(TaskCreate(
        task_type="modeling", model_name="modeling", model_version="v1",
        validator="qa", source_dir=str(tmp_path),
    ))
    assert unrelated.validation_workflow_version == 0
    init_db(db_path)
    assert TaskRepository(db_path).get_task(task.id).validation_workflow_version == 2
    repo = ValidationContractRepository(db_path)
    first = repo.replace_candidates(task.id, make_candidate_contract(
        material_hashes={role: sha256_file(path) for role, path in paths.items()}
    ))
    confirmed = repo.confirm(
        task.id, make_validation_confirmation(), expected_revision=first.revision,
    )
    assert confirmed.revision == first.revision + 1
    assert confirmed.status == "ready"
    with pytest.raises(ValidationContractRevisionConflict):
        repo.confirm(
            task.id, make_validation_confirmation(), expected_revision=first.revision,
        )


def test_migration_versions_only_historical_validation_rows(tmp_path):
    db_path = tmp_path / "schema-v2.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, task_type TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO tasks (id, task_type) VALUES (?, ?)",
            [("old-validation", "validation"), ("old-modeling", "modeling")],
        )
        conn.execute("PRAGMA user_version = 2")
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        versions = dict(conn.execute(
            "SELECT id, validation_workflow_version FROM tasks ORDER BY id"
        ))
    assert versions == {"old-modeling": 0, "old-validation": 1}
```

Expose the shared deterministic fixtures as soon as the types exist:

```python
# tests/conftest.py
import pytest

from marvis.validation.input_contracts import PmmlInputManifest, StressUnit
from tests.validation_builders import make_ready_contract


@pytest.fixture
def ready_contract():
    return make_ready_contract()


@pytest.fixture
def pmml_contract(ready_contract):
    return ready_contract


@pytest.fixture
def direct_manifest(ready_contract):
    return ready_contract.require_pmml_manifest()


@pytest.fixture
def derived_manifest():
    return PmmlInputManifest(
        schema_version="marvis.pmml_input_manifest.v1",
        raw_required_fields=("age", "income"),
        derived_fields=("age_bucket",),
        model_features=("age_bucket", "income"),
        stress_units=(
            StressUnit("age_bucket", ("age",), ("age_bucket <- age",)),
            StressUnit("income", ("income",), ()),
        ),
        unsupported_derivations=(),
        output_candidates=("probability_1",),
        algorithm="xgb",
    )
```

Append migration 3 to `marvis/db_schema.py`:

```python
def _migration_003_validation_input_contracts(conn):
    _ensure_column(
        conn,
        table="tasks",
        column="validation_workflow_version",
        definition="INTEGER NOT NULL DEFAULT 0",
    )
    conn.execute(
        "UPDATE tasks SET validation_workflow_version = 1 "
        "WHERE task_type = 'validation' AND validation_workflow_version = 0"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS validation_input_contracts (
            task_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            candidate_json TEXT NOT NULL,
            confirmed_json TEXT NOT NULL DEFAULT '{}',
            material_hashes_json TEXT NOT NULL,
            sample_schema_json TEXT NOT NULL,
            pmml_manifest_json TEXT NOT NULL,
            metadata_resolution_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
    """)


SCHEMA_VERSION = 3
_MIGRATIONS = (
    (1, _migration_001_baseline),
    (2, _migration_002_strategy_versioning),
    (3, _migration_003_validation_input_contracts),
)
```

Add `validation_workflow_version` to `TaskCreate`/`TaskRecord` and every task query/payload. The migration first adds the column with default `0`, then backfills only pre-existing `task_type='validation'` rows from `0` to immutable version `1`; unrelated workflows remain `0`, and an already-populated `2` is never overwritten. `TaskRepository.create_task` writes version `2` for every newly created validation task (and `0` for unrelated workflow task types). Scan, stage, Agent, evidence, retry, and report dispatch branch only on this field—not on missing output files, timestamps, or Notebook contents. No public update endpoint may change it. Repository tests must upgrade a v2-schema database snapshot and prove old validation rows remain `1`, old unrelated rows remain `0`, new validation rows are `2`, and reruns preserve the original value.

Implement the repository with this public record and method surface:

```python
# marvis/repositories/validation_contracts.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    SampleSchema,
    ValidationInputConfirmation,
    ValidationInputContract,
    input_contract_to_dict,
)


@dataclass(frozen=True)
class ValidationInputContractRecord:
    task_id: str
    revision: int
    status: str
    contract: ValidationInputContract
    created_at: str
    updated_at: str

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "revision": self.revision,
            "status": self.status,
            "needs_confirmation": self.status == "pending_confirmation",
            "read_only": True,
            "contract": input_contract_to_dict(self.contract),
        }


class ValidationContractRevisionConflict(RuntimeError):
    pass


class ValidationContractRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get(self, task_id: str) -> ValidationInputContractRecord | None:
        return _read_contract_record(self.db_path, task_id)

    def replace_candidates(
        self, task_id: str, contract: ValidationInputContract
    ) -> ValidationInputContractRecord:
        return _replace_contract_candidates(self.db_path, task_id, contract)

    def confirm(
        self, task_id: str, confirmation: ValidationInputConfirmation,
        *, expected_revision: int,
        resolved_sample_schema: SampleSchema | None = None,
        resolved_feature_metadata: FeatureMetadataResolution | None = None,
    ) -> ValidationInputContractRecord:
        return _confirm_contract(
            self.db_path, task_id, confirmation,
            expected_revision=expected_revision,
            resolved_sample_schema=resolved_sample_schema,
            resolved_feature_metadata=resolved_feature_metadata,
        )
```

`ValidationContractRepository.confirm` must always require an existing task and update `tasks.target_col`, `split_col`, `time_col`, and `algorithm` in the same transaction.

Implement the private repository operations with these exact transaction rules:

- `get`: one read transaction, decode every JSON column, and reconstruct a single `ValidationInputContract`; corrupt JSON/schema raises a repository data error rather than returning `None`.
- `replace_candidates`: `BEGIN IMMEDIATE`, verify the task exists, upsert revision `1` or `previous + 1`, replace candidate/manifest/metadata JSON, clear `confirmed_json`, and set status from the new contract. A material hash change always invalidates prior confirmation.
- `confirm`: receives only the already validated confirmation produced by Task 6, then `BEGIN IMMEDIATE`, selects the row, compares `expected_revision`, resolves each relative selected path against the persisted `task.source_dir` (rejecting traversal/escape), re-hashes all four currently selected files, applies the confirmed values to the stored candidate contract, requires status `ready`, and runs `UPDATE ... WHERE revision = ?`. A zero row count raises `ValidationContractRevisionConflict`. In the same connection, synchronize the four legacy task columns and append an audit event; any failure rolls back both records. Task 6 owns raw metadata/transformation validation before this transaction. Reuse the same selected-material resolver in scan, confirmation, scoring, and stress so a UI-relative path never resolves against process CWD.
- Store canonical JSON with `sort_keys=True`, compact separators, and `allow_nan=False`. Never persist Notebook source, sample rows, PMML XML, or raw metadata rows outside the normalized allowlisted fields.

- [ ] **Step 5: Run contract, migration, and repository tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_input_contracts.py \
  tests/test_validation_contract_repository.py \
  tests/test_db.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the contract foundation**

```bash
git add marvis/validation/input_contracts.py marvis/repositories/validation_contracts.py \
  marvis/db_schema.py marvis/domain.py marvis/db.py \
  tests/validation/test_input_contracts.py tests/validation_builders.py \
  tests/test_validation_contract_repository.py tests/conftest.py
git commit -m "feat: add versioned validation input contracts"
```

### Task 3: Parse the PMML Input Manifest and Inspect Sample Schema

**Files:**
- Create: `marvis/validation/pmml_manifest.py`
- Create: `marvis/validation/sample_schema.py`
- Create: `tests/validation/test_pmml_manifest.py`
- Create: `tests/validation/test_sample_schema.py`
- Create: `tests/fixtures/pmml/derived_fields.pmml`
- Modify: `marvis/validation/input_contracts.py`

**Interfaces:**
- Consumes: PMML path and selected sample path.
- Produces: `PmmlInputManifest`, `OutputFieldResolution`, and bounded `SampleSchema` without loading the complete sample.

- [ ] **Step 1: Write failing PMML manifest tests**

```python
# tests/validation/test_pmml_manifest.py
from pathlib import Path
import pytest

from marvis.validation.pmml_manifest import parse_pmml_input_manifest


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_manifest_resolves_direct_and_derived_field_dependencies():
    manifest = parse_pmml_input_manifest(FIXTURES / "pmml" / "derived_fields.pmml")
    assert manifest.raw_required_fields == ("age", "income")
    assert manifest.model_features == ("age_bucket", "income")
    assert manifest.stress_units[0].model_feature == "age_bucket"
    assert manifest.stress_units[0].raw_input_fields == ("age",)
    assert manifest.output_candidates == ("probability_1",)


def test_manifest_rejects_doctype_before_xml_parse(tmp_path):
    pmml = tmp_path / "unsafe.pmml"
    pmml.write_text("<!DOCTYPE PMML [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><PMML/>")
    with pytest.raises(ValueError, match="DOCTYPE and ENTITY are not allowed"):
        parse_pmml_input_manifest(pmml)


def test_manifest_rejects_utf16_doctype_without_bypassing_guard(tmp_path):
    pmml = tmp_path / "unsafe-utf16.pmml"
    pmml.write_bytes(
        "<?xml version='1.0' encoding='UTF-16'?><!DOCTYPE PMML><PMML/>".encode(
            "utf-16"
        )
    )
    with pytest.raises(ValueError, match="UTF-8 or ASCII"):
        parse_pmml_input_manifest(pmml)


def test_manifest_rejects_utf16_without_bom_before_ascii_guard(tmp_path):
    pmml = tmp_path / "unsafe-utf16-no-bom.pmml"
    pmml.write_bytes(
        "<?xml version='1.0'?><!DOCTYPE PMML><PMML/>".encode("utf-16-le")
    )
    with pytest.raises(ValueError, match="UTF-8 or ASCII"):
        parse_pmml_input_manifest(pmml)


def test_manifest_excludes_target_and_supplementary_fields_from_scoring_inputs():
    manifest = parse_pmml_input_manifest(FIXTURES / "min_lr.pmml")
    assert manifest.raw_required_fields == ("x1", "x2")
    assert "y" not in manifest.raw_required_fields


def test_manifest_maps_malformed_xml_to_bounded_validation_error(tmp_path):
    pmml = tmp_path / "malformed.pmml"
    pmml.write_text("<PMML><broken></PMML>", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid PMML XML"):
        parse_pmml_input_manifest(pmml)
```

- [ ] **Step 2: Run and verify missing module failure**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_pmml_manifest.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the bounded PMML manifest parser**

```python
# marvis/validation/pmml_manifest.py
from dataclasses import dataclass
from pathlib import Path
import re
from xml.etree import ElementTree

from marvis.validation.input_contracts import PmmlInputManifest, StressUnit

MAX_PMML_MANIFEST_BYTES = 512 * 1024 * 1024


def _declared_xml_encoding(prefix: bytes) -> str | None:
    match = re.search(br"encoding\s*=\s*['\"]([^'\"]+)['\"]", prefix, re.I)
    return match.group(1).decode("ascii").lower() if match else None


def parse_pmml_input_manifest(pmml_path: Path) -> PmmlInputManifest:
    if pmml_path.stat().st_size > MAX_PMML_MANIFEST_BYTES:
        raise ValueError("PMML file exceeds manifest inspection limit")
    with pmml_path.open("rb") as handle:
        prefix = handle.read(512)
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff", b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        raise ValueError("PMML must use UTF-8 or ASCII XML encoding")
    if b"\x00" in prefix:
        raise ValueError("PMML must use UTF-8 or ASCII XML encoding")
    declared_encoding = _declared_xml_encoding(prefix)
    if declared_encoding not in {None, "utf-8", "utf8", "us-ascii", "ascii"}:
        raise ValueError("PMML must use UTF-8 or ASCII XML encoding")
    tail = b""
    with pmml_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            inspected = (tail + chunk).upper()
            if b"<!DOCTYPE" in inspected or b"<!ENTITY" in inspected:
                raise ValueError("PMML DOCTYPE and ENTITY are not allowed")
            tail = inspected[-16:]
    try:
        root = ElementTree.parse(pmml_path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError("invalid PMML XML") from exc
    namespace = _namespace(root.tag)
    derived = _derived_field_dependencies(root, namespace)
    model_features = _model_features(root, namespace)
    stress_units, unsupported = _resolve_stress_units(model_features, derived)
    return PmmlInputManifest(
        schema_version="marvis.pmml_input_manifest.v1",
        raw_required_fields=_raw_required_fields(root, namespace, derived),
        derived_fields=tuple(sorted(derived)),
        model_features=model_features,
        stress_units=stress_units,
        unsupported_derivations=unsupported,
        output_candidates=_output_candidates(root, namespace),
        algorithm=_infer_algorithm(root, namespace),
    )


@dataclass(frozen=True)
class OutputFieldResolution:
    selected: str | None
    candidates: tuple[str, ...]
    source: str
    needs_confirmation: bool


def choose_pmml_output_field(
    manifest: PmmlInputManifest,
    *, notebook_hint: str | None,
    user_confirmation: str | None,
) -> OutputFieldResolution:
    if user_confirmation:
        if user_confirmation not in manifest.output_candidates:
            raise ValueError("confirmed PMML output field is not present in the model")
        return OutputFieldResolution(user_confirmation, manifest.output_candidates, "user", False)
    if notebook_hint and notebook_hint in manifest.output_candidates:
        return OutputFieldResolution(notebook_hint, manifest.output_candidates, "notebook", False)
    if len(manifest.output_candidates) == 1:
        return OutputFieldResolution(
            manifest.output_candidates[0], manifest.output_candidates, "pmml", False
        )
    return OutputFieldResolution(None, manifest.output_candidates, "ambiguous", True)
```

`_resolve_stress_units` must recurse through `FieldRef` dependencies, detect cycles, keep declaration order, and place unsupported Apply/MapValues chains in `unsupported_derivations` instead of guessing.

Implement every parser helper with the following boundaries:

- `_namespace(tag)`: return the namespace URI from `{uri}local`; reject a non-PMML root local name.
- `_derived_field_dependencies`: index global and local `DerivedField` nodes by declaration scope/order; record direct `FieldRef` edges and the expression node type. Duplicate names in the same scope and references to unknown fields are errors.
- `_model_features`: inspect the selected top-level scoring model's `MiningSchema`, include only active fields, and retain declaration order. Multiple top-level scoring models without an unambiguous active model block confirmation.
- `_raw_required_fields`: depth-first expand active model features through derived edges, de-duplicate at first occurrence, and return only `DataField` leaves; target/supplementary/weight fields never become roots.
- `_resolve_stress_units`: for each model feature, return all raw leaves only when every derivation node is an allowlisted deterministic PMML transform. Cycles or unsupported expressions produce an explicit unsupported entry and no guessed unit.
- `_output_candidates`: retain explicit probability `OutputField`s first; if PMML omits output declarations, synthesize only standards-defined probability candidates from binary target values. Multiple positive-class candidates require confirmation.
- `_infer_algorithm`: map PMML model node/`algorithmName` to canonical `xgb`, `lgb`, or the existing normalized fallback; `RMC_ALGORITHM` never overrides it.

Tests cover global/local shadowing, multi-level derivation, cycles, unknown references, target exclusion, multiple model/output ambiguity, XGBoost MiningModel segmentation, LightGBM MiningModel segmentation, and deterministic order.

- [ ] **Step 4: Write failing bounded sample-schema tests**

```python
# tests/validation/test_sample_schema.py
import pandas as pd

from marvis.validation.sample_schema import inspect_sample_schema


def test_csv_schema_inspection_uses_encoding_fallback_without_full_read(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes("年龄,标签\n20,0\n30,1\n".encode("gb18030"))
    schema = inspect_sample_schema(path)
    assert schema.columns == ("年龄", "标签")
    assert schema.preview_row_count == 2
    assert schema.encoding == "gb18030"


def test_parquet_schema_reports_columns_and_row_count(tmp_path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_parquet(path)
    schema = inspect_sample_schema(path)
    assert schema.columns == ("x", "y")
    assert schema.row_count == 2
```

- [ ] **Step 5: Implement format-specific schema inspection**

```python
# marvis/validation/sample_schema.py
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from marvis.validation.input_contracts import SampleSchema


def inspect_sample_schema(path: Path) -> SampleSchema:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _inspect_csv(path, encodings=("utf-8-sig", "utf-8", "gb18030"), preview_rows=20)
    if suffix == ".parquet":
        return _inspect_parquet(path)
    if suffix == ".feather":
        return _inspect_feather(path)
    if suffix in {".xlsx", ".xls"}:
        return _inspect_excel(path, preview_rows=20)
    raise ValueError(f"unsupported validation sample format: {suffix}")


def iter_sample_projection(
    path: Path, *, columns: tuple[str, ...], chunk_size: int,
    schema: SampleSchema | None = None,
) -> Iterator[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError("sample chunk size must be positive")
    selected_schema = schema or inspect_sample_schema(path)
    missing = [name for name in columns if name not in selected_schema.columns]
    if missing:
        raise ValueError("sample projection missing columns: " + ", ".join(missing))
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for frame in pd.read_csv(
            path, usecols=list(columns), encoding=selected_schema.encoding,
            chunksize=chunk_size,
        ):
            yield frame.loc[:, list(columns)]
        return
    if suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=chunk_size, columns=list(columns)
        ):
            yield batch.to_pandas().loc[:, list(columns)]
        return
    if suffix == ".feather":
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.RecordBatchFileReader(source)
            indices = [reader.schema.get_field_index(name) for name in columns]
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index).select(indices)
                for start in range(0, batch.num_rows, chunk_size):
                    yield batch.slice(start, chunk_size).to_pandas().loc[
                        :, list(columns)
                    ]
        return
    if suffix in {".xlsx", ".xls"}:
        # inspect_sample_schema already proved there is one non-empty sheet.
        frame = pd.read_excel(path, usecols=list(columns))
        for start in range(0, len(frame), chunk_size):
            yield frame.iloc[start:start + chunk_size].loc[:, list(columns)].copy()
        return
    raise ValueError(f"unsupported validation sample format: {suffix}")
```

Use PyArrow metadata for Parquet/Feather and bounded pandas reads for CSV/Excel; never load all rows in this task.

`_inspect_csv` opens the file once per allowed encoding, reads header plus at most 20 rows, and records the successful encoding; `_inspect_parquet` uses `ParquetFile.schema_arrow` and row-group metadata; `_inspect_feather` uses Arrow IPC schema and record-batch counts. `_inspect_excel` uses `ExcelFile` plus `nrows=20` and accepts exactly one non-empty sheet; because the v1 contract intentionally has no `sample_sheet` selector, multiple non-empty sample sheets are a hard, actionable scan error: “样本工作簿包含多个非空 sheet，请另存为单 sheet 样本后重新选择”. `iter_sample_projection` is the single selected-column reader used by Task 6 confirmation and wrapped with row IDs by Task 8; callers after confirmation pass `contract.require_sample_schema()` so stress categories do not re-hash/re-inspect the million-row file. Material hash validation stays at stage/cache boundaries. Excel may load its sole sheet because the format has no stable streaming reader, while the other formats remain chunked/batched. All readers normalize column names to exact strings, reject duplicates/blank names, and compute the source hash by streaming only during inspection. Add tests for projected column order, every format, missing columns, chunk-size validation, supplied-schema reuse (inspection called zero times), and multi-sheet rejection so the implementation never claims a confirmation path that does not exist.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_pmml_manifest.py \
  tests/validation/test_sample_schema.py -q
```

Expected: PASS.

```bash
git add marvis/validation/input_contracts.py marvis/validation/pmml_manifest.py \
  marvis/validation/sample_schema.py tests/validation/test_pmml_manifest.py \
  tests/validation/test_sample_schema.py tests/fixtures/pmml/derived_fields.pmml
git commit -m "feat: inspect PMML and validation sample schemas"
```

### Task 4: Recognize Notebook Fields Without Executing the Notebook

**Files:**
- Create: `marvis/validation/field_recognition.py`
- Create: `marvis/validation/field_transformations.py`
- Create: `tests/validation/test_field_recognition.py`
- Create: `tests/validation/test_field_transformations.py`
- Modify: `marvis/validation/input_contracts.py`
- Read only: `marvis/notebook_contract.py`

**Interfaces:**
- Consumes: Notebook path.
- Produces: `FieldRecognitionResult` and allowlisted `TransformationSpec` candidates with source evidence; does not touch legacy Notebook runtime functions.

- [ ] **Step 1: Write the security and literal-recognition tests**

```python
# tests/validation/test_field_recognition.py
import nbformat

from marvis.validation.field_recognition import recognize_notebook_fields


def test_recognizer_extracts_rmc_literals_without_executing_code(tmp_path, monkeypatch):
    notebook = tmp_path / "model.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(
            "open('/should/not/run')\n"
            "RMC_TARGET_COL = 'y'\n"
            "RMC_SPLIT_COL = 'model_flag'\n"
            "RMC_TIME_COL = 'loan_month'\n"
            "RMC_PMML_OUTPUT_FIELD = 'probability(1)'\n"
            "RMC_MODEL_PARAMS = {'learning_rate': 0.05, 'n_estimators': 300}\n"
        )
    ]), notebook)
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    result = recognize_notebook_fields(notebook)
    assert result.candidates["target_col"][0].value == "y"
    assert result.candidates["split_col"][0].value == "model_flag"
    assert result.candidates["time_col"][0].value == "loan_month"
    assert result.candidates["model_params"][0].value["n_estimators"] == 300
```

- [ ] **Step 2: Run and verify missing module failure**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_field_recognition.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement AST-only field recognition**

```python
# marvis/validation/field_recognition.py
import ast
from hashlib import sha256
from pathlib import Path
import re

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

from marvis.validation.field_transformations import extract_safe_transformations
from marvis.validation.input_contracts import FieldRecognitionResult


def recognize_notebook_fields(notebook_path: Path) -> FieldRecognitionResult:
    raw = notebook_path.read_bytes()
    if len(raw) > MAX_NOTEBOOK_BYTES:
        raise ValueError("Notebook exceeds static inspection limit")
    notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
    if len(notebook.cells) > MAX_NOTEBOOK_CELLS:
        raise ValueError("Notebook has too many cells for static inspection")
    candidates = _empty_candidate_map()
    transformations = []
    conflicts = []
    for cell_index, cell in enumerate(notebook.cells):
        source = str(cell.source)
        if len(source) > MAX_NOTEBOOK_CELL_CHARS:
            conflicts.append(f"cell {cell_index} exceeds static inspection limit")
            continue
        if cell.cell_type == "markdown":
            _collect_markdown_candidates(candidates, source, cell_index)
            continue
        if cell.cell_type != "code":
            continue
        tree = _safe_ast(source, cell_index)
        if tree is None:
            continue
        _collect_literal_candidates(candidates, tree, cell_index, source)
        _collect_model_param_candidates(candidates, tree, cell_index, source)
        _collect_comment_candidates(candidates, source, cell_index)
        _collect_saved_output_candidates(candidates, cell.get("outputs", ()), cell_index)
        transformations.extend(extract_safe_transformations(tree, cell_index=cell_index))
    return FieldRecognitionResult.from_candidates(
        notebook_sha256=sha256(raw).hexdigest(),
        candidates=candidates,
        transformations=tuple(transformations),
        conflicts=tuple(conflicts) + _same_priority_conflicts(candidates),
    )
```

`_raw_required_fields` starts only from active model features (`MiningField.usageType` absent/`active`); it must exclude target, predicted, supplementary, weight, frequency, group, and order fields before recursively resolving derived dependencies. Preserve deterministic PMML declaration order for scoring columns and use a separately sorted representation only when hashing.

Use `IPython.core.inputtransformer2.TransformerManager` only to make valid Python AST from magics; never execute transformed source. Collect candidates from RMC names first, then aliases such as `LABEL`, `TARGET`, `SPLIT_COL`, `TIME_COL`, estimator constructors, and literal parameter dictionaries.

Implement the called helpers, rather than leaving them as heuristic placeholders:

- `_safe_ast` runs only `TransformerManager().transform_cell(source)` followed by `ast.parse`; it catches `SyntaxError`/`IndentationError` and records bounded evidence, and never compiles, imports, evaluates, or calls `get_ipython`.
- `_collect_literal_candidates` walks `Assign` and `AnnAssign` nodes whose left side is a plain name from an explicit RMC/alias map. It accepts values only through `ast.literal_eval`, rejects calls/comprehensions/attribute access, truncates source evidence to 240 characters, and assigns priority `RMC literal > recognized alias > Markdown`.
- `_collect_model_param_candidates` recognizes allowlisted estimator constructor names (`XGBClassifier`, `XGBRegressor`, `LGBMClassifier`, `LGBMRegressor`, and their qualified forms) and literal keyword arguments/`**` literal dictionaries only. A non-literal value is evidence that confirmation is required, never a partially guessed parameter.
- `_collect_markdown_candidates` accepts only anchored `RMC_<NAME>[:=] <JSON-or-Python-literal>` lines inside Markdown/fenced configuration blocks. Free prose and substring matches do not become candidates.
- `_collect_comment_candidates` tokenizes Python comments and applies the same anchored literal grammar at lower confidence; it never treats arbitrary commented code as executable evidence.
- `_collect_saved_output_candidates` inspects only bounded `application/json` objects and anchored `text/plain` literal lines from already-saved outputs. It rejects HTML/JavaScript/images, tracebacks, payloads above the byte limit, and any object keys outside the explicit field alias map. Saved outputs are evidence only and never auto-confirm labels/mappings.
- `_same_priority_conflicts` returns a hard conflict only when the same field has distinct values from equal-priority explicit RMC assignments. Lower-priority disagreement remains visible candidate ambiguity; labels, split mappings, time granularity, and output class are never inferred from names.
- Notebook byte/cell/source bounds are constants covered by tests. Evidence stores cell index, source kind, confidence, and a bounded excerpt, never the full Notebook.

- [ ] **Step 4: Write and implement allowlisted transformation tests**

```python
# tests/validation/test_field_transformations.py
import ast

from marvis.validation.field_transformations import extract_safe_transformations


def test_extracts_date_to_month_and_constant_split_mapping():
    tree = ast.parse(
        "df['apply_month'] = df['apply_dt'].astype(str).str[:7]\n"
        "df['model_flag'] = df['source_tag'].map({'dev': 'train', 'holdout': 'oot'})\n"
    )
    specs = extract_safe_transformations(tree, cell_index=2)
    assert [(row.operation, row.output_field) for row in specs] == [
        ("date_to_month", "apply_month"),
        ("constant_mapping", "model_flag"),
    ]


def test_ignores_arbitrary_function_calls():
    specs = extract_safe_transformations(
        ast.parse("df['x'] = private_package.make_feature(df)"), cell_index=0
    )
    assert specs == ()
```

Implement only the six operations listed in `TransformationSpec`; no generic expression serializer. The same module owns deterministic materialization so scan, scoring, metrics, and stress never implement transformations differently:

The extractor has an exact AST shape table: `df['out'] = df['in']` (`copy`), an assigned `df.rename(columns={'old': 'new'})` with a literal one-to-one map (`rename`), `astype(str).str[:7]` or `to_datetime(...).dt.to_period('M').astype(str)` (`date_to_month`), `Series.map({...})` with a literal finite mapping (`constant_mapping`), `np.where(Series <op> literal, literal, literal)` (`constant_threshold`), and scalar JSON-literal column assignment (`constant_source_label`). Subscripts must use literal string column names and all dataframe roots must refer to the same statically named frame. Chained methods, lambdas, arbitrary calls, dynamic dictionaries, inplace rename, and overlapping rename outputs are rejected as unsupported evidence rather than serialized or executed.

```python
import ast
from collections.abc import Callable, Collection, Sequence
import operator

import numpy as np
import pandas as pd

from marvis.validation.input_contracts import (
    JsonScalar,
    TransformationSpec,
)


_THRESHOLD_OPERATORS: dict[str, Callable[[pd.Series, JsonScalar], pd.Series]] = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}


def topologically_sorted_transformations(
    specs: Sequence[TransformationSpec],
) -> tuple[TransformationSpec, ...]:
    by_output: dict[str, TransformationSpec] = {}
    for spec in specs:
        if spec.output_field in by_output:
            raise ValueError(f"duplicate transformation output: {spec.output_field}")
        by_output[spec.output_field] = spec
    state: dict[str, int] = {}
    ordered: list[TransformationSpec] = []

    def visit(output: str) -> None:
        if state.get(output) == 1:
            raise ValueError(f"transformation cycle includes: {output}")
        if state.get(output) == 2:
            return
        state[output] = 1
        spec = by_output[output]
        for input_field in spec.input_fields:
            if input_field in by_output:
                visit(input_field)
        state[output] = 2
        ordered.append(spec)

    for spec in specs:
        visit(spec.output_field)
    return tuple(ordered)


def required_transformation_inputs(
    output_fields: Collection[str], specs: Sequence[TransformationSpec]
) -> tuple[str, ...]:
    ordered = topologically_sorted_transformations(specs)
    by_output = {spec.output_field: spec for spec in ordered}
    inputs: list[str] = []

    def resolve(field: str) -> None:
        spec = by_output.get(field)
        if spec is None:
            if field not in inputs:
                inputs.append(field)
            return
        for input_field in spec.input_fields:
            resolve(input_field)

    for output_field in output_fields:
        resolve(output_field)
    return tuple(inputs)


def validate_transformation_plan(
    specs: Sequence[TransformationSpec], *, sample_columns: Collection[str]
) -> tuple[TransformationSpec, ...]:
    ordered = topologically_sorted_transformations(specs)
    raw_columns = set(sample_columns)
    for spec in ordered:
        if spec.output_field in raw_columns:
            raise ValueError(f"transformation overwrites raw sample field: {spec.output_field}")
        _validate_operation_arity_and_params(spec, threshold_operators=_THRESHOLD_OPERATORS)
    return ordered


def apply_confirmed_transformations(
    frame: pd.DataFrame, specs: Sequence[TransformationSpec]
) -> pd.DataFrame:
    result = frame.copy()
    for spec in topologically_sorted_transformations(specs):
        missing = [name for name in spec.input_fields if name not in result.columns]
        if missing:
            raise ValueError(f"transformation {spec.output_field} missing inputs: {missing}")
        if spec.operation in {"copy", "rename"}:
            result[spec.output_field] = result[spec.input_fields[0]]
        elif spec.operation == "date_to_month":
            result[spec.output_field] = result[spec.input_fields[0]].astype("string").str.slice(0, 7)
        elif spec.operation == "constant_mapping":
            mapped = result[spec.input_fields[0]].map(spec.params["mapping"])
            if mapped.isna().any():
                raise ValueError(f"transformation {spec.output_field} has unmapped values")
            result[spec.output_field] = mapped
        elif spec.operation == "constant_threshold":
            compare = _THRESHOLD_OPERATORS[str(spec.params["operator"])]
            mask = compare(result[spec.input_fields[0]], spec.params["threshold"])
            result[spec.output_field] = np.where(
                mask, spec.params["true_value"], spec.params["false_value"]
            )
        elif spec.operation == "constant_source_label":
            result[spec.output_field] = spec.params["value"]
        else:
            raise ValueError(f"unsupported confirmed transformation: {spec.operation}")
    return result
```

`_validate_operation_arity_and_params` uses an explicit required-parameter table for all six operations and rejects extra parameters as well as missing ones. Call `validate_transformation_plan` during scan/confirmation; runtime materialization calls `topologically_sorted_transformations` again as a defensive cycle/duplicate check. Add round-trip tests proving the same confirmed specs produce byte-for-byte equal target/split/time/model-input columns in scoring, metrics, and stress.

- [ ] **Step 5: Run focused tests and protect legacy behavior**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_field_recognition.py \
  tests/validation/test_field_transformations.py \
  tests/test_notebook_contract.py -q
```

Expected: PASS. `git diff -- marvis/notebook_contract.py tests/test_notebook_contract.py` must be empty in the isolated worktree.

- [ ] **Step 6: Commit the static recognizer**

```bash
git add marvis/validation/input_contracts.py marvis/validation/field_recognition.py \
  marvis/validation/field_transformations.py tests/validation/test_field_recognition.py \
  tests/validation/test_field_transformations.py
git commit -m "feat: recognize validation fields without notebook execution"
```

### Task 5: Normalize Feature Metadata and Enforce Complete Importance Coverage

**Files:**
- Create: `marvis/validation/feature_metadata.py`
- Create: `tests/validation/test_feature_metadata.py`
- Modify: `marvis/validation/input_contracts.py`
- Modify: `marvis/validation/feature_categories.py`

**Interfaces:**
- Consumes: selected dictionary/metadata file and `PmmlInputManifest`.
- Produces: candidate column selections and a confirmed `FeatureMetadataResolution` with exact `feature/category/importance` coverage.

- [ ] **Step 1: Write failing encoding, alias, and coverage tests**

```python
# tests/validation/test_feature_metadata.py
import pandas as pd
import pytest

from marvis.validation.feature_metadata import inspect_feature_metadata, normalize_feature_metadata


def test_gb18030_metadata_accepts_zero_importance_and_alias_columns(tmp_path, direct_manifest):
    path = tmp_path / "数据字典.csv"
    path.write_bytes(
        "指标英文,分类,feature_importance\nx1,征信,1.5\nx2,内部,0\n".encode("gb18030")
    )
    inspection = inspect_feature_metadata(path, direct_manifest)
    selection = inspection.only_valid_selection()
    resolution = normalize_feature_metadata(path, selection=selection, manifest=direct_manifest)
    assert resolution.coverage.feature == 1.0
    assert resolution.coverage.category == 1.0
    assert resolution.coverage.importance == 1.0
    assert resolution.rows[1].importance == 0.0


def test_conflicting_duplicate_importance_blocks(tmp_path, direct_manifest):
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "feature,category,importance\nx1,征信,1\nx1,征信,2\nx2,内部,0\n",
        encoding="utf-8",
    )
    inspection = inspect_feature_metadata(path, direct_manifest)
    with pytest.raises(ValueError, match="conflicting feature metadata for x1"):
        normalize_feature_metadata(
            path, selection=inspection.only_valid_selection(), manifest=direct_manifest
        )


def test_missing_importance_is_blocking_not_selection_ambiguity(tmp_path, direct_manifest):
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "feature,category,importance\nx1,征信,1\nx2,内部,\n",
        encoding="utf-8",
    )
    inspection = inspect_feature_metadata(path, direct_manifest)
    assert inspection.selections == ()
    assert any("importance" in value for value in inspection.blocking_errors)
    with pytest.raises(ValueError, match="importance"):
        inspection.only_valid_selection()
```

- [ ] **Step 2: Run and verify missing module failure**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_feature_metadata.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement deterministic candidate inspection and normalization**

```python
# marvis/validation/feature_metadata.py
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    PmmlInputManifest,
)


FEATURE_ALIASES = ("feature", "特征名", "特征名称", "指标英文", "feature_name")
CATEGORY_ALIASES = ("category", "类别", "分类", "数据源", "来源", "厂商名称", "供应商")
IMPORTANCE_ALIASES = ("importance", "feature_importance", "gain", "权重")


@dataclass(frozen=True)
class FeatureMetadataSelection:
    sheet_name: str | None
    feature_col: str
    category_col: str
    importance_col: str


@dataclass(frozen=True)
class FeatureMetadataInspection:
    path: str
    selections: tuple[FeatureMetadataSelection, ...]
    blocking_errors: tuple[str, ...]

    def only_valid_selection(self) -> FeatureMetadataSelection:
        if self.blocking_errors:
            raise ValueError("; ".join(self.blocking_errors))
        if len(self.selections) != 1:
            raise ValueError("feature metadata selection requires user confirmation")
        return self.selections[0]


def inspect_feature_metadata(path: Path, manifest: PmmlInputManifest) -> FeatureMetadataInspection:
    tables = _read_candidate_tables(path)
    selections = []
    diagnostics = []
    for sheet_name, frame in tables.items():
        valid, rejected = _column_selections(sheet_name, frame, manifest)
        selections.extend(valid)
        diagnostics.extend(rejected)
    return FeatureMetadataInspection(
        path=str(path),
        selections=tuple(selections),
        blocking_errors=tuple(diagnostics) if not selections else (),
    )


def normalize_feature_metadata(
    path: Path, *, selection: FeatureMetadataSelection, manifest: PmmlInputManifest
) -> FeatureMetadataResolution:
    frame = _read_selected_table(path, selection.sheet_name)
    rows = _normalize_rows(frame, selection)
    rows = _merge_identical_and_reject_conflicts(rows)
    return _resolve_against_manifest(rows, manifest, require_complete=True)
```

CSV decoding order is `utf-8-sig`, `utf-8`, `gb18030`; Excel reads every sheet with `sheet_name=None`. Convert importance with `pd.to_numeric(errors="raise")`, then reject NaN and infinity. Do not modify global `pd.read_csv`.

Implement the private helpers with deterministic, bounded behavior:

- `_read_candidate_tables` accepts only CSV/XLSX/XLS, rejects oversized files and sheets above the configured metadata row/column limits, reads CSV with the exact encoding fallback above, and returns Excel sheets in workbook order. Blank sheets are ignored; duplicate or blank headers block that sheet.
- `_column_selections` returns `(valid_selections, rejection_diagnostics)`. It takes the Cartesian product of distinct feature/category/importance alias matches on a sheet, forbids reusing one physical column for two roles, and keeps a selection only when every PMML `model_feature` appears exactly once after string trimming and every matched row has nonblank category and finite numeric importance. Missing importance aliases, missing PMML rows, blank category, non-finite importance, and duplicate conflicts produce bounded diagnostics naming sheet/column/feature. If at least one valid selection exists, preserve every equally valid selection for user confirmation; if none exists, all diagnostics become hard `blocking_errors` rather than ordinary ambiguity. Do not choose by column position.
- `_normalize_rows` trims surrounding whitespace from feature/category strings without lowercasing, Unicode folding, fuzzy matching, or coercing IDs such as `001` to numbers. Importance uses `pd.to_numeric(errors='raise')` and `np.isfinite`; zero and negative finite values are retained because they are source evidence, while blank/NaN/infinity fail.
- `_merge_identical_and_reject_conflicts` merges exact duplicate rows for a feature only when category and importance are equal; any disagreement reports the feature name and blocks confirmation.
- `_resolve_against_manifest` joins by exact `manifest.model_features` in PMML declaration order, requires 100% feature/category/importance coverage, retains metadata-only extras in `extra_features`, and builds `per_category_raw_fields` by expanding each feature through its resolved `StressUnit`. A missing/unsupported stress unit or an empty category is a hard error.

Add tests for ambiguous alias columns, multi-sheet ambiguity, oversize bounds, whitespace-only names, non-finite/negative/zero importance, exact duplicate merging, case-sensitive feature names, extras, and a feature whose PMML derivation cannot resolve to raw stress inputs.

- [ ] **Step 4: Add exact stress-unit coverage tests**

```python
def test_derived_feature_requires_a_resolved_stress_unit(tmp_path, derived_manifest):
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "feature,category,importance\nage_bucket,征信,0.6\nincome,内部,0.4\n",
        encoding="utf-8",
    )
    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(path, derived_manifest).only_valid_selection(),
        manifest=derived_manifest,
    )
    assert resolution.per_category_raw_fields == {"征信": ("age",), "内部": ("income",)}
```

Update `resolve_feature_categories` to adapt a complete `FeatureMetadataResolution` into the existing `FeatureCategoryResolution`; keep its legacy call signature for historical tasks.

- [ ] **Step 5: Run focused and legacy category tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_feature_metadata.py \
  tests/validation/test_feature_categories.py \
  tests/validation/test_platform_metrics_stress_categories.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit metadata normalization**

```bash
git add marvis/validation/input_contracts.py marvis/validation/feature_metadata.py \
  marvis/validation/feature_categories.py tests/validation/test_feature_metadata.py
git commit -m "feat: require complete validation feature metadata"
```

### Task 6: Build, Persist, and Confirm the Validation Input Contract

**Files:**
- Create: `marvis/routers/validation_contracts.py`
- Create: `marvis/validation/input_confirmation.py`
- Create: `marvis/validation_materials.py`
- Create: `tests/test_validation_input_contract_api.py`
- Create: `tests/validation/test_input_confirmation.py`
- Create: `tests/validation_material_builders.py`
- Modify: `marvis/api_schemas.py`
- Modify: `marvis/api_scan_helpers.py`
- Modify: `marvis/app.py`
- Modify: `marvis/routers/scans.py`
- Modify: `marvis/agent/validation_stages.py`
- Modify: `marvis/validation/config.py`
- Modify: `marvis/validation/checks.py`
- Modify: `marvis/repositories/validation_contracts.py`
- Modify: `tests/test_api_scan_helpers.py`
- Modify: `tests/test_api_v2.py`

**Interfaces:**
- Consumes: four selected material paths plus Task 2–5 inspectors.
- Produces: `GET/PUT /api/tasks/{task_id}/validation-input-contract`, scan payload evidence, and an execution gate requiring contract status `ready`.

Create one real four-material test builder used by both repository-level scan tests and HTTP tests:

```python
# tests/validation_material_builders.py
from dataclasses import dataclass
from pathlib import Path
import shutil

import nbformat
import pandas as pd

from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskRecord
from marvis.settings import Settings, build_settings


@dataclass(frozen=True)
class ValidationMaterialBundle:
    root: Path
    notebook_path: Path
    sample_path: Path
    pmml_path: Path
    dictionary_path: Path


def write_validation_material_bundle(
    root: Path, *, notebook_source: str
) -> ValidationMaterialBundle:
    root.mkdir(parents=True, exist_ok=True)
    notebook_path = root / "model.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(notebook_source)]),
        notebook_path,
    )
    sample_path = root / "sample.parquet"
    pd.DataFrame({
        "x1": [0.0, 1.0, 2.0, 3.0], "x2": [0.0, 1.0, 0.0, 1.0],
        "y": [0, 1, 0, 1], "split": ["train", "test", "oot", "oot"],
        "apply_month": ["202601", "202602", "202603", "202603"],
    }).to_parquet(sample_path, index=False)
    pmml_path = root / "model.pmml"
    shutil.copy2(Path("tests/fixtures/min_lr.pmml"), pmml_path)
    dictionary_path = root / "metadata.csv"
    pd.DataFrame({
        "feature": ["x1", "x2"],
        "category": ["内部", "征信"],
        "importance": [0.6, 0.4],
    }).to_csv(dictionary_path, index=False)
    return ValidationMaterialBundle(
        root, notebook_path, sample_path, pmml_path, dictionary_path
    )


def create_repository_validation_task(
    tmp_path: Path, *, notebook_source: str
) -> tuple[TaskRecord, TaskRepository, Settings]:
    settings = build_settings(tmp_path / "workspace")
    bundle = write_validation_material_bundle(
        settings.workspace / "bundle", notebook_source=notebook_source
    )
    init_db(settings.db_path)
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(TaskCreate(
        model_name="fixture", model_version="v2", validator="pytest",
        source_dir=str(bundle.root), notebook_path=bundle.notebook_path.name,
        sample_path=bundle.sample_path.name, pmml_path=bundle.pmml_path.name,
        dictionary_path=bundle.dictionary_path.name,
    ))
    return task, repo, settings


def create_api_validation_task(client, bundle: ValidationMaterialBundle) -> tuple[str, dict]:
    created = client.post("/api/tasks", json={
        "task_type": "validation", "model_name": "fixture",
        "model_version": "v2", "validator": "pytest",
        "source_dir": str(bundle.root), "run_mode": "manual",
    })
    assert created.status_code == 200, created.text
    task_id = str(created.json()["id"])
    selected = client.put(f"/api/tasks/{task_id}/materials", json={
        "notebook_path": bundle.notebook_path.name,
        "sample_path": bundle.sample_path.name,
        "pmml_path": bundle.pmml_path.name,
        "dictionary_path": bundle.dictionary_path.name,
    })
    assert selected.status_code == 200, selected.text
    scanned = client.post(f"/api/tasks/{task_id}/scan")
    assert scanned.status_code == 200, scanned.text
    return task_id, scanned.json()
```

The helper creates files below the workspace allowlisted root and always selects paths through the same route used by users.

- [ ] **Step 1: Write failing scan behavior tests**

```python
# tests/test_api_scan_helpers.py
from tests.validation_material_builders import create_repository_validation_task


def _validation_task_with_four_materials(tmp_path, *, notebook_source):
    return create_repository_validation_task(
        tmp_path, notebook_source=notebook_source
    )


def test_scan_builds_read_only_validation_contract_without_rmc_score_fn(
    tmp_path, monkeypatch
):
    task, repo, settings = _validation_task_with_four_materials(
        tmp_path,
        notebook_source="RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\nRMC_TIME_COL='month'",
    )
    monkeypatch.setattr(
        "marvis.notebooks.run_notebook",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    payload = perform_scan_task(repo, task, settings)
    assert payload["validation_input_contract"]["status"] == "pending_confirmation"
    assert payload["validation_input_contract"]["needs_confirmation"] is True
    assert payload["validation_input_contract"]["read_only"] is True
    assert repo.get_task(task.id).status == TaskStatus.SCANNED
```

- [ ] **Step 2: Run and verify the old RMC precheck fails the test**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/test_api_scan_helpers.py::test_scan_builds_read_only_validation_contract_without_rmc_score_fn -q
```

Expected: FAIL because `precheck_notebook_contract` still requires `RMC_SCORE_FN` and `RMC_SAMPLE_DF`.

- [ ] **Step 3: Replace the new-task scan seam, preserving material selection/history**

Create the shared orchestration-level path resolver in `marvis/validation_materials.py` (outside the pure `validation/` package and outside `api_scan_helpers`, which already imports `pipeline`):

```python
# marvis/validation_materials.py
from dataclasses import dataclass
from pathlib import Path

from marvis.domain import TaskRecord


@dataclass(frozen=True)
class ResolvedValidationMaterials:
    notebook: Path
    sample: Path
    pmml: Path
    dictionary: Path


def resolve_selected_validation_materials(
    task: TaskRecord,
) -> ResolvedValidationMaterials:
    source_dir = Path(task.source_dir).resolve()

    def selected(attribute: str, label: str) -> Path:
        raw_value = str(getattr(task, attribute, "") or "").strip()
        if not raw_value:
            raise ValueError(f"selected {label} path is missing")
        raw = Path(raw_value).expanduser()
        candidate = (raw if raw.is_absolute() else source_dir / raw).resolve()
        try:
            candidate.relative_to(source_dir)
        except ValueError as exc:
            raise ValueError(f"selected {label} escapes source directory") from exc
        if not candidate.is_file():
            raise ValueError(f"selected {label} file does not exist")
        return candidate

    return ResolvedValidationMaterials(
        notebook=selected("notebook_path", "Notebook"),
        sample=selected("sample_path", "sample"),
        pmml=selected("pmml_path", "PMML"),
        dictionary=selected("dictionary_path", "feature metadata"),
    )
```

Then add the deterministic contract builder to `marvis/api_scan_helpers.py`:

```python


def build_validation_input_contract(
    *, notebook_path: Path, sample_path: Path, pmml_path: Path, dictionary_path: Path
) -> ValidationInputContract:
    sample_schema = inspect_sample_schema(sample_path)
    fields = recognize_notebook_fields(notebook_path)
    manifest = parse_pmml_input_manifest(pmml_path)
    metadata_inspection = inspect_feature_metadata(dictionary_path, manifest)
    metadata = (
        normalize_feature_metadata(
            dictionary_path,
            selection=metadata_inspection.selections[0],
            manifest=manifest,
        )
        if len(metadata_inspection.selections) == 1
        else None
    )
    return assemble_validation_input_contract(
        material_hashes={
            "notebook": sha256_file(notebook_path),
            "sample": sha256_file(sample_path),
            "pmml": sha256_file(pmml_path),
            "dictionary": sha256_file(dictionary_path),
        },
        sample_schema=sample_schema,
        fields=fields,
        manifest=manifest,
        metadata=metadata,
        metadata_selections=metadata_inspection.selections,
        metadata_errors=metadata_inspection.blocking_errors,
    )
```

Implement assembly as the single deterministic readiness decision:

```python
REQUIRED_CONFIRMED_KEYS = frozenset({
    "target_col", "positive_label", "negative_label", "split_col",
    "split_value_mapping", "time_col", "time_granularity",
    "pmml_output_field", "model_params", "metadata_sheet",
    "feature_col", "category_col", "importance_col",
})


def assemble_validation_input_contract(
    *, material_hashes: dict[str, str], sample_schema: SampleSchema,
    fields: FieldRecognitionResult, manifest: PmmlInputManifest,
    metadata: FeatureMetadataResolution | None,
    metadata_selections: tuple[FeatureMetadataSelection, ...],
    metadata_errors: tuple[str, ...],
) -> ValidationInputContract:
    candidates = dict(fields.candidates)
    candidates["algorithm"] = (_pmml_candidate(manifest.algorithm),)
    candidates["pmml_output_field"] = tuple(
        _pmml_candidate(value) for value in manifest.output_candidates
    )
    candidates.update(_metadata_selection_candidates(metadata_selections))
    conflicts = list(fields.conflicts)
    conflicts.extend(f"feature metadata: {message}" for message in metadata_errors)
    if manifest.unsupported_derivations:
        conflicts.append(
            "unsupported PMML stress dependencies: "
            + ", ".join(manifest.unsupported_derivations)
        )
    transformations = validate_transformation_plan(
        fields.transformations, sample_columns=sample_schema.columns
    )
    required_sample_inputs = required_transformation_inputs(
        manifest.raw_required_fields, transformations
    )
    missing = [name for name in required_sample_inputs if name not in sample_schema.columns]
    if missing:
        conflicts.append("sample missing required PMML inputs: " + ", ".join(missing))
    confirmed = _unique_high_confidence_values(candidates)
    confirmed["algorithm"] = manifest.algorithm
    if metadata is not None:
        confirmed.update(_selection_values(metadata_selections[0]))
    ready = (
        not conflicts
        and metadata is not None
        and REQUIRED_CONFIRMED_KEYS <= set(confirmed)
        and _metadata_coverage_is_complete(metadata)
    )
    return ValidationInputContract(
        schema_version=INPUT_CONTRACT_SCHEMA,
        material_hashes=material_hashes,
        status="blocked" if conflicts else ("ready" if ready else "pending_confirmation"),
        candidates=candidates,
        sample_schema=sample_schema,
        pmml_manifest=manifest,
        feature_metadata=metadata,
        confirmed=confirmed if ready else {
            key: value for key, value in confirmed.items() if key == "algorithm"
        },
        transformations=transformations,
        conflicts=tuple(conflicts),
    )
```

`_unique_high_confidence_values` only auto-confirms a field when all highest-priority evidence agrees; it never guesses labels, split mapping, or time granularity from column names. Confirmation reruns metadata normalization with the chosen sheet/columns, validates the submitted transformations and complete coverage, then writes the complete `confirmed` mapping. A non-empty `conflicts` tuple is a hard 422; ordinary ambiguity stays `pending_confirmation`.

In `perform_scan_task`, call this after the existing four-role path resolution; stop calling `precheck_notebook_contract` for new schema tasks. Keep scan history and artifact transaction semantics unchanged. A valid but ambiguous contract leaves the task `SCANNED` with `pending_confirmation`; when assembly returns `status="blocked"`, persist bounded scan evidence/history and raise a validation-material error so the route returns 422. Missing/invalid importance is therefore never presented as user-selectable ambiguity.

Update `routers/scans.py` so v2 material/contract `ValueError` becomes `422` with prefix `validation materials invalid:` rather than the misleading legacy `source dir invalid:` text. Preserve the old message for actual `FileNotFoundError`/`NotADirectoryError` and historical v1 scans. The missing-importance E2E asserts the structured v2 detail.

- [ ] **Step 4: Add the confirmation API schema and repository wiring**

```python
# marvis/api_schemas.py
from marvis.validation.input_contracts import JsonScalar, JsonValue


class ValidationInputConfirmationRequest(BaseModel):
    revision: int
    target_col: str
    positive_label: str | int | float | bool
    negative_label: str | int | float | bool | None = None
    split_col: str
    split_value_mapping: dict[str, JsonScalar]
    time_col: str
    time_granularity: str
    pmml_output_field: str
    model_params: dict[str, JsonValue]
    metadata_sheet: str | None = None
    feature_col: str
    category_col: str
    importance_col: str
    transformations: list[dict[str, object]] = Field(default_factory=list)
```

`model_params` is intentionally required on confirmation. The UI pre-fills statically recognized values; when the Notebook contains no recoverable parameters, the user must explicitly confirm an empty object, which renders the existing “no model parameters found” evidence instead of blocking later inside `require_model_params()`. `split_value_mapping` is canonical group name to exact raw `JsonScalar` (`{"train": 0, "test": 1, "oot": 2}` is valid); update `ValidationConfig.split_values` and `validate_split_values` type hints/checks accordingly, preserving scalar types during comparisons.

Do not pass this raw request directly into the repository. Add a deterministic validation service:

```python
# marvis/validation/input_confirmation.py
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.files import sha256_file
from marvis.validation.feature_metadata import normalize_feature_metadata
from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    required_transformation_inputs,
    validate_transformation_plan,
)
from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    SampleSchema,
    ValidationInputConfirmation,
    ValidationInputContract,
)
from marvis.validation.pmml_manifest import choose_pmml_output_field
from marvis.validation.sample_schema import inspect_sample_schema, iter_sample_projection


@dataclass(frozen=True)
class ValidatedConfirmation:
    values: ValidationInputConfirmation
    sample_schema: SampleSchema
    feature_metadata: FeatureMetadataResolution


@dataclass(frozen=True)
class ObservedConfirmationValues:
    target_values: tuple[object, ...]
    split_values: tuple[object, ...]
    row_count: int


def json_scalar_identity(value: object) -> tuple[str, str]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or not isinstance(value, (str, int, float, bool)):
        raise ValueError("field value is not a JSON scalar")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("field value is not finite")
    return type(value).__name__, json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False
    )


def normalize_binary_target(
    values: pd.Series, *, positive: object, negative: object
) -> pd.Series:
    positive_key = json_scalar_identity(positive)
    negative_key = json_scalar_identity(negative)
    if positive_key == negative_key:
        raise ValueError("positive and negative labels must differ")
    normalized = []
    for value in values.tolist():
        if pd.isna(value):
            raise ValueError("target contains null values")
        key = json_scalar_identity(value)
        if key == positive_key:
            normalized.append(1)
        elif key == negative_key:
            normalized.append(0)
        else:
            raise ValueError(f"target contains value outside confirmed labels: {value!r}")
    return pd.Series(normalized, index=values.index, dtype="int8")


def inspect_confirmation_values(
    sample_path, *, columns, transformations, target_col, split_col, time_col,
    time_granularity, sample_schema, chunk_size=100_000,
) -> ObservedConfirmationValues:
    target_values: dict[tuple[str, str], object] = {}
    split_values: dict[tuple[str, str], object] = {}
    row_count = 0
    for frame in iter_sample_projection(
        sample_path, columns=tuple(columns), chunk_size=chunk_size,
        schema=sample_schema,
    ):
        frame = apply_confirmed_transformations(frame, transformations)
        _collect_bounded_json_scalars(
            target_values, frame[target_col], limit=100, field="target"
        )
        _collect_bounded_json_scalars(
            split_values, frame[split_col], limit=100, field="split"
        )
        _validate_time_series(frame[time_col], time_granularity)
        row_count += len(frame)
    if row_count == 0:
        raise ValueError("validation sample contains no rows")
    return ObservedConfirmationValues(
        tuple(target_values.values()), tuple(split_values.values()), row_count
    )


def validate_confirmation_against_materials(
    *, contract: ValidationInputContract, sample_path: Path,
    dictionary_path: Path, requested: ValidationInputConfirmation,
) -> ValidatedConfirmation:
    schema = inspect_sample_schema(sample_path)
    if schema.sha256 != contract.material_hashes["sample"]:
        raise ValueError("selected sample changed; rescan before confirmation")
    if sha256_file(dictionary_path) != contract.material_hashes["dictionary"]:
        raise ValueError("selected feature metadata changed; rescan before confirmation")
    manifest = contract.require_pmml_manifest()
    output = choose_pmml_output_field(
        manifest, notebook_hint=None,
        user_confirmation=requested.pmml_output_field,
    )
    if output.needs_confirmation or output.selected is None:
        raise ValueError("PMML output field requires a valid selection")
    transformations = validate_transformation_plan(
        requested.transformations, sample_columns=schema.columns
    )
    required_columns = required_transformation_inputs(
        (
            *manifest.raw_required_fields,
            requested.target_col,
            requested.split_col,
            requested.time_col,
        ),
        transformations,
    )
    missing = [name for name in required_columns if name not in schema.columns]
    if missing:
        raise ValueError("confirmed fields are missing from sample: " + ", ".join(missing))
    observed = inspect_confirmation_values(
        sample_path,
        columns=required_columns,
        transformations=transformations,
        target_col=requested.target_col,
        split_col=requested.split_col,
        time_col=requested.time_col,
        time_granularity=requested.time_granularity,
        sample_schema=schema,
    )
    validate_binary_labels(
        observed.target_values,
        positive=requested.positive_label,
        negative=requested.negative_label,
    )
    validate_split_mapping(
        observed.split_values, requested.split_value_mapping,
        required_canonical=("train", "test", "oot"),
    )
    metadata = normalize_feature_metadata(
        dictionary_path,
        selection=feature_metadata_selection_from_confirmation(requested),
        manifest=manifest,
    )
    if not metadata_has_complete_coverage(metadata):
        raise ValueError("feature metadata coverage must be 100%")
    return ValidatedConfirmation(
        values=requested,
        sample_schema=schema,
        feature_metadata=metadata,
    )
```

Keep this module inside the pure `validation/` boundary: it accepts only a deterministic `ValidationInputContract`, resolved file paths, and requested values; it never imports `TaskRecord`, repositories, FastAPI, DB, or task status. `resolve_selected_validation_materials` stays in root-level `marvis/validation_materials.py` as the shared source-dir-relative, traversal-safe orchestration resolver used by scan/router/pipeline without creating the existing `api_scan_helpers -> pipeline` circular dependency. `_collect_bounded_json_scalars` rejects null/non-JSON scalar values and keys values by `(type-name, canonical-value)` so `1`, `1.0`, and `"1"` cannot collapse. `inspect_confirmation_values` synchronously streams only the required control columns, applies the same allowlisted transformations, collects bounded unique target/split values, and validates every non-null time value against the confirmed granularity; it never loads the full wide sample. It fails if unique-value bounds are exceeded rather than truncating. `validate_binary_labels` requires the positive label to occur, requires exactly two type-stable label values for binary validation, and verifies the optional negative label. `validate_split_mapping` requires exactly the canonical keys `train/test/oot`, requires their raw scalar values to be type-stably unique and to cover every observed split value, and requires each group to be non-empty. This supports numeric raw splits without string coercion. Wrap metadata errors with a `feature metadata:` prefix so the 422 response identifies the failing material. Cancellation is required for long-running scoring, metrics, and stress jobs below; this synchronous three-column confirmation request does not advertise a cancellation endpoint.

Extend `ValidationContractRepository.confirm` with keyword-only `resolved_sample_schema` and `resolved_feature_metadata` arguments. The repository uses these validated values to replace the candidate schema/metadata in the same optimistic transaction; if omitted in Task 2 unit tests, it requires the already stored values to be complete. It still re-hashes all materials under `BEGIN IMMEDIATE` to close the validation/write race.

```python
# marvis/routers/validation_contracts.py
from fastapi import APIRouter, Request

from marvis.api_schemas import ValidationInputConfirmationRequest
from marvis.db import TaskRepository
from marvis.errors import conflict, not_found, unprocessable
from marvis.repositories.validation_contracts import (
    ValidationContractRepository,
    ValidationContractRevisionConflict,
)
from marvis.validation.input_confirmation import validate_confirmation_against_materials
from marvis.validation.input_contracts import (
    ValidationInputConfirmation,
    transformation_spec_from_dict,
)
from marvis.validation_materials import resolve_selected_validation_materials


router = APIRouter(prefix="/api", tags=["validation-contracts"])


def _task_repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _contract_repo(request: Request) -> ValidationContractRepository:
    return ValidationContractRepository(request.app.state.settings.db_path)


@router.get("/tasks/{task_id}/validation-input-contract")
def get_validation_input_contract(task_id: str, request: Request) -> dict:
    record = _contract_repo(request).get(task_id)
    if record is None:
        raise not_found("validation input contract not found")
    return record.to_api_payload()


@router.put("/tasks/{task_id}/validation-input-contract")
def confirm_validation_input_contract(
    task_id: str, payload: ValidationInputConfirmationRequest, request: Request
) -> dict:
    task = _task_repo(request).get_task(task_id)
    if task is None:
        raise not_found("validation task not found")
    current = _contract_repo(request).get(task_id)
    if current is None:
        raise not_found("validation input contract not found")
    try:
        paths = resolve_selected_validation_materials(task)
        requested = ValidationInputConfirmation(
                target_col=payload.target_col,
                positive_label=payload.positive_label,
                negative_label=payload.negative_label,
                split_col=payload.split_col,
                split_value_mapping=payload.split_value_mapping,
                time_col=payload.time_col,
                time_granularity=payload.time_granularity,
                pmml_output_field=payload.pmml_output_field,
                model_params=payload.model_params,
                metadata_sheet=payload.metadata_sheet,
                feature_col=payload.feature_col,
                category_col=payload.category_col,
                importance_col=payload.importance_col,
                transformations=tuple(
                    transformation_spec_from_dict(item) for item in payload.transformations
                ),
            )
        validated = validate_confirmation_against_materials(
            contract=current.contract,
            sample_path=paths.sample,
            dictionary_path=paths.dictionary,
            requested=requested,
        )
        record = _contract_repo(request).confirm(
            task_id, validated.values,
            expected_revision=payload.revision,
            resolved_sample_schema=validated.sample_schema,
            resolved_feature_metadata=validated.feature_metadata,
        )
    except ValidationContractRevisionConflict as exc:
        raise conflict(str(exc)) from exc
    except ValueError as exc:
        raise unprocessable(str(exc)) from exc
    return record.to_api_payload()
```

- [ ] **Step 5: Write API tests for ambiguity, confirmation, stale revision, and material invalidation**

First pin the pure validation boundary independently of FastAPI:

```python
# tests/validation/test_input_confirmation.py
from dataclasses import replace

import pandas as pd
import pytest

from marvis.files import sha256_file
from marvis.validation.input_confirmation import validate_confirmation_against_materials
from tests.validation_builders import (
    make_candidate_contract,
    make_validation_confirmation,
)
from tests.validation_material_builders import write_validation_material_bundle


def test_confirmation_validator_reads_real_selected_columns_and_metadata(tmp_path):
    bundle = write_validation_material_bundle(
        tmp_path / "bundle",
        notebook_source="RMC_TARGET_COL='y'",
    )
    contract = make_candidate_contract(material_hashes={
        "notebook": sha256_file(bundle.notebook_path),
        "sample": sha256_file(bundle.sample_path),
        "pmml": sha256_file(bundle.pmml_path),
        "dictionary": sha256_file(bundle.dictionary_path),
    })
    validated = validate_confirmation_against_materials(
        contract=contract,
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=replace(make_validation_confirmation(), metadata_sheet=None),
    )
    assert validated.sample_schema.row_count == 4
    assert validated.feature_metadata.coverage.importance == 1.0


def test_confirmation_validator_rejects_unobserved_positive_label(tmp_path):
    bundle = write_validation_material_bundle(
        tmp_path / "bundle",
        notebook_source="RMC_TARGET_COL='y'",
    )
    contract = make_candidate_contract(material_hashes={
        "notebook": sha256_file(bundle.notebook_path),
        "sample": sha256_file(bundle.sample_path),
        "pmml": sha256_file(bundle.pmml_path),
        "dictionary": sha256_file(bundle.dictionary_path),
    })
    requested = replace(
        make_validation_confirmation(), positive_label=99, metadata_sheet=None
    )
    with pytest.raises(ValueError, match="positive label"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=requested,
        )


def test_confirmation_validator_preserves_numeric_split_values(tmp_path):
    bundle = write_validation_material_bundle(
        tmp_path / "numeric-split",
        notebook_source="RMC_TARGET_COL='y'",
    )
    sample = pd.read_parquet(bundle.sample_path)
    sample["split"] = [0, 1, 2, 2]
    sample.to_parquet(bundle.sample_path, index=False)
    contract = make_candidate_contract(material_hashes={
        "notebook": sha256_file(bundle.notebook_path),
        "sample": sha256_file(bundle.sample_path),
        "pmml": sha256_file(bundle.pmml_path),
        "dictionary": sha256_file(bundle.dictionary_path),
    })
    requested = replace(
        make_validation_confirmation(),
        split_value_mapping={"train": 0, "test": 1, "oot": 2},
        metadata_sheet=None,
    )
    validated = validate_confirmation_against_materials(
        contract=contract,
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=requested,
    )
    assert validated.values.split_value_mapping["oot"] == 2
```

```python
# tests/test_validation_input_contract_api.py
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from tests.validation_material_builders import (
    create_api_validation_task,
    write_validation_material_bundle,
)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace")) as value:
        yield value


def _valid_confirmation(*, revision: int) -> dict:
    return {
        "revision": revision,
        "target_col": "y",
        "positive_label": 1,
        "negative_label": 0,
        "split_col": "split",
        "split_value_mapping": {"train": "train", "test": "test", "oot": "oot"},
        "time_col": "apply_month",
        "time_granularity": "month",
        "pmml_output_field": "probability_1",
        "model_params": {},
        "metadata_sheet": None,
        "feature_col": "feature",
        "category_col": "category",
        "importance_col": "importance",
        "transformations": [],
    }


@pytest.fixture
def ambiguous_validation_task(client) -> str:
    root = client.app.state.settings.workspace / "ambiguous-bundle"
    bundle = write_validation_material_bundle(
        root,
        notebook_source=(
            "RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\n"
            "RMC_TIME_COL='apply_month'\n"
            "RMC_PMML_OUTPUT_FIELD='probability_1'\nRMC_MODEL_PARAMS={}\n"
        ),
    )
    task_id, scan = create_api_validation_task(client, bundle)
    contract = scan["validation_input_contract"]
    assert contract["status"] == "pending_confirmation"
    assert contract["needs_confirmation"] is True
    return task_id


@pytest.fixture
def ready_task(client, ambiguous_validation_task) -> str:
    before = client.get(
        f"/api/tasks/{ambiguous_validation_task}/validation-input-contract"
    ).json()
    response = client.put(
        f"/api/tasks/{ambiguous_validation_task}/validation-input-contract",
        json=_valid_confirmation(revision=before["revision"]),
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ready"
    return ambiguous_validation_task


def _replace_selected_pmml(client, task_id: str, new_name: str) -> None:
    materials = client.get(f"/api/tasks/{task_id}/materials").json()
    source_dir = Path(materials["source_dir"])
    selection = dict(materials["selection"])
    original = source_dir / selection["pmml_path"]
    replacement = source_dir / new_name
    replacement.write_bytes(original.read_bytes() + b"\n")
    selection["pmml_path"] = new_name
    response = client.put(f"/api/tasks/{task_id}/materials", json=selection)
    assert response.status_code == 200, response.text


def test_confirm_contract_rejects_stale_revision(client, ambiguous_validation_task):
    before = client.get(
        f"/api/tasks/{ambiguous_validation_task}/validation-input-contract"
    ).json()
    payload = _valid_confirmation(revision=before["revision"])
    assert client.put(
        f"/api/tasks/{ambiguous_validation_task}/validation-input-contract", json=payload
    ).status_code == 200
    stale = client.put(
        f"/api/tasks/{ambiguous_validation_task}/validation-input-contract", json=payload
    )
    assert stale.status_code == 409


def test_changing_selected_pmml_invalidates_previous_confirmation(client, ready_task):
    _replace_selected_pmml(client, ready_task, "second.pmml")
    rescanned = client.post(f"/api/tasks/{ready_task}/scan").json()
    assert rescanned["validation_input_contract"]["status"] != "ready"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("positive_label", 99, "positive label"),
        ("split_value_mapping", {"train": "train", "test": "test"}, "oot"),
        ("importance_col", "missing_importance", "metadata"),
        ("transformations", [{
            "operation": "python_eval", "output_field": "x",
            "input_fields": ["x1"], "params": {},
        }], "unsupported transformation"),
    ],
)
def test_confirmation_validation_errors_are_structured_422(
    client, ambiguous_validation_task, field, value, message
):
    before = client.get(
        f"/api/tasks/{ambiguous_validation_task}/validation-input-contract"
    ).json()
    payload = _valid_confirmation(revision=before["revision"])
    payload[field] = value
    response = client.put(
        f"/api/tasks/{ambiguous_validation_task}/validation-input-contract",
        json=payload,
    )
    assert response.status_code == 422
    assert message.lower() in response.json()["detail"].lower()


def test_confirmation_for_missing_task_is_404(client):
    response = client.put(
        "/api/tasks/missing/validation-input-contract",
        json=_valid_confirmation(revision=1),
    )
    assert response.status_code == 404
```

- [ ] **Step 6: Gate both manual and Agent execution on a ready contract**

Add one reusable guard to `marvis/repositories/validation_contracts.py`:

```python
# marvis/repositories/validation_contracts.py
def require_confirmed_validation_input_contract(
    repo: ValidationContractRepository, task_id: str
) -> ValidationInputContractRecord:
    record = repo.get(task_id)
    if record is None or record.status != "ready":
        raise ValueError("validation input contract requires confirmation")
    return record
```

Call it before the new PMML scoring stage in manual and Agent paths. Do not add a new `TaskStatus`; keep confirmation state in the contract record.

- [ ] **Step 7: Run scan/API/Agent-focused tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_input_confirmation.py \
  tests/test_api_scan_helpers.py \
  tests/test_api_v2.py \
  tests/test_validation_input_contract_api.py \
  tests/test_agent_api.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the confirmed input contract flow**

```bash
git add marvis/api_schemas.py marvis/api_scan_helpers.py marvis/app.py \
  marvis/routers/scans.py \
  marvis/routers/validation_contracts.py marvis/validation/input_confirmation.py \
  marvis/validation_materials.py marvis/validation/config.py marvis/validation/checks.py \
  marvis/repositories/validation_contracts.py \
  marvis/agent/validation_stages.py \
  tests/test_api_scan_helpers.py tests/test_api_v2.py \
  tests/test_validation_input_contract_api.py tests/validation/test_input_confirmation.py \
  tests/validation_material_builders.py
git commit -m "feat: confirm validation fields before scoring"
```

---

## Wave 2 — Batch PMML Scoring, Metrics, and Mandatory Stress

### Task 7: Replace Per-Row PMML Calls with DataFrame Batch Scoring

**Files:**
- Modify: `marvis/validation/pmml_scoring.py`
- Modify: `marvis/validation/results.py`
- Modify: `tests/validation/test_pmml_scoring.py`
- Modify: `tests/validation/test_results.py`
- Create: `tests/validation/test_pmml_scoring_performance_contract.py`

**Interfaces:**
- Consumes: one loaded pypmml `Model`, a DataFrame chunk, and confirmed output field.
- Produces: `PmmlScorer.score_chunk(dataframe: pd.DataFrame) -> pd.Series` with one JVM call per chunk and no silent row loss.

- [ ] **Step 1: Write a failing one-call batch test**

```python
# tests/validation/test_pmml_scoring.py
def test_dataframe_scoring_calls_model_predict_once_for_the_whole_chunk():
    class BatchModel:
        def __init__(self):
            self.calls = []

        def predict(self, value):
            self.calls.append(value.copy())
            return pd.DataFrame({"probability(1.0)": [0.1, 0.2, 0.3]})

    model = BatchModel()
    scorer = PmmlScorer(model=model, positive_output_field="probability_1")
    scores = scorer.score_chunk(pd.DataFrame({"x": [1, 2, 3]}))
    assert len(model.calls) == 1
    assert model.calls[0].shape == (3, 1)
    assert scores.tolist() == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run and verify the current scorer has no batch API**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_pmml_scoring.py::test_dataframe_scoring_calls_model_predict_once_for_the_whole_chunk -q
```

Expected: FAIL because `PmmlScorer` has no `score_chunk`, or because the current implementation calls `predict` three times.

- [ ] **Step 3: Implement one-call DataFrame scoring and vectorized output extraction**

First add the scoring audit type used by this and every later task:

```python
# marvis/validation/results.py
import dataclasses
from dataclasses import asdict, dataclass, field
import math
import re


@dataclass(frozen=True)
class PmmlScoringResult:
    schema_version: str
    cache_key: str
    pmml_sha256: str
    sample_sha256: str
    engine: str
    engine_version: str
    output_field: str
    input_row_count: int
    success_count: int
    failure_count: int
    null_count: int
    non_finite_count: int
    elapsed_seconds: float
    rows_per_second: float
    chunk_size: int
    required_input_count: int
    missing_inputs: list[str]
    score_artifact_path: str
    score_artifact_sha256: str
    status: str
    bounded_errors: list[str]


def validate_pmml_scoring_result_fields(
    result: PmmlScoringResult,
) -> PmmlScoringResult:
    integer_fields = (
        "input_row_count", "success_count", "failure_count", "null_count",
        "non_finite_count", "chunk_size", "required_input_count",
    )
    for name in integer_fields:
        value = getattr(result, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid non-negative PMML scoring integer: {name}")
    for name in ("elapsed_seconds", "rows_per_second"):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid PMML scoring number: {name}")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"invalid finite PMML scoring number: {name}")
    if result.input_row_count <= 0 or result.chunk_size <= 0:
        raise ValueError("PMML scoring row count and chunk size must be positive")
    if result.success_count + result.failure_count != result.input_row_count:
        raise ValueError("PMML scoring success/failure counts do not add to input")
    if result.failure_count != result.null_count + result.non_finite_count:
        raise ValueError("PMML scoring failure detail counts are inconsistent")
    if result.status not in {"pass", "failed"}:
        raise ValueError("invalid PMML scoring status")
    if not isinstance(result.missing_inputs, list) or not all(
        isinstance(value, str) and value for value in result.missing_inputs
    ):
        raise ValueError("invalid PMML scoring missing_inputs")
    if not isinstance(result.bounded_errors, list) or not all(
        isinstance(value, str) and value for value in result.bounded_errors
    ):
        raise ValueError("invalid PMML scoring bounded_errors")
    if result.status == "pass" and (
        result.failure_count or result.null_count or result.non_finite_count
        or result.missing_inputs or result.bounded_errors
    ):
        raise ValueError("passing PMML scoring evidence contains failures")
    for name in ("cache_key", "pmml_sha256", "sample_sha256", "score_artifact_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(result, name))):
            raise ValueError(f"invalid PMML scoring SHA-256 field: {name}")
    for name in ("engine", "engine_version", "output_field", "score_artifact_path"):
        if not isinstance(getattr(result, name), str) or not getattr(result, name).strip():
            raise ValueError(f"missing PMML scoring identity field: {name}")
    return result


def pmml_scoring_result_from_dict(payload: dict[str, Any]) -> PmmlScoringResult:
    expected = {field.name for field in dataclasses.fields(PmmlScoringResult)}
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown or missing:
        raise ValueError(
            f"invalid PMML scoring result; missing={missing}, unknown={unknown}"
        )
    result = PmmlScoringResult(**payload)
    if result.schema_version != "marvis.pmml_scoring.v1":
        raise ValueError(f"unsupported PMML scoring schema: {result.schema_version}")
    return validate_pmml_scoring_result_fields(result)
```

```python
# marvis/validation/pmml_scoring.py
from collections import OrderedDict
import threading


@dataclass(frozen=True)
class PmmlScorer:
    model: Model
    positive_output_field: str

    def score_chunk(self, dataframe: pd.DataFrame) -> pd.Series:
        prediction = self.model.predict(dataframe)
        frame = _prediction_frame(prediction, expected_rows=len(dataframe))
        output_column = _resolve_output_column(frame.columns, self.positive_output_field)
        values = pd.to_numeric(frame[output_column], errors="coerce")
        if len(values) != len(dataframe):
            raise ValueError(
                f"PMML scorer returned {len(values)} rows for {len(dataframe)} inputs"
            )
        return pd.Series(values.to_numpy(), index=dataframe.index, name="pmml_score")

    def score(self, dataframe: pd.DataFrame) -> list[float | None]:
        values = self.score_chunk(dataframe)
        return [None if pd.isna(value) else float(value) for value in values]


class TaskPmmlScorerRegistry:
    """Bounded process-local reuse across baseline and stress stages."""

    def __init__(self, max_tasks: int = 4):
        self.max_tasks = max_tasks
        self.lock = threading.RLock()
        self.entries: OrderedDict[
            str, tuple[tuple[str, str], PmmlScorer]
        ] = OrderedDict()

    def get(
        self, *, task_id: str, pmml_path: Path, pmml_sha256: str,
        output_field: str,
    ) -> PmmlScorer:
        identity = (pmml_sha256, output_field)
        with self.lock:
            current = self.entries.get(task_id)
            if current is not None and current[0] == identity:
                self.entries.move_to_end(task_id)
                return current[1]
            scorer = PmmlScorer(
                model=Model.fromFile(str(pmml_path)),
                positive_output_field=output_field,
            )
            self.entries[task_id] = (identity, scorer)
            self.entries.move_to_end(task_id)
            while len(self.entries) > self.max_tasks:
                self.entries.popitem(last=False)
            return scorer

    def clear(self, task_id: str) -> None:
        with self.lock:
            self.entries.pop(task_id, None)


TASK_PMML_SCORERS = TaskPmmlScorerRegistry()


def _prediction_frame(prediction: Any, *, expected_rows: int) -> pd.DataFrame:
    if isinstance(prediction, pd.DataFrame):
        return prediction.reset_index(drop=True)
    if isinstance(prediction, pd.Series):
        return prediction.to_frame().T if expected_rows == 1 else prediction.to_frame()
    if isinstance(prediction, list):
        return pd.DataFrame.from_records(prediction)
    if isinstance(prediction, dict) and expected_rows == 1:
        return pd.DataFrame.from_records([prediction])
    raise TypeError(f"unsupported PMML batch prediction type: {type(prediction).__name__}")
```

`_resolve_output_column` must reuse the existing probability alias rules and raise a bounded error listing available output columns when no alias matches.

- [ ] **Step 4: Add null, non-finite, order, and no-row-loop tests**

```python
# tests/validation/test_pmml_scoring_performance_contract.py
import inspect

import pandas as pd

from marvis.validation.pmml_scoring import PmmlScorer, TaskPmmlScorerRegistry


def test_scorer_source_contains_no_record_loop():
    source = inspect.getsource(PmmlScorer.score_chunk)
    assert "to_dict(orient=\"records\")" not in source
    assert "for record" not in source


def test_score_chunk_preserves_input_order_and_marks_nulls():
    class StaticBatchModel:
        def predict(self, _frame):
            return pd.DataFrame({"probability_1": [0.3, None, float("inf")]})
    model = StaticBatchModel()
    scorer = PmmlScorer(model=model, positive_output_field="probability_1")
    scores = scorer.score_chunk(pd.DataFrame({"x": [30, 10, 20]}, index=[7, 2, 9]))
    assert scores.index.tolist() == [7, 2, 9]
    assert scores.iloc[0] == 0.3
    assert pd.isna(scores.iloc[1])
    assert scores.iloc[2] == float("inf")


def test_real_pmml_dataframe_batch_matches_legacy_single_row_semantics():
    from pathlib import Path

    import numpy as np
    from pypmml import Model

    from marvis.validation.pmml_scoring import _prediction_score_value

    frame = pd.DataFrame({
        "x1": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "x2": [0.0, 1.0, 0.0, 1.0, 0.0],
    }, index=[8, 3, 7, 1, 9])
    model = Model.fromFile(str(Path("tests/fixtures/min_lr.pmml")))
    legacy = np.asarray([
        float(_prediction_score_value(model.predict(record), "probability_1"))
        for record in frame.to_dict(orient="records")
    ])
    batch = PmmlScorer(model, "probability_1").score_chunk(frame).to_numpy(dtype=float)
    np.testing.assert_allclose(batch, legacy, rtol=1e-12, atol=1e-12)


def test_task_scorer_registry_reuses_one_loaded_model_and_invalidates_on_hash(monkeypatch):
    from pathlib import Path

    loads = []

    class LoadedModel:
        def predict(self, frame):
            return pd.DataFrame({"probability_1": [0.5] * len(frame)})

    monkeypatch.setattr(
        "marvis.validation.pmml_scoring.Model.fromFile",
        lambda path: loads.append(path) or LoadedModel(),
    )
    registry = TaskPmmlScorerRegistry(max_tasks=2)
    first = registry.get(
        task_id="t1", pmml_path=Path("one.pmml"),
        pmml_sha256="a", output_field="probability_1",
    )
    second = registry.get(
        task_id="t1", pmml_path=Path("one.pmml"),
        pmml_sha256="a", output_field="probability_1",
    )
    third = registry.get(
        task_id="t1", pmml_path=Path("one.pmml"),
        pmml_sha256="b", output_field="probability_1",
    )
    assert first is second and third is not second
    assert loads == ["one.pmml", "one.pmml"]
```

The stage-level code in Task 8, not `PmmlScorer`, owns the hard finite-value gate. Retain `_prediction_score_value` as the historical one-record semantic oracle until this parity test and the later XGBoost/LightGBM fixture smokes pass; it is not used by the production batch loop. Task state prevents baseline/stress overlap for one task, so a task-owned scorer is never used concurrently; distinct tasks receive distinct scorer objects. The registry is a bounded acceleration cache, not persisted state: process restart safely reloads by PMML hash/output identity, and terminal task cleanup calls `clear(task_id)`.

- [ ] **Step 5: Run PMML scoring tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_pmml_scoring.py \
  tests/validation/test_pmml_scoring_performance_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the batch scorer**

```bash
git add marvis/validation/pmml_scoring.py tests/validation/test_pmml_scoring.py \
  marvis/validation/results.py tests/validation/test_results.py \
  tests/validation/test_pmml_scoring_performance_contract.py
git commit -m "perf: batch PMML scoring across the JVM boundary"
```

### Task 8: Stream the Sample into a Hashed PMML Score Sidecar

**Files:**
- Modify: `pyproject.toml`
- Create: `marvis/validation/sample_chunks.py`
- Create: `marvis/validation/pmml_score_artifacts.py`
- Create: `tests/validation/test_sample_chunks.py`
- Create: `tests/validation/test_pmml_score_artifacts.py`
- Modify: `marvis/validation/input_contracts.py`

**Interfaces:**
- Consumes: confirmed `ValidationInputContract`, `PmmlInputManifest`, sample path, PMML path, output path, cancellation callback.
- Produces: `PmmlScoringResult` JSON data plus `pmml_scores.parquet` with stable `row_id` and `pmml_score`.

Add explicit runtime dependency `filelock>=3.13,<4`; Tasks 11–12 reuse it for cross-process cache locks.

- [ ] **Step 1: Write failing chunk-reader tests for all supported formats**

```python
# tests/validation/test_sample_chunks.py
import pandas as pd

from marvis.validation.sample_chunks import iter_sample_chunks


def test_csv_chunks_project_columns_and_assign_contiguous_row_ids(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": range(5), "y": [0, 1, 0, 1, 0], "unused": range(5)}).to_csv(
        path, index=False
    )
    chunks = list(iter_sample_chunks(path, columns=("x", "y"), chunk_size=2))
    assert [chunk.frame.columns.tolist() for chunk in chunks] == [["x", "y"]] * 3
    assert [chunk.row_ids.tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
```

- [ ] **Step 2: Run and verify missing module failure**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_sample_chunks.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Wrap the shared projected reader with stable row IDs**

```python
# marvis/validation/sample_chunks.py
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.validation.input_contracts import SampleSchema
from marvis.validation.sample_schema import iter_sample_projection


@dataclass(frozen=True)
class SampleChunk:
    row_ids: np.ndarray
    frame: pd.DataFrame


def iter_sample_chunks(
    path: Path, *, columns: tuple[str, ...], chunk_size: int,
    schema: SampleSchema | None = None,
) -> Iterator[SampleChunk]:
    offset = 0
    for frame in iter_sample_projection(
        path, columns=columns, chunk_size=chunk_size, schema=schema
    ):
        frame = frame.reset_index(drop=True)
        row_ids = np.arange(offset, offset + len(frame), dtype=np.int64)
        yield SampleChunk(row_ids=row_ids, frame=frame)
        offset += len(frame)


def read_selected_columns(
    path: Path, *, columns: tuple[str, ...], chunk_size: int = 100_000,
    schema: SampleSchema | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    frames = []
    for chunk in iter_sample_chunks(
        path, columns=columns, chunk_size=chunk_size, schema=schema
    ):
        if cancellation_check is not None:
            cancellation_check()
        frames.append(chunk.frame)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, axis=0, ignore_index=True)
```

All format/encoding/sheet behavior comes from Task 3's `iter_sample_projection`; this module adds only stable contiguous row IDs. Add tests that pass the inspected GB18030 CSV schema and the confirmed sole-sheet XLSX schema through `read_selected_columns`, proving metrics use the exact same encoding/sheet decision as scoring. Excel may read its sole sheet into memory because the format lacks a stable streaming reader, but it still emits scoring chunks; XLSX size/row caps remain mandatory and it is never described as bounded-memory.

- [ ] **Step 4: Write the failing full-scoring artifact test**

```python
# tests/validation/test_pmml_score_artifacts.py
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from marvis.validation.pmml_score_artifacts import run_pmml_scoring


MIN_LR_PMML = Path("tests/fixtures/min_lr.pmml")


def test_run_pmml_scoring_writes_all_rows_and_audit_counts(tmp_path, ready_contract):
    sample = tmp_path / "sample.parquet"
    pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [0.0, 0.0, 1.0]}).to_parquet(sample)
    result = run_pmml_scoring(
        contract=ready_contract,
        sample_path=sample,
        pmml_path=MIN_LR_PMML,
        score_path=tmp_path / "pmml_scores.parquet",
        chunk_size=2,
    )
    scored = pq.read_table(tmp_path / "pmml_scores.parquet").to_pandas()
    assert scored["row_id"].tolist() == [0, 1, 2]
    assert result.input_row_count == result.success_count == 3
    assert result.failure_count == result.null_count == result.non_finite_count == 0
    assert result.status == "pass"
```

- [ ] **Step 5: Implement atomic Parquet scoring and cache identity**

```python
# marvis/validation/pmml_score_artifacts.py
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from time import monotonic
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock, Timeout

from marvis.files import sha256_file
from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    required_transformation_inputs,
)
from marvis.validation.input_contracts import ValidationInputContract
from marvis.validation.pmml_scoring import PmmlScorer, load_pmml_scorer
from marvis.validation.results import (
    PmmlScoringResult,
    validate_pmml_scoring_result_fields,
)
from marvis.validation.sample_chunks import iter_sample_chunks


SCORING_SCHEMA = "marvis.pmml_scoring.v1"


class AtomicScoreWriter:
    def __init__(self, final_path: Path):
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self.final_path = final_path
        self.staging_path = final_path.with_name(
            f".{final_path.name}.{uuid4().hex}.staging"
        )
        self.schema = pa.schema([("row_id", pa.int64()), ("pmml_score", pa.float64())])
        self.writer = pq.ParquetWriter(self.staging_path, self.schema)
        self.closed = False

    def write(self, row_ids: np.ndarray, scores: np.ndarray) -> None:
        if len(row_ids) != len(scores):
            raise ValueError("row_id and PMML score lengths differ")
        self.writer.write_table(
            pa.table({"row_id": row_ids, "pmml_score": scores}, schema=self.schema)
        )

    def commit(self) -> tuple[Path, str]:
        if not self.closed:
            self.writer.close()
            self.closed = True
        os.replace(self.staging_path, self.final_path)
        return self.final_path, sha256_file(self.final_path)

    def rollback(self) -> None:
        if not self.closed:
            self.writer.close()
            self.closed = True
        self.staging_path.unlink(missing_ok=True)


def atomic_score_writer(final_path: Path) -> AtomicScoreWriter:
    return AtomicScoreWriter(final_path)


@dataclass
class _ScoreCounts:
    input_row_count: int = 0
    null_count: int = 0
    non_finite_count: int = 0
    latest_invalid_count: int = 0

    def observe(self, scores: pd.Series) -> None:
        numeric = pd.to_numeric(scores, errors="coerce")
        null_mask = numeric.isna().to_numpy()
        non_finite_mask = (~null_mask) & ~np.isfinite(numeric.to_numpy(dtype=float))
        self.input_row_count += len(numeric)
        self.null_count += int(null_mask.sum())
        self.non_finite_count += int(non_finite_mask.sum())
        self.latest_invalid_count = int(null_mask.sum() + non_finite_mask.sum())

    def latest_invalid_message(self) -> str:
        return (
            "null PMML score or non-finite PMML score in latest chunk: "
            f"count={self.latest_invalid_count}"
        )

    def to_result(self, **kwargs: Any) -> PmmlScoringResult:
        if self.input_row_count == 0:
            raise ValueError("validation sample contains no rows")
        failure_count = self.null_count + self.non_finite_count
        elapsed = float(kwargs.pop("elapsed_seconds"))
        return PmmlScoringResult(
            input_row_count=self.input_row_count,
            success_count=self.input_row_count - failure_count,
            failure_count=failure_count,
            null_count=self.null_count,
            non_finite_count=self.non_finite_count,
            elapsed_seconds=elapsed,
            rows_per_second=self.input_row_count / elapsed if elapsed > 0 else 0.0,
            status="pass" if failure_count == 0 else "failed",
            bounded_errors=[],
            **kwargs,
        )


def raise_if_cancelled(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def pypmml_engine_version() -> str:
    return importlib.metadata.version("pypmml")


def sha256_file_cancellable(
    path: Path, cancellation_check: Callable[[], None] | None,
    *, block_size: int = 8 * 1024 * 1024,
) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            raise_if_cancelled(cancellation_check)
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    raise_if_cancelled(cancellation_check)
    return digest.hexdigest()


def copy_file_cancellable(
    source: Path, destination: Path,
    *, cancellation_check: Callable[[], None] | None,
    block_size: int = 8 * 1024 * 1024,
) -> None:
    with source.open("rb") as reader, destination.open("wb") as writer:
        while True:
            raise_if_cancelled(cancellation_check)
            block = reader.read(block_size)
            if not block:
                break
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, destination)
    raise_if_cancelled(cancellation_check)


@contextmanager
def cancellable_file_lock(
    path: Path, cancellation_check: Callable[[], None] | None,
) -> Iterator[None]:
    lock = FileLock(str(path))
    while True:
        raise_if_cancelled(cancellation_check)
        try:
            lock.acquire(timeout=0.25)
            break
        except Timeout:
            continue
    try:
        yield
    finally:
        lock.release()


def validate_pmml_score_artifact(
    result: PmmlScoringResult,
    score_path: Path,
    *,
    expected_cache_key: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> PmmlScoringResult:
    """Validate the exact artifact a downstream stage is about to consume."""
    validate_pmml_scoring_result_fields(result)
    raise_if_cancelled(cancellation_check)
    if result.schema_version != SCORING_SCHEMA or result.status != "pass":
        raise ValueError("PMML scoring result is not a passing v1 artifact")
    if expected_cache_key is not None and result.cache_key != expected_cache_key:
        raise ValueError("PMML scoring result/cache key mismatch")
    if result.failure_count or result.success_count != result.input_row_count:
        raise ValueError("PMML scoring result counts are incomplete")
    if result.input_row_count <= 0:
        raise ValueError("PMML scoring result contains no rows")
    if not score_path.is_file():
        raise ValueError("PMML score sidecar is missing")
    if result.score_artifact_sha256 != sha256_file_cancellable(
        score_path, cancellation_check
    ):
        raise ValueError("PMML score sidecar hash mismatch")
    parquet = pq.ParquetFile(score_path)
    if parquet.schema_arrow != pa.schema([
        ("row_id", pa.int64()), ("pmml_score", pa.float64())
    ]):
        raise ValueError("PMML score sidecar schema mismatch")
    offset = 0
    for batch in parquet.iter_batches(
        columns=["row_id", "pmml_score"], batch_size=100_000
    ):
        raise_if_cancelled(cancellation_check)
        frame = batch.to_pandas()
        row_ids = frame["row_id"].to_numpy(dtype=np.int64)
        expected = np.arange(offset, offset + len(frame), dtype=np.int64)
        if not np.array_equal(row_ids, expected):
            raise ValueError("PMML score sidecar row_id is not contiguous")
        if not np.isfinite(frame["pmml_score"].to_numpy(dtype=float)).all():
            raise ValueError("PMML score sidecar contains a non-finite score")
        offset += len(frame)
    if offset != result.input_row_count:
        raise ValueError("PMML score sidecar row count mismatch")
    raise_if_cancelled(cancellation_check)
    return replace(result, score_artifact_path=str(score_path))


def pmml_scoring_cache_key(
    *, pmml_sha256: str, sample_sha256: str, output_field: str,
    engine_version: str, transformation_sha256: str,
) -> str:
    canonical = json.dumps({
        "schema": SCORING_SCHEMA,
        "pmml": pmml_sha256,
        "sample": sample_sha256,
        "output": output_field,
        "engine": engine_version,
        "transformations": transformation_sha256,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def run_pmml_scoring(
    *, contract: ValidationInputContract, sample_path: Path, pmml_path: Path,
    score_path: Path, chunk_size: int, scorer: PmmlScorer | None = None,
    pmml_sha256: str | None = None, sample_sha256: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> PmmlScoringResult:
    manifest = contract.require_pmml_manifest()
    resolved_pmml_sha256 = pmml_sha256 or sha256_file(pmml_path)
    resolved_sample_sha256 = sample_sha256 or sha256_file(sample_path)
    active_scorer = scorer or load_pmml_scorer(
        pmml_path, contract.require_output_field()
    )
    writer = atomic_score_writer(score_path)
    started = monotonic()
    counts = _ScoreCounts()
    try:
        for chunk in iter_sample_chunks(
            sample_path,
            columns=required_transformation_inputs(
                manifest.raw_required_fields, contract.transformations
            ),
            chunk_size=chunk_size,
            schema=contract.require_sample_schema(),
        ):
            raise_if_cancelled(cancellation_check)
            scoring_frame = apply_confirmed_transformations(chunk.frame, contract.transformations)
            scoring_frame = scoring_frame.loc[:, list(manifest.raw_required_fields)]
            scores = active_scorer.score_chunk(scoring_frame)
            counts.observe(scores)
            if counts.latest_invalid_count:
                raise ValueError(counts.latest_invalid_message())
            writer.write(chunk.row_ids, scores.to_numpy(dtype=float))
        if counts.input_row_count == 0:
            raise ValueError("validation sample contains no rows")
        raise_if_cancelled(cancellation_check)
        final_path, digest = writer.commit()
    except Exception:
        writer.rollback()
        raise
    return counts.to_result(
        schema_version=SCORING_SCHEMA,
        cache_key=pmml_scoring_cache_key(
            pmml_sha256=resolved_pmml_sha256,
            sample_sha256=resolved_sample_sha256,
            output_field=contract.require_output_field(),
            engine_version=pypmml_engine_version(),
            transformation_sha256=sha256(
                json.dumps(
                    [asdict(row) for row in contract.transformations],
                    ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        ),
        pmml_sha256=resolved_pmml_sha256,
        sample_sha256=resolved_sample_sha256,
        output_field=contract.require_output_field(),
        engine="pypmml-pmml4s-batch",
        engine_version=pypmml_engine_version(),
        chunk_size=chunk_size,
        required_input_count=len(manifest.raw_required_fields),
        missing_inputs=[],
        elapsed_seconds=monotonic() - started,
        score_artifact_path=str(final_path),
        score_artifact_sha256=digest,
    )
```

Use `pyarrow.parquet.ParquetWriter` with columns `row_id: int64` and `pmml_score: float64`; write to an artifact staging path and promote only after every row succeeds. The cache key is SHA-256 over PMML hash, sample hash, output field, scorer version, transformation hash, and chunk-independent schema version. Import `replace` alongside `asdict`/`dataclass`. `validate_pmml_score_artifact` is the single public verifier for both shared cache entries and the copied task-local sidecar; it hashes the exact file being consumed and validates schema, audit-field invariants, counts, contiguous row IDs, and finite scores in bounded cancellable batches. Add parameterized tampered-JSON tests for nonzero/negative null or non-finite counts, inconsistent success/failure totals, nonempty `missing_inputs`/`bounded_errors` on pass, non-finite/negative elapsed or throughput, malformed hashes, and invalid list types; every case must fail before any downstream metric.

- [ ] **Step 6: Add hard-failure and cancellation tests**

```python
def test_scoring_rolls_back_on_any_null_score(tmp_path, ready_contract, monkeypatch):
    sample_path = tmp_path / "two_rows.parquet"
    pd.DataFrame({"x1": [0.0, 1.0], "x2": [1.0, 0.0]}).to_parquet(
        sample_path, index=False
    )
    class NullScorer:
        def score_chunk(self, _frame):
            return pd.Series([0.1, None])
    monkeypatch.setattr(
        "marvis.validation.pmml_score_artifacts.load_pmml_scorer",
        lambda *_a, **_k: NullScorer(),
    )
    with pytest.raises(ValueError, match="null PMML score"):
        run_pmml_scoring(
            contract=ready_contract,
            sample_path=sample_path,
            pmml_path=MIN_LR_PMML,
            score_path=tmp_path / "scores.parquet",
            chunk_size=2,
        )
    assert not (tmp_path / "scores.parquet").exists()
```

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_sample_chunks.py \
  tests/validation/test_pmml_score_artifacts.py -q
```

Expected: PASS.

```bash
git add pyproject.toml marvis/validation/input_contracts.py marvis/validation/sample_chunks.py \
  marvis/validation/pmml_score_artifacts.py tests/validation/test_sample_chunks.py \
  tests/validation/test_pmml_score_artifacts.py
git commit -m "feat: persist full-sample PMML scores"
```

### Task 9: Introduce the New Result Schema with Historical Compatibility

**Files:**
- Modify: `marvis/validation/results.py`
- Modify: `tests/validation/test_results.py`
- Modify: `tests/output/test_excel.py`
- Modify: `tests/test_metric_tables.py`
- Create: `tests/fixtures/legacy_validation_results.json`

**Interfaces:**
- Consumes: `PmmlScoringResult` defined in Task 7 or historical `ReproducibilityResult`.
- Produces: versioned `ValidationResults` where new tasks use `pmml_scoring` and old payloads remain readable without rewriting.

- [ ] **Step 1: Write failing new- and legacy-schema tests**

```python
# tests/validation/test_results.py
import json
from pathlib import Path

from marvis.validation.results import (
    ValidationResults,
    validation_results_from_dict,
    validation_results_to_dict,
)
from tests.output.test_excel import _make_pmml_results


LEGACY_FIXTURE = Path("tests/fixtures/legacy_validation_results.json")


def _make_pmml_scoring_results() -> ValidationResults:
    return _make_pmml_results()


def test_new_results_round_trip_uses_pmml_scoring_not_reproducibility():
    results = _make_pmml_scoring_results()
    payload = validation_results_to_dict(results)
    assert payload["schema_version"] == "marvis.validation_results.v2"
    assert payload["pmml_scoring"]["status"] == "pass"
    assert "reproducibility" not in payload
    assert validation_results_from_dict(payload) == results


def test_legacy_results_fixture_remains_readable():
    payload = json.loads(LEGACY_FIXTURE.read_text(encoding="utf-8"))
    results = validation_results_from_dict(payload)
    assert results.schema_version == "marvis.validation_results.v1"
    assert results.reproducibility is not None
    assert results.pmml_scoring is None
```

- [ ] **Step 2: Run and verify missing new schema fields**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_results.py::test_new_results_round_trip_uses_pmml_scoring_not_reproducibility \
  tests/validation/test_results.py::test_legacy_results_fixture_remains_readable -q
```

Expected: FAIL because `ValidationResults` has only `reproducibility` and no explicit serializer.

- [ ] **Step 3: Add the explicit versioned validation envelope and serializer**

```python
# marvis/validation/results.py
@dataclass(frozen=True)
class ValidationResults:
    model_name: str
    model_version: str
    algorithm: str
    target_type: Literal["binary"]
    basic_info: BasicInfoResult
    effectiveness: EffectivenessResult
    stress_test: StressTestResult
    # Compatibility-safe default: every pre-migration constructor remains v1.
    # New PMML code must opt into v2 explicitly.
    schema_version: str = "marvis.validation_results.v1"
    pmml_scoring: PmmlScoringResult | None = None
    reproducibility: ReproducibilityResult | None = None


def validation_results_to_dict(results: ValidationResults) -> dict[str, Any]:
    payload = asdict(results)
    if results.schema_version == "marvis.validation_results.v2":
        if results.pmml_scoring is None or results.reproducibility is not None:
            raise ValueError("v2 validation results require only pmml_scoring")
        payload.pop("reproducibility", None)
    elif results.schema_version == "marvis.validation_results.v1":
        if results.reproducibility is None or results.pmml_scoring is not None:
            raise ValueError("v1 validation results require only reproducibility")
        payload.pop("pmml_scoring", None)
    else:
        raise ValueError(f"unsupported validation results schema: {results.schema_version}")
    return payload


def validation_results_from_dict(payload: dict[str, Any]) -> ValidationResults:
    schema = str(payload.get("schema_version") or "")
    if not schema:
        schema = (
            "marvis.validation_results.v1"
            if "reproducibility" in payload
            else "marvis.validation_results.v2"
        )
    if schema == "marvis.validation_results.v2":
        if "reproducibility" in payload or "pmml_scoring" not in payload:
            raise ValueError("v2 validation results require only pmml_scoring")
        scoring = pmml_scoring_result_from_dict(dict(payload["pmml_scoring"]))
        reproducibility = None
    elif schema == "marvis.validation_results.v1":
        if "pmml_scoring" in payload:
            raise ValueError("v1 validation results cannot contain pmml_scoring")
        scoring = None
        reproducibility = _reproducibility_from_dict(
            dict(payload.get("reproducibility") or {})
        )
    else:
        raise ValueError(f"unsupported validation results schema: {schema}")
    return ValidationResults(
        model_name=str(payload.get("model_name") or ""),
        model_version=str(payload.get("model_version") or ""),
        algorithm=str(payload.get("algorithm") or ""),
        target_type="binary",
        basic_info=_basic_info_from_dict(dict(payload.get("basic_info") or {})),
        effectiveness=_effectiveness_from_dict(dict(payload.get("effectiveness") or {})),
        stress_test=_stress_test_from_dict(dict(payload.get("stress_test") or {})),
        schema_version=schema,
        pmml_scoring=scoring,
        reproducibility=reproducibility,
    )
```

The legacy fixture is a verbatim serialized current `_make_results()` payload captured before changing the dataclass. Keep the dataclass default at v1 so untouched historical constructors and the legacy `run_platform_validation` path cannot silently become an invalid “v2 + reproducibility” object. Every new PMML assembly point and `_make_pmml_results()` passes v2 explicitly. Reject a payload containing both canonical sections; tests that need both must keep two separate objects rather than weakening the production decoder. Add a repository-wide constructor test/search covering `tests/output/test_e2e_results_to_outputs.py`, legacy platform metrics, and every `ValidationResults(` call so each is explicitly legacy or new.

- [ ] **Step 4: Update shared test builders to expose separate legacy/new fixtures**

Keep `_make_results()` as a legacy result for historical renderer tests and change its constructor explicitly to `schema_version="marvis.validation_results.v1"`. Add this complete shared builder in `tests/output/test_excel.py`; do not mutate one fixture to contain both schemas:

```python
from dataclasses import replace

from marvis.validation.results import PmmlScoringResult


def _make_pmml_results() -> ValidationResults:
    legacy = _make_results()
    scoring = PmmlScoringResult(
        schema_version="marvis.pmml_scoring.v1",
        cache_key="c" * 64,
        pmml_sha256="p" * 64,
        sample_sha256="s" * 64,
        engine="pypmml-pmml4s-batch",
        engine_version="1.5.5",
        output_field="probability_1",
        input_row_count=3,
        success_count=3,
        failure_count=0,
        null_count=0,
        non_finite_count=0,
        elapsed_seconds=0.1,
        rows_per_second=30.0,
        chunk_size=2,
        required_input_count=2,
        missing_inputs=[],
        score_artifact_path="pmml_scores.parquet",
        score_artifact_sha256="a" * 64,
        status="pass",
        bounded_errors=[],
    )
    return replace(
        legacy,
        schema_version="marvis.validation_results.v2",
        pmml_scoring=scoring,
        reproducibility=None,
    )
```

- [ ] **Step 5: Run result and existing renderer tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_results.py \
  tests/output/test_excel.py \
  tests/test_metric_tables.py -q
```

Expected: PASS for both schemas.

- [ ] **Step 6: Commit the versioned result schema**

```bash
git add marvis/validation/results.py tests/validation/test_results.py \
  tests/output/test_excel.py tests/test_metric_tables.py \
  tests/fixtures/legacy_validation_results.json
git commit -m "feat: version PMML scoring validation results"
```

### Task 10: Compute Existing Metrics from the PMML Score Sidecar

**Files:**
- Modify: `marvis/validation/platform_metrics.py`
- Modify: `marvis/validation/sample_stats.py`
- Create: `tests/validation/test_platform_metrics_pmml_scores.py`
- Modify: `tests/validation/test_sample_stats.py`
- Modify: `tests/test_pipeline_v2.py`

**Interfaces:**
- Consumes: confirmed input contract, selected sample, `pmml_scores.parquet`, and normalized feature metadata.
- Produces: the same `BasicInfoResult` and `EffectivenessResult` fields as today, with `score_col` sourced only from PMML.

- [ ] **Step 1: Write a failing proof that sample score columns are ignored**

```python
# tests/validation/test_platform_metrics_pmml_scores.py
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.validation.binning import compute_ks
from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    FeatureMetadataRow,
    MetadataCoverage,
)
from marvis.validation.platform_metrics import compute_platform_validation_results
from tests.output.test_excel import _make_pmml_results


def test_platform_metrics_uses_pmml_sidecar_not_existing_sample_score(tmp_path, pmml_contract):
    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame({
        "pred": [0.99, 0.99, 0.99, 0.99],
        "y": [0, 1, 0, 1],
        "split": ["train", "test", "oot", "oot"],
        "apply_month": ["202601", "202602", "202603", "202603"],
    }).to_parquet(sample_path, index=False)
    score_path = tmp_path / "pmml_scores.parquet"
    pd.DataFrame({
        "row_id": [0, 1, 2, 3],
        "pmml_score": [0.1, 0.2, 0.3, 0.4],
    }).to_parquet(score_path, index=False)
    fixture_results = _make_pmml_results()
    results = compute_platform_validation_results(
        task=SimpleNamespace(model_name="fixture", model_version="v1"),
        contract=pmml_contract,
        sample_path=sample_path,
        score_path=score_path,
        scoring_result=fixture_results.pmml_scoring,
        metadata_resolution=FeatureMetadataResolution(
            schema_version="marvis.feature_metadata.v1",
            rows=(
                FeatureMetadataRow("x1", "内部", 0.6, "features", True),
                FeatureMetadataRow("x2", "征信", 0.4, "features", True),
            ),
            coverage=MetadataCoverage(1.0, 1.0, 1.0, 1.0),
            per_category_raw_fields={"内部": ("x1",), "征信": ("x2",)},
            extra_features=(),
            conflicts=(),
        ),
        stress_test=fixture_results.stress_test,
        settings=SimpleNamespace(
            bin_count=10, random_sample_size=1_000, random_seed=42
        ),
    )
    assert results.effectiveness.overall[-1].ks == pytest.approx(
        compute_ks([0.3, 0.4], [0, 1])
    )
    assert results.pmml_scoring is not None
```

The local `pmml_contract` fixture is a `ready` `ValidationInputContract` whose confirmed values are `target_col=y`, labels `0/1`, `split_col=split`, canonical split mapping, `time_col=apply_month`, output `probability_1`, and model params `{}`; it uses the two-feature manifest/metadata builders from Tasks 2–5 rather than mocking `require_*` methods.

- [ ] **Step 2: Run and verify the current code requires code-model scores**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_platform_metrics_pmml_scores.py -q
```

Expected: FAIL because `write_platform_validation_metrics` loads `code_model_scores_path`.

- [ ] **Step 3: Add a narrow analysis-frame loader**

```python
from collections.abc import Callable

from marvis.validation.input_confirmation import normalize_binary_target


def load_pmml_analysis_frame(
    *, sample_path: Path, score_path: Path, contract: ValidationInputContract,
    cancellation_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    target_col = str(contract.confirmed["target_col"])
    split_col = str(contract.confirmed["split_col"])
    time_col = str(contract.confirmed["time_col"])
    sample = read_selected_columns(
        sample_path,
        columns=required_transformation_inputs(
            (target_col, split_col, time_col), contract.transformations
        ),
        schema=contract.require_sample_schema(),
        cancellation_check=cancellation_check,
    )
    if cancellation_check is not None:
        cancellation_check()
    sample = apply_confirmed_transformations(sample, contract.transformations)
    scores = pd.read_parquet(score_path, columns=["row_id", "pmml_score"])
    if cancellation_check is not None:
        cancellation_check()
    if not np.array_equal(
        scores["row_id"].to_numpy(dtype=np.int64),
        np.arange(len(sample), dtype=np.int64),
    ):
        raise ValueError("PMML score sidecar row_id does not match the validation sample")
    positive = contract.confirmed["positive_label"]
    negative = contract.confirmed["negative_label"]
    target_codes = normalize_binary_target(
        sample[target_col], positive=positive, negative=negative
    )
    frame = pd.DataFrame({
        "__target__": target_codes,
        "__split__": sample[split_col],
        "__time__": sample[time_col],
    }).reset_index(drop=True)
    frame["__pmml_score__"] = scores["pmml_score"].to_numpy(dtype=float)
    return frame


def validation_config_from_input_contract(
    contract: ValidationInputContract, settings: Any
) -> ValidationConfig:
    split_values = dict(contract.confirmed["split_value_mapping"])
    if set(split_values) != {"train", "test", "oot"}:
        raise ValueError("split_value_mapping must map train/test/oot to sample values")
    return ValidationConfig(
        target_col="__target__",
        score_col="__pmml_score__",
        split_col="__split__",
        time_col="__time__",
        feature_columns=[],
        bin_count=int(settings.bin_count),
        random_sample_size=int(settings.random_sample_size),
        random_seed=int(settings.random_seed),
        score_decimal_places=6,
        split_values=split_values,
        data_dict_feature_col="feature",
        data_dict_category_col="category",
    )
```

This deliberately does not read every model feature into the metrics DataFrame.

- [ ] **Step 4: Split pure computation from artifact writing**

```python
def compute_platform_validation_results(
    *, task: TaskRecord, contract: ValidationInputContract, sample_path: Path,
    score_path: Path, scoring_result: PmmlScoringResult,
    metadata_resolution: FeatureMetadataResolution,
    stress_test: StressTestResult, settings: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> ValidationResults:
    if cancellation_check is not None:
        cancellation_check()
    config = validation_config_from_input_contract(contract, settings)
    sample_scored = load_pmml_analysis_frame(
        sample_path=sample_path,
        score_path=score_path,
        contract=contract,
        cancellation_check=cancellation_check,
    )
    basic_info = run_basic_info_from_metadata(
        sample=sample_scored,
        config=config,
        model_params=contract.require_model_params(),
        feature_metadata=metadata_resolution.rows,
        cancellation_check=cancellation_check,
    )
    if cancellation_check is not None:
        cancellation_check()
    effectiveness = compute_existing_effectiveness(
        sample_scored, config, cancellation_check=cancellation_check
    )
    if cancellation_check is not None:
        cancellation_check()
    return ValidationResults(
        model_name=task.model_name,
        model_version=task.model_version,
        algorithm=contract.require_algorithm(),
        target_type="binary",
        schema_version="marvis.validation_results.v2",
        pmml_scoring=scoring_result,
        basic_info=basic_info,
        effectiveness=effectiveness,
        stress_test=stress_test,
    )
```

Implement the two extracted pure helpers by moving, not rewriting, the current deterministic code:

```python
# marvis/validation/sample_stats.py
def run_basic_info_from_metadata(
    *, sample: pd.DataFrame, config: ValidationConfig,
    model_params: Mapping[str, JsonValue],
    feature_metadata: Sequence[FeatureMetadataRow],
    cancellation_check: Callable[[], None] | None = None,
) -> BasicInfoResult:
    if cancellation_check is not None:
        cancellation_check()
    split_summary, monthly_distribution, sample_period = _sample_distribution_rows(
        sample=sample, config=config
    )
    if cancellation_check is not None:
        cancellation_check()
    ranked = sorted(feature_metadata, key=lambda row: (-row.importance, row.feature))
    return BasicInfoResult(
        sample_period=sample_period,
        split_summary=split_summary,
        monthly_distribution=monthly_distribution,
        hyperparameters=dict(model_params),
        feature_importance=[
            FeatureImportanceRow(
                rank=index,
                feature=row.feature,
                importance=row.importance,
                category=row.category,
            )
            for index, row in enumerate(ranked, start=1)
        ],
    )


# marvis/validation/platform_metrics.py
def compute_existing_effectiveness(
    sample_scored: pd.DataFrame, config: ValidationConfig,
    *, cancellation_check: Callable[[], None] | None = None,
) -> EffectivenessResult:
    check = cancellation_check or (lambda: None)
    check()
    context = prepare_effectiveness_context(sample=sample_scored, config=config)
    check()
    overall = compute_overall_ks(sample=sample_scored, config=config)
    check()
    overall = compute_overall_psi(
        sample=sample_scored, config=config, context=context, overall=overall
    )
    check()
    bin_tables = compute_bin_tables(sample=sample_scored, config=config, context=context)
    check()
    monthly_ks = compute_monthly_ks(sample=sample_scored, config=config)
    check()
    monthly_psi = compute_monthly_psi(
        sample=sample_scored, config=config, context=context
    )
    check()
    psi_table = compute_psi_stability_table(sample=sample_scored, config=config)
    check()
    curves = compute_roc_ks_curves(sample=sample_scored, config=config)
    check()
    return build_effectiveness_result(
        overall=overall, bin_tables=bin_tables, monthly_ks=monthly_ks,
        monthly_psi=monthly_psi, psi_stability_table=psi_table,
        roc_ks_curves=curves,
    )
```

Extract `_sample_distribution_rows` from the current `run_basic_info`; both legacy `run_basic_info(model_meta_path=...)` and new `run_basic_info_from_metadata(...)` must call it. Keep all current effectiveness calls and defaults unchanged. Where the existing helpers iterate splits or months, add the same optional callback and checkpoint once per iteration so cancellation is not delayed until an entire multi-period calculation completes; legacy callers omit it. New v2 contracts use the normalized `__target__`, `__split__`, `__time__`, and `__pmml_score__` columns; legacy tasks retain `validation_config_from_contract` and `run_basic_info(model_meta_path=...)`.

- [ ] **Step 5: Add parity assertions for every deterministic metric section**

Create one fixture where code scores equal PMML sidecar scores and assert the old and new paths produce identical `basic_info` and `effectiveness` dictionaries.

```python
assert asdict(new_results.basic_info) == asdict(legacy_results.basic_info)
assert asdict(new_results.effectiveness) == asdict(legacy_results.effectiveness)
```

- [ ] **Step 6: Run metrics parity tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_platform_metrics_pmml_scores.py \
  tests/validation/test_sample_stats.py \
  tests/validation/test_effectiveness.py \
  tests/validation/test_binning.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit PMML-backed metrics**

```bash
git add marvis/validation/platform_metrics.py marvis/validation/sample_stats.py \
  tests/validation/test_platform_metrics_pmml_scores.py \
  tests/validation/test_sample_stats.py tests/test_pipeline_v2.py
git commit -m "feat: compute validation metrics from PMML scores"
```

### Task 11: Rescore Every OOT Stress Category with the Same PMML Engine

**Files:**
- Create: `marvis/validation/pmml_stress.py`
- Modify: `marvis/validation/stress_test.py`
- Create: `tests/validation/test_pmml_stress.py`
- Modify: `tests/validation/test_stress_test.py`
- Modify: `tests/validation/test_platform_metrics_stress_categories.py`

**Interfaces:**
- Consumes: baseline PMML sidecar, confirmed category-to-raw-fields mapping, OOT membership, sample path, loaded PMML scorer.
- Produces: one Parquet scenario per category and a strict `StressTestResult(status="completed")` or a stage failure.

- [ ] **Step 1: Write failing baseline-reuse and complete-OOT tests**

```python
# tests/validation/test_pmml_stress.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marvis.validation.config import ValidationConfig
from marvis.files import sha256_file
from marvis.validation.pmml_stress import (
    ScenarioScoreArtifact,
    load_or_run_stress_scenario,
    run_pmml_stress,
)


MIN_LR_PMML = Path("tests/fixtures/min_lr.pmml")


class RecordingBatchScorer:
    def __init__(self):
        self.calls: list[pd.DataFrame] = []

    def score_chunk(self, frame: pd.DataFrame) -> pd.Series:
        self.calls.append(frame.copy())
        return pd.Series(1.0 / (1.0 + np.exp(-(frame["x1"] - frame["x2"]))))


def test_stress_reuses_baseline_and_scores_each_category_on_complete_oot(
    tmp_path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame({
        "x1": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "x2": [0.0, 1.0, 0.0, 1.0, 0.0],
        "y": [0, 1, 0, 1, 0],
        "split": ["train", "test", "oot", "oot", "oot"],
        "apply_month": ["202601", "202602", "202603", "202603", "202603"],
    }).to_parquet(sample_path, index=False)
    baseline_score_path = tmp_path / "baseline.parquet"
    pd.DataFrame({
        "row_id": range(5),
        "pmml_score": [0.1, 0.2, 0.3, 0.7, 0.4],
    }).to_parquet(baseline_score_path, index=False)
    scorer = RecordingBatchScorer()
    monkeypatch.setattr("marvis.validation.pmml_stress.load_pmml_scorer", lambda *_a, **_k: scorer)
    result = run_pmml_stress(
        contract=ready_contract,
        config=ValidationConfig(
            target_col="__target__", score_col="__pmml_score__",
            split_col="__split__", time_col="__time__", bin_count=2,
        ),
        sample_path=sample_path,
        baseline_score_path=baseline_score_path,
        pmml_path=MIN_LR_PMML,
        scenario_dir=tmp_path / "stress",
        feature_categories={"征信": ("x1",), "内部": ("x2",)},
        chunk_size=2,
    )
    assert result.baseline.sample_count == 3
    assert sum(len(call) for call in scorer.calls if (call["x2"] == -9999).all()) == 3
    assert sum(len(call) for call in scorer.calls if (call["x1"] == -9999).all()) == 3
    assert [row.category for row in result.per_category] == ["征信", "内部"]
    assert result.status == "completed"
```

- [ ] **Step 2: Run and verify missing module failure**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_pmml_stress.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement chunked OOT scenario production**

Use the explicit `filelock>=3.13,<4` runtime dependency added in Task 8 for cross-process coordination on both POSIX and packaged Windows runtimes.

```python
# marvis/validation/pmml_stress.py
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import threading
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from marvis.artifacts import ArtifactUnitOfWork
from marvis.files import sha256_file
from marvis.validation.binning import (
    bin_distribution,
    bin_table,
    compute_ks,
    compute_psi,
    equal_frequency_bin_edges,
)
from marvis.validation.config import ValidationConfig
from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    required_transformation_inputs,
)
from marvis.validation.input_contracts import ValidationInputContract
from marvis.validation.pmml_score_artifacts import (
    atomic_score_writer,
    cancellable_file_lock,
    copy_file_cancellable,
    raise_if_cancelled,
    sha256_file_cancellable,
)
from marvis.validation.pmml_scoring import PmmlScorer, load_pmml_scorer
from marvis.validation.platform_metrics import load_pmml_analysis_frame
from marvis.validation.results import (
    StressBaseline,
    StressCategoryResult,
    StressTestResult,
)
from marvis.validation.sample_chunks import iter_sample_chunks
from marvis.validation.stress_test import STRESS_MISSING_VALUE


@dataclass(frozen=True)
class OotStressContext:
    row_ids: np.ndarray
    labels: np.ndarray
    baseline_scores: np.ndarray


@dataclass(frozen=True)
class ScenarioScoreArtifact:
    category: str
    path: Path
    row_count: int
    sha256: str


@dataclass(frozen=True)
class OotInputArtifact:
    path: Path
    row_count: int
    sha256: str


def materialize_oot_pmml_inputs(
    *, contract: ValidationInputContract, sample_path: Path,
    oot_row_ids: np.ndarray, output_path: Path, chunk_size: int,
    cancellation_check: Callable[[], None] | None,
) -> OotInputArtifact:
    manifest = contract.require_pmml_manifest()
    required = required_transformation_inputs(
        manifest.raw_required_fields, contract.transformations
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.{uuid4().hex}.staging")
    writer = None
    written = 0
    try:
        for chunk in iter_sample_chunks(
            sample_path, columns=required, chunk_size=chunk_size,
            schema=contract.require_sample_schema(),
        ):
            raise_if_cancelled(cancellation_check)
            mask = np.isin(chunk.row_ids, oot_row_ids, assume_unique=True)
            if not mask.any():
                continue
            frame = apply_confirmed_transformations(
                chunk.frame.loc[mask].reset_index(drop=True),
                contract.transformations,
            ).loc[:, list(manifest.raw_required_fields)]
            frame.insert(0, "row_id", chunk.row_ids[mask])
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(staging, table.schema)
            writer.write_table(table)
            written += len(frame)
        if writer is not None:
            writer.close()
            writer = None
        if written != len(oot_row_ids):
            raise ValueError(
                f"OOT PMML input row count mismatch: {written} != {len(oot_row_ids)}"
            )
        os.replace(staging, output_path)
        return OotInputArtifact(output_path, written, sha256_file(output_path))
    except Exception:
        if writer is not None:
            writer.close()
        staging.unlink(missing_ok=True)
        raise


def stress_cache_key(
    *, baseline_cache_key: str, category: str,
    raw_fields: tuple[str, ...], sentinel: float,
) -> str:
    payload = json.dumps({
        "schema": "marvis.pmml_stress.v1",
        "baseline_cache_key": baseline_cache_key,
        "category": category,
        "raw_fields": list(raw_fields),
        "sentinel": sentinel,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def load_oot_stress_context(
    *, sample_path: Path, baseline_score_path: Path,
    contract: ValidationInputContract, config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> OotStressContext:
    frame = load_pmml_analysis_frame(
        sample_path=sample_path, score_path=baseline_score_path, contract=contract,
        cancellation_check=cancellation_check,
    )
    mask = frame[config.split_col].eq(config.split_values["oot"]).to_numpy()
    row_ids = np.flatnonzero(mask).astype(np.int64)
    if len(row_ids) == 0:
        raise ValueError("OOT sample is required for model stress test")
    return OotStressContext(
        row_ids=row_ids,
        labels=frame.loc[mask, config.target_col].to_numpy(dtype=int),
        baseline_scores=frame.loc[mask, config.score_col].to_numpy(dtype=float),
    )


def _score_oot_category(
    *, category: str, raw_fields: tuple[str, ...], scorer: PmmlScorer,
    contract: ValidationInputContract, oot_input_path: Path,
    expected_row_ids: np.ndarray, output_path: Path, chunk_size: int,
    cancellation_check: Callable[[], None] | None,
) -> ScenarioScoreArtifact:
    manifest = contract.require_pmml_manifest()
    writer = atomic_score_writer(output_path)
    written = 0
    try:
        for batch in pq.ParquetFile(oot_input_path).iter_batches(
            batch_size=chunk_size,
            columns=["row_id", *manifest.raw_required_fields],
        ):
            raise_if_cancelled(cancellation_check)
            selected = batch.to_pandas()
            selected_ids = selected.pop("row_id").to_numpy(dtype=np.int64)
            for field in raw_fields:
                if field not in selected.columns:
                    raise ValueError(f"stress field is missing after transformations: {field}")
                selected[field] = STRESS_MISSING_VALUE
            scores = scorer.score_chunk(
                selected.loc[:, list(manifest.raw_required_fields)]
            )
            numeric = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
            if len(numeric) != len(selected_ids) or not np.isfinite(numeric).all():
                raise ValueError(f"invalid PMML stress scores for category {category}")
            writer.write(selected_ids, numeric)
            written += len(selected_ids)
        if written != len(expected_row_ids):
            raise ValueError(
                f"stress row count mismatch for {category}: "
                f"{written} != {len(expected_row_ids)}"
            )
        final_path, digest = writer.commit()
    except Exception:
        writer.rollback()
        raise
    return ScenarioScoreArtifact(category, final_path, written, digest)


def _load_aligned_scenario_scores(
    artifact: ScenarioScoreArtifact, expected_row_ids: np.ndarray,
    cancellation_check: Callable[[], None] | None,
) -> np.ndarray:
    value_chunks = []
    offset = 0
    for batch in pq.ParquetFile(artifact.path).iter_batches(
        columns=["row_id", "pmml_score"], batch_size=100_000
    ):
        raise_if_cancelled(cancellation_check)
        frame = batch.to_pandas()
        ids = frame["row_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(ids, expected_row_ids[offset:offset + len(ids)]):
            raise ValueError(f"stress row alignment failed for category {artifact.category}")
        values = frame["pmml_score"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite stress score for category {artifact.category}")
        value_chunks.append(values)
        offset += len(ids)
    if offset != len(expected_row_ids):
        raise ValueError(f"stress row count mismatch for category {artifact.category}")
    raise_if_cancelled(cancellation_check)
    return np.concatenate(value_chunks)


def _stress_rows(
    *, labels: np.ndarray, baseline_scores: np.ndarray,
    scenario_scores: np.ndarray, raw_fields: tuple[str, ...],
    category: str, edges: np.ndarray,
) -> StressCategoryResult:
    scenario_frame = pd.DataFrame({"__target__": labels, "__score__": scenario_scores})
    baseline_distribution = bin_distribution(baseline_scores, edges)
    return StressCategoryResult(
        category=category,
        dropped_features=list(raw_fields),
        ks_after=float(compute_ks(scenario_scores, labels)),
        ks_delta=float(compute_ks(scenario_scores, labels) - compute_ks(baseline_scores, labels)),
        psi_vs_baseline=float(
            compute_psi(baseline_distribution, bin_distribution(scenario_scores, edges))
        ),
        bin_table=bin_table(
            scenario_frame, edges, score_col="__score__", target_col="__target__"
        ),
        error=None,
        status="completed",
    )


def run_pmml_stress(
    *, contract: ValidationInputContract, config: ValidationConfig, sample_path: Path,
    baseline_score_path: Path, pmml_path: Path, scenario_dir: Path,
    feature_categories: dict[str, tuple[str, ...]], chunk_size: int,
    scorer: PmmlScorer | None = None,
    baseline_cache_key: str | None = None, cache_dir: Path | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> StressTestResult:
    context = load_oot_stress_context(
        sample_path=sample_path,
        baseline_score_path=baseline_score_path,
        contract=contract,
        config=config,
        cancellation_check=cancellation_check,
    )
    raise_if_cancelled(cancellation_check)
    edges = equal_frequency_bin_edges(context.baseline_scores, config.bin_count)
    baseline_frame = pd.DataFrame({
        "__target__": context.labels,
        "__score__": context.baseline_scores,
    })
    baseline = StressBaseline(
        ks=float(compute_ks(context.baseline_scores, context.labels)),
        sample_count=len(context.row_ids),
        bin_table=bin_table(
            baseline_frame, edges, score_col="__score__", target_col="__target__"
        ),
    )
    active_scorer = scorer or load_pmml_scorer(
        pmml_path, contract.require_output_field()
    )
    per_category = []
    scenario_dir.mkdir(parents=True, exist_ok=True)
    oot_inputs = materialize_oot_pmml_inputs(
        contract=contract,
        sample_path=sample_path,
        oot_row_ids=context.row_ids,
        output_path=scenario_dir / "oot_pmml_inputs.parquet",
        chunk_size=chunk_size,
        cancellation_check=cancellation_check,
    )
    for index, (category, raw_fields) in enumerate(feature_categories.items(), start=1):
        raise_if_cancelled(cancellation_check)
        if not raw_fields:
            raise ValueError(f"stress category {category} has no resolved raw input fields")
        output_path = scenario_dir / f"category_{index:03d}.parquet"
        runner = lambda: _score_oot_category(
            category=category, raw_fields=raw_fields, scorer=active_scorer,
            contract=contract, oot_input_path=oot_inputs.path,
            expected_row_ids=context.row_ids, output_path=output_path,
            chunk_size=chunk_size, cancellation_check=cancellation_check,
        )
        if baseline_cache_key is not None and cache_dir is not None:
            key = stress_cache_key(
                baseline_cache_key=baseline_cache_key,
                category=category,
                raw_fields=raw_fields,
                sentinel=STRESS_MISSING_VALUE,
            )
            cached = load_or_run_stress_scenario(
                cache_dir=cache_dir, cache_key=key,
                expected_row_ids=context.row_ids, runner=runner,
                cancellation_check=cancellation_check,
            )
            artifact = materialize_stress_scenario(
                cached, output_path, cancellation_check=cancellation_check
            )
        else:
            artifact = runner()
        raise_if_cancelled(cancellation_check)
        per_category.append(_stress_rows(
            labels=context.labels,
            baseline_scores=context.baseline_scores,
            scenario_scores=_load_aligned_scenario_scores(
                artifact, context.row_ids, cancellation_check
            ),
            raw_fields=raw_fields,
            category=category,
            edges=edges,
        ))
        raise_if_cancelled(cancellation_check)
    return StressTestResult(
        baseline=baseline,
        per_category=per_category,
        status="completed",
        unclassified_features=[],
        category_source_counts={key: len(value) for key, value in feature_categories.items()},
    )


@contextmanager
def _stress_cache_lock(
    cache_dir: Path, cache_key: str,
    cancellation_check: Callable[[], None] | None,
):
    lock_dir = cache_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with cancellable_file_lock(
        lock_dir / f"{cache_key}.lock", cancellation_check
    ):
        yield


def _valid_cached_stress_scenario(
    cache_dir: Path, cache_key: str, expected_row_ids: np.ndarray,
    *, cancellation_check: Callable[[], None] | None = None,
) -> ScenarioScoreArtifact | None:
    entry = cache_dir / cache_key
    metadata_path = entry / "scenario.json"
    score_path = entry / "scores.parquet"
    if not metadata_path.is_file() or not score_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if set(payload) != {"category", "row_count", "sha256"}:
            return None
        artifact = ScenarioScoreArtifact(
            category=str(payload["category"]), path=score_path,
            row_count=int(payload["row_count"]), sha256=str(payload["sha256"]),
        )
        if not artifact.category or artifact.sha256 != sha256_file_cancellable(
            score_path, cancellation_check
        ):
            return None
        offset = 0
        for batch in pq.ParquetFile(score_path).iter_batches(
            columns=["row_id", "pmml_score"], batch_size=100_000
        ):
            raise_if_cancelled(cancellation_check)
            frame = batch.to_pandas()
            ids = frame["row_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(ids, expected_row_ids[offset:offset + len(ids)]):
                return None
            if not np.isfinite(frame["pmml_score"].to_numpy(dtype=float)).all():
                return None
            offset += len(ids)
        if artifact.row_count != offset or offset != len(expected_row_ids):
            return None
        return artifact
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, pa.ArrowInvalid):
        # Catch only known cache/data failures. A cancellation callback's
        # JobCancelled exception must propagate to the stage unchanged.
        return None


def load_or_run_stress_scenario(
    *, cache_dir: Path, cache_key: str, expected_row_ids: np.ndarray,
    runner: Callable[[], ScenarioScoreArtifact],
    cancellation_check: Callable[[], None] | None = None,
) -> ScenarioScoreArtifact:
    with _stress_cache_lock(cache_dir, cache_key, cancellation_check):
        raise_if_cancelled(cancellation_check)
        cached = _valid_cached_stress_scenario(
            cache_dir, cache_key, expected_row_ids,
            cancellation_check=cancellation_check,
        )
        if cached is not None:
            return cached
        shutil.rmtree(cache_dir / cache_key, ignore_errors=True)
        result = runner()
        entry = cache_dir / cache_key
        uow = ArtifactUnitOfWork()
        try:
            score = uow.stage_file(entry, "scores.parquet")
            shutil.copy2(result.path, score.path)
            metadata = uow.stage_file(entry, "scenario.json")
            metadata.path.write_text(json.dumps({
                "category": result.category,
                "row_count": result.row_count,
                "sha256": result.sha256,
            }, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            uow.promote_all()
            uow.commit()
        except Exception:
            uow.rollback()
            raise
        cached = _valid_cached_stress_scenario(
            cache_dir, cache_key, expected_row_ids,
            cancellation_check=cancellation_check,
        )
        if cached is None:
            raise ValueError("stored PMML stress cache failed verification")
        return cached


def materialize_stress_scenario(
    cached: ScenarioScoreArtifact, output_path: Path,
    *, cancellation_check: Callable[[], None] | None = None,
) -> ScenarioScoreArtifact:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.{uuid4().hex}.staging")
    try:
        try:
            os.link(cached.path, staging)
        except OSError:
            copy_file_cancellable(
                cached.path, staging, cancellation_check=cancellation_check
            )
        raise_if_cancelled(cancellation_check)
        os.replace(staging, output_path)
    finally:
        staging.unlink(missing_ok=True)
    return replace(
        cached, path=output_path,
        sha256=sha256_file_cancellable(output_path, cancellation_check),
    )
```

The full sample is projected and transformed exactly once into `oot_pmml_inputs.parquet`; every category thereafter reads only complete OOT model inputs, avoiding a full million-row source scan and repeated transformations per category. Preserve original `row_id` and verify its exact order against the baseline. Scenario filenames are ordinal to avoid filesystem-unsafe category labels; `StressCategoryResult.category` and `category_source_counts` retain both the original labels and the normalized metadata's first-seen insertion order. Do not alphabetically sort categories, because that would regress pressure-summary rows, Excel sheet order, and Word image order. Cache-key JSON remains independently canonicalized. Each category is scored and reduced into result rows before the next scenario is loaded, so memory is bounded by one OOT score vector rather than `OOT rows × categories`. The two-category test asserts `[row.category for row in result.per_category] == ["征信", "内部"]` for that input order and monkeypatches `iter_sample_chunks` to prove the source sample is traversed once, not once per category.

`load_or_run_stress_scenario` mirrors the baseline cache transaction: a cross-process per-key `FileLock`, JSON metadata plus Parquet promoted together, cancellable SHA-256 verification, batched exact ordered `row_id` verification, and cache deletion on corruption. `copy_file_cancellable` performs a bounded block copy plus metadata preservation and checks the callback between blocks. `materialize_stress_scenario` atomically hard-links when source/target share a filesystem and otherwise uses that cancellable copy; it always returns verified metadata for the task-local path. Task 12 passes the baseline PMML cache key and `workspace/cache/pmml_stress`, so retrying metrics after a later failure reuses every complete matching category while a changed sample, PMML, output field, transformation, category mapping, or sentinel invalidates it.

Add a real `multiprocessing`/spawn test (not only threads) in which two workers target the same key behind a barrier and a file-backed producer counter proves exactly one producer ran. Add cancellation tests during cached-file hashing, cached Parquet batch validation, cross-filesystem materialization, and aligned-score loading; each must propagate `JobCancelled` rather than treating cancellation as recoverable cache corruption.

- [ ] **Step 4: Make new stress results strict without changing historical rendering**

```python
def require_complete_stress_result(result: StressTestResult) -> StressTestResult:
    if result.status != "completed":
        raise ValueError(f"model stress test is incomplete: {result.status}")
    failed = [row.category for row in result.per_category if row.status != "completed"]
    if failed:
        raise ValueError("model stress test failed categories: " + ", ".join(failed))
    return result
```

Keep the existing `run_stress_test` and partial-status reader for legacy results. New pipeline code calls only `run_pmml_stress` plus `require_complete_stress_result`.

- [ ] **Step 5: Add hard-gate tests**

```python
def test_stress_scenario_cache_hits_and_corruption_recomputes(tmp_path):
    calls = []
    row_ids = np.array([2, 3], dtype=np.int64)
    def produce():
        calls.append(len(calls) + 1)
        path = tmp_path / f"produced-{len(calls)}.parquet"
        pd.DataFrame({"row_id": row_ids, "pmml_score": [0.2, 0.8]}).to_parquet(
            path, index=False
        )
        return ScenarioScoreArtifact(
            "内部", path, 2, sha256_file(path)
        )
    first = load_or_run_stress_scenario(
        cache_dir=tmp_path / "cache", cache_key="k", expected_row_ids=row_ids,
        runner=produce,
    )
    second = load_or_run_stress_scenario(
        cache_dir=tmp_path / "cache", cache_key="k", expected_row_ids=row_ids,
        runner=produce,
    )
    assert first.sha256 == second.sha256
    assert calls == [1]
    first.path.write_bytes(first.path.read_bytes() + b"corrupt")
    load_or_run_stress_scenario(
        cache_dir=tmp_path / "cache", cache_key="k", expected_row_ids=row_ids,
        runner=produce,
    )
    assert calls == [1, 2]


@pytest.mark.parametrize("mode", ["json", "row_count", "non_finite"])
def test_stress_scenario_cache_recomputes_all_invalid_metadata_and_scores(
    tmp_path, mode
):
    calls = []
    row_ids = np.array([2, 3], dtype=np.int64)

    def produce():
        calls.append(len(calls) + 1)
        path = tmp_path / f"valid-{len(calls)}.parquet"
        pd.DataFrame({"row_id": row_ids, "pmml_score": [0.2, 0.8]}).to_parquet(
            path, index=False
        )
        return ScenarioScoreArtifact("内部", path, 2, sha256_file(path))

    load_or_run_stress_scenario(
        cache_dir=tmp_path / "cache", cache_key="k", expected_row_ids=row_ids,
        runner=produce,
    )
    entry = tmp_path / "cache" / "k"
    metadata_path = entry / "scenario.json"
    if mode == "json":
        metadata_path.write_text("{", encoding="utf-8")
    elif mode == "row_count":
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["row_count"] = 999
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        score_path = entry / "scores.parquet"
        pd.DataFrame({"row_id": row_ids, "pmml_score": [0.2, np.nan]}).to_parquet(
            score_path, index=False
        )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["sha256"] = sha256_file(score_path)
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    load_or_run_stress_scenario(
        cache_dir=tmp_path / "cache", cache_key="k", expected_row_ids=row_ids,
        runner=produce,
    )
    assert calls == [1, 2]


def _hard_gate_inputs(tmp_path, *, has_oot: bool = True) -> tuple[Path, Path]:
    sample_path = tmp_path / "gate_sample.parquet"
    splits = ["train", "test", "oot", "oot"] if has_oot else ["train", "test", "test", "test"]
    pd.DataFrame({
        "x1": [0.0, 1.0, 2.0, 3.0], "x2": [0.0, 1.0, 0.0, 1.0],
        "y": [0, 1, 0, 1], "split": splits,
        "apply_month": ["202601", "202602", "202603", "202603"],
    }).to_parquet(sample_path, index=False)
    baseline_path = tmp_path / "gate_baseline.parquet"
    pd.DataFrame({
        "row_id": [0, 1, 2, 3], "pmml_score": [0.1, 0.8, 0.2, 0.7]
    }).to_parquet(baseline_path, index=False)
    return sample_path, baseline_path


def _gate_config() -> ValidationConfig:
    return ValidationConfig(
        target_col="__target__", score_col="__pmml_score__",
        split_col="__split__", time_col="__time__", bin_count=2,
    )


def test_new_pmml_stress_rejects_empty_category(tmp_path, ready_contract):
    sample_path, baseline_path = _hard_gate_inputs(tmp_path)
    with pytest.raises(ValueError):
        run_pmml_stress(
            contract=ready_contract, config=_gate_config(), sample_path=sample_path,
            baseline_score_path=baseline_path, pmml_path=MIN_LR_PMML,
            scenario_dir=tmp_path / "stress", feature_categories={"空类别": ()},
            chunk_size=2,
        )


@pytest.mark.parametrize("mode", ["null", "row_count"])
def test_new_pmml_stress_rejects_invalid_category_scores(
    tmp_path, ready_contract, monkeypatch, mode
):
    sample_path, baseline_path = _hard_gate_inputs(tmp_path)
    class InvalidScorer:
        def score_chunk(self, frame):
            if mode == "null":
                return pd.Series([np.nan] * len(frame))
            return pd.Series([0.5] * max(0, len(frame) - 1))
    monkeypatch.setattr(
        "marvis.validation.pmml_stress.load_pmml_scorer",
        lambda *_args, **_kwargs: InvalidScorer(),
    )
    with pytest.raises(ValueError):
        run_pmml_stress(
            contract=ready_contract, config=_gate_config(), sample_path=sample_path,
            baseline_score_path=baseline_path, pmml_path=MIN_LR_PMML,
            scenario_dir=tmp_path / "stress", feature_categories={"内部": ("x1",)},
            chunk_size=2,
        )
    assert not list(tmp_path.rglob("*.staging"))


def test_new_pmml_stress_rejects_empty_oot(tmp_path, ready_contract):
    sample_path, baseline_path = _hard_gate_inputs(tmp_path, has_oot=False)
    with pytest.raises(ValueError, match="OOT sample is required"):
        run_pmml_stress(
            contract=ready_contract, config=_gate_config(), sample_path=sample_path,
            baseline_score_path=baseline_path, pmml_path=MIN_LR_PMML,
            scenario_dir=tmp_path / "stress", feature_categories={"内部": ("x1",)},
            chunk_size=2,
        )
```

- [ ] **Step 6: Run stress and category tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_pmml_stress.py \
  tests/validation/test_stress_test.py \
  tests/validation/test_platform_metrics_stress_categories.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the PMML stress path**

```bash
git add marvis/validation/pmml_stress.py marvis/validation/stress_test.py \
  tests/validation/test_pmml_stress.py tests/validation/test_stress_test.py \
  tests/validation/test_platform_metrics_stress_categories.py
git commit -m "feat: require complete PMML model stress tests"
```

### Task 12: Rewire the Pipeline, Cancellation, Cache, and Stage Routes

**Files:**
- Modify: `marvis/pipeline.py`
- Modify: `marvis/settings.py`
- Modify: `marvis/validation/pmml_score_artifacts.py`
- Modify: `marvis/routers/validation_stages.py`
- Modify: `marvis/routers/stage_controls.py`
- Modify: `marvis/api_task_payloads.py`
- Modify: `marvis/agent/validation_app_service.py`
- Modify: `marvis/agent/validation_runner.py`
- Modify: `marvis/packs/v1_compat/tools.py`
- Modify: `marvis/notebook_steps.py`
- Modify: `tests/test_pipeline_v2.py`
- Modify: `tests/test_notebook_cancellation.py`
- Modify: `tests/test_job_watchdog.py`
- Modify: `tests/test_api_v2.py`
- Create: `tests/test_settings.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: ready validation input contract and Tasks 7–11 functions.
- Produces: `run_pmml_scoring_stage`, PMML-backed metrics/stress stage, unchanged report-stage boundary, and historical legacy dispatch.

- [ ] **Step 1: Write the failing no-Notebook pipeline test**

```python
# tests/test_pipeline_v2.py
def test_new_validation_pipeline_never_opens_a_notebook_session(
    tmp_path, monkeypatch, ready_validation_task, pipeline_settings
):
    monkeypatch.setattr(
        "marvis.pipeline._notebook_step_v3",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("notebook execution forbidden")),
    )
    run_pmml_scoring_stage(
        task_id=ready_validation_task.id, settings=pipeline_settings
    )
    run_metrics_stage(
        task_id=ready_validation_task.id, settings=pipeline_settings
    )
    task = TaskRepository(pipeline_settings.db_path).get_task(ready_validation_task.id)
    assert task.status == TaskStatus.WRITING_ARTIFACTS
    assert (
        pipeline_settings.workspace / "tasks" / task.id / "outputs" /
        "pmml_scoring_result.json"
    ).exists()
```

- [ ] **Step 2: Run and verify the new PMML stage is not implemented yet**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/test_pipeline_v2.py::test_new_validation_pipeline_never_opens_a_notebook_session -q
```

Expected: FAIL because `run_pmml_scoring_stage` is not implemented. This Task 12 gate intentionally stops at `WRITING_ARTIFACTS`; Task 13 adds the minimum PMML-aware presentation/report seam and Tasks 14–17 own complete report success. No Task 12 test may require later-task Word/Agent behavior.

- [ ] **Step 3: Add the canonical PMML scoring stage**

```python
# marvis/pipeline.py import
from contextlib import contextmanager, nullcontext

from marvis.job_cancellation import JobCancelled
from marvis.repositories.validation_contracts import (
    ValidationContractRepository,
    require_confirmed_validation_input_contract,
)
from marvis.validation.pmml_scoring import TASK_PMML_SCORERS
from marvis.validation.pmml_score_artifacts import (
    copy_file_cancellable,
    raise_if_cancelled,
    sha256_file_cancellable,
)
from marvis.validation_materials import resolve_selected_validation_materials


@contextmanager
def _pmml_job_cancellation(job_id: str | None) -> Iterator[Callable[[], None] | None]:
    if not job_id:
        yield None
        return
    token = register_job_cancellation(job_id)
    try:
        yield token.raise_if_cancelled
    finally:
        unregister_job_cancellation(job_id, token)


def run_pmml_scoring_stage(
    *, task_id: str, settings: PipelineSettings, stage_claimed: bool = False,
    cancellation_job_id: str | None = None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    if not stage_claimed:
        repo.update_status(
            task_id, TaskStatus.RUNNING, "PMML打分测试进行中",
            expected=TaskStatus.SCANNED,
        )
    try:
        with _pmml_job_cancellation(cancellation_job_id) as cancellation_check:
            _execute_pmml_scoring_stage(
                task_id=task_id, settings=settings,
                cancellation_check=cancellation_check,
            )
    except JobCancelled as exc:
        _mark_cancelled(repo, task_id, TaskStatus.SCANNED, "PMML打分测试已取消")
    except PipelineCancelled as exc:
        _mark_cancelled(repo, task_id, exc.resume_status, str(exc))
    except Exception:
        _restore_scoring_resume_state(repo, task_id, stage_claimed=True)
        raise


def _execute_pmml_scoring_stage(
    *, task_id: str, settings: PipelineSettings,
    cancellation_check: Callable[[], None] | None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    raise_if_cancelled(cancellation_check)
    try:
        materials = resolve_selected_validation_materials(task)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    contract_record = require_confirmed_validation_input_contract(
        ValidationContractRepository(settings.db_path), task_id
    )
    current_hashes = {
        "notebook": sha256_file_cancellable(materials.notebook, cancellation_check),
        "sample": sha256_file_cancellable(materials.sample, cancellation_check),
        "pmml": sha256_file_cancellable(materials.pmml, cancellation_check),
        "dictionary": sha256_file_cancellable(materials.dictionary, cancellation_check),
    }
    if current_hashes != contract_record.contract.material_hashes:
        raise PipelineError("selected validation materials changed; rescan and reconfirm")
    outputs_dir = settings.workspace / "tasks" / task_id / "outputs"
    sample_path = materials.sample
    pmml_path = materials.pmml
    pmml_digest = current_hashes["pmml"]
    sample_digest = current_hashes["sample"]
    scorer = TASK_PMML_SCORERS.get(
        task_id=task_id,
        pmml_path=pmml_path,
        pmml_sha256=pmml_digest,
        output_field=contract_record.contract.require_output_field(),
    )
    transformation_hash = sha256(
        json.dumps(
            [asdict(row) for row in contract_record.contract.transformations],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_key = pmml_scoring_cache_key(
        pmml_sha256=pmml_digest,
        sample_sha256=sample_digest,
        output_field=contract_record.contract.require_output_field(),
        engine_version=pypmml_engine_version(),
        transformation_sha256=transformation_hash,
    )
    uow = ArtifactUnitOfWork()
    staged_scores = uow.stage_file(outputs_dir, "pmml_scores.parquet")
    staged_result = uow.stage_file(outputs_dir, "pmml_scoring_result.json")
    try:
        working_score = (
            settings.workspace / "cache" / "pmml_scoring_work" / task_id /
            f"{cache_key}.parquet"
        )
        working_score.parent.mkdir(parents=True, exist_ok=True)
        with nullcontext(cancellation_check) as cancellation_check:
            result = load_or_run_pmml_scoring(
                cache_dir=settings.workspace / "cache" / "pmml_scoring",
                cache_key=cache_key,
                runner=lambda: run_pmml_scoring(
                    contract=contract_record.contract,
                    sample_path=sample_path,
                    pmml_path=pmml_path,
                    score_path=working_score,
                    chunk_size=settings.pmml_scoring_chunk_size,
                    scorer=scorer,
                    pmml_sha256=pmml_digest,
                    sample_sha256=sample_digest,
                    cancellation_check=cancellation_check,
                ),
                cancellation_check=cancellation_check,
            )
            if cancellation_check is not None:
                cancellation_check()
            copy_file_cancellable(
                Path(result.score_artifact_path), staged_scores.path,
                cancellation_check=cancellation_check,
            )
            result = validate_pmml_score_artifact(
                result, staged_scores.path, expected_cache_key=cache_key,
                cancellation_check=cancellation_check,
            )
            staged_result.path.write_text(
                json.dumps(asdict(replace(
                    result, score_artifact_path=str(staged_scores.final_path)
                )), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if cancellation_check is not None:
                cancellation_check()
            uow.finalize_with_connection(
                repo.transaction,
                lambda conn: repo.update_status_on_connection(
                    conn, task_id, TaskStatus.EXECUTED,
                    message="PMML打分测试完成", expected=TaskStatus.RUNNING,
                    begin_immediate=True,
                ),
            )
    except JobCancelled as exc:
        uow.rollback()
        raise PipelineCancelled("PMML打分测试已取消", TaskStatus.SCANNED) from exc
    except Exception:
        uow.rollback()
        raise
    finally:
        working_score.unlink(missing_ok=True)
        remove_empty_directory_chain(
            working_score.parent,
            stop_at=settings.workspace / "cache" / "pmml_scoring_work",
        )


# marvis/pipeline.py
@dataclass(frozen=True)
class PipelineSettings:
    # Retain every existing field and add this bounded batch setting.
    pmml_scoring_chunk_size: int = 10_000


# marvis/validation/pmml_score_artifacts.py
@contextmanager
def _scoring_cache_lock(
    cache_dir: Path, cache_key: str,
    cancellation_check: Callable[[], None] | None,
):
    lock_dir = cache_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with cancellable_file_lock(
        lock_dir / f"{cache_key}.lock", cancellation_check
    ):
        yield


def load_or_run_pmml_scoring(
    *, cache_dir: Path, cache_key: str,
    runner: Callable[[], PmmlScoringResult],
    cancellation_check: Callable[[], None] | None = None,
) -> PmmlScoringResult:
    with _scoring_cache_lock(cache_dir, cache_key, cancellation_check):
        raise_if_cancelled(cancellation_check)
        cached = _load_valid_scoring_cache(
            cache_dir, cache_key, cancellation_check=cancellation_check
        )
        if cached is not None:
            return cached
        shutil.rmtree(cache_dir / cache_key, ignore_errors=True)
        result = runner()
        if result.cache_key != cache_key:
            raise ValueError("PMML scoring result/cache key mismatch")
        _store_scoring_cache(cache_dir, cache_key, result)
        cached = _load_valid_scoring_cache(
            cache_dir, cache_key, cancellation_check=cancellation_check
        )
        if cached is None:
            raise ValueError("stored PMML scoring cache failed verification")
        return cached


def _load_valid_scoring_cache(
    cache_dir: Path, cache_key: str,
    *, cancellation_check: Callable[[], None] | None = None,
) -> PmmlScoringResult | None:
    entry = cache_dir / cache_key
    result_path = entry / "pmml_scoring_result.json"
    score_path = entry / "pmml_scores.parquet"
    if not result_path.exists() or not score_path.exists():
        return None
    try:
        result = pmml_scoring_result_from_dict(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
        return validate_pmml_score_artifact(
            result, score_path, expected_cache_key=cache_key,
            cancellation_check=cancellation_check,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, pa.ArrowInvalid):
        return None


def _store_scoring_cache(
    cache_dir: Path, cache_key: str, result: PmmlScoringResult
) -> None:
    entry = cache_dir / cache_key
    entry.mkdir(parents=True, exist_ok=True)
    uow = ArtifactUnitOfWork()
    try:
        staged_score = uow.stage_file(entry, "pmml_scores.parquet")
        shutil.copy2(result.score_artifact_path, staged_score.path)
        staged_json = uow.stage_file(entry, "pmml_scoring_result.json")
        staged_json.path.write_text(
            json.dumps(asdict(replace(result, score_artifact_path="pmml_scores.parquet")),
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        uow.promote_all()
        uow.commit()
    except Exception:
        uow.rollback()
        raise
```

Export `pypmml_engine_version()`, `sha256_file_cancellable()`, and `validate_pmml_score_artifact()` from `pmml_score_artifacts.py`; the engine helper must return the exact value written into `PmmlScoringResult`. Use the explicit `filelock>=3.13,<4` runtime dependency added in Task 8. Add `MARVIS_PMML_SCORING_CHUNK_SIZE` to `marvis/settings.py`, validate it as a positive integer, and thread it through `pipeline_settings_from_request`. The per-key `FileLock` is cross-process and lives outside the replaceable cache entry; `_store_scoring_cache` is called only while that lock is held and promotes both files together, so concurrent retries across multiple workers cannot delete or observe a partial entry. Invalid JSON, schema, hash, key, counts, row order, or non-finite score causes the locked caller to remove the complete entry and recompute. Convert `JobCancelled` to the pipeline's existing cancellation state and leave the task resumable at `SCANNED`; `stage_controls.py` must call `request_job_cancellation(job_id)` for the new `pmml-scoring` job kind.

Close disk lifecycle explicitly. Add positive settings `MARVIS_PMML_CACHE_MAX_BYTES` (default 10 GiB) and `MARVIS_PMML_CACHE_MAX_AGE_DAYS` (default 30). Before starting a new v2 scoring job, `prune_pmml_caches` takes one cross-process `.prune.lock`, ignores `.locks`, skips any per-key lock it cannot acquire immediately, deletes expired entries, then removes least-recently-used unlocked entries until the combined scoring/stress cache is under the size cap. A verified cache hit touches only an `access.json`/mtime under its key lock. Pruning failure is logged and does not corrupt an active validation. The task-specific `pmml_scoring_work/<task>/<key>.parquet` is always deleted in `_execute_pmml_scoring_stage`'s `finally`, followed by removal of empty task/work directories, so a million-row job does not retain an accidental third sidecar copy. Add settings, expiry/size/LRU, active-lock-skip, and no-leftover-work-file tests.

Do not delete legacy `run_notebook_stage`; rename it internally to `run_legacy_notebook_stage` or leave a compatibility wrapper used only when the task's result schema is v1. New task creation always selects v2.

- [ ] **Step 4: Rewire metrics and mandatory stress in one transactional stage**

`run_metrics_stage` for v2 must:

1. load `pmml_scoring_result.json`, recompute all four selected-material hashes, and require the stored PMML/sample/output identities to match the confirmed contract;
2. call `validate_pmml_score_artifact` on the **task-local** `outputs/pmml_scores.parquet` immediately before metrics; validation of the shared cache copy is not sufficient;
3. reacquire the same task-owned scorer from `TASK_PMML_SCORERS` using the verified PMML hash/output field, then run complete PMML OOT stress scenarios with that `scorer`, `baseline_cache_key=scoring_result.cache_key`, and `cache_dir=settings.workspace / "cache" / "pmml_stress"`;
4. compute basic/effectiveness results from the verified sidecar and assemble `ValidationResults(schema_version=v2)`;
5. write JSON and the preliminary Excel into a private metrics work directory, then promote both atomically with the status transition;
6. move to `WRITING_ARTIFACTS` only after stress status is `completed` and a final cancellation checkpoint passes.

Keep the current legacy function body intact behind a version dispatcher and put the new behavior in a narrow helper:

```python
def run_metrics_stage(
    *, task_id: str, settings: PipelineSettings, stage_claimed: bool = False,
    cancellation_job_id: str | None = None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    if task.validation_workflow_version == 2:
        if not stage_claimed and task.status != TaskStatus.EXECUTED:
            raise PipelineError(
                f"PMML metrics require completed scoring; current status is {task.status.value}"
            )
        if not stage_claimed:
            repo.update_status(
                task_id, TaskStatus.COMPUTING_METRICS,
                message="模型压力测试进行中", expected=TaskStatus.EXECUTED,
            )
        try:
            with _pmml_job_cancellation(cancellation_job_id) as cancellation_check:
                return _run_pmml_metrics_stage(
                    task_id=task_id, settings=settings, stage_claimed=True,
                    cancellation_check=cancellation_check,
                )
        except JobCancelled:
            _mark_cancelled(repo, task_id, TaskStatus.EXECUTED, "模型压力测试已取消")
            return
        except PipelineCancelled as exc:
            _mark_cancelled(repo, task_id, exc.resume_status, str(exc))
            return
        except Exception:
            _restore_metrics_resume_state(repo, task_id, stage_claimed=True)
            raise
    return _run_legacy_metrics_stage(
        task_id=task_id, settings=settings, stage_claimed=stage_claimed,
        cancellation_job_id=cancellation_job_id,
    )


def _run_pmml_metrics_stage(
    *, task_id: str, settings: PipelineSettings, stage_claimed: bool,
    cancellation_check: Callable[[], None] | None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    raise_if_cancelled(cancellation_check)
    materials = resolve_selected_validation_materials(task)
    contract = require_confirmed_validation_input_contract(
        ValidationContractRepository(settings.db_path), task_id
    ).contract
    current_hashes = {
        "notebook": sha256_file_cancellable(materials.notebook, cancellation_check),
        "sample": sha256_file_cancellable(materials.sample, cancellation_check),
        "pmml": sha256_file_cancellable(materials.pmml, cancellation_check),
        "dictionary": sha256_file_cancellable(materials.dictionary, cancellation_check),
    }
    if current_hashes != contract.material_hashes:
        _restore_scoring_resume_state(repo, task_id, stage_claimed=stage_claimed)
        raise PipelineError("selected validation materials changed; rescan and reconfirm")
    outputs_dir = settings.workspace / "tasks" / task_id / "outputs"
    score_path = outputs_dir / "pmml_scores.parquet"
    scoring_result = pmml_scoring_result_from_dict(
        json.loads((outputs_dir / "pmml_scoring_result.json").read_text("utf-8"))
    )
    if (
        scoring_result.pmml_sha256 != current_hashes["pmml"]
        or scoring_result.sample_sha256 != current_hashes["sample"]
        or scoring_result.output_field != contract.require_output_field()
    ):
        _restore_scoring_resume_state(repo, task_id, stage_claimed=stage_claimed)
        raise PipelineError("PMML scoring evidence does not match selected materials")
    try:
        validate_pmml_score_artifact(
            scoring_result, score_path, cancellation_check=cancellation_check
        )
    except ValueError as exc:
        _restore_scoring_resume_state(repo, task_id, stage_claimed=stage_claimed)
        raise PipelineError(
            "task-local PMML score artifact is invalid; rerun PMML打分测试"
        ) from exc
    work_dir = outputs_dir / ".pmml-metrics-stage-work"
    uow: ArtifactUnitOfWork | None = None
    try:
        _remove_dir_if_exists(work_dir)
        work_dir.mkdir(parents=True)
        with nullcontext(cancellation_check) as cancellation_check:
            if cancellation_check is not None:
                cancellation_check()
            scorer = TASK_PMML_SCORERS.get(
                task_id=task_id, pmml_path=materials.pmml,
                pmml_sha256=current_hashes["pmml"],
                output_field=scoring_result.output_field,
            )
            config = validation_config_from_input_contract(contract, settings)
            stress = run_pmml_stress(
                contract=contract, config=config, sample_path=materials.sample,
                baseline_score_path=score_path, pmml_path=materials.pmml,
                scenario_dir=work_dir / "stress", scorer=scorer,
                feature_categories=contract.require_feature_metadata().per_category_raw_fields,
                chunk_size=settings.pmml_scoring_chunk_size,
                baseline_cache_key=scoring_result.cache_key,
                cache_dir=settings.workspace / "cache" / "pmml_stress",
                cancellation_check=cancellation_check,
            )
            require_complete_stress_result(stress)
            results = compute_platform_validation_results(
                task=task, contract=contract, sample_path=materials.sample,
                score_path=score_path, scoring_result=scoring_result,
                metadata_resolution=contract.require_feature_metadata(),
                stress_test=stress, settings=settings,
                cancellation_check=cancellation_check,
            )
            write_validation_results_json(results, work_dir / "validation_results.json")
            write_validation_metrics_excel(results, work_dir / "validation.xlsx")
            if cancellation_check is not None:
                cancellation_check()
            uow = _stage_pmml_metrics_outputs_for_commit(
                outputs_dir=outputs_dir, work_dir=work_dir
            )
            if cancellation_check is not None:
                cancellation_check()
            uow.finalize_with_connection(
                repo.transaction,
                lambda conn: repo.update_status_on_connection(
                    conn, task_id, TaskStatus.WRITING_ARTIFACTS,
                    message="模型效果与模型压力测试完成",
                    expected=TaskStatus.COMPUTING_METRICS, begin_immediate=True,
                ),
            )
    except JobCancelled as exc:
        if uow is not None:
            uow.rollback()
        raise PipelineCancelled("模型压力测试已取消", TaskStatus.EXECUTED) from exc
    except Exception as exc:
        if uow is not None:
            uow.rollback()
        _restore_metrics_resume_state(repo, task_id, stage_claimed=stage_claimed)
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(
            f"模型效果或模型压力测试失败: {exc.__class__.__name__}: {exc}"
        ) from exc
    finally:
        _remove_dir_if_exists(work_dir)
```

The v2 dispatcher catches `PipelineCancelled`, marks the task back to `EXECUTED`, and returns. It must not mark the task failed, publish `validation_results.json`/Excel, or clear complete shared stress-cache entries. `_restore_scoring_resume_state` atomically returns `RUNNING`, `EXECUTED`, or a preclaimed `COMPUTING_METRICS` task to `SCANNED` when selected hashes, scoring identity, or the task-local sidecar are invalid, so the only available recovery is to rescan when needed and rerun `PMML打分测试`; it never masks a material/hash mismatch as a metrics retry. `_restore_metrics_resume_state` atomically returns an ordinary computation/stress failure from `COMPUTING_METRICS` to `EXECUTED`, preserving the verified baseline sidecar so the existing retry test can rerun only metrics/stress. Both helpers use explicit expected statuses and audit messages; neither writes directly around repository transition validation. `_stage_pmml_metrics_outputs_for_commit` copies only the completed JSON/workbook (and their generated image directory) into one `ArtifactUnitOfWork`; work-directory scenario files are disposable.

Terminal report status must no longer inspect `results.reproducibility.summary.status`; technical stage failures already prevent report generation.

Clear `TASK_PMML_SCORERS` only after report success, task deletion, or unrecoverable terminal failure; do not clear it between PMML scoring and model stress. A process restart naturally starts with an empty registry and reloads safely. Add a pipeline test that monkeypatches `Model.fromFile`, runs baseline plus metrics/stress, and asserts one load for the task; a changed PMML hash must trigger a second load.

- [ ] **Step 5: Add canonical routes and legacy aliases**

Add `POST /api/tasks/{task_id}/pmml-scoring`. Keep the old Notebook-stage route only for historical v1 tasks and return `409` if called for a new v2 task. Update job kind copy and watchdog recovery messages to `PMML打分测试` and `模型压力测试`.

Version the existing metrics cancellation route instead of replacing its legacy behavior:

```python
@router.post("/tasks/{task_id}/metrics/cancel", status_code=202)
def cancel_task_metrics(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    if task.status != TaskStatus.COMPUTING_METRICS:
        raise conflict(f"cannot cancel metrics in status {task.status.value}")
    job_id = _active_job_id(repo, task_id, "metrics-stress", "metrics", "pipeline")
    if job_id is None:
        raise conflict("task has no active metrics job")
    if task.validation_workflow_version == 2:
        # The registry accepts a pending request in the short interval before
        # the worker registers its token.
        request_job_cancellation(job_id)
        message = "模型压力测试取消请求已提交"
    else:
        if not request_active_notebook_cancellation(
            task_id, expected_job_id=job_id
        ):
            raise conflict("active metrics job has no cancellable execution token")
        _write_metrics_cancel_marker(request.app.state.settings.tasks_dir / task_id)
        message = "metrics cancellation requested"
    return {"task_id": task_id, "status": "accepted", "message": message}
```

The v2 route never writes the legacy Notebook metrics marker. The v1 branch remains byte-for-byte compatible.

Update `_SYSTEM_STEP_TITLES`:

```python
_SYSTEM_STEP_TITLES.update({
    "field-recognition": ("system-field-recognition", "材料与字段识别"),
    "pmml-scoring": ("system-pmml-scoring", "PMML打分测试"),
    "metrics-stress": ("system-metrics-stress", "模型压力测试"),
})
```

New v2 task evidence must not include original Notebook cell execution progress.

- [ ] **Step 6: Add cancellation, cache, retry, and hash-invalidation tests**

```python
# additions to tests/test_pipeline_v2.py imports/constants
import numpy as np

from marvis.files import sha256_file
from marvis.pipeline import (
    PipelineCancelled,
    run_metrics_stage,
    run_pmml_scoring_stage,
    run_staged_pipeline,
)
from marvis.validation.pmml_score_artifacts import (
    load_or_run_pmml_scoring,
    pmml_scoring_cache_key,
    run_pmml_scoring,
)
from marvis.validation.pmml_scoring import TASK_PMML_SCORERS
from marvis.validation.pmml_stress import stress_cache_key
from marvis.validation.results import PmmlScoringResult
from marvis.validation_materials import resolve_selected_validation_materials


MIN_LR_PMML = Path("tests/fixtures/min_lr.pmml")


def _successful_scoring_result(tmp_path: Path, *, cache_key: str) -> PmmlScoringResult:
    score_path = tmp_path / f"{cache_key}.parquet"
    pd.DataFrame({"row_id": [0, 1], "pmml_score": [0.1, 0.9]}).to_parquet(
        score_path, index=False
    )
    return PmmlScoringResult(
        schema_version="marvis.pmml_scoring.v1",
        cache_key=cache_key,
        pmml_sha256="a" * 64,
        sample_sha256="b" * 64,
        engine="pypmml-pmml4s-batch",
        engine_version="1.5.5",
        output_field="probability_1",
        input_row_count=2,
        success_count=2,
        failure_count=0,
        null_count=0,
        non_finite_count=0,
        elapsed_seconds=0.1,
        rows_per_second=20.0,
        chunk_size=2,
        required_input_count=2,
        missing_inputs=[],
        score_artifact_path=str(score_path),
        score_artifact_sha256=sha256_file(score_path),
        status="pass",
        bounded_errors=[],
    )


def test_pmml_scoring_retry_reuses_only_matching_hash_cache(tmp_path):
    calls = []
    key = pmml_scoring_cache_key(
        pmml_sha256="p1", sample_sha256="s1", output_field="probability_1",
        engine_version="1.5.5", transformation_sha256="t1",
    )
    def produce():
        calls.append(key)
        return _successful_scoring_result(tmp_path, cache_key=key)
    first = load_or_run_pmml_scoring(cache_dir=tmp_path / "cache", cache_key=key, runner=produce)
    second = load_or_run_pmml_scoring(cache_dir=tmp_path / "cache", cache_key=key, runner=produce)
    assert first.score_artifact_sha256 == second.score_artifact_sha256
    assert calls == [key]


@pytest.mark.parametrize("mode", ["json", "row_order", "non_finite"])
def test_pmml_scoring_cache_corruption_is_deleted_and_recomputed(tmp_path, mode):
    calls = []
    key = pmml_scoring_cache_key(
        pmml_sha256="p1", sample_sha256="s1", output_field="probability_1",
        engine_version="1.5.5", transformation_sha256="t1",
    )

    def produce():
        calls.append(len(calls) + 1)
        return _successful_scoring_result(tmp_path, cache_key=key)

    load_or_run_pmml_scoring(
        cache_dir=tmp_path / "cache", cache_key=key, runner=produce
    )
    entry = tmp_path / "cache" / key
    result_path = entry / "pmml_scoring_result.json"
    score_path = entry / "pmml_scores.parquet"
    if mode == "json":
        result_path.write_text("{", encoding="utf-8")
    else:
        scores = [0.1, np.nan] if mode == "non_finite" else [0.9, 0.1]
        row_ids = [0, 1] if mode == "non_finite" else [1, 0]
        pd.DataFrame({"row_id": row_ids, "pmml_score": scores}).to_parquet(
            score_path, index=False
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["score_artifact_sha256"] = sha256_file(score_path)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
    load_or_run_pmml_scoring(
        cache_dir=tmp_path / "cache", cache_key=key, runner=produce
    )
    assert calls == [1, 2]


def test_pmml_scoring_cache_lock_allows_only_one_concurrent_producer(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    key = pmml_scoring_cache_key(
        pmml_sha256="p1", sample_sha256="s1", output_field="probability_1",
        engine_version="1.5.5", transformation_sha256="t1",
    )
    calls = 0
    calls_lock = Lock()

    def produce():
        nonlocal calls
        with calls_lock:
            calls += 1
        return _successful_scoring_result(tmp_path, cache_key=key)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _index: load_or_run_pmml_scoring(
                cache_dir=tmp_path / "cache", cache_key=key, runner=produce
            ),
            range(2),
        ))
    assert calls == 1
    assert results[0].score_artifact_sha256 == results[1].score_artifact_sha256


def test_pmml_scoring_cancel_rolls_back_partial_parquet(tmp_path, ready_contract):
    from marvis.job_cancellation import JobCancelled

    sample_path = tmp_path / "four_rows.parquet"
    pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 1.0, 0.0, 0.0]}).to_parquet(
        sample_path, index=False
    )
    check_count = 0
    def cancel_after_first_chunk():
        nonlocal check_count
        check_count += 1
        if check_count >= 2:
            raise JobCancelled("cancelled")
    with pytest.raises(JobCancelled):
        run_pmml_scoring(
            contract=ready_contract, sample_path=sample_path,
            pmml_path=MIN_LR_PMML, score_path=tmp_path / "pmml_scores.parquet",
            chunk_size=2, cancellation_check=cancel_after_first_chunk,
        )
    assert not (tmp_path / "pmml_scores.parquet").exists()
    assert not list(tmp_path.rglob("*.staging*"))


def test_scoring_rejects_material_changed_after_confirmation(
    ready_validation_task, pipeline_settings
):
    task = TaskRepository(pipeline_settings.db_path).get_task(ready_validation_task.id)
    materials = resolve_selected_validation_materials(task)
    materials.sample.write_bytes(materials.sample.read_bytes() + b"changed")
    with pytest.raises(PipelineError, match="rescan and reconfirm"):
        run_pmml_scoring_stage(task_id=task.id, settings=pipeline_settings)
    persisted = TaskRepository(pipeline_settings.db_path).get_task(task.id)
    assert persisted.status == TaskStatus.SCANNED
    outputs = pipeline_settings.workspace / "tasks" / task.id / "outputs"
    assert not (outputs / "pmml_scoring_result.json").exists()


def test_changing_sample_hash_invalidates_score_and_stress_cache(tmp_path):
    first = pmml_scoring_cache_key(
        pmml_sha256="p1", sample_sha256="s1", output_field="probability_1",
        engine_version="1.5.5", transformation_sha256="t1",
    )
    second = pmml_scoring_cache_key(
        pmml_sha256="p1", sample_sha256="s2", output_field="probability_1",
        engine_version="1.5.5", transformation_sha256="t1",
    )
    assert first != second
    assert stress_cache_key(
        baseline_cache_key=first, category="征信", raw_fields=("x1",), sentinel=-9999
    ) != (
        stress_cache_key(
            baseline_cache_key=second, category="征信", raw_fields=("x1",), sentinel=-9999
        )
    )


def test_metrics_failure_resumes_from_executed_without_rescoring_baseline(
    tmp_path, ready_validation_task, pipeline_settings, monkeypatch
):
    run_pmml_scoring_stage(task_id=ready_validation_task.id, settings=pipeline_settings)
    score_path = (
        pipeline_settings.workspace / "tasks" / ready_validation_task.id /
        "outputs" / "pmml_scores.parquet"
    )
    baseline_digest = sha256_file(score_path)
    baseline_mtime = score_path.stat().st_mtime_ns
    original_stress = pipeline_module.run_pmml_stress
    monkeypatch.setattr(
        pipeline_module,
        "run_pmml_stress",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected stress failure")),
    )
    with pytest.raises(PipelineError):
        run_metrics_stage(task_id=ready_validation_task.id, settings=pipeline_settings)
    monkeypatch.setattr(pipeline_module, "run_pmml_stress", original_stress)
    run_metrics_stage(task_id=ready_validation_task.id, settings=pipeline_settings)
    assert sha256_file(score_path) == baseline_digest
    assert score_path.stat().st_mtime_ns == baseline_mtime


def test_baseline_and_all_stress_scenarios_load_pmml_once_per_task(
    ready_validation_task, pipeline_settings, monkeypatch
):
    from pypmml import Model

    TASK_PMML_SCORERS.clear(ready_validation_task.id)
    original = Model.fromFile
    loads = []

    def counted(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr("marvis.validation.pmml_scoring.Model.fromFile", counted)
    run_pmml_scoring_stage(
        task_id=ready_validation_task.id, settings=pipeline_settings
    )
    run_metrics_stage(task_id=ready_validation_task.id, settings=pipeline_settings)
    assert len(loads) == 1


def test_metrics_rejects_a_corrupt_task_local_score_sidecar_before_analysis(
    ready_validation_task, pipeline_settings
):
    run_pmml_scoring_stage(
        task_id=ready_validation_task.id, settings=pipeline_settings
    )
    outputs = (
        pipeline_settings.workspace / "tasks" / ready_validation_task.id / "outputs"
    )
    score_path = outputs / "pmml_scores.parquet"
    frame = pd.read_parquet(score_path)
    frame.loc[0, "pmml_score"] = np.nan
    frame.to_parquet(score_path, index=False)
    with pytest.raises(PipelineError, match="rerun PMML打分测试"):
        run_metrics_stage(
            task_id=ready_validation_task.id, settings=pipeline_settings
        )
    task = TaskRepository(pipeline_settings.db_path).get_task(ready_validation_task.id)
    assert task.status == TaskStatus.SCANNED
    assert not (outputs / "validation_results.json").exists()
    assert not (outputs / "validation.xlsx").exists()


def test_metrics_cancel_during_second_stress_category_rolls_back_outputs(
    ready_validation_task, pipeline_settings, monkeypatch
):
    import marvis.validation.pmml_stress as stress_module
    from marvis.job_cancellation import request_job_cancellation

    run_pmml_scoring_stage(
        task_id=ready_validation_task.id, settings=pipeline_settings
    )
    job_id = "metrics-cancel-during-second-category"
    calls = 0
    original = stress_module.load_or_run_stress_scenario

    def cancel_during_second_category(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            request_job_cancellation(job_id)
        return original(**kwargs)

    monkeypatch.setattr(
        stress_module, "load_or_run_stress_scenario", cancel_during_second_category
    )
    run_metrics_stage(
        task_id=ready_validation_task.id, settings=pipeline_settings,
        cancellation_job_id=job_id,
    )
    task = TaskRepository(pipeline_settings.db_path).get_task(ready_validation_task.id)
    outputs = pipeline_settings.workspace / "tasks" / task.id / "outputs"
    assert calls == 2
    assert task.status == TaskStatus.EXECUTED
    assert not (outputs / "validation_results.json").exists()
    assert not (outputs / "validation.xlsx").exists()
    assert not (outputs / ".pmml-metrics-stage-work").exists()
    assert any((pipeline_settings.workspace / "cache" / "pmml_stress").iterdir())
```

Extend `tests/test_notebook_cancellation.py` with API-level assertions that a v2 metrics cancellation calls `request_job_cancellation(job_id)`, never calls `request_active_notebook_cancellation`, and writes no legacy marker; retain the inverse assertions for a historical v1 task. The second-category test above is the behavioral proof that the same token reaches the sample reader and every stress scenario, not merely the HTTP handler.

Retain the fast thread cache test, and add one spawn-based multi-process test with a barrier and file-backed counter proving one scoring producer for one key across workers. Add stage-level real-registry tests that start scoring/metrics in a worker thread, pause inside cancellable sample hashing, task-local sidecar validation, and a valid cache-hit validation batch, call the actual cancel route, then assert prompt cancellation, correct resume state, no published partial artifacts, and no swallowed `JobCancelled`.

Add the following fixtures in `tests/conftest.py`; they initialize a real DB/workspace, copy the default report template, write four real materials, run scan, and persist an optimistic confirmation. They do not inject a mock contract:

```python
from dataclasses import replace
from pathlib import Path
import shutil

from marvis.api_scan_helpers import perform_scan_task
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.pipeline import PipelineSettings
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.settings import build_settings
from marvis.validation.input_confirmation import validate_confirmation_against_materials
from marvis.validation_materials import resolve_selected_validation_materials
from tests.validation_builders import make_validation_confirmation
from tests.validation_material_builders import write_validation_material_bundle


@pytest.fixture
def pipeline_settings(tmp_path):
    app_settings = build_settings(tmp_path / "workspace")
    app_settings.report_template_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path("workspace/report_templates/default.docx"),
        app_settings.report_template_path,
    )
    init_db(app_settings.db_path)
    return PipelineSettings(
        workspace=app_settings.workspace,
        db_path=app_settings.db_path,
        report_template_path=app_settings.report_template_path,
        pmml_scoring_chunk_size=2,
    )


@pytest.fixture
def ready_validation_task(pipeline_settings):
    app_settings = build_settings(pipeline_settings.workspace)
    bundle = write_validation_material_bundle(
        app_settings.workspace / "ready-bundle",
        notebook_source=(
            "RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\n"
            "RMC_TIME_COL='apply_month'\n"
            "RMC_PMML_OUTPUT_FIELD='probability_1'\nRMC_MODEL_PARAMS={}\n"
        ),
    )
    repo = TaskRepository(app_settings.db_path)
    task = repo.create_task(TaskCreate(
        model_name="ready", model_version="v2", validator="pytest",
        source_dir=str(bundle.root), notebook_path=bundle.notebook_path.name,
        sample_path=bundle.sample_path.name, pmml_path=bundle.pmml_path.name,
        dictionary_path=bundle.dictionary_path.name,
    ))
    scan = perform_scan_task(repo, task, app_settings)
    assert scan["validation_input_contract"]["status"] == "pending_confirmation"
    contracts = ValidationContractRepository(app_settings.db_path)
    candidate = contracts.get(task.id)
    persisted_task = repo.get_task(task.id)
    paths = resolve_selected_validation_materials(persisted_task)
    validated = validate_confirmation_against_materials(
        contract=candidate.contract,
        sample_path=paths.sample,
        dictionary_path=paths.dictionary,
        requested=replace(make_validation_confirmation(), metadata_sheet=None),
    )
    confirmed = contracts.confirm(
        task.id,
        validated.values,
        expected_revision=candidate.revision,
        resolved_sample_schema=validated.sample_schema,
        resolved_feature_metadata=validated.feature_metadata,
    )
    assert confirmed.status == "ready"
    persisted = repo.get_task(task.id)
    assert persisted.status == TaskStatus.SCANNED
    assert persisted.validation_workflow_version == 2
    return persisted
```

Keep the earlier `ready_contract` fixture for unit tests. Each pipeline test must additionally assert final task status/artifact presence, absence of staging files, and scorer call count appropriate to the tested path.

- [ ] **Step 7: Run focused pipeline tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/test_pipeline_v2.py \
  tests/test_notebook_cancellation.py \
  tests/test_job_watchdog.py \
  tests/test_api_v2.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the new pipeline**

```bash
git add marvis/pipeline.py marvis/settings.py marvis/validation/pmml_score_artifacts.py \
  marvis/routers/validation_stages.py marvis/routers/stage_controls.py \
  marvis/api_task_payloads.py marvis/agent/validation_app_service.py \
  marvis/agent/validation_runner.py marvis/packs/v1_compat/tools.py \
  marvis/notebook_steps.py \
  tests/test_pipeline_v2.py tests/test_notebook_cancellation.py \
  tests/test_job_watchdog.py tests/test_api_v2.py tests/test_settings.py tests/conftest.py
git commit -m "feat: replace notebook consistency with PMML scoring"
```

---

## Wave 3 — Shared Reports, Agent/UI Parity, and Release Gates

### Task 13: Build One Presentation Payload for Word, Excel, Web, and Agent

**Files:**
- Create: `marvis/validation/presentation.py`
- Create: `tests/validation/test_presentation.py`
- Modify: `marvis/report_texts.py`
- Modify: `marvis/output/word.py`
- Modify: `marvis/output/image_render.py`
- Modify: `marvis/template_reports.py`
- Modify: `marvis/pipeline.py`
- Modify: `tests/test_report_texts_v2.py`
- Modify: `tests/output/test_word.py`
- Modify: `tests/test_report_renderer_charts.py`
- Modify: `tests/test_template_reports.py`
- Modify: `tests/test_pipeline_v2.py`

**Interfaces:**
- Consumes: new or legacy `ValidationResults`, confirmed report values, field recognition, feature metadata, and rendered metric tables.
- Produces: `FinalValidationPresentation` used unchanged by all output adapters plus a hash-verified `validation_presentation.json` that Agent/Web readers consume.

- [ ] **Step 1: Write the failing shared-presentation test**

```python
# tests/validation/test_presentation.py
from pathlib import Path

import pytest

from marvis.template_reports import find_placeholder_occurrences
from marvis.validation.presentation import (
    build_validation_presentation,
    presentation_from_dict,
    presentation_to_dict,
)
from tests.output.test_excel import _make_pmml_results
from tests.validation_output_contract import load_validation_non_regression_contract


def test_pmml_presentation_has_new_summary_and_all_existing_images(tmp_path):
    presentation = build_validation_presentation(
        _make_pmml_results(),
        report_values={"TEXT:final_validation_conclusion": "经复核，报告结论成立。"},
        image_output_dir=tmp_path / "images",
        template_path=Path("workspace/report_templates/default.docx"),
    )
    assert "TEXT:pmml_scoring_summary" in presentation.text_values
    assert "100%" in presentation.text_values["TEXT:pmml_scoring_summary"]
    assert presentation.text_values["TEXT:reproducibility_summary"] == (
        presentation.text_values["TEXT:pmml_scoring_summary"]
    )
    assert set(load_validation_non_regression_contract()["rendered_image_keys"]) <= set(
        presentation.image_values
    )
    assert presentation.text_values["TEXT:final_validation_conclusion"] == "经复核，报告结论成立。"
    assert presentation.template_placeholder_occurrences == tuple(
        find_placeholder_occurrences(Path("workspace/report_templates/default.docx"))
    )


def test_persisted_presentation_round_trip_is_hash_verified(tmp_path):
    presentation = build_validation_presentation(
        _make_pmml_results(), report_values={},
        image_output_dir=tmp_path / "presentation_images",
        template_path=Path("workspace/report_templates/default.docx"),
    )
    payload = presentation_to_dict(presentation, artifact_root=tmp_path)
    restored = presentation_from_dict(payload, artifact_root=tmp_path)
    assert restored.text_values == presentation.text_values
    assert restored.metric_sections == presentation.metric_sections
    assert restored.pmml_scoring == presentation.pmml_scoring
    first = next(iter(restored.image_values.values()))
    first_path = first[0] if isinstance(first, list) else first
    first_path.write_bytes(first_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        presentation_from_dict(payload, artifact_root=tmp_path)
```

- [ ] **Step 2: Run and verify the presentation module is missing**

Run:

```bash
conda run -n py_313 python -m pytest tests/validation/test_presentation.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Introduce the immutable shared payload**

```python
# addition to marvis/template_reports.py
from collections.abc import Iterator

from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph


def _ordered_template_paragraphs(
    document: DocumentObject,
) -> Iterator[Paragraph]:
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            for cell in _iter_unique_table_cells([table]):
                yield from cell.paragraphs


def find_placeholder_occurrences(template_path: Path) -> list[str]:
    document = Document(template_path)
    non_body = find_non_body_placeholders(template_path)
    if non_body:
        raise ValueError(
            "validation report placeholders are supported only in the document body: "
            + ", ".join(non_body)
        )
    occurrences = []
    for paragraph in _ordered_template_paragraphs(document):
        occurrences.extend(PLACEHOLDER_PATTERN.findall(paragraph.text))
    return occurrences


# addition to tests/test_template_reports.py
def test_find_placeholder_occurrences_preserves_duplicates_and_body_order(tmp_path):
    from marvis.template_reports import find_placeholder_occurrences

    path = tmp_path / "occurrences.docx"
    document = Document()
    document.add_paragraph("{{TEXT:a}} and {{TEXT:a}}")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "{{IMAGE:b}}"
    document.add_paragraph("{{TEXT:c}}")
    document.save(path)
    assert find_placeholder_occurrences(path) == [
        "{{TEXT:a}}", "{{TEXT:a}}", "{{IMAGE:b}}", "{{TEXT:c}}"
    ]
```

Implement `find_non_body_placeholders` by scanning every unique referenced header/footer OOXML part for the same placeholder pattern. This version deliberately hard-rejects any header/footer `TEXT:`, `IMAGE:`, or `TABLE:` placeholder before rendering; static header/footer text and images are still preserved and mirrored by Task 14. Add a header-placeholder rejection test and assert the default template has none. Therefore every generated replacement record in this plan uses `source_part="word/document.xml"`; the field remains explicit so it cannot be confused with static header/footer images.

```python
# marvis/validation/presentation.py
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from marvis.files import sha256_file
from marvis.metric_tables import metric_table_sections_from_payload
from marvis.output.image_render import RenderedImageValue, render_all_images
from marvis.report_texts import report_text_values_from_results
from marvis.template_reports import find_placeholder_occurrences
from marvis.validation.results import (
    ValidationResults,
    validation_results_to_dict,
)


@dataclass(frozen=True)
class FinalValidationPresentation:
    schema_version: str
    source_results_sha256: str
    text_values: dict[str, str]
    image_values: dict[str, RenderedImageValue]
    metric_sections: list[dict[str, Any]]
    field_recognition: dict[str, Any]
    feature_metadata: dict[str, Any]
    pmml_scoring: dict[str, Any]
    template_placeholder_occurrences: tuple[str, ...]


def presentation_to_dict(
    presentation: FinalValidationPresentation, *, artifact_root: Path
) -> dict[str, Any]:
    root = artifact_root.resolve()
    image_values = {}
    for key, value in presentation.image_values.items():
        paths = value if isinstance(value, list) else [value]
        items = []
        for path in paths:
            resolved = Path(path).resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"presentation image escapes artifact root: {path}") from exc
            items.append({
                "path": relative.as_posix(),
                "sha256": sha256_file(resolved),
            })
        image_values[key] = {"many": isinstance(value, list), "items": items}
    return {
        "schema_version": presentation.schema_version,
        "source_results_sha256": presentation.source_results_sha256,
        "text_values": dict(presentation.text_values),
        "image_values": image_values,
        "metric_sections": presentation.metric_sections,
        "field_recognition": presentation.field_recognition,
        "feature_metadata": presentation.feature_metadata,
        "pmml_scoring": presentation.pmml_scoring,
        "template_placeholder_occurrences": list(
            presentation.template_placeholder_occurrences
        ),
    }


def presentation_from_dict(
    payload: dict[str, Any], *, artifact_root: Path
) -> FinalValidationPresentation:
    if payload.get("schema_version") != "marvis.validation_presentation.v1":
        raise ValueError("unsupported validation presentation schema")
    root = artifact_root.resolve()
    image_values: dict[str, RenderedImageValue] = {}
    for key, manifest in payload["image_values"].items():
        paths = []
        for item in manifest["items"]:
            candidate = (root / str(item["path"])).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError("presentation image path traversal") from exc
            if not candidate.is_file() or sha256_file(candidate) != item["sha256"]:
                raise ValueError(f"presentation image is missing or corrupt: {key}")
            paths.append(candidate)
        image_values[str(key)] = paths if manifest["many"] else paths[0]
    return FinalValidationPresentation(
        schema_version=str(payload["schema_version"]),
        source_results_sha256=str(payload["source_results_sha256"]),
        text_values={str(k): str(v) for k, v in payload["text_values"].items()},
        image_values=image_values,
        metric_sections=list(payload["metric_sections"]),
        field_recognition=dict(payload["field_recognition"]),
        feature_metadata=dict(payload["feature_metadata"]),
        pmml_scoring=dict(payload["pmml_scoring"]),
        template_placeholder_occurrences=tuple(
            str(value) for value in payload["template_placeholder_occurrences"]
        ),
    )


def build_validation_presentation(
    results: ValidationResults,
    *, report_values: dict[str, str] | None,
    image_output_dir: Path,
    template_path: Path | None = None,
    field_recognition: dict[str, Any] | None = None,
    feature_metadata: dict[str, Any] | None = None,
) -> FinalValidationPresentation:
    text_values = report_text_values_from_results(results, report_values=report_values)
    image_values = render_all_images(results, image_output_dir)
    metric_sections = metric_table_sections_from_payload(validation_results_to_dict(results))
    return FinalValidationPresentation(
        schema_version="marvis.validation_presentation.v1",
        source_results_sha256=sha256(json.dumps(
            validation_results_to_dict(results), ensure_ascii=True,
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        text_values=text_values,
        image_values=image_values,
        metric_sections=metric_sections,
        field_recognition=dict(field_recognition or {}),
        feature_metadata=dict(feature_metadata or {}),
        pmml_scoring=asdict(results.pmml_scoring) if results.pmml_scoring else {},
        template_placeholder_occurrences=(
            tuple(find_placeholder_occurrences(template_path)) if template_path else ()
        ),
    )
```

The JSON representation stores only relative paths plus hashes, never image bytes or staging paths. Validate the exact allowed top-level keys, the 64-hex `source_results_sha256`, image manifest keys, booleans, list/dict shapes, duplicate keys, and non-empty single-image manifests during deserialization; the compact code above shows the identity contract, while the implementation must reject extra/malformed fields rather than silently coerce them. Add a round-trip test, a `../` traversal test, a missing-image test, a byte-corruption/hash test, and a results-hash mismatch test.

After Task 12 computes v2 results, extend `_run_pmml_metrics_stage` to build a preliminary presentation from those exact results, current field-recognition/metadata evidence, the selected template, and the current confirmed report values. Render its images under `work_dir/presentation_images`, serialize it with `artifact_root=work_dir`, and write `work_dir/validation_presentation.json`; `_stage_pmml_metrics_outputs_for_commit` must promote the JSON and image directory in the same transaction as results/Excel. Task 14 uses the identical layout in a private `report_work_dir`: generate `report_work_dir/presentation_images`, serialize relative to `report_work_dir`, then copy/stage that directory and JSON without changing their relative names. After promotion, `presentation_from_dict(..., artifact_root=outputs_dir)` must succeed. Consequently the persisted presentation is preliminary during conclusion review and final after report completion, but is never independently recomputed by a reader.

- [ ] **Step 4: Replace the report-text semantics without removing the legacy alias**

```python
def _pmml_scoring_text(results: ValidationResults) -> str:
    scoring = results.pmml_scoring
    if scoring is None:
        return _reproducibility_text(results)
    return (
        f"PMML打分测试覆盖 {scoring.input_row_count} 行，成功 {scoring.success_count} 行，"
        f"失败 {scoring.failure_count} 行，空值 {scoring.null_count} 行，"
        f"非有限值 {scoring.non_finite_count} 行，完整评分率 "
        f"{scoring.success_count / scoring.input_row_count:.2%}，测试通过。"
    )


if results.pmml_scoring is not None:
    pmml_summary = _pmml_scoring_text(results)
    values["TEXT:pmml_scoring_summary"] = pmml_summary
    values["TEXT:reproducibility_summary"] = pmml_summary  # old customer-template alias
```

Add both keys to the computed-value boundary, with the new key canonical:

```python
COMPUTED_REPORT_TEXT_KEYS = COMPUTED_REPORT_TEXT_KEYS | {
    "TEXT:pmml_scoring_summary",
    "TEXT:reproducibility_summary",
}


def _computed_scoring_text_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    if payload.get("schema_version") != "marvis.validation_results.v2":
        return {}
    results = validation_results_from_dict(payload)
    value = _pmml_scoring_text(results)
    return {
        "TEXT:pmml_scoring_summary": value,
        "TEXT:reproducibility_summary": value,
    }
```

Merge `_computed_scoring_text_from_payload` into `computed_report_text_values_from_payload`. Neither key is included in `AGENT_CONFIRMED_REPORT_TEXT_KEYS`; submitted `report_values`/manual values cannot override them. Add tests that attempt to inject fake text through both APIs and prove the computed PMML row counts remain. For legacy v1 results, preserve the old consistency wording exactly. Update final-conclusion prompts separately in Task 15.

- [ ] **Step 5: Make Word consume the prepared presentation**

```python
from marvis.template_reports import find_placeholder_occurrences


@dataclass(frozen=True)
class TemplateReplacementOccurrence:
    placeholder_key: str
    kind: Literal["TEXT", "IMAGE", "TABLE"]
    occurrence_index: int
    source_part: str
    source_body_order: int
    physical_item_count: int
    relationship_ids: tuple[str, ...] = ()
    rendered_blip_orders: tuple[int, ...] = ()
    rendered_table_orders: tuple[int, ...] = ()


def write_validation_word_from_presentation(
    presentation: FinalValidationPresentation,
    *, template_path: Path,
    output_path: Path,
) -> TemplateReportResult:
    current_placeholders = tuple(find_placeholder_occurrences(template_path))
    if presentation.template_placeholder_occurrences != current_placeholders:
        raise ValueError("presentation/template placeholder inventory mismatch")
    return render_template_report(TemplateReportPayload(
        template_path=template_path,
        output_path=output_path,
        text_values=_with_prefix(presentation.text_values, "TEXT:"),
        image_values=_with_prefix(presentation.image_values, "IMAGE:"),
    ))
```

Extend `TemplateReportResult` compatibly with `replacement_occurrences: tuple[TemplateReplacementOccurrence, ...] = ()`. Each record contains the exact placeholder key, kind (`TEXT|IMAGE|TABLE`), duplicate occurrence index, source OOXML part, source direct-child/body order, physical rendered item count, and the exact image relationship ids plus blip ordinals or rendered table ordinals created for that occurrence. Populate it inside the existing replacement loop before placeholder text is destroyed; when adding a picture, read the new inline shape's `a:blip/@r:embed` and retain both that relationship id and its one-based `a:blip` ordinal inside the source direct child. This is renderer metadata, not business evidence, and is required for Excel's exact duplicate/text/table/image mapping. A count or relationship id alone is insufficient because python-docx may reuse one relationship for identical image bytes in the same paragraph. Keep `write_validation_word` as a compatibility wrapper with its current public parameters; it builds one presentation and delegates. New pipeline code builds the presentation itself and passes both the presentation and returned replacement manifest to Excel.

Add a renderer test whose single paragraph contains two occurrences of the same `IMAGE:*` placeholder backed by identical bytes. Require one replacement record per placeholder occurrence; assert two records, the same `source_body_order`, distinct `rendered_blip_orders`, and a one-to-one inventory even if python-docx reuses the relationship id. A list-valued image still uses one occurrence record with ordered multiple relationship/blip entries. Add an analogous duplicate `TEXT:*` and two-table test so source locations, not rendered string matching, drive Task 14.

- [ ] **Step 6: Run text, image, Word, and style tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/validation/test_presentation.py \
  tests/test_report_texts_v2.py \
  tests/output/test_word.py \
  tests/test_report_renderer_charts.py \
  tests/test_template_reports.py \
  tests/test_pipeline_v2.py -q
```

Expected: PASS, including the existing run-`rPr` preservation assertions.

- [ ] **Step 7: Commit the shared presentation seam**

```bash
git add marvis/validation/presentation.py marvis/report_texts.py \
  marvis/output/word.py marvis/output/image_render.py marvis/template_reports.py \
  marvis/pipeline.py \
  tests/validation/test_presentation.py tests/test_report_texts_v2.py \
  tests/output/test_word.py tests/test_report_renderer_charts.py \
  tests/test_template_reports.py tests/test_pipeline_v2.py
git commit -m "refactor: share validation presentation across outputs"
```

### Task 14: Mirror the Final Word Report and Every Chart into Excel

**Files:**
- Create: `marvis/output/report_mirror.py`
- Create: `tests/output/test_report_mirror.py`
- Modify: `marvis/output/excel.py`
- Modify: `tests/output/test_excel.py`
- Modify: `marvis/pipeline.py`
- Modify: `tests/test_pipeline_v2.py`

**Interfaces:**
- Consumes: final rendered Word path and the same `FinalValidationPresentation` used to render it.
- Produces: final Excel with all legacy sheets plus `PMML打分测试`, `字段识别`, `特征元数据覆盖`, `报告全文`, `报告图表`, `报告静态资源`, and `报告内容索引`.

- [ ] **Step 1: Write the failing Word-mirror extraction test**

```python
# tests/output/test_report_mirror.py
from docx import Document

from marvis.output.report_mirror import extract_report_mirror


def test_report_mirror_keeps_paragraph_and_table_order(tmp_path):
    path = tmp_path / "report.docx"
    document = Document()
    document.add_paragraph("第一章 模型概述")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "指标"
    table.cell(0, 1).text = "取值"
    table.cell(1, 0).text = "OOT KS"
    table.cell(1, 1).text = "0.35"
    document.add_paragraph("最终结论")
    document.save(path)
    mirror = extract_report_mirror(path)
    assert [(block.kind, block.text) for block in mirror.blocks] == [
        ("paragraph", "第一章 模型概述"),
        ("table", "指标\t取值\nOOT KS\t0.35"),
        ("paragraph", "最终结论"),
    ]
    assert mirror.blocks[1].table_rows == (
        ("指标", "取值"), ("OOT KS", "0.35")
    )


def test_report_mirror_extracts_every_body_header_and_footer_image_occurrence(tmp_path):
    from PIL import Image

    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    path = tmp_path / "report-with-images.docx"
    document = Document()
    document.add_picture(str(image_path))
    document.add_picture(str(image_path))
    document.sections[0].header.paragraphs[0].add_run().add_picture(str(image_path))
    document.sections[0].footer.paragraphs[0].add_run().add_picture(str(image_path))
    document.save(path)
    mirror = extract_report_mirror(path)
    assert len(mirror.images) == 4
    assert [image.scope for image in mirror.images] == [
        "body", "body", "footer", "header"
    ] or [image.scope for image in mirror.images] == [
        "body", "body", "header", "footer"
    ]
    assert len({image.sha256 for image in mirror.images}) == 1
```

- [ ] **Step 2: Run and verify the mirror module is missing**

Run:

```bash
conda run -n py_313 python -m pytest tests/output/test_report_mirror.py -q
```

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement ordered DOCX block extraction**

```python
# marvis/output/report_mirror.py
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import posixpath
from typing import Literal
from zipfile import ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


@dataclass(frozen=True)
class ReportBlock:
    order: int
    kind: Literal["paragraph", "table"]
    text: str
    table_rows: tuple[tuple[str, ...], ...] = ()
    scope: Literal["body", "header", "footer"] = "body"
    part_name: str = "word/document.xml"


@dataclass(frozen=True)
class ReportImage:
    occurrence_order: int
    word_order: int
    blip_order: int
    scope: Literal["body", "header", "footer"]
    part_name: str
    relationship_id: str
    media_name: str
    content_type: str
    sha256: str
    blob: bytes


@dataclass(frozen=True)
class ReportMirror:
    blocks: tuple[ReportBlock, ...]
    images: tuple[ReportImage, ...]


def extract_report_mirror(report_path: Path) -> ReportMirror:
    blocks = []
    images = []
    with ZipFile(report_path) as archive:
        part_names = ["word/document.xml", *sorted(
            name for name in archive.namelist()
            if name.startswith("word/header") or name.startswith("word/footer")
        )]
        content_types = _content_type_map(archive)
        image_order = 0
        for part_name in part_names:
            root = etree.fromstring(archive.read(part_name))
            scope = _scope_for_part(part_name)
            container = (
                root.find(f"{{{W_NS}}}body") if scope == "body" else root
            )
            if container is None:
                raise ValueError(f"DOCX part has no content container: {part_name}")
            relationships = _image_relationships(archive, part_name)
            for word_order, element in enumerate(container, start=1):
                local_name = etree.QName(element).localname
                if local_name == "p":
                    text = "".join(element.xpath(".//w:t/text()", namespaces={"w": W_NS}))
                    if text.strip():
                        blocks.append(ReportBlock(
                            word_order, "paragraph", text.strip(),
                            scope=scope, part_name=part_name,
                        ))
                elif local_name == "tbl":
                    rows = _table_rows(element)
                    blocks.append(ReportBlock(
                        word_order, "table",
                        "\n".join("\t".join(row) for row in rows), rows,
                        scope=scope, part_name=part_name,
                    ))
                for blip_order, blip in enumerate(element.xpath(
                    ".//a:blip[@r:embed]", namespaces={
                    "a": A_NS, "r": R_NS,
                }), start=1):
                    relationship_id = str(blip.get(f"{{{R_NS}}}embed"))
                    media_name = relationships[relationship_id]
                    blob = archive.read(media_name)
                    image_order += 1
                    images.append(ReportImage(
                        occurrence_order=image_order,
                        word_order=word_order,
                        blip_order=blip_order,
                        scope=scope,
                        part_name=part_name,
                        relationship_id=relationship_id,
                        media_name=media_name,
                        content_type=content_types[media_name],
                        sha256=sha256(blob).hexdigest(),
                        blob=blob,
                    ))
    return ReportMirror(blocks=tuple(blocks), images=tuple(images))
```

Implement `_rels_name`, `_image_relationships`, `_content_type_map`, `_scope_for_part`, and `_table_rows` with traversal-safe OOXML resolution: relationship targets are resolved relative to the owning part, must remain inside the ZIP, and must have an image relationship type plus a declared content type. Scan each unique body/header/footer OOXML part once and every `a:blip` occurrence in document order, even when several occurrences reuse the same media blob. Iterate direct `w:p` and `w:tbl` children so order is preserved; recurse through table cells only when extracting cell text. Keep both searchable flattened text and rectangular `table_rows`, so Excel reproduces Word tables as cells rather than one tab-delimited string. `ReportMirror` therefore represents all final Word body/header/footer text/tables and all physical image occurrences, including static template logos.

- [ ] **Step 4: Write the failing final Excel parity test**

```python
# tests/output/test_excel.py
from pathlib import Path

from marvis.output.excel import render_final_validation_excel
from marvis.output.report_mirror import extract_report_mirror
from marvis.output.word import write_validation_word_from_presentation
from marvis.validation.presentation import build_validation_presentation
from tests.validation_output_contract import load_validation_non_regression_contract


def _render_pmml_report(tmp_path):
    results = _make_pmml_results()
    presentation = build_validation_presentation(
        results,
        report_values={"TEXT:final_validation_conclusion": "经复核，报告结论成立。"},
        image_output_dir=tmp_path / "word_images",
        template_path=Path("workspace/report_templates/default.docx"),
    )
    word_path = tmp_path / "validation_report.docx"
    word_result = write_validation_word_from_presentation(
        presentation,
        template_path=Path("workspace/report_templates/default.docx"),
        output_path=word_path,
    )
    return presentation, word_path, word_result


def test_final_excel_contains_word_text_all_images_and_legacy_sheets(tmp_path):
    presentation, word_path, word_result = _render_pmml_report(tmp_path)
    report_mirror = extract_report_mirror(word_path)
    output = tmp_path / "validation_metrics.xlsx"
    render_final_validation_excel(
        _make_pmml_results(), output,
        staged_image_dir=tmp_path / "excel_images",
        presentation=presentation,
        report_mirror=report_mirror,
        template_replacements=word_result.replacement_occurrences,
    )
    workbook = load_workbook(output, data_only=True)
    contract = load_validation_non_regression_contract()
    assert set(contract["excel_sheets"]) <= set(workbook.sheetnames)
    assert {
        "PMML打分测试", "字段识别", "特征元数据覆盖",
        "报告全文", "报告图表", "报告静态资源", "报告内容索引",
    } <= set(workbook.sheetnames)
    full_text = "\n".join(
        str(cell.value or "") for row in workbook["报告全文"] for cell in row
    )
    assert "经复核，报告结论成立" in full_text
    word_text = "\n".join(block.text for block in report_mirror.blocks)
    final_text = presentation.text_values["TEXT:final_validation_conclusion"]
    assert final_text in word_text
    assert final_text in full_text
    assert len(workbook["报告图表"]._images) == len(report_mirror.images)
    index_rows = list(
        workbook["报告内容索引"].iter_rows(min_row=2, values_only=True)
    )
    image_rows = [row for row in index_rows if row[0] == "image"]
    assert len(image_rows) == len(report_mirror.images)
    assert any(str(row[1]).startswith("WORD_STATIC_IMAGE:") for row in image_rows)
    dynamic_rows = [row for row in image_rows if str(row[1]).startswith("IMAGE:")]
    assert dynamic_rows
    assert all(row[4] and row[5] for row in dynamic_rows)
    assert any(row[0] == "text" for row in index_rows)
    assert any(row[0] == "table" for row in index_rows)
```

- [ ] **Step 5: Extend the Excel adapter without changing existing sheets**

Add `TemplateReplacementOccurrence` to the `marvis.template_reports` imports at the top of `marvis/output/excel.py`.

```python
def _populate_validation_metric_sheets(
    workbook: Workbook, results: ValidationResults, *, image_dir: Path
) -> None:
    # Keep the current writer calls and order exactly.
    _write_overview(workbook, results)
    _write_basic_info(workbook, results)
    _write_monthly_distribution(workbook, results)
    _write_hyperparameters(workbook, results)
    _write_feature_importance(workbook, results)
    _write_effectiveness_overall(workbook, results)
    _write_psi_stability(workbook, results)
    _write_roc_ks_images(workbook, results, image_dir)
    for split in ("train", "test", "oot"):
        _write_bins(
            workbook,
            f"分箱_{split}",
            results.effectiveness.bin_tables.get(split, []),
            first_header=f"{split}(独立分箱)",
        )
    _write_monthly_effectiveness(workbook, results)
    _write_stress_summary(workbook, results)
    for category_result in results.stress_test.per_category:
        _write_bins(
            workbook,
            f"压力测试_分箱_{category_result.category}",
            category_result.bin_table,
            first_header=category_result.category,
        )
    if results.pmml_scoring is not None:
        _write_pmml_scoring(workbook, results.pmml_scoring)


def _new_validation_workbook(
    results: ValidationResults, *, image_dir: Path
) -> Workbook:
    workbook = Workbook()
    workbook.remove(workbook.active)
    _populate_validation_metric_sheets(workbook, results, image_dir=image_dir)
    return workbook


def write_validation_metrics_excel(
    results: ValidationResults, output_path: Path
) -> Path:
    """Transactional metrics workbook used before final report confirmation."""
    uow = ArtifactUnitOfWork()
    workbook_artifact = uow.stage_file(output_path.parent, output_path.name)
    image_artifact = uow.stage_directory(output_path.parent, "excel_images")
    try:
        workbook = _new_validation_workbook(results, image_dir=image_artifact.path)
        workbook.save(workbook_artifact.path)
        uow.promote_all()
        uow.commit()
        return workbook_artifact.final_path
    except Exception:
        uow.rollback()
        raise


def write_validation_excel(results: ValidationResults, output_path: Path) -> Path:
    """Backward-compatible public name for legacy and preliminary callers."""
    return write_validation_metrics_excel(results, output_path)


def render_final_validation_excel(
    results: ValidationResults,
    staged_output_path: Path,
    *,
    staged_image_dir: Path,
    presentation: FinalValidationPresentation,
    report_mirror: ReportMirror,
    template_replacements: tuple[TemplateReplacementOccurrence, ...],
) -> Path:
    if results.pmml_scoring is None:
        raise ValueError("final PMML validation Excel requires pmml_scoring")
    staged_image_dir.mkdir(parents=True, exist_ok=True)
    workbook = _new_validation_workbook(results, image_dir=staged_image_dir)
    _write_field_recognition(workbook, presentation.field_recognition)
    _write_feature_metadata_coverage(workbook, presentation.feature_metadata)
    _write_report_full_text(workbook, report_mirror)
    image_anchors = _write_report_images(workbook, report_mirror.images)
    _write_report_content_index(
        workbook, presentation, report_mirror,
        template_replacements=template_replacements,
        image_anchors=image_anchors,
    )
    workbook.save(staged_output_path)
    return staged_output_path
```

Task 12 calls `write_validation_metrics_excel`; Task 14's `run_report_stage` calls `render_final_validation_excel` with paths already staged by its outer `ArtifactUnitOfWork`. This resolves the lifecycle explicitly: the preliminary workbook remains available during Agent conclusion review, and the final workbook atomically replaces it only alongside the final Word. For v2 results, replace only the old consistency rows inside `验证总览` with total/success/failure/null/non-finite/output-field/elapsed/throughput. For v1 results, retain the current overview exactly.

`_write_report_images` embeds `ReportImage.blob` for every physical final-DOCX occurrence in `report_mirror.images`; it never reopens `presentation.image_values` as the image source. It writes one row per occurrence to `报告图表` and returns anchors keyed by `(part_name, word_order, blip_order, relationship_id)`. This guarantees that template logos, repeated images, headers, and footers are mirrored exactly as Word rendered them.

`报告内容索引` must use this deterministic image-to-data mapping for generated report images; every generated image occurrence gets one row containing its exact placeholder key, caption, `报告图表` anchor, source sheet, and source cell range:

| Word image key | Excel source data |
|---|---|
| `IMAGE:sample_overall_distribution` | `样本基本信息` |
| `IMAGE:sample_month_distribution` | `样本逐月分布` |
| `IMAGE:top20_feature_ranking` | `特征重要性` |
| `IMAGE:ranking_table` (alias of top-20 importance) | `特征重要性` |
| `IMAGE:model_parameters` | `模型超参` |
| `IMAGE:overall_model_effect` | `模型效果` |
| `IMAGE:dataset_model_effect` (alias of overall effect) | `模型效果` |
| `IMAGE:loan_month_effect` | `逐月效果` |
| `IMAGE:psi_stability_table` | `PSI稳定性` |
| `IMAGE:ks_discrimination_table` | `分箱_oot` |
| `IMAGE:pressure_ks_table`, `IMAGE:pressure_psi_table` | `压力测试_汇总` |
| actual `IMAGE:pressure_score_shift_<N>` and the corresponding list entry in `IMAGE:pressure_score_shift` | category at index `N`, using `压力测试_分箱_<category>` |
| fallback `IMAGE:pressure_score_shift_<N>` slots beyond the number of categories | `压力测试_汇总` |
| `IMAGE:roc_ks_graph_<split>` | `ROC_KS曲线`, split `<split>` |
| `IMAGE:ranking_table_<split>` | `分箱_<split>` |

For each mapped sheet, write the actual range returned by `worksheet.calculate_dimension()` as `'<sheet>'!<range>` after all rows are populated; for category/split images, resolve the dynamic sheet first and then calculate its range. Resolve a generated image occurrence by joining `TemplateReplacementOccurrence.(source_part, source_body_order, rendered_blip_order, relationship_id)` to the corresponding physical `ReportImage`; then verify its SHA-256 equals the hash of the exact presentation image path for that placeholder/list ordinal. This location-and-relationship join, not hash matching alone, disambiguates duplicate placeholders and aliases with identical bytes. The adapter must reject a generated image with no replacement mapping, a missing anchor, a hash mismatch, or a mapping to a missing/empty sheet; it may not silently write a chart without its underlying data index.

Static template images that have no `IMAGE:*` replacement record are indexed as `WORD_STATIC_IMAGE:<occurrence_order>` and written to a separate `报告静态资源` sheet with part, relationship id, media name, content type, hash, scope, and Word order. They have no fabricated metric source range. The final Excel parity test requires the physical Word image count to equal the embedded Excel image count and requires every dynamic chart—only dynamic charts—to carry a valid underlying source sheet/range.

`报告内容索引` has fixed columns `kind`, `key_or_block`, `caption`, `excel_anchor`, `source_sheet`, `source_range`, `word_order`, and `source_kind` (`word_block|placeholder|generated_image|static_image`). Populate it with five classes of rows:

1. every rendered body/header/footer paragraph/table block from `ReportMirror`, mapped to its exact row range in `报告全文` (`kind=text|table`);
2. every `TEXT:*` placeholder present in the selected template, mapped to the corresponding rendered block and `报告全文` range (`kind=text`, key retained);
3. every physical generated `IMAGE:*` occurrence written to `报告图表`, joined by renderer relationship metadata and mapped to the data sheet/range table above (`kind=image`);
4. every physical static Word image occurrence written to `报告图表` and `报告静态资源`, with blank metric source fields (`kind=image`, stable `WORD_STATIC_IMAGE:*` key);
5. every `TABLE:*` replacement occurrence from `template_replacements`, mapped to the exact rectangular `报告全文` range of its rendered Word table (`kind=table`, key retained).

`_write_report_full_text` writes a paragraph block into one text row and a table block into its own rectangular cell range from `ReportBlock.table_rows`, preserving row/column boundaries and blank cells. It returns `{(part_name, word_order): excel_range}`; the index uses those exact ranges for every body/header/footer paragraph/table block. Read placeholder identities and source locations from `template_replacements`, captured from the exact template during Word rendering; `presentation.template_placeholder_occurrences` remains the pre-render inventory cross-check but is not used to guess post-render locations. Do not recover placeholder keys by string matching the already rendered DOCX or rescan a possibly changed template. Duplicate placeholder occurrences produce separate index rows with exact renderer-provided locations. Tests compare this inventory with Task 1 and fail on any missing index row, and reconstruct each Word table from its indexed Excel range to assert exact cell equality.

- [ ] **Step 6: Regenerate Word and final Excel atomically after conclusion confirmation**

In `run_report_stage`:

1. load results, field evidence, metadata evidence, and final `report_values`;
2. build one presentation with `template_path=settings.report_template_path`, thereby freezing exact placeholder occurrence order;
3. create a private `report_work_dir`, render images at `report_work_dir/presentation_images`, and serialize paths relative to that stable logical root;
4. render Word and retain `TemplateReportResult.replacement_occurrences`;
5. extract `ReportMirror` from staged Word;
6. stage and write final Excel from the same presentation/mirror plus the retained replacement manifest;
7. write `report_work_dir/validation_presentation.json`, then stage/copy the work directory's JSON and `presentation_images` under those exact final names;
8. promote Word, presentation JSON/images, and Excel in one `ArtifactUnitOfWork` transaction, delete `report_work_dir`, and reload the committed JSON from `outputs_dir` as an integrity assertion.

If Excel or presentation serialization/reload fails, the previous Word, Excel, presentation JSON, and presentation images must all remain untouched. Add an integration test that commits the UOW and then successfully calls `presentation_from_dict(payload, artifact_root=outputs_dir)`; asserting only against staging paths is insufficient.

- [ ] **Step 7: Run Excel, Word, and rollback tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/output/test_report_mirror.py \
  tests/output/test_excel.py \
  tests/output/test_word.py \
  tests/test_pipeline_v2.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the full report mirror**

```bash
git add marvis/output/report_mirror.py marvis/output/excel.py marvis/pipeline.py \
  tests/output/test_report_mirror.py tests/output/test_excel.py tests/test_pipeline_v2.py
git commit -m "feat: mirror final Word content into Excel"
```

### Task 15: Migrate Agent Evidence, Prompts, UI Stages, and Charts Without Regression

**Files:**
- Modify: `marvis/routers/evidence.py`
- Modify: `marvis/agent/validation_evidence.py`
- Modify: `marvis/agent/validation_messages.py`
- Modify: `marvis/agent/validation_stages.py`
- Modify: `marvis/agent/service.py`
- Modify: `marvis/llm_prompts.py`
- Modify: `marvis/agent/prompts.py`
- Modify: `marvis/metric_tables.py`
- Modify: `marvis/static/app.js`
- Modify: `marvis/static/js/state.js`
- Modify: `tests/test_agent_api.py`
- Modify: `tests/test_agent_service.py`
- Modify: `tests/test_metric_tables.py`
- Modify: `tests/test_frontend_shell_static.py`
- Modify: `tests/test_frontend_static_v2.py`
- Modify: `tests/test_frontend_v2_api_state.py`

**Interfaces:**
- Consumes: historical v1 artifacts or the hash-verified persisted `validation_presentation.json` for v2.
- Produces: v2 PMML scoring stage messages/evidence plus all current non-consistency Agent and Web visualizations without independently rebuilding metric sections or report text.

- [ ] **Step 1: Write failing versioned evidence tests**

```python
# tests/test_agent_api.py
import json
import sqlite3

import pytest

from marvis.db import TaskRepository
from marvis.domain import TaskCreate
from marvis.validation.presentation import (
    build_validation_presentation,
    presentation_to_dict,
)
from marvis.validation.results import validation_results_to_dict
from tests.output.test_excel import _make_pmml_results, _make_results
from tests.validation_output_contract import load_validation_non_regression_contract


def _persist_evidence_task(client, results, *, historical_v1: bool = False) -> str:
    settings = client.app.state.settings
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(TaskCreate(
        model_name="fixture", model_version="v1", validator="qa",
        source_dir=str(settings.workspace),
    ))
    if historical_v1:
        # Test-only seed of a row that existed before migration 003. There is
        # intentionally no repository/public mutation API for this field.
        with sqlite3.connect(settings.db_path) as conn:
            conn.execute(
                "UPDATE tasks SET validation_workflow_version = 1 WHERE id = ?",
                (task.id,),
            )
    outputs = settings.tasks_dir / task.id / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    payload = validation_results_to_dict(results)
    (outputs / "validation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    if results.pmml_scoring is not None:
        (outputs / "pmml_scoring_result.json").write_text(
            json.dumps(payload["pmml_scoring"], ensure_ascii=False), encoding="utf-8"
        )
        presentation = build_validation_presentation(
            results, report_values={},
            image_output_dir=outputs / "presentation_images",
            template_path=settings.report_template_path,
            field_recognition={"target_col": "y"},
            feature_metadata={"importance_coverage": 1.0},
        )
        persisted = presentation_to_dict(presentation, artifact_root=outputs)
        (outputs / "validation_presentation.json").write_text(
            json.dumps(persisted, ensure_ascii=False), encoding="utf-8"
        )
    else:
        (outputs / "reproducibility_result.json").write_text(
            json.dumps(payload["reproducibility"], ensure_ascii=False), encoding="utf-8"
        )
    return task.id


@pytest.fixture
def completed_pmml_task(client):
    return _persist_evidence_task(client, _make_pmml_results())


@pytest.fixture
def legacy_task(client):
    return _persist_evidence_task(client, _make_results(), historical_v1=True)


def test_new_task_evidence_exposes_pmml_scoring_and_preserves_metric_sections(
    client, completed_pmml_task
):
    evidence = client.get(f"/api/tasks/{completed_pmml_task}/evidence").json()
    outputs = client.app.state.settings.tasks_dir / completed_pmml_task / "outputs"
    persisted = json.loads(
        (outputs / "validation_presentation.json").read_text(encoding="utf-8")
    )
    assert evidence["pmml_scoring"] == persisted["pmml_scoring"]
    assert evidence["report_texts"] == persisted["text_values"]
    assert evidence["validation_results"]["metric_sections"] == persisted["metric_sections"]
    assert evidence["field_recognition"] == persisted["field_recognition"]
    assert evidence["feature_metadata"] == persisted["feature_metadata"]
    assert "reproducibility" not in evidence
    sections = evidence["validation_results"]["metric_sections"]
    contract_sections = load_validation_non_regression_contract()["agent_sections"]
    assert [
        {
            "title": row["title"],
            "table_keys": [table["key"] for table in row.get("tables", [])],
            "chart_keys": [chart["key"] for chart in row.get("charts", [])],
        }
        for row in sections
    ] == contract_sections


def test_v2_web_and_agent_read_persisted_sentinels_without_recomputation(
    client, completed_pmml_task
):
    outputs = client.app.state.settings.tasks_dir / completed_pmml_task / "outputs"
    path = outputs / "validation_presentation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["text_values"]["TEXT:persisted_reader_sentinel"] = "PERSISTED_ONLY_TEXT"
    payload["metric_sections"][0]["title"] = "PERSISTED_ONLY_SECTION"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    evidence = client.get(f"/api/tasks/{completed_pmml_task}/evidence").json()
    assert evidence["report_texts"]["TEXT:persisted_reader_sentinel"] == (
        "PERSISTED_ONLY_TEXT"
    )
    assert evidence["validation_results"]["metric_sections"][0]["title"] == (
        "PERSISTED_ONLY_SECTION"
    )
    agent_context = build_validation_agent_context(
        client.app.state.settings, completed_pmml_task
    )
    assert "PERSISTED_ONLY_TEXT" in json.dumps(agent_context, ensure_ascii=False)
    web_payload = client.get(f"/api/tasks/{completed_pmml_task}").json()
    assert "PERSISTED_ONLY_SECTION" in json.dumps(web_payload, ensure_ascii=False)


@pytest.mark.parametrize("mode", ["missing", "bad_json", "bad_image", "results_mismatch"])
def test_v2_readers_fail_closed_on_presentation_integrity_error(
    client, completed_pmml_task, mode
):
    _corrupt_presentation_fixture(client, completed_pmml_task, mode)
    response = client.get(f"/api/tasks/{completed_pmml_task}/evidence")
    assert response.status_code == 409
    assert "evidence integrity" in response.json()["detail"].lower()


def test_legacy_task_evidence_still_exposes_reproducibility(client, legacy_task):
    evidence = client.get(f"/api/tasks/{legacy_task}/evidence").json()
    assert "reproducibility" in evidence
    assert "pmml_scoring" not in evidence
```

- [ ] **Step 2: Run and verify the API still emits only reproducibility**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/test_agent_api.py::test_new_task_evidence_exposes_pmml_scoring_and_preserves_metric_sections -q
```

Expected: FAIL because evidence readers still hard-code `reproducibility_result.json`.

- [ ] **Step 3: Make evidence readers schema-aware**

```python
def _load_v2_presentation(task_dir: Path) -> FinalValidationPresentation:
    outputs = task_dir / "outputs"
    payload = _read_json_required(outputs / "validation_presentation.json")
    return presentation_from_dict(payload, artifact_root=outputs)


def _versioned_validation_evidence(
    task: TaskRecord, task_dir: Path, results: dict
) -> dict[str, Any]:
    if task.validation_workflow_version == 2:
        if results.get("schema_version") != "marvis.validation_results.v2":
            raise EvidenceIntegrityError("v2 task has non-v2 validation results")
        presentation = _load_v2_presentation(task_dir)
        expected_results_hash = sha256(json.dumps(
            results, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if presentation.source_results_sha256 != expected_results_hash:
            raise EvidenceIntegrityError("presentation/results identity mismatch")
        return {
            "pmml_scoring": presentation.pmml_scoring,
            "report_texts": presentation.text_values,
            "field_recognition": presentation.field_recognition,
            "feature_metadata": presentation.feature_metadata,
            "validation_results": {
                **results,
                "metric_sections": presentation.metric_sections,
            },
        }
    if task.validation_workflow_version != 1:
        raise EvidenceIntegrityError("unsupported validation workflow version")
    return {
        "reproducibility": results.get("reproducibility")
        or _read_json(task_dir / "outputs" / "reproducibility_result.json")
    }
```

All dispatch first on immutable `task.validation_workflow_version`, never on artifact presence or result schema inference; the result schema and canonical results hash are then mandatory integrity checks. All v2 Agent evidence builders, stage summaries, report prompts, and Web payload adapters receive this one loaded presentation object or the fields above. They must not call `metric_table_sections_from_payload`, `report_text_values_from_results`, or `build_validation_presentation` while reading a task. The only builders are the metrics-stage preliminary write and report-stage final replacement. A missing/corrupt presentation, source-results mismatch, or image hash failure is an evidence-integrity error, not a cue to silently recompute. Historical v1 readers retain their current result/reproducibility path. Do not return execution environment or Notebook cell progress as v2 stage evidence.

- [ ] **Step 4: Replace only consistency-stage Agent copy and prompts**

```python
def agent_stage_label(stage: str) -> str:
    return {
        "scan": "材料与字段识别",
        "pmml_scoring": "PMML打分测试",
        "metrics": "模型效果与稳定性验证",
        "stress": "模型压力测试",
        "word_conclusion_draft": "验证报告结论确认",
    }.get(stage, stage)
```

The PMML scoring stage summary prompt may discuss only row coverage, input coverage, output field, null/non-finite counts, elapsed time, throughput, engine, and remediation. Remove instructions to discuss Notebook reproducibility, code-model scores, match percentage, or maximum score difference. Preserve the existing metrics/stress risk tiers and three confirmed Word conclusion fields.

- [ ] **Step 5: Preserve all metric-table sections and add a scoring audit table**

Do not rename or remove current section/table keys. Add a separate scoring audit section only in the PMML stage:

```python
def pmml_scoring_table(scoring: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "PMML打分测试",
        "tables": [{
            "key": "PMML_SCORING_AUDIT",
            "title": "PMML打分测试",
            "headers": ["状态", "总行数", "成功", "失败", "空值", "非有限值", "耗时", "吞吐"],
            "rows": [[
                scoring["status"], scoring["input_row_count"], scoring["success_count"],
                scoring["failure_count"], scoring["null_count"], scoring["non_finite_count"],
                scoring["elapsed_seconds"], scoring["rows_per_second"],
            ]],
        }],
    }
```

- [ ] **Step 6: Replace the frontend evidence component without deleting legacy rendering**

New v2 tasks render `pmmlScoringSummary`; legacy tasks continue using `reproducibilitySummary`. The v2 panel shows status, total/success/failure/null/non-finite/output field/engine/elapsed/throughput and no precision comparison chart. Keep every existing metric chart and report-confirmation component.

Add static assertions:

```python
app_source = Path("marvis/static/app.js").read_text(encoding="utf-8")
state_source = Path("marvis/static/js/state.js").read_text(encoding="utf-8")
assert 'headingHtml: "<h3>PMML打分测试</h3>"' in app_source
assert 'headingHtml: "<h3>分数一致性</h3>"' in app_source  # legacy branch remains
assert '模型压力测试' in state_source
```

- [ ] **Step 7: Run Agent, metric-table, and frontend tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/test_agent_api.py tests/test_agent_service.py tests/test_metric_tables.py \
  tests/test_frontend_shell_static.py tests/test_frontend_static_v2.py \
  tests/test_frontend_v2_api_state.py -q
node --check marvis/static/app.js
node --check marvis/static/js/state.js
```

Expected: PASS.

- [ ] **Step 8: Commit Agent/UI migration**

```bash
git add marvis/routers/evidence.py marvis/agent/validation_evidence.py \
  marvis/agent/validation_messages.py marvis/agent/validation_stages.py \
  marvis/agent/service.py marvis/llm_prompts.py marvis/agent/prompts.py \
  marvis/metric_tables.py marvis/static/app.js marvis/static/js/state.js \
  tests/test_agent_api.py tests/test_agent_service.py tests/test_metric_tables.py \
  tests/test_frontend_shell_static.py tests/test_frontend_static_v2.py \
  tests/test_frontend_v2_api_state.py
git commit -m "feat: present PMML scoring across Agent and UI"
```

### Task 16: Update the Default Word Template and Public Contracts

**Files:**
- Create: `scripts/migrate_pmml_validation_template.py`
- Modify: `workspace/report_templates/default.docx`
- Modify: `docs/roadmap.md`
- Modify: `docs/notebook_contract.md`
- Modify: `docs/对notebook的要求.md`
- Modify: `docs/runbook.md`
- Modify: `AGENTS.md`
- Modify: `DESIGN.md`
- Modify: `tests/test_template_reports.py`
- Modify: `tests/test_validation_debt.py`

**Interfaces:**
- Consumes: finalized new workflow terminology and `TEXT:pmml_scoring_summary` alias behavior.
- Produces: a default template and documentation that no longer promise Notebook execution or code/PMML consistency for new tasks.

- [ ] **Step 1: Write the failing default-template wording test**

```python
# tests/test_template_reports.py
from scripts.migrate_pmml_validation_template import (
    REPLACEMENTS,
    document_text_parts,
    image_relationship_signature,
    main,
    migrate_template_atomic,
    validate_migrated_template,
)


def test_default_template_uses_pmml_scoring_not_notebook_reproducibility():
    document = Document(Path("workspace/report_templates/default.docx"))
    text = "\n".join(document_text_parts(document))
    assert "代码模型与PMML" not in text
    assert "Notebook可复现" not in text
    assert "开发过程和结果的有效性和可复现性" not in text
    assert "整个训练过程是否可以完全复现" not in text
    assert "验证和开发人员是否一致" not in text
    assert "PMML打分测试" in text
    assert "模型压力测试" in text
```

- [ ] **Step 2: Run and verify the existing fixed template wording fails**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/test_template_reports.py::test_default_template_uses_pmml_scoring_not_notebook_reproducibility -q
```

Expected: FAIL on the current fixed reproducibility statements.

- [ ] **Step 3: Add and run a deterministic template migration script**

```python
# scripts/migrate_pmml_validation_template.py
import argparse
from collections.abc import Iterator
from hashlib import sha256
import os
from pathlib import Path
import posixpath
from uuid import uuid4
from zipfile import ZipFile

from docx import Document
from docx.document import Document as DocumentObject
from docx.text.paragraph import Paragraph
from lxml import etree

from marvis.template_reports import find_placeholders


REPLACEMENTS = {
    "本次模型验证的首要目标是确保开发过程和结果的有效性和可复现性":
        "本次模型验证的首要目标是验证拟投产PMML在提交样本上的可执行性、模型效果、稳定性和压力表现",
    "模型开发验证主要包括模型开发过程的验证和模型结果的验证，确保模型参数的选择合理、变量分析和模型训练的过程真实准确、模型结果有效和可复现性。":
        "模型验证主要包括材料与字段识别、PMML打分测试、模型效果与稳定性验证和模型压力测试。",
    "开发过程验证": "材料与字段识别",
    "模型开发过程的验证，包括验证入模变量的筛选、分析过程是否清晰合理、单一变量的权重是否过高、变量的分组是否与业务逻辑一致且满足应用要求、最终模型变量的维度是否覆盖全面，以及整个训练过程是否可以完全复现。":
        "材料与字段识别包括静态识别Notebook中的目标、分组、时间和模型参数，核验PMML输入输出，以及检查全部入模特征的分类和重要性元数据；Notebook不执行。",
    "开发结果的验证主要是通过训练集train、验证集test和测试集oot的比对，验证模型的区分能力、稳定性、准确性等指标，以验证和开发人员是否一致。":
        "模型结果验证使用PMML对提交样本全量打分，并在train、test和oot上验证区分能力、稳定性、排序性和压力表现。",
}


def iter_document_paragraphs(document: DocumentObject) -> Iterator[Paragraph]:
    for paragraph in document.paragraphs:
        yield paragraph
    seen_cells: set[int] = set()
    pending_tables = list(document.tables)
    while pending_tables:
        table = pending_tables.pop(0)
        for row in table.rows:
            for cell in row.cells:
                cell_id = id(cell._tc)
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)
                yield from cell.paragraphs
                pending_tables.extend(cell.tables)


def document_text_parts(document: DocumentObject) -> list[str]:
    return [paragraph.text for paragraph in iter_document_paragraphs(document)]


def replace_text_preserving_run_properties(
    paragraph: Paragraph, old: str, new: str
) -> None:
    while old in paragraph.text:
        full_text = "".join(run.text for run in paragraph.runs)
        start = full_text.index(old)
        end = start + len(old)
        offsets = []
        cursor = 0
        for run in paragraph.runs:
            offsets.append((cursor, cursor + len(run.text)))
            cursor += len(run.text)
        start_index = next(i for i, (_a, b) in enumerate(offsets) if start < b)
        end_index = next(i for i, (a, b) in enumerate(offsets) if a < end <= b)
        start_run = paragraph.runs[start_index]
        start_offset = start - offsets[start_index][0]
        end_offset = end - offsets[end_index][0]
        if start_index == end_index:
            start_run.text = start_run.text[:start_offset] + new + start_run.text[end_offset:]
            continue
        suffix = paragraph.runs[end_index].text[end_offset:]
        start_run.text = start_run.text[:start_offset] + new
        for index in range(start_index + 1, end_index + 1):
            paragraph.runs[index].text = ""
        paragraph.runs[end_index].text = suffix


def migrate_template(source: Path, destination: Path) -> None:
    document = Document(source)
    for paragraph in iter_document_paragraphs(document):
        for old, new in REPLACEMENTS.items():
            if old in paragraph.text:
                replace_text_preserving_run_properties(paragraph, old, new)
    document.save(destination)


def image_relationship_signature(path: Path) -> tuple[tuple[str, str, str, str], ...]:
    rows = []
    with ZipFile(path) as archive:
        for rels_name in sorted(
            name for name in archive.namelist() if name.endswith(".rels")
        ):
            root = etree.fromstring(archive.read(rels_name))
            owner_dir = posixpath.dirname(posixpath.dirname(rels_name))
            for relation in root:
                if not str(relation.get("Type", "")).endswith("/image"):
                    continue
                target = str(relation.get("Target"))
                member = posixpath.normpath(posixpath.join(owner_dir, target))
                rows.append((
                    rels_name, str(relation.get("Id")), target,
                    sha256(archive.read(member)).hexdigest(),
                ))
    return tuple(rows)


def validate_migrated_template(source: Path, candidate: Path) -> None:
    if find_placeholders(source) != find_placeholders(candidate):
        raise ValueError("template placeholder inventory changed during migration")
    if image_relationship_signature(source) != image_relationship_signature(candidate):
        raise ValueError("template image relationships changed during migration")
    text = "\n".join(document_text_parts(Document(candidate)))
    if any(old in text for old in REPLACEMENTS):
        raise ValueError("legacy validation wording remains after migration")
    if "PMML打分" not in text:
        raise ValueError("migrated template does not contain PMML scoring wording")


def migrate_template_atomic(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid4().hex}.staging")
    try:
        migrate_template(source, staging)
        validate_migrated_template(source, staging)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    migrate_template_atomic(args.source, args.destination)


if __name__ == "__main__":
    main()
```

The relationship signature derives the owning OOXML part correctly for document/header/footer relationships and hashes the related media bytes; add a unit test with an image in a header as well as the body. Assigning `Run.text` above leaves each run's existing `rPr` in place; tests must compare the XML `rPr` of the first affected run before and after migration, including a replacement split across three runs. The script may not clear and rebuild paragraphs with plain runs.

Add these deterministic tests:

```python
def test_template_migration_preserves_placeholders_images_and_first_run_rpr(tmp_path):
    from PIL import Image

    source = tmp_path / "source.docx"
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    document = Document()
    paragraph = document.add_paragraph()
    old = next(iter(REPLACEMENTS))
    first = paragraph.add_run(old[:8])
    first.bold = True
    first.font.name = "宋体"
    paragraph.add_run(old[8:18])
    paragraph.add_run(old[18:] + " {{TEXT:pmml_scoring_summary}}")
    document.add_picture(str(image_path))
    document.sections[0].header.paragraphs[0].add_run().add_picture(str(image_path))
    document.save(source)
    before_rpr = Document(source).paragraphs[0].runs[0]._r.rPr.xml
    before_placeholders = find_placeholders(source)
    before_images = image_relationship_signature(source)
    destination = tmp_path / "migrated.docx"
    migrate_template_atomic(source, destination)
    migrated = Document(destination)
    assert migrated.paragraphs[0].runs[0]._r.rPr.xml == before_rpr
    assert find_placeholders(destination) == before_placeholders
    assert image_relationship_signature(destination) == before_images
    assert "PMML打分" in "\n".join(document_text_parts(migrated))


def test_template_cli_migrates_in_place_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "default.docx"
    shutil.copy2(Path("workspace/report_templates/default.docx"), destination)
    monkeypatch.setattr(sys, "argv", [
        "migrate_pmml_validation_template.py", str(destination), str(destination)
    ])
    main()
    assert not list(tmp_path.glob(".*.staging"))
    validate_migrated_template(destination, destination)
```

Add the necessary `shutil`/`sys` imports to the test module. The second validation call intentionally proves the post-migration template is idempotently valid; `validate_migrated_template` sees identical source/candidate inventories.

Run the CLI on the real default template only after the tests pass:

```bash
conda run -n py_313 python scripts/migrate_pmml_validation_template.py \
  workspace/report_templates/default.docx workspace/report_templates/default.docx
```

- [ ] **Step 4: Rewrite the Notebook contract as a read-only identification contract**

The docs must state exactly:

- Notebook is one of four mandatory files but is never executed;
- `RMC_TARGET_COL`, `RMC_SPLIT_COL`, `RMC_TIME_COL`, `RMC_PMML_OUTPUT_FIELD`, and `RMC_MODEL_PARAMS` are preferred static hints;
- `RMC_SAMPLE_DF`, `RMC_SCORE_FN`, `RMC_ALGORITHM`, and environment package setup are not required by new tasks;
- complex derived fields must be materialized by the user;
- feature metadata requires `feature/category/importance` and exact PMML coverage;
- PMML scores every row and model stress tests every complete OOT category;
- legacy tasks remain readable under their old semantics.

Update `docs/roadmap.md`, `AGENTS.md`, `DESIGN.md`, and `docs/runbook.md` in the same commit so no current source still claims the old workflow is the new-task compatibility boundary.

- [ ] **Step 5: Add documentation debt assertions**

```python
from pathlib import Path


CURRENT_VALIDATION_DOCS = (
    Path("docs/roadmap.md"),
    Path("docs/notebook_contract.md"),
    Path("docs/对notebook的要求.md"),
    Path("docs/runbook.md"),
    Path("AGENTS.md"),
    Path("DESIGN.md"),
)


def test_current_validation_docs_do_not_require_notebook_execution():
    for path in CURRENT_VALIDATION_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "Notebook 必须能从头到尾运行完成" not in text
        assert "RMC_SCORE_FN 是平台调用内存模型打分的唯一入口" not in text
        assert "PMML打分测试" in text
```

- [ ] **Step 6: Run template and documentation tests**

Run:

```bash
conda run -n py_313 python -m pytest \
  tests/test_template_reports.py tests/test_validation_debt.py -q
git diff --check
```

Expected: PASS.

- [ ] **Step 7: Commit template and docs**

```bash
git add scripts/migrate_pmml_validation_template.py workspace/report_templates/default.docx \
  docs/roadmap.md docs/notebook_contract.md docs/对notebook的要求.md docs/runbook.md \
  AGENTS.md DESIGN.md tests/test_template_reports.py tests/test_validation_debt.py
git commit -m "docs: migrate validation contracts to PMML scoring"
```

### Task 17: Add End-to-End, Performance, Windows, and Non-Regression Gates

**Files:**
- Create: `tests/e2e/test_pmml_validation_flow.py`
- Create: `tests/e2e/validation_bundle.py`
- Create: `tests/slow/test_pmml_batch_benchmark.py`
- Create: `tests/fixtures/xgb_binary_small.pmml`
- Create: `tests/fixtures/lightgbm_binary_small.pmml`
- Create: `tests/fixtures/pmml_fixture_provenance.json`
- Create: `scripts/build_tree_pmml_fixtures.py`
- Create: `scripts/benchmark_pmml_scoring.py`
- Create: `scripts/benchmark_pmml_validation_flow.py`
- Create: `packaging/windows/smoke_pmml_scoring.py`
- Modify: `packaging/windows/build-installer.ps1`
- Modify: `.github/workflows/windows-installer.yml`
- Modify: `tests/test_python_compat.py`
- Modify: `tests/test_windows_packaging.py`
- Create: `tests/test_validation_non_regression_contract.py`

**Interfaces:**
- Consumes: the complete implementation from Tasks 1–16.
- Produces: release-readiness evidence; does not tag, push, or publish.

- [ ] **Step 1: Write the complete manual-mode E2E journey**

```python
# tests/e2e/test_pmml_validation_flow.py
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from docx import Document
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from marvis.app import create_app
from marvis.output.report_mirror import extract_report_mirror
from marvis.validation.presentation import presentation_from_dict
from marvis.validation.pmml_scoring import load_pmml_scorer
from tests.e2e.validation_bundle import build_validation_bundle
from tests.validation_output_contract import load_validation_non_regression_contract


@pytest.fixture
def app_client(tmp_path: Path) -> TestClient:
    with TestClient(create_app(tmp_path / "workspace")) as client:
        yield client


@pytest.fixture
def validation_bundle(tmp_path: Path):
    return build_validation_bundle(
        tmp_path / "bundle",
        pmml_fixture=Path("tests/fixtures/xgb_binary_small.pmml"),
        sample_format="parquet",
        metadata_format="xlsx",
    )


def _ok(response, expected: int = 200) -> dict:
    assert response.status_code == expected, response.text
    return response.json()


def _wait_for_status(client, task_id: str, expected: set[str], timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = _ok(client.get(f"/api/tasks/{task_id}"))
        if task["status"] in expected:
            return task
        if task["status"] == "failed":
            raise AssertionError(task.get("status_message") or task)
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {sorted(expected)}")


def _create_validation_task(client, bundle, *, run_mode: str) -> str:
    payload = _ok(client.post("/api/tasks", json={
        "task_type": "validation",
        "model_name": bundle.model_name,
        "model_version": "e2e",
        "validator": "pytest",
        "source_dir": str(bundle.source_dir),
        "run_mode": run_mode,
    }))
    return str(payload["id"])


def _select_four_materials(client, task_id: str, bundle) -> None:
    _ok(client.put(f"/api/tasks/{task_id}/materials", json={
        "notebook_path": str(bundle.notebook_path),
        "sample_path": str(bundle.sample_path),
        "pmml_path": str(bundle.pmml_path),
        "dictionary_path": str(bundle.dictionary_path),
    }))


def _confirm_contract(client, task_id: str, scanned_contract: dict, bundle) -> None:
    _ok(client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json={"revision": scanned_contract["revision"], **bundle.confirmation},
    ))


def _run_pmml_scoring(client, task_id: str) -> None:
    _ok(client.post(f"/api/tasks/{task_id}/pmml-scoring", json={}), expected=202)
    _wait_for_status(client, task_id, {"executed"})


def _run_metrics_and_stress(client, task_id: str) -> None:
    _ok(client.post(f"/api/tasks/{task_id}/metrics"), expected=202)
    _wait_for_status(client, task_id, {"writing_artifacts", "review_required"})


def _finish_manual_report(client, task_id: str) -> None:
    fields = _ok(client.get(f"/api/tasks/{task_id}/report-fields"))
    _ok(client.put(
        f"/api/tasks/{task_id}/report-fields",
        headers={"If-Match": str(fields["revision"])},
        json={"text_values": fields["text_values"]},
    ))
    _ok(client.post(f"/api/tasks/{task_id}/report"), expected=202)


def _finish_agent_report(client, task_id: str) -> None:
    draft = _ok(client.post(f"/api/tasks/{task_id}/agent/report-draft", json={}))
    fields = _ok(client.get(f"/api/tasks/{task_id}/report-fields"))
    _ok(client.post(
        f"/api/tasks/{task_id}/agent/report-draft/confirm",
        json={"revision": fields["revision"], "text_values": draft["text_values"]},
    ), expected=202)


def _assert_word_excel_agent_non_regression(client, task_id: str) -> None:
    settings = client.app.state.settings
    outputs = settings.tasks_dir / task_id / "outputs"
    word_path = outputs / "validation_report.docx"
    excel_path = outputs / "validation.xlsx"
    assert word_path.is_file() and excel_path.is_file()

    contract = load_validation_non_regression_contract()
    document = Document(word_path)
    mirror = extract_report_mirror(word_path)
    word_text = "\n".join(block.text for block in mirror.blocks)
    assert "{{TEXT:" not in word_text and "{{IMAGE:" not in word_text
    assert len(document.inline_shapes) > 0

    workbook = load_workbook(excel_path, data_only=True)
    assert set(contract["excel_sheets"]) <= set(workbook.sheetnames)
    assert {"报告全文", "报告图表", "报告静态资源", "报告内容索引"} <= set(
        workbook.sheetnames
    )
    excel_report_text = "\n".join(
        str(cell.value or "") for row in workbook["报告全文"] for cell in row
    )
    for block in mirror.blocks:
        assert block.text in excel_report_text
    image_count = len(workbook["报告图表"]._images)
    index_rows = [
        row for row in workbook["报告内容索引"].iter_rows(min_row=2, values_only=True)
        if any(value not in (None, "") for value in row)
    ]
    image_rows = [row for row in index_rows if row[0] == "image"]
    dynamic_image_rows = [
        row for row in image_rows if str(row[1]).startswith("IMAGE:")
    ]
    word_table_rows = [
        row for row in index_rows if row[0] == "table" and row[7] == "word_block"
    ]
    placeholder_table_rows = [
        row for row in index_rows if row[0] == "table" and row[7] == "placeholder"
    ]
    assert image_count == len(mirror.images) == len(image_rows) > 0
    assert all(row[4] and row[5] for row in dynamic_image_rows)
    assert len(word_table_rows) == sum(block.kind == "table" for block in mirror.blocks)
    assert all(row[4] and row[5] for row in word_table_rows)
    assert all(row[4] and row[5] for row in placeholder_table_rows)
    assert any(row[0] == "text" for row in index_rows)

    evidence = _ok(client.get(f"/api/tasks/{task_id}/evidence"))
    presentation_payload = json.loads(
        (outputs / "validation_presentation.json").read_text(encoding="utf-8")
    )
    presentation = presentation_from_dict(
        presentation_payload, artifact_root=outputs
    )
    assert evidence["pmml_scoring"] == presentation_payload["pmml_scoring"]
    assert evidence["report_texts"] == presentation_payload["text_values"]
    assert evidence["validation_results"]["metric_sections"] == (
        presentation_payload["metric_sections"]
    )
    for key, value in presentation.text_values.items():
        placeholder = "{{" + key + "}}"
        if value and placeholder in presentation.template_placeholder_occurrences:
            assert value in word_text
            assert value in excel_report_text
    sections = evidence["validation_results"]["metric_sections"]
    assert [
        {
            "title": row["title"],
            "table_keys": [table["key"] for table in row.get("tables", [])],
            "chart_keys": [chart["key"] for chart in row.get("charts", [])],
        }
        for row in sections
    ] == contract["agent_sections"]


@pytest.mark.e2e
@pytest.mark.parametrize("run_mode", ["manual", "agent"])
def test_pmml_only_validation_journey_preserves_all_outputs(
    app_client, validation_bundle, run_mode, monkeypatch
):
    if run_mode == "agent":
        monkeypatch.setattr(
            "marvis.routers.validation_agent.generate_word_conclusions",
            lambda **_kwargs: ({
                "TEXT:pressure_test_summary": "压力测试完成。",
                "TEXT:pressure_impact_recommendation": "持续监控。",
                "TEXT:final_validation_conclusion": "验证通过。",
            }, {"source": "deterministic-e2e"}),
        )
    task_id = _create_validation_task(app_client, validation_bundle, run_mode=run_mode)
    _select_four_materials(app_client, task_id, validation_bundle)
    scan = _ok(app_client.post(f"/api/tasks/{task_id}/scan"))
    if scan["validation_input_contract"]["needs_confirmation"]:
        _confirm_contract(
            app_client, task_id, scan["validation_input_contract"], validation_bundle
        )
    _run_pmml_scoring(app_client, task_id)
    _run_metrics_and_stress(app_client, task_id)
    if run_mode == "manual":
        _finish_manual_report(app_client, task_id)
    else:
        _finish_agent_report(app_client, task_id)
    task = _wait_for_status(app_client, task_id, {"succeeded", "review_required"})
    assert task["status"] in {"succeeded", "review_required"}
    evidence = _ok(app_client.get(f"/api/tasks/{task_id}/evidence"))
    assert evidence["pmml_scoring"]["success_count"] == evidence["pmml_scoring"]["input_row_count"]
    assert evidence["validation_results"]["stress_test"]["status"] == "completed"
    if run_mode == "agent":
        messages = _ok(app_client.get(f"/api/tasks/{task_id}/agent/messages"))["messages"]
        stages = {message["stage"] for message in messages}
        assert "word_conclusion_draft" in stages
        assert any("确认" in message["content"] for message in messages)
    _assert_word_excel_agent_non_regression(app_client, task_id)
```

`tests/e2e/validation_bundle.py` defines an immutable `ValidationBundle` and a fixture factory that writes the four selected files plus the exact required confirmation payload (including `model_params`). The deterministic patch above replaces only the LLM response, not the Agent orchestration or report gate.

```python
# tests/e2e/validation_bundle.py
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import nbformat
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationBundle:
    model_name: str
    source_dir: Path
    notebook_path: Path
    sample_path: Path
    pmml_path: Path
    dictionary_path: Path
    confirmation: dict[str, Any]


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix == ".feather":
        frame.to_feather(path)
    elif path.suffix == ".xlsx":
        frame.to_excel(path, index=False)
    else:
        raise ValueError(path.suffix)


def _write_metadata(
    frame: pd.DataFrame, path: Path, *, encoding: str, sheet_name: str
) -> None:
    if path.suffix == ".csv":
        frame.to_csv(path, index=False, encoding=encoding)
        return
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"说明": ["fixture"]}).to_excel(
            writer, sheet_name="README", index=False
        )
        frame.to_excel(writer, sheet_name=sheet_name, index=False)


def build_validation_bundle(
    root: Path, *, pmml_fixture: Path, sample_format: str = "parquet",
    metadata_format: str = "xlsx", derive_month: bool = False,
    missing_importance: bool = False, ambiguous_target: bool = False,
) -> ValidationBundle:
    root.mkdir(parents=True, exist_ok=True)
    notebook_path = root / "model.ipynb"
    source_time = "apply_dt" if derive_month else "apply_month"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        (("TARGET = 'y'\nLABEL = 'alternate_y'\n") if ambiguous_target
         else "RMC_TARGET_COL = 'y'\n") +
        "RMC_SPLIT_COL = 'split'\n"
        f"RMC_TIME_COL = '{'apply_month' if derive_month else source_time}'\n"
        "RMC_PMML_OUTPUT_FIELD = 'probability_1'\n"
        "RMC_MODEL_PARAMS = {'fixture': True}\n"
        + ("df['apply_month'] = df['apply_dt'].astype(str).str[:7]\n" if derive_month else "")
    )]), notebook_path)
    sample = pd.DataFrame({
        "x1": [-2, -1, 0, 1] * 3,
        "x2": [0, 1, 0, 1] * 3,
        "y": [0, 1, 0, 1] * 3,
        "alternate_y": [0, 1, 0, 1] * 3,
        "split": ["train"] * 4 + ["test"] * 4 + ["oot"] * 4,
        source_time: ["2026-01-01"] * 4 + ["2026-02-01"] * 4 + ["2026-03-01"] * 4,
    })
    sample_path = root / f"sample.{sample_format}"
    _write_table(sample, sample_path)
    pmml_path = root / "model.pmml"
    shutil.copy2(pmml_fixture, pmml_path)
    metadata = pd.DataFrame({
        "feature": ["x1", "x2"],
        "category": ["内部", "征信"],
        "importance": [0.6, np.nan if missing_importance else 0.4],
    })
    dictionary_path = root / ("metadata.csv" if metadata_format == "gb18030" else "metadata.xlsx")
    _write_metadata(metadata, dictionary_path, encoding="gb18030", sheet_name="features")
    transformations = ([{
        "operation": "date_to_month",
        "output_field": "apply_month",
        "input_fields": ["apply_dt"],
        "params": {},
    }] if derive_month else [])
    return ValidationBundle(
        model_name=pmml_fixture.stem,
        source_dir=root,
        notebook_path=notebook_path,
        sample_path=sample_path,
        pmml_path=pmml_path,
        dictionary_path=dictionary_path,
        confirmation={
            "target_col": "y",
            "positive_label": 1,
            "negative_label": 0,
            "split_col": "split",
            "split_value_mapping": {"train": "train", "test": "test", "oot": "oot"},
            "time_col": "apply_month" if derive_month else source_time,
            "time_granularity": "month" if derive_month else "date",
            "pmml_output_field": "probability_1",
            "model_params": {"fixture": True},
            "metadata_sheet": None if metadata_format == "gb18030" else "features",
            "feature_col": "feature",
            "category_col": "category",
            "importance_col": "importance",
            "transformations": transformations,
        },
    )
```

The two committed PMML fixtures must be real exports, not hand-labeled generic trees. `scripts/build_tree_pmml_fixtures.py` fits deterministic two-feature binary `XGBClassifier` and `LGBMClassifier` models (`random_state=42`, shallow trees), wraps each in the exporter-supported pipeline, exports through the installed `sklearn2pmml`, removes only volatile timestamp/application-version metadata, and writes canonical UTF-8 XML. `pmml_fixture_provenance.json` records SHA-256, exporter/XGBoost/LightGBM versions, algorithm, model feature names, tree/segment counts, and the generation command. A test regenerates into a temp directory and compares normalized hashes when the exporter toolchain is available; the committed fixtures are always exercised by direct pypmml batch smoke before the flow matrix. Never copy customer model metadata into these fixtures.

```python
@pytest.mark.parametrize(
    "pmml_name", ["xgb_binary_small.pmml", "lightgbm_binary_small.pmml"]
)
def test_tree_pmml_fixture_supports_dataframe_batch_scoring(pmml_name):
    scorer = load_pmml_scorer(
        Path("tests/fixtures") / pmml_name, "probability_1"
    )
    frame = pd.DataFrame({"x1": [0.0, 1.0], "x2": [1.0, 0.0]})
    scores = scorer.score_chunk(frame)
    assert len(scores) == len(frame)
    assert np.isfinite(pd.to_numeric(scores).to_numpy(dtype=float)).all()
```

- [ ] **Step 2: Add representative LGB/XGB and format matrix cases**

Parametrize fixture bundles for:

- XGBoost PMML + CSV + GB18030 metadata;
- LightGBM PMML + Parquet + multi-sheet XLSX metadata;
- Feather sample with persisted fields;
- allowed `date_to_month` transformation;
- ambiguous field candidates requiring confirmation;
- one missing importance row that must block before scoring.

```python
@pytest.mark.e2e
@pytest.mark.parametrize(
    ("pmml_name", "sample_format", "metadata_format", "derive_month", "ambiguous"),
    [
        ("xgb_binary_small.pmml", "csv", "gb18030", False, False),
        ("lightgbm_binary_small.pmml", "parquet", "xlsx", False, False),
        ("xgb_binary_small.pmml", "feather", "xlsx", True, False),
        ("xgb_binary_small.pmml", "parquet", "xlsx", False, True),
    ],
)
def test_pmml_validation_format_and_algorithm_matrix(
    app_client, tmp_path, pmml_name, sample_format, metadata_format,
    derive_month, ambiguous
):
    bundle = build_validation_bundle(
        tmp_path / f"case-{pmml_name}-{sample_format}",
        pmml_fixture=Path("tests/fixtures") / pmml_name,
        sample_format=sample_format,
        metadata_format=metadata_format,
        derive_month=derive_month,
        ambiguous_target=ambiguous,
    )
    task_id = _create_validation_task(app_client, bundle, run_mode="manual")
    _select_four_materials(app_client, task_id, bundle)
    scan = _ok(app_client.post(f"/api/tasks/{task_id}/scan"))
    if ambiguous:
        assert scan["validation_input_contract"]["needs_confirmation"] is True
    if scan["validation_input_contract"]["needs_confirmation"]:
        _confirm_contract(
            app_client, task_id, scan["validation_input_contract"], bundle
        )
    _run_pmml_scoring(app_client, task_id)
    _run_metrics_and_stress(app_client, task_id)
    _finish_manual_report(app_client, task_id)
    _wait_for_status(app_client, task_id, {"succeeded", "review_required"})
    evidence = _ok(app_client.get(f"/api/tasks/{task_id}/evidence"))
    assert evidence["pmml_scoring"]["failure_count"] == 0
    assert evidence["validation_results"]["stress_test"]["status"] == "completed"
    _assert_word_excel_agent_non_regression(app_client, task_id)


@pytest.mark.e2e
def test_missing_importance_blocks_before_pmml_scoring(app_client, tmp_path):
    bundle = build_validation_bundle(
        tmp_path / "missing-importance",
        pmml_fixture=Path("tests/fixtures/xgb_binary_small.pmml"),
        missing_importance=True,
    )
    task_id = _create_validation_task(app_client, bundle, run_mode="manual")
    _select_four_materials(app_client, task_id, bundle)
    scan = app_client.post(f"/api/tasks/{task_id}/scan")
    assert scan.status_code == 422
    scoring = app_client.post(f"/api/tasks/{task_id}/pmml-scoring", json={})
    assert scoring.status_code == 409
    outputs = app_client.app.state.settings.tasks_dir / task_id / "outputs"
    assert not (outputs / "pmml_scores.parquet").exists()
```

- [ ] **Step 3: Add the batch performance contract**

```python
# tests/slow/test_pmml_batch_benchmark.py
import math

import numpy as np
import pandas as pd
import pytest

from marvis.validation.pmml_scoring import PmmlScorer


CHUNK_SIZE = 4_096


class CountingModel:
    def __init__(self):
        self.calls = 0

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame({
            "probability_1": 1.0 / (1.0 + np.exp(-(frame["x1"] - frame["x2"])))
        })


@pytest.mark.slow
def test_batch_scoring_uses_bounded_jvm_calls_instead_of_row_calls():
    frame = pd.DataFrame({
        "x1": np.linspace(-3, 3, 20_000),
        "x2": np.linspace(3, -3, 20_000),
    })
    batch_model = CountingModel()
    batch_scorer = PmmlScorer(batch_model, "probability_1")
    [
        batch_scorer.score_chunk(frame.iloc[start:start + CHUNK_SIZE])
        for start in range(0, len(frame), CHUNK_SIZE)
    ]
    row_model = CountingModel()
    row_scorer = PmmlScorer(row_model, "probability_1")
    [
        row_scorer.score_chunk(frame.iloc[index:index + 1])
        for index in range(len(frame))
    ]
    assert batch_model.calls == math.ceil(len(frame) / CHUNK_SIZE)
    assert row_model.calls == len(frame)
```

The synthetic test is a deterministic JVM-call-count regression guard, not a hardware throughput claim. `scripts/benchmark_pmml_scoring.py` is the real-engine **scorer microbenchmark**: it must print model hash, rows, columns, tree/model metadata when available, load time, warm rows/sec, chunk size, and a transparent estimate for repeated scoring. It is not the end-to-end acceptance gate. Memory sampling covers the Python parent and recursive children throughout model load/warm/score, with separate Python, Java, and total process-tree peaks; measuring only Python RSS is invalid for pypmml. It must not claim a fixed throughput target across hardware.

```python
# scripts/benchmark_pmml_scoring.py
import argparse
import json
from pathlib import Path
import threading
import time
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import psutil

from marvis.files import sha256_file
from marvis.validation.pmml_manifest import parse_pmml_input_manifest
from marvis.validation.pmml_scoring import PmmlScorer, load_pmml_scorer


def _model_node_counts(pmml_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, element in ElementTree.iterparse(pmml_path, events=("start",)):
        name = element.tag.rsplit("}", 1)[-1]
        if name.endswith("Model") or name == "Segment":
            counts[name] = counts.get(name, 0) + 1
    return counts


class ProcessTreePeakRss:
    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.python_peak = 0
        self.java_peak = 0
        self.total_peak = 0

    def _sample(self) -> None:
        try:
            parent = psutil.Process()
            python_rss = parent.memory_info().rss
            java_rss = 0
            child_rss = 0
            for child in parent.children(recursive=True):
                try:
                    rss = child.memory_info().rss
                    child_rss += rss
                    name = child.name().lower()
                    command = " ".join(child.cmdline()[:1]).lower()
                    if "java" in name or "java" in command:
                        java_rss += rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.python_peak = max(self.python_peak, python_rss)
            self.java_peak = max(self.java_peak, java_rss)
            self.total_peak = max(self.total_peak, python_rss + child_rss)
        except psutil.NoSuchProcess:
            return

    def _run(self) -> None:
        self._sample()
        while not self.stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "ProcessTreePeakRss":
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self._sample()

    def as_megabytes(self) -> dict[str, float]:
        divisor = 1024 * 1024
        return {
            "peak_python_rss_mb": self.python_peak / divisor,
            "peak_java_rss_mb": self.java_peak / divisor,
            "peak_process_tree_rss_mb": self.total_peak / divisor,
        }


def _score_all(scorer: PmmlScorer, frame: pd.DataFrame, chunk_size: int) -> int:
    count = 0
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start:start + chunk_size]
        scores = scorer.score_chunk(chunk)
        values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
        if len(values) != len(chunk) or not np.isfinite(values).all():
            raise RuntimeError("benchmark scorer returned invalid rows")
        count += len(values)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmml", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--stress-category-count", type=int, default=1)
    args = parser.parse_args()
    if min(args.rows, args.chunk_size, args.stress_category_count) <= 0:
        parser.error("rows, chunk-size, and stress-category-count must be positive")
    manifest = parse_pmml_input_manifest(args.pmml)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        field: rng.normal(size=args.rows)
        for field in manifest.raw_required_fields
    })
    memory = ProcessTreePeakRss()
    with memory:
        load_started = time.perf_counter()
        scorer = load_pmml_scorer(args.pmml, manifest.output_candidates[0])
        load_seconds = time.perf_counter() - load_started
        scorer.score_chunk(frame.iloc[:min(args.chunk_size, len(frame))])
        started = time.perf_counter()
        scored = _score_all(scorer, frame, args.chunk_size)
        elapsed = time.perf_counter() - started
    rows_per_second = scored / elapsed
    one_million_seconds = 1_000_000 / rows_per_second
    print(json.dumps({
        "pmml_sha256": sha256_file(args.pmml),
        "algorithm": manifest.algorithm,
        "rows": scored,
        "columns": len(manifest.raw_required_fields),
        "model_feature_count": len(manifest.model_features),
        "model_node_counts": _model_node_counts(args.pmml),
        "load_seconds": load_seconds,
        "warm_rows_per_second": rows_per_second,
        **memory.as_megabytes(),
        "chunk_size": args.chunk_size,
        "estimated_baseline_million_seconds": one_million_seconds,
        "estimated_baseline_plus_stress_seconds": one_million_seconds * (
            1 + args.stress_category_count
        ),
        "stress_category_count": args.stress_category_count,
    }, ensure_ascii=False, indent=2))
```

The script imports the same production manifest/scorer/hash functions and exits non-zero on any row-count or non-finite output mismatch. Its gate test parses the emitted JSON and requires `peak_process_tree_rss_mb >= peak_python_rss_mb > 0` and `peak_java_rss_mb > 0`, proving the private Java gateway was included. For non-numeric real models, add `--sample` and draw benchmark chunks from the selected sample rather than fabricating values; never coerce categorical fields to random floats.

Add a separate full-flow benchmark that performs the actual production validation path rather than multiplying a baseline estimate:

```python
# scripts/benchmark_pmml_validation_flow.py (interface and mandatory phase order)
def run_validation_benchmark(
    *, pmml_path: Path, output_dir: Path, rows: int,
    chunk_size: int, category_count: int,
) -> dict[str, Any]:
    if rows != 1_000_000:
        raise ValueError("release gate requires exactly one million rows")
    if category_count < 2:
        raise ValueError("release gate requires at least two stress categories")
    _require_absent_or_empty_directory(output_dir)
    sample_path = output_dir / "benchmark_sample.parquet"
    _write_streamed_parquet_sample(
        sample_path, pmml_path=pmml_path, rows=rows, batch_size=chunk_size
    )
    contract, metadata = _build_ready_benchmark_contract(
        sample_path=sample_path, pmml_path=pmml_path,
        category_count=category_count,
    )
    phase_seconds = {}
    with ProcessTreePeakRss() as memory:
        scorer = CountingScorer(load_pmml_scorer(
            pmml_path, contract.require_output_field()
        ))
        with measured(phase_seconds, "baseline_scoring"):
            scoring = run_pmml_scoring(
                contract=contract, sample_path=sample_path, pmml_path=pmml_path,
                score_path=output_dir / "pmml_scores.parquet",
                chunk_size=chunk_size, scorer=scorer,
            )
        with measured(phase_seconds, "sidecar_validation"):
            validate_pmml_score_artifact(
                scoring, output_dir / "pmml_scores.parquet"
            )
        config = validation_config_from_input_contract(
            contract, _benchmark_metric_settings()
        )
        with measured(phase_seconds, "stress"):
            stress = run_pmml_stress(
                contract=contract, config=config, sample_path=sample_path,
                baseline_score_path=output_dir / "pmml_scores.parquet",
                pmml_path=pmml_path, scenario_dir=output_dir / "stress",
                feature_categories=metadata.per_category_raw_fields,
                chunk_size=chunk_size, scorer=scorer,
                baseline_cache_key=scoring.cache_key,
                cache_dir=output_dir / "stress_cache",
            )
            require_complete_stress_result(stress)
        with measured(phase_seconds, "metrics"):
            results = compute_platform_validation_results(
                task=SimpleNamespace(model_name="benchmark", model_version="v1"),
                contract=contract, sample_path=sample_path,
                score_path=output_dir / "pmml_scores.parquet",
                scoring_result=scoring, metadata_resolution=metadata,
                stress_test=stress, settings=_benchmark_metric_settings(),
            )
    return _benchmark_evidence(
        rows=rows, sample_path=sample_path, scoring=scoring, stress=stress,
        results=results, scorer=scorer, phase_seconds=phase_seconds,
        memory=memory,
    )
```

The implementation is self-contained but calls the production functions shown above. `_write_streamed_parquet_sample` writes the actual PMML raw fields plus binary target, train/test/oot split, and time columns in bounded Arrow batches; fixture-specific generators may be selected by explicit CLI arguments, never guessed from field names. `_build_ready_benchmark_contract` uses the parsed PMML manifest and creates finite 100%-covered metadata with at least two categories. `CountingScorer` delegates one loaded real pypmml scorer and counts `score_chunk` calls; the benchmark must report `scorer_load_count=1`, baseline calls, stress calls, OOT rows, actual category count, input bytes, sidecar bytes, every phase duration, total duration, and Python/Java/process-tree peak RSS.

Acceptance requires: an absent or empty output directory (enforced before creating files, so no prior baseline/stress cache can satisfy the run); exactly 1,000,000 baseline rows; exact sidecar row IDs and finite values; actual metrics completion; actual execution of every configured stress category over every OOT row; `stress.status == "completed"`; no estimated stress duration substituted for execution; one scorer load; and exact scorer calls equal to `ceil(rows/chunk_size) + category_count * ceil(oot_rows/chunk_size)`. This exact bound proves chunk batching, rejects cache hits, and rejects row-wise invocation. Sample generation time is reported separately from validation time. The release gate uses Parquet because CSV/Parquet/Feather scoring and stress are chunked. Metrics intentionally hold only the three control columns plus one score column (`O(rows)` narrow memory), and the measured peak covers that reality. XLSX is not a streaming million-row path: enforce the existing XLSX file/row caps and do not describe it as bounded-memory or use it for this benchmark.

- [ ] **Step 4: Extend Windows packaging smoke coverage**

Add a real packaged-runtime smoke command:

```python
# packaging/windows/smoke_pmml_scoring.py
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from pypmml import Model


def main(pmml_path: Path) -> None:
    java_home = Path(os.environ["JAVA_HOME"])
    if not java_home.is_dir():
        raise RuntimeError(f"private JAVA_HOME does not exist: {java_home}")
    model = Model.fromFile(str(pmml_path))
    frame = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 0.0]})
    prediction = model.predict(frame)
    if len(prediction) != len(frame):
        raise RuntimeError("pypmml batch smoke returned the wrong row count")
    values = pd.to_numeric(prediction["probability_1"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("pypmml batch smoke returned non-finite scores")
    print("private Java + pypmml DataFrame batch smoke ok")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
```

In `build-installer.ps1`, after creating the private runtime and before building the installer, set `JAVA_HOME` and prepend its `bin` directory exactly as the launcher does, then execute:

```powershell
& $RuntimePython "packaging\windows\smoke_pmml_scoring.py" "tests\fixtures\min_lr.pmml"
if ($LASTEXITCODE -ne 0) { throw "Private runtime PMML batch smoke failed" }
```

The Windows CI workflow must run the non-skipped payload build so this command executes. Add static assertions in `tests/test_windows_packaging.py` for the script invocation, private `JAVA_HOME`, error text, DataFrame prediction, row-count check, and finite-value check. Do not add JPMML jars or a system-wide Java dependency.

Also extend `tests/test_python_compat.py`:

```python
import ast


NEW_PMML_RUNTIME_MODULES = (
    "marvis/validation/input_contracts.py",
    "marvis/validation/field_transformations.py",
    "marvis/validation/pmml_manifest.py",
    "marvis/validation/pmml_scoring.py",
    "marvis/validation/pmml_score_artifacts.py",
    "marvis/validation/pmml_stress.py",
)


def test_new_pmml_runtime_modules_parse_as_supported_python_311():
    for relative_path in NEW_PMML_RUNTIME_MODULES:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        ast.parse(source, filename=relative_path, feature_version=(3, 11))
```

- [ ] **Step 5: Run the focused E2E and performance gates**

Run:

```bash
conda run -n py_313 python -m pytest tests/e2e/test_pmml_validation_flow.py -q
conda run -n py_313 python -m pytest tests/slow/test_pmml_batch_benchmark.py -q
conda run -n py_313 python scripts/benchmark_pmml_scoring.py \
  --pmml tests/fixtures/min_lr.pmml --rows 1000000 --chunk-size 10000 \
  --stress-category-count 2
conda run -n py_313 python scripts/benchmark_pmml_validation_flow.py \
  --pmml tests/fixtures/xgb_binary_small.pmml \
  --output-dir .artifacts/pmml-validation-benchmark-xgb \
  --rows 1000000 --chunk-size 10000 --stress-category-count 2
conda run -n py_313 python scripts/benchmark_pmml_validation_flow.py \
  --pmml tests/fixtures/lightgbm_binary_small.pmml \
  --output-dir .artifacts/pmml-validation-benchmark-lightgbm \
  --rows 1000000 --chunk-size 10000 --stress-category-count 2
```

Expected: all E2E cases PASS. The scorer microbenchmark actually scores one million rows and reports its explicit estimate. Each XGBoost and LightGBM full-flow gate separately reads a real one-million-row Parquet file, writes and validates the sidecar, computes metrics, and actually scores all OOT rows for both stress categories before printing measured per-phase/total duration, call counts, artifact sizes, and process-tree RSS. Save all three JSON outputs in the implementation handoff as measured evidence; a full-flow result must contain no estimated stress field and no smaller-row extrapolation may substitute for it.

- [ ] **Step 6: Run the complete repository validation**

Run:

```bash
conda run -n py_313 python -m pytest -q
conda run -n py_313 python -m ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
node --check marvis/static/js/state.js
git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 7: Compare the output inventory against the allowed-change contract**

Run:

```bash
conda run -n py_313 python -m pytest tests/test_validation_non_regression_contract.py -q
```

Expected: PASS. Any unexpected missing Word key/image, Excel sheet, Agent section, chart key, or conclusion field blocks completion.

- [ ] **Step 8: Commit the release gates**

```bash
git add tests/e2e/test_pmml_validation_flow.py tests/e2e/validation_bundle.py \
  tests/slow/test_pmml_batch_benchmark.py tests/fixtures/xgb_binary_small.pmml \
  tests/fixtures/lightgbm_binary_small.pmml tests/fixtures/pmml_fixture_provenance.json \
  scripts/build_tree_pmml_fixtures.py scripts/benchmark_pmml_scoring.py \
  scripts/benchmark_pmml_validation_flow.py \
  packaging/windows/smoke_pmml_scoring.py packaging/windows/build-installer.ps1 \
  .github/workflows/windows-installer.yml tests/test_python_compat.py \
  tests/test_windows_packaging.py \
  tests/test_validation_non_regression_contract.py
git commit -m "test: gate PMML validation non-regression"
```

### Task 18: Final Review and Implementation Handoff

**Files:**
- Review: every file changed by Tasks 1–17
- Update only if needed: `docs/superpowers/plans/2026-07-12-pmml-scoring-validation-non-regression.md`

**Interfaces:**
- Consumes: all task commits and validation output.
- Produces: a clean, reviewable branch ready for user acceptance; no release or push without explicit approval.

- [ ] **Step 1: Verify branch and worktree scope**

Run:

```bash
git status --short
git log --oneline b63ce09e..HEAD
git diff --stat b63ce09e..HEAD
```

Expected: clean implementation worktree; only the planned commits are present; the original main worktree's two dirty files remain untouched.

- [ ] **Step 2: Run a requirements traceability check**

For each numbered acceptance item in
`docs/superpowers/specs/2026-07-12-pmml-scoring-validation-non-regression-design.md`, record the implementing task and test command in the final handoff. Missing traceability blocks completion.

- [ ] **Step 3: Request code review using the required review skill**

Invoke `superpowers:requesting-code-review` and provide:

```text
BASE_SHA=b63ce09e
HEAD_SHA=<current HEAD>
Scope: PMML-only validation migration; all non-consistency outputs must remain stable.
Primary risks: historical result compatibility, row alignment, pressure completeness,
Word/Excel parity, Agent/UI output loss, cancellation and cache invalidation.
```

- [ ] **Step 4: Address accepted review findings with focused tests and commits**

Use `superpowers:receiving-code-review` before applying feedback. Each accepted finding gets a reproducing test, minimal fix, focused test run, and narrow commit; rejected findings get an evidence-backed explanation.

- [ ] **Step 5: Re-run the full validation after review fixes**

Run:

```bash
conda run -n py_313 python -m pytest -q
conda run -n py_313 python -m ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
node --check marvis/static/js/state.js
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 6: Stop at release readiness**

Report exact test counts, benchmark results, branch, commit list, remaining risks, and whether the Windows installer was tested on a real Windows host. Do not run `scripts/release_push.py`, create a tag, push, or publish unless the user explicitly authorizes release.
