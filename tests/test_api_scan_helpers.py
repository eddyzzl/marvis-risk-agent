import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from marvis.api_scan_helpers import (
    assemble_validation_input_contract,
    format_notebook_contract_error,
    material_candidates_payload,
    perform_scan_task,
)
from marvis.db import TaskRepository, connect, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.notebook_contract import NotebookContractError, precheck_notebook_contract
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.validation.feature_metadata import FeatureMetadataSelection
from marvis.validation.input_contracts import (
    FIELD_RECOGNITION_SCHEMA,
    PMML_INPUT_MANIFEST_SCHEMA,
    FieldRecognitionResult,
    PmmlInputManifest,
    SampleSchema,
    StressUnit,
    TransformationSpec,
)
from tests.validation_material_builders import create_repository_validation_task


class FailingScanStatusRepository(TaskRepository):
    def update_status_on_connection(self, *args, **kwargs):
        raise RuntimeError("status write failed")


def _as_historical_validation_task(repo: TaskRepository, task):
    with connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET validation_workflow_version = 1 WHERE id = ?",
            (task.id,),
        )
    return repo.get_task(task.id)


def test_v2_scan_builds_pending_contract_without_legacy_notebook_precheck_or_sample_rehash(
    tmp_path, monkeypatch
):
    task, repo, settings = create_repository_validation_task(
        tmp_path,
        notebook_source=(
            "RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\n"
            "RMC_TIME_COL='apply_month'\nRMC_PMML_OUTPUT_FIELD='probability_1'\n"
            "RMC_MODEL_PARAMS={}\n"
        ),
    )
    sample_path = (Path(task.source_dir) / str(task.sample_path)).resolve()
    original_sha256_file = __import__(
        "marvis.api_scan_helpers", fromlist=["sha256_file"]
    ).sha256_file

    def guarded_sha256_file(path):
        if Path(path).resolve() == sample_path:
            raise AssertionError("sample hash must be reused from SampleSchema")
        return original_sha256_file(path)

    monkeypatch.setattr(
        "marvis.api_scan_helpers.precheck_notebook_contract",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("legacy Notebook precheck must not run")
        ),
    )
    monkeypatch.setattr(
        "marvis.api_scan_helpers.inspect_notebook_contract",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("legacy Notebook contract scan must not run")
        ),
    )
    monkeypatch.setattr("marvis.api_scan_helpers.sha256_file", guarded_sha256_file)

    payload = perform_scan_task(repo, task, settings)

    contract_payload = payload["validation_input_contract"]
    assert contract_payload["status"] == "pending_confirmation"
    assert contract_payload["needs_confirmation"] is True
    assert contract_payload["read_only"] is True
    assert contract_payload["revision"] == 1
    contract = contract_payload["contract"]
    assert contract["material_hashes"]["sample"] == contract["sample_schema"]["sha256"]
    assert repo.get_task(task.id).status == TaskStatus.SCANNED
    assert ValidationContractRepository(settings.db_path).get(task.id) is not None


def _scan_assembly_inputs(*, transformations=()):
    schema = SampleSchema(
        path="sample.parquet",
        columns=("raw",),
        dtypes={"raw": "double"},
        row_count=1,
        preview_row_count=0,
        encoding=None,
        sha256="a" * 64,
        sheet_name=None,
    )
    fields = FieldRecognitionResult(
        schema_version=FIELD_RECOGNITION_SCHEMA,
        notebook_sha256="b" * 64,
        candidates={},
        transformations=tuple(transformations),
        conflicts=(),
        diagnostics=(),
    )
    manifest = PmmlInputManifest(
        schema_version=PMML_INPUT_MANIFEST_SCHEMA,
        raw_required_fields=("raw",),
        derived_fields=(),
        model_features=("raw",),
        stress_units=(StressUnit("raw", ("raw",), ()),),
        unsupported_derivations=(),
        output_candidates=("probability_1",),
        algorithm="xgb",
    )
    return schema, fields, manifest


def test_scan_assembly_handles_long_transformation_candidates_iteratively():
    transformations = []
    previous = "raw"
    for index in range(1_500):
        output = f"derived_{index}"
        transformations.append(
            TransformationSpec("copy", output, (previous,), {})
        )
        previous = output
    schema, fields, manifest = _scan_assembly_inputs(
        transformations=transformations
    )
    manifest = replace(
        manifest,
        raw_required_fields=(previous,),
        model_features=(previous,),
        stress_units=(StressUnit(previous, ("raw",), ()),),
    )

    contract = assemble_validation_input_contract(
        material_hashes={
            "notebook": "b" * 64,
            "sample": "a" * 64,
            "pmml": "c" * 64,
            "dictionary": "d" * 64,
        },
        sample_schema=schema,
        fields=fields,
        manifest=manifest,
        metadata=None,
        metadata_selections=(
            FeatureMetadataSelection(None, "feature", "category", "importance"),
        ),
        conflicts=(),
    )

    assert contract.status == "pending_confirmation"
    assert not contract.conflicts


@pytest.mark.parametrize(
    ("manifest_update", "expected"),
    [({"algorithm": ""}, "algorithm"), ({"output_candidates": ()}, "output")],
)
def test_scan_assembly_blocks_unconfirmable_pmml_identity(manifest_update, expected):
    schema, fields, manifest = _scan_assembly_inputs()
    manifest = replace(manifest, **manifest_update)

    contract = assemble_validation_input_contract(
        material_hashes={
            "notebook": "b" * 64,
            "sample": "a" * 64,
            "pmml": "c" * 64,
            "dictionary": "d" * 64,
        },
        sample_schema=schema,
        fields=fields,
        manifest=manifest,
        metadata=None,
        metadata_selections=(),
        conflicts=(),
    )

    assert contract.status == "blocked"
    assert any(expected in conflict.lower() for conflict in contract.conflicts)


def test_scan_assembly_blocks_when_metadata_has_no_confirmable_selection():
    schema, fields, manifest = _scan_assembly_inputs()

    contract = assemble_validation_input_contract(
        material_hashes={
            "notebook": "b" * 64,
            "sample": "a" * 64,
            "pmml": "c" * 64,
            "dictionary": "d" * 64,
        },
        sample_schema=schema,
        fields=fields,
        manifest=manifest,
        metadata=None,
        metadata_selections=(),
        conflicts=(),
    )

    assert contract.status == "blocked"
    assert "feature metadata selection is unavailable" in contract.conflicts


def test_v2_scan_bounds_combined_contract_error_message(tmp_path):
    task, repo, settings = create_repository_validation_task(
        tmp_path,
        notebook_source="RMC_TARGET_COL='y'",
    )
    aliases = {
        "feature": ["feature", "特征名", "特征名称", "指标英文"],
        "category": ["category", "类别", "分类", "数据源"],
        "importance": ["importance", "feature_importance", "gain", "权重"],
    }
    header = [*aliases["feature"], *aliases["category"], *aliases["importance"]]
    values = ["missing"] * 4 + ["内部"] * 4 + ["1"] * 4
    (Path(task.source_dir) / str(task.dictionary_path)).write_text(
        ",".join(header) + "\n" + ",".join(values) + "\n",
        encoding="utf-8",
    )

    payload = perform_scan_task(repo, task, settings)

    contract_check = next(
        check for check in payload["checks"] if check["id"] == "validation_input_contract"
    )
    assert contract_check["status"] == "error"
    assert len(contract_check["message"]) <= 2_000


def test_perform_scan_task_rolls_back_artifacts_when_status_update_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(TaskCreate(
        model_name="A卡",
        model_version="v1",
        validator="qa",
        source_dir=str(tmp_path / "source"),
    ))
    task = _as_historical_validation_task(repo, task)
    task_dir = tmp_path / "tasks" / task.id
    execution_dir = task_dir / "execution"
    outputs_dir = task_dir / "outputs"
    images_dir = task_dir / "images"
    execution_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    (execution_dir / "scan_result.json").write_text('{"old": true}', encoding="utf-8")
    (execution_dir / "notebook_steps.json").write_text('{"steps": ["old"]}', encoding="utf-8")
    (outputs_dir / "validation.xlsx").write_bytes(b"old-xlsx")
    (images_dir / "roc.png").write_bytes(b"old-png")
    monkeypatch.setattr("marvis.api_scan_helpers.scan_source_dir", lambda _path: [])

    failing_repo = FailingScanStatusRepository(db_path)
    settings = SimpleNamespace(tasks_dir=tmp_path / "tasks")
    with pytest.raises(RuntimeError, match="status write failed"):
        perform_scan_task(failing_repo, task, settings)

    assert (execution_dir / "scan_result.json").read_text(encoding="utf-8") == '{"old": true}'
    assert (execution_dir / "notebook_steps.json").read_text(encoding="utf-8") == '{"steps": ["old"]}'
    assert (outputs_dir / "validation.xlsx").read_bytes() == b"old-xlsx"
    assert (images_dir / "roc.png").read_bytes() == b"old-png"
    assert repo.get_task(task.id).status == TaskStatus.CREATED


def test_v2_scan_rolls_back_contract_and_artifacts_with_failed_status_write(tmp_path):
    task, repo, settings = create_repository_validation_task(
        tmp_path,
        notebook_source=(
            "RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\n"
            "RMC_TIME_COL='apply_month'\nRMC_MODEL_PARAMS={}\n"
        ),
    )
    execution_dir = settings.tasks_dir / task.id / "execution"
    execution_dir.mkdir(parents=True)
    (execution_dir / "scan_result.json").write_text('{"old": true}', encoding="utf-8")

    failing_repo = FailingScanStatusRepository(settings.db_path)
    with pytest.raises(RuntimeError, match="status write failed"):
        perform_scan_task(failing_repo, task, settings)

    assert (execution_dir / "scan_result.json").read_text(encoding="utf-8") == '{"old": true}'
    assert ValidationContractRepository(settings.db_path).get(task.id) is None
    assert repo.get_task(task.id).status == TaskStatus.CREATED


def test_material_candidates_payload_lists_extra_csv_without_selecting_it(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    notebook_path = source / "model.ipynb"
    sample_path = source / "sample.parquet"
    extra_csv = source / "feature_importance_best.csv"
    pmml_path = source / "model.pmml"
    dictionary_path = source / "dictionary.csv"
    nbformat.write(nbformat.v4.new_notebook(cells=[]), notebook_path)
    sample_path.write_bytes(b"PAR1")
    extra_csv.write_text("feature,importance\nx1,1\n", encoding="utf-8")
    pmml_path.write_text("<PMML/>", encoding="utf-8")
    dictionary_path.write_text("特征名,类别\nx1,基础信息\n", encoding="utf-8")

    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source),
            sample_path="sample.parquet",
        )
    )
    task = _as_historical_validation_task(repo, task)

    payload = material_candidates_payload(task)

    sample_candidates = payload["candidates"]["sample"]
    assert {item["relative_path"] for item in sample_candidates} == {
        "dictionary.csv",
        "feature_importance_best.csv",
        "sample.parquet",
    }
    by_path = {item["relative_path"]: item for item in sample_candidates}
    assert by_path["feature_importance_best.csv"]["role"] == "data_dictionary"
    assert payload["selection"]["sample_path"] == "sample.parquet"


def test_contract_syntax_error_reports_notebook_revision_and_source_excerpt(tmp_path):
    notebook_path = tmp_path / "model.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "RMC_SAMPLE_DF = sample_df\n"
                    "RMC_TARGET_COL = 'y'\n"
                    "RMC_ALGORITHM = 'xgb'\n"
                    "def RMC_SCORE_FN(df):\n"
                    "return [0.1] * len(df)\n"
                )
            ]
        ),
        notebook_path,
    )

    with pytest.raises(NotebookContractError) as excinfo:
        precheck_notebook_contract(notebook_path)

    message = format_notebook_contract_error(excinfo.value)
    expected_sha256 = sha256(notebook_path.read_bytes()).hexdigest()
    assert str(notebook_path.resolve()) in message
    assert expected_sha256 in message
    assert "代码单元 0，第 5 行" in message
    assert "L5: return [0.1] * len(df)" in message


def test_missing_contract_error_reports_notebook_revision(tmp_path):
    notebook_path = tmp_path / "model.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "RMC_SAMPLE_DF = sample_df\n"
                    "RMC_TARGET_COL = 'y'\n"
                    "RMC_ALGORITHM = 'xgb'\n"
                )
            ]
        ),
        notebook_path,
    )

    with pytest.raises(NotebookContractError) as excinfo:
        precheck_notebook_contract(notebook_path)

    message = format_notebook_contract_error(excinfo.value)
    assert str(notebook_path.resolve()) in message
    assert sha256(notebook_path.read_bytes()).hexdigest() in message


def test_perform_scan_task_archives_previous_notebook_revision(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    notebook_path = source / "model.ipynb"
    _write_contract_notebook(notebook_path, algorithm="xgb")
    (source / "sample.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    (source / "model.pmml").write_text("<PMML/>", encoding="utf-8")
    (source / "dictionary.csv").write_text("feature,category\nx,base\n", encoding="utf-8")

    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source),
            notebook_path="model.ipynb",
            sample_path="sample.csv",
            pmml_path="model.pmml",
            dictionary_path="dictionary.csv",
        )
    )
    task = _as_historical_validation_task(repo, task)
    settings = SimpleNamespace(tasks_dir=tmp_path / "tasks")

    first = perform_scan_task(repo, task, settings)
    _write_contract_notebook(notebook_path, algorithm="lgb")
    second = perform_scan_task(repo, repo.get_task(task.id), settings)

    assert first["scan_id"] != second["scan_id"]
    assert first["notebook_revision"]["sha256"] != second["notebook_revision"]["sha256"]
    assert second["previous_scan"]["scan_id"] == first["scan_id"]
    assert (
        second["previous_scan"]["notebook_revision"]["sha256"]
        == first["notebook_revision"]["sha256"]
    )
    history_files = list(
        (tmp_path / "tasks" / task.id / "execution" / "scan_history").glob("*.json")
    )
    assert len(history_files) == 1
    archived = json.loads(history_files[0].read_text(encoding="utf-8"))
    assert archived["scan_id"] == first["scan_id"]


def _write_contract_notebook(path, *, algorithm):
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "RMC_SAMPLE_DF = sample_df\n"
                    "RMC_TARGET_COL = 'y'\n"
                    f"RMC_ALGORITHM = {algorithm!r}\n"
                    "def RMC_SCORE_FN(df):\n"
                    "    return [0.1] * len(df)\n"
                )
            ]
        ),
        path,
    )
