import json
from hashlib import sha256
from types import SimpleNamespace

import nbformat
import pytest

from marvis.api_scan_helpers import (
    format_notebook_contract_error,
    material_candidates_payload,
    perform_scan_task,
)
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.notebook_contract import NotebookContractError, precheck_notebook_contract


class FailingScanStatusRepository(TaskRepository):
    def update_status_on_connection(self, *args, **kwargs):
        raise RuntimeError("status write failed")


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

    payload = material_candidates_payload(task)

    sample_candidates = payload["candidates"]["sample"]
    assert {item["relative_path"] for item in sample_candidates} == {
        "dictionary.csv",
        "feature_importance_best.csv",
        "sample.parquet",
    }
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
