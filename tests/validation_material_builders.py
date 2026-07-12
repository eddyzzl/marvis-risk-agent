from __future__ import annotations

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
    root: Path,
    *,
    notebook_source: str,
    sample: pd.DataFrame | None = None,
    dictionary: pd.DataFrame | None = None,
) -> ValidationMaterialBundle:
    root.mkdir(parents=True, exist_ok=True)
    notebook_path = root / "model.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(notebook_source)]
        ),
        notebook_path,
    )
    sample_path = root / "sample.parquet"
    source_sample = sample if sample is not None else pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "y": [0, 1, 0, 1],
            "split": ["train", "test", "oot", "oot"],
            "apply_month": ["202601", "202602", "202603", "202603"],
        }
    )
    source_sample.to_parquet(sample_path, index=False)
    pmml_path = root / "model.pmml"
    shutil.copy2(Path(__file__).parent / "fixtures" / "min_lr.pmml", pmml_path)
    dictionary_path = root / "metadata.csv"
    source_dictionary = dictionary if dictionary is not None else pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "category": ["内部", "征信"],
            "importance": [0.6, 0.4],
        }
    )
    source_dictionary.to_csv(dictionary_path, index=False)
    return ValidationMaterialBundle(
        root=root,
        notebook_path=notebook_path,
        sample_path=sample_path,
        pmml_path=pmml_path,
        dictionary_path=dictionary_path,
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
    task = repo.create_task(
        TaskCreate(
            model_name="fixture",
            model_version="v2",
            validator="pytest",
            source_dir=str(bundle.root),
            notebook_path=bundle.notebook_path.name,
            sample_path=bundle.sample_path.name,
            pmml_path=bundle.pmml_path.name,
            dictionary_path=bundle.dictionary_path.name,
        )
    )
    return task, repo, settings


def create_api_validation_task(
    client, bundle: ValidationMaterialBundle
) -> tuple[str, dict]:
    created = client.post(
        "/api/tasks",
        json={
            "task_type": "validation",
            "model_name": "fixture",
            "model_version": "v2",
            "validator": "pytest",
            "source_dir": str(bundle.root),
            "run_mode": "manual",
        },
    )
    assert created.status_code == 200, created.text
    task_id = str(created.json()["id"])
    selected = client.put(
        f"/api/tasks/{task_id}/materials",
        json={
            "notebook_path": bundle.notebook_path.name,
            "sample_path": bundle.sample_path.name,
            "pmml_path": bundle.pmml_path.name,
            "dictionary_path": bundle.dictionary_path.name,
        },
    )
    assert selected.status_code == 200, selected.text
    scanned = client.post(f"/api/tasks/{task_id}/scan")
    assert scanned.status_code == 200, scanned.text
    return task_id, scanned.json()
