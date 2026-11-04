"""Native local journeys; no paid/model transport and no mocked computation.

Notebook loopback isolation applies to executable V1 Notebook jobs. Standard V2
only reads Notebook literals and calculates with the host's PMML/JVM scorer; its
Notebook setting must never be reported as isolation evidence for that JVM.
"""

from dataclasses import replace
import asyncio
import json
from pathlib import Path
import sys

from docx import Document
from fastapi import BackgroundTasks
import nbformat
import numpy as np
import pandas as pd
import pytest

from marvis.agent.service import fallback_word_conclusions
from marvis.agent.validation_app_service import (
    confirm_agent_report_conclusions,
    run_agent_validation_job,
)
from marvis.agent.validation_evidence import agent_evidence_from_settings
from marvis.api_stage_helpers import agent_pipeline_settings
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.execution_environment import (
    ExecutionEnvironmentSettings,
    save_execution_environment,
)
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.settings import build_settings
from marvis.validation.input_confirmation import validate_confirmation_against_materials
from marvis.validation_materials import resolve_selected_validation_materials
from marvis.validation_batch_runner import run_validation_batch, run_validation_batch_job
from tests.test_local_execution_network import _NATIVE_PROBE
from tests.test_validation_batch_runner import _batch_with_valid_child
from tests.validation_builders import make_validation_confirmation
from tests.validation_material_builders import write_validation_material_bundle


pytestmark = [
    pytest.mark.pmml_runtime,
    pytest.mark.skipif(sys.platform != "darwin", reason="macOS native execution journeys"),
]


def _confirm_report(settings, repo, task_id):
    task = repo.get_task(task_id)
    _values, revision = repo.get_report_values(task_id)
    background = BackgroundTasks()
    confirm_agent_report_conclusions(
        repo_=repo, task=task, task_id=task_id, settings=settings,
        text_values=fallback_word_conclusions(
            task=task, evidence=agent_evidence_from_settings(settings, task_id),
        ),
        expected_revision=revision, background_tasks=background,
    )
    asyncio.run(background())


def _confirm_v2_materials(settings, repo, task_id):
    task = repo.get_task(task_id)
    contracts = ValidationContractRepository(settings.db_path)
    candidate = contracts.get(task_id)
    assert candidate is not None
    materials = resolve_selected_validation_materials(task)
    validated = validate_confirmation_against_materials(
        contract=candidate.contract, sample_path=materials.sample,
        dictionary_path=materials.dictionary,
        requested=replace(make_validation_confirmation(), metadata_sheet=None),
    )
    contracts.confirm(
        task_id, validated.values, expected_revision=candidate.revision,
        resolved_sample_schema=validated.sample_schema,
        resolved_feature_metadata=validated.feature_metadata,
    )


def test_agent_v1_propagates_loopback_mode_to_real_notebook_and_report(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    # The property may fall back to the packaged template until a workspace
    # override exists. Always write the explicit temporary workspace path.
    template_path = settings.workspace / "report_templates" / "default.docx"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template = Document()
    template.add_paragraph("Native local Agent report: {{TEXT:model_name}}")
    template.save(template_path)
    assert settings.report_template_path == template_path
    sample = pd.DataFrame([
        {"x1": (i + 1) / 41, "x2": 0.0, "y": int(i >= 20), "split": split,
         "apply_month": "202601" if split == "train" else "202602"}
        for split in ("train", "test", "oot") for i in range(40)
    ])
    code = _NATIVE_PROBE + f"""
child = subprocess.run([sys.executable, '-c', {_NATIVE_PROBE!r}], capture_output=True, text=True, timeout=10)
assert child.returncode == 0, child.stderr
print('agent-native-descendant-denied')
import pandas as pd
import numpy as np
RMC_SAMPLE_DF = pd.read_parquet('sample.parquet')
RMC_TARGET_COL = 'y'
RMC_ALGORITHM = 'lr'
RMC_SPLIT_COL = 'split'
RMC_TIME_COL = 'apply_month'
class LocalModel:
    def score(self, frame):
        return 1 / (1 + np.exp(-(frame['x1'] - frame['x2'])))
local_model = LocalModel()
def RMC_SCORE_FN(frame):
    return local_model.score(frame)
RMC_FEATURE_IMPORTANCE = pd.DataFrame({{'feature': ['x1', 'x2'], 'importance': [0.8, 0.2]}})
RMC_MODEL_PARAMS = {{'synthetic_fixture': True}}
"""
    bundle = write_validation_material_bundle(
        settings.workspace / "materials", notebook_source=code, sample=sample,
        dictionary=pd.DataFrame({"特征名": ["x1", "x2"], "类别": ["基础", "基础"], "importance": [0.8, 0.2]}),
    )
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(TaskCreate(
        model_name="Synthetic local LR", model_version="native", validator="qa",
        source_dir=str(bundle.root), run_mode="agent",
        notebook_path=str(bundle.notebook_path), sample_path=str(bundle.sample_path),
        pmml_path=str(bundle.pmml_path), dictionary_path=str(bundle.dictionary_path),
    ))
    with repo.transaction() as conn:
        conn.execute("UPDATE tasks SET validation_workflow_version = 1 WHERE id = ?", (task.id,))
    save_execution_environment(settings.workspace, ExecutionEnvironmentSettings(notebook_network_mode="loopback_only"))
    # Advance the three deterministic stages through real Agent jobs. An empty
    # profile takes stage-summary fallback before transport; report prose is
    # explicitly supplied below, rather than pretending a model generated it.
    for stage in ("scan", "reproducibility", "metrics"):
        job_id = repo.start_job(task.id, "agent")
        run_agent_validation_job(job_id, settings, task.id, {}, stage=stage)
        job = repo.get_job(job_id)
        assert job["status"] == "succeeded", job
    assert repo.get_task(task.id).status is TaskStatus.WRITING_ARTIFACTS
    execution = settings.tasks_dir / task.id / "execution"
    policy = json.loads((execution / "notebook_execution.json").read_text())
    assert policy["mode"] == "loopback_only"
    assert policy["backend"] == "macos_seatbelt"
    assert policy["state"] == "completed"
    assert policy["network_restricted"] and policy["subprocess_worker"]
    executed = list(execution.glob("*.ipynb"))
    assert executed
    outputs = "".join(
        output.get("text", "")
        for path in executed for cell in nbformat.read(path, as_version=4).cells
        for output in cell.get("outputs", [])
    )
    assert "agent-native-descendant-denied" in outputs
    artifacts = settings.tasks_dir / task.id / "outputs"
    reproducibility = json.loads((artifacts / "reproducibility_result.json").read_text())
    assert reproducibility["summary"]["mismatch_count"] == 0
    assert reproducibility["summary"]["match_count"] == len(sample)
    results = json.loads((artifacts / "validation_results.json").read_text())
    assert {item["split"] for item in results["effectiveness"]["overall"]} == {"train", "test", "oot"}
    assert not (artifacts / "validation_report.docx").exists()
    assert any(message["metadata"].get("fallback") for message in repo.list_agent_messages(task.id))
    _confirm_report(settings, repo, task.id)
    assert repo.get_task(task.id).status in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}
    assert (artifacts / "validation_report.docx").stat().st_size > 0
    assert (artifacts / "validation.xlsx").stat().st_size > 0


def test_standard_v2_two_model_journey_keeps_notebook_static_and_scores_real_pmml(tmp_path):
    settings, batches, batch = _batch_with_valid_child(tmp_path, child_count=2)
    repo = TaskRepository(settings.db_path)
    save_execution_environment(settings.workspace, ExecutionEnvironmentSettings(notebook_network_mode="loopback_only"))
    sentinels = []
    for item in batches.list_items(batch.parent_task_id):
        task = repo.get_task(item.child_task_id)
        sentinel = Path(task.source_dir) / "must-not-execute.txt"
        sentinels.append(sentinel)
        notebook = nbformat.read(task.notebook_path, as_version=4)
        notebook.cells.append(nbformat.v4.new_code_cell(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n"
            "raise AssertionError('standard V2 must only inspect this Notebook')"
        ))
        nbformat.write(notebook, task.notebook_path)
        assert agent_pipeline_settings(settings, task).notebook_network_mode == "loopback_only"

    run_validation_batch(settings=settings, parent_task_id=batch.parent_task_id)
    items = batches.list_items(batch.parent_task_id)
    assert all(item.stage == "input_confirmation" for item in items)
    for item in items:
        _confirm_v2_materials(settings, repo, item.child_task_id)
    job_id = repo.start_job(batch.parent_task_id, "validation_batch")
    run_validation_batch_job(job_id=job_id, settings=settings, parent_task_id=batch.parent_task_id)
    assert repo.get_job(job_id)["status"] == "succeeded"
    pending = batches.list_items(batch.parent_task_id)
    assert all(item.stage == "report_conclusion" for item in pending)
    assert batches.get_batch(batch.parent_task_id).status == "awaiting_confirmation"
    for item in pending:
        task = repo.get_task(item.child_task_id)
        task_dir = settings.tasks_dir / item.child_task_id
        outputs = task_dir / "outputs"
        assert not (task_dir / "execution" / "notebook_execution.json").exists()
        assert not list((task_dir / "execution").glob("*.ipynb"))
        evidence = json.loads((outputs / "pmml_scoring_result.json").read_text())
        assert evidence["status"] == "pass"
        assert evidence["input_row_count"] == evidence["success_count"] == 12
        scores = pd.read_parquet(outputs / "pmml_scores.parquet")
        sample = pd.read_parquet(task.sample_path)
        expected = 1 / (1 + np.exp(-(sample["x1"] - sample["x2"])))
        np.testing.assert_allclose(scores["pmml_score"], expected)
        metrics = json.loads((outputs / "validation_results.json").read_text())
        assert {row["split"] for row in metrics["effectiveness"]["overall"]} == {"train", "test", "oot"}
        assert not (outputs / "validation_report.docx").exists()
        _confirm_report(settings, repo, task.id)
        assert (outputs / "validation_report.docx").stat().st_size > 0
        assert (outputs / "validation.xlsx").stat().st_size > 0
    assert not any(path.exists() for path in sentinels)
    completed = batches.get_batch(batch.parent_task_id)
    assert completed.status == "completed"
    assert Path(completed.summary_path).stat().st_size > 0
    assert all(item.status == "succeeded" for item in batches.list_items(batch.parent_task_id))
