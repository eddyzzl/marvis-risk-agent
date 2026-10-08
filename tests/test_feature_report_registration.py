import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.packs.feature.tools import tool_generate_feature_report
from marvis.repositories.task_artifacts import TaskArtifactRepository
from test_feature_pack import _runtime


def test_feature_report_registers_exact_bytes_and_inputs_and_preserves_old_output(tmp_path):
    _runtime(tmp_path)
    ctx = SimpleNamespace(workspace=tmp_path / "workspace", task_id="task-feature")
    inputs = {"metrics": [{"feature": "x", "iv": .1}], "binning": [], "collinear": None}
    first = tool_generate_feature_report(inputs, ctx)
    path = Path(first["report_path"])
    before = path.read_bytes()
    repo = TaskArtifactRepository(ctx.workspace / "marvis.sqlite")
    row = repo.get_for_task(ctx.task_id, first["artifact_id"])
    assert row["kind"] == "feature_report_xlsx"
    assert row["origin_tool"] == "feature.generate_feature_report"
    assert row["content_hash"] == first["artifact_content_hash"] == digest(before)
    assert row["provenance"]["input_hash"] == digest(inputs)
    assert row["provenance"]["feature_count"] == 1
    second = tool_generate_feature_report({"metrics": [{"feature": "x", "iv": .9}]}, ctx)
    assert second["artifact_id"] != first["artifact_id"]
    assert Path(second["report_path"]).exists() and path.read_bytes() == before
    assert len(repo.list_for_task(ctx.task_id)) == 2


def test_feature_report_registration_failure_rolls_back_files_and_registry(tmp_path, monkeypatch):
    _runtime(tmp_path)
    ctx = SimpleNamespace(workspace=tmp_path / "workspace", task_id="task-feature")
    first = tool_generate_feature_report({"metrics": [{"feature": "x", "iv": .1}]}, ctx)
    path = Path(first["report_path"])
    before = path.read_bytes()
    repo = TaskArtifactRepository(ctx.workspace / "marvis.sqlite")
    rows = repo.list_for_task(ctx.task_id)
    register = TaskArtifactRepository.register_on_connection

    def fail(self, conn, **kwargs):
        register(self, conn, **kwargs)
        raise RuntimeError("registration interrupted")
    monkeypatch.setattr(TaskArtifactRepository, "register_on_connection", fail)
    with pytest.raises(RuntimeError, match="registration interrupted"):
        tool_generate_feature_report({"metrics": [{"feature": "x", "iv": .9}]}, ctx)
    assert repo.list_for_task(ctx.task_id) == rows
    assert list(path.parent.glob("*.xlsx")) == [path]
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".feature-report-*"))
    assert not (path.parent / ".staging").exists()


def test_feature_report_output_schema_declares_registered_identity():
    manifest = json.loads((Path(__file__).parents[1] / "marvis/packs/feature/manifest.json").read_text())
    schema = next(t["output_schema"] for t in manifest["tools"] if t["name"] == "generate_feature_report")
    assert {"artifact_id", "artifact_content_hash"} <= set(schema["properties"])
