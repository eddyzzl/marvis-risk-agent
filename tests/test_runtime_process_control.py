"""Actual owned-service crash/restart, isolated workspace and no remote model."""
import json
import time
from pathlib import Path

from test_runtime_agent_benchmark import base_case, fixture_model

from marvis.orchestrator.eval.runtime_contracts import RuntimeCase
from marvis.orchestrator.eval.runtime_runner import _application, _source_identity


def test_service_restart_uses_new_process_and_retains_both_original_journals(tmp_path):
    case = RuntimeCase.model_validate(base_case("restart-local", "data_join", []))
    source = _source_identity()
    state = {}
    with fixture_model() as (model, calls):
        with _application(tmp_path, case, model.profile("fixture_model"), tmp_path, {}, source["source_sha256"], (),
                          process_state=state, origin_ns=time.monotonic_ns()) as (client, workspace, _deadline):
            control = client._marvis_process_control
            before = control.snapshot()
            marker = workspace / "preserved-original.txt"
            marker.write_bytes(b"local fixture workspace survives actual crash")
            restarted = control.restart_same_workspace()
            assert restarted["before"] == before
            assert restarted["after"]["pid"] != before["pid"]
            assert restarted["after"]["create_time"] != before["create_time"]
            assert restarted["after"]["workspace_sha256"] == before["workspace_sha256"]
            assert marker.read_bytes() == b"local fixture workspace survives actual crash"
            assert client.get("/api/branding").status_code == 200
            assert restarted["old_returncode"] < 0 and restarted["signal"] == "SIGKILL"
    assert not calls and state["stopped"]
    assert len(state["process_epochs"]) == 2
    first, second = state["process_epochs"]
    original = Path(first["process_observations_path"]).read_bytes()
    resumed = Path(second["process_observations_path"]).read_bytes()
    assert original and resumed and original != resumed
    assert json.loads(resumed.splitlines()[-1])["event"] == "closed"
    assert all(Path(e["process_timing_path"]).exists() for e in state["process_epochs"])
