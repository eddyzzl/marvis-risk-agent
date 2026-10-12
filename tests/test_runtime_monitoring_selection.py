"""Final selection is an authenticated output pointer, not historical status."""
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.runtime_monitoring import _selected_monitoring_experiment
from marvis.orchestrator.eval.runtime_runner import RuntimeJourneyError


def fixture(*, selected="child", artifact="child-artifact", plans=1):
    experiments = [{"id": name, "task_id": "task", "status": "selected", "artifact_id": name + "-artifact"}
                   for name in ("parent", "child")]
    calls = []
    def request(method, path, *, label):
        calls.append((method, path, label))
        return {"selected_experiment_id": selected, "artifact_id": artifact}
    journey = SimpleNamespace(task_id="task", json_request=request,
        plans=lambda: [{"task_id": "task", "status": "done", "steps": [
            {"id": f"select-{i}", "tool_ref": {"plugin": "modeling", "tool": "select_experiment"},
             "status": "done", "output_ref": f"metrics:select-{i}:v2"}]} for i in range(plans)])
    return journey, experiments, calls


def test_parent_retained_status_does_not_hide_exact_final_child():
    journey, experiments, calls = fixture()
    assert _selected_monitoring_experiment(journey, list(reversed(experiments))) == experiments[1]
    assert calls == [("GET", "/api/step-outputs/select-0:v2", "read_monitoring_final_selection")]


@pytest.mark.parametrize("kwargs", [{"artifact": "parent-artifact"}, {"selected": "foreign"}, {"plans": 0}, {"plans": 2}])
def test_ambiguous_or_rebound_final_pointer_is_rejected(kwargs):
    journey, experiments, _ = fixture(**kwargs)
    with pytest.raises(RuntimeJourneyError):
        _selected_monitoring_experiment(journey, experiments)


def test_legacy_single_selection_retains_behavior_without_guessing_another_task():
    journey, experiments, calls = fixture()
    assert _selected_monitoring_experiment(journey, experiments[:1]) == experiments[0]
    assert not calls
    experiments[0]["task_id"] = "other"
    with pytest.raises(RuntimeJourneyError):
        _selected_monitoring_experiment(journey, experiments[:1])
