import json

from marvis.agent.strategy_request_compiler import compile_strategy_request
from marvis.data.backend import DataBackend
from marvis.data.field_directions import ApprovedFieldDirections, approved_field_directions
from marvis.data.registry import DatasetRegistry
from marvis.repositories.datasets import DatasetRepository
from test_field_directions import _setup, _base, _confirm_body


class Recorder:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return {"clarification": "请指定策略版本。"}


def test_confirmed_and_unknown_directions_enter_compiler_only_as_sealed_physical_fields(tmp_path):
    client, settings, task, _ = _setup(tmp_path)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
    llm = Recorder()
    sealed = approved_field_directions(registry, task.id)
    compile_strategy_request("分析 loan_amount 的策略", allowed_columns=["loan_amount"], llm=llm, field_directions=sealed)
    prompt = json.loads(llm.calls[0]["user_prompt"])
    fields = prompt["field_directions"]["fields"]
    assert len(fields) == 1 and fields[0]["column"] == "loan_amount"
    assert fields[0]["direction"] == "unknown" and not fields[0]["actor_confirmed"]
    response = client.post(_base(task) + "/confirm", json=_confirm_body(client, task, {"loan_amount": "lower_is_riskier"}))
    assert response.status_code == 200
    llm = Recorder()
    sealed = approved_field_directions(registry, task.id)
    compile_strategy_request("分析 loan_amount 的策略", allowed_columns=["loan_amount"], llm=llm, field_directions=sealed)
    wire = llm.calls[0]["user_prompt"]
    fields = json.loads(wire)["field_directions"]["fields"]
    assert fields[0]["direction"] == "lower_is_riskier" and fields[0]["actor_confirmed"]
    assert fields[0]["confirmation_hash"]
    assert "maker-1" not in wire and "risk decreases automatically" not in wire
    for forged in (sealed.payload(), ApprovedFieldDirections(sealed.binding_json, object())):
        rejected = Recorder()
        result = compile_strategy_request("分析 loan_amount 的策略", allowed_columns=["loan_amount"], llm=rejected, field_directions=forged)
        assert result.clarification_code == "outbound_policy"
        assert rejected.calls == []
