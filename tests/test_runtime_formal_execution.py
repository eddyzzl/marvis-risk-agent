"""One actual local HTTP/tool candidate with formal admission, never paid."""
from pathlib import Path
from types import SimpleNamespace

from marvis.orchestrator.eval.hidden_formal_runner import _execute_attempt
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, RuntimeSuite, digest
from marvis.orchestrator.eval.runtime_formal_budget import ProviderLimits
from marvis.orchestrator.eval.runtime_runner import _source_identity
from marvis.orchestrator.eval.runtime_scoring import ExpectedCase, ExpectedSuite
from test_runtime_agent_benchmark import base_case, fixture_model, materials
from test_runtime_aggregate_budget import price


def test_actual_local_candidate_uses_formal_gateway_and_retains_custody(tmp_path, monkeypatch):
    material = materials(tmp_path / "data")
    raw = base_case("formal_local", "data_join", [material["sample.parquet"], material["features.parquet"]])
    raw["actions"] = [{"kind": "approve_step", "tool": "data_ops.execute_join", "content": "批准本地合成数据拼接。"}]
    raw["budget"] = {"max_cost": 25., "currency": "CNY", "max_total_tokens": 2000000}
    case = RuntimeCase.model_validate(raw)
    expected = ExpectedCase(result="done", assertions=[{"kind": "dataset_rows", "tool": "data_ops.execute_join", "value": 80}])
    prices = tmp_path / "prices.json"
    prices.write_bytes(price())
    load_spec = {"profile_load": {"concurrent_users": 2}, "load_policy": {"concurrent_workers": 2, "warmup": None}}
    load = SimpleNamespace(load_sha256=digest(load_spec), model_dump=lambda: {"load_sha256": digest(load_spec)})
    for name in ("inputs", "runs", "timing"):
        (tmp_path / name).mkdir(mode=0o700)
    with fixture_model() as (model, calls):
        model = model.model_copy(update={"api_key_env": "LOCAL_FORMAL_TEST_KEY"})
        limits = ProviderLimits(protocol="single_text_utf8_bound_v1", context_window=model.context_window,
            max_output_tokens=model.max_output_tokens, framing_token_allowance=32, price_book_sha256=digest(prices.read_bytes()),
            retain_full_reservations=True, approval={"path": "local-fixture-only.json", "sha256": "a" * 64})
        prepared = SimpleNamespace(suite=RuntimeSuite(cases=[case]), expected=ExpectedSuite(cases={case.id: expected}),
            model=model, price_book_path=prices, provider_limits=limits, source_identity=_source_identity(),
            environment={"load": load_spec}, benchmark=SimpleNamespace(environment=load, model=SimpleNamespace(model_dump=lambda: {"source": "local-fixture-only"})),
            dataset_root=tmp_path / "data")
        # Change the authority file immediately after its validated read. The
        # forwarded accounting must keep the first bytes, never read low prices
        # a second time after checking the approved expensive price digest.
        original_read = Path.read_bytes
        price_reads = []
        def changing_read(path):
            data = original_read(path)
            if path == prices:
                price_reads.append(data)
                prices.write_bytes(price(prompt_per_million=0., completion_per_million=0.))
            return data
        monkeypatch.setattr(Path, "read_bytes", changing_read)
        row = _execute_attempt(prepared, SimpleNamespace(id="local-r0", case_id=case.id, load_mode="cold_start"),
                               tmp_path / "inputs", tmp_path / "runs", tmp_path / "custody",
                               {"LOCAL_FORMAL_TEST_KEY": "local-fixture-placeholder"})
    assert len(price_reads) == 1
    assert calls and row["status"] == "completed", row
    assert row["local_predicates_passed"] and row["evidence_custody"]["status"] == "retained"
    assert Path(row["run_dir"]).is_dir() and row["telemetry"]["sha256"]
    assert row["independent_acceptance"] == "not_established"
