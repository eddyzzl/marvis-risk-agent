"""Independent native DSL, typed backtest and retained strategy identity replay."""
import hashlib
import json
from pathlib import Path
import tempfile

from pydantic import Field, model_validator

from .runtime_archive_domain import FrozenDomainBinding, DomainArchive, HASH, ID, numbers, one_step
from .runtime_archive_datasets import _dataset, _source_input
from .runtime_archive_reader import _Unsupported
from .runtime_strategy_reference import decisions_reference, typed_impact_reference, native_economics_bindings
from .runtime_strategy_sample_reference import authenticated_membership


class FrozenStrategyBinding(FrozenDomainBinding):
    dataset_id: str = Field(pattern=ID)
    dataset_sha256: str = Field(pattern=HASH)
    strategy_id: str = Field(pattern=ID)
    strategy_version: int = Field(gt=0)
    strategy_spec_sha256: str = Field(pattern=HASH)
    sample_design: FrozenDomainBinding | None = None

    @model_validator(mode="after")
    def strategy_contract(self):
        if self.case.family != "strategy" or self.case.task.task_type != "strategy" or not self.case.materials:
            raise ValueError("declared native strategy case required")
        if self.sample_design is not None:
            for key in ("case", "run_id", "task_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"):
                if getattr(self, key) != getattr(self.sample_design, key):
                    raise ValueError("strategy sample belongs to another frozen case/run/task/source")
            if self.sample_design.plan_id == self.plan_id or set(self.sample_design.step_ids.values()) & set(self.step_ids.values()):
                raise ValueError("strategy sample requires its separate original plan and steps")
        return self


def strategy_spec(session, strategy_id, *, version=None, content_hash=None):
    row = session.conn.execute("SELECT * FROM strategies WHERE id=?", (strategy_id,)).fetchone()
    if row is None or row["task_id"] != session.binding.task_id or not row["dsl_json"]:
        raise ValueError("canonical strategy task owner mismatch")
    spec = json.loads(row["dsl_json"])
    sha = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if row["dsl_content_hash"] != sha or content_hash is not None and sha != content_hash or version is not None and row["version"] != version:
        raise ValueError("frozen strategy version/effect changed")
    if spec["strategy_type"] != row["strategy_type"] or spec["schema_version"] != row["dsl_schema_version"]:
        raise ValueError("canonical strategy type/schema differs")
    return dict(row), spec


def _bad_target(inputs, output, request):
    """Replay the declared label meaning, never the producer's chosen meaning."""
    bad = 1 if request is None else request["target_bad_value"]
    if type(bad) is not int or bad not in {0, 1}:
        raise _Unsupported("binary strategy target declaration required")
    if request is not None and inputs["target_col"] != request["target_col"]:
        raise ValueError("native backtest target differs from frozen sample request")
    declared = output.get("normalized_input", {}).get("target_encoding", {}).get("bad")
    if type(declared) is not int or declared != bad:
        raise ValueError("native backtest bad-label direction differs from frozen source contract")
    return bad


def revalidate_strategy_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenStrategyBinding):
    if not isinstance(frozen_binding, FrozenStrategyBinding):
        raise ValueError("explicit FrozenStrategyBinding required")
    binding = FrozenStrategyBinding.model_validate(frozen_binding.model_dump())
    with DomainArchive(archive, expected_manifest_sha256, binding, family="strategy",
                       checks=("registered_dataset", "source_material", "strategy_identity", "independent_dsl_impact", "native_backtest_record", "adoption_scope")) as session:
        session.open()
        session.current = "registered_dataset"
        source, frame = _dataset(session.conn, session.archive, binding.task_id, binding.dataset_id, binding.dataset_sha256)
        session.mark("registered_dataset", rows=len(frame))
        session.current = "source_material"
        scratch = Path(session.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        _source_input(session.archive, session.manifest, binding, source, frame, scratch)
        session.mark("source_material")
        session.current = "strategy_identity"
        row, spec = strategy_spec(session, binding.strategy_id, version=binding.strategy_version, content_hash=binding.strategy_spec_sha256)
        session.mark("strategy_identity", version=binding.strategy_version)
        session.current = "independent_dsl_impact"
        backtest = one_step(session.steps, "strategy.backtest_strategy")
        inputs, output = backtest["input"], backtest["output"]
        if inputs["strategy_id"] != binding.strategy_id or inputs["dataset_id"] != binding.dataset_id:
            raise ValueError("native backtest source or policy changed")
        baseline = None
        if inputs.get("baseline_strategy_id"):
            _, baseline = strategy_spec(session, inputs["baseline_strategy_id"])
        if inputs.get("profit_params") or inputs.get("ead_col") or inputs.get("pd_col"):
            raise _Unsupported("approval profit requires its independent declared economic profile")
        if inputs.get("runtime_requirements"):
            raise _Unsupported("model-dependent strategy fields require independently replayed model dependencies")
        masks = request = None
        if inputs.get("sample_design_ref"):
            frame, masks, request = authenticated_membership(session, inputs["sample_design_ref"], frame)
        bad = _bad_target(inputs, output, request)
        values = typed_impact_reference(frame.to_dict("records"), spec, inputs["target_col"], baseline=baseline,
                                       economics=native_economics_bindings(inputs.get("economics_inputs"), spec["strategy_type"], frame.columns), bad_value=bad)
        numbers(output, values)
        decisions = decisions_reference(frame.to_dict("records"), spec)
        session.mark("independent_dsl_impact", rows=len(decisions), profile=spec["strategy_type"])
        session.current = "native_backtest_record"
        record = session.conn.execute("SELECT * FROM backtests WHERE id=? AND strategy_id=? AND dataset_id=?",
            (output.get("backtest_id"), binding.strategy_id, binding.dataset_id)).fetchone()
        if record is None or output.get("source_dataset_content_hash") != binding.dataset_sha256:
            raise ValueError("native typed backtest record does not match producing output")
        retained = json.loads(record["result_json"])
        if any(key not in output or output[key] != value for key, value in retained.items()):
            raise ValueError("retained native backtest envelope differs from producing output")
        numbers(retained, values)
        session.mark("native_backtest_record", scope="task_policy_dataset_and_independent_values")
        session.current = "adoption_scope"
        adoption = [s for s in session.steps.values() if s["tool"] == "strategy.adopt_strategy"]
        if adoption:
            if len(adoption) != 1 or adoption[0]["input"]["strategy_id"] != binding.strategy_id or adoption[0]["output"]["strategy_id"] != binding.strategy_id:
                raise ValueError("adoption refers to another native strategy")
            if row["status"] != "adopted":
                raise ValueError("native local adoption state missing")
        session.mark("adoption_scope", scope="local_state_only_not_deployment_or_business_approval")
        public_values = {**values, "economics": {k: v for k, v in values["economics"].items() if k != "by_row"}}
        session.result["values"] = {**public_values, "strategy_id": binding.strategy_id, "strategy_type": spec["strategy_type"],
            "decision_rows": len(decisions), "business_acceptance": "not_established"}
        session.result["unsupported"]["sample_design_and_stability_time_truth"] = "requires_authenticated_exact_member_and_source_time_contract_not_inferred_from_backtest"
    return session.result
