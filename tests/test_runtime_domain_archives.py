"""Actual HTTP/Agent/worker originals; independent normal-domain revalidation."""
import hashlib
import json
import shutil

import pytest

from marvis.orchestrator.eval.runtime_archive_modeling import FrozenFitRepresentation, FrozenModelingBinding, revalidate_modeling_archive
from marvis.orchestrator.eval.runtime_archive_monitoring import (
    DEFAULT_DECLARED_THRESHOLDS, FrozenMonitoringBinding, revalidate_monitoring_archive,
)
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_archive_strategy import FrozenStrategyBinding, revalidate_strategy_archive
from marvis.orchestrator.evidence import payload_hash
from tests.test_runtime_agent_benchmark import fixture_model


def _protocol(request, answer, payload):
    from tests.test_runtime_workflow_families import _business_protocol, _gate_route_request
    from tests.test_runtime_normal_strategy import _sample_request
    if "instruction_origin" in request and "allowed_columns" in request:
        if "固化 V2 策略样本设计" in request["instruction"]:
            return _sample_request()
        return {"operation": "develop", "strategy_type": "approval", "objective": "max_approval",
                "max_bad_rate": .25, "min_approval_rate": .4}
    if "allowed_intents" in request and ("固化 V2 策略样本设计" in request["instruction"] or "做完整审批策略开发" in request["instruction"]):
        answer = _business_protocol(request, answer, payload)
        answer["intent"] = "strategy_workflow"
        return answer
    if _gate_route_request(payload):
        purpose = (request.get("gate_context") or {}).get("purpose")
        if purpose in {"modeling_setup", "create_workflow_plan"}:
            return {"action": "adjust", "params": {"recipes": ["lr"], "n_trials": 1, "target_type": "binary"},
                "constraint": "", "reason": "合成用户规格", "confidence": "high", "explicit_authorization": False}
        return {"action": "confirm", "params": {}, "constraint": "", "reason": "用户明确当前授权",
            "confidence": "high", "explicit_authorization": True}
    if "review_instruction" in request and "first_pass" in request:
        return request["first_pass"]
    return _business_protocol(request, answer, payload)


def _binding(archive, case, template):
    import sqlite3
    with sqlite3.connect(archive / "workspace/marvis.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        tables = {name: [dict(row) for row in connection.execute("SELECT * FROM " + name)] for name in (
            "plans", "plan_steps", "plan_step_output_versions", "plan_step_runs", "experiments", "datasets", "strategies")}
    common = json.loads((archive / "bindings.json").read_text())
    plan = next(row for row in tables["plans"] if row["template_id"] == template)
    values = {key: common[key] for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")}
    values.update(case=case, task_id=plan["task_id"], plan_id=plan["id"], step_ids={}, output_sha256={})
    outputs = {}
    for step in (row for row in tables["plan_steps"] if row["plan_id"] == plan["id"]):
        assert step["status"] == "done"
        version = next(row for row in tables["plan_step_output_versions"] if row["step_id"] == step["id"])
        output = json.loads(version["output_json"])
        key = step["tool_plugin"] + "." + step["tool_name"]
        try:
            sha = digest(output)
        except ValueError:
            sha = payload_hash(output).removeprefix("sha256:")
        values["step_ids"][key], values["output_sha256"][key], outputs[key] = step["id"], sha, output
    return values, outputs, tables


def _model_binding(archive, case):
    values, outputs, tables = _binding(archive, case, "modeling")
    selected = outputs["modeling.select_experiment"]
    identity = selected["selected_experiment_id"]
    selection = next(action for action in case["actions"] if action["kind"] == "select_recommended_experiment")
    development_refit = selection["refit_on_train_plus_test"]
    assert selected["refit"]["applied"] is development_refit
    experiment = next(row for row in tables["experiments"] if row["id"] == identity)
    dataset_id = json.loads(experiment["config_json"])["dataset_id"]
    dataset = next(row for row in tables["datasets"] if row["id"] == dataset_id)
    values.update(dataset_id=dataset_id, dataset_sha256=dataset["content_hash"], selected_experiment_id=identity,
        final_artifact_id=selected["artifact_id"], fit_representations={row["id"]: {
            "representation": "source_native" if development_refit and row["id"] == identity else "compact_float32",
            "noncompact_features": []} for row in tables["experiments"]})
    return FrozenModelingBinding.model_validate(values)


def _derive_development_refit_suite(paths, root):
    """Declare a new public input/answer pair; retain its no-refit source bytes."""
    from marvis.orchestrator.eval.runtime_cases import _write_suite_files

    source_bytes = {key: paths[key].read_bytes() for key in ("cases", "expected")}
    cases = json.loads(source_bytes["cases"])["cases"]
    expected = json.loads(source_bytes["expected"])["cases"]
    assert len(cases) == 1
    case = cases[0]
    source_id = case["id"]
    selection = next(action for action in case["actions"] if action["kind"] == "select_recommended_experiment")
    assert selection["refit_on_train_plus_test"] is False
    selection["refit_on_train_plus_test"] = True
    selection["content"] = "我已审阅当前候选及限制，明确采用推荐实验，并授权仅用原 train+test 的 480 行生成独立重新拟合子模型；保留原候选的独立 test 指标为选择证据，合成 OOT 不参与拟合或选择，仅用于本次本地流程验证。"
    case["id"] = source_id + "_development_refit"
    case["revision"] += "-explicit-refit"
    source_contract = "fold-frozen training retains the authenticated selected candidate without legacy refit"
    assert source_contract in case["business_constraints_source"]
    case["business_constraints_source"] = case["business_constraints_source"].replace(
        source_contract,
        "human explicitly authorizes a distinct 480-row train+test development-refit child; "
        "the original held-out candidate and test metrics remain bound selection evidence, "
        "and synthetic OOT is excluded from fitting and selection",
    )
    answer = expected.pop(source_id)
    refit = [row for row in answer["assertions"] if row.get("tool") == "modeling.select_experiment"
             and row.get("path") == ["refit", "applied"]]
    assert len(refit) == 1 and refit[0]["value"] is False
    refit[0]["value"] = True
    expected[case["id"]] = answer
    root.mkdir()
    private = root / "private"
    private.mkdir(mode=0o700)
    data = root / "data"
    shutil.copytree(paths["dataset_root"], data)
    derived = _write_suite_files(root, private, data, cases, expected)
    assert all(paths[key].read_bytes() == source_bytes[key] for key in source_bytes)
    assert all(digest(derived[key].read_bytes()) != digest(source_bytes[key]) for key in source_bytes)
    return derived


def _actual_archive(tmp_path_factory, family, *, development_refit=False):
    suffix = "-development-refit" if development_refit else ""
    parent = tmp_path_factory.mktemp("independent-archive-" + family + suffix)
    paths = write_synthetic_suite(parent / ("source-suite" if development_refit else "suite"),
        **{"normal_" + family + "_only": True})
    source_paths = paths
    source_bytes = {key: paths[key].read_bytes() for key in ("cases", "expected")}
    if development_refit:
        paths = _derive_development_refit_suite(paths, parent / "suite")
    declared_bytes = {key: paths[key].read_bytes() for key in ("cases", "expected")}
    with fixture_model(answer_factory=_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=parent / "public",
            evidence_custody_dir=parent / "custody", model=model, model_source="fixture_model")
    assert all(source_paths[key].read_bytes() == value for key, value in source_bytes.items())
    assert all(paths[key].read_bytes() == value for key, value in declared_bytes.items())
    observed = report["cases"][0]
    archive = parent / "custody" / report["run_id"] / observed["case_id"]
    case = json.loads((archive / "case.json").read_text())
    model_binding = _model_binding(archive, case) if family != "strategy" else None
    binding = model_binding
    if family == "monitoring":
        values, outputs, tables = _binding(archive, case, "monitoring_run")
        score = outputs["modeling.score_dataset"]
        inputs = json.loads(next(row for row in tables["plan_step_runs"] if row["tool_ref"] == "modeling.score_dataset")["input_json"])
        source = next(row for row in tables["datasets"] if row["id"] == inputs["dataset_id"])
        child = next(row for row in tables["datasets"] if row["id"] == score["result_dataset_id"])
        values.update(modeling=model_binding.model_dump(), dataset_id=source["id"], dataset_sha256=source["content_hash"],
            scored_dataset_id=child["id"], scored_dataset_sha256=child["content_hash"],
            experiment_id=model_binding.selected_experiment_id, artifact_id=model_binding.final_artifact_id,
            monitoring_binding_sha256=digest(inputs["monitoring_binding"]), thresholds=DEFAULT_DECLARED_THRESHOLDS)
        binding = FrozenMonitoringBinding.model_validate(values)
    elif family == "strategy":
        values, outputs, tables = _binding(archive, case, "strategy_development")
        strategy_id = outputs["strategy.backtest_strategy"]["strategy_id"]
        strategy = next(row for row in tables["strategies"] if row["id"] == strategy_id)
        inputs = json.loads(next(row for row in tables["plan_step_runs"] if row["tool_ref"] == "strategy.backtest_strategy")["input_json"])
        source = next(row for row in tables["datasets"] if row["id"] == inputs["dataset_id"])
        values.update(dataset_id=source["id"], dataset_sha256=source["content_hash"], strategy_id=strategy_id,
                      strategy_version=strategy["version"], strategy_spec_sha256=strategy["dsl_content_hash"])
        prerequisite, _, _ = _binding(archive, case, "strategy_sample_design_v2_native")
        values["sample_design"] = prerequisite
        binding = FrozenStrategyBinding.model_validate(values)
    manifest_sha = hashlib.sha256((archive / "manifest.json").read_bytes()).hexdigest()
    assert report["all_passed"] and observed["score"]["passed"], observed["score"]
    assert all(row["passed"] for row in observed["score"]["assertions"])
    if family != "strategy":
        selection = next(action for action in case["actions"] if action["kind"] == "select_recommended_experiment")
        assert selection["refit_on_train_plus_test"] is development_refit
        _, outputs, tables = _binding(archive, case, "modeling")
        native = next(row for row in tables["plan_step_runs"] if row["tool_ref"] == "modeling.select_experiment")
        assert json.loads(native["input_json"])["refit_on_train_plus_test"] is development_refit
        assert outputs["modeling.select_experiment"]["refit"]["applied"] is development_refit
    return archive, manifest_sha, binding


@pytest.fixture(scope="module", params=["modeling", "monitoring", "strategy"])
def actual_archive(tmp_path_factory, request):
    return _actual_archive(tmp_path_factory, request.param)


@pytest.fixture(scope="module", params=["modeling", "monitoring"])
def actual_refit_archive(tmp_path_factory, request):
    return _actual_archive(tmp_path_factory, request.param, development_refit=True)


def _revalidate(archive, sha, binding):
    reader = revalidate_monitoring_archive if isinstance(binding, FrozenMonitoringBinding) else revalidate_strategy_archive if isinstance(binding, FrozenStrategyBinding) else revalidate_modeling_archive
    return reader(archive, expected_manifest_sha256=sha, frozen_binding=binding)


def _forbid_producer_functions(monkeypatch):
    import marvis.packs.modeling.recipes.common as common
    import marvis.packs.modeling.scoring as scoring
    def forbidden(*args, **kwargs):
        pytest.fail("producer math/scoring must not establish independent correctness")
    for name in ("compute_model_metrics", "compute_regression_metrics", "compute_multiclass_model_metrics"):
        monkeypatch.setattr(common, name, forbidden)
    monkeypatch.setattr(scoring, "_ModelArtifactScorer", forbidden)


def test_actual_archive_recomputes_fit_baseline_and_drift_without_producer_functions(actual_archive, monkeypatch):
    _forbid_producer_functions(monkeypatch)
    result = _revalidate(*actual_archive)
    assert result["supported_checks_verified"], result["checks"]
    assert result["archive_unchanged_during_revalidation"]
    if isinstance(actual_archive[2], FrozenMonitoringBinding):
        assert result["values"]["overall_level"] == "red"
        assert result["values"]["label_maturity_assurance"] == "unknown"
        scope = result["checks"]["model_fit_and_baseline"]["fit_scope"]
        assert scope["baseline_verification"] == "independently_recomputed"
        assert scope["fit_scope"] == "outer_train" and scope["fit_rows"] == 360
        assert scope["test_evaluation"] == "independent_heldout"
    elif isinstance(actual_archive[2], FrozenModelingBinding):
        assert result["values"]["scope"]["fit_scope"] == "outer_train"
        assert result["values"]["scope"]["fit_rows"] == 360
        assert result["values"]["scope"]["test_evaluation"] == "independent_heldout"
        assert result["values"]["scope"]["parent_experiment_id"] is None
        assert result["values"]["metrics"]["test_auc"] is not None
        assert result["values"]["selection_evidence"]["metrics"]["test_auc"] is not None
    else:
        assert result["values"]["population_count"] == 60
        assert result["values"]["metrics"]["approve_count"] == 48


def test_explicit_refit_archive_recomputes_development_child_and_heldout_parent(actual_refit_archive, monkeypatch):
    _forbid_producer_functions(monkeypatch)
    archive, sha, binding = actual_refit_archive
    model = binding.modeling if isinstance(binding, FrozenMonitoringBinding) else binding
    result = revalidate_modeling_archive(archive, expected_manifest_sha256=sha, frozen_binding=model)
    assert result["supported_checks_verified"], result["checks"]
    assert result["archive_unchanged_during_revalidation"]
    values = result["values"]
    assert values["scope"]["fit_scope"] == "full_development"
    assert values["scope"]["fit_rows"] == 480
    assert values["scope"]["test_evaluation"] == "unavailable_consumed_in_fit"
    assert values["scope"]["baseline_verification"] == "independently_recomputed"
    assert values["metrics"]["test_auc"] is None
    assert values["selection_evidence"]["metrics"]["test_auc"] is not None
    assert values["scope"]["parent_experiment_id"] == values["selection_evidence"]["experiment_id"]
    assert values["scope"]["parent_experiment_id"] != model.selected_experiment_id
    if isinstance(binding, FrozenMonitoringBinding):
        monitored = _revalidate(*actual_refit_archive)
        assert monitored["supported_checks_verified"], monitored["checks"]
        assert monitored["archive_unchanged_during_revalidation"]
        assert monitored["values"]["overall_level"] == "red"
        assert monitored["values"]["label_maturity_assurance"] == "unknown"
        assert monitored["checks"]["model_fit_and_baseline"]["fit_scope"] == values["scope"]


def _reject_wrong_output_and_fit_representation(archive, sha, binding):
    wrong = binding.model_copy(update={"output_sha256": {key: "0" * 64 for key in binding.output_sha256}})
    result = _revalidate(archive, sha, wrong)
    assert not result["supported_checks_verified"] and result["checks"]["native_steps"]["status"] == "mismatch"
    if isinstance(binding, FrozenStrategyBinding):
        wrong = binding.model_copy(update={"strategy_spec_sha256": "0" * 64})
        assert not _revalidate(archive, sha, wrong)["supported_checks_verified"]
        return
    model = binding.modeling if isinstance(binding, FrozenMonitoringBinding) else binding
    profiles = dict(model.fit_representations)
    profile = profiles[model.selected_experiment_id]
    opposite = "compact_float32" if profile.representation == "source_native" else "source_native"
    profiles[model.selected_experiment_id] = FrozenFitRepresentation(representation=opposite, noncompact_features=[])
    wrong_model = model.model_copy(update={"fit_representations": profiles})
    wrong = binding.model_copy(update={"modeling": wrong_model}) if isinstance(binding, FrozenMonitoringBinding) else wrong_model
    result = _revalidate(archive, sha, wrong)
    assert not result["supported_checks_verified"]
    profiles.pop(model.selected_experiment_id)
    wrong_model = model.model_copy(update={"fit_representations": profiles})
    wrong = binding.model_copy(update={"modeling": wrong_model}) if isinstance(binding, FrozenMonitoringBinding) else wrong_model
    with pytest.raises(ValueError, match="final fit numerical representation must be independently frozen"):
        _revalidate(archive, sha, wrong)


def test_actual_archive_rejects_frozen_wrong_output_and_unbound_fit_representation(actual_archive):
    _reject_wrong_output_and_fit_representation(*actual_archive)


def test_explicit_refit_archive_rejects_wrong_output_and_fit_representation(actual_refit_archive):
    _reject_wrong_output_and_fit_representation(*actual_refit_archive)


def test_monitoring_cannot_omit_fit_source_or_select_another_model(actual_archive):
    if not isinstance(actual_archive[2], FrozenMonitoringBinding):
        return
    binding = actual_archive[2]
    payload = binding.model_dump()
    payload.pop("modeling")
    with pytest.raises(ValueError):
        FrozenMonitoringBinding.model_validate(payload)
    payload = binding.model_dump()
    payload["modeling"]["final_artifact_id"] = "another-native-model"
    with pytest.raises(ValueError):
        FrozenMonitoringBinding.model_validate(payload)
