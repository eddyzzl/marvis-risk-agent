"""Synthetic offline selection-reader contracts; no producer metrics or archives."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace

import pytest

import marvis.orchestrator.eval.runtime_archive_modeling as reader
from marvis.orchestrator.eval.runtime_archive_modeling import (
    _selection_metrics,
    _selection_projection_allowed,
)


# Independent static contract, copied from the native field declarations rather
# than inferred from the payload under inspection or a producer computation.
FULL_KEYS = (
    "train_ks", "test_ks", "oot_ks", "train_auc", "test_auc", "oot_auc",
    "psi_test_vs_train", "psi_oot_vs_train", "overfit_train_test_gap",
    "overfit_train_oot_gap", "overfit_flag", "weighted_train_ks",
    "weighted_test_ks", "weighted_oot_ks", "weighted_train_auc",
    "weighted_test_auc", "weighted_oot_auc", "weighted_psi_test_vs_train",
    "weighted_psi_oot_vs_train", "train_rmse", "test_rmse", "oot_rmse",
    "train_mae", "test_mae", "oot_mae", "train_r2", "test_r2", "oot_r2",
    "train_macro_auc", "test_macro_auc", "oot_macro_auc", "train_logloss",
    "test_logloss", "oot_logloss", "train_accuracy", "test_accuracy",
    "oot_accuracy", "test_ks_ci_low", "test_ks_ci_high", "test_ks_ci_std",
    "oot_ks_ci_low", "oot_ks_ci_high", "oot_ks_ci_std", "ks_ci_n_boot",
    "test_lift_head_5", "test_lift_tail_5", "test_lift_head_10",
    "test_lift_tail_10", "oot_lift_head_5", "oot_lift_tail_5",
    "oot_lift_head_10", "oot_lift_tail_10",
)
COMPARISON_KEYS = (
    "train_ks", "test_ks", "oot_ks", "train_auc", "test_auc", "oot_auc",
    "test_ks_ci_low", "test_ks_ci_high", "test_ks_ci_std", "oot_ks_ci_low",
    "oot_ks_ci_high", "oot_ks_ci_std", "test_lift_head_5",
    "test_lift_head_10", "test_lift_tail_5", "test_lift_tail_10",
    "oot_lift_head_5", "oot_lift_head_10", "oot_lift_tail_5",
    "oot_lift_tail_10", "psi_test_vs_train", "psi_oot_vs_train", "overfit_flag",
)


def _binary_values():
    # Teaching numbers supplied by this test, never read from actual output.
    return {
        "train_ks": .43, "test_ks": .40, "oot_ks": .38,
        "train_auc": .76, "test_auc": .73, "oot_auc": .71,
        "psi_test_vs_train": .02, "psi_oot_vs_train": .03,
        "overfit_train_test_gap": .03, "overfit_train_oot_gap": .05,
        "overfit_flag": False, "test_ks_ci_low": .32,
        "test_ks_ci_high": .48, "test_ks_ci_std": .04,
        "oot_ks_ci_low": .30, "oot_ks_ci_high": .46,
        "oot_ks_ci_std": .04, "ks_ci_n_boot": 200,
        "test_lift_head_5": 2.2, "test_lift_head_10": 1.9,
        "test_lift_tail_5": .2, "test_lift_tail_10": .3,
        "oot_lift_head_5": 2.1, "oot_lift_head_10": 1.8,
        "oot_lift_tail_5": .25, "oot_lift_tail_10": .35,
    }


def _unknown_oot_values():
    values = _binary_values()
    for key in (
        "oot_ks", "oot_auc", "oot_ks_ci_low", "oot_ks_ci_high",
        "oot_ks_ci_std", "oot_lift_head_5", "oot_lift_head_10",
        "oot_lift_tail_5", "oot_lift_tail_10", "overfit_train_oot_gap",
    ):
        values[key] = None
    return values


def _regression_values():
    return {
        "train_rmse": 1.1, "test_rmse": 1.2, "oot_rmse": None,
        "train_mae": .8, "test_mae": .9, "oot_mae": None,
        "train_r2": .72, "test_r2": .69, "oot_r2": None,
        "overfit_train_test_gap": .03, "overfit_train_oot_gap": None,
        "overfit_flag": False,
    }


def _multiclass_values():
    return {
        "train_macro_auc": .80, "test_macro_auc": .76, "oot_macro_auc": None,
        "train_logloss": .44, "test_logloss": .50, "oot_logloss": None,
        "train_accuracy": .79, "test_accuracy": .75, "oot_accuracy": None,
        "overfit_train_test_gap": .04, "overfit_train_oot_gap": None,
        "overfit_flag": False,
    }


def _full(values):
    return {key: values.get(key) for key in FULL_KEYS}


def _projection(values):
    return {key: values[key] for key in COMPARISON_KEYS
            if key in values and values[key] is not None}


def _scope(profile="lr"):
    return {
        "profile": profile, "fit_scope": "outer_train",
        "test_evaluation": "independent_heldout", "parent_experiment_id": None,
        "metric_numeric_tolerance": {"rel_tol": 1e-9, "abs_tol": 1e-11},
    }


def _retained_select(values):
    return {
        "tool": "modeling.select_experiment",
        "input": {
            "experiment_ids": ["candidate"], "selected_experiment_id": "candidate",
            "refit_on_train_plus_test": False,
        },
        "output": {
            "selected_experiment_id": "candidate", "artifact_id": "candidate-artifact",
            "metrics": _projection(values),
            "refit": {"requested": False, "applied": False},
        },
    }


@pytest.mark.parametrize("factory", [
    _binary_values, _unknown_oot_values, _regression_values, _multiclass_values,
])
@pytest.mark.parametrize("allow_projection", [False, True])
def test_complete_native_schema_includes_explicit_other_target_nulls(factory, allow_projection):
    values = factory()
    actual = _full(values)
    assert len(actual) == 52
    assert _selection_metrics(actual, values, allow_projection=allow_projection) is None


@pytest.mark.parametrize("factory", [_binary_values, _unknown_oot_values])
def test_fixed_comparison_projection_omits_only_independently_undefined_values(factory):
    values = factory()
    actual = _projection(values)
    assert set(actual) == {key for key in COMPARISON_KEYS if values.get(key) is not None}
    assert _selection_metrics(actual, values, allow_projection=True) is None
    with pytest.raises(ValueError):
        _selection_metrics(actual, values)


@pytest.mark.parametrize("key", COMPARISON_KEYS)
@pytest.mark.parametrize("full", [False, True])
def test_a_declared_metric_cannot_disappear_from_either_accepted_schema(key, full):
    values = _binary_values()
    actual = _full(values) if full else _projection(values)
    del actual[key]
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)


@pytest.mark.parametrize("key", FULL_KEYS)
def test_every_full_schema_value_is_checked_against_independent_expectation(key):
    values = _binary_values()
    actual = _full(values)
    expected = actual[key]
    actual[key] = not expected if isinstance(expected, bool) else 1.0 if expected is None else expected + 1
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)


@pytest.mark.parametrize("key", COMPARISON_KEYS)
def test_every_projected_value_is_checked_against_independent_expectation(key):
    values = _binary_values()
    actual = _projection(values)
    actual[key] = True if key == "overfit_flag" else actual[key] + 1
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)


@pytest.mark.parametrize("replacement", [0, 1, "false", "true", None])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("expected", [False, True])
def test_overfit_flag_requires_an_actual_bool(replacement, full, expected):
    values = _binary_values()
    values["overfit_flag"] = expected
    actual = _full(values) if full else _projection(values)
    actual["overfit_flag"] = replacement
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)


def test_full_schema_allows_null_flag_only_when_independently_undefined():
    values = _binary_values()
    values["overfit_flag"] = None
    assert _selection_metrics(_full(values), values) is None
    actual = _full(values)
    actual["overfit_flag"] = False
    with pytest.raises(ValueError):
        _selection_metrics(actual, values)


@pytest.mark.parametrize("replacement", [True, "0.73", float("nan"), float("inf")])
@pytest.mark.parametrize("full", [False, True])
def test_numeric_values_cannot_be_bool_text_or_nonfinite(replacement, full):
    values = _binary_values()
    actual = _full(values) if full else _projection(values)
    actual["test_auc"] = replacement
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)


@pytest.mark.parametrize("key", ["weighted_train_ks", "train_rmse", "oot_accuracy"])
@pytest.mark.parametrize("allow_projection", [False, True])
def test_full_payload_missing_a_null_field_never_becomes_a_projection(key, allow_projection):
    values = _binary_values()
    actual = _full(values)
    del actual[key]
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=allow_projection)


@pytest.mark.parametrize("full", [False, True])
def test_unexpected_fields_are_rejected_in_both_schemas(full):
    values = _binary_values()
    actual = _full(values) if full else _projection(values)
    actual["invented_metric"] = .73
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)


def test_projection_cannot_reintroduce_independently_null_metrics():
    values = _unknown_oot_values()
    actual = _projection(values)
    actual["oot_auc"] = None
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)


@pytest.mark.parametrize("full", [False, True])
def test_caller_numeric_tolerance_is_preserved(full):
    values = _binary_values()
    actual = _full(values) if full else _projection(values)
    actual["test_auc"] += 5e-8
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True)
    assert _selection_metrics(actual, values, allow_projection=True,
                              rel_tol=1e-6, abs_tol=1e-7) is None
    actual["test_auc"] += .01
    with pytest.raises(ValueError):
        _selection_metrics(actual, values, allow_projection=True,
                           rel_tol=1e-6, abs_tol=1e-7)


@pytest.mark.parametrize("profile", ["lr", "xgb", "lgb"])
def test_retained_binary_candidate_with_real_heldout_metrics_allows_projection(profile):
    values = _binary_values()
    assert _selection_projection_allowed(_retained_select(values), _scope(profile), values=values) is True


@pytest.mark.parametrize("field,replacement", [
    ("profile", "lr_regressor"), ("profile", "xgb_multiclass"),
    ("profile", "lgb_regressor"), ("profile", "unknown"),
    ("fit_scope", "full_development"),
    ("test_evaluation", "unavailable_consumed_in_fit"),
    ("parent_experiment_id", "ancestor"), ("parent_experiment_id", 0),
    ("parent_experiment_id", False),
])
def test_projection_gate_rejects_other_profiles_and_parent_lineage(field, replacement):
    values = _binary_values()
    scope = _scope()
    scope[field] = replacement
    assert _selection_projection_allowed(_retained_select(values), scope, values=values) is False


@pytest.mark.parametrize("branch,field,replacement", [
    ("refit", "applied", True), ("refit", "applied", 0),
    ("refit", "applied", None), ("refit", "applied", "false"),
    ("refit", "requested", True), ("refit", "requested", 0),
    ("refit", "requested", None), ("refit", "requested", "false"),
    ("input", "refit_on_train_plus_test", True),
    ("input", "refit_on_train_plus_test", 0),
    ("input", "refit_on_train_plus_test", None),
    ("input", "refit_on_train_plus_test", "false"),
    ("input", "selected_experiment_id", "different"),
    ("input", "experiment_ids", ["different"]),
    ("output", "selected_experiment_id", "different"),
    ("output", "selection_evidence", {}),
    ("output", "selection_evidence", {"experiment_id": "ancestor"}),
])
def test_projection_gate_requires_explicit_false_controls_and_selected_membership(branch, field, replacement):
    values = _binary_values()
    select = _retained_select(values)
    target = select["output"]["refit"] if branch == "refit" else select[branch]
    target[field] = replacement
    assert _selection_projection_allowed(select, _scope(), values=values) is False


@pytest.mark.parametrize("branch,field", [
    ("refit", "applied"), ("refit", "requested"),
    ("input", "refit_on_train_plus_test"),
    ("input", "selected_experiment_id"), ("input", "experiment_ids"),
    ("output", "selected_experiment_id"),
])
def test_projection_gate_does_not_invent_missing_retention_controls(branch, field):
    values = _binary_values()
    select = _retained_select(values)
    target = select["output"]["refit"] if branch == "refit" else select[branch]
    del target[field]
    assert _selection_projection_allowed(select, _scope(), values=values) is False


@pytest.mark.parametrize("key", ["test_auc", "test_ks"])
@pytest.mark.parametrize("replacement", [None, float("nan"), float("inf"), float("-inf"), True, False, "0.73"])
def test_projection_gate_requires_real_finite_recomputed_heldout_auc_and_ks(key, replacement):
    values = _binary_values()
    values[key] = replacement
    assert _selection_projection_allowed(_retained_select(values), _scope(), values=values) is False


@pytest.mark.parametrize("key", ["test_auc", "test_ks"])
def test_projection_gate_rejects_missing_recomputed_heldout_metric(key):
    values = _binary_values()
    select = _retained_select(values)
    del values[key]
    assert _selection_projection_allowed(select, _scope(), values=values) is False


class _SyntheticBinding(SimpleNamespace):
    def model_dump(self):
        return vars(self).copy()

    @classmethod
    def model_validate(cls, value):
        return cls(**value)


def _wire_reader(monkeypatch, select, final_values, final_scope, *, parent_values=None, parent_scope=None):
    """Only test reader wiring; authentication/native math are separate gates."""
    calls = []
    binding = _SyntheticBinding(
        task_id="synthetic-task", dataset_id="synthetic-data",
        dataset_sha256="a" * 64,
        selected_experiment_id=select["output"]["selected_experiment_id"],
        final_artifact_id=select["output"]["artifact_id"],
    )

    class Connection:
        def execute(self, query, params):
            assert query == "SELECT artifact_id FROM experiments WHERE id=?"
            assert params == ("candidate",)
            return SimpleNamespace(fetchone=lambda: {"artifact_id": "candidate-artifact"})

    class Archive:
        def __init__(self, path, sha, supplied_binding, *, family, checks):
            assert supplied_binding.model_dump() == binding.model_dump()
            assert family == "modeling"
            self.binding = supplied_binding
            self.steps = {"synthetic-select": deepcopy(select)}
            self.stack = ExitStack()
            self.conn = Connection()
            self.archive = path
            self.result = {"unsupported": {}}
            self.marks = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.stack.close()
            return False

        def open(self):
            return None

        def mark(self, check, **details):
            self.marks.append(check)

    def model_values(session, experiment_id, artifact_id, frame):
        calls.append((experiment_id, artifact_id))
        if experiment_id == binding.selected_experiment_id:
            assert artifact_id == binding.final_artifact_id
            return deepcopy(final_values), deepcopy(final_scope)
        assert experiment_id == "candidate" and artifact_id == "candidate-artifact"
        assert parent_values is not None and parent_scope is not None
        return deepcopy(parent_values), deepcopy(parent_scope)

    monkeypatch.setattr(reader, "FrozenModelingBinding", _SyntheticBinding)
    monkeypatch.setattr(reader, "DomainArchive", Archive)
    monkeypatch.setattr(reader, "_dataset", lambda *args: ({"id": "synthetic-data"}, [0]))
    monkeypatch.setattr(reader, "source_material_reference", lambda *args: "synthetic-source")
    monkeypatch.setattr(reader, "_model_values", model_values)
    return binding, calls


def _run_reader(tmp_path, binding):
    return reader.revalidate_modeling_archive(
        tmp_path / "synthetic-no-archive", expected_manifest_sha256="b" * 64,
        frozen_binding=binding,
    )


def _refitted_select(parent_values, *, full_parent=False):
    final_values = _binary_values()
    for key in final_values:
        if key.startswith(("test_", "oot_")) or key in {
            "psi_test_vs_train", "psi_oot_vs_train", "overfit_train_test_gap",
            "overfit_train_oot_gap", "overfit_flag", "ks_ci_n_boot",
        }:
            final_values[key] = None
    final_scope = _scope()
    final_scope.update(fit_scope="full_development", test_evaluation="unavailable_consumed_in_fit",
                       parent_experiment_id="candidate")
    select = _retained_select(parent_values)
    select["input"]["refit_on_train_plus_test"] = True
    select["output"].update(
        selected_experiment_id="refitted", artifact_id="refitted-artifact",
        metrics=_full(final_values), refit={"applied": True, "requested": True},
        selection_evidence={"experiment_id": "candidate", "artifact_id": "candidate-artifact",
                            "metrics": _full(parent_values) if full_parent else _projection(parent_values)},
    )
    return select, final_values, final_scope


def test_retained_projection_is_used_by_archive_reader(tmp_path, monkeypatch):
    values = _binary_values()
    select = _retained_select(values)
    binding, calls = _wire_reader(monkeypatch, select, values, _scope())
    result = _run_reader(tmp_path, binding)
    assert result["values"]["metrics"] == values
    assert calls == [("candidate", "candidate-artifact")]


@pytest.mark.parametrize("full_parent", [False, True])
def test_refitted_final_requires_full_metrics_but_independent_parent_accepts_exact_schemas(tmp_path, monkeypatch, full_parent):
    parent_values = _binary_values()
    select, final_values, final_scope = _refitted_select(parent_values, full_parent=full_parent)
    binding, calls = _wire_reader(monkeypatch, select, final_values, final_scope,
                                 parent_values=parent_values, parent_scope=_scope())
    result = _run_reader(tmp_path, binding)
    assert result["values"]["selection_evidence"] == select["output"]["selection_evidence"]
    assert calls == [("refitted", "refitted-artifact"), ("candidate", "candidate-artifact")]


@pytest.mark.parametrize("refit", [False, True])
def test_archive_reader_rejects_incomplete_full_null_schema_without_projection_fallback(tmp_path, monkeypatch, refit):
    values = _binary_values()
    if refit:
        select, values, scope = _refitted_select(values)
    else:
        select, scope = _retained_select(values), _scope()
        select["output"]["metrics"] = _full(values)
    del select["output"]["metrics"]["weighted_train_ks"]
    binding, calls = _wire_reader(monkeypatch, select, values, scope)
    with pytest.raises(ValueError):
        _run_reader(tmp_path, binding)
    assert calls == [(select["output"]["selected_experiment_id"], select["output"]["artifact_id"])]


def test_archive_reader_does_not_allow_refit_final_comparison_projection(tmp_path, monkeypatch):
    select, values, scope = _refitted_select(_binary_values())
    select["output"]["metrics"] = _projection(values)
    binding, calls = _wire_reader(monkeypatch, select, values, scope)
    with pytest.raises(ValueError):
        _run_reader(tmp_path, binding)
    assert calls == [("refitted", "refitted-artifact")]


@pytest.mark.parametrize("replacement", [None, "unrelated-artifact"])
def test_archive_reader_requires_parent_artifact_id_to_match_authenticated_experiment(tmp_path, monkeypatch, replacement):
    select, values, scope = _refitted_select(_binary_values())
    if replacement is None:
        del select["output"]["selection_evidence"]["artifact_id"]
    else:
        select["output"]["selection_evidence"]["artifact_id"] = replacement
    binding, calls = _wire_reader(monkeypatch, select, values, scope)
    with pytest.raises(ValueError, match="artifact reference mismatch"):
        _run_reader(tmp_path, binding)
    assert calls == [("refitted", "refitted-artifact")]


def test_archive_reader_rejects_output_parent_without_independently_verified_parent_lineage(tmp_path, monkeypatch):
    values = _binary_values()
    select = _retained_select(values)
    select["output"]["metrics"] = _full(values)
    select["output"]["selection_evidence"] = {
        "experiment_id": "candidate", "artifact_id": "candidate-artifact",
        "metrics": _projection(values),
    }
    binding, calls = _wire_reader(monkeypatch, select, values, _scope())
    with pytest.raises(ValueError, match="unexpected pre-refit"):
        _run_reader(tmp_path, binding)
    assert calls == [("candidate", "candidate-artifact")]


@pytest.mark.parametrize("field,replacement", [
    ("profile", "lr_regressor"), ("fit_scope", "full_development"),
    ("test_evaluation", "unavailable_consumed_in_fit"),
    ("parent_experiment_id", "ancestor"),
])
def test_archive_reader_parent_projection_requires_independent_binary_outer_train_scope(tmp_path, monkeypatch, field, replacement):
    parent_values = _binary_values()
    select, values, scope = _refitted_select(parent_values)
    parent_scope = _scope()
    parent_scope[field] = replacement
    binding, _ = _wire_reader(monkeypatch, select, values, scope,
                              parent_values=parent_values, parent_scope=parent_scope)
    with pytest.raises(ValueError):
        _run_reader(tmp_path, binding)


@pytest.mark.parametrize("key", ["test_auc", "test_ks"])
@pytest.mark.parametrize("replacement", [None, float("nan"), True])
def test_archive_reader_parent_projection_requires_real_recomputed_heldout_metrics(tmp_path, monkeypatch, key, replacement):
    parent_values = _binary_values()
    parent_values[key] = replacement
    select, values, scope = _refitted_select(parent_values)
    binding, _ = _wire_reader(monkeypatch, select, values, scope,
                              parent_values=parent_values, parent_scope=_scope())
    with pytest.raises(ValueError):
        _run_reader(tmp_path, binding)
