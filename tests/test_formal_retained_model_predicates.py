"""Synthetic predicate wiring; native authentication is verified separately."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.acceptance_trust import TrustInputError
from marvis.orchestrator.eval.formal_domain_adapters import independently_expected
import marvis.orchestrator.eval.formal_native_predicates as predicates
from marvis.orchestrator.eval.runtime_scoring import ExpectedCase


def _world(monkeypatch, *, refit=False, comparison=None):
    original = "parent" if refit else "candidate"
    selected_metrics = {"train_auc": .76, "train_ks": .43,
                        "test_auc": None if refit else .73,
                        "test_ks": None if refit else .40}
    parent_metrics = {"train_auc": .75, "train_ks": .42, "test_auc": .72, "test_ks": .38}
    evidence = {"experiment_id": original,
                "metrics": deepcopy(parent_metrics if refit else selected_metrics)}
    if refit:
        evidence["artifact_id"] = "parent-artifact"
    scope = {"profile": "lr", "fit_scope": "full_development" if refit else "outer_train",
             "test_evaluation": "unavailable_consumed_in_fit" if refit else "independent_heldout",
             "parent_experiment_id": "parent" if refit else None}
    values = {"metrics": selected_metrics, "scope": scope, "selection_evidence": evidence}
    binding = SimpleNamespace(selected_experiment_id="candidate",
                              final_artifact_id="candidate-artifact", task_id="task")
    rows = {
        "candidate": {"id": "candidate", "artifact_id": "candidate-artifact",
                      "recipe_id": "lr", "status": "selected", "created_at": "2026-10-12T02:00:00Z"},
        "parent": {"id": "parent", "artifact_id": "parent-artifact",
                   "recipe_id": "lr", "status": "selected", "created_at": "2026-10-12T01:00:00Z"},
    }
    sections = [{"section": name, "available": False} for name in
                ("sample_analysis", "vintage", "amount_bin", "low_pricing", "product_list")]
    if refit:
        sections.append({"section": "development_refit_scope", "available": True})
    if comparison is None:
        comparison = {"champion": {"experiment_id": "parent", "artifact_id": "parent-artifact",
                                    "label": "previous_selected_experiment", "metrics": deepcopy(parent_metrics)}} if refit else {}
    steps = {
        "select": {"tool": "modeling.select_experiment",
                   "input": {"selected_experiment_id": original, "experiment_ids": [original],
                             "refit_on_train_plus_test": refit},
                   "output": {"selected_experiment_id": "candidate", "artifact_id": "candidate-artifact",
                              "recipe": "lr", "refit": {"applied": refit, "requested": refit},
                              "policy_decision": {"explicit_selection": True}}},
        "reports": {"tool": "modeling.generate_model_reports",
                    "input": {"selected_experiment_id": "candidate"},
                    "output": {"report_path": "/synthetic/report.docx",
                               "reports": [{"experiment_id": "candidate", "report_path": "/synthetic/report.docx"}],
                               "section_status": sections}},
        "post": {"tool": "modeling.post_training_action", "input": {"experiment_id": "candidate"},
                 "output": {"challenger_comparison": comparison}},
    }
    if refit:
        steps["select"]["output"]["selection_evidence"] = evidence
    queries = []

    class Connection:
        def execute(self, query, params):
            assert query == "SELECT * FROM experiments WHERE id=? AND task_id=?"
            assert params[1] == binding.task_id
            queries.append(params[0])
            return SimpleNamespace(fetchone=lambda: rows.get(params[0]))

    captured = []
    monkeypatch.setattr(predicates, "_artifact", lambda session, step, field: captured.append((step["tool"], field)) or field)
    session = SimpleNamespace(steps=steps, conn=Connection())
    return session, binding, values, rows, queries, captured


def test_retained_candidate_with_no_native_comparison_does_not_invent_a_previous_selected_fact(monkeypatch):
    session, binding, values, _, queries, captured = _world(monkeypatch)
    facts, artifacts = predicates._model_facts(session, binding, values)
    assert facts["modeling.select_experiment"]["refit"] == {"applied": False}
    assert facts["modeling.post_training_action"] == {"challenger_comparison": {}}
    assert queries == ["candidate"]
    assert captured == [("modeling.generate_model_reports", "report_path"),
                        ("modeling.post_training_action", "model_card_path"),
                        ("modeling.post_training_action", "approval_package_path")]
    assert artifacts["modeling.post_training_action"] == ["model_card_path", "approval_package_path"]


@pytest.mark.parametrize("profile", ["lr", "xgb", "lgb"])
@pytest.mark.parametrize("status", ["selected", "handed_off", "validated", "delivered", "approved", "champion"])
def test_native_retained_profiles_and_selected_lifecycle_states_remain_supported(monkeypatch, profile, status):
    session, binding, values, rows, _, _ = _world(monkeypatch)
    values["scope"]["profile"] = profile
    rows["candidate"].update(status=status, recipe_id=profile)
    session.steps["select"]["output"]["recipe"] = profile
    facts, _ = predicates._model_facts(session, binding, values)
    assert facts["modeling.post_training_action"]["challenger_comparison"] == {}


@pytest.mark.parametrize("comparison", [
    None, False, 0, [], {"status": "unavailable"}, {"status": "missing"},
    {"champion": {}}, {"champion": {"label": "previous_selected_experiment"}},
    {"champion": {"experiment_id": "parent", "artifact_id": "parent-artifact"}},
    {"unexpected": True},
])
def test_retained_boundary_requires_the_producers_exact_empty_native_comparison(monkeypatch, comparison):
    session, binding, values, _, queries, captured = _world(monkeypatch)
    session.steps["post"]["output"]["challenger_comparison"] = comparison
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)
    assert queries == ["candidate"]
    assert captured == []


def test_missing_comparison_is_not_interpreted_as_a_captured_empty_object(monkeypatch):
    session, binding, values, _, _, _ = _world(monkeypatch)
    del session.steps["post"]["output"]["challenger_comparison"]
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("branch,field", [("refit", "applied"), ("refit", "requested"),
                                         ("input", "refit_on_train_plus_test")])
@pytest.mark.parametrize("replacement", [0, True, None, "false"])
def test_retained_branch_requires_three_strict_false_native_controls(monkeypatch, branch, field, replacement):
    session, binding, values, _, _, _ = _world(monkeypatch)
    target = session.steps["select"]["output"]["refit"] if branch == "refit" else session.steps["select"][branch]
    target[field] = replacement
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("branch,field", [("refit", "applied"), ("refit", "requested"),
                                         ("input", "refit_on_train_plus_test")])
def test_missing_refit_controls_do_not_authorize_retention(monkeypatch, branch, field):
    session, binding, values, _, _, _ = _world(monkeypatch)
    target = session.steps["select"]["output"]["refit"] if branch == "refit" else session.steps["select"][branch]
    del target[field]
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("field,replacement", [
    ("profile", "lr_regressor"), ("profile", "xgb_multiclass"),
    ("profile", "unknown"), ("fit_scope", "full_development"),
    ("test_evaluation", "unavailable_consumed_in_fit"),
])
def test_scope_labels_cannot_authorize_unverified_retained_model_facts(monkeypatch, field, replacement):
    session, binding, values, _, _, _ = _world(monkeypatch)
    values["scope"][field] = replacement
    if field == "fit_scope" and replacement == "full_development":
        session.steps["reports"]["output"]["section_status"].append(
            {"section": "development_refit_scope", "available": True}
        )
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("metric", ["test_auc", "test_ks"])
@pytest.mark.parametrize("replacement", [None, True, False, "0.73", float("nan"), float("inf")])
def test_retention_requires_real_finite_independently_recomputed_heldout_metrics(monkeypatch, metric, replacement):
    session, binding, values, _, _, _ = _world(monkeypatch)
    values["metrics"][metric] = replacement
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("parent", [False, 0, "", "parent", "untrusted"])
def test_present_unverified_parent_cannot_use_the_no_baseline_escape(monkeypatch, parent):
    session, binding, values, _, _, _ = _world(monkeypatch)
    values["scope"]["parent_experiment_id"] = parent
    with pytest.raises(ValueError):
        predicates._model_facts(session, binding, values)


def test_missing_verified_parent_scope_is_not_guessed_to_be_none(monkeypatch):
    session, binding, values, _, _, _ = _world(monkeypatch)
    del values["scope"]["parent_experiment_id"]
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("evidence", [{}, {"experiment_id": "parent"}])
def test_native_selection_with_a_parent_claim_cannot_become_retained_no_baseline(monkeypatch, evidence):
    session, binding, values, _, _, _ = _world(monkeypatch)
    session.steps["select"]["output"]["selection_evidence"] = evidence
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("owner", ["row", "output", "binding"])
def test_final_retained_artifact_must_match_the_authenticated_final_binding(monkeypatch, owner):
    session, binding, values, rows, _, _ = _world(monkeypatch)
    if owner == "row":
        rows["candidate"]["artifact_id"] = "different-artifact"
    elif owner == "output":
        session.steps["select"]["output"]["artifact_id"] = "different-artifact"
    else:
        binding.final_artifact_id = "different-artifact"
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("status", ["trained", "failed", "pending", "cancelled", "handoff", "handed-off"])
def test_a_model_outside_selected_lifecycle_cannot_certify_retained_selection(monkeypatch, status):
    session, binding, values, rows, _, _ = _world(monkeypatch)
    rows["candidate"]["status"] = status
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("guard", ["artifact", "controls", "comparison", "evidence_metrics"])
def test_legal_handed_off_terminal_state_does_not_bypass_other_native_guards(monkeypatch, guard):
    session, binding, values, rows, _, _ = _world(monkeypatch)
    rows["candidate"]["status"] = "handed_off"
    if guard == "artifact":
        rows["candidate"]["artifact_id"] = "different-artifact"
    elif guard == "controls":
        session.steps["select"]["output"]["refit"]["requested"] = True
    elif guard == "comparison":
        session.steps["post"]["output"]["challenger_comparison"] = {"champion": {"experiment_id": "unverified"}}
    else:
        values["selection_evidence"]["metrics"]["test_auc"] = .99
    with pytest.raises(ValueError):
        predicates._model_facts(session, binding, values)


def test_retained_self_selection_evidence_cannot_replace_recomputed_metrics(monkeypatch):
    session, binding, values, _, _, _ = _world(monkeypatch)
    values["selection_evidence"]["metrics"]["test_auc"] = .99
    with pytest.raises(ValueError, match="domain identity mismatch"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("step,branch,field,replacement", [
    ("select", "input", "selected_experiment_id", "different"),
    ("select", "input", "experiment_ids", ["different"]),
    ("select", "output", "selected_experiment_id", "different"),
    ("select", "output", "recipe", "different"),
    ("select", "output", "policy_decision", {"explicit_selection": False}),
    ("select", "output", "policy_decision", {"explicit_selection": 1}),
    ("reports", "input", "selected_experiment_id", "different"),
    ("reports", "output", "reports", []),
    ("reports", "output", "report_path", "/synthetic/different.docx"),
    ("reports", "output", "section_status", []),
    ("post", "input", "experiment_id", "different"),
])
def test_original_selection_report_and_delivery_identity_guards_remain_strict(monkeypatch, step, branch, field, replacement):
    session, binding, values, _, _, _ = _world(monkeypatch)
    session.steps[step][branch][field] = replacement
    with pytest.raises(ValueError):
        predicates._model_facts(session, binding, values)


def test_authenticated_refit_parent_still_produces_previous_selected_fact(monkeypatch):
    session, binding, values, _, queries, captured = _world(monkeypatch, refit=True)
    facts, _ = predicates._model_facts(session, binding, values)
    assert facts["modeling.select_experiment"]["refit"] == {"applied": True}
    assert facts["modeling.post_training_action"]["challenger_comparison"]["champion"] == {"label": "previous_selected_experiment"}
    assert queries == ["candidate", "parent"]
    assert len(captured) == 3


def test_refit_without_authenticated_parent_never_uses_retained_empty_comparison(monkeypatch):
    session, binding, values, _, _, _ = _world(monkeypatch, refit=True)
    values["scope"]["parent_experiment_id"] = None
    session.steps["post"]["output"]["challenger_comparison"] = {}
    with pytest.raises(ValueError, match="empty no-baseline boundary"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("owner,field,replacement", [
    ("parent", "status", "trained"), ("parent", "status", "handed_off"),
    ("parent", "artifact_id", "different"),
    ("parent", "created_at", "2026-10-12T02:00:00Z"),
    ("parent", "created_at", "2026-10-12T03:00:00Z"),
    ("champion", "experiment_id", "different"), ("champion", "artifact_id", "different"),
    ("champion", "metrics", {"test_auc": .99}),
])
def test_original_refit_parent_identity_age_status_and_numeric_guards_remain_strict(monkeypatch, owner, field, replacement):
    session, binding, values, rows, _, _ = _world(monkeypatch, refit=True)
    target = rows["parent"] if owner == "parent" else session.steps["post"]["output"]["challenger_comparison"]["champion"]
    target[field] = replacement
    with pytest.raises(ValueError):
        predicates._model_facts(session, binding, values)


def test_missing_refit_parent_row_remains_a_failure(monkeypatch):
    session, binding, values, rows, _, _ = _world(monkeypatch, refit=True)
    del rows["parent"]
    with pytest.raises(ValueError, match="champion identity mismatch"):
        predicates._model_facts(session, binding, values)


@pytest.mark.parametrize("family", ["modeling", "monitoring"])
@pytest.mark.parametrize("refit", [False, True])
def test_declared_previous_selected_predicate_is_unavailable_without_authenticated_parent(monkeypatch, family, refit):
    session, binding, values, _, _, _ = _world(monkeypatch, refit=refit)
    facts, artifacts = predicates._model_facts(session, binding, values)
    expected = ExpectedCase(result="done", assertions=[{
        "kind": "output_equals", "tool": "modeling.post_training_action",
        "path": ["challenger_comparison", "champion", "label"], "value": "previous_selected_experiment",
    }])
    recomputed = {"supported_checks_verified": True, "values": values,
                  "checks": {"native_steps": {"status": "verified"}}}
    native_facts = {"values_by_tool": facts, "artifacts": artifacts}
    if refit:
        assert independently_expected(expected, {}, recomputed, family=family, native_facts=native_facts)
    else:
        with pytest.raises(TrustInputError, match="predicate_unavailable"):
            independently_expected(expected, {}, recomputed, family=family, native_facts=native_facts)
