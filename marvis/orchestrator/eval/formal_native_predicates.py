"""Bounded native facts accompanying independently recomputed domain values.

Report existence establishes captured bytes and their task/producer binding;
it does not establish report-cell accuracy, source maturity or business approval.
No producer calculation, original workspace, pickle or network is invoked.
"""
from pathlib import Path
import json

from .acceptance_trust import TrustInputError
from .runtime_archive_domain import DomainArchive, equal, one_step
from .runtime_archive_modeling import _selection_projection_allowed
from .runtime_archive_reader import _original_path


def _artifact(session, step, field):
    path = step["output"].get(field)
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("native artifact path missing")
    original = Path(session.manifest["original_workspace"]).resolve()
    relative = Path(path).relative_to(original)
    if relative.parts[:2] != ("tasks", session.binding.task_id):
        raise ValueError("native artifact belongs to another task")
    captured = _original_path(session.archive, str(original), path)
    if not captured.is_file() or not captured.stat().st_size:
        raise ValueError("native artifact bytes missing")
    return field


def _model_facts(session, binding, values):
    selected = one_step(session.steps, "modeling.select_experiment")
    original = values["selection_evidence"]["experiment_id"]
    if (selected["input"].get("selected_experiment_id") != original
            or original not in selected["input"].get("experiment_ids", [])
            or selected["output"].get("selected_experiment_id") != binding.selected_experiment_id):
        raise ValueError("explicit selection native identity mismatch")
    row = session.conn.execute("SELECT * FROM experiments WHERE id=? AND task_id=?",
                               (binding.selected_experiment_id, binding.task_id)).fetchone()
    if row is None:
        raise ValueError("selected experiment missing")
    select_values = {"metrics": values["metrics"], "selection_evidence": values["selection_evidence"],
        "refit": {"applied": values["scope"]["fit_scope"] == "full_development"},
        "policy_decision": {"explicit_selection": True}, "recipe": row["recipe_id"]}
    if selected["output"].get("recipe") != row["recipe_id"]:
        raise ValueError("selected recipe differs from native experiment")
    equal(selected["output"].get("policy_decision", {}).get("explicit_selection"), True)
    reports = one_step(session.steps, "modeling.generate_model_reports")
    if reports["input"].get("selected_experiment_id") != binding.selected_experiment_id:
        raise ValueError("report primary differs from selected model")
    report_items = reports["output"].get("reports", [])
    if (not report_items or report_items[0].get("experiment_id") != binding.selected_experiment_id
            or report_items[0].get("report_path") != reports["output"].get("report_path")):
        raise ValueError("report primary native file binding mismatch")
    business = reports["input"].get("business_columns") or {}
    requirements = (("sample_analysis", ("loan_month_col",)),
        ("vintage", ("loan_month_col", "mob_observe_cols")),
        ("amount_bin", ("loan_amount_col",)), ("low_pricing", ("interest_rate_col",)),
        ("product_list", ()))
    sections = [{"section": name, "available": all(bool(business.get(k)) for k in keys)
                 and (name != "product_list" or bool(reports["input"].get("feature_dictionary_id")))}
                for name, keys in requirements]
    if values["scope"]["fit_scope"] == "full_development":
        sections.append({"section": "development_refit_scope", "available": True})
    retained = reports["output"].get("section_status", [])
    equal([{k: item.get(k) for k in ("section", "available")} for item in retained], sections)
    post = one_step(session.steps, "modeling.post_training_action")
    if post["input"].get("experiment_id") != binding.selected_experiment_id:
        raise ValueError("delivery experiment differs from final selected model")
    if values["scope"].get("parent_experiment_id") is None:
        # The native producer returns exactly {} when no champion is available.
        # A missing field or an invented unavailable marker is not that record.
        comparison = post["output"].get("challenger_comparison")
        if ("parent_experiment_id" not in values["scope"]
                or not _selection_projection_allowed(selected, values["scope"], values=values["metrics"])
                or type(comparison) is not dict or comparison
                or row["artifact_id"] != binding.final_artifact_id
                or selected["output"].get("artifact_id") != binding.final_artifact_id
                or row["status"] not in {"selected", "handed_off", "validated", "delivered", "approved", "champion"}):
            raise ValueError("retained candidate requires authenticated native empty no-baseline boundary")
        equal(values["selection_evidence"]["metrics"], values["metrics"])
        return {
            "modeling.select_experiment": select_values,
            "modeling.generate_model_reports": {"section_status": sections},
            "modeling.post_training_action": {"challenger_comparison": {}},
        }, {"modeling.generate_model_reports": [_artifact(session, reports, "report_path")],
            "modeling.post_training_action": [_artifact(session, post, name) for name in ("model_card_path", "approval_package_path")]}
    champion = post["output"].get("challenger_comparison", {}).get("champion", {})
    parent_id = values["scope"].get("parent_experiment_id")
    if not parent_id:
        raise ValueError("previous-selected predicate requires authenticated refit parent")
    parent = session.conn.execute("SELECT * FROM experiments WHERE id=? AND task_id=?",
                                  (parent_id, binding.task_id)).fetchone()
    if (parent is None or parent["status"] not in {"selected", "validated", "delivered", "approved", "champion"}
            or parent["created_at"] >= row["created_at"]
            or champion.get("experiment_id") != parent_id or champion.get("artifact_id") != parent["artifact_id"]):
        raise ValueError("previous selected champion identity mismatch")
    equal(champion.get("metrics"), values["selection_evidence"]["metrics"])
    return {
        "modeling.select_experiment": select_values,
        "modeling.generate_model_reports": {"section_status": sections},
        "modeling.post_training_action": {"challenger_comparison": {"champion": {"label": "previous_selected_experiment"}}},
    }, {"modeling.generate_model_reports": [_artifact(session, reports, "report_path")],
        "modeling.post_training_action": [_artifact(session, post, name) for name in ("model_card_path", "approval_package_path")]}


def _strategy_facts(session, binding, native_tools):
    reference = one_step(session.steps, "strategy.backtest_strategy")["input"].get("sample_design_ref")
    if reference:
        if binding.sample_design is None:
            raise ValueError("strategy sample prerequisite original binding missing")
        with DomainArchive(session.original, session.sha, binding.sample_design,
                           family="predicate-sample", checks=("exact_sample_reference",)) as sample:
            sample.open()
            sample.current = "exact_sample_reference"
            native_tools.update({identity: step["tool"] for identity, step in sample.steps.items()})
            produced = one_step(sample.steps, "strategy.materialize_sample_design_v2_native")["output"]
            design = produced.get("bundle", {}).get("sample_design", {})
            if (design.get("sample_design_id") != reference["sample_design_id"]
                    or design.get("content_hash") != reference["sample_design_content_hash"]):
                raise ValueError("strategy sample prerequisite output differs from consumed source")
            descriptor = produced.get("artifacts", {}).get("bundle", {})
            artifact = sample.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?",
                (reference["artifact_id"], binding.task_id)).fetchone()
            if (artifact is None or artifact["origin_tool"] != "strategy.materialize_sample_design_v2_native"
                    or artifact["kind"] != "strategy_sample_design_v2_json"
                    or artifact["content_hash"] != reference["artifact_content_hash"]
                    or descriptor.get("content_hash") != reference["artifact_content_hash"]
                    or descriptor.get("filename") != Path(artifact["path"]).name):
                raise ValueError("strategy sample prerequisite artifact differs")
            equal(json.loads(artifact["provenance_json"])["request"],
                  one_step(sample.steps, "strategy.materialize_sample_design_v2_native")["input"])
            sample.mark("exact_sample_reference")
        if not sample.result["supported_checks_verified"]:
            raise ValueError("strategy sample prerequisite originals mismatch")
    adopt = one_step(session.steps, "strategy.adopt_strategy")
    row = session.conn.execute("SELECT * FROM strategies WHERE id=? AND task_id=?",
                               (binding.strategy_id, binding.task_id)).fetchone()
    if (row is None or row["status"] != "adopted" or adopt["input"].get("strategy_id") != binding.strategy_id
            or adopt["output"].get("strategy_id") != binding.strategy_id):
        raise ValueError("local strategy adoption native identity mismatch")
    challenger = one_step(session.steps, "strategy.render_challenger_report")
    if challenger["input"].get("strategy_id") != binding.strategy_id or challenger["input"].get("champion_strategy_id"):
        raise ValueError("no-baseline predicate has a different native input")
    document = one_step(session.steps, "strategy.render_strategy_doc")
    if document["input"].get("strategy_id") != binding.strategy_id:
        raise ValueError("strategy document owner mismatch")
    return {"strategy.adopt_strategy": {"asset_status": "adopted_local"},
            "strategy.render_challenger_report": {"status": "no_baseline"}}, {
            "strategy.render_strategy_doc": [_artifact(session, document, "doc_path")]}


def native_predicates(archive, sha, binding, *, family, model_values=None):
    """Reopen only a bounded authenticated copy; preserve incomplete originals."""
    facts, artifacts, native_tools = {}, {}, {}
    bindings = [binding.modeling, binding] if family == "monitoring" else [binding]
    for current in bindings:
        with DomainArchive(archive, sha, current, family="predicate-facts", checks=("predicate_facts",)) as session:
            session.open()
            session.current = "predicate_facts"
            native_tools.update({identity: step["tool"] for identity, step in session.steps.items()})
            if current is bindings[0] and family in {"modeling", "monitoring"}:
                values, paths = _model_facts(session, current, model_values)
            elif family == "strategy":
                values, paths = _strategy_facts(session, current, native_tools)
            else:
                score = one_step(session.steps, "modeling.score_dataset")
                row = session.conn.execute("SELECT row_count FROM datasets WHERE id=? AND task_id=?",
                    (score["output"].get("result_dataset_id"), binding.task_id)).fetchone()
                if row is None:
                    raise ValueError("native scored dataset missing")
                values, paths = {"modeling.score_dataset": {"dataset_rows": row["row_count"]}}, {}
            facts.update(values)
            artifacts.update(paths)
            session.mark("predicate_facts", scope="native_inputs_identity_and_captured_files; excludes_report_cell_accuracy")
        if not session.result["supported_checks_verified"]:
            raise TrustInputError("formal_native_predicate_originals_mismatch")
    return {"values_by_tool": facts, "artifacts": artifacts, "tools_by_step": native_tools}
