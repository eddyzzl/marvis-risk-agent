"""Prepare exact report sources around explicitly selected asset evidence."""

import hashlib
from types import SimpleNamespace

from marvis.agent.strategy_request_compiler.contracts import (
    StandardWorkflowRequestDraft,
)
from marvis.agent.strategy_workflows._foundation_delivery import (
    validate_report_bundle_v2_inputs,
)
from marvis.agent.turn_handlers.strategy_candidates import (
    _strategy_report_bundle_v2_plan_slots,
)
from marvis.packs.strategy.asset_stability import canonical
from marvis.packs.strategy.asset_stability_report import (
    add_asset_stability_report_evidence,
)
from marvis.packs.strategy.asset_stability_tools import load_asset_stability_artifact
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.report_bundle_tools import (
    _load_sources,
    _validate_inputs,
    run_build_strategy_report_bundle_v2,
)
from marvis.packs.strategy.report_bundle_adapters import (
    build_strategy_report_bundle_source_inputs,
)


def prepare_asset_stability_report(inputs, *, task, runtime):
    if not isinstance(inputs, dict) or set(inputs) != {
        "title",
        "status",
        "asset_stability_refs",
    }:
        raise StrategyError(
            "asset report requires explicit title, status and asset_stability_refs"
        )
    controls = validate_report_bundle_v2_inputs(
        {"title": inputs["title"], "status": inputs["status"]}, None
    )
    refs = inputs["asset_stability_refs"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= 16:
        raise StrategyError(
            "asset report requires 1 to 16 explicitly selected exact evidence refs"
        )
    for ref in refs:
        load_asset_stability_artifact(runtime, task_id=task.id, ref=ref)
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_report_bundle_v2", workflow_inputs=controls
    )
    proposal = _strategy_report_bundle_v2_plan_slots(
        SimpleNamespace(settings=runtime.settings), task, draft, source_message=None
    )
    proposal["asset_stability_refs"] = refs
    proposal = _validate_inputs(proposal)
    sources = _load_sources(runtime, task_id=task.id, request=proposal)
    source_inputs = build_strategy_report_bundle_source_inputs(
        project_context=sources.project_context,
        sample_design=sources.sample_design,
        candidate_pool=sources.candidate_pool,
        pool_validations=sources.pool_validations,
        candidate_stability=sources.candidate_stability,
        pool_stability=sources.pool_stability,
        voting_candidate_search=sources.voting_candidate_search,
        cross_candidate_search=sources.cross_candidate_search,
        cross_rule_search=sources.cross_rule_search,
        pool_impact=sources.pool_impact,
        impact_cube=sources.impact_cube,
        model_evidence=sources.model_evidence,
        training_evidence=sources.training_evidence,
        score_evidence=sources.score_evidence,
    )
    add_asset_stability_report_evidence(
        source_inputs,
        bindings=sources.asset_stabilities,
        sample_design=sources.sample_design,
        task_id=task.id,
    )
    return {
        "proposal": proposal,
        "proposal_hash": hashlib.sha256(canonical(proposal)).hexdigest(),
    }


def publish_asset_stability_report(inputs, ctx, runtime):
    if not isinstance(inputs, dict) or set(inputs) != {"proposal", "proposal_hash"}:
        raise StrategyError("asset report publish requires one exact proposal")
    proposal = inputs["proposal"]
    if hashlib.sha256(canonical(proposal)).hexdigest() != inputs["proposal_hash"]:
        raise StrategyError("asset report proposal changed before publication")
    return run_build_strategy_report_bundle_v2(proposal, ctx, runtime)
