"""Semantic authorization for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from marvis.agent.instruction_router import route_instruction
from marvis.agent.plan_driver import confirmation_is_explicitly_withheld
from marvis.agent.semantic_authorization import review_semantic_authorization
import hashlib
import json
import re
from . import c1_state as c1_state_lane
from . import contracts as contracts_lane


def _semantic_exact_gate_authorization(
    runtime: contracts_lane.DriverTurnRuntime,
    text: str,
    *,
    gate_context: str,
    proposed_params: Mapping[str, object],
) -> dict[str, object] | None:
    """Authorize prose only after independent strict route and review passes."""

    if (
        runtime.llm_client is None
        or not text
        or confirmation_is_explicitly_withheld(text)
    ):
        return None
    try:
        route = route_instruction(
            runtime.llm_client,
            gate_context=gate_context,
            instruction=text,
            param_schema=[],
            strict_contract=True,
        )
    except Exception:
        return None
    if (
        route.get("action") != "confirm"
        or route.get("confidence") != "high"
        or route.get("explicit_authorization") is not True
        or bool(route.get("params"))
        or bool(str(route.get("constraint") or "").strip())
    ):
        return None
    review = review_semantic_authorization(
        runtime.llm_client,
        gate_context=gate_context,
        instruction=text,
        proposed_params=dict(proposed_params),
    )
    if not review.authorized:
        return None
    return {
        "source": "llm_two_pass",
        "route_reason": str(route.get("reason") or "").strip(),
        "evidence_quote": review.evidence_quote,
        "review_reason": review.reason,
        "confidence": review.confidence,
    }


def _semantic_c1_recommendation_authorization(
    text: str,
    c1_state: dict,
    llm_client,
    *,
    proposed_assignment: dict,
) -> dict | None:
    """Authorize a contextual C1 reply through the same independent two-pass gate.

    Exact ``确认`` and the typed ``[C1]`` payload remain deterministic. Any longer
    sentence that accepts the proposed file roles is interpreted by the LLM and
    independently reviewed; a missing client, malformed response, conditional
    wording, requested change, or client failure leaves the C1 gate open.
    """

    if llm_client is None or not text:
        return None
    proposed_assignment = {
        "anchor_id": proposed_assignment.get("anchor_id"),
        "feature_ids": list(proposed_assignment.get("feature_ids") or []),
        "target_col": proposed_assignment.get("target_col"),
    }
    try:
        route = route_instruction(
            llm_client,
            gate_context=(
                "文件角色与目标列授权：平台已把用户原话约束到当前数据集中的"
                "一个具体角色/目标列方案；confirm 仅表示用户明确、即时、无条件地"
                "授权采用下列精确方案："
                + json.dumps(
                    proposed_assignment,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            instruction=text,
            param_schema=[],
            strict_contract=True,
        )
    except Exception:
        return None
    if (
        route.get("action") != "confirm"
        or route.get("confidence") != "high"
        or route.get("explicit_authorization") is not True
        or bool(route.get("params"))
        or bool(str(route.get("constraint") or "").strip())
    ):
        return None
    review = review_semantic_authorization(
        llm_client,
        gate_context="采用当前界面展示的文件角色与目标列建议",
        instruction=text,
        proposed_params=proposed_assignment,
    )
    if not review.authorized:
        return None
    snapshot = c1_state_lane._c1_snapshot(c1_state)
    assignment_payload = json.dumps(
        proposed_assignment,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "source": "llm_two_pass",
        "route_reason": str(route.get("reason") or "").strip(),
        "evidence_quote": review.evidence_quote,
        "review_reason": review.reason,
        "confidence": review.confidence,
        "c1_snapshot_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
        "proposed_assignment_sha256": hashlib.sha256(
            assignment_payload.encode("utf-8")
        ).hexdigest(),
        "proposed_assignment": proposed_assignment,
    }


def _instruction_explicitly_names_identifier(
    instruction: str,
    identifier: str,
) -> bool:
    """Ground an LLM-selected binding in exact user-supplied operands.

    This does not classify intent or look for action keywords.  It only proves
    that the already-selected bounded operation names the platform candidate it
    would mutate, preventing an erroneous route from silently binding a
    different file or column.
    """

    text = str(instruction or "")
    value = str(identifier or "").strip()
    if not text or not value:
        return False
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
                text,
                re.IGNORECASE,
            )
        )
    return value in text
