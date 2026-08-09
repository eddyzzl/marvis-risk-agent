import json

import pytest

from marvis.agent.turn_handlers import (
    _c1_semantic_snapshot_matches,
    _c1_snapshot,
    _parse_c1_reply,
)


def _state() -> dict:
    return {
        "files": [
            {
                "dataset_id": "ds-old",
                "name": "vintage_panel.parquet",
                "columns": ["account_id", "bad"],
                "target_candidates": ["bad"],
            },
            {
                "dataset_id": "ds-reg",
                "name": "MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet",
                "columns": ["application_id", "case_weight", "loss_amount_target"],
                "target_candidates": [],
            },
            {
                "dataset_id": "ds-features",
                "name": "bureau_features.parquet",
                "columns": ["application_id", "bureau_score"],
                "target_candidates": [],
            },
        ],
        "anchor_id": "ds-old",
        "feature_ids": ["ds-reg", "ds-features"],
        "target_col": "bad",
    }


def test_c1_natural_language_can_switch_anchor_ignore_rest_and_set_target() -> None:
    assignment = _parse_c1_reply(
        "只使用 MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet 作为样本主表，"
        "目标列设为 loss_amount_target；其余文件全部忽略，不要作为特征表。",
        _state(),
    )

    assert assignment == {
        "anchor_id": "ds-reg",
        "feature_ids": [],
        "target_col": "loss_amount_target",
    }


def test_c1_natural_language_keeps_explicit_feature_when_ignoring_rest() -> None:
    assignment = _parse_c1_reply(
        "MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet 是样本主表，"
        "bureau_features.parquet 作为特征表，其他文件忽略，"
        "目标列 loss_amount_target。",
        _state(),
    )

    assert assignment == {
        "anchor_id": "ds-reg",
        "feature_ids": ["ds-features"],
        "target_col": "loss_amount_target",
    }


class _SemanticClient:
    def __init__(
        self,
        *,
        valid_review: bool,
        route_constraint: str = "",
        route_payload: dict | None = None,
    ):
        self.valid_review = valid_review
        self.route_constraint = route_constraint
        self.route_payload = route_payload
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("prompt_name") == "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS":
            if not self.valid_review:
                return json.dumps(
                    {
                        "action": "confirm",
                        "params": {},
                        "constraint": "",
                        "reason": "伪造授权",
                        "confidence": "high",
                        "explicit_authorization": True,
                    },
                    ensure_ascii=False,
                )
            instruction = json.loads(kwargs["user_prompt"])["instruction"]
            return json.dumps(
                {
                    "verdict": "authorize",
                    "evidence_quote": instruction,
                    "reason": "用户明确接受当前展示的推荐角色。",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_authorization": False,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            self.route_payload
            or {
                "action": "confirm",
                "params": {},
                "constraint": self.route_constraint,
                "reason": "用户明确接受当前建议。",
                "confidence": "high",
                "explicit_authorization": True,
            },
            ensure_ascii=False,
        )


def test_c1_contextual_recommendation_uses_independent_two_pass_review() -> None:
    client = _SemanticClient(valid_review=True)
    text = "我确认无误，请按推荐的样本主表和特征表继续。"

    assignment = _parse_c1_reply(
        text,
        _state(),
        llm_client=client,
        require_semantic_authorization=True,
    )

    assert assignment is not None
    authorization = assignment.pop("_semantic_authorization")
    assert assignment == {
        "anchor_id": "ds-old",
        "feature_ids": ["ds-reg", "ds-features"],
        "target_col": "bad",
    }
    assert authorization == {
        "source": "llm_two_pass",
        "route_reason": "用户明确接受当前建议。",
        "evidence_quote": text,
        "review_reason": "用户明确接受当前展示的推荐角色。",
        "confidence": "high",
        "c1_snapshot_sha256": authorization["c1_snapshot_sha256"],
        "proposed_assignment_sha256": authorization[
            "proposed_assignment_sha256"
        ],
        "proposed_assignment": assignment,
    }
    assert len(authorization["c1_snapshot_sha256"]) == 64
    assert len(authorization["proposed_assignment_sha256"]) == 64
    assert [call.get("prompt_name") for call in client.calls] == [
        "GATE_INSTRUCTION_ROUTER_SYS",
        "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS",
    ]


def test_c1_contextual_recommendation_fails_closed_without_independent_review() -> None:
    text = "我确认无误，请按推荐的样本主表和特征表继续。"
    client = _SemanticClient(valid_review=False)

    assert _parse_c1_reply(
        text,
        _state(),
        require_semantic_authorization=True,
    ) is None
    assert _parse_c1_reply(
        text,
        _state(),
        llm_client=client,
        require_semantic_authorization=True,
    ) is None
    assert [call.get("prompt_name") for call in client.calls] == [
        "GATE_INSTRUCTION_ROUTER_SYS",
        "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS",
    ]


def test_c1_explicit_role_change_is_grounded_then_semantically_authorized() -> None:
    client = _SemanticClient(valid_review=True)

    assignment = _parse_c1_reply(
        "确认继续，但把 MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet 作为样本主表，"
        "目标列改为 loss_amount_target，其余文件全部忽略。",
        _state(),
        llm_client=client,
        require_semantic_authorization=True,
    )

    assert assignment is not None
    authorization = assignment.pop("_semantic_authorization")
    assert assignment == {
        "anchor_id": "ds-reg",
        "feature_ids": [],
        "target_col": "loss_amount_target",
    }
    assert authorization["proposed_assignment"] == assignment
    assert [call.get("prompt_name") for call in client.calls] == [
        "GATE_INSTRUCTION_ROUTER_SYS",
        "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS",
    ]


def test_c1_semantic_route_with_constraint_fails_closed_before_review() -> None:
    client = _SemanticClient(
        valid_review=True,
        route_constraint="先改目标列再继续",
    )

    assert _parse_c1_reply(
        "我确认无误，请按推荐的样本主表和特征表继续。",
        _state(),
        llm_client=client,
        require_semantic_authorization=True,
    ) is None
    assert [call.get("prompt_name") for call in client.calls] == [
        "GATE_INSTRUCTION_ROUTER_SYS",
    ]


@pytest.mark.parametrize(
    "route_payload",
    [
        {
            "action": "confirm",
            "params": {},
            "reason": "缺少 constraint",
            "confidence": "high",
            "explicit_authorization": True,
        },
        {
            "action": "confirm",
            "params": [],
            "constraint": "",
            "reason": "params 类型错误",
            "confidence": "high",
            "explicit_authorization": True,
        },
        {
            "action": "confirm",
            "params": {},
            "constraint": False,
            "reason": "constraint 类型错误",
            "confidence": "high",
            "explicit_authorization": True,
        },
        {
            "action": "confirm",
            "params": {},
            "constraint": "",
            "reason": "含额外字段",
            "confidence": "high",
            "explicit_authorization": True,
            "unexpected": "value",
        },
        {
            "action": "confirm",
            "params": {},
            "constraint": "",
            "reason": "授权布尔类型错误",
            "confidence": "high",
            "explicit_authorization": 1,
        },
    ],
)
def test_c1_malformed_first_pass_never_reaches_authorization_review(
    route_payload: dict,
) -> None:
    client = _SemanticClient(valid_review=True, route_payload=route_payload)

    assert _parse_c1_reply(
        "我确认无误，请按推荐的样本主表和特征表继续。",
        _state(),
        llm_client=client,
        require_semantic_authorization=True,
    ) is None
    assert [call.get("prompt_name") for call in client.calls] == [
        "GATE_INSTRUCTION_ROUTER_SYS",
    ]


def test_c1_snapshot_changes_with_recommendation_or_schema_drift() -> None:
    original = _state()
    changed_assignment = {**_state(), "anchor_id": "ds-reg"}
    changed_schema = _state()
    changed_schema["files"] = [dict(item) for item in changed_schema["files"]]
    changed_schema["files"][0]["columns"] = ["account_id", "bad", "new_col"]

    assert _c1_snapshot(original) != _c1_snapshot(changed_assignment)
    assert _c1_snapshot(original) != _c1_snapshot(changed_schema)


def test_c1_semantic_receipt_is_bound_to_reviewed_snapshot() -> None:
    assignment = _parse_c1_reply(
        "我确认无误，请按推荐的样本主表和特征表继续。",
        _state(),
        llm_client=_SemanticClient(valid_review=True),
        require_semantic_authorization=True,
    )
    assert assignment is not None
    assert _c1_semantic_snapshot_matches(assignment, _state())

    changed = _state()
    changed["files"] = [dict(item) for item in changed["files"]]
    changed["files"][0]["content_hash"] = "changed-content"
    assert not _c1_semantic_snapshot_matches(assignment, changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("anchor_id", "ds-reg"),
        ("feature_ids", ["ds-features"]),
        ("target_col", "loss_amount_target"),
    ],
)
def test_c1_semantic_receipt_rejects_assignment_tampering(field, value) -> None:
    assignment = _parse_c1_reply(
        "我确认无误，请按推荐的样本主表和特征表继续。",
        _state(),
        llm_client=_SemanticClient(valid_review=True),
        require_semantic_authorization=True,
    )
    assert assignment is not None

    assignment[field] = value

    assert not _c1_semantic_snapshot_matches(assignment, _state())


def test_agent_exact_confirmation_still_requires_two_pass_review() -> None:
    client = _SemanticClient(valid_review=True)

    assignment = _parse_c1_reply(
        "确认",
        _state(),
        llm_client=client,
        require_semantic_authorization=True,
    )

    assert assignment is not None
    assert "_semantic_authorization" in assignment
    assert [call.get("prompt_name") for call in client.calls] == [
        "GATE_INSTRUCTION_ROUTER_SYS",
        "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS",
    ]


def test_agent_mixed_withholding_cannot_authorize_extracted_assignment() -> None:
    client = _SemanticClient(valid_review=True)

    assignment = _parse_c1_reply(
        "不要把 MODEL-REGRESSION-RANDOM-OOT_abcd1234.parquet 作为样本主表，"
        "也不要用 loss_amount_target，先别继续。",
        _state(),
        llm_client=client,
        require_semantic_authorization=True,
    )

    assert assignment is None
    assert client.calls == []


def test_structured_c1_payload_requires_validated_ui_action() -> None:
    payload = "[C1]" + json.dumps(
        {
            "anchor_id": "ds-reg",
            "feature_ids": ["ds-features"],
            "target_col": "loss_amount_target",
        }
    )

    assert _parse_c1_reply(payload, _state()) is None
    assert _parse_c1_reply(
        payload,
        _state(),
        trusted_ui_action=True,
    ) == {
        "anchor_id": "ds-reg",
        "feature_ids": ["ds-features"],
        "target_col": "loss_amount_target",
    }
