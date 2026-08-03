"""Agent-mode gate-instruction router: classify a free-text reply at a gate into
confirm / adjust / replan / clarify and extract parameters. Pure + FakeLLM-driven.
"""

from __future__ import annotations

from marvis.agent.instruction_router import parse_route, route_instruction


class _FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.payload


class _SequencedLLM:
    def __init__(self, payloads: list[str]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if len(self.calls) <= len(self.payloads):
            return self.payloads[len(self.calls) - 1]
        return self.payloads[-1]


def test_parse_route_adjust_extracts_params():
    out = parse_route('{"action":"adjust","params":{"n_trials":20},"reason":"调大搜索"}')
    assert out["action"] == "adjust"
    assert out["params"] == {"n_trials": 20}


def test_parse_route_extracts_json_from_markdown():
    out = parse_route('模型判断如下:\n```json\n{"action":"adjust","params":{"n_trials":20},"reason":"调大搜索"}\n```')
    assert out["action"] == "adjust"
    assert out["params"] == {"n_trials": 20}


def test_parse_route_adjust_without_params_falls_back_to_clarify():
    out = parse_route('{"action":"adjust","params":{},"reason":""}')
    assert out["action"] == "clarify"
    assert out["reason"]


def test_parse_route_replan_keeps_constraint():
    out = parse_route('{"action":"replan","constraint":"改用 xgb 重新建模","reason":"x"}')
    assert out["action"] == "replan"
    assert out["constraint"] == "改用 xgb 重新建模"


def test_parse_route_junk_is_clarify():
    assert parse_route("not json at all")["action"] == "clarify"


def test_parse_route_unknown_action_is_clarify():
    assert parse_route('{"action":"frobnicate","params":{}}')["action"] == "clarify"


def test_route_instruction_passes_context_and_instruction_to_llm():
    fake = _FakeLLM('{"action":"confirm","reason":"同意"}')
    out = route_instruction(fake, gate_context="特征筛选完成", instruction="可以,继续")
    assert out["action"] == "confirm"
    prompt = fake.calls[0]["user_prompt"]
    assert "特征筛选完成" in prompt
    assert "可以,继续" in prompt


def test_route_instruction_returns_semantic_authorization_and_confidence():
    fake = _FakeLLM(
        '{"action":"confirm","params":{},"constraint":"",'
        '"reason":"用户明确要求按当前方案继续",'
        '"confidence":"high","explicit_authorization":true}'
    )

    out = route_instruction(
        fake,
        gate_context="特征筛选完成",
        instruction="这版筛选结果符合预期，接着完成剩余步骤。",
    )

    assert out["action"] == "confirm"
    assert out["confidence"] == "high"
    assert out["explicit_authorization"] is True


def test_route_instruction_retries_once_after_unparseable_reply():
    fake = _SequencedLLM(["not json", '{"action":"adjust","params":{"n_trials":30},"reason":"重试可解析"}'])

    out = route_instruction(fake, gate_context="调参节点", instruction="n_trials 到 30")

    assert out["action"] == "adjust"
    assert out["params"] == {"n_trials": 30}
    assert len(fake.calls) == 2
    assert "上一次返回无法解析" in fake.calls[1]["user_prompt"]


def test_route_instruction_injects_param_schema_into_prompt():
    """AGT-5: the gate's adjustable-parameter schema (name/type/current value/
    bounds) is rendered into the prompt so the router extracts adjust params
    against real parameter names instead of guessing them from the instruction."""
    fake = _FakeLLM('{"action":"adjust","params":{"n_trials":30},"reason":"调大轮数"}')

    route_instruction(
        fake,
        gate_context="调参节点",
        instruction="n_trials 调到 30",
        param_schema=[
            {"name": "n_trials", "type": "integer", "current": 20, "bounds": {"min": 1}},
            {"name": "leakage_ks", "type": "number", "current": 0.4, "bounds": {"min": 0, "max": 1}},
        ],
    )

    prompt = fake.calls[0]["user_prompt"]
    assert "【可调参数】" in prompt
    assert "n_trials" in prompt
    assert "当前值=20" in prompt
    assert "leakage_ks" in prompt
    assert "min=0, max=1" in prompt
    assert fake.calls[0]["prompt_name"] == "GATE_INSTRUCTION_ROUTER_SYS"
    assert fake.calls[0]["prompt_version"] == 7


def test_route_instruction_omits_param_schema_section_when_empty():
    fake = _FakeLLM('{"action":"confirm","reason":"同意"}')

    route_instruction(fake, gate_context="计划总览", instruction="可以")

    assert "【可调参数】" not in fake.calls[0]["user_prompt"]


def test_route_instruction_recovers_declared_candidate_adoption_from_replan():
    """Candidate adoption is a semantic decision at the current gate, not a DAG
    rewrite. The bounded review is triggered by the declared selection control
    and must ground the returned id in its enum."""
    selected_id = "experiment_f1eb251544394fefb8092301676b5a20"
    fake = _SequencedLLM(
        [
            (
                '{"action":"replan","params":{},"constraint":"进入报告",'
                '"reason":"用户要求转入下一阶段","confidence":"high",'
                '"explicit_authorization":false}'
            ),
            (
                '{"action":"confirm","params":{"selected_experiment_id":"'
                + selected_id
                + '"},"constraint":"","reason":"用户明确采用当前候选并进入报告",'
                '"confidence":"high","explicit_authorization":true}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="选择实验",
        instruction=f"采用平台标星的 LR 实验 {selected_id} 作为主实验，转入报告阶段。",
        param_schema=[
            {
                "name": "selected_experiment_id",
                "type": "string",
                "current": selected_id,
                "enum": [selected_id, "experiment_other"],
                "bounds": {"enum": [selected_id, "experiment_other"]},
            }
        ],
    )

    assert out == {
        "action": "confirm",
        "params": {"selected_experiment_id": selected_id},
        "constraint": "",
        "reason": "用户明确采用当前候选并进入报告",
        "confidence": "high",
        "explicit_authorization": True,
    }
    assert len(fake.calls) == 2
    assert "候选选择节点语义复核" in fake.calls[1]["user_prompt"]
    assert selected_id in fake.calls[0]["user_prompt"]


def test_route_instruction_selection_review_uses_candidate_semantics():
    """The LLM receives the candidate's stable id, recipe, visible label, and
    recommendation marker, so a natural algorithm-name choice can be resolved
    semantically without a code-side Chinese keyword matcher.
    """
    selected_id = "experiment_lr"
    fake = _SequencedLLM(
        [
            (
                '{"action":"clarify","params":{},"constraint":"",'
                '"reason":"请提供具体实验ID","confidence":"low",'
                '"explicit_authorization":false}'
            ),
            (
                '{"action":"confirm","params":{"selected_experiment_id":"'
                + selected_id
                + '"},"constraint":"","reason":"唯一候选是平台推荐的逻辑回归",'
                '"confidence":"high","explicit_authorization":true}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="选择实验",
        instruction="采用平台推荐的逻辑回归作为最终模型。",
        param_schema=[
            {
                "name": "selected_experiment_id",
                "type": "string",
                "current": selected_id,
                "enum": ["experiment_lgb", selected_id],
                "bounds": {"enum": ["experiment_lgb", selected_id]},
                "candidates": [
                    {
                        "experiment_id": "experiment_lgb",
                        "recipe": "lgb_multiclass",
                        "display_name": "LightGBM（多分类）",
                        "recommended": False,
                    },
                    {
                        "experiment_id": selected_id,
                        "recipe": "lr_multiclass",
                        "display_name": "逻辑回归（多分类）",
                        "recommended": True,
                    },
                ],
            }
        ],
    )

    assert out["action"] == "confirm"
    assert out["params"] == {"selected_experiment_id": selected_id}
    prompt = fake.calls[0]["user_prompt"]
    assert '"recipe":"lr_multiclass"' in prompt
    assert '"display_name":"逻辑回归（多分类）"' in prompt
    assert '"recommended":true' in prompt
    assert "仅在名称或 recipe 唯一匹配" in fake.calls[1]["user_prompt"]
    review_schema = fake.calls[1]["json_schema"]
    assert review_schema["name"] == "candidate_selection_route"
    assert (
        review_schema["schema"]["properties"]["params"]["properties"][
            "selected_experiment_id"
        ]["enum"]
        == ["experiment_lgb", selected_id]
    )
    assert (
        review_schema["schema"]["properties"]["params"]["additionalProperties"]
        is False
    )


def test_route_instruction_selection_review_keeps_ambiguous_name_as_clarify():
    """If more than one candidate shares the requested recipe/name family, the
    structured LLM must ask the user to disambiguate instead of guessing an id.
    """
    fake = _SequencedLLM(
        [
            (
                '{"action":"replan","params":{},"constraint":"采用逻辑回归",'
                '"reason":"候选不明确","confidence":"medium",'
                '"explicit_authorization":false}'
            ),
            (
                '{"action":"clarify","params":{},"constraint":"",'
                '"reason":"有两个逻辑回归候选，请选择具体实验",'
                '"confidence":"low","explicit_authorization":false}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="选择实验",
        instruction="采用逻辑回归作为最终模型。",
        param_schema=[
            {
                "name": "selected_experiment_id",
                "type": "string",
                "current": "experiment_lr_a",
                "enum": ["experiment_lr_a", "experiment_lr_b"],
                "bounds": {"enum": ["experiment_lr_a", "experiment_lr_b"]},
                "candidates": [
                    {
                        "experiment_id": "experiment_lr_a",
                        "recipe": "lr",
                        "display_name": "逻辑回归",
                        "recommended": True,
                    },
                    {
                        "experiment_id": "experiment_lr_b",
                        "recipe": "lr",
                        "display_name": "逻辑回归",
                        "recommended": False,
                    },
                ],
            }
        ],
    )

    assert out["action"] == "clarify"
    assert "两个逻辑回归候选" in out["reason"]


def test_route_instruction_canonicalizes_grounded_confirm_to_experiment_id_only():
    """A structured model may echo recipe/display evidence alongside the enum
    id. Once the id, confidence, and explicit authorization are all valid, the
    platform keeps only the executable selected_experiment_id field.
    """
    selected_id = "experiment_lr"
    fake = _FakeLLM(
        '{"action":"confirm","params":{"selected_experiment_id":"'
        + selected_id
        + '","recipe":"lr_regressor"},"constraint":"",'
        '"reason":"采用平台推荐的逻辑回归","confidence":"high",'
        '"explicit_authorization":true}'
    )

    out = route_instruction(
        fake,
        gate_context="选择实验",
        instruction="采用平台推荐的逻辑回归作为最终模型。",
        param_schema=[
            {
                "name": "selected_experiment_id",
                "type": "string",
                "current": selected_id,
                "enum": ["experiment_xgb", selected_id],
                "bounds": {"enum": ["experiment_xgb", selected_id]},
                "candidates": [
                    {
                        "experiment_id": selected_id,
                        "recipe": "lr_regressor",
                        "display_name": "逻辑回归（回归）",
                        "recommended": True,
                    }
                ],
            }
        ],
    )

    assert out["action"] == "confirm"
    assert out["params"] == {"selected_experiment_id": selected_id}
    assert len(fake.calls) == 1


def test_route_instruction_recovers_modeling_parameter_adjustment_from_replan():
    """A modeling setup request that only changes controls already declared by
    the current gate is an adjust, even when the first router pass calls it a
    structural replan.

    This reproduces the live UI wording used to request time OOT plus multiple
    algorithms.  The second pass must extract a governed, gate-scoped patch
    instead of sending the whole plan through best-effort replanning.
    """

    instruction = (
        "请先调整本次建模方案：按日期列 apply_month 做时间外推 OOT；"
        "同时使用 lr、xgb、lgb、catboost、scorecard、mlp，每种调参 1 轮；"
        "ensemble 界面显示不可用，请保留其不可用状态并记录原因。"
        "调整后等待我继续确认。"
    )
    fake = _SequencedLLM(
        [
            '{"action":"replan","params":{},"constraint":"重做建模方案","reason":"涉及切分和算法"}',
            (
                '{"action":"adjust","params":{'
                '"split_config":{"test_size":0.25,"oot_by_time":"apply_month","oot_size":0.2},'
                '"recipes":["lr","xgb","lgb","catboost","scorecard","mlp"],'
                '"n_trials":1},"constraint":"","reason":"调整当前建模设置"}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="特征筛选",
        instruction=instruction,
        param_schema=[
            {"name": "split_config", "type": "object", "current": {}},
            {"name": "recipes", "type": "array", "current": ["lgb"]},
            {"name": "n_trials", "type": "integer", "current": 1},
        ],
    )

    assert out["action"] == "adjust"
    assert out["params"] == {
        "split_config": {
            "test_size": 0.25,
            "oot_by_time": "apply_month",
            "oot_size": 0.2,
        },
        "recipes": ["lr", "xgb", "lgb", "catboost", "scorecard", "mlp"],
        "n_trials": 1,
    }
    assert len(fake.calls) == 2
    assert "当前节点参数调整" in fake.calls[1]["user_prompt"]


def test_route_instruction_completeness_reviews_partial_modeling_adjustment():
    """A first-pass adjust can still omit one of several controls named in the
    same sentence. The bounded modeling review must compare against the whole
    instruction and return a complete canonical patch.
    """

    instruction = (
        "按 apply_month 做时间 OOT，同时用 lr、xgb、lgb，每种调参 1 轮。"
    )
    fake = _SequencedLLM(
        [
            (
                '{"action":"adjust","params":{"recipes":["lr","xgb","lgb"],'
                '"n_trials":1},"constraint":"","reason":"调整算法和轮数"}'
            ),
            (
                '{"action":"adjust","params":{'
                '"split_col":"apply_month",'
                '"split_config":{"method":"time_outer","date_col":"apply_month"},'
                '"recipes":["lr","xgb","lgb"],"n_trials":1},'
                '"constraint":"","reason":"补全全部明确控件"}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="特征筛选",
        instruction=instruction,
        param_schema=[
            {"name": "split_col", "type": "string", "current": "split"},
            {"name": "split_config", "type": "object", "current": {}},
            {"name": "recipes", "type": "array", "current": ["lgb"]},
            {"name": "n_trials", "type": "integer", "current": 12},
        ],
    )

    assert out["action"] == "adjust"
    assert out["params"] == {
        "split_config": {"oot_by_time": "apply_month"},
        "recipes": ["lr", "xgb", "lgb"],
        "n_trials": 1,
    }
    assert len(fake.calls) == 2
    assert "不得遗漏" in fake.calls[1]["user_prompt"]


def test_route_instruction_normalizes_llm_split_ratio_aliases():
    fake = _SequencedLLM(
        [
            (
                '{"action":"adjust","params":{"split_config":{'
                '"random_oot":true,"test_ratio":0.2,"oot_ratio":0.2,'
                '"train_ratio":0.6}},"constraint":"",'
                '"reason":"按用户比例随机留出 OOT"}'
            ),
            (
                '{"action":"adjust","params":{"split_config":{'
                '"random_oot":true,"test_ratio":0.2,"oot_ratio":0.2,'
                '"train_ratio":0.6}},"constraint":"",'
                '"reason":"按用户比例随机留出 OOT"}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="选择建模规格",
        instruction="随机留出 OOT，test 占 20%，OOT 占 20%。",
        param_schema=[
            {"name": "split_config", "type": "object", "current": {}},
        ],
    )

    assert out["action"] == "adjust"
    assert out["params"] == {
        "split_config": {
            "random_oot": True,
            "test_size": 0.2,
            "oot_size": 0.2,
        }
    }


def test_route_instruction_canonicalizes_one_weight_candidate_as_selection():
    """The LLM owns the semantic decision; the router then canonicalizes the
    single selected column into the gate's writable sample_weight_col field.

    This reproduces the live UI failure where the router understood the named
    column but wrote it to the read-only diagnostic candidates list, leaving
    training unweighted.
    """

    fake = _SequencedLLM(
        [
            (
                '{"action":"adjust","params":{"sample_weight_candidates":["case_weight"]},'
                '"constraint":"","reason":"启用用户点名的样本权重列"}'
            ),
            (
                '{"action":"adjust","params":{"sample_weight_candidates":["case_weight"]},'
                '"constraint":"","reason":"启用用户点名的样本权重列"}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="选择建模规格",
        instruction="把样本权重列设置为 case_weight，其他规格保持不变。",
        param_schema=[
            {"name": "sample_weight_col", "type": "string", "current": ""},
            {
                "name": "sample_weight_candidates",
                "type": "array",
                "current": ["case_weight"],
            },
        ],
    )

    assert out["action"] == "adjust"
    assert out["params"] == {"sample_weight_col": "case_weight"}
    assert len(fake.calls) == 2
    assert "sample_weight_candidates 是只读诊断" in fake.calls[1]["user_prompt"]


def test_route_instruction_does_not_choose_between_multiple_weight_candidates():
    fake = _SequencedLLM(
        [
            (
                '{"action":"adjust","params":{'
                '"sample_weight_candidates":["weight_a","weight_b"]},'
                '"constraint":"","reason":"检测到多个候选"}'
            ),
            (
                '{"action":"adjust","params":{'
                '"sample_weight_candidates":["weight_a","weight_b"]},'
                '"constraint":"","reason":"检测到多个候选"}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="选择建模规格",
        instruction="给这次训练加上合适的样本权重。",
        param_schema=[
            {"name": "sample_weight_col", "type": "string", "current": ""},
            {
                "name": "sample_weight_candidates",
                "type": "array",
                "current": ["weight_a", "weight_b"],
            },
        ],
    )

    assert out["action"] == "clarify"
    assert out["params"] == {}
    assert "多个" in out["reason"]


def test_route_instruction_keeps_explicit_no_weight_selection():
    fake = _SequencedLLM(
        [
            (
                '{"action":"adjust","params":{"sample_weight_col":""},'
                '"constraint":"","reason":"用户明确不使用权重"}'
            ),
            (
                '{"action":"adjust","params":{"sample_weight_col":""},'
                '"constraint":"","reason":"用户明确不使用权重"}'
            ),
        ]
    )

    out = route_instruction(
        fake,
        gate_context="选择建模规格",
        instruction="这次不使用样本权重，其他规格保持不变。",
        param_schema=[
            {"name": "sample_weight_col", "type": "string", "current": "case_weight"},
            {
                "name": "sample_weight_candidates",
                "type": "array",
                "current": ["case_weight"],
            },
        ],
    )

    assert out["action"] == "adjust"
    assert out["params"] == {"sample_weight_col": ""}


def test_route_instruction_keeps_actual_step_structure_change_as_replan():
    fake = _FakeLLM(
        '{"action":"replan","params":{},"constraint":"删除调参步骤","reason":"改变步骤结构"}'
    )

    out = route_instruction(
        fake,
        gate_context="特征筛选",
        instruction="删除调参步骤，直接训练。",
        param_schema=[
            {"name": "recipes", "type": "array", "current": ["lgb"]},
            {"name": "n_trials", "type": "integer", "current": 1},
        ],
    )

    assert out["action"] == "replan"
    assert len(fake.calls) == 1
