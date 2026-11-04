"""Outbound DTO for workflow insight producers; no rendered table prose."""
from marvis.llm_outbound import OutboundContextError, _mint_prompt
from marvis.llm_outbound_reports import _feature, _labels, _numeric, _rows
from marvis.llm_outbound_validation import _hyperparameters
from marvis.validation.overfitting import OVERFIT_TRAIN_OOT_ABS, OVERFIT_TRAIN_TEST_REL


_MODEL_RECIPES = frozenset({"lr", "lgb", "xgb", "catboost", "scorecard", "mlp",
    "lgb_regressor", "xgb_regressor", "lr_regressor", "mlp_regressor", "lgb_multiclass",
    "xgb_multiclass", "lr_multiclass", "mlp_multiclass"})
_MODEL_METRICS = ("train_ks", "test_ks", "oot_ks", "train_auc", "test_auc", "oot_auc",
                  "psi_oot_vs_train", "overfit_train_test_gap", "overfit_train_oot_gap")
# The interpretation is read by business users: platform state codes travel
# with their Chinese label, and measured values with the 4-decimal display
# precision the cards use. Stored metrics are untouched.
_FEATURE_STATE_LABELS = {"recommended": "推荐", "candidate": "候选", "not_recommended": "不推荐",
                         "caution": "慎用", "unevaluated": "未评估"}
_DISPLAY_DECIMALS = 4


def _display_rounded(value):
    if isinstance(value, float):
        return round(value, _DISPLAY_DECIMALS)
    if isinstance(value, dict):
        return {key: _display_rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_display_rounded(item) for item in value]
    return value


def _feature_state(row):
    state = _labels(row, {"state": set(_FEATURE_STATE_LABELS)})
    if state:
        state["state_label"] = _FEATURE_STATE_LABELS[state["state"]]
    return state


def workflow_insight_prompt(context):
    milestone = context.get("milestone")
    if milestone not in {"join_completed", "join_diagnosed", "feature_reported", "feature_analyzed",
                         "model_completed", "model_review"}:
        raise OutboundContextError("工作流分析缺少已知阶段，未发送请求。")
    source = context.get("measured_evidence")
    if not isinstance(source, dict) or not source:
        raise OutboundContextError("工作流分析缺少结构化度量，未将报告文字作为证据发送。")
    clean = {}
    if milestone.startswith("join_"):
        clean.update(_numeric(source, ("anchor_rows", "joined_rows")))
        clean["join_tables"] = [{**_numeric(row, ("table_index", "match_rate", "missing_rate")),
                                **{key: row[key] for key in ("key_unique", "row_inflation")
                                   if type(row.get(key)) is bool}}
                               for row in _rows(source.get("join_tables"))]
    elif milestone.startswith("feature_"):
        clean.update(_numeric(source, ("feature_count",)))
        clean["features"] = [{**_feature(row), **_feature_state(row)}
                             for row in _rows(source.get("features")) if _feature(row)]
        clean["quality"] = [{**_feature(row), **_numeric(row, ("missing_rate", "single_value_rate", "zero_rate"))}
                            for row in _rows(source.get("quality")) if _feature(row)]
    else:
        clean.update(_labels(source, {
            "recipe": _MODEL_RECIPES,
            "evidence_basis": {"pre_refit_heldout", "selected_experiment", "delivered_experiment",
                               "delivered_metrics", "candidate"},
        }))
        if clean.get("recipe"):
            # The algorithm's name next to its code (eval S4 read 「recipe=lr」).
            from marvis.agent.presenters.display_labels import recipe_label

            clean["recipe_label"] = recipe_label(clean["recipe"])
        metrics = source.get("metrics")
        clean["metrics"] = _numeric(metrics if isinstance(metrics, dict) else {}, _MODEL_METRICS)
        if type(source.get("overfit_flag")) is bool:
            clean["overfit_flag"] = source["overfit_flag"]
        goal = source.get("goal")
        if isinstance(goal, dict) and goal.get("metric") == "oot_ks" and _numeric(goal, ("min",)).get("min") is not None:
            clean["goal"] = {"metric": "oot_ks", "min": goal["min"],
                             "met": goal["met"] if type(goal.get("met")) is bool else None}
        clean["candidates"] = [
            {**_labels(row, {"recipe": _MODEL_RECIPES}),
             "recommended": row.get("recommended") is True,
             "metrics": _numeric(row.get("metrics") if isinstance(row.get("metrics"), dict) else {}, _MODEL_METRICS),
             **({"overfit_flag": row["overfit_flag"]} if type(row.get("overfit_flag")) is bool else {})}
            for row in _rows(source.get("candidates"))[:8]
        ]
        clean["parameters"] = _hyperparameters(source.get("parameters"))
        flags = source.get("risk_flags")
        clean["risk_flags"] = [item for item in flags if isinstance(item, str) and item in {
            "train_test_ks_gap", "test_oot_ks_gap", "oot_ks_unavailable", "overfit_flag",
            "oot_ks_goal_unmet", "oot_ks_goal_unverifiable",
        }] if isinstance(flags, list) else []
    # Hyperparameters keep their exact values; measured metrics use display precision.
    clean = {key: (value if key == "parameters" else _display_rounded(value)) for key, value in clean.items()}
    # Each workflow gets only its own reading rules: one shared instruction made
    # feature analysis say 「平台未提供 OOT KS 目标、过拟合标记、重复键…」
    # (eval S3) — modeling and join checks that never apply there.
    if milestone.startswith("join_"):
        specific = (
            "拼接表序号仅用于本次结果，不要推测文件名或机构名。"
            "join_tables 就是本次全部拼接表，不要推测还有其他表。"
            "拼接后行数等于锚点行数即说明没有行数膨胀。"
            "某项度量没有给出时一句话说明即可，不要把它展开成多条风险和建议。"
            "解释低命中率、重复键和行数膨胀的风险。"
        )
    elif milestone.startswith("feature_"):
        specific = (
            "这是单表特征分析，不涉及模型训练、OOT 目标、候选模型或拼接；不要提这些没有给出的内容。"
            "特征建议状态由平台提供；区分推荐、候选、不推荐、慎用和未评估，不能把未评估写成推荐。"
            "mode_rate/最常见值占比是最常见取值所占比例：越高越集中，很低说明取值分散。"
            "解释缺失率、取值集中、PSI 稳定性和疑似泄漏（单变量区分度异常高）的风险。"
        )
    else:
        specific = (
            "evidence_basis=pre_refit_heldout 表示指标来自所选实验的留出集，交付模型在 train+test 上重新拟合，"
            "应据此评价而不是称指标缺失；文字里写「所选实验的留出集」，不要写 pre_refit_heldout 等代码。"
            "overfit_flag=true 必须在 summary 中点明，写成「平台标记了过拟合风险」，不要写 overfit_flag 这个键名。"
            f"平台规则：overfit_train_test_gap（训练/测试 KS 相对差距）超过 {OVERFIT_TRAIN_TEST_REL:.2f}，"
            f"或 overfit_train_oot_gap（训练/OOT KS 差距）超过 {OVERFIT_TRAIN_OOT_ABS:.2f}，即标记过拟合风险；"
            "说明是哪一项触发的。OOT KS 高于训练集时，这更像 OOT 样本差异或样本量小带来的波动，不要说成模型记住了训练集。"
            "goal 给出用户的 OOT KS 目标及平台判定的是否达成，summary 必须说明；"
            "goal 缺失表示这次没有设置 OOT KS 目标，直接说明未设置，不要说未在输入中给出。"
            "candidates 有多个时逐一比较；为空时不要提候选比较，也不要建议增加候选。"
            "提到算法时用 recipe_label 的名称，不要写 recipe= 或英文键名。"
            "解释过拟合和样本外表现的风险。"
        )
    if not milestone.startswith("join_"):
        # The platform's own stability bands (gate_payloads._stability_label
        # and packs.feature.tools._feature_recommendation share them);
        # eval S4 read PSI 0.0672 as 「分布存在一定差异」.
        specific += (
            "PSI 按平台口径解读：低于 0.10 为稳定，0.10 及以上、低于 0.25 需关注，"
            "0.25 及以上为明显漂移（恰好 0.10 算需关注，恰好 0.25 算明显漂移）。"
        )
    from marvis.agent.suggested_reply import SUGGESTED_REPLY_INSTRUCTION

    # UI-6: whether the platform now waits on the user decides the suggestion.
    awaiting = context.get("awaiting_confirmation") is True
    return _mint_prompt({"milestone": milestone, "measured_evidence": clean,
                         "awaiting_confirmation": awaiting, "instruction": (
        "只解释已给出的平台度量。缺失值代表未知，不能推定为通过。"
        + specific
        + "提到状态时使用 state_label 的中文写法，不要写英文状态码；数值已按展示精度保留 4 位小数，引用时照用，不要扩展位数。"
        # A rule, not a fact to report: eval S4 turned it into a finding, a
        # risk and a recommendation about 「缺少报告原文」.
        "不要声称做过历史对比；也不要在结论里提到报告原文、原始记录或历史记忆没有提供。"
        "返回 JSON，含 summary、findings、risks、recommendations，以及可选的 suggested_reply 字符串。"
        + SUGGESTED_REPLY_INSTRUCTION
        + "awaiting_confirmation=true 表示平台正在等待用户确认。"
    )})
