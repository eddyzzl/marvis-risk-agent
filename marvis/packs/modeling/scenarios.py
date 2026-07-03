from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from marvis.packs.modeling.contracts import TrainConfig
from marvis.packs.modeling.errors import ModelingError


@dataclass(frozen=True)
class ScenarioTemplate:
    id: str
    description: str
    target_type: str
    default_recipe: str
    target_hint: str
    param_overrides: dict[str, Any]
    feature_guidance: str
    eval_metric: str
    notes: str


SCENARIO_TEMPLATES = {
    "loan_pre_a": ScenarioTemplate(
        "loan_pre_a",
        "贷前A卡（准入评分）",
        "binary",
        "scorecard",
        "首逾FPD/30+ @ mob",
        {"max_depth": 3},
        "强解释性、单调分箱、跨渠道稳定",
        "ks_auc",
        "评分卡优先；OOT KS 衰减需关注",
    ),
    "pre_screen": ScenarioTemplate(
        "pre_screen",
        "前筛（粗筛坏客户）",
        "binary",
        "lgb",
        "坏客户/拒绝后表现",
        {},
        "外部数据源覆盖率优先",
        "ks_auc",
        "高拒绝率场景，注意样本偏差（拒绝推断另议）",
    ),
    "loan_in": ScenarioTemplate(
        "loan_in",
        "贷中（行为评分B卡）",
        "binary",
        "lgb",
        "未来N期逾期",
        {},
        "行为/还款/支用类特征为主",
        "ks_auc",
        "观察期+表现期定义要清晰；账龄影响",
    ),
    "loan_post": ScenarioTemplate(
        "loan_post",
        "贷后（早期预警/催收前）",
        "binary",
        "lgb",
        "滚动恶化/进入M2+",
        {},
        "近期还款行为+逾期状态迁移",
        "ks_auc",
        "目标常是状态迁移，配合 roll_rate",
    ),
    "marketing": ScenarioTemplate(
        "marketing",
        "营销响应",
        "binary",
        "lgb",
        "是否响应/转化",
        {},
        "触达/活跃/历史响应特征",
        "response_lift",
        "目标是响应非风险；用 lift/响应率评估，不混淆 KS",
    ),
    "transaction": ScenarioTemplate(
        "transaction",
        "交易（反欺诈/异常）",
        "binary",
        "lgb",
        "欺诈/异常交易",
        {"scale_pos_weight": "auto"},
        "交易序列/设备/关系网特征",
        "ks_auc",
        "极度不平衡，关注 recall/精确率而非只看 KS",
    ),
    "recall": ScenarioTemplate(
        "recall",
        "捞回（流失/沉睡唤醒）",
        "binary",
        "lgb",
        "唤醒后用信/复借",
        {},
        "沉睡时长/历史用信/营销触达",
        "response_lift",
        "目标是被唤醒后的正向行为，类营销口径",
    ),
    "income": ScenarioTemplate(
        "income",
        "收入预测",
        "continuous",
        "lgb_regressor",
        "月收入/可支配收入(连续值)",
        {"objective": "regression"},
        "工资/公积金/交易流水/消费类特征",
        "rmse_mae",
        "回归任务：指标用 RMSE/MAE/R2，不是 KS/AUC；需 continuous recipe",
    ),
    "credit_limit": ScenarioTemplate(
        "credit_limit",
        "额度",
        "binary",
        "lgb",
        "提额后风险/用信意愿",
        {},
        "额度使用率/收入/风险分",
        "ks_auc",
        "常与额度策略联动（Phase 7）",
    ),
    "pricing": ScenarioTemplate(
        "pricing",
        "定价（风险定价）",
        "binary",
        "lgb",
        "风险分层支撑定价",
        {},
        "风险+价格敏感度特征",
        "ks_auc",
        "输出风险分供定价策略用（Phase 7）",
    ),
}


def get_scenario(scenario_id: str) -> ScenarioTemplate:
    return SCENARIO_TEMPLATES[scenario_id]


def list_scenarios() -> list[ScenarioTemplate]:
    return list(SCENARIO_TEMPLATES.values())


def apply_scenario(config: TrainConfig, scenario_id: str) -> TrainConfig:
    template = get_scenario(scenario_id)
    user_params = dict(config.params)
    recipe_override = (
        user_params.pop("recipe_id", None)
        or user_params.pop("recipe", None)
        or config.recipe_id
    )
    recipe_id = str(recipe_override or template.default_recipe)
    _assert_recipe_matches_target(recipe_id, template.target_type)
    return replace(
        config,
        params={**template.param_overrides, **user_params},
        recipe_id=recipe_id,
        scenario_id=template.id,
        target_type=template.target_type,
        eval_metric=template.eval_metric,
    )


def _assert_recipe_matches_target(recipe_id: str, target_type: str) -> None:
    is_regression_recipe = recipe_id.endswith("_regressor")
    is_multiclass_recipe = "multiclass" in recipe_id
    if target_type == "continuous":
        if not is_regression_recipe:
            raise ModelingError(
                f"continuous scenario requires a regression recipe, got: {recipe_id}"
            )
        return
    if target_type == "multiclass":
        if not is_multiclass_recipe:
            raise ModelingError(
                f"multiclass scenario requires a multiclass recipe, got: {recipe_id}"
            )
        return
    if target_type == "binary":
        if is_regression_recipe or is_multiclass_recipe:
            raise ModelingError(
                f"binary scenario requires a classification recipe, got: {recipe_id}"
            )
        return
    raise ModelingError(f"unsupported scenario target_type: {target_type}")


__all__ = [
    "SCENARIO_TEMPLATES",
    "ScenarioTemplate",
    "apply_scenario",
    "get_scenario",
    "list_scenarios",
]
