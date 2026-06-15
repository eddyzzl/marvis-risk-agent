from __future__ import annotations

import re


DEFAULT_ALGORITHM = "lgb"
SUPPORTED_ALGORITHM_TEXT = "xgb, lgb, lr, catboost, scorecard, dnn"

ALGORITHM_LABELS: dict[str, str] = {
    "xgb": "XGBoost",
    "lgb": "LightGBM",
    "lr": "逻辑回归",
    "catboost": "CatBoost",
    "scorecard": "评分卡",
    "dnn": "DNN",
}

ALLOWED_ALGORITHMS = frozenset(ALGORITHM_LABELS)

_ALGORITHM_ALIASES = {
    "xgb": "xgb",
    "xgboost": "xgb",
    "xgbclassifier": "xgb",
    "xgboostclassifier": "xgb",
    "xgboostxgbclassifier": "xgb",
    "xgboostsklearnxgbclassifier": "xgb",
    "lgb": "lgb",
    "lgbm": "lgb",
    "lightgbm": "lgb",
    "lighgbm": "lgb",
    "lgbclassifier": "lgb",
    "lgbmclassifier": "lgb",
    "lightgbmclassifier": "lgb",
    "lightgbmsklearnlgbmclassifier": "lgb",
    "lr": "lr",
    "logit": "lr",
    "logistic": "lr",
    "logisticregression": "lr",
    "logisticregressioncv": "lr",
    "sklearnlinearmodellogisticregression": "lr",
    "逻辑回归": "lr",
    "cat": "catboost",
    "catboost": "catboost",
    "catboostclassifier": "catboost",
    "catboostcatboostclassifier": "catboost",
    "scorecard": "scorecard",
    "scorecards": "scorecard",
    "评分卡": "scorecard",
    "记分卡": "scorecard",
    "dnn": "dnn",
    "deeplearning": "dnn",
    "deepneuralnetwork": "dnn",
    "neuralnetwork": "dnn",
    "mlp": "dnn",
    "mlpclassifier": "dnn",
    "神经网络": "dnn",
    "深度神经网络": "dnn",
}

MODEL_TRAINING_DESCRIPTIONS: dict[str, str] = {
    "xgb": (
        "XGBoost 是梯度提升树集成算法，以多棵 CART 树逐轮拟合前序模型残差或梯度，"
        "持续提升对违约概率的排序能力。它支持正则化、列采样、行采样、学习率收缩、"
        "缺失值默认方向和树剪枝，能在变量较多、非线性关系明显、特征交互复杂的信贷"
        "风控场景中保持较强拟合能力。验证时需重点关注训练集与测试集 KS、AUC 差异，"
        "OOT 稳定性、变量重要性集中度、单调性合理性、缺失和极端值敏感性，以及参数"
        "设置是否存在过深树、过高学习率或过度迭代导致的过拟合风险。报告撰写时应说明"
        "模型由多轮弱学习器叠加形成风险排序，不宜将树模型解释为单一变量因果关系，并应"
        "结合样本变化、监控口径和投产约束判断验证结论，避免单点指标决策。"
    ),
    "lgb": (
        "LightGBM 是基于梯度提升树的高效集成算法，采用直方图分裂、叶子优先生长和"
        "特征并行等机制，在样本量较大、变量维度较高的信贷风控建模中能兼顾训练效率"
        "和区分能力。模型通过连续迭代拟合损失函数梯度，自动捕捉非线性关系、变量交互"
        "和局部风险差异。验证时需关注训练集、测试集和 OOT 的 KS、AUC、PSI 表现，"
        "检查叶子数、学习率、最小样本数、迭代轮数等参数是否合理，评估变量重要性是否"
        "过度集中，并结合分箱、缺失值和压力测试确认模型在客群迁移、数据漂移和异常输入"
        "下仍保持稳健。报告撰写时应说明该算法偏向自动学习复杂分裂结构，结论应同时依赖"
        "样本外表现、稳定性和业务可解释性，保持审慎，并关注投产一致性。"
    ),
    "lr": (
        "逻辑回归是信贷风控中常用的二分类线性模型，通过特征加权求和并经 Logit 函数"
        "转换为违约概率，模型结构清晰、可解释性强，适合评分卡、准入策略和基准模型建设。"
        "其优势在于变量方向、权重大小、边际影响和策略含义容易审查，也便于与业务规则、"
        "单调约束和分箱结果结合。验证时需关注样本代表性、变量筛选依据、多重共线性、"
        "系数方向、显著性、训练集与测试集 KS/AUC 差异、OOT 稳定性、PSI 漂移、校准"
        "表现和分数分布。同时应检查缺失值、极端值、类别合并及变量标准化处理是否与投产"
        "链路一致，避免口径偏差。报告撰写时应说明其线性假设和变量处理边界，结论应结合"
        "稳定性、校准性和业务可解释性共同判断，注意线性边界。"
    ),
    "catboost": (
        "CatBoost 是梯度提升树集成算法，重点优化类别特征处理和有序 boosting，能够在"
        "变量中包含较多枚举、渠道、地区或行为类别字段时减少手工编码负担。它通过多棵树"
        "逐轮拟合损失梯度，自动学习非线性关系和特征交互。验证时需关注训练、测试和 OOT "
        "样本的 KS/AUC、PSI、类别取值漂移、缺失值处理、树深、学习率和迭代轮数，并检查"
        "投产链路中的类别编码、未知类别和模型导出格式是否与开发环境一致。报告撰写时应"
        "说明模型依赖树集成结构和类别特征处理机制，结论需结合稳定性、解释性和投产一致性"
        "共同判断。"
    ),
    "scorecard": (
        "评分卡通常以分箱后的变量、WOE 转换和逻辑回归系数为基础，将违约概率映射为便于"
        "业务使用的分数体系。它强调变量方向、分箱单调性、分数贡献和策略解释，适用于准入、"
        "额度、定价或贷后预警等需要稳定解释的场景。验证时需重点检查分箱样本量、坏账率"
        "单调性、WOE/IV、系数方向、训练测试/OOT 的 KS 和 AUC、分数分布、PSI、拒绝阈值"
        "敏感性，以及投产评分公式和分箱边界是否与开发版本一致。报告应说明评分卡的线性"
        "假设和分箱口径，避免只凭单一排序指标判断有效性。"
    ),
    "dnn": (
        "DNN 是多层神经网络模型，通过若干隐藏层和非线性激活函数学习复杂的变量组合关系，"
        "适合特征维度较高、交互关系明显或传统线性模型表达不足的场景。其可解释性通常弱于"
        "树模型和评分卡，验证时需更关注样本外 KS/AUC、OOT 稳定性、校准表现、训练过程、"
        "正则化、早停、特征标准化、缺失值处理、随机种子和模型版本一致性。还应检查输入"
        "张量构造、预处理流水线、阈值选择和投产推理环境，避免训练推理口径不一致。报告"
        "撰写时应明确其黑盒属性，并结合稳定性、压力测试和监控要求给出审慎结论。"
    ),
}


def normalize_algorithm(value: str | None, *, allow_empty: bool = False) -> str:
    key = str(value or "").strip()
    if not key:
        if allow_empty:
            return ""
        raise ValueError(
            f"model algorithm is required; supported algorithms: {SUPPORTED_ALGORITHM_TEXT}"
        )
    if key in ALLOWED_ALGORITHMS:
        return key
    normalized = _algorithm_key(key)
    if normalized in _ALGORITHM_ALIASES:
        return _ALGORITHM_ALIASES[normalized]
    raise ValueError(
        f"unsupported model algorithm; supported algorithms: {SUPPORTED_ALGORITHM_TEXT}"
    )


def model_training_description(algorithm: str | None) -> str:
    normalized = normalize_algorithm(algorithm)
    return MODEL_TRAINING_DESCRIPTIONS[normalized]


def _algorithm_key(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)
