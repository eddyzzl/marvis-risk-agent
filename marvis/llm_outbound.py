"""Reconstruct external LLM context from an explicit structural allowlist.

Unknown fields and free text are withheld, rather than scanned for particular
customers or file formats. Raw local artifacts remain available to deterministic
tools and the local UI. This module never opens a business file.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field as dataclass_field


POLICY_VERSION = "marvis.llm_structural_context.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_REFERENCE = re.compile(r"^\$ref:[A-Za-z0-9_-]{1,96}\.output(?:\.[A-Za-z_][A-Za-z0-9_]{0,95}){1,8}$")
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,95}$")
_HASH = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_MAX_DEPTH = 12
_DROP = object()
_PROMPT_TOKEN = object()


class OutboundContextError(ValueError):
    pass


class StructuralPrompt(str):
    """An in-process capability minted only by the typed context builders."""

    def __new__(cls, value, *, _token=None):
        if _token is not _PROMPT_TOKEN:
            raise OutboundContextError("LLM context must be built from approved structure")
        return super().__new__(cls, value)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Copying an existing capability must not reconstruct it without its
        # authority token, nor mint authority from an ordinary string.
        memo[id(self)] = self
        return self


class SafeInstruction(str):
    def __new__(cls, value, *, _token=None):
        if _token is not _PROMPT_TOKEN:
            raise OutboundContextError("instruction is not an approved control expression")
        return super().__new__(cls, value)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


_CONTROL_PHRASES = frozenset({
    "继续", "确认", "开始", "停止", "取消", "同意", "拒绝", "禁止继续", "不要继续", "先不要继续",
    "立即继续", "可以继续", "请继续", "按当前方案继续", "立即按当前方案继续", "按当前配置继续",
    "确认并继续", "确认继续", "确认执行", "开始执行", "暂停执行", "先暂停", "不要执行", "先不要执行",
    "继续执行", "执行当前方案", "请按当前方案继续", "不要修改参数", "保持当前参数",
    "continue", "confirm", "stop", "cancel", "do not continue", "proceed", "do not execute",
})

# A lexical data boundary, not an intent classifier: complete coverage is
# required and negation, conditions, questions and word order remain verbatim.
_CONTROL_WORDS = frozenset("""
确认 继续 开始 停止 取消 同意 拒绝 禁止 不要 不 不可 立即 可以 请 按 当前 方案 配置
并 执行 暂停 先 保持 参数 修改 采用 平台 标星 推荐 的 实验 作为 主实验 最终 模型
转入 进入 后续 下一 剩余 完成 步骤 这版 筛选 结果 符合 预期 接着 特征 筛选完成
这些 这个 这次 本次 就 口径 汇总 数据 稀疏 风险 也 保留 在 报告 里 是不是 是否
如果 核验 通过 再 但 新增 未经 结论 删掉 其余 弱化 改写 成 无影响 停 这里
看起来 还行 没问题 有问题 不确定 调整 建模 调参 节点 选择 规格 算法 组合
逻辑回归 多分类 二分类 回归 集成 神经网络 评分卡 决策树 随机森林
随机 留出 时间 外推 日期 日期列 列 同时 使用 用 每种 轮 轮数
界面 显示 不可用 其 状态 记录 原因 后 等待 我 其他 不变
样本 权重 样本权重 样本权重列 设置 为 给 加上 合适 删除 直接 训练
阈值 放宽 到 调到 改为 占 改成 增加 减少 重排 移除 跳过 工作流 候选
组合 会 原样 已 展示 总览 把 与 阶段
比较 对比 解释 为什么 怎样 什么 影响 分析 检查 验证 正常 异常
continue confirm stop cancel do not proceed execute please if then otherwise
train test oot lr xgb lgb catboost scorecard mlp ensemble lightgbm xgboost
""".split())
_CONTROL_PUNCTUATION = re.compile(r"[\s,，。.!！?？;；:：、()（）“”‘’\"']+")
_NUMERIC_CONTROL = re.compile(
    r"(?:n_trials|num_boost_round|seed|leakage_ks|max_missing_rate|iv_min|corr_max|"
    r"调参轮数|调参|阈值|test|oot|train)\s*(?:调到|设置为|改为|改成|放宽到|到|占|=)?\s*"
    r"(?:0|[1-9]\d{0,5})(?:\.\d{1,8})?%?(?![A-Za-z0-9_.])", re.IGNORECASE,
)
_COLUMN_CONTROL = re.compile(
    r"按\s+(?:日期列\s+)?[A-Za-z_][A-Za-z0-9_]{0,127}\s+做\s*时间"
    r"|按日期列\s+[A-Za-z_][A-Za-z0-9_]{0,127}\s+做\s*时间"
)


def control_instruction(value, *, allowed_identifiers=()):
    """Validate a complete control utterance; never extract a record from prose.

    Registered field/experiment identifiers and bounded numeric control
    expressions may accompany control language. Arbitrary literals, record
    syntax and incomplete coverage fail closed before a request exists.
    """
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise OutboundContextError("指令不能安全转换为控制结构，请使用当前节点的结构化控件。")
    if value.strip().rstrip("。.!！") in _CONTROL_PHRASES:
        return SafeInstruction(value.strip(), _token=_PROMPT_TOKEN)
    tokens = set(_CONTROL_WORDS)
    tokens.update(item for item in allowed_identifiers if _identifier(item, opaque=True) is not _DROP)
    ordered = sorted(tokens, key=len, reverse=True)
    offset = 0
    while offset < len(value):
        match = (_CONTROL_PUNCTUATION.match(value, offset) or _NUMERIC_CONTROL.match(value, offset)
                 or _COLUMN_CONTROL.match(value, offset))
        if match:
            offset = match.end()
            continue
        found = next((word for word in ordered if value[offset:offset + len(word)].casefold() == word.casefold()
            and (not word[-1].isascii() or offset + len(word) == len(value)
                 or not re.match(r"[A-Za-z0-9_-]", value[offset + len(word)]))), None)
        if found is None:
            raise OutboundContextError("指令含未批准的文字或数值，未发送；请使用当前节点的结构化控件。")
        offset += len(found)
    return SafeInstruction(value.strip(), _token=_PROMPT_TOKEN)

# These keys may carry raw data, prose copied from files, or exception messages.
# They are withheld even when nested beneath an otherwise approved container.
_PRIVATE_KEYS = frozenset({
    "sample_values", "sample_rows", "preview_rows", "rows", "records", "data",
    "values", "head", "tail", "preview", "content", "source", "text", "body",
    "file", "files", "file_content", "file_contents", "raw", "raw_output",
    "raw_response", "traceback", "exception", "error", "error_message", "error_value",
    "last_error", "reason", "reasons", "warnings", "summary", "description", "notes",
    "memory_anchor", "dictionary", "source_code", "code", "notebook", "notebook_source",
    "model", "model_content", "model_bytes", "pmml", "pkl", "pickle", "payload",
    "path", "paths", "source_dir", "filename", "file_name", "model_path", "api_key",
    "connection_string", "credentials", "password", "secret", "token", "url",
})
_CONTAINERS = frozenset({
    "task_context", "memory_context", "observation", "output", "output_summary", "metadata",
    "available_tools", "tools", "datasets", "dataset", "columns", "column_profiles",
    "feature_profiles", "schema", "metrics", "best_metrics", "metrics_summary", "parameters",
    "params", "best_params", "inputs", "controls", "control_values", "modeling_setup",
    "split_summary", "split_counts", "split_values", "gate", "screen", "dedup", "selection",
    "experiment", "experiments", "steps", "remaining_steps", "completed_steps", "plan",
    "planner_constraints", "required_tools", "allowed_tools", "literal_inputs",
    "business_acceptance", "target", "evidence", "criteria", "training", "validation",
    "dataset_split", "stability", "sample_analysis", "vintage", "feature_importance",
    "score_bands", "calibration", "univariate", "stress_product_removal", "stress_low_pricing",
    "structured_summary", "report", "sections", "post_checks", "spec", "range", "counts",
    "items", "input_fields", "output_fields", "risk_flags", "allowed_actions", "dedup_strategies",
    "sample_weight_diagnostics", "downstream_reset_policy", "runtime", "budget",
    "required_tool_refs", "forbidden_tool_refs", "required_literal_inputs", "tool_ref",
    "granted_tools", "policy", "bounds", "retry_policy", "effect_target", "safety", "workflow", "split_config",
    "gate_context", "proposed_params", "param_schema",
})
_IDENTIFIER_KEYS = frozenset({
    "column", "feature", "target_col", "split_col", "time_col", "date_col", "group_col",
    "sample_weight_col", "oot_by_time", "field", "plugin", "tool", "recipe", "recipe_id", "algorithm",
    "method", "metric", "eval_metric", "aggregation", "dtype", "type", "target_type",
    "status", "assurance", "stage", "effect_stage", "mode", "novel_mode", "scope", "kind",
    "action", "rule", "operator", "determinism", "family", "role", "search_stage",
    "human_decision_gate", "effect_authorization", "id_input", "result_status", "strategy_type",
    "workflow_family", "entrypoint",
})
_IDENTIFIER_LISTS = frozenset({
    "features", "feature_cols", "selected", "candidates", "recipes", "group_cols", "valid_group_cols",
    "required_inputs", "required", "sample_weight_candidates", "allowed_actions", "risk_flags",
    "request_topics", "depends_on", "step_ids",
})
_ID_KEYS = frozenset({
    "task_id", "dataset_id", "source_dataset_id", "result_dataset_id", "anchor_id",
    "experiment_id", "artifact_id", "plan_id", "step_id", "best_experiment_id",
    "target_step_id", "selected_experiment_id",
})
_BOOL_KEYS = frozenset({
    "ok", "passed", "valid", "enabled", "confirmed", "required", "nullable", "unique",
    "labels_mature", "drop_nan_labels", "goal_met", "goal_doubt", "execution_completed",
    "tune_enabled", "needs_confirmation", "has_target", "has_time", "redacted_unstructured",
    "requires_confirmation", "safety_critical", "decision_point", "random_oot",
})
_NUMERIC_KEYS = frozenset({
    "row_count", "column_count", "feature_count", "step_count", "count", "n", "n_rows",
    "train_count", "test_count", "oot_count", "bad_count", "good_count", "missing_count",
    "missing_rate", "null_rate", "unique_count", "bad_rate", "positive_rate", "positive_count",
    "train", "test", "oot", "seed", "n_trials", "total_n_trials", "cv_folds", "max_steps",
    "early_stopping_rounds", "max_boost_round", "num_boost_round", "n_estimators", "iterations",
    "max_depth", "learning_rate", "num_leaves", "min_child_samples", "min_child_weight",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "scale_pos_weight", "C",
    "top_k", "leakage_ks", "max_missing_rate", "iv_min", "corr_max", "vif_max", "overfit_penalty",
    "importance", "confidence", "ks", "auc", "psi", "csi", "iv", "rmse", "mae", "r2",
    "logloss", "accuracy", "precision", "recall", "f1", "lift", "gini", "mape", "brier",
    "selection_score", "score_std", "test_ks_std", "cv_score_std", "best_iteration",
    "coverage", "coverage_rate", "match_rate", "duplicate_rate", "row_retention", "criteria_count",
    "max_tokens", "max_trials", "max_recipes", "num_threads", "n_jobs", "thread_count",
    "red_flag_count", "measured",
    "workspace_revision", "analysis_generation", "dataset_revision", "min", "max", "test_size", "oot_size",
})
_METRIC = re.compile(r"^(?:(?:weighted|train|test|oot|valid|cv|mean|std|macro|micro)_){0,3}(?:ks|auc|psi|csi|iv|rmse|mae|r2|logloss|accuracy|precision|recall|f1|lift|gini|mape|brier)$")
_SAFE_ENUMS = frozenset({
    "object", "array", "string", "number", "integer", "boolean", "null", "binary", "continuous",
    "multiclass", "train", "test", "oot", "confirm", "adjust", "replan", "halt", "cancel",
    "skip", "retry", "pending", "done", "failed", "blocked", "passed", "unknown", "unavailable",
    "lgb", "xgb", "lr", "catboost", "scorecard", "mlp", "lgb_regressor", "xgb_regressor",
    "lr_regressor", "mlp_regressor", "lgb_multiclass", "xgb_multiclass", "lr_multiclass",
    "mlp_multiclass", "left", "inner", "outer", "right", "auto", "all", "any", "none",
    "random", "time", "group", "stratified", "mean", "median", "min", "max", "sum", "count",
    "first", "last", "error", "drop", "mask", "retain", "safe", "agent", "manual",
    "authorize", "reject", "ambiguous", "clarify", "high", "medium", "low",
    "required", "draft", "validated", "confirmed", "running", "awaiting_confirm", "review",
    "checking", "skipped", "cancelled", "partial", "complete", "completed", "ok",
    "plan_ahead", "explore", "deterministic", "seeded", "stochastic", "nondeterministic",
    "adopted", "proposed", "queued", "approval", "limit", "pricing", "fixed", "adaptive",
})
_FIXED_ENUM_FIELDS = {
    "status", "type", "target_type", "mode", "novel_mode", "action", "human_decision_gate",
    "effect_authorization", "determinism", "assurance", "result_status", "strategy_type", "workflow_family",
}
_TOPICS = {
    "modeling": ("建模", "训练", "model", "train"),
    "validation": ("验证", "validation"), "comparison": ("比较", "对比", "compare", "baseline"),
    "feature_analysis": ("特征", "feature"), "data_preparation": ("数据", "样本", "data", "sample"),
    "time_split": ("时间", "时序", "time", "oot"), "group_isolation": ("主体", "分组", "group"),
    "monitoring": ("监控", "monitor"), "report": ("报告", "report"),
    "strategy": ("策略", "strategy"), "join": ("拼接", "关联", "join"),
}


def _identifier(value, *, opaque=False):
    pattern = _OPAQUE_ID if opaque else _IDENTIFIER
    return value if isinstance(value, str) and pattern.fullmatch(value) else _DROP


def _finite(value):
    return type(value) in (int, float) and abs(value) <= 10**15 and math.isfinite(value)


def control_identifiers(param_schema=(), proposed_params=None):
    """Names come only from declared controls, never from pasted instruction text."""
    result = set()
    for spec in param_schema or ():
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if _identifier(name) is not _DROP:
            result.add(name)
        if name in {"sample_weight_col", "sample_weight_candidates", "features", "group_cols", "time_col", "date_col"}:
            current = spec.get("current")
            values = current if isinstance(current, list) else [current]
            result.update(value for value in values if _identifier(value) is not _DROP)
        if name == "selected_experiment_id":
            result.update(value for value in spec.get("enum", []) if _experiment_id(value))
    if isinstance(proposed_params, dict) and _experiment_id(proposed_params.get("selected_experiment_id")):
        result.add(proposed_params["selected_experiment_id"])
    return result


def _experiment_id(value):
    return isinstance(value, str) and re.fullmatch(r"experiment[-_][A-Za-z0-9_-]{1,96}", value) is not None


def gate_instruction_prompt(gate_context, instruction, *, param_schema=(), proposed_params=None):
    from marvis.llm_outbound_instructions import user_instruction
    from marvis.llm_outbound_gate_context import ApprovedGateContext
    if not isinstance(gate_context, (str, dict, ApprovedGateContext)):
        raise OutboundContextError("gate context must be an explicit structure")
    identifiers = control_identifiers(param_schema, proposed_params)
    safe_instruction = user_instruction(instruction, allowed_identifiers=identifiers)
    context = structural_summary(gate_context) if isinstance(gate_context, dict) else {}
    if isinstance(gate_context, ApprovedGateContext):
        context = gate_context.payload()
    if isinstance(gate_context, str):
        try:
            context = str(control_instruction(gate_context, allowed_identifiers=identifiers))
        except OutboundContextError:
            pass  # Unclassified UI prose is local only; the full instruction is still required.
    return _mint_prompt({"instruction": str(safe_instruction), "gate_context": context,
        "param_schema": _gate_parameters(param_schema),
        "proposed_params": structural_summary(proposed_params or {})})


def gate_review_prompt(prompt, *, kind, first_pass=None):
    if not isinstance(prompt, StructuralPrompt):
        raise OutboundContextError("gate review requires approved context")
    instructions = {
        "retry": "上一次返回无法解析；请按原始结构契约重新回答，不推测缺失事实。",
        "parameters": (
            "当前节点参数调整复核：对照完整用户指令和 param_schema，逐项抽取所有明确控件，不得遗漏。"
            "算法组合、调参轮数、目标类型、样本权重和样本切分属于 adjust；仅新增、删除、重排步骤属于 replan。"
            "明确不可用、不要或不选择的算法不得放进 recipes。时间 OOT 用 split_config.oot_by_time；"
            "随机 OOT 用 split_config.random_oot=true；无 OOT 不设置两项。"
            "sample_weight_candidates 是只读诊断，不能写入 params。明确使用某列返回 sample_weight_col；"
            "明确不使用权重返回空字符串，有多个候选而没有选择时 clarify。不得猜列名或比例。"
        ),
        "selection": (
            "候选选择节点语义复核：仅在名称或 recipe 唯一匹配某个候选，且用户明确授权采用时 confirm；"
            "params 只返回枚举内的 selected_experiment_id，confidence=high、explicit_authorization=true。"
            "候选 id、recipe、recommended 是已批准的结构，私有显示名未发送。"
            "同名多候选、仅询问或评价、否定、条件未满足均不得当作采用授权；不明确时 clarify。"
            "候选选择不是重写 DAG，只有新增、删除、重排步骤或切换 Workflow 才是 replan。"
        ),
    }
    if kind not in instructions:
        raise OutboundContextError("unknown gate review")
    payload = json.loads(prompt)
    payload["review_instruction"] = instructions[kind]
    if isinstance(first_pass, dict):
        payload["first_pass"] = structural_summary(first_pass)
    return _mint_prompt(payload)


def _gate_parameters(specs):
    out = []
    for spec in specs or ():
        if not isinstance(spec, dict) or _identifier(spec.get("name")) is _DROP:
            continue
        name = spec["name"]
        item = {"name": name, **safe_schema({"type": spec.get("type")})}
        value = spec.get("current")
        if (name in _NUMERIC_KEYS or name == "bins") and _finite(value):
            item["current"] = value
        elif name in {"recipes", "target_type"}:
            values = value if isinstance(value, list) else [value]
            valid = [entry for entry in values if isinstance(entry, str) and entry in _SAFE_ENUMS]
            if valid:
                item["current"] = valid if isinstance(value, list) else valid[0]
        elif name in {"sample_weight_col", "sample_weight_candidates"}:
            values = value if isinstance(value, list) else [value]
            valid = [entry for entry in values if entry == "" or _identifier(entry) is not _DROP]
            if valid:
                item["current"] = valid if isinstance(value, list) else valid[0]
        elif name == "split_config" and isinstance(value, dict):
            from marvis.llm_outbound_gates import split_configuration

            item["current"] = split_configuration(value)
        if name == "split_config":
            from marvis.modeling_split_schema import split_configuration_schema, split_configuration_semantics

            # Only the installed contract is admissible. Never project schema
            # annotations or arbitrary properties supplied by a caller.
            item["schema"] = split_configuration_schema()
            item["semantics"] = split_configuration_semantics()
        elif name in {"features", "selection"} and isinstance(value, list):
            item["current"] = [entry for entry in value if _identifier(entry) is not _DROP]
        elif name == "selected_experiment_id":
            allowed = [entry for entry in spec.get("enum", []) if _experiment_id(entry)]
            item["enum"] = allowed
            if value in allowed:
                item["current"] = value
            item["candidates"] = []
            for candidate in spec.get("candidates", []):
                if not isinstance(candidate, dict) or candidate.get("experiment_id") not in allowed:
                    continue
                row = {"experiment_id": candidate["experiment_id"]}
                if isinstance(candidate.get("recipe"), str) and candidate["recipe"] in _SAFE_ENUMS:
                    row["recipe"] = candidate["recipe"]
                    base, _, family = candidate["recipe"].partition("_")
                    label = {"lgb": "LightGBM", "xgb": "XGBoost", "lr": "逻辑回归",
                             "mlp": "MLP", "catboost": "CatBoost", "scorecard": "评分卡"}.get(base)
                    if label:
                        row["display_name"] = label + {"regressor": "（回归）", "multiclass": "（多分类）"}.get(family, "")
                if type(candidate.get("recommended")) is bool:
                    row["recommended"] = candidate["recommended"]
                item["candidates"].append(row)
        bounds = spec.get("bounds")
        if isinstance(bounds, dict):
            item["bounds"] = {key: bounds[key] for key in ("min", "max")
                              if key in bounds and _finite(bounds[key])}
            if name in {"recipes", "target_type"} and isinstance(bounds.get("enum"), list):
                item["bounds"]["enum"] = [entry for entry in bounds["enum"]
                                          if isinstance(entry, str) and entry in _SAFE_ENUMS]
        out.append(item)
    return out


def structural_summary(value, *, _depth=0, _context=""):
    """Allow only named structural fields and deterministic aggregate metrics."""
    if _depth > _MAX_DEPTH:
        raise OutboundContextError("结构深度超限，未截断执行约束或发送请求。")
    if not isinstance(value, dict):
        return {}
    out = {}
    if len(value) > 10000:
        raise OutboundContextError("结构包含过多字段，未静默截断或发送请求。")
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        # Structural wiring is not the referenced local content. Preserve it
        # before applying value/container rules, including metrics and files.
        if isinstance(item, str) and _REFERENCE.fullmatch(item) and _IDENTIFIER.fullmatch(key):
            out[key] = item
            continue
        if key.lower() in _PRIVATE_KEYS:
            continue
        safe = _DROP
        if key == "instruction" and isinstance(item, SafeInstruction):
            safe = str(item)
        elif key == "name" and _context in {
            "columns", "column_profiles", "feature_profiles", "input_fields", "output_fields", "param_schema",
        }:
            safe = _identifier(item)
        elif key == "id" and _context in {"steps", "completed_steps", "remaining_steps", "controls"}:
            safe = _identifier(item, opaque=True)
        elif key == "schema_version" and isinstance(item, str) and item in {"tool-policy.v1", "gate.v1", "failure.v1", "evidence.v1"}:
            safe = item
        elif key == "expected_statuses" and isinstance(item, (list, tuple)):
            safe = [entry for entry in item if isinstance(entry, str) and entry in _SAFE_ENUMS]
        elif key in {"goal", "user_message", "user_instruction", "user_constraint"}:
            if isinstance(item, SafeInstruction):
                safe = str(item)
            else:
                text = item.lower() if isinstance(item, str) else ""
                out["request_topics"] = [name for name, words in _TOPICS.items() if any(word in text for word in words)]
        elif key in {"input_schema", "output_schema", "json_schema", "schema"}:
            safe = safe_schema(item)
        elif key == "split_config" and isinstance(item, dict):
            from marvis.llm_outbound_gates import split_configuration

            safe = split_configuration(item)
        elif key in _BOOL_KEYS and type(item) is bool:
            safe = item
        elif (key in _NUMERIC_KEYS or _METRIC.fullmatch(key)) and _finite(item):
            safe = item
        elif key == "version" and isinstance(item, str) and (item == "" or re.fullmatch(r"\d{1,4}\.\d{1,4}\.\d{1,4}", item)):
            safe = item
        elif key in _IDENTIFIER_KEYS:
            if isinstance(item, dict) and key in {"tool", "target", "schema"}:
                safe = structural_summary(item, _depth=_depth + 1, _context=key)
            elif key in _FIXED_ENUM_FIELDS:
                if isinstance(item, str) and item in _SAFE_ENUMS:
                    safe = item
            else:
                safe = _identifier(item)
        elif key in _ID_KEYS:
            safe = _identifier(item, opaque=True)
        elif key.endswith(("_hash", "_sha256")) and isinstance(item, str) and _HASH.fullmatch(item):
            safe = item
        elif key in _IDENTIFIER_LISTS and isinstance(item, (list, tuple)):
            if len(item) > 10000:
                raise OutboundContextError("结构字段列表超过允许大小，未截断任务约束或发送请求。")
            safe = [result for raw in item
                    if (result := _identifier(raw, opaque=key in {"depends_on", "step_ids"})) is not _DROP]
        elif key == "metrics" and isinstance(item, (list, tuple)):
            safe = [raw for raw in item if isinstance(raw, str) and (raw in _NUMERIC_KEYS or _METRIC.fullmatch(raw))]
        elif key == "outputs" and isinstance(item, dict):
            if len(item) > 10000:
                raise OutboundContextError("步骤输出过多，未截断审阅证据或发送请求。")
            safe = [{"step_id": _identifier(name, opaque=True), "output": structural_summary(child, _depth=_depth + 1, _context="output")}
                    for name, child in item.items()
                    if _identifier(name, opaque=True) is not _DROP and isinstance(child, dict)]
        elif key in _CONTAINERS:
            if isinstance(item, dict):
                safe = structural_summary(item, _depth=_depth + 1, _context=key)
            elif isinstance(item, (list, tuple)):
                limit = 10000
                if len(item) > limit:
                    raise OutboundContextError("结构列表超过允许大小，未截断执行契约或发送请求。")
                safe = [structural_summary(child, _depth=_depth + 1, _context=key)
                        for child in item if isinstance(child, dict)]
        # References are executable structural wiring, never file paths or URLs.
        elif isinstance(item, str) and _REFERENCE.fullmatch(item) and _IDENTIFIER.fullmatch(key):
            safe = item
        if safe is not _DROP:
            out[key] = safe
    return out


_MAX_SCHEMA_DEPTH = 64
_MAX_SCHEMA_BRANCHES = 256
_MAX_SCHEMA_ITEMS = 10000
_MAX_SCHEMA_NODES = 100000
_MAX_SCHEMA_BYTES = 1000000
_SCHEMA_TOKEN = object()
_SCHEMA_ANNOTATIONS = frozenset({
    "description", "examples", "default", "title", "$comment", "deprecated", "readOnly", "writeOnly",
})


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ApplicationSchema:
    """Immutable installed output contract; callers cannot bless a dictionary."""

    contract: str
    _encoded: str = dataclass_field(repr=False)

    def __init__(self, contract, encoded, *, _token=None):
        if _token is not _SCHEMA_TOKEN:
            raise OutboundContextError("输出契约必须来自已安装的应用契约，未发送请求。")
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "_encoded", encoded)

    def __getitem__(self, key):
        return json.loads(self._encoded)[key]

    def get(self, key, default=None):
        return json.loads(self._encoded).get(key, default)

    def __eq__(self, other):
        if isinstance(other, ApplicationSchema):
            return self.contract == other.contract and self._encoded == other._encoded
        if isinstance(other, dict):
            return json.loads(self._encoded) == other
        return NotImplemented


def application_schema(contract, **bindings):
    """Build a named installed contract, with family-specific bounded bindings.

    This is an in-process provenance capability, like StructuralPrompt. It is
    not a registration API for caller schemas or a sandbox against hostile
    Python code. Dynamic identifiers must come from the caller's authenticated
    task controls; identifier shape alone does not establish anonymity.
    """
    if not isinstance(contract, str):
        raise OutboundContextError("应用输出契约名称无效，未发送请求。")
    source = _installed_application_schema(contract, bindings)
    return _mint_application_schema(contract, source)


def _mint_application_schema(contract, source):
    """Internal constructor used only after installed-source resolution."""
    _check_schema_limits(source)
    clean = _copy_application_schema(source)
    _check_output_schema(clean)
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_SCHEMA_BYTES:
        raise OutboundContextError("输出契约超过字节限制，未截断或发送请求。")
    return ApplicationSchema(contract, encoded, _token=_SCHEMA_TOKEN)


def _installed_application_schema(contract, bindings):
    if contract == "sql_select_compile":
        from marvis.llm_outbound_sql import installed_sql_schema

        return installed_sql_schema(bindings)
    if contract in {"adhoc_slice_spec", "feature_cross_recommendations", "draft_authoring", "memory_distillation", "learning_note"}:
        from marvis.llm_outbound_noncore_schemas import installed_noncore_schema

        return installed_noncore_schema(contract, bindings)
    if contract == "workflow_intent":
        from marvis.orchestrator.templates import list_templates

        if set(bindings) != {"candidates"}:
            raise OutboundContextError("工作流意图契约绑定无效，未发送请求。")
        candidates = _schema_binding_list(bindings["candidates"])
        installed = {template.id for template in list_templates()} | {"novel"}
        if not candidates or any(item not in installed for item in candidates):
            raise OutboundContextError("候选工作流不属于已安装目录，未发送请求。")
        return {"name": "workflow_intent", "strict": True, "schema": {
            "type": "object", "properties": {"choice": {"type": "string", "enum": candidates}},
            "required": ["choice"], "additionalProperties": False,
        }}
    if contract == "semantic_intent":
        from marvis.agent.semantic_intent import ALL_SEMANTIC_INTENTS, _schema

        if set(bindings) != {"allowed_intents"}:
            raise OutboundContextError("语义意图契约绑定无效，未发送请求。")
        allowed = _schema_binding_list(bindings["allowed_intents"])
        if not allowed or any(item not in ALL_SEMANTIC_INTENTS for item in allowed):
            raise OutboundContextError("语义意图枚举不属于应用契约，未发送请求。")
        return _schema(allowed)
    if contract == "feature_meaning_direction":
        from marvis.agent.feature_setup import _MEANING_CONFIDENCE, _MEANING_DIRECTIONS
        from marvis.llm_outbound_instructions import column_identifiers

        if set(bindings) != {"features"}:
            raise OutboundContextError("特征方向契约绑定无效，未发送请求。")
        features = _schema_binding_list(bindings["features"])
        features = column_identifiers(features)
        if not features or len(set(features)) != len(features):
            raise OutboundContextError("特征方向字段绑定无效，未发送请求。")
        return {"name": contract, "strict": True, "schema": {
            "type": "object", "properties": {"directions": {"type": "array", "items": {
                "type": "object", "properties": {
                    "feature": {"type": "string", "enum": features},
                    "direction": {"type": "string", "enum": sorted(_MEANING_DIRECTIONS)},
                    "confidence": {"type": "string", "enum": sorted(_MEANING_CONFIDENCE)},
                    "rationale": {"type": "string", "maxLength": 300},
                }, "required": ["feature", "direction", "confidence", "rationale"],
                "additionalProperties": False,
            }, "maxItems": len(features)}}, "required": ["directions"], "additionalProperties": False,
        }}
    if bindings:
        raise OutboundContextError("此应用契约不接受调用者覆盖，未发送请求。")
    if contract == "strategy_request":
        from marvis.agent.strategy_request_compiler.core import STRATEGY_REQUEST_JSON_SCHEMA

        return STRATEGY_REQUEST_JSON_SCHEMA
    if contract == "sample_design_v2_correction":
        from marvis.agent.strategy_request_compiler.sample_design import SAMPLE_DESIGN_V2_CORRECTION_JSON_SCHEMA

        return SAMPLE_DESIGN_V2_CORRECTION_JSON_SCHEMA
    if contract == "plan_steps":
        from marvis.orchestrator.planner import PLAN_STEPS_SCHEMA

        return PLAN_STEPS_SCHEMA
    if contract == "gate_instruction_route":
        from marvis.agent.instruction_router import _ROUTE_SCHEMA

        return _ROUTE_SCHEMA
    if contract == "semantic_authorization":
        from marvis.agent.semantic_authorization import _REVIEW_SCHEMA

        return _REVIEW_SCHEMA
    if contract in {"step_critique", "plan_review_summary"}:
        from marvis.orchestrator.reviewer import _CRITIQUE_SCHEMA, _NARRATIVE_SCHEMA

        return _CRITIQUE_SCHEMA if contract == "step_critique" else _NARRATIVE_SCHEMA
    raise OutboundContextError("未知应用输出契约，未发送请求。")


def _schema_binding_list(value):
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_SCHEMA_ITEMS:
        raise OutboundContextError("输出契约绑定超限或无效，未发送请求。")
    if any(not isinstance(item, str) for item in value) or len(set(value)) != len(value):
        raise OutboundContextError("输出契约绑定包含非法或重复值，未发送请求。")
    return list(value)


def _check_schema_limits(value):
    """Bound the complete input before projection; no branch is truncated."""
    pending = [(value, 0)]
    nodes = 0
    byte_count = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > _MAX_SCHEMA_DEPTH or nodes > _MAX_SCHEMA_NODES:
            raise OutboundContextError("输出契约深度或节点数超限，未截断或发送请求。")
        if isinstance(current, dict):
            if len(current) > _MAX_SCHEMA_ITEMS or any(not isinstance(key, str) for key in current):
                raise OutboundContextError("输出契约字段超限或无效，未发送请求。")
            for key, child in current.items():
                byte_count += len(key.encode("utf-8"))
                if key in {"anyOf", "oneOf", "allOf"} and isinstance(child, list) and len(child) > _MAX_SCHEMA_BRANCHES:
                    raise OutboundContextError("输出契约分支超限，未截断或发送请求。")
                pending.append((child, depth + 1))
        elif isinstance(current, (list, tuple)):
            if len(current) > _MAX_SCHEMA_ITEMS:
                raise OutboundContextError("输出契约列表超限，未截断或发送请求。")
            pending.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            byte_count += len(current.encode("utf-8"))
        elif current is not None and type(current) is not bool:
            if not _schema_number(current):
                raise OutboundContextError("输出契约包含非法值，未发送请求。")
            byte_count += len(str(current))
        if byte_count > _MAX_SCHEMA_BYTES:
            raise OutboundContextError("输出契约超过字节限制，未截断或发送请求。")


def _copy_application_schema(value):
    """Copy the installed contract verbatim, including its application prose."""
    if isinstance(value, dict):
        return {key: _copy_application_schema(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_application_schema(child) for child in value]
    return value


def _schema_number(value):
    # JSON Schema bounds may legitimately include the full IEEE safe-integer
    # range; the aggregate-metric bound in _finite is not a schema contract.
    return ((type(value) is int and value.bit_length() <= 4096)
            or (type(value) is float and math.isfinite(value)))


def _check_output_schema(value):
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    root = value.get("schema", value) if isinstance(value, dict) else value
    if isinstance(value, dict) and "schema" in value:
        if set(value) - ({"name", "strict", "schema"} | _SCHEMA_ANNOTATIONS) or _identifier(value.get("name")) is _DROP:
            raise OutboundContextError("输出契约信封无效，未发送请求。")
        if "strict" in value and type(value["strict"]) is not bool:
            raise OutboundContextError("输出契约严格模式无效，未发送请求。")
    try:
        Draft202012Validator.check_schema(root)
    except (SchemaError, TypeError, ValueError, RecursionError):
        raise OutboundContextError("输出契约不是有效 JSON Schema，未发送请求。") from None
    pending = [root]
    references = []
    anchors = set()
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            if "$id" in current:
                # A nested resource changes the base of otherwise local refs.
                # The supported installed contract is one resource with local
                # pointers/anchors; do not certify a different lookup scope.
                raise OutboundContextError("输出契约不支持带资源作用域的引用，未发送请求。")
            for key in ("$ref", "$dynamicRef"):
                if key in current:
                    references.append(current[key])
            for key in ("$anchor", "$dynamicAnchor"):
                if key in current:
                    anchors.add(current[key])
            for key in {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}:
                if isinstance(current.get(key), dict):
                    pending.extend(current[key].values())
            for key in {"items", "additionalItems", "additionalProperties", "contains", "propertyNames", "not",
                        "if", "then", "else", "unevaluatedItems", "unevaluatedProperties", "contentSchema"}:
                if isinstance(current.get(key), dict):
                    pending.append(current[key])
            for key in {"anyOf", "oneOf", "allOf", "prefixItems"}:
                if isinstance(current.get(key), list):
                    pending.extend(current[key])
    for reference in references:
        if not isinstance(reference, str) or not reference.startswith("#"):
            raise OutboundContextError("输出契约只能引用本地定义，未发送请求。")
        target = root
        if reference.startswith("#/"):
            try:
                for part in reference[2:].split("/"):
                    part = part.replace("~1", "/").replace("~0", "~")
                    target = target[int(part)] if isinstance(target, list) else target[part]
            except (KeyError, IndexError, TypeError, ValueError):
                raise OutboundContextError("输出契约引用未绑定，未发送请求。") from None
            if not isinstance(target, (dict, bool)):
                raise OutboundContextError("输出契约引用目标无效，未发送请求。")
        elif reference != "#" and reference[1:] not in anchors:
            raise OutboundContextError("输出契约锚点未绑定，未发送请求。")


def safe_schema(value, _depth=0, _field=None):
    """Preserve installed contracts; project unowned schemas restrictively."""
    if isinstance(value, ApplicationSchema):
        # The capability holds an immutable snapshot validated at construction.
        # Rechecking every installed schema during catalog ranking is both
        # redundant and costly; the actual output-schema transport is checked
        # once more by the client before a request.
        return json.loads(value._encoded)
    if _depth > _MAX_SCHEMA_DEPTH:
        raise OutboundContextError("输出契约深度超限，未截断或发送请求。")
    if not isinstance(value, dict):
        raise OutboundContextError("输出契约必须是结构对象，未发送请求。")
    if _depth == 0:
        _check_schema_limits(value)
    if set(value) & {"const", "$ref", "$defs", "definitions", "$dynamicRef", "if", "then", "else", "not"}:
        raise OutboundContextError("完整输出约束需要已安装的应用契约，未发送请求。")
    out = {}
    def property_name(name):
        if _identifier(name) is not _DROP:
            return True
        # These colon-prefixed names are actual report output fields, not
        # arbitrary user-provided schema labels.
        from marvis.repositories.tasks import AGENT_REPORT_WRITABLE_KEYS
        return isinstance(name, str) and name in AGENT_REPORT_WRITABLE_KEYS

    for key, item in value.items():
        if key == "properties" and isinstance(item, dict):
            if len(item) > 10000:
                raise OutboundContextError("输出结构过大，未截断约束或发送请求。")
            out[key] = {name: safe_schema(child, _depth + 1, name) for name, child in item.items()
                        if property_name(name)}
        elif key in {"schema", "items", "additionalProperties"} and isinstance(item, dict):
            out[key] = safe_schema(item, _depth + 1)
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(item, list):
            out[key] = [safe_schema(child, _depth + 1) for child in item]
        elif key in {"required"} and isinstance(item, list):
            if len(item) > 10000:
                raise OutboundContextError("输出必填字段过多，未截断约束或发送请求。")
            out[key] = [raw for raw in item if property_name(raw)]
        elif key in {"type", "enum"}:
            values = item if isinstance(item, list) else [item]
            if len(values) > 10000:
                raise OutboundContextError("输出枚举过多，未截断约束或发送请求。")
            safe = [raw for raw in values if isinstance(raw, str) and raw in _SAFE_ENUMS]
            if key == "enum" and _field == "selected_experiment_id":
                safe = [raw for raw in values if _experiment_id(raw)]
            if key == "enum":
                # Never remove a rejected enum and accidentally widen the
                # contract to every string. An empty enum remains unsatisfiable.
                out[key] = safe
            elif safe:
                out[key] = safe if isinstance(item, list) else safe[0]
        elif key in {"strict", "additionalProperties"} and type(item) is bool:
            out[key] = item
        elif key == "name" and _identifier(item) is not _DROP:
            out[key] = item
        elif key in {"minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength"} and _schema_number(item):
            out[key] = item
    return out


def outbound_prompt(value):
    """Accept a caller-built structure, never interpret arbitrary text as data."""
    if isinstance(value, StructuralPrompt):
        return value
    if not isinstance(value, dict):
        raise OutboundContextError("LLM requests require an explicit structural context")
    summary = structural_summary(value)
    return _mint_prompt(summary)


def _mint_prompt(summary):
    encoded = json.dumps({"outbound_policy": POLICY_VERSION, **summary}, ensure_ascii=False,
                         sort_keys=True, allow_nan=False, separators=(",", ":"))
    return StructuralPrompt(encoded, _token=_PROMPT_TOKEN)


def planning_prompt(payload):
    from marvis.orchestrator.planner import PLANNING_EXAMPLES
    from marvis.llm_outbound_instructions import user_instruction
    from marvis.llm_outbound_catalog import attach_planning_catalog, project_planning_catalog
    from marvis.llm_outbound_planning import attach_planning_controls

    approved_payload = dict(project_planning_catalog(payload))
    for field in ("goal", "user_constraint"):
        value = approved_payload.get(field)
        if value is not None:
            approved_payload[field] = user_instruction(value)
    clean = structural_summary(approved_payload)
    catalog_contracts = attach_planning_catalog(clean, payload)
    planning_controls = attach_planning_controls(clean, payload)
    clean["instruction_origin"] = "interactive_user"
    clean["planning_examples"] = PLANNING_EXAMPLES
    clean["instruction"] = (
        "Return a JSON object with steps. Give every step a unique `id`; omit optional fields "
        "instead of null. Use $ref:<step_id>.output.<field> for upstream outputs. "
        "planner_constraints are hard server-authored constraints. Preserve completed steps and "
        "all required binding inputs. Do not infer unavailable raw data or compute metrics."
    )
    if "remaining_steps" in payload:
        clean["instruction"] += (
            " Treat remaining_steps as reusable source contracts: copy retained steps "
            "field-for-field unless the user asks to change them. Remove or rewrite every "
            "depends_on edge and $ref targeting a deleted step. Return the full revised remaining plan; "
            "when work remains, steps must be non-empty. Do not include completed steps."
        )
    elif "completed_steps" in payload:
        clean["instruction"] += (
            " Return done=true and steps=[] only if the goal and all required planner constraints are complete. "
            "Otherwise return no more than max_steps next-segment steps. After prior exploration, "
            "the next segment must advance at least one currently unmet required "
            "tool or literal-input constraint; do not repeat work."
        )
    if payload.get("last_error") or payload.get("repair_context"):
        clean["repair_context"] = _planner_repair_context(payload)
    prompt = _mint_prompt(clean)
    prompt._catalog_contracts = catalog_contracts
    prompt._planning_controls = planning_controls
    return prompt


def _planner_repair_context(payload):
    # Classify known platform diagnostic categories; never echo exceptions or
    # parse arbitrary JSON embedded in a prior response.
    error = payload.get("last_error")
    if not isinstance(error, str) and isinstance(payload.get("repair_context"), dict):
        categories = payload["repair_context"].get("categories")
        if isinstance(categories, list):
            error = "; ".join(item for item in categories if isinstance(item, str))
    if not isinstance(error, str):
        return {"categories": ["previous_plan_invalid"]}
    categories = [label for label in (
        "not json", "invalid plan fields", "invalid explore fields", "required literal inputs mismatch",
        "done must be omitted rather than null", "non-empty steps", "unknown dependency",
        "invalid ref", "conflicting id and step_id", "constraint progress",
        "outside the allowed replan catalog", "made no progress toward unmet planner constraints",
    ) if label in error]
    constraints = payload.get("planner_constraints") or {}
    references = [*payload.get("available_tools", [])]
    if isinstance(constraints, dict):
        references += list(constraints.get("required_tool_refs") or [])
        references += list(constraints.get("forbidden_tool_refs") or [])
    for ref in references:
        if not isinstance(ref, dict):
            continue
        plugin, tool = ref.get("plugin"), ref.get("tool")
        if _identifier(plugin) is _DROP or _identifier(tool) is _DROP:
            continue
        for prefix, suffix in (("forbidden tool ", ""), ("required tool ", " is missing")):
            label = prefix + plugin + "." + tool + suffix
            if label in error:
                categories.append(label)
    if not categories:
        categories = ["previous plan failed validation; check required inputs, references and tools missing from scoped catalog"]
    return {"categories": list(dict.fromkeys(categories))}


def report_prompt(summary):
    from marvis.llm_outbound_reports import modeling_report_prompt
    return modeling_report_prompt(summary)


def review_repair_prompt(prompt, *, kind):
    """Repair response shape using the frozen context, never the rejected text."""
    if not isinstance(prompt, StructuralPrompt) or kind not in {"critique", "summary"}:
        raise OutboundContextError("审阅修复需要已批准的原始上下文和已知响应类型。")
    clean = json.loads(prompt)
    clean["response_repair"] = {
        "kind": kind,
        "instruction": (
            "Previous reply was not parseable JSON or did not match the requested response schema. "
            "Return one complete JSON object conforming to the same schema. "
            "Use only the unchanged measured evidence in this request; do not invent missing facts."
        ),
    }
    return _mint_prompt(clean)


def safe_system_prompt(system_prompt, prompt_name=None):
    """Use installed application instructions, never caller-supplied file text."""
    from marvis.llm_prompts import ALL_PROMPTS, GATE_SYSTEM_TEMPLATE

    for spec in ALL_PROMPTS:
        if spec == GATE_SYSTEM_TEMPLATE:
            if prompt_name == spec.name or str(system_prompt).startswith(spec.text.split("{allowed_actions}")[0]):
                return spec.text.format(allowed_actions="confirm, adjust, replan, clarify, halt")
        elif prompt_name == spec.name or str(system_prompt).startswith(spec.text):
            return spec.text
    return (
        "You are MARVIS. Use only the supplied structural context and platform-computed aggregates. "
        "Raw data and local files are withheld. Do not infer missing facts or invent metrics. "
        "Return the requested structured decision or explanation; ask for clarification when evidence is insufficient."
    )


def safe_extra_request_fields(value):
    """Provider tuning knobs cannot override messages, schemas, URLs or metadata."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("LLM extra request fields violate the outbound contract")
    out = {}
    for key, item in value.items():
        if key in {"top_k", "top_p", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty", "seed"} and _finite(item):
            out[key] = item
        elif key == "chat_template_kwargs" and isinstance(item, dict) and set(item) <= {"enable_thinking"} and all(type(v) is bool for v in item.values()):
            out[key] = item
        else:
            raise ValueError("LLM extra request fields violate the outbound contract")
    return out
