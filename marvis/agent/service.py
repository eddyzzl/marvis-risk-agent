from __future__ import annotations

from collections.abc import Callable
import json
import re

from marvis.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    RISK_METRIC_INTERPRETATION_GUIDANCE,
    WORD_CONCLUSION_SYSTEM_PROMPT,
    WORD_CONCLUSION_V2_SYSTEM_PROMPT,
)
from marvis.db import AGENT_REPORT_CONCLUSION_KEYS
from marvis.domain import TaskRecord
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.agent_memory.prompting import (
    add_memory_to_prompt_payload,
    attach_memory_metadata,
    memory_context_was_truncated,
    normalize_memory_context,
)


REQUIRED_AGENT_REPORT_KEYS = tuple(sorted(AGENT_REPORT_CONCLUSION_KEYS))
_V2_FINAL_CONCLUSION_FORBIDDEN_PATTERNS = (
    re.compile(r"材料(?:扫描|识别|完备)"),
    re.compile(r"验证输入契约"),
    re.compile(r"PMML\s*(?:全量)?(?:打分|评分)(?:测试|完成|通过|成功|覆盖|样本|耗时)?", re.IGNORECASE),
    re.compile(r"报告(?:已|进入|生成|产出|定稿)"),
    re.compile(r"最终定稿|建议(?:在)?投产前审阅|建议审阅(?:报告|结论)|确认\s*Word", re.IGNORECASE),
    re.compile(r"可直接(?:部署|投产)"),
)
GLOBAL_AGENT_EVIDENCE_KEYS = frozenset(
    {
        "scan",
        "notebook_steps",
        "contract",
        "reproducibility",
        "pmml_scoring",
        "validation_results",
        "report_fields",
        "visible_stage_summaries",
    }
)
STAGE_EVIDENCE_KEYS = {
    "scan": ("scan", "contract", "notebook_steps"),
    "reproducibility": (
        "notebook_steps",
        "contract",
        "reproducibility",
        "pmml_scoring",
    ),
    "report": ("report_fields",),
}
SCAN_REQUIRED_CHECK_LABELS = (
    "Notebook 文件",
    "样本数据",
    "PMML 模型",
    "数据字典",
    "Notebook RMC 契约",
)
SCAN_SUCCESS_STATUSES = {"success", "passed", "pass", "ok", "通过", "已通过"}
METRICS_VALIDATION_RESULT_KEYS = (
    "model_name",
    "model_version",
    "algorithm",
    "target_type",
    "basic_info",
    "effectiveness",
    "overfitting_check",
    "stress_test",
)
WORD_CONCLUSION_MONTHLY_LIMIT = 12
WORD_CONCLUSION_FEATURE_LIMIT = 20
WORD_CONCLUSION_STRESS_CATEGORY_LIMIT = 20
WORD_CONCLUSION_PSI_BIN_LIMIT = 12
WORD_CONCLUSION_VISIBLE_SUMMARY_LIMIT = 8
WORD_CONCLUSION_VISIBLE_SUMMARY_CHARS = 1200
REPRODUCIBILITY_NOTEBOOK_STEP_LIMIT = 16
REPRODUCIBILITY_CONTRACT_FEATURE_LIMIT = 20
NOTEBOOK_CONTRACT_SOURCE_PREVIEW_LIMIT = 3
NOTEBOOK_CONTRACT_SOURCE_PREVIEW_CHARS = 500
NOTEBOOK_FAILURE_CELL_LIMIT = 3
NOTEBOOK_FAILURE_SOURCE_CHARS = 1200
NOTEBOOK_FAILURE_FILE_LIMIT = 20
NOTEBOOK_FAILURE_LINE_LIMIT = 12
# Raw chart series that balloon the prompt without helping LLM reasoning
# (each curve carries thousands of (x, y) samples — easily 10+ MB total).
# AUC/KS summary numbers used in image-1's analysis already live in
# effectiveness.overall, so dropping the raw curves keeps the analysis honest
# while shrinking the prompt by >99%.
OVERSIZED_EFFECTIVENESS_KEYS = frozenset({"roc_ks_curves"})
START_VALIDATION_GUIDANCE_FRAGMENTS = (
    "可以协助启动验证",
    "需要开始时，请输入",
    "请输入“开始验证”",
    '请输入"开始验证"',
    "输入“开始验证”",
    '输入"开始验证"',
)
CONVERSATION_MEMORY_MAX_MESSAGES = 48
CONVERSATION_MEMORY_MAX_CHARS = 32000
CONVERSATION_MEMORY_MESSAGE_MAX_CHARS = 2400
GREETING_CHAT_FALLBACK = (
    "你好，我在。你可以直接问我验证结果、指标含义、PMML 或报告结论，"
    "也可以告诉我下一步想处理什么。"
)
STAGE_SUMMARY_MAX_TOKENS = {
    "metrics": 4096,
}
AGENT_RESPONSE_PREAMBLE_PATTERNS = (
    r"^(?:好的|好|收到|明白|可以)[，,。！!\s]*",
    r"^(?:我会|我将)?(?:遵照|根据|按照)(?:您|你)?的?(?:指示|要求)[，,。！!\s]*",
    # The trailing terminator is required (+, not *): otherwise the bare 分析/总结
    # alternatives match mid-sentence and delete the opening clause of legitimate
    # content like "以下是关于本次验证分析的详细内容。" → "的详细内容。".
    r"^以下是(?:针对|关于|基于)[^\n]{0,160}?(?:验证分析|阶段分析|分析说明|验证总结|分析|总结)[。:：\s]+",
)
AGENT_RESPONSE_LEADING_SEPARATOR_PATTERN = re.compile(r"^(?:[-*_]\s*){3,}\s*")

# Start-family negation markers — mirror is_continue_validation_intent's
# negation_markers so "先别开始验证" / "不要开始验证" short-circuit to chat
# before the direct-phrase substring branch can fire. Includes the generic
# negators plan_driver._NEGATED_CONFIRM already trusts (先别/别/不要/不用/
# 不需要/先不/暂不) so phrasings not enumerated below are still caught.
START_VALIDATION_NEGATION_MARKERS = (
    "不开始",
    "不要开始",
    "先别开始",
    "别开始",
    "暂不开始",
    "暂时不开始",
    "不用开始",
    "无需开始",
    "不需要开始",
    "不想开始",
    "没必要开始",
    "不打算开始",
    "不会开始",
    "先不开始",
    "不启动",
    "不要启动",
    "别启动",
    "不执行",
    "不要执行",
    "别执行",
    "不运行",
    "不要运行",
    "不跑",
    "不要跑",
    "先别",
    "别",
    "不要",
    "不用",
    "不需要",
    "先不",
    "暂不",
    "暂停",
)
# English-negation parity with plan_driver._NEGATED_CONFIRM's do-not / don't
# family, so "do not start validation" / "don't run validation" stay chat.
_START_VALIDATION_ENGLISH_NEGATION = re.compile(
    r"\b(?:do\s*not|don't|dont|not)\s+(?:start|run|validate)\b",
    re.IGNORECASE,
)
# Interrogative guard — mirror plan_driver._QUESTION's particle set plus
# start-context interrogatives. NOTE: bare trailing '吧' is intentionally
# EXCLUDED (unlike plan_driver._QUESTION) because '开始吧'/'启动吧'/'运行吧'/
# '跑吧' are legitimate start affirmatives in direct_commands, not questions.
_START_QUESTION = re.compile(
    r"[?？]|吗|呢$|什么时候|何时|要不要|需不需要|能不能|可不可以|是不是",
    re.IGNORECASE,
)


def is_start_validation_intent(content: str) -> bool:
    text = content.strip().lower()
    if not text:
        return False
    # Question guard on RAW text (like plan_driver.is_confirm) so trailing
    # particles / question marks disqualify interrogatives before any positive
    # match. '吧$' is deliberately not treated as a question here.
    if _START_QUESTION.search(text):
        return False
    if _START_VALIDATION_ENGLISH_NEGATION.search(text):
        return False
    compact = "".join(text.split())
    # Drop interior punctuation for the negation scan, same as
    # is_continue_validation_intent, so "先别，开始验证" normalizes cleanly.
    compact_np = compact
    for ch in "，。、；：,;:":
        compact_np = compact_np.replace(ch, "")
    if any(marker in compact_np for marker in START_VALIDATION_NEGATION_MARKERS):
        return False
    direct_phrases = (
        "开始验证",
        "开始模型验证",
        "启动验证",
        "启动模型验证",
        "执行验证",
        "执行模型验证",
        "运行验证",
        "运行模型验证",
        "跑验证",
        "跑模型验证",
        "开始执行",
        "开始运行",
        "开始跑",
        "跑一下",
        "跑一遍",
        "跑起来",
        "启动任务",
        "开始任务",
        "startvalidation",
        "runvalidation",
        "validatethistask",
    )
    if any(phrase in compact for phrase in direct_phrases):
        return True
    direct_commands = {
        "开始",
        "开始吧",
        "启动",
        "启动吧",
        "运行",
        "运行吧",
        "执行",
        "执行吧",
        "跑吧",
        "start",
        "run",
        "validate",
    }
    return compact.strip("。.!！?？") in direct_commands


def is_continue_validation_intent(content: str) -> bool:
    text = content.strip().lower()
    if not text:
        return False
    compact = "".join(text.split()).strip("。.!！?？")
    # Drop interior punctuation so acknowledged-continue phrasings like
    # "明白了，继续验证吧" share a normal form with "继续验证吧". Without
    # this, the Chinese comma blocks affix stripping and the matcher
    # routes the user's intent into the chat-question branch.
    for ch in "，。、；：,;:":
        compact = compact.replace(ch, "")
    negation_markers = (
        "不继续",
        "不要继续",
        "先不继续",
        "暂不继续",
        "暂时不继续",
        "不用继续",
        "别继续",
        "无需继续",
        "不需要继续",
        "不想继续",
        "没必要继续",
        "不打算继续",
        "不会继续",
    )
    if any(marker in compact for marker in negation_markers):
        return False
    direct_phrases = {
        "继续",
        "继续吧",
        "请继续",
        "下一步",
        "继续下一步",
        "执行下一步",
        "继续执行",
        "继续验证",
        "往下走",
        "可以继续",
        "确认继续",
        "goon",
        "continue",
        "next",
    }
    if compact in direct_phrases:
        return True
    if _is_continue_report_draft_intent(compact):
        return True
    # Substring fallback for unambiguous continue fragments. After the
    # negation markers above were ruled out, finding any of these inside
    # the compacted message means the user wants to advance even though
    # they prefixed an acknowledgment.
    unambiguous_continue_fragments = (
        "继续验证",
        "继续执行",
        "继续下一步",
        "执行下一步",
    )
    if any(fragment in compact for fragment in unambiguous_continue_fragments):
        return True
    normalized = _strip_continue_command_affixes(compact)
    return normalized in direct_phrases or _is_continue_report_draft_intent(normalized)


def _is_continue_report_draft_intent(value: str) -> bool:
    if "继续" not in value and "下一步" not in value and "执行" not in value:
        return False
    report_actions = (
        "继续生成报告",
        "继续生成word",
        "继续写报告",
        "继续起草报告",
        "继续生成草稿",
        "继续起草草稿",
        "继续生成结论",
        "继续起草结论",
        "继续生成三段",
        "继续起草三段",
        "下一步生成报告",
        "下一步生成word",
        "执行报告",
        "执行报告结论",
    )
    return any(action in value for action in report_actions)


def _strip_continue_command_affixes(value: str) -> str:
    text = value
    # Sorted longest-first so multi-char acknowledgments ("明白了") strip
    # cleanly before the shorter overlap ("明白") gets a chance to leave a
    # trailing "了" stuck on the front of the remaining phrase.
    prefixes = (
        "明白了",
        "了解了",
        "知道了",
        "懂了",
        "收到",
        "了解",
        "知道",
        "明白",
        "好的",
        "麻烦",
        "帮我",
        "确认",
        "可以",
        "请",
        "那",
        "先",
        "好",
        "嗯",
        "ok",
    )
    suffixes = ("一下", "下", "吧", "了")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix) :]
                changed = True
        for suffix in suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text


def is_agent_advance_intent(content: str) -> bool:
    return is_start_validation_intent(content) or is_continue_validation_intent(content)


def agent_rerun_stage(content: str) -> str | None:
    compact = "".join(str(content or "").strip().lower().split()).strip("。.!！?？")
    if not compact:
        return None
    rerun_markers = (
        "重新",
        "重跑",
        "重做",
        "重来",
        "重写",
        "再跑",
        "再执行",
        "再生成",
        "再写",
        "从头",
        "从一开始",
        "从最开始",
        "从开始",
    )
    if not any(marker in compact for marker in rerun_markers):
        return None
    if _matches_rerun_stage(
        compact,
        (
            "从头",
            "从一开始",
            "从最开始",
            "从开始",
            "从第一步",
            "从第1步",
            "从步骤1",
            "从步骤一",
            "全流程",
            "全部流程",
            "完整流程",
            "完整执行",
            "完整重跑",
            "全量",
            "全部重新",
            "全部重跑",
            "重新执行全部",
            "重新执行一遍全部",
            "重新跑全部",
            "整体重跑",
        ),
    ):
        return "scan"
    if _matches_rerun_stage(
        compact,
        (
            "第四步",
            "第4步",
            "4步",
            "步骤4",
            "步骤四",
            "第四阶段",
            "第4阶段",
            "阶段4",
            "阶段四",
            "step4",
            "第4个步骤",
            "第四个步骤",
            "报告",
            "word",
            "草稿",
            "结论",
            "三段",
            "三段总结",
            "三段结论",
            "写报告",
            "生成报告",
            "报告生成",
            "word报告",
            "最终报告",
        ),
    ):
        return "word_conclusion_draft"
    if _matches_rerun_stage(
        compact,
        (
            "第三步",
            "第3步",
            "3步",
            "步骤3",
            "步骤三",
            "第三阶段",
            "第3阶段",
            "阶段3",
            "阶段三",
            "step3",
            "第3个步骤",
            "第三个步骤",
            "效果",
            "稳定性",
            "效果稳定性",
            "效果与稳定性",
            "模型效果",
            "模型效果&稳定性",
            "指标",
            "指标概览",
            "ks",
            "psi",
            "auc",
            "分箱",
            "压力测试",
            "过拟合",
            "oot",
        ),
    ):
        return "metrics"
    if _matches_rerun_stage(
        compact,
        (
            "第二步",
            "第2步",
            "2步",
            "步骤2",
            "步骤二",
            "第二阶段",
            "第2阶段",
            "阶段2",
            "阶段二",
            "step2",
            "第2个步骤",
            "第二个步骤",
            "复现",
            "复现性",
            "可复现",
            "可复现性",
            "模型复现",
            "模型可复现",
            "复现验证",
            "可复现性验证",
            "notebook",
            "分数一致",
            "分数一致性",
            "建模代码",
            "pmml打分",
            "部署一致性",
        ),
    ):
        return "reproducibility"
    if _matches_rerun_stage(
        compact,
        (
            "第一步",
            "第1步",
            "1步",
            "步骤1",
            "步骤一",
            "第一阶段",
            "第1阶段",
            "阶段1",
            "阶段一",
            "step1",
            "第1个步骤",
            "第一个步骤",
            "材料",
            "完备",
            "完备性",
            "材料完备性",
            "完备性验证",
            "完备性检查",
            "材料完备性验证",
            "材料识别",
            "材料扫描",
            "读取",
            "识别",
            "扫描",
            "目录",
            "文件",
            "rmc契约",
        ),
    ):
        return "scan"
    return None


def _matches_rerun_stage(compact: str, markers: tuple[str, ...]) -> bool:
    return any(marker in compact for marker in markers)


def is_stop_validation_intent(content: str) -> bool:
    text = content.strip().lower()
    if not text:
        return False
    negated_phrases = ("不要停止", "不用停止", "无需停止", "别停止")
    if any(phrase in text for phrase in negated_phrases):
        return False
    keywords = (
        "停止",
        "停下",
        "终止",
        "中止",
        "取消",
        "别跑",
        "不用跑",
        "stop",
        "cancel",
        "abort",
        "terminate",
    )
    return any(keyword in text for keyword in keywords)


def _is_greeting_message(content: str) -> bool:
    compact = "".join(content.strip().lower().split()).strip("。.!！?？~～")
    return compact in {
        "你好",
        "您好",
        "hello",
        "hi",
        "hey",
        "嗨",
        "在吗",
        "在不在",
    }


def _chat_fallback_for_message(user_message: str) -> str:
    if _is_greeting_message(user_message):
        return GREETING_CHAT_FALLBACK
    return "我现在无法调用大模型完成这次回答。请检查大模型 API 配置、网络连通性或稍后重试。"


def _looks_like_start_validation_guidance(content: str) -> bool:
    return any(fragment in content for fragment in START_VALIDATION_GUIDANCE_FRAGMENTS)


def agent_conclusions_confirmed(values: dict[str, str]) -> bool:
    return all(str(values.get(key) or "").strip() for key in REQUIRED_AGENT_REPORT_KEYS)


def summarize_stage(
    *,
    task: TaskRecord,
    stage: str,
    evidence: dict,
    memory_context: dict | None = None,
    model_profile: dict,
    fallback: str,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    prompt = _stage_prompt(
        task=task,
        stage=stage,
        evidence=evidence,
        memory_context=memory_context,
    )
    # LLM-5: memory injection is one of the three named highest-volume prompt
    # touch points; surface the truncation flag on this call's audit record.
    truncated = memory_context_was_truncated(normalize_memory_context(memory_context))
    try:
        content = _client(model_profile).complete(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=STAGE_SUMMARY_MAX_TOKENS.get(stage),
            on_delta=on_delta,
            truncated=truncated,
        )
    except LLMClientError as exc:
        return fallback, {"llm_error": str(exc), "fallback": True}
    cleaned = _strip_agent_response_preamble(content or fallback)
    guarded = _guard_stage_summary(stage=stage, evidence=evidence, content=cleaned)
    metadata = {"fallback": False}
    if guarded != cleaned:
        metadata["guarded_scan_summary"] = True
    return guarded, attach_memory_metadata(metadata, memory_context, use_reason=stage)


def compose_agent_start_message(
    *,
    task: TaskRecord,
    model_profile: dict,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    prompt = _stage_prompt(
        task=task,
        stage="agent_start",
        evidence={
            "next_tool": "scan_materials",
            "tool_purpose": "识别 Notebook、样本数据、PMML 模型、数据字典并检查 Notebook RMC 契约",
        },
    )
    fallback = (
        "我将先以信贷风控模型验证专家身份检查本次验证材料的完备性，"
        "随后调用材料识别工具读取目录、识别关键文件并检查 Notebook RMC 契约。"
    )
    try:
        content = _client(model_profile).complete(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            on_delta=on_delta,
        )
    except LLMClientError as exc:
        return fallback, {"llm_error": str(exc), "fallback": True}
    return _strip_agent_response_preamble(content or fallback), {"fallback": False}


def failure_summary(
    *,
    task: TaskRecord,
    stage: str,
    error: str,
    evidence: dict | None = None,
    memory_context: dict | None = None,
    model_profile: dict,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    failure_evidence = {"failed_stage": stage, "error": error}
    if isinstance(evidence, dict):
        notebook_failure = _compact_notebook_failure_evidence(
            evidence.get("notebook_steps")
        )
        if notebook_failure:
            failure_evidence["notebook_failure"] = notebook_failure
        scan = _compact_scan_evidence(evidence.get("scan"))
        if scan:
            failure_evidence["scan"] = scan
    prompt = _stage_prompt(
        task=task,
        stage="failure",
        evidence=failure_evidence,
        memory_context=memory_context,
    )
    fallback = f"失败阶段：{stage}\n直接原因：{error}\n可能原因：请检查该阶段输入材料、执行环境和上游产物。\n下一步：修正后重新从失败阶段执行。"
    try:
        content = _client(model_profile).complete(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.1,
            on_delta=on_delta,
        )
    except LLMClientError as exc:
        return fallback, {"llm_error": str(exc), "fallback": True}
    return _strip_agent_response_preamble(content or fallback), attach_memory_metadata(
        {"fallback": False},
        memory_context,
        use_reason="failure",
    )


def answer_chat_message(
    *,
    task: TaskRecord,
    user_message: str,
    conversation: list[dict],
    evidence: dict,
    memory_context: dict | None = None,
    model_profile: dict,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    prompt = _chat_prompt(
        task=task,
        user_message=user_message,
        conversation=conversation,
        evidence=evidence,
        memory_context=memory_context,
    )
    fallback = _chat_fallback_for_message(user_message)
    # LLM-5: memory injection is one of the three named highest-volume prompt
    # touch points; surface the truncation flag on this call's audit record.
    truncated = memory_context_was_truncated(normalize_memory_context(memory_context))
    try:
        raw_content = _client(model_profile).complete(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            on_delta=on_delta,
            truncated=truncated,
        )
        content = (raw_content or "").strip()
    except LLMClientError as exc:
        return fallback, {"llm_error": str(exc), "fallback": True}
    if not content:
        return fallback, {"fallback": True, "empty_llm_response": True}
    if _is_greeting_message(user_message) and _looks_like_start_validation_guidance(
        content
    ):
        return fallback, {"fallback": True, "llm_response_replaced": True}
    return _strip_agent_response_preamble(content), attach_memory_metadata(
        {"fallback": False},
        memory_context,
        use_reason="chat",
    )


def _strip_agent_response_preamble(content: str) -> str:
    original = str(content or "").strip()
    if not original:
        return ""
    text = original
    for _ in range(6):
        before = text
        text = AGENT_RESPONSE_LEADING_SEPARATOR_PATTERN.sub("", text).lstrip()
        for pattern in AGENT_RESPONSE_PREAMBLE_PATTERNS:
            text = re.sub(pattern, "", text, count=1).lstrip()
            text = AGENT_RESPONSE_LEADING_SEPARATOR_PATTERN.sub("", text).lstrip()
        if text == before:
            break
    return text.strip() or original


def generate_word_conclusions(
    *,
    task: TaskRecord,
    evidence: dict,
    memory_context: dict | None = None,
    model_profile: dict,
    user_instruction: str | None = None,
) -> tuple[dict[str, str], dict]:
    prompt = _stage_prompt(
        task=task,
        stage="word_conclusion_draft",
        evidence=evidence,
        memory_context=memory_context,
        user_instruction=user_instruction,
    )
    try:
        content = _client(model_profile).complete(
            system_prompt=(
                WORD_CONCLUSION_V2_SYSTEM_PROMPT
                if task.validation_workflow_version == 2
                else WORD_CONCLUSION_SYSTEM_PROMPT
            ),
            user_prompt=prompt,
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=False,
        )
        values = _parse_conclusion_json(content)
        if task.validation_workflow_version == 2:
            _validate_v2_word_conclusions(values)
    except (LLMClientError, ValueError) as exc:
        return {}, {"llm_error": str(exc), "fallback": True, "confirmable": False}
    return values, attach_memory_metadata(
        {"fallback": False},
        memory_context,
        use_reason="word_conclusion_draft",
    )


def fallback_word_conclusions(*, task: TaskRecord) -> dict[str, str]:
    name = task.model_name or "本模型"
    if task.validation_workflow_version == 2:
        return {
            "TEXT:pressure_test_summary": (
                "平台已完成模型压力测试相关指标产出。由于当前未能生成更细的模型解释文本，"
                "建议验证人员结合结构化明细复核各压力场景下 KS、PSI 和打分分布变化。"
            ),
            "TEXT:pressure_impact_recommendation": (
                "建议对模型压力测试中影响较大的数据源和特征类别设置上线后监控阈值；"
                "如出现显著稳定性或区分能力下降，应先复核信源质量和样本分布后再继续使用。"
            ),
            "TEXT:final_validation_conclusion": (
                f"当前缺少足以直接评价{name}区分效果、样本外稳定性、过拟合和压力风险的结构化证据，"
                "因此不能形成模型可用性结论。PMML部署可用。"
            ),
        }
    return {
        "TEXT:pressure_test_summary": (
            "平台已完成压力测试相关指标产出。由于当前未能生成更细的模型解释文本，"
            "建议验证人员结合 Excel 明细复核各压力场景下 KS、PSI 和打分分布变化。"
        ),
        "TEXT:pressure_impact_recommendation": (
            "建议对压力测试中影响较大的数据源和特征类别设置上线后监控阈值；"
            "如出现显著稳定性或区分能力下降，应先复核信源质量和样本分布后再继续使用。"
        ),
        "TEXT:final_validation_conclusion": (
            f"本次验证已围绕{name}开展材料完备性、Notebook 可复现性、模型效果、稳定性和压力测试检查。"
            "从当前平台产物看，核心验证流程已执行至报告结论候选生成阶段；最终是否符合要求仍应以结构化指标、"
            "压力测试明细和验证人员复核意见为准。建议在确认 Word 结论前重点核对 OOT 区分效果、PSI 稳定性和关键变量压力表现。"
        ),
    }


def _client(model_profile: dict) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(model_profile)


def _stage_prompt(
    *,
    task: TaskRecord,
    stage: str,
    evidence: dict,
    memory_context: dict | None = None,
    user_instruction: str | None = None,
) -> str:
    instructions = _stage_instructions(
        stage,
        validation_workflow_version=task.validation_workflow_version,
    )
    payload = {
        "stage": stage,
        "task": _llm_task_meta(task),
        "evidence": _sanitize_llm_payload(
            _stage_scoped_evidence(
                stage,
                evidence,
                validation_workflow_version=task.validation_workflow_version,
            ),
            task,
        ),
        "instructions": instructions,
    }
    if user_instruction:
        payload["user_instruction"] = _sanitize_llm_text(user_instruction, task)
        payload["instructions"] = (
            instructions
            + "本次是用户要求重新生成该阶段内容，必须优先满足 user_instruction 中的修改要求；"
            "不得因为已有旧草稿而复用旧措辞。"
        )
    payload = add_memory_to_prompt_payload(payload, memory_context)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _stage_scoped_evidence(
    stage: str,
    evidence: dict,
    *,
    validation_workflow_version: int | None = None,
) -> dict:
    if not isinstance(evidence, dict):
        return evidence
    if stage == "scan":
        return _scan_stage_evidence(evidence)
    if stage == "reproducibility":
        if validation_workflow_version == 2:
            return _pmml_scoring_stage_evidence(evidence)
        return _reproducibility_stage_evidence(evidence)
    if stage == "metrics":
        return _metrics_stage_evidence(evidence)
    if stage == "word_conclusion_draft":
        return _word_conclusion_stage_evidence(
            evidence,
            validation_workflow_version=validation_workflow_version,
        )
    keys = STAGE_EVIDENCE_KEYS.get(stage)
    if not keys or not _looks_like_global_agent_evidence(evidence):
        return _slim_evidence_for_llm(evidence)
    return _slim_evidence_for_llm(
        {
            key: evidence.get(key)
            for key in keys
            if evidence.get(key) is not None
        }
    )


def _metrics_stage_evidence(evidence: dict) -> dict:
    if not _looks_like_global_agent_evidence(evidence):
        return _slim_evidence_for_llm(evidence)
    validation_results = evidence.get("validation_results")
    if not isinstance(validation_results, dict):
        return {"validation_results": validation_results}
    return {
        "validation_results": _slim_validation_results_for_llm(
            {
                key: validation_results.get(key)
                for key in METRICS_VALIDATION_RESULT_KEYS
                if key in validation_results
            }
        )
    }


def _pmml_scoring_stage_evidence(evidence: dict) -> dict:
    validation_results = evidence.get("validation_results")
    scoring = _compact_pmml_scoring_evidence(
        evidence.get("pmml_scoring"),
        validation_results.get("pmml_scoring")
        if isinstance(validation_results, dict)
        else None,
    )
    return {"pmml_scoring": scoring}


def _reproducibility_stage_evidence(evidence: dict) -> dict:
    if not _looks_like_global_agent_evidence(evidence):
        return _slim_evidence_for_llm(evidence)
    validation_results = evidence.get("validation_results")
    scoped: dict[str, object] = {}

    notebook_steps = _compact_notebook_steps_for_reproducibility(
        evidence.get("notebook_steps")
    )
    if notebook_steps or "notebook_steps" in evidence:
        scoped["notebook_steps"] = notebook_steps

    notebook_failure = _compact_notebook_failure_evidence(
        evidence.get("notebook_steps")
    )
    if notebook_failure:
        scoped["notebook_failure"] = notebook_failure

    contract = _compact_contract_for_reproducibility(evidence.get("contract"))
    if contract or "contract" in evidence:
        scoped["contract"] = contract

    reproducibility = _compact_reproducibility_evidence(
        evidence.get("reproducibility"),
        validation_results.get("reproducibility") if isinstance(validation_results, dict) else None,
    )
    if reproducibility or "reproducibility" in evidence:
        scoped["reproducibility"] = reproducibility

    return scoped


def _compact_notebook_steps_for_reproducibility(steps: object) -> list[dict]:
    if isinstance(steps, dict):
        steps = steps.get("steps")
    if not isinstance(steps, list):
        return []
    compact = []
    for step in steps[:REPRODUCIBILITY_NOTEBOOK_STEP_LIMIT]:
        if not isinstance(step, dict):
            continue
        compact.append(
            {
                key: step.get(key)
                for key in (
                    "id",
                    "title",
                    "status",
                    "cell_count",
                    "elapsed_seconds",
                    "system",
                )
                if step.get(key) not in (None, "")
            }
        )
    return compact


def _compact_notebook_failure_evidence(notebook_steps: object) -> dict:
    if not isinstance(notebook_steps, dict):
        return {}
    cells = notebook_steps.get("cells")
    if not isinstance(cells, list):
        return {}
    steps_by_id = _notebook_steps_by_id(notebook_steps.get("steps"))
    failed_cells: list[dict] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if str(cell.get("status") or "").lower() != "failed":
            continue
        compact = {
            key: cell.get(key)
            for key in (
                "cell_index",
                "step_id",
                "exception_name",
                "exception_value",
                "traceback_preview",
            )
            if cell.get(key) not in (None, "")
        }
        step = steps_by_id.get(str(cell.get("step_id") or ""))
        if step:
            compact["step_title"] = step.get("title")
        source_preview = str(cell.get("source_preview") or "").strip()
        if source_preview:
            compact["source_preview"] = _truncate_llm_text(
                source_preview,
                NOTEBOOK_FAILURE_SOURCE_CHARS,
            )
        referenced_files = cell.get("referenced_files")
        if isinstance(referenced_files, list) and referenced_files:
            compact["referenced_files"] = [
                str(item)
                for item in referenced_files[:NOTEBOOK_FAILURE_FILE_LIMIT]
                if str(item).strip()
            ]
        access_lines = cell.get("file_access_lines")
        if isinstance(access_lines, list) and access_lines:
            compact["file_access_lines"] = [
                _truncate_llm_text(str(item), NOTEBOOK_FAILURE_SOURCE_CHARS)
                for item in access_lines[:NOTEBOOK_FAILURE_LINE_LIMIT]
                if str(item).strip()
            ]
        failed_cells.append(compact)
        if len(failed_cells) >= NOTEBOOK_FAILURE_CELL_LIMIT:
            break
    if not failed_cells:
        return {}
    return {
        "source": "notebook_execution_progress",
        "read_only": True,
        "failed_cells": failed_cells,
        "interpretation_rule": (
            "回答 Notebook 缺文件或失败 cell 问题时，优先使用 failed_cells 中的 "
            "source_preview、referenced_files 和 file_access_lines；不要只根据错误文本猜测。"
        ),
    }


def _notebook_steps_by_id(steps: object) -> dict[str, dict]:
    if not isinstance(steps, list):
        return {}
    return {
        str(step.get("id") or ""): step
        for step in steps
        if isinstance(step, dict) and step.get("id")
    }


def _compact_contract_for_reproducibility(contract: object) -> dict:
    if not isinstance(contract, dict):
        return {}
    compact = {
        key: contract.get(key)
        for key in (
            "algorithm",
            "target_col",
            "score_col",
            "split_col",
            "time_col",
            "pmml_output_field",
            "score_decimal_places",
        )
        if contract.get(key) not in (None, "")
    }
    feature_columns = contract.get("feature_columns")
    if isinstance(feature_columns, list):
        compact["feature_count"] = len(feature_columns)
        compact["feature_columns_sample"] = feature_columns[
            :REPRODUCIBILITY_CONTRACT_FEATURE_LIMIT
        ]
    return compact


def _slim_evidence_for_llm(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        return evidence
    slim = dict(evidence)
    validation_results = slim.get("validation_results")
    if isinstance(validation_results, dict):
        slim["validation_results"] = _slim_validation_results_for_llm(validation_results)
    return slim


def _word_conclusion_stage_evidence(
    evidence: dict,
    *,
    validation_workflow_version: int | None = None,
) -> dict:
    if not _looks_like_global_agent_evidence(evidence):
        return _slim_evidence_for_llm(evidence)
    validation_results = evidence.get("validation_results")
    scoped: dict[str, object] = {}

    scan = _compact_scan_evidence(evidence.get("scan"))
    if scan:
        scoped["scan"] = scan

    pmml_scoring = _compact_pmml_scoring_evidence(
        evidence.get("pmml_scoring"),
        validation_results.get("pmml_scoring")
        if isinstance(validation_results, dict)
        else None,
    )
    if pmml_scoring:
        scoped["pmml_scoring"] = pmml_scoring

    if validation_workflow_version != 2:
        reproducibility = _compact_reproducibility_evidence(
            evidence.get("reproducibility"),
            validation_results.get("reproducibility")
            if isinstance(validation_results, dict)
            else None,
        )
        if reproducibility:
            scoped["reproducibility"] = reproducibility

    compact_validation = _compact_word_validation_results(
        validation_results,
        validation_workflow_version=validation_workflow_version,
    )
    if compact_validation:
        scoped["validation_results"] = compact_validation

    report_fields = evidence.get("report_fields")
    if report_fields is not None:
        scoped["report_fields"] = report_fields

    visible_summaries = _compact_visible_stage_summaries(
        evidence.get("visible_stage_summaries")
    )
    if visible_summaries:
        scoped["visible_stage_summaries"] = visible_summaries

    return scoped


def _compact_scan_evidence(scan: object) -> dict:
    if not isinstance(scan, dict):
        return {}
    checks = scan.get("checks")
    if not isinstance(checks, list):
        checks = []
    compact_checks = []
    for check in checks[:20]:
        if not isinstance(check, dict):
            continue
        compact_checks.append(
            {
                key: check.get(key)
                for key in ("id", "label", "status", "message")
                if check.get(key) not in (None, "")
            }
        )
    compact = {
        "checks": compact_checks,
        "scan_interpretation": _scan_interpretation(compact_checks),
    }
    notebook_contract = _compact_notebook_contract_evidence(
        scan.get("notebook_contract")
    )
    if notebook_contract:
        compact["notebook_contract"] = notebook_contract
    return compact


def _compact_notebook_contract_evidence(contract: object) -> dict:
    if not isinstance(contract, dict):
        return {}
    compact = {
        key: contract.get(key)
        for key in (
            "read_only",
            "source",
            "sample_df_defined",
            "score_fn_defined",
            "target_col_defined",
            "algorithm_defined",
            "target_col",
            "algorithm",
            "algorithm_raw",
            "algorithm_source",
            "algorithm_valid",
            "algorithm_error",
            "error",
        )
        if contract.get(key) not in (None, "")
    }
    for key in ("missing_names", "invalid_names", "contract_cell_indexes"):
        value = contract.get(key)
        if isinstance(value, list) and value:
            compact[key] = value[:20]
    previews = contract.get("source_previews")
    if isinstance(previews, list) and previews:
        compact["source_previews"] = [
            _truncate_llm_text(str(preview), NOTEBOOK_CONTRACT_SOURCE_PREVIEW_CHARS)
            for preview in previews[:NOTEBOOK_CONTRACT_SOURCE_PREVIEW_LIMIT]
            if str(preview).strip()
        ]
    return compact


def _compact_reproducibility_evidence(*candidates: object) -> dict:
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        compact = {
            key: candidate.get(key)
            for key in ("summary", "sample_size", "seed")
            if candidate.get(key) is not None
        }
        if compact:
            return compact
    return {}


def _compact_pmml_scoring_evidence(*candidates: object) -> dict:
    fields = (
        "schema_version",
        "engine",
        "engine_version",
        "output_field",
        "input_row_count",
        "success_count",
        "failure_count",
        "null_count",
        "non_finite_count",
        "elapsed_seconds",
        "rows_per_second",
        "chunk_size",
        "required_input_count",
        "missing_inputs",
        "status",
        "bounded_errors",
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        compact = {
            key: candidate.get(key)
            for key in fields
            if candidate.get(key) is not None
        }
        if compact:
            return compact
    return {}


def _compact_word_validation_results(
    validation_results: object,
    *,
    validation_workflow_version: int | None = None,
) -> dict:
    if not isinstance(validation_results, dict):
        return {}
    slim = _slim_validation_results_for_llm(validation_results)
    compact: dict[str, object] = {}
    for key in ("model_name", "model_version", "algorithm", "target_type"):
        if slim.get(key) is not None:
            compact[key] = slim.get(key)

    basic_info = _compact_basic_info_for_word(slim.get("basic_info"))
    if basic_info:
        compact["basic_info"] = basic_info

    if validation_workflow_version != 2:
        reproducibility = _compact_reproducibility_evidence(
            slim.get("reproducibility")
        )
        if reproducibility:
            compact["reproducibility"] = reproducibility

    pmml_scoring = _compact_pmml_scoring_evidence(slim.get("pmml_scoring"))
    if pmml_scoring:
        compact["pmml_scoring"] = pmml_scoring

    effectiveness = _compact_effectiveness_for_word(slim.get("effectiveness"))
    if effectiveness:
        compact["effectiveness"] = effectiveness

    overfitting = slim.get("overfitting_check")
    if isinstance(overfitting, dict):
        compact["overfitting_check"] = dict(overfitting)

    stress_test = _compact_stress_test_for_word(slim.get("stress_test"))
    if stress_test:
        compact["stress_test"] = stress_test
    return compact


def _compact_basic_info_for_word(basic_info: object) -> dict:
    if not isinstance(basic_info, dict):
        return {}
    compact: dict[str, object] = {}
    for key in ("sample_period", "split_summary", "monthly_distribution"):
        value = basic_info.get(key)
        if isinstance(value, list):
            compact[key] = _tail_list(value, WORD_CONCLUSION_MONTHLY_LIMIT)
        elif value is not None:
            compact[key] = value
    feature_importance = basic_info.get("feature_importance")
    if isinstance(feature_importance, list):
        compact["feature_importance"] = feature_importance[:WORD_CONCLUSION_FEATURE_LIMIT]
    return compact


def _compact_effectiveness_for_word(effectiveness: object) -> dict:
    if not isinstance(effectiveness, dict):
        return {}
    compact: dict[str, object] = {}
    for key in ("overall", "monthly_ks", "monthly_psi", "psi_stability_table"):
        value = effectiveness.get(key)
        if not isinstance(value, list):
            if value is not None:
                compact[key] = value
            continue
        if key in {"monthly_ks", "monthly_psi"}:
            compact[key] = _tail_list(value, WORD_CONCLUSION_MONTHLY_LIMIT)
        elif key == "psi_stability_table":
            compact[key] = value[:WORD_CONCLUSION_PSI_BIN_LIMIT]
        else:
            compact[key] = value
    return compact


def _compact_stress_test_for_word(stress_test: object) -> dict:
    if not isinstance(stress_test, dict):
        return {}
    compact: dict[str, object] = {}
    baseline = stress_test.get("baseline")
    if isinstance(baseline, dict):
        compact["baseline"] = {
            key: value
            for key, value in baseline.items()
            if key != "bin_table"
        }
    elif baseline is not None:
        compact["baseline"] = baseline
    per_category = stress_test.get("per_category")
    if isinstance(per_category, list):
        compact["per_category"] = [
            _compact_stress_category(item)
            for item in per_category[:WORD_CONCLUSION_STRESS_CATEGORY_LIMIT]
            if isinstance(item, dict)
        ]
    if stress_test.get("status") is not None:
        compact["status"] = stress_test.get("status")
    return compact


def _compact_stress_category(category: dict) -> dict:
    compact = {
        key: value
        for key, value in category.items()
        if key != "bin_table"
    }
    dropped = compact.get("dropped_features")
    if isinstance(dropped, list):
        compact["dropped_features"] = dropped[:WORD_CONCLUSION_FEATURE_LIMIT]
    return compact


def _compact_visible_stage_summaries(summaries: object) -> list[dict]:
    if not isinstance(summaries, list):
        return []
    compact = []
    for item in summaries[-WORD_CONCLUSION_VISIBLE_SUMMARY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        compact.append(
            {
                "stage": str(item.get("stage") or ""),
                "content": _truncate_llm_text(
                    content,
                    WORD_CONCLUSION_VISIBLE_SUMMARY_CHARS,
                ),
            }
        )
    return compact


def _tail_list(value: list, limit: int) -> list:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _slim_validation_results_for_llm(validation_results: dict) -> dict:
    if not isinstance(validation_results, dict):
        return validation_results
    slim = dict(validation_results)
    effectiveness = slim.get("effectiveness")
    if isinstance(effectiveness, dict):
        slim["effectiveness"] = {
            key: value
            for key, value in effectiveness.items()
            if key not in OVERSIZED_EFFECTIVENESS_KEYS
        }
    return slim


def _scan_stage_evidence(evidence: dict) -> dict:
    scoped = dict(evidence)
    checks = scoped.get("checks")
    if not isinstance(checks, list) and isinstance(scoped.get("scan"), dict):
        checks = scoped["scan"].get("checks")
        notebook_contract = _compact_notebook_contract_evidence(
            scoped["scan"].get("notebook_contract")
        )
        if notebook_contract:
            scoped["scan"] = {**scoped["scan"], "notebook_contract": notebook_contract}
    else:
        notebook_contract = _compact_notebook_contract_evidence(
            scoped.get("notebook_contract")
        )
        if notebook_contract:
            scoped["notebook_contract"] = notebook_contract
    scoped["scan_interpretation"] = _scan_interpretation(
        checks if isinstance(checks, list) else []
    )
    return scoped


def _scan_interpretation(checks: list[dict]) -> dict:
    checks_by_label = {
        str(check.get("label") or check.get("name") or ""): check
        for check in checks
        if isinstance(check, dict)
    }
    detected: list[str] = []
    missing: list[str] = []
    for label in SCAN_REQUIRED_CHECK_LABELS:
        check = checks_by_label.get(label)
        if check and _scan_check_passed(check):
            detected.append(label)
        else:
            missing.append(label)
    return {
        "required_materials": list(SCAN_REQUIRED_CHECK_LABELS),
        "detected_required_materials": detected,
        "missing_required_materials": missing,
        "required_materials_complete": not missing,
        "not_required_in_scan_stage": [
            "pickle/pkl 模型",
            "验证样本与评分输出中间结果",
            "KS/PSI/AUC 等效果指标产物",
        ],
        "interpretation_rule": (
            "checks 中 success/passed/通过 表示对应材料或契约已经满足，"
            "不得再描述为缺失。pickle/pkl、评分输出和 KS/PSI/AUC 是可选材料或后续阶段结果，"
            "不属于材料扫描阶段的必需输入。"
        ),
    }


def _scan_check_passed(check: dict) -> bool:
    status = str(check.get("status") or "").strip().lower()
    message = str(check.get("message") or "")
    return (
        status in SCAN_SUCCESS_STATUSES
        or message.startswith("已识别")
        or message.startswith("已定义")
    )


def _guard_stage_summary(*, stage: str, evidence: dict, content: str) -> str:
    if stage != "scan" or not isinstance(evidence, dict):
        return content
    interpretation = _scan_stage_evidence(evidence).get("scan_interpretation", {})
    if not interpretation.get("required_materials_complete"):
        return content
    if not _scan_summary_claims_missing_complete_materials(content):
        return content
    return (
        "材料完备性检查已完成，Notebook、样本数据、PMML 模型、数据字典和 "
        "Notebook RMC 契约均已通过；当前未发现必需材料缺失。"
    )


def _scan_summary_claims_missing_complete_materials(content: str) -> bool:
    text = str(content or "")
    if not text:
        return False
    if re.search(r"(未发现|无).{0,12}(缺少|缺失|缺乏)", text):
        return False
    return bool(
        re.search(
            r"缺少|缺乏|未提供|没有|未识别|未检测到|未检出|建议补充|补充.{0,12}(文件|材料|输出|结果)|缺失",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_global_agent_evidence(evidence: dict) -> bool:
    return any(key in evidence for key in GLOBAL_AGENT_EVIDENCE_KEYS)


def _chat_prompt(
    *,
    task: TaskRecord,
    user_message: str,
    conversation: list[dict],
    evidence: dict,
    memory_context: dict | None = None,
) -> str:
    conversation_memory = _conversation_memory(
        conversation=conversation,
        task=task,
        current_user_message=user_message,
    )
    payload = {
            "stage": "chat",
            "task": _llm_task_meta(task),
            "user_message": _sanitize_llm_text(user_message, task),
            "conversation_memory": conversation_memory,
            "available_evidence": _sanitize_llm_payload(
                _chat_evidence_for_llm(evidence), task
            ),
            "instructions": _chat_instructions(user_message),
        }
    payload = add_memory_to_prompt_payload(payload, memory_context)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _conversation_memory(
    *,
    conversation: list[dict],
    task: TaskRecord,
    current_user_message: str,
) -> dict:
    messages = _conversation_memory_messages(conversation, task)
    previous_messages = messages
    if messages and messages[-1]["role"] == "user" and messages[-1]["content"] == _sanitize_llm_text(
        current_user_message, task
    ):
        previous_messages = messages[:-1]
    previous_user_question = _last_message_content(previous_messages, role="user")
    previous_assistant_answer = _last_message_content(previous_messages, role="assistant")
    kept_messages, omitted_count = _fit_conversation_memory(messages)
    return {
        "scope": "same_agent_task",
        "description": "同一个验证任务内的用户问题、Agent 回复和阶段输出历史，用于理解后续追问。",
        "messages": kept_messages,
        "omitted_message_count": omitted_count,
        "previous_user_question": previous_user_question,
        "previous_assistant_answer": previous_assistant_answer,
        "follow_up_guidance": (
            "如果当前 user_message 是省略主语、承接上一轮或只问“有什么影响/为什么/继续说”的追问，"
            "必须优先按 previous_user_question 和 previous_assistant_answer 中的主题补全问题，"
            "不要扩展到无关验证阶段或泛化成整体验证分析。"
        ),
    }


def _conversation_memory_messages(conversation: list[dict], task: TaskRecord) -> list[dict]:
    messages = []
    for message in conversation:
        content = _truncate_llm_text(
            _sanitize_llm_text(str(message.get("content") or ""), task),
            CONVERSATION_MEMORY_MESSAGE_MAX_CHARS,
        )
        if not content:
            continue
        messages.append(
            {
                "role": str(message.get("role") or ""),
                "stage": str(message.get("stage") or ""),
                "content": content,
            }
        )
    return messages


def _fit_conversation_memory(messages: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    total_chars = 0
    for message in reversed(messages):
        content_length = len(message["content"])
        if kept and (
            len(kept) >= CONVERSATION_MEMORY_MAX_MESSAGES
            or total_chars + content_length > CONVERSATION_MEMORY_MAX_CHARS
        ):
            break
        kept.append(message)
        total_chars += content_length
    kept.reverse()
    return kept, max(0, len(messages) - len(kept))


def _chat_evidence_for_llm(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        return evidence
    if not _looks_like_global_agent_evidence(evidence):
        return _slim_evidence_for_llm(evidence)
    scoped = _word_conclusion_stage_evidence(evidence)
    notebook_failure = _compact_notebook_failure_evidence(
        evidence.get("notebook_steps")
    )
    if notebook_failure:
        scoped["notebook_failure"] = notebook_failure
    return scoped


def _last_message_content(messages: list[dict], *, role: str) -> str:
    for message in reversed(messages):
        if message["role"] == role and message["content"]:
            return message["content"]
    return ""


def _truncate_llm_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 8)].rstrip() + "…[截断]"


def _chat_instructions(user_message: str) -> str:
    instructions = (
        _llm_task_reference_instruction()
        + "直接回答用户当前问题。若用户只是问候、寒暄或普通对话，应自然、简短回应，"
        "不要给出固定启动口令或流程引导。若 conversation 显示用户连续多轮明显偏离当前模型验证任务，"
        "可以简短回答后提醒你主要用于解释验证结果、PMML、指标含义和报告结论。"
        "若问题是 PMML、验证流程、指标含义或当前任务结果解释，"
        "请以信贷风控模型验证专家身份给出清晰中文说明；能引用平台证据时引用证据，"
        "没有证据时明确说明这是通用解释。若用户提到界面里的图、表、阶段文字或结论，"
        "优先使用 available_evidence.validation_results、report_fields.metric_values、"
        "report_fields.text_values 和 visible_stage_summaries 作答；图表应按其底层结构化数据解释，"
        "不要假装直接看到了像素图。"
        "若用户询问 Notebook、RMC 契约字段或 RMC_ALGORITHM 当前值，优先使用 "
        "available_evidence.scan.notebook_contract；该证据来自平台对 Notebook 的只读静态扫描，"
        "可以说明原始填写值、归一化算法和错误原因，但不得声称修改过 Notebook，也不得要求用户手动打开"
        "Notebook 来查找已在 evidence 中提供的字段。"
        "若用户询问 Notebook 执行失败、缺少哪个文件、或某个 cell 为什么报错，优先使用 "
        "available_evidence.notebook_failure.failed_cells 中的失败 cell 源码摘要、referenced_files "
        "和 file_access_lines；不要只根据 FileNotFoundError 文本、任务名、文件名或历史经验猜测。"
        "同一验证任务内的 conversation_memory 是任务内会话记忆。若当前 user_message 是承接式追问、"
        "省略主语或只问影响/原因/继续说明，必须先根据 conversation_memory.previous_user_question "
        "和 previous_assistant_answer 补全主题；回答时要围绕上一轮用户问题的主题，不要自动扩展到"
        "无关验证阶段或整体验证结论。"
        "如果回答涉及 KS、PSI、AUC、稳定性或区分能力等指标，必须按以下业务口径解释，不能脱离模型场景："
        + RISK_METRIC_INTERPRETATION_GUIDANCE
    )
    if _is_greeting_message(user_message):
        instructions += "当前 user_message 是问候或寒暄，直接打招呼并表示可以继续回答即可。"
    return instructions


def _llm_task_meta(task: TaskRecord) -> dict:
    status_message = task.status_message
    if status_message is not None:
        status_message = _sanitize_llm_text(status_message, task)
    return {
        "model_display_name": _model_display_name(task),
        "model_name": task.model_name or "当前模型",
        "model_version": task.model_version,
        "algorithm": task.algorithm,
        "status": task.status.value,
        "status_message": status_message,
    }


def _model_display_name(task: TaskRecord) -> str:
    name = task.model_name or "当前模型"
    version = str(task.model_version or "").strip()
    return f"{name}（{version}）" if version else name


def _llm_task_reference_instruction() -> str:
    return (
        "涉及当前任务或当前模型时，必须使用 task.model_display_name / 模型名称来称呼，"
        "不要输出、复述或引用内部任务 ID。"
    )


def _sanitize_llm_payload(value: object, task: TaskRecord) -> object:
    if isinstance(value, str):
        return _sanitize_llm_text(value, task)
    if isinstance(value, list):
        return [_sanitize_llm_payload(item, task) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_llm_payload(item, task) for item in value]
    if isinstance(value, dict):
        # Only sanitize values, never keys. Keys are structural field names; a
        # key that happens to contain the task id should not be rewritten into a
        # display name (which would corrupt the payload schema seen by the LLM).
        return {
            key: _sanitize_llm_payload(item, task)
            for key, item in value.items()
        }
    return value


def _sanitize_llm_text(value: str, task: TaskRecord) -> str:
    task_id = str(task.id or "")
    if not task_id:
        return value
    return value.replace(task_id, _model_display_name(task))


def _stage_instructions(
    stage: str,
    *,
    validation_workflow_version: int | None = None,
) -> str:
    reference_instruction = _llm_task_reference_instruction()
    if stage == "agent_start":
        return (
            reference_instruction
            + "最多 2 句。先说明你将以信贷风控模型验证专家身份启动任务，"
            "再说明下一步会调用材料识别工具；不要声称材料已经完成识别。"
        )
    if stage == "scan":
        return (
            reference_instruction
            + "只针对当前材料完备性阶段，最多 2 句，说明材料识别、材料缺失或 Notebook RMC 契约检查状况。"
            "如果 evidence.scan.notebook_contract 或 evidence.notebook_contract 存在，"
            "它是平台只读静态扫描得到的 Notebook 契约摘要，可用于说明 RMC_ALGORITHM、"
            "RMC_TARGET_COL 等字段的原始值和归一化结果；不得声称无法读取这些已提供证据。"
            "必须以 evidence.scan_interpretation.required_materials_complete 和 "
            "missing_required_materials 为准；当 required_materials_complete=true 时，"
            "只能说明 Notebook、样本数据、PMML 模型、数据字典和 Notebook RMC 契约均已满足，"
            "不得说缺少、未检测到模型文件、PMML、pickle/pkl、验证样本与评分输出或 KS/PSI/AUC 中间结果，"
            "也不要建议补充这些已满足或非必需材料。"
            "pickle/pkl 不是当前扫描必需材料；KS/PSI/AUC 是后续效果稳定性阶段计算结果，"
            "不是材料扫描阶段的输入文件。"
            "不要分析分数一致性、AUC、KS、PSI、压力测试或最终报告结论。"
        )
    if stage == "reproducibility":
        if validation_workflow_version == 2:
            return (
                reference_instruction
                + "只针对当前 PMML 打分测试阶段，分为“结论、评分覆盖、异常情况、性能、建议”。"
                "只使用 evidence.pmml_scoring，解读 input_row_count、success_count、failure_count、"
                "null_count、non_finite_count、missing_inputs、status、elapsed_seconds 和 rows_per_second。"
                "不得声称执行过 Notebook、代码模型评分或代码模型与 PMML 分数一致性比较；"
                "不得建议当前或后续补做 Notebook 模型执行、代码模型评分、"
                "代码模型与 PMML 分数一致性验证；"
                "不要分析材料完备性、AUC、KS、PSI、分箱、逐月稳定性、模型压力测试或报告输出，"
                "不要给出整体验证结论。"
            )
        return (
            reference_instruction
            + "只针对当前模型可复现性/分数一致性阶段，分为“结论、证据、风险含义、建议”。"
            "只使用 evidence.reproducibility、contract 和 notebook_steps 中的证据。"
            "不要分析材料完备性，不要分析 AUC、KS、PSI、分箱、逐月稳定性、压力测试或报告输出，"
            "不要给出整体验证结论，也不要输出整份验证工作底稿结构。"
        )
    if stage == "metrics":
        return (
            reference_instruction
            + "只针对当前效果与稳定性阶段，分为“总体判断、效果表现、稳定性表现、压力测试风险、建议”。"
            "压力测试风险必须完整覆盖高/中/低风险分层或明确说明证据不足，并用一句完整结论收束；"
            "不得停在半句话、项目符号中途或只有“KS 降幅”等未完成表述。"
            "不要回顾材料完备性或分数一致性的执行过程，不要生成最终报告综合结论。"
            "解释 KS、PSI、AUC、稳定性或区分能力时必须按以下业务口径判断，不能脱离模型场景："
            + RISK_METRIC_INTERPRETATION_GUIDANCE
        )
    if stage == "report":
        return (
            reference_instruction
            + "只针对当前报告输出阶段，最多 2 句，说明报告产物生成、预览或下载状态。"
            "不要重新分析材料完备性、分数一致性或效果稳定性指标。"
        )
    if stage == "word_conclusion_draft":
        if validation_workflow_version == 2:
            return (
                reference_instruction
                + "生成最终 Word 报告中的三段候选文字。只能使用 evidence 中已给出的 PMML 打分测试、"
                "效果与稳定性指标、模型压力测试分层和阶段总结；不得编造未提供的 KS、AUC、PSI、样本量、"
                "数据源名称或监管结论，也不得声称执行过 Notebook、代码模型评分或分数一致性比较。"
                "输出必须是 JSON 对象，且必须完整包含三段键值："
                "TEXT:pressure_test_summary、TEXT:pressure_impact_recommendation、"
                "TEXT:final_validation_conclusion。TEXT:pressure_test_summary 必须说明模型压力测试目的、"
                "方法和观察到的高/中/低风险数据源分层；证据不足时明确说明无法完成某一档分层。"
                "TEXT:pressure_impact_recommendation 必须围绕上述风险分层给出监控、替代、降级、"
                "人工复核或上线限制建议。TEXT:final_validation_conclusion 必须比前两段更完整，"
                "建议 1 到 2 个自然段，直接评价区分效果、样本外稳定性、过拟合风险、"
                "模型压力测试主要发现和最终审慎判断；最多用一个短句说明 PMML 部署可用，"
                "不得写可直接部署或可直接投产。"
                "不得复述材料扫描、材料完备性、验证输入契约、PMML 打分样本量或耗时、"
                "平台执行步骤、报告产出状态、最终定稿阶段、投产前审阅安排等流程信息；"
                "如果 evidence.visible_stage_summaries 中已有模型效果稳定性解读，必须吸收其模型评价要点；"
                "PMML 相关证据最多用于判断是否可简短表述为 PMML 部署可用。"
            )
        return (
            reference_instruction
            + "生成最终 Word 报告中的三段候选文字。只能使用 evidence 中已给出的结构化指标、"
            "复现结论、压力测试分层和阶段总结；不得编造未提供的 KS、AUC、PSI、样本量、"
            "数据源名称或监管结论。输出必须是 JSON 对象，且必须完整包含三段键值："
            "TEXT:pressure_test_summary、TEXT:pressure_impact_recommendation、"
            "TEXT:final_validation_conclusion。TEXT:pressure_test_summary 必须说明压力测试目的、"
            "方法和观察到的高/中/低风险数据源分层；证据不足时明确说明无法完成某一档分层。"
            "TEXT:pressure_impact_recommendation 必须围绕上述风险分层给出监控、替代、降级、"
            "人工复核或上线限制建议。TEXT:final_validation_conclusion 必须比前两段更完整，"
            "建议 1 到 2 个自然段，覆盖开发过程或材料完备性、Notebook 可复现性、分数一致性、"
            "区分效果、稳定性、压力测试主要发现、报告产出状态和最终审慎判断；"
            "如果 evidence.visible_stage_summaries 中已有复现性或效果稳定性解读，必须吸收其要点，"
            "不能退化成一句泛泛结论。"
        )
    if stage == "failure":
        return (
            reference_instruction
            + "分为“失败阶段、直接原因、可能原因、下一步”。如果 evidence.notebook_failure 存在，"
            "必须优先基于其中的失败 cell 源码摘要、referenced_files 和 file_access_lines 判断 Notebook "
            "实际引用了哪些文件；不要只根据错误文本猜测，也不要把任务名或模型名误当成缺失文件。"
        )
    return reference_instruction + "生成审慎、专业、基于证据的中文说明。"


def _parse_conclusion_json(content: str) -> dict[str, str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM 返回不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("word conclusion response must be a JSON object")
    values = {
        key: str(payload.get(key) or "").strip()
        for key in REQUIRED_AGENT_REPORT_KEYS
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError("word conclusion response missing keys: " + ", ".join(missing))
    return values


def _validate_v2_word_conclusions(values: dict[str, str]) -> None:
    conclusion = values.get("TEXT:final_validation_conclusion", "")
    for pattern in _V2_FINAL_CONCLUSION_FORBIDDEN_PATTERNS:
        if pattern.search(conclusion):
            raise ValueError(
                "V2 final validation conclusion contains forbidden process narration"
            )
