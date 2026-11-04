from __future__ import annotations

from collections.abc import Callable
import json
import re

from marvis.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    RISK_METRIC_INTERPRETATION_GUIDANCE,
)
from marvis.agent.validation_narrative import (
    generate_v2_metrics_summary, generate_v2_sections, has_report_revision_context,
    validate_v2_word_conclusions as _validate_v2_word_conclusions,
)
from marvis.llm_prompts import AGENT_SYSTEM_PROMPT as _AGENT_PROMPT_SPEC
from marvis.llm_prompts import WORD_CONCLUSION_SYSTEM_PROMPT as _WORD_PROMPT_SPEC
from marvis.agent.instruction_router import route_instruction
from marvis.agent.semantic_authorization import review_semantic_authorization
from marvis.model_algorithms import (
    is_platform_default_training_description,
    model_training_report_text,
)
from marvis.repositories.tasks import (
    AGENT_REPORT_CONCLUSION_KEYS, AGENT_REPORT_NARRATIVE_KEYS, AGENT_REPORT_WRITABLE_KEYS,
)
from marvis.domain import TaskRecord
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.report_texts import report_text_values_from_results
from marvis.validation.lift_ranking import assess_lift_ranking
from marvis.validation.overfitting import overfitting_check_from_validation_results
from marvis.validation.results import validation_results_from_dict
from marvis.validation.stress_risk import (
    stress_ks_risk,
    stress_psi_risk,
    stress_risk_label,
    worst_stress_risk,
)
from marvis.validation_report_copy import narrative_report_values
from marvis.agent_memory.prompting import attach_memory_metadata


REQUIRED_AGENT_REPORT_KEYS = tuple(sorted(AGENT_REPORT_CONCLUSION_KEYS))
_V2_PMML_SCORING_LEGACY_PATTERN = re.compile(
    r"可复现(?:性)?|(?:分数|评分|部署)?一致性|Notebook\s*模型|代码模型|原始代码模型",
    re.IGNORECASE,
)
START_VALIDATION_GUIDANCE_FRAGMENTS = (
    "可以协助启动验证",
    "需要开始时，请输入",
    "请输入“开始验证”",
    '请输入"开始验证"',
    "输入“开始验证”",
    '输入"开始验证"',
)
GREETING_CHAT_FALLBACK = (
    "你好，我在。你可以直接问我验证结果、指标含义、PMML 或报告结论，"
    "也可以告诉我下一步想处理什么。"
)
STAGE_SUMMARY_MAX_TOKENS = {
    "metrics": 4096,
}
METRICS_SUMMARY_REQUIRED_SECTIONS = (
    "总体判断",
    "效果表现",
    "稳定性表现",
    "压力测试风险",
    "建议",
)
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
_VALIDATION_AUTHORIZATION_QUESTION = re.compile(
    r"[?？]|吗(?:[\s。.!！]*$)|呢(?:[\s。.!！]*$)|为什么|怎么(?:样)?|如何|"
    r"什么是|是什么|什么(?:意思|影响|原因|作用)|有什么(?:影响|区别|含义)|"
    r"要不要|需不需要|能不能|可不可以|行不行|好不好|对不对|是不是",
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


def review_validation_instruction_authorization(
    model_profile: dict,
    *,
    gate_context: str,
    instruction: str,
) -> dict | None:
    """Return audited evidence only for a two-pass, fail-closed authorization.

    Legacy model validation does not have a PlanDriver gate object, but Agent mode
    still exposes a free-text confirmation seam.  Reuse the same router and
    independent semantic reviewer as PlanDriver while keeping the executable
    decision parameter-free: a condition, question, adjustment, rejection,
    malformed reply, or client failure can never become an advance command.
    """

    normalized_instruction = str(instruction or "").strip()
    if (
        not model_profile
        or not normalized_instruction
        or _VALIDATION_AUTHORIZATION_QUESTION.search(normalized_instruction)
        or _is_greeting_message(normalized_instruction)
    ):
        return None
    client = OpenAICompatibleLLMClient(model_profile)
    try:
        route = route_instruction(
            client,
            gate_context=gate_context,
            instruction=instruction,
            strict_contract=True,
        )
    except Exception:
        return None
    if (
        route.get("action") != "confirm"
        or route.get("confidence") != "high"
        or route.get("explicit_authorization") is not True
        or bool(route.get("constraint"))
        or bool(route.get("params"))
    ):
        return None
    review = review_semantic_authorization(
        client,
        gate_context=gate_context,
        instruction=instruction,
        proposed_params={},
    )
    if not review.authorized:
        return None
    return {
        "evidence_quote": review.evidence_quote,
        "reason": review.reason,
        "confidence": review.confidence,
        "route_reason": str(route.get("reason") or ""),
    }


def _compact_agent_command(content: str) -> str:
    return "".join(str(content or "").strip().lower().split()).strip("。.!！?？")


def is_agent_material_reselection_intent(content: str) -> bool:
    compact = _compact_agent_command(content)
    if not compact:
        return False
    selection_markers = (
        "重新选择",
        "重新选",
        "重选",
        "更换",
        "换一下",
        "换一个",
        "重新绑定",
        "修改材料选择",
        "修改文件选择",
    )
    material_markers = (
        "材料",
        "验证文件",
        "notebook",
        "样本",
        "pmml",
        "数据字典",
    )
    return any(marker in compact for marker in selection_markers) and any(
        marker in compact for marker in material_markers
    )


def is_agent_report_revision_intent(content: str) -> bool:
    from marvis.llm_outbound_revision import report_revision_button_field

    # The draft card's「请 Agent 修订」button names its field explicitly.
    if report_revision_button_field(content) is not None:
        return True
    compact = _compact_agent_command(content)
    if not compact:
        return False
    revision_markers = (
        "修订",
        "修正",
        "更正",
        "纠正",
        "修改",
        "调整",
        "改写",
        "改成",
        "改为",
        "重写",
        "替换",
        "删除",
        "删掉",
        "不要提",
        "不再提",
        "只评价",
        "只保留",
        "补充",
        "补上",
        "补全",
        "写上",
        "加上",
        "增加",
        "添加",
        "完善",
    )
    report_markers = (
        "报告",
        "草稿",
        "三段",
        "结论",
        "压力测试总结",
        "压力影响建议",
        "最终验证结论",
        "概述",
        "适用范围",
        "坏样本",
        "好样本",
        "样本人群",
        "训练说明",
        "word",
        "excel",
        "支用",
        "授信",
    )
    return any(marker in compact for marker in revision_markers) and any(
        marker in compact for marker in report_markers
    )


def agent_rerun_stage(content: str) -> str | None:
    compact = _compact_agent_command(content)
    if not compact:
        return None
    if is_agent_material_reselection_intent(content):
        return None
    if is_agent_report_revision_intent(content):
        return "word_conclusion_draft"
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
    negated_phrases = (
        "不要停止",
        "不用停止",
        "无需停止",
        "别停止",
        "先别停止",
        "不要取消",
        "不用取消",
        "别取消",
        "先别取消",
        "不要中止",
        "别中止",
    )
    if any(phrase in text for phrase in negated_phrases):
        return False
    if re.search(r"[?？]|为什么|为何|怎么|如何|是否|有没有|停止了没|停止了吗", text):
        return False
    if re.search(
        r"(?:"
        r"(?:到|等到|待|运行到|执行到).{1,40}(?:处|后|时)?"
        r"|在.{1,40}(?:处|时)"
        r")(?:停止|停下|暂停|中止)",
        text,
    ):
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


AGENT_REPORT_CONCLUSION_LABELS = {
    "TEXT:final_validation_conclusion": "最终验证结论",
    "TEXT:pressure_test_summary": "压力测试总结",
    "TEXT:pressure_impact_recommendation": "压力影响建议",
}


def agent_conclusions_confirmed(values: dict[str, str]) -> bool:
    return all(str(values.get(key) or "").strip() for key in REQUIRED_AGENT_REPORT_KEYS)


def missing_agent_conclusion_labels(values: dict | None) -> list[str]:
    """Display names of required report paragraphs that are still empty."""
    row = values if isinstance(values, dict) else {}
    return [
        AGENT_REPORT_CONCLUSION_LABELS[key]
        for key in ("TEXT:final_validation_conclusion", "TEXT:pressure_test_summary",
                    "TEXT:pressure_impact_recommendation")
        if not str(row.get(key) or "").strip()
    ]


def latest_report_draft_context(messages: list[dict]) -> dict:
    """Read the current editable narrative, including deliberate empty values.

    A streaming placeholder has no draft values yet; it must not hide the saved
    draft that a rewrite is based on. An incomplete saved draft is still the
    current draft and must never fall back to an older confirmable version.
    """
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("stage") == "word_conclusion_confirmed":
            return {}
        if message.get("stage") != "word_conclusion_draft":
            continue
        metadata = message.get("metadata") or {}
        values = metadata.get("draft_values")
        revision = metadata.get("report_revision")
        if not isinstance(values, dict):
            continue
        if not isinstance(revision, int) or isinstance(revision, bool):
            return {}
        return {
            "message_id": message.get("id"),
            "report_revision": revision,
            "draft_edit_revision": metadata.get("draft_edit_revision", 0),
            "text_values": {
                key: value for key, value in values.items()
                if key in AGENT_REPORT_WRITABLE_KEYS and isinstance(value, str)
            },
        }
    return {}


def saved_report_narrative_context(values: dict, revision: int) -> dict:
    """Saved wording is a revision baseline, never measured validation evidence."""
    return {
        "report_revision": revision,
        "text_values": {
            key: value for key, value in values.items()
            if key in AGENT_REPORT_WRITABLE_KEYS and isinstance(value, str)
        },
    }


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
    """Validation stage summary. It feeds the Word conclusion draft
    (``visible_stage_summaries``), and reports keep their language (D-10), so
    the reply-language rule is not applied to these model calls."""

    from marvis.ui_locale import use_reply_language

    with use_reply_language(None):
        return _summarize_stage(
            task=task, stage=stage, evidence=evidence, memory_context=memory_context,
            model_profile=model_profile, fallback=fallback, on_delta=on_delta,
        )


def _summarize_stage(
    *,
    task: TaskRecord,
    stage: str,
    evidence: dict,
    memory_context: dict | None = None,
    model_profile: dict,
    fallback: str,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    metrics_v2 = stage == "metrics" and task.validation_workflow_version == 2
    metrics_v1 = stage == "metrics" and not metrics_v2
    scoring_v2 = stage == "reproducibility" and task.validation_workflow_version == 2
    comparison_v1 = stage == "reproducibility" and not scoring_v2
    scan_stage = stage == "scan"
    typed_stage = metrics_v2 or metrics_v1 or scoring_v2 or comparison_v1 or scan_stage
    guard_before_emit = metrics_v1 or scoring_v2 or comparison_v1 or scan_stage
    if not typed_stage:
        # Only stages with a typed structural adapter may reach a model. The
        # former plain-string prompt builder was always refused by the client
        # before any request; keep that outcome without building the text.
        return fallback, {"llm_error": "此调用尚未提供已批准的结构摘要，未发送模型请求。", "fallback": True}
    # Native aggregates must reach their typed adapter before legacy text
    # compaction or sanitization can truncate periods or change numeric values.
    prompt = {"stage": stage, "evidence": evidence}
    truncated = False
    client = _client(model_profile)
    history = None
    try:
        if stage == "metrics":
            from marvis.llm_outbound_memory import model_history_context
            history = model_history_context(task=task, evidence=evidence, memory_context=memory_context)
            prompt["model_history"] = history
        if scoring_v2:
            from marvis.llm_outbound_validation import pmml_scoring_prompt
            prompt = pmml_scoring_prompt(evidence)
        elif comparison_v1:
            from marvis.llm_outbound_validation import reproducibility_prompt
            prompt = reproducibility_prompt(evidence)
        elif scan_stage:
            from marvis.llm_outbound_scan import scan_prompt
            prompt = scan_prompt(evidence, workflow_version=task.validation_workflow_version)
        elif metrics_v1:
            from marvis.llm_outbound_validation import validation_metrics_summary_prompt
            prompt = validation_metrics_summary_prompt(evidence, history=history)
        if metrics_v2:
            content = generate_v2_metrics_summary(client, prompt, truncated=truncated)
            if on_delta:
                on_delta(content)
        else:
            content = client.complete(
                system_prompt=AGENT_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=STAGE_SUMMARY_MAX_TOKENS.get(stage),
                on_delta=None if guard_before_emit else on_delta,
                stream=not guard_before_emit,
                truncated=truncated,
                caller=f"validation_stage_{stage}",
                prompt_name=_AGENT_PROMPT_SPEC.name,
                prompt_version=_AGENT_PROMPT_SPEC.version,
            )
    except (LLMClientError, ValueError) as exc:
        guarded_fallback = (
            _sectioned_metrics_summary(fallback, fallback)
            if stage == "metrics"
            else fallback
        )
        return guarded_fallback, {"llm_error": str(exc), "fallback": True}
    cleaned = _strip_agent_response_preamble(content or fallback)
    metadata = {"fallback": False}
    if stage == "metrics":
        missing_sections = _missing_metrics_summary_sections(cleaned)
        if missing_sections:
            try:
                from marvis.llm_outbound_validation import validation_metrics_summary_prompt
                repair_prompt = validation_metrics_summary_prompt(evidence, missing_sections=missing_sections, history=history)
                repaired_content = client.complete(
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    user_prompt=repair_prompt,
                    temperature=0.1,
                    max_tokens=STAGE_SUMMARY_MAX_TOKENS.get(stage),
                    stream=False,
                    truncated=truncated,
                    caller=f"validation_stage_{stage}_repair",
                    prompt_name=_AGENT_PROMPT_SPEC.name,
                    prompt_version=_AGENT_PROMPT_SPEC.version,
                )
            except (LLMClientError, ValueError) as exc:
                repaired_content = ""
                metadata["llm_repair_error"] = str(exc)
            repaired = _strip_agent_response_preamble(repaired_content or "")
            if repaired and not _missing_metrics_summary_sections(repaired):
                cleaned = repaired
                metadata["summary_repaired"] = True
            else:
                cleaned = _sectioned_metrics_summary(cleaned, fallback)
                metadata["fallback"] = True
                metadata["summary_repair_failed"] = True
    guarded = _guard_stage_summary(
        task=task,
        stage=stage,
        evidence=evidence,
        content=cleaned,
        fallback=fallback,
    )
    if guarded != cleaned:
        metadata["guarded_stage_summary"] = True
        if stage == "scan":
            metadata["guarded_scan_summary"] = True
    if guard_before_emit and on_delta:
        on_delta(guarded)
    if typed_stage:
        metadata.update(history.metadata(use_reason=stage) if history else {"history_comparison_available": False})
        return guarded, metadata
    return guarded, attach_memory_metadata(metadata, memory_context, use_reason=stage)


def _missing_metrics_summary_sections(content: str) -> list[str]:
    text = str(content or "")
    missing: list[str] = []
    for section in METRICS_SUMMARY_REQUIRED_SECTIONS:
        heading = re.compile(
            rf"(?m)^\s{{0,3}}(?:#{{1,6}}\s*|[-+*]\s+|\d+[.、)]\s*)?"
            rf"(?:\*\*|__)?{re.escape(section)}"
            r"(?:\*\*|__)?\s*(?::|：|$)",
        )
        if not heading.search(text):
            missing.append(section)
    return missing


def _sectioned_metrics_summary(content: str, fallback: str) -> str:
    overall = str(content or fallback or "平台指标已生成，请结合页面证据复核。").strip()
    return (
        f"总体判断\n{overall}\n\n"
        "效果表现\n请以平台已计算的 KS、AUC 和样本分层结果为准。\n\n"
        "稳定性表现\n请以平台已计算的 PSI、逐期表现和样本外结果为准。\n\n"
        "压力测试风险\n请以平台生成的高、中、低风险分层和对应数据源影响为准。\n\n"
        "建议\n结合上述确定性指标与压力测试明细进行持续监测和人工复核。"
    )


def compose_agent_start_message(
    *,
    task: TaskRecord,
    model_profile: dict,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    from marvis.llm_outbound_validation import validation_start_prompt
    prompt = validation_start_prompt(workflow_version=task.validation_workflow_version)
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
            caller="validation_agent_start",
            prompt_name=_AGENT_PROMPT_SPEC.name,
            prompt_version=_AGENT_PROMPT_SPEC.version,
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
    from marvis.llm_outbound_failure import failure_prompt

    # Keep actionable source diagnostics on the local display path. The model
    # only receives positions, statuses and approved error categories.
    diagnostics = [f"失败阶段：{stage}", f"直接原因：{error}"]
    notebook = evidence.get("notebook_steps") if isinstance(evidence, dict) else None
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("status") != "failed":
                continue
            location = {}
            if type(cell.get("cell_index")) is int:
                location["cell_index"] = cell["cell_index"]
            references = cell.get("referenced_files")
            if isinstance(references, list):
                location["referenced_files"] = [name for name in references if isinstance(name, str)]
            if location:
                diagnostics.append("本地失败位置：" + json.dumps(location, ensure_ascii=False))
    local_diagnostics = "\n".join(diagnostics)
    fallback = local_diagnostics + "\n可能原因：请检查该阶段输入材料、执行环境和上游产物。\n下一步：修正后重新从失败阶段执行。"
    try:
        prompt = failure_prompt(stage=stage, error=error, evidence=evidence,
                                workflow_version=task.validation_workflow_version)
        content = _client(model_profile).complete(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.1,
            stream=False,
            caller="validation_failure_summary",
            prompt_name=_AGENT_PROMPT_SPEC.name,
            prompt_version=_AGENT_PROMPT_SPEC.version,
        )
    except (LLMClientError, ValueError) as exc:
        return fallback, {"llm_error": str(exc), "fallback": True, "history_comparison_available": False}
    explanation = _strip_agent_response_preamble(content or "")
    if not explanation:
        return fallback, {"fallback": True, "empty_llm_response": True, "history_comparison_available": False}
    combined = local_diagnostics + "\n\n分析与建议：\n" + explanation
    if on_delta:
        on_delta(combined)
    return combined, {"fallback": False, "history_comparison_available": False}


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
    from marvis.llm_outbound import OutboundContextError
    from marvis.llm_outbound_chat import chat_prompt
    dictionary_answer = _dictionary_requirement_answer(user_message, evidence)
    if dictionary_answer is not None:
        # The platform's own contract answers this; a model must not invent
        # extra "required" dictionary columns.
        if on_delta:
            on_delta(dictionary_answer)
        return dictionary_answer, {"fallback": False, "deterministic_answer": "dictionary_requirements",
                                   "history_comparison_available": False}
    fallback = _chat_fallback_for_message(user_message)
    # LLM-5: memory injection is one of the three named highest-volume prompt
    # touch points; surface the truncation flag on this call's audit record.
    try:
        prompt, history = chat_prompt(task=task, user_message=user_message, conversation=conversation,
                                     evidence=evidence, memory_context=memory_context)
        raw_content = _client(model_profile).complete(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            on_delta=None,
            stream=False,
            truncated=False,
            caller="agent_chat",
            prompt_name=_AGENT_PROMPT_SPEC.name,
            prompt_version=_AGENT_PROMPT_SPEC.version,
        )
        content = (raw_content or "").strip()
        # UI-6: the trailing 「建议回复：…」 line is metadata, never prose.
        from marvis.agent.suggested_reply import split_suggested_reply

        content, raw_suggestion = split_suggested_reply(content)
        content = (content or "").strip()
    except OutboundContextError as exc:
        return str(exc), {"llm_error": str(exc), "fallback": True, "needs_clarification": True,
                          "history_comparison_available": False}
    except (LLMClientError, ValueError):
        return fallback, {"llm_error": "llm_request_failed", "fallback": True,
                          "needs_clarification": False, "history_comparison_available": False}
    if not content:
        return fallback, {"fallback": True, "empty_llm_response": True}
    if (not json.loads(history.payload_json)["records"]
            and re.search(
                r"历史(?:同类)?模型|(?:上一版|上次|以前的|之前的)模型|对比历史|历史\s*(?:KS|AUC|PSI)|"
                r"(?:和|比|相比|对比).{0,12}(?:之前|以前|上一版|上次)|"
                r"(?:previous|historical)\s+(?:model|performance|KS|AUC|PSI)",
                user_message, re.IGNORECASE,
            )):
        guarded = "当前没有可直接对照的认证历史模型指标，尚不能判断较上一版是否改善。请提供同口径、同分层的历史证据；本次指标以平台结果为准。"
        if on_delta:
            on_delta(guarded)
        return guarded, {"fallback": False, "guarded_history_comparison": True, **history.metadata(use_reason="chat")}
    if _is_greeting_message(user_message) and _looks_like_start_validation_guidance(
        content
    ):
        return fallback, {"fallback": True, "llm_response_replaced": True}
    content = _strip_agent_response_preamble(content)
    if on_delta:
        on_delta(content)
    from marvis.agent.suggested_reply import attach_suggested_reply

    return content, attach_suggested_reply({"fallback": False, **history.metadata(use_reason="chat")},
                                           raw_suggestion, user_message=user_message)


_DICTIONARY_SUBJECT = re.compile(r"数据字典|字典文件|特征元数据")
_DICTIONARY_QUESTION = re.compile(
    r"哪些列|什么列|几列|哪几列|必填|需要.{0,6}列|要.{0,4}列|列名|什么要求|有什么要求|格式要求|"
    r"不能用|不可用|用不了|不完整|缺什么|缺了什么|缺哪|哪里不对|有什么问题|不合格|不通过"
)


def _dictionary_requirement_answer(user_message: str, evidence: dict | None) -> str | None:
    """Answer data-dictionary questions from the platform contract and checks."""
    from marvis.validation_report_copy import METADATA_REASON_LABELS, dictionary_requirement_text

    text = str(user_message or "")
    if not (_DICTIONARY_SUBJECT.search(text) and _DICTIONARY_QUESTION.search(text)):
        return None
    diagnostics = evidence.get("material_diagnostics") if isinstance(evidence, dict) else None
    if not isinstance(diagnostics, dict):
        return None
    rows = [row for row in diagnostics.get("data_dictionary") or [] if isinstance(row, dict)]
    lines = []
    usable = [row for row in rows if row.get("status") == "compatible"]
    for row in rows:
        name = str(row.get("name") or "数据字典")
        if row.get("status") == "compatible":
            lines.append(f"- {name}：与所选 PMML 匹配，可以使用。")
        elif row.get("status") == "incompatible":
            reasons = [METADATA_REASON_LABELS[code] for code in row.get("reasons") or [] if code in METADATA_REASON_LABELS]
            lines.append(f"- {name}：不能用，" + ("；".join(reasons) or "内容不符合数据字典要求") + "。")
        else:
            lines.append(f"- {name}：平台未能自动检查，请人工确认。")
    if not lines:
        lines.append("- 材料目录里没有找到数据字典文件。")
    parts = ["平台对当前材料的数据字典检查结果：", *lines, "", dictionary_requirement_text()]
    if not usable:
        parts.append("补齐后点「重新选择验证材料」，在弹窗里用「上传补充文件」传入，选中新字典再重新扫描。")
    return "\n".join(parts)


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
    return humanize_internal_terms(text.strip() or original)


# Internal evidence keys and tool names stay in metadata; user text gets labels.
_INTERNAL_TERM_LABELS = (
    ("material_data_dictionary", "数据字典"),
    ("material_model_pmml", "PMML 模型"),
    ("material_notebook", "Notebook"),
    ("material_sample", "样本数据"),
    ("validation_input_contract", "验证输入契约"),
    ("notebook_contract", "Notebook RMC 契约"),
    ("missing_or_unsatisfied_checks", "未满足的检查项"),
    ("satisfied_checks", "已满足的检查项"),
    ("required_checks", "必需检查项"),
    ("failed_check_count", "失败检查数"),
    ("unknown_check_count", "未识别检查数"),
    ("scan_materials", "材料识别工具"),
)
_INTERNAL_TERM_PATTERNS = (
    (re.compile(r"`?required_materials_complete`?\s*(?:=|＝|为|：|:)?\s*`?false`?", re.IGNORECASE), "必需材料尚未齐全"),
    (re.compile(r"`?required_materials_complete`?\s*(?:=|＝|为|：|:)?\s*`?true`?", re.IGNORECASE), "必需材料已齐全"),
    (re.compile(r"`?required_materials_complete`?"), "必需材料是否齐全"),
    (re.compile(
        r"`?pending_confirmation`?\s*[（(]\s*`?needs_confirmation\s*=\s*true`?\s*[，,、]\s*"
        r"`?revision\s*[=：:]?\s*(\d+)`?\s*[）)]", re.IGNORECASE,
    ), r"待确认（第 \1 版）"),
    (re.compile(r"`?needs_confirmation\s*=\s*true`?", re.IGNORECASE), "仍需确认"),
    (re.compile(r"`?needs_confirmation\s*=\s*false`?", re.IGNORECASE), "无需再确认"),
    (re.compile(r"`?pending_confirmation`?"), "待确认"),
)


def humanize_internal_terms(text: str) -> str:
    """Replace internal keys the model echoed with the labels users see."""
    value = str(text or "")
    if not value:
        return value
    for pattern, label in _INTERNAL_TERM_PATTERNS:
        value = pattern.sub(label, value)
    for key, label in _INTERNAL_TERM_LABELS:
        # "数据字典（material_data_dictionary）" keeps only the label once.
        value = re.sub(r"\s*[（(]\s*`?" + key + r"`?\s*[）)]", "", value)
        value = re.sub(r"`?" + key + r"`?", label, value)
    return value


def generate_word_conclusions(
    *,
    task: TaskRecord,
    evidence: dict,
    memory_context: dict | None = None,
    model_profile: dict,
    user_instruction: str | None = None,
) -> tuple[dict[str, str], dict]:
    from marvis.llm_outbound import SafeInstruction
    from marvis.llm_outbound_instructions import user_instruction as approve_user_instruction
    from marvis.llm_outbound_memory import model_history_context
    from marvis.llm_outbound_revision import (
        APPEND_OPERATIONS, EVIDENCE_OPERATIONS, ReportRevisionError,
        parse_report_revision_response, report_revision_baseline,
        report_revision_prompt, report_revision_request, revision_fields,
    )

    fresh_report = not has_report_revision_context(evidence, user_instruction, algorithm=task.algorithm)
    request, baseline, fields, provenance = None, {}, (), {}
    try:
        history = model_history_context(task=task, evidence=evidence, memory_context=memory_context)
        if fresh_report:
            prompt = {"stage": "word_conclusion_draft", "evidence": {
                key: value for key, value in evidence.items() if key != "saved_report_narrative"
            }, "model_history": history}
        else:
            instruction = user_instruction
            if instruction is not None and not isinstance(instruction, SafeInstruction):
                instruction = approve_user_instruction(instruction)
            request = report_revision_request(instruction)
            baseline, provenance = report_revision_baseline(
                evidence, report_revision=task.report_values_revision,
            )
            fields = revision_fields(request, baseline)
        client = _client(model_profile) if fresh_report or (fields and request.operation in EVIDENCE_OPERATIONS) else None
        local_base = {}
        if request is not None and request.operation in {"replace_local_text", "replace_and_append"}:
            # Check the local edit before any external request is sent.
            if any(request.local_old_text not in baseline.get(key, "") for key in fields):
                raise ReportRevisionError(
                    "当前报告字段未包含指定的待替换片段，请核对当前草稿后重新说明修改。",
                    code="revision_text_mismatch",
                )
            local_base = {key: baseline[key].replace(request.local_old_text, request.local_text) for key in fields}
        elif request is not None and request.operation == "append_from_evidence":
            local_base = {key: baseline.get(key, "") for key in fields}
        if request is not None and request.operation == "clear":
            values = {key: "" for key in fields}
        elif request is not None and request.operation == "set_local_text":
            values = {key: request.local_text for key in fields}
        elif request is not None and request.operation == "replace_local_text":
            values = dict(local_base)
        elif request is not None and not fields:
            values = {}
        elif task.validation_workflow_version == 2 and request is not None:
            from marvis.agent.validation_narrative import generate_v2_revision_sections

            values = generate_v2_revision_sections(
                client, evidence, request=request, provenance=provenance, fields=fields, history=history,
            )
        elif task.validation_workflow_version == 2:
            values = generate_v2_sections(client, prompt)
        else:
            from marvis.llm_outbound_validation import validation_v1_report_prompt

            prompt = validation_v1_report_prompt(evidence, history=history)
            if request is not None:
                prompt = report_revision_prompt(
                    prompt, request=request, provenance=provenance, requested_fields=fields,
                )
            response_contract = {"json_schema": {
                "name": "validation_v1_report_revision", "strict": False, "schema": {
                    "type": "object", "properties": {key: {"type": "string"} for key in fields},
                    "required": list(fields), "additionalProperties": False,
                },
            }} if request is not None else {"response_format": {"type": "json_object"}}
            from marvis.llm_prompts import WORD_CONCLUSION_REVISION_SYSTEM_PROMPT

            report_spec = WORD_CONCLUSION_REVISION_SYSTEM_PROMPT if request is not None else _WORD_PROMPT_SPEC
            content = client.complete(
                system_prompt=report_spec.text,
                user_prompt=prompt,
                temperature=0.2,
                **response_contract,
                stream=False,
                caller="validation_v1_report",
                prompt_name=report_spec.name,
                prompt_version=report_spec.version,
            )
            if request is not None:
                values = parse_report_revision_response(content, fields)
            else:
                values = _parse_conclusion_json(content)
                values = {key: value for key, value in values.items()
                          if key in AGENT_REPORT_CONCLUSION_KEYS or key == "TEXT:model_training_description"}
        if request is not None and request.operation in APPEND_OPERATIONS:
            # The measured addendum follows the locally kept paragraph verbatim.
            values = {
                key: "\n\n".join(part for part in (local_base.get(key, "").strip(), values.get(key, "").strip()) if part)
                for key in fields
            }
        if fresh_report or (request is not None and request.operation in EVIDENCE_OPERATIONS
                            and _FINAL_CONCLUSION_KEY in fields):
            values = _with_platform_risk_findings(values, evidence)
        if request is not None:
            values = {**narrative_report_values(task.model_name), **baseline, **values}
            if "TEXT:model_training_description" not in values:
                values = _with_training_description_seed(task, values, evidence)
        else:
            values = _with_narrative_report_seeds(task, values, evidence)
            values = _with_training_description_seed(task, values, evidence)
        if task.validation_workflow_version == 2 and (request is None or request.operation == "regenerate_from_evidence"):
            _validate_v2_word_conclusions(values)
    except (LLMClientError, ValueError) as exc:
        from marvis.llm_outbound import OutboundContextError

        metadata = {"llm_error": str(exc), "fallback": True, "confirmable": False,
                    "history_comparison_available": False}
        if isinstance(exc, OutboundContextError):
            metadata.update({"outbound_context_status": "clarification_required",
                             "revision_error_code": getattr(exc, "code", "unapproved_instruction")})
        if request is not None:
            metadata.update({"report_revision_status": "failed",
                             "report_revision_operation": request.operation,
                             "report_revision_fields": list(fields),
                             "report_revision_provenance": provenance})
        return {}, metadata
    history_metadata = (history.metadata(use_reason="word_conclusion_draft")
                        if fresh_report or (request.operation in EVIDENCE_OPERATIONS and fields)
                        else {"history_comparison_available": False})
    metadata = {"fallback": False, **history_metadata}
    if not fresh_report:
        metadata.update({"report_revision_operation": request.operation,
                         "report_revision_fields": list(fields),
                         "report_revision_provenance": provenance,
                         "confirmable": agent_conclusions_confirmed(values)})
    return values, metadata


def fallback_word_conclusions(
    *,
    task: TaskRecord,
    evidence: dict | None = None,
) -> dict[str, str]:
    name = task.model_name or "本模型"
    deterministic = _fallback_conclusions_from_validation_evidence(
        evidence=evidence,
        validation_workflow_version=task.validation_workflow_version,
    )
    if deterministic is not None:
        return _with_training_description_seed(
            task,
            _with_narrative_report_seeds(task, deterministic, evidence),
            evidence,
        )
    if task.validation_workflow_version == 2:
        return _with_training_description_seed(
            task,
            _with_narrative_report_seeds(
                task,
                {
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
                },
                evidence,
            ),
            evidence,
        )
    return _with_training_description_seed(
        task,
        _with_narrative_report_seeds(
            task,
            {
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
            },
            evidence,
        ),
        evidence,
    )


def _fallback_conclusions_from_validation_evidence(
    *,
    evidence: dict | None,
    validation_workflow_version: int | None = None,
) -> dict[str, str] | None:
    if not isinstance(evidence, dict):
        return None
    raw_results = evidence.get("validation_results")
    if not isinstance(raw_results, dict):
        return None
    payload = dict(raw_results)
    overfitting = payload.pop("overfitting_check", None)
    try:
        results = validation_results_from_dict(payload)
    except ValueError:
        return None
    if not isinstance(overfitting, dict):
        overfitting = overfitting_check_from_validation_results(payload)

    computed = report_text_values_from_results(results)
    pressure_summary = computed.get(
        "TEXT:pressure_test_summary",
        "平台已完成压力测试，具体指标见 Excel 明细。",
    )
    baseline_ks = results.stress_test.baseline.ks
    category_risks: list[str | None] = []
    for item in results.stress_test.per_category:
        if item.status != "completed":
            category_risks.append(None)
            continue
        category_risks.append(worst_stress_risk(
            stress_ks_risk(baseline_ks, item.ks_after)
            if item.ks_after is not None
            else None,
            stress_psi_risk(item.psi_vs_baseline),
        ))
    pressure_risk = worst_stress_risk(*category_risks)
    pressure_label = stress_risk_label(pressure_risk)
    oot = next(
        (row for row in results.effectiveness.overall if row.split == "oot"),
        None,
    )
    oot_text = (
        f"OOT KS {oot.ks:.4f}（仅展示，不参与通过判定），OOT PSI {oot.psi_vs_train:.4f}"
        if oot is not None
        else "OOT KS/PSI 数据缺失"
    )
    if results.pmml_scoring is not None:
        scoring_text = f"PMML 打分测试{('通过' if results.pmml_scoring.status == 'pass' else '未通过')}"
    else:
        scoring_text = "Notebook 分数一致性结果见结构化验证明细"

    if pressure_risk == "high":
        recommendation = (
            "压力测试存在高风险场景，应核对缺失信源、样本分布与分箱迁移；"
            "投产前应配置监控和熔断，并准备备用评分或人工复核方案。"
        )
    elif pressure_risk == "medium":
        recommendation = (
            "压力测试达到预警阈值，建议人工核验异常类别并通知模型团队；"
            "上线后需加强 KS、PSI 与信源缺失率监控。"
        )
    else:
        recommendation = (
            "压力测试未发现中高风险场景，可结合其他确定性门禁继续评估；"
            "上线后仍应执行常规 KS、PSI 与信源稳定性监控。"
        )
    metrics_text = _effectiveness_metric_sentence(results.effectiveness.overall)
    stability_text = _oot_stability_sentence(oot)
    overfit_text = _overfitting_sentence(overfitting)
    bin_text = _binning_sentence(results)
    conclusion_parts = [
        metrics_text,
        oot_text,
        stability_text,
        overfit_text,
        bin_text,
        f"压力测试最高等级为{pressure_label}",
    ]
    if validation_workflow_version != 2:
        conclusion_parts.insert(0, scoring_text)
    conclusion = (
        "；".join(part for part in conclusion_parts if part)
        + "。本段为大模型文本生成失败后的确定性保守回退结论，指标取自平台结果；"
        "最终使用决定仍需结合稳定性、压力测试和报告完整性门禁复核。"
    )
    return {
        "TEXT:pressure_test_summary": pressure_summary,
        "TEXT:pressure_impact_recommendation": recommendation,
        "TEXT:final_validation_conclusion": conclusion,
        "TEXT:model_training_description": model_training_report_text(
            results.algorithm,
            results.basic_info.hyperparameters,
        ),
    }


_FINAL_CONCLUSION_KEY = "TEXT:final_validation_conclusion"


def _platform_risk_findings(evidence: dict | None) -> list[tuple[tuple[tuple[str, ...], ...], str]]:
    """Failed or high-risk items the platform measured, with mention keywords.

    Each finding pairs keyword groups (every group needs one hit in the
    conclusion) with a red-bold sentence quoting platform values only. Reads
    the persisted result dict directly so a partial payload still reports.
    """
    raw = evidence.get("validation_results") if isinstance(evidence, dict) else None
    if not isinstance(raw, dict):
        return []

    def number(value):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    findings: list[tuple[tuple[tuple[str, ...], ...], str]] = []
    stress = raw.get("stress_test") if isinstance(raw.get("stress_test"), dict) else {}
    baseline_ks = number((stress.get("baseline") or {}).get("ks")) if isinstance(stress.get("baseline"), dict) else None
    high: list[str] = []
    categories = stress.get("per_category") if isinstance(stress.get("per_category"), list) else []
    for index, item in enumerate(categories, start=1):
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        ks_after = number(item.get("ks_after"))
        risk = worst_stress_risk(
            stress_ks_risk(baseline_ks, ks_after) if baseline_ks is not None and ks_after is not None else None,
            stress_psi_risk(number(item.get("psi_vs_baseline"))),
        )
        if risk != "high":
            continue
        name = str(item.get("category") or "").strip()[:24] or f"类别{index}"
        after = f"KS {ks_after:.3f}" if ks_after is not None else "KS 未知"
        high.append(f"「{name}」类（剔除后 {after}）")
    if high:
        baseline_text = f"基线 KS {baseline_ks:.3f}，" if baseline_ks is not None else ""
        findings.append((
            (("压力",), ("高风险", "失效", "丧失")),
            f"!!压力测试显示{'、'.join(high[:6])}为高风险数据源!!，{baseline_text}"
            "相关数据缺失或口径变化会显著削弱模型区分能力",
        ))
    overfitting = raw.get("overfitting_check")
    if not isinstance(overfitting, dict):
        try:
            overfitting = overfitting_check_from_validation_results(raw)
        except Exception:  # noqa: BLE001 - a partial payload just has no overfitting finding
            overfitting = None
    if isinstance(overfitting, dict) and overfitting.get("status") == "fail":
        findings.append(((("过拟合",), ("未通过", "不通过", "风险")), _overfitting_sentence(overfitting)))
    effectiveness = raw.get("effectiveness") if isinstance(raw.get("effectiveness"), dict) else {}
    overall = effectiveness.get("overall") if isinstance(effectiveness.get("overall"), list) else []
    oot = next((row for row in overall if isinstance(row, dict) and row.get("split") == "oot"), None)
    oot_psi = number(oot.get("psi_vs_train")) if oot else None
    if oot_psi is not None and oot_psi >= 0.25:
        findings.append((
            (("PSI",), ("迁移", "不稳定", "偏移", "超过", "超出")),
            f"!!OOT 相对 Train 的 PSI 为 {oot_psi:.4f}，样本外分布迁移明显!!",
        ))
    return findings


def _with_platform_risk_findings(values: dict, evidence: dict | None) -> dict:
    """A final conclusion must not omit platform-measured failed/high-risk items."""
    conclusion = str(values.get(_FINAL_CONCLUSION_KEY) or "")
    if not conclusion.strip():
        return values
    missing = [
        sentence for groups, sentence in _platform_risk_findings(evidence)
        if not all(any(word.lower() in conclusion.lower() for word in group) for group in groups)
    ]
    if not missing:
        return values
    return {
        **values,
        _FINAL_CONCLUSION_KEY: conclusion.rstrip() + "\n\n平台确定性检查提示：" + "；".join(missing) + "。",
    }


def _effectiveness_metric_sentence(overall: list) -> str:
    by_split = {row.split: row for row in overall}
    parts: list[str] = []
    for split, label in (("train", "Train"), ("test", "Test"), ("oot", "OOT")):
        row = by_split.get(split)
        if row is None:
            continue
        parts.append(
            f"{label} KS {row.ks:.4f}、AUC {row.auc:.4f}、PSI {row.psi_vs_train:.4f}"
        )
    return "；".join(parts) if parts else "KS/AUC/PSI 数据缺失"


def _oot_stability_sentence(oot) -> str:
    if oot is None:
        return "样本外稳定性证据不足"
    psi = float(oot.psi_vs_train)
    if psi < 0.10:
        return f"样本外稳定性可接受（OOT PSI {psi:.4f}）"
    if psi < 0.25:
        return f"样本外稳定性需关注（OOT PSI {psi:.4f}）"
    return f"样本外分布迁移明显（OOT PSI {psi:.4f}）"


def _overfitting_sentence(overfitting: dict | None) -> str:
    if not isinstance(overfitting, dict):
        return "过拟合检查证据不足"
    status = str(overfitting.get("status") or "")
    relative_diff = overfitting.get("train_test_relative_diff")
    abs_diff = overfitting.get("train_oot_abs_diff")
    relative_text = (
        f"{relative_diff:.2%}"
        if isinstance(relative_diff, (int, float))
        else "未知"
    )
    abs_text = (
        f"{abs_diff:.4f}"
        if isinstance(abs_diff, (int, float))
        else "未知"
    )
    if status == "fail":
        return (
            f"!!过拟合检查未通过（train-test 相对差 {relative_text}，"
            f"train-OOT 绝对差 {abs_text}）!!，存在过拟合或样本外效果衰减风险"
        )
    if status == "pass":
        return "过拟合检查通过，train/test/OOT 区分能力缺口在阈值内"
    return "过拟合检查证据不足"


_LIFT_STRENGTH_LABELS = {
    "fail": "未达标",
    "weak": "偏弱",
    "ok": "尚可",
    "strong": "较好",
    "unknown": "证据不足",
}
_MONOTONICITY_LABELS = {
    "monotonic": "逾期率大致单调",
    "mostly_monotonic": "逾期率基本单调",
    "broken": "逾期率存在多处倒挂",
    "unknown": "单调性证据不足",
}


def _mark_fail_phrase(text: str, *, fail: bool) -> str:
    return f"!!{text}!!" if fail else text


def _binning_sentence(results) -> str:
    ranking = assess_lift_ranking(_effectiveness_payload_for_lift(results))
    oot = next(
        (item for item in ranking.get("splits") or [] if item.get("split") == "oot"),
        None,
    )
    if not isinstance(oot, dict):
        return "分箱排序性明细见结构化结果"
    bins = oot.get("independent_quantile") or oot.get("train_aligned") or {}
    if not isinstance(bins, dict):
        bins = {}
    head_value = oot.get("head_lift_5pct")
    tail_value = oot.get("tail_lift_5pct")
    head_strength = str(oot.get("head_5pct_strength") or "unknown")
    tail_strength = str(oot.get("tail_5pct_strength") or "unknown")
    if head_value is None:
        head_value = bins.get("head_group_lift")
        head_strength = str(bins.get("head_group_strength") or head_strength)
    if tail_value is None:
        tail_value = bins.get("tail_group_lift")
        tail_strength = str(bins.get("tail_group_strength") or tail_strength)
    if head_value is None and tail_value is None:
        return "分箱排序性明细见结构化结果"
    kind = "独立10等分分箱" if oot.get("independent_quantile") else "分箱"
    head_text = (
        f"头部 lift {head_value:.2f}（{_LIFT_STRENGTH_LABELS.get(head_strength, '证据不足')}）"
        if isinstance(head_value, (int, float))
        else "头部 lift 缺失"
    )
    tail_text = (
        f"尾部 lift {tail_value:.2f}（{_LIFT_STRENGTH_LABELS.get(tail_strength, '证据不足')}）"
        if isinstance(tail_value, (int, float))
        else "尾部 lift 缺失"
    )
    monotonicity = str(bins.get("bad_rate_monotonicity") or "unknown")
    mono_text = _MONOTONICITY_LABELS.get(monotonicity, "单调性证据不足")
    return (
        f"OOT {kind}"
        f"{_mark_fail_phrase(head_text, fail=head_strength in {'fail', 'weak'})}、"
        f"{_mark_fail_phrase(tail_text, fail=tail_strength in {'fail', 'weak'})}，"
        f"{_mark_fail_phrase(mono_text, fail=monotonicity == 'broken')}"
    )


def _effectiveness_payload_for_lift(results) -> dict:
    def tables(payload: object) -> dict[str, list[dict[str, object]]]:
        if not isinstance(payload, dict):
            return {}
        converted: dict[str, list[dict[str, object]]] = {}
        for split, rows in payload.items():
            converted[str(split)] = [
                {
                    "lift": getattr(row, "lift", None),
                    "bad_rate": getattr(row, "bad_rate", None),
                }
                for row in rows or []
            ]
        return converted

    return {
        "overall": [
            {
                "split": row.split,
                "head_lift_5pct": row.head_lift_5pct,
                "tail_lift_5pct": row.tail_lift_5pct,
            }
            for row in results.effectiveness.overall
        ],
        "bin_tables": tables(results.effectiveness.bin_tables),
        "independent_quantile_bin_tables": tables(
            results.effectiveness.independent_quantile_bin_tables
        ),
    }


def _client(model_profile: dict) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(model_profile)


def _guard_stage_summary(
    *,
    task: TaskRecord,
    stage: str,
    evidence: dict,
    content: str,
    fallback: str,
) -> str:
    if (
        stage == "reproducibility"
        and task.validation_workflow_version == 2
        and _V2_PMML_SCORING_LEGACY_PATTERN.search(content)
    ):
        return fallback
    if stage != "scan" or not isinstance(evidence, dict):
        return content
    from marvis.llm_outbound_scan import scan_context
    context = scan_context(evidence, workflow_version=task.validation_workflow_version)
    interpretation = context["scan_interpretation"]
    if (not interpretation["required_materials_complete"] or interpretation["failed_check_count"]
            or context["scan"].get("status") == "failed"):
        return content
    if not _scan_summary_claims_missing_complete_materials(content):
        return content
    if task.validation_workflow_version == 2:
        return (
            "材料识别与验证输入契约候选检查已完成，当前未发现必需材料缺失。"
            "字段和元数据是否已确认，请以当前输入契约状态为准。"
        )
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


def _llm_task_reference_instruction() -> str:
    return (
        "涉及当前任务或当前模型时，必须使用 task.model_display_name / 模型名称来称呼，"
        "不要输出、复述或引用内部任务 ID。"
    )


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
            "效果表现的分析深度必须对齐稳定性表现和压力测试风险：不能只写 KS/AUC 数字或 lift 是否跨过 1。"
            "必须结合 evidence.validation_results.effectiveness.lift_ranking_assessment "
            "评价头尾 5% lift 的幅度、分箱逾期率/单组 lift 是否单调、头尾区分是否拉开；"
            "头部 0.99 虽小于 1 仍应判为弱区分，不得写成通过。"
            "未通过、过拟合、弱 lift、分箱倒挂、PSI≥0.25、高风险数据源等明显不好的判断，"
            "必须用 !!关键短语!! 包住（只包短语），通过项不要用 !! !!。"
            "若 cross_task_memory 含同类模型，必须对比历史 KS/AUC/PSI（及过拟合、头尾 lift 如有），"
            "并标明来自历史记忆；没有可比记忆时在总体判断写「本次未见可比历史模型」，不要编造对比。"
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
                "输出必须是 JSON 对象，且必须完整包含："
                "TEXT:pressure_test_summary、TEXT:pressure_impact_recommendation、"
                "TEXT:final_validation_conclusion、TEXT:model_training_description。"
                "TEXT:pressure_test_summary 必须说明模型压力测试目的、"
                "方法和观察到的高/中/低风险数据源或特征类别分层，写清基线 KS 与各类别剔除后 KS/PSI；"
                "不得只复述「置 -9999」机械清单。证据不足时明确说明无法完成某一档分层。"
                "TEXT:model_training_description 必须介绍本模型实际算法，并引用 "
                "evidence.validation_results.basic_info.hyperparameters 的关键参数"
                "（如 max_depth、learning_rate、num_boost_round/best_iteration）；"
                "不得只粘贴该算法的通用教科书介绍，也不得在有超参证据时写成待确认算法。"
                "TEXT:pressure_impact_recommendation 必须围绕上述风险分层给出监控、替代、降级、"
                "人工复核或上线限制建议。TEXT:final_validation_conclusion 必须针对本模型撰写专属叙事，"
                "写入 Train/Test/OOT 的 KS、AUC、PSI，并评价稳定性、过拟合、压力测试与分箱排序性；"
                "分箱与 lift 须使用 lift_ranking_assessment：看单调性、头尾幅度和区分是否拉开，"
                "不得只因 lift 跨过 1 就判好。未通过或明显不好的短语用 !! !! 包住。"
                "不得套用同一套套话。如 cross_task_memory 含历史同类模型，必须做一次克制对比；"
                "没有则写「本次未见可比历史模型」。引用须克制且不得改写平台指标。"
                "TEXT:final_validation_conclusion 必须比前两段更完整，"
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
            "数据源名称或监管结论。输出必须是 JSON 对象，且必须完整包含："
            "TEXT:pressure_test_summary、TEXT:pressure_impact_recommendation、"
            "TEXT:final_validation_conclusion、TEXT:model_training_description。"
            "TEXT:pressure_test_summary 必须说明压力测试目的、方法和观察到的高/中/低风险数据源分层；"
            "不得只复述机械清单。证据不足时明确说明无法完成某一档分层。"
            "TEXT:model_training_description 必须介绍本模型实际算法并引用已给出的训练超参，"
            "不得只输出通用教科书介绍。"
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
    if any(key in payload and not isinstance(payload[key], str) for key in AGENT_REPORT_WRITABLE_KEYS):
        raise ValueError("word conclusion fields must contain text")
    values = {
        key: str(payload.get(key) or "").strip()
        for key in REQUIRED_AGENT_REPORT_KEYS
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError("word conclusion response missing keys: " + ", ".join(missing))
    for key in AGENT_REPORT_WRITABLE_KEYS - AGENT_REPORT_CONCLUSION_KEYS:
        extra = str(payload.get(key) or "").strip()
        if extra or (key in AGENT_REPORT_NARRATIVE_KEYS and isinstance(payload.get(key), str)):
            values[key] = extra
    return values


def _with_narrative_report_seeds(
    task: TaskRecord, values: dict[str, str], evidence: dict | None = None,
) -> dict[str, str]:
    # Model output is a patch for optional narratives. Preserve raw saved text
    # rather than its privacy-filtered prompt copy, including deliberate blanks.
    merged = narrative_report_values(task.model_name)
    for name in ("saved_report_narrative", "report_draft"):
        context = evidence.get(name) if isinstance(evidence, dict) else None
        saved = context.get("text_values") if isinstance(context, dict) else None
        if isinstance(saved, dict):
            merged.update({key: value for key, value in saved.items()
                           if key in AGENT_REPORT_NARRATIVE_KEYS and isinstance(value, str)})
    merged.update(values)
    return merged


def _with_training_description_seed(
    task: TaskRecord,
    values: dict[str, str],
    evidence: dict | None = None,
) -> dict[str, str]:
    merged = dict(values)
    current = merged.get("TEXT:model_training_description")
    if not is_platform_default_training_description(current, task.algorithm):
        return merged
    seeded = _training_description_from_evidence(task, evidence)
    if seeded:
        merged["TEXT:model_training_description"] = seeded
    return merged


def _training_description_from_evidence(
    task: TaskRecord,
    evidence: dict | None,
) -> str:
    algorithm = str(task.algorithm or "").strip()
    hyperparameters = None
    if isinstance(evidence, dict):
        raw_results = evidence.get("validation_results")
        if isinstance(raw_results, dict):
            algorithm = str(raw_results.get("algorithm") or algorithm).strip()
            basic_info = raw_results.get("basic_info")
            if isinstance(basic_info, dict) and isinstance(basic_info.get("hyperparameters"), dict):
                hyperparameters = basic_info.get("hyperparameters")
    if not algorithm:
        return ""
    try:
        return model_training_report_text(algorithm, hyperparameters)
    except ValueError:
        return ""
