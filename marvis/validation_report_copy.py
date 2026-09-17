"""Deterministic validation-report copy: identity fields, A/T card wording, samples.

Agent may draft or revise narrative keys and the model-training description,
but must not invent validator names or KS/AUC/PSI. Identity fields are filled
by the platform at task create time. Platform default algorithm blurbs are
only a fallback; Word prefers Agent text or this model's recorded hyperparameters.
"""

from __future__ import annotations

from datetime import date
import re

from marvis.model_algorithms import (
    PENDING_MODEL_TRAINING_DESCRIPTION,
    model_training_report_text,
)


IDENTITY_REPORT_KEYS = frozenset({
    "TEXT:report_title",
    "TEXT:drafter",
    "TEXT:draft_date",
    "TEXT:revision_version",
    "TEXT:revision_date",
    "TEXT:revision_author",
    "TEXT:revision_description",
    "TEXT:model_training_description",
})
NARRATIVE_REPORT_KEYS = frozenset({
    "TEXT:model_overview",
    "TEXT:model_scope",
    "TEXT:bad_sample_definition",
    "TEXT:good_sample_definition",
    "TEXT:sample_audience",
})
MISSING_IMPORTANCE_GUIDANCE = (
    "数据字典缺少 importance（特征重要性）列。"
    "请补充 importance 或 feature_importance 列后重新扫描；"
    "平台不会自动填 1.0，也不会把该列改成可选。"
)
METRIC_REWRITE_REFUSAL = (
    "KS、AUC、PSI 和分数一致性由平台确定性计算，不能按对话改写。"
    "请查看效果稳定性证据；如需改报告叙事（概述、样本定义、结论口径），说明要改的文案即可。"
)
_IMPORTANCE_DIAGNOSTIC_MARKERS = (
    "missing importance",
    "importance column alias",
)
_METRIC_REWRITE_MARKERS = (
    "ks",
    "auc",
    "psi",
    "分数一致性",
    "oot ks",
    "区分度数字",
)


def display_model_name(model_name: str) -> str:
    text = str(model_name or "").strip() or "本模型"
    return text[:-2] if text.endswith("模型") else text


def infer_scorecard_kind(model_name: str) -> str | None:
    text = str(model_name or "")
    has_t = "T卡" in text or "t卡" in text.lower()
    has_a = "A卡" in text or "a卡" in text.lower()
    if has_t and not has_a:
        return "t"
    if has_a and not has_t:
        return "a"
    return None


def infer_mob_window(model_name: str) -> str | None:
    match = re.search(r"MOB\s*([36])", str(model_name or ""), flags=re.IGNORECASE)
    if match is None:
        return None
    return f"MOB{match.group(1)}"


def scorecard_stage_phrases(kind: str | None) -> tuple[str, str]:
    if kind == "t":
        return "支用环节", "支用申请阶段"
    if kind == "a":
        return "授信环节", "授信申请阶段"
    return "xx", "xx"


def sample_audience_phrase(kind: str | None) -> str:
    if kind == "t":
        return "申请支用的用户"
    if kind == "a":
        return "申请授信的用户"
    return "申请授信的用户"


def channel_from_model_name(model_name: str) -> str:
    """Use only the channel prefix supplied by the user, never a client list.

    A scorecard or observation-window marker separates the prefix from model
    details. Names without that boundary remain unspecified for user review.
    """
    text = str(model_name or "").strip()
    parts = re.split(r"[AT]卡|MOB\s*\d+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return ""
    prefix = parts[0].strip(" _-/：:")
    if prefix in {"自营", "自营通用"}:
        return "自营"
    return prefix.removeprefix("自营").strip(" _-/：:")


def cohort_from_model_name(model_name: str) -> str:
    channel = channel_from_model_name(model_name)
    if not channel:
        return "xx"
    kind = infer_scorecard_kind(model_name)
    if channel == "自营":
        channel = "自营通用"
    return f"{channel}{kind.upper()}卡" if kind else channel


def narrative_report_values(model_name: str) -> dict[str, str]:
    """Unknown narrative facts stay blank until provided or grounded in evidence."""
    return {key: "" for key in NARRATIVE_REPORT_KEYS}


def identity_report_values(
    model_name: str,
    model_version: str,
    validator: str,
    algorithm: str = "",
) -> dict[str, str]:
    today = date.today().isoformat()
    display_name = display_model_name(model_name)
    version_suffix = f"{model_version}版" if model_version else ""
    title_name = display_name if display_name.endswith("模型") else f"{display_name}模型"
    training_description = (
        model_training_report_text(algorithm)
        if str(algorithm or "").strip()
        else PENDING_MODEL_TRAINING_DESCRIPTION
    )
    return {
        "TEXT:report_title": f"{title_name}{version_suffix}验证文档",
        "TEXT:drafter": validator,
        "TEXT:draft_date": today,
        "TEXT:revision_version": "V1",
        "TEXT:revision_date": today,
        "TEXT:revision_author": validator,
        "TEXT:revision_description": "初稿",
        "TEXT:model_training_description": training_description,
    }


def seed_report_values(
    model_name: str,
    model_version: str,
    validator: str,
    algorithm: str = "",
) -> dict[str, str]:
    return {
        **identity_report_values(model_name, model_version, validator, algorithm),
        # A model name is not evidence for customer scope or bad-label rules.
        # Leave narrative facts empty until supplied or drafted from evidence.
        **{key: "" for key in NARRATIVE_REPORT_KEYS},
    }


def merge_seed_report_values(
    existing: dict[str, str] | None,
    *,
    model_name: str,
    model_version: str,
    validator: str,
    algorithm: str = "",
) -> dict[str, str]:
    seeded = seed_report_values(model_name, model_version, validator, algorithm)
    provided = {
        key: str(value).strip()
        for key, value in (existing or {}).items()
        if str(value or "").strip()
    }
    return {**seeded, **provided}


def revision_identity_values(
    validator: str,
    *,
    description: str,
    version: str = "V2",
) -> dict[str, str]:
    today = date.today().isoformat()
    return {
        "TEXT:revision_version": version,
        "TEXT:revision_date": today,
        "TEXT:revision_author": validator,
        "TEXT:revision_description": description or "按对话修订报告文案",
    }


def humanize_scan_check_message(message: str) -> str:
    text = str(message or "").strip()
    if not text:
        return text
    lowered = text.lower()
    if any(marker in lowered for marker in _IMPORTANCE_DIAGNOSTIC_MARKERS):
        return MISSING_IMPORTANCE_GUIDANCE
    return text


def looks_like_metric_rewrite_request(content: str) -> bool:
    compact = "".join(str(content or "").lower().split())
    if not compact:
        return False
    rewrite = any(
        marker in compact
        for marker in ("改成", "改写", "修改", "调整", "换成", "把ks", "把auc", "把psi")
    )
    mentions_metric = any(marker in compact for marker in _METRIC_REWRITE_MARKERS)
    return rewrite and mentions_metric


def looks_like_accept_suggested_contracts(content: str) -> bool:
    compact = "".join(str(content or "").lower().split())
    if not compact:
        return False
    return any(
        marker in compact
        for marker in (
            "都按这个",
            "都按识别结果",
            "全部按这个",
            "按识别结果确认",
            "都按建议",
        )
    )
