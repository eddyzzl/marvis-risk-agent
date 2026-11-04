"""Deterministic validation-report copy: identity fields, A/T card wording, samples.

Agent may draft or revise narrative keys and the model-training description,
but must not invent validator names or KS/AUC/PSI. Identity fields are filled
by the platform at task create time. Platform default algorithm blurbs are
only a fallback; Word prefers Agent text or this model's recorded hyperparameters.
"""

from __future__ import annotations

from datetime import date
import re

from marvis.formatting import model_version_display
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
# A narrative may mention a metric ("补充剔除后 KS 降到 0 的风险"); only a
# request whose object is a metric value is a rewrite of platform evidence.
_METRIC_TOKEN = r"(?:(?:train|test|oot|训练集|测试集)?(?:ks|auc|psi)|分数一致性|区分度数字)"
_METRIC_VALUE_VERBS = (
    r"(?:改成|改为|改写为|改写成|改写|修改为|修改成|调整为|调整成|调整到|换成|写成|写为"
    r"|设为|设置为|设成|填成|填为|提高到|提升到|降低到|调高|调低|改高|改低|上调|下调)"
)
_METRIC_REWRITE_PATTERNS = (
    # "KS 改成 0.5" / "AUC 从 0.7 改成 0.8"
    re.compile(_METRIC_TOKEN + r"(?:值|的值|数值|指标|结果)?(?:从[^，。；,;]{0,12})?" + _METRIC_VALUE_VERBS),
    # "把 KS 改成 0.5" / "把结论里的 AUC 写成…" / "把“KS 为 1.0”改成…"
    re.compile(
        r"(?:把|将)(?:[^，。；,;“”\"]{0,12}?(?:里|中)的?)?[“\"「]?" + _METRIC_TOKEN
        + r"[^，。；,;]{0,10}?(?:改|调|换|写|设|填|降|升)"
    ),
    # "修改 KS" / "调高报告里的 AUC"
    re.compile(
        r"(?:修改|调整|改写|改动|篡改|美化|调高|调低|改高|改低|上调|下调)(?:一下)?"
        r"(?:报告|结论)?(?:里|中)?的?" + _METRIC_TOKEN
    ),
)


def display_model_name(model_name: str) -> str:
    text = str(model_name or "").strip() or "本模型"
    return text[:-2] if text.endswith("模型") else text


_BEHAVIOR_SCORECARD_MARKERS = ("b卡", "行为评分卡", "行为卡", "行为评分模型", "贷中")


def infer_scorecard_kind(model_name: str) -> str | None:
    """Only an explicit card marker decides the business stage; else unknown."""
    lowered = str(model_name or "").lower()
    kinds = {
        kind for kind, present in (
            ("t", "t卡" in lowered),
            ("a", "a卡" in lowered),
            ("b", any(marker in lowered for marker in _BEHAVIOR_SCORECARD_MARKERS)),
        ) if present
    }
    return kinds.pop() if len(kinds) == 1 else None


def scorecard_business_stage(*texts: str) -> str | None:
    """The latest user-supplied text naming exactly one card kind wins."""
    for text in reversed(texts):
        kind = infer_scorecard_kind(text)
        if kind is not None:
            return kind
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
    if kind == "b":
        return "贷中环节", "贷中管理阶段"
    return "xx", "xx"


def sample_audience_phrase(kind: str | None) -> str:
    if kind == "t":
        return "申请支用的用户"
    if kind == "a":
        return "申请授信的用户"
    if kind == "b":
        return "存量在贷用户"
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
    readable_version = model_version_display(model_version)
    version_suffix = f"{readable_version}版" if readable_version else ""
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


def dictionary_required_columns() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """The real data-dictionary contract: one column from each alias group."""
    from marvis.validation.feature_metadata import CATEGORY_ALIASES, FEATURE_ALIASES, IMPORTANCE_ALIASES

    return (
        ("feature", "特征名", tuple(FEATURE_ALIASES)),
        ("category", "特征类别/数据源", tuple(CATEGORY_ALIASES)),
        ("importance", "特征重要性", tuple(IMPORTANCE_ALIASES)),
    )


_METADATA_MISSING_COLUMN = {
    "feature": "missing_feature_column",
    "category": "missing_category_column",
    "importance": "missing_importance_column",
}
METADATA_REASON_LABELS = {
    "missing_feature_column": "缺少 feature（特征名）列",
    "missing_category_column": "缺少 category（特征类别/数据源）列",
    "missing_importance_column": "缺少 importance（特征重要性）列",
    "sample_file": "这是样本数据文件，不是数据字典",
    "not_evaluated": "文件过大或候选过多，平台未能自动检查",
    "no_table": "没有读到可用的表格",
    "pmml_mismatch": "没有覆盖所选 PMML 的入模特征，或类别/重要性取值无效",
}


def metadata_reason_codes(blocking_errors) -> list[str]:
    """Fixed reason codes for metadata blocking errors (no raw text)."""
    codes: list[str] = []
    for error in blocking_errors if isinstance(blocking_errors, (list, tuple)) else ():
        text = str(error or "").lower()
        found = [code for column, code in _METADATA_MISSING_COLUMN.items()
                 if re.search(rf"missing\b[^;]*\b{column} column alias", text)]
        if not found:
            if "sample data file" in text:
                found = ["sample_file"]
            elif "limit" in text:
                found = ["not_evaluated"]
            elif "no candidate table" in text:
                found = ["no_table"]
            else:
                found = ["pmml_mismatch"]
        codes.extend(code for code in found if code not in codes)
    return codes


def humanize_metadata_blocking_errors(blocking_errors) -> str:
    """One Chinese sentence explaining why a dictionary file is unusable."""
    codes = metadata_reason_codes(blocking_errors)
    if not codes:
        return ""
    reasons = "；".join(METADATA_REASON_LABELS[code] for code in codes)
    if any(code.startswith("missing_") for code in codes):
        return f"{reasons}，请补充后重新上传（平台不会自动填 importance）。"
    return f"{reasons}。"


def dictionary_requirement_text() -> str:
    parts = [
        f"{name}（{label}，列名可用 {'、'.join(aliases[:6])}{' 等' if len(aliases) > 6 else ''}）"
        for name, label, aliases in dictionary_required_columns()
    ]
    return (
        "平台对数据字典只要求三类列：" + "；".join(parts)
        + "。三类都必填，importance 不会被自动补 1.0；字典还需覆盖所选 PMML 的入模特征。其他列不是必需的。"
    )


def looks_like_metric_rewrite_request(content: str) -> bool:
    compact = "".join(str(content or "").lower().split())
    if not compact:
        return False
    return any(pattern.search(compact) for pattern in _METRIC_REWRITE_PATTERNS)


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
