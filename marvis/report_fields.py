from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

from marvis.domain import TaskRecord
from marvis.model_algorithms import model_training_description


@dataclass(frozen=True)
class ReportField:
    key: str
    label: str
    stage: str
    multiline: bool = False


REPORT_FIELDS: tuple[ReportField, ...] = (
    ReportField("TEXT:report_title", "报告标题", "cover"),
    ReportField("TEXT:drafter", "撰写人", "cover"),
    ReportField("TEXT:draft_date", "撰写日期", "cover"),
    ReportField("TEXT:revision_version", "修订版本", "cover"),
    ReportField("TEXT:revision_date", "修订日期", "cover"),
    ReportField("TEXT:revision_author", "修订人", "cover"),
    ReportField("TEXT:revision_description", "修订说明", "cover", True),
    ReportField("TEXT:model_overview", "模型概述", "before", True),
    ReportField("TEXT:model_scope", "适用范围", "before", True),
    ReportField("TEXT:bad_sample_definition", "坏样本定义", "before"),
    ReportField("TEXT:good_sample_definition", "好样本定义", "before"),
    ReportField("TEXT:model_training_description", "模型训练说明", "during", True),
    ReportField("TEXT:pressure_recommendation_summary", "压力测试建议", "after", True),
    ReportField("TEXT:pressure_recommendation_action", "压力风险处置", "after", True),
    ReportField("TEXT:pressure_recommendation_monitoring", "压力监控建议", "after", True),
    ReportField("TEXT:pressure_recommendation_high_impact", "高影响压力建议", "after", True),
    ReportField("TEXT:pressure_recommendation_medium_impact", "中影响压力建议", "after", True),
    ReportField("TEXT:pressure_recommendation_low_impact", "低影响压力建议", "after", True),
    ReportField("TEXT:final_validation_conclusion", "最终验证结论", "after", True),
)


def default_report_values(
    model_name: str,
    model_version: str,
    validator: str,
    algorithm: str = "",
) -> dict[str, str]:
    today = date.today().isoformat()
    version_suffix = f"{model_version}版" if model_version else ""
    display_name = model_name or "本模型"
    training_description = (
        model_training_description(algorithm)
        if str(algorithm or "").strip()
        else "待 Notebook 契约 RMC_ALGORITHM 确认后自动生成模型训练说明。"
    )
    return {
        "TEXT:report_title": f"{display_name}模型{version_suffix}验证文档",
        "TEXT:drafter": validator,
        "TEXT:draft_date": today,
        "TEXT:revision_version": "V1",
        "TEXT:revision_date": today,
        "TEXT:revision_author": validator,
        "TEXT:revision_description": "初稿",
        "TEXT:model_overview": (
            f"为了更好的对xx用户进行授信环节风险管控，现开发{display_name}模型，"
            "对xx客群做前置风险拦截，从授信申请阶段做好风险防范。"
        ),
        "TEXT:model_scope": "本模型适用于xx渠道用户。",
        "TEXT:bad_sample_definition": "xx逾期 >= xx天",
        "TEXT:good_sample_definition": "xx未逾期",
        "TEXT:model_training_description": training_description,
        "TEXT:pressure_recommendation_summary": "待补充压力测试结果和风险提示。",
        "TEXT:pressure_impact_recommendation": "待补充压力测试结果和风险提示。",
        "TEXT:pressure_recommendation_action": "建议结合压力测试表现制定差异化准入和监控策略。",
        "TEXT:pressure_recommendation_monitoring": "建议上线后持续监控模型区分度、稳定性和关键特征漂移。",
        "TEXT:pressure_recommendation_high_impact": "对于 KS 或 PSI 变化较大的特征类别，建议复核变量依赖和策略兜底方案。",
        "TEXT:pressure_recommendation_medium_impact": "对于中等影响的特征类别，建议纳入上线后重点监控并设置预警阈值。",
        "TEXT:pressure_recommendation_low_impact": "对于影响较低的特征类别，建议保持常规监控并定期复核稳定性。",
        "TEXT:final_validation_conclusion": "待补充最终验证结论。",
    }


def report_field_payload(
    task: TaskRecord,
    values: dict[str, str],
    revision: int,
    metric_values: dict[str, str] | None = None,
    metric_table_sections: list[dict] | None = None,
) -> dict:
    text_values = default_report_values(
        task.model_name,
        task.model_version,
        task.validator,
        task.algorithm,
    )
    text_values.update(values)
    return {
        "fields": [asdict(field) for field in REPORT_FIELDS],
        "text_values": text_values,
        "revision": revision,
        "metric_values": metric_values or {},
        "metric_table_sections": metric_table_sections or [],
    }
