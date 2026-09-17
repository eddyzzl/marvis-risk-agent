from __future__ import annotations

from dataclasses import dataclass, asdict

from marvis.domain import TaskRecord
from marvis.validation_report_copy import seed_report_values


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
    values = seed_report_values(model_name, model_version, validator, algorithm)
    values.update({
        "TEXT:pressure_recommendation_summary": "待补充压力测试结果和风险提示。",
        "TEXT:pressure_impact_recommendation": "待补充压力测试结果和风险提示。",
        "TEXT:pressure_recommendation_action": "建议结合压力测试表现制定差异化准入和监控策略。",
        "TEXT:pressure_recommendation_monitoring": "建议上线后持续监控模型区分度、稳定性和关键特征漂移。",
        "TEXT:pressure_recommendation_high_impact": "对于 KS 或 PSI 变化较大的特征类别，建议复核变量依赖和策略兜底方案。",
        "TEXT:pressure_recommendation_medium_impact": "对于中等影响的特征类别，建议纳入上线后重点监控并设置预警阈值。",
        "TEXT:pressure_recommendation_low_impact": "对于影响较低的特征类别，建议保持常规监控并定期复核稳定性。",
        "TEXT:final_validation_conclusion": "待补充最终验证结论。",
    })
    return values


def report_field_payload(
    task: TaskRecord,
    values: dict[str, str],
    revision: int,
    metric_values: dict[str, str] | None = None,
    metric_table_sections: list[dict] | None = None,
) -> dict:
    display_defaults = default_report_values(
        task.model_name,
        task.model_version,
        task.validator,
        task.algorithm,
    )
    stored_values = dict(values)
    text_values = dict(display_defaults)
    text_values.update(stored_values)
    return {
        "fields": [asdict(field) for field in REPORT_FIELDS],
        "text_values": text_values,
        "stored_values": stored_values,
        "display_defaults": display_defaults,
        "revision": revision,
        "metric_values": metric_values or {},
        "metric_table_sections": metric_table_sections or [],
    }
