from __future__ import annotations


def agent_stage_opening_text(stage: str) -> str:
    if stage == "reproducibility":
        return "收到，我将继续执行模型可复现性验证，运行 Notebook 并检查代码模型分数与提交 PMML 分数的一致性。"
    if stage == "metrics":
        return "收到，我将继续执行模型效果与稳定性验证，计算 KS、PSI、分箱和压力测试等指标。"
    if stage == "word_conclusion_draft":
        return "收到，我将基于已完成的验证结果起草 Word 报告中的三段结论，完成后会等你确认。"
    return "收到，我将继续执行下一步验证。"


def agent_stage_label(stage: str) -> str:
    if stage == "scan":
        return "模型材料完备性验证"
    if stage == "reproducibility":
        return "模型可复现性验证"
    if stage == "metrics":
        return "模型效果&稳定性验证"
    if stage == "word_conclusion_draft":
        return "报告结论草稿生成"
    return "下一步验证"


def format_conclusion_values(values: dict[str, str]) -> str:
    labels = {
        "TEXT:pressure_test_summary": "压力测试总结",
        "TEXT:pressure_impact_recommendation": "压力影响建议",
        "TEXT:final_validation_conclusion": "最终验证结论",
    }
    ordered_keys = [
        "TEXT:pressure_test_summary",
        "TEXT:pressure_impact_recommendation",
        "TEXT:final_validation_conclusion",
    ]
    ordered_keys.extend(key for key in values if key not in labels)
    return "\n\n".join(
        f"{labels.get(key, key)}\n{value}"
        for key in ordered_keys
        if (value := values.get(key))
    )
