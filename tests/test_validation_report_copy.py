from __future__ import annotations

import pytest

from marvis.agent.validation_app_service import WIRED_AGENT_TASK_TYPES
from marvis.domain import TASK_TYPE_VALIDATION_BATCH
from marvis.validation_report_copy import (
    METRIC_REWRITE_REFUSAL,
    MISSING_IMPORTANCE_GUIDANCE,
    channel_from_model_name,
    cohort_from_model_name,
    humanize_scan_check_message,
    looks_like_accept_suggested_contracts,
    looks_like_metric_rewrite_request,
    narrative_report_values,
)


def test_humanize_scan_check_message_explains_missing_importance():
    assert humanize_scan_check_message(
        "missing importance column alias feature_importance"
    ) == MISSING_IMPORTANCE_GUIDANCE
    assert "请补充 importance" in MISSING_IMPORTANCE_GUIDANCE
    assert "不会自动填 1.0" in MISSING_IMPORTANCE_GUIDANCE


def test_looks_like_metric_rewrite_request_detects_ks_auc_psi_changes():
    assert looks_like_metric_rewrite_request("把KS改成0.50") is True
    assert looks_like_metric_rewrite_request("请把 AUC 改写为 0.8") is True
    assert looks_like_metric_rewrite_request("PSI 调整成 0.01") is True
    assert looks_like_metric_rewrite_request("概述改成支用环节") is False
    assert looks_like_metric_rewrite_request("KS 现在多少") is False


def test_t_card_narrative_uses_drawdown_not_credit():
    values = narrative_report_values("自营渠道甲T卡")
    assert "支用" in values["TEXT:model_overview"]
    assert "授信" not in values["TEXT:model_overview"]
    assert "申请支用" in values["TEXT:sample_audience"]


@pytest.mark.parametrize(
    ("name", "channel", "cohort"),
    [
        ("自营渠道甲T卡多维MOB6", "渠道甲", "渠道甲T卡"),
        ("渠道甲、渠道乙A卡MOB3", "渠道甲、渠道乙", "渠道甲、渠道乙A卡"),
        ("渠道丙MOB6", "渠道丙", "渠道丙"),
        ("自营通用T卡多头MOB6", "自营", "自营通用T卡"),
        ("T卡", "", "xx"),
        ("普通模型", "", "xx"),
    ],
)
def test_report_cohort_uses_only_user_supplied_name(name, channel, cohort):
    assert channel_from_model_name(name) == channel
    assert cohort_from_model_name(name) == cohort


def test_wired_agent_types_include_validation_batch():
    assert TASK_TYPE_VALIDATION_BATCH in WIRED_AGENT_TASK_TYPES


def test_metric_rewrite_refusal_points_to_platform_metrics():
    assert "确定性" in METRIC_REWRITE_REFUSAL
    assert "KS" in METRIC_REWRITE_REFUSAL


def test_looks_like_accept_suggested_contracts():
    assert looks_like_accept_suggested_contracts("都按这个") is True
    assert looks_like_accept_suggested_contracts("都按识别结果确认") is True
    assert looks_like_accept_suggested_contracts("概述改成支用环节") is False
