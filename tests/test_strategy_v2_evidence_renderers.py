from __future__ import annotations

from copy import deepcopy

import pytest

import marvis.agent.renderers as renderers
from marvis.agent.renderers import render_tool_output
from marvis.packs.strategy.model_evidence_tools import (
    run_materialize_model_evidence_v2,
)
from test_strategy_model_evidence_tool import _fixture as _model_evidence_fixture


SAMPLE_TOOL = "materialize_sample_design_v2"
MODEL_TOOL = "materialize_model_evidence_v2"


@pytest.fixture(scope="module")
def strategy_v2_outputs(tmp_path_factory) -> dict:
    fx = _model_evidence_fixture(tmp_path_factory.mktemp("strategy-v2-renderers"))
    model_output = run_materialize_model_evidence_v2(
        fx["inputs"], fx["ctx"], fx["runtime"]
    )
    return {"sample": fx["sample_v2"], "model": model_output}


def test_sample_v2_renderer_surfaces_only_authenticated_design_facts(
    strategy_v2_outputs: dict,
) -> None:
    output = strategy_v2_outputs["sample"]
    text, tables = render_tool_output(SAMPLE_TOOL, output)

    assert "策略样本设计 V2 已固化" in text
    assert "范围 `strategy_development`" in text
    assert "人群关系 `nested_same_cohort`" in text
    assert "风险样本成熟度：`confirmed_matured`" in text
    assert "历史评分：`available / legacy_score / higher_is_riskier`" in text
    assert "未创建策略、未采纳、未部署" in text
    assert "group_coverage_gap: warn" in text
    assert output["artifacts"]["bundle"]["filename"] in text
    assert output["artifacts"]["bundle"]["content_hash"] in text
    assert output["artifacts"]["membership"]["filename"] in text
    assert output["membership_content_hash"] in text
    assert "统一产物栏" in text
    assert "不会自行拼接链接" in text
    assert "/api/tasks/" not in text
    assert "download?" not in text
    assert "](" not in text

    split = next(table for table in tables if table["title"] == "双人群样本分区")
    assert split["rows"] == [
        ["approval", "6", "2", "2", "2", "not_applicable"],
        ["risk", "6", "2", "2", "2", "confirmed_matured"],
    ]
    diagnostics = next(table for table in tables if table["title"] == "样本诊断")
    assert {row[1] for row in diagnostics["rows"]} >= {
        "target_selector",
        "maturity",
        "sufficiency",
    }
    artifacts = next(table for table in tables if table["title"] == "样本产物摘要")
    assert {row[0] for row in artifacts["rows"]} == {
        "sample-design bundle",
        "membership",
    }
    membership_row = next(row for row in artifacts["rows"] if row[0] == "membership")
    assert membership_row[3] == f"semantic:{output['membership_content_hash']}"


def test_model_evidence_v2_renderer_is_univariate_only_and_never_builds_a_link(
    strategy_v2_outputs: dict,
) -> None:
    text, tables = render_tool_output(MODEL_TOOL, strategy_v2_outputs["model"])

    assert "策略分析证据 V2 已固化" in text
    assert "已认证 **2** 条单变量证据" in text
    assert "`risk/development`" in text
    assert "univariate-only" in text
    assert "未创建模型、未进行模型比较、未采纳、未部署" in text
    assert "统一产物栏" in text
    assert "不会自行拼接链接" in text
    assert "/api/tasks/" not in text
    assert "download?" not in text
    assert "](" not in text

    evidence = next(table for table in tables if table["title"] == "单变量分析证据")
    rows = {row[0]: row for row in evidence["rows"]}
    assert rows["channel"][1:3] == ["categorical", "2"]
    assert rows["legacy_score"][1:3] == ["equal_width", "3"]
    assert all(int(row[3]) == sum(int(value) for value in row[4:]) for row in rows.values())


@pytest.mark.parametrize(
    ("output_name", "mutation"),
    [
        ("sample", "top_hash"),
        ("sample", "bundle"),
        ("sample", "observation"),
        ("sample", "artifact"),
        ("sample", "extra"),
        ("model", "top_hash"),
        ("model", "bundle"),
        ("model", "observation"),
        ("model", "artifact"),
        ("model", "extra"),
    ],
)
def test_strategy_v2_renderers_fail_closed_on_every_authenticated_layer(
    strategy_v2_outputs: dict,
    output_name: str,
    mutation: str,
) -> None:
    output = deepcopy(strategy_v2_outputs[output_name])
    tool = SAMPLE_TOOL if output_name == "sample" else MODEL_TOOL

    if mutation == "top_hash":
        output["content_hash"] = "f" * 64
    elif mutation == "bundle" and output_name == "sample":
        output["bundle"]["sample_design"]["relationship"] = "parallel_time_cohorts"
    elif mutation == "bundle":
        output["bundle"]["producer_version"] = "forged-id"
    elif mutation == "observation" and output_name == "sample":
        output["bundle"]["metric_observations"][0]["sample_count"] = 999999
    elif mutation == "observation":
        output["bundle"]["univariate_evidence"][0]["observations"][0][
            "sample_count"
        ] = 999999
    elif mutation == "artifact" and output_name == "sample":
        output["artifacts"]["bundle"]["download_url"] = "/forged-id/999999"
    elif mutation == "artifact":
        output["artifact"]["filename"] = "forged-id-999999.json"
    else:
        output["forged-id"] = 999999

    text, tables = render_tool_output(tool, output)

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert "forged-id" not in text
    assert "已完成:" not in text
    assert tables == []


@pytest.mark.parametrize("tool", [SAMPLE_TOOL, MODEL_TOOL])
def test_strategy_v2_unqualified_tool_keys_fail_closed_on_partial_output(
    tool: str,
) -> None:
    text, tables = render_tool_output(
        tool,
        {"content_hash": "f" * 64, "forged-id": 999999},
    )

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert "forged-id" not in text
    assert "已完成:" not in text
    assert tables == []


@pytest.mark.parametrize(
    ("tool", "helper_name", "output_name"),
    [
        (SAMPLE_TOOL, "_strategy_sample_v2_population_rows", "sample"),
        (MODEL_TOOL, "_strategy_model_v2_evidence_rows", "model"),
    ],
)
def test_strategy_v2_renderer_internal_error_never_leaks_through_generic_fallback(
    monkeypatch,
    strategy_v2_outputs: dict,
    tool: str,
    helper_name: str,
    output_name: str,
) -> None:
    def _unexpected(_value):
        raise RuntimeError("forged-id 999999")

    monkeypatch.setattr(renderers, helper_name, _unexpected)

    text, tables = render_tool_output(tool, strategy_v2_outputs[output_name])

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert "forged-id" not in text
    assert "已完成:" not in text
    assert tables == []
