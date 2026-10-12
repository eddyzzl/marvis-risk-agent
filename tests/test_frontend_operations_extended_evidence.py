"""Installed Operations envelopes expose actual metrics without null coercion."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _node(value, assertions):
    script = """
import assert from 'node:assert/strict';
import {evidenceHtml,scheduleHtml} from './marvis/static/js/operations-view.js';
let input=''; for await (const chunk of process.stdin) input+=chunk;
const value=JSON.parse(input);
""" + assertions
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(value), cwd=ROOT, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_native_feedback_flat_output_displays_red_counts_rates_and_unknowns():
    _node({
        "schema_version": "operations.decision_feedback_monitor.v1",
        "level": "red",
        "binding": {"task_id": "task", "environment": "local-reference:production",
                    "package_hash": "package", "complete_through": "2026-10-01T00:00:00Z",
                    "label_maturity_seconds": 0},
        "tool_output": {"overall_level": "red", "population_count": 2,
                        "labeled_count": 2, "label_coverage": 1, "bad_count": 1,
                        "bad_rate": .5, "approval_labeled_count": 1,
                        "approval_bad_rate": 0, "fallback_count": 0, "fallback_rate": 0,
                        "score_psi": None, "economic_outcome": None,
                        "rule_replay": "unavailable_original_features_not_retained",
                        "outcome_assurance": "publisher_declared",
                        "label_maturity_assurance": "publisher_declared_elapsed_time_checked",
                        "source_refs": [{"dataset_id": "source1", "content_hash": "hash1"},
                                        {"dataset_id": "source2", "content_hash": "hash2"}]},
    }, r"""
const html=evidenceHtml(value);
for(const row of ['实际决策数</td><td>2','已成熟标签数</td><td>2','标签覆盖率</td><td>100%',
 '坏账率</td><td>50%','通过决策坏账率</td><td>0%',
 '分数 PSI</td><td>未知 / 无可用值','经济结果</td><td>未知 / 无可用值',
 'source1</td><td>hash1','source2</td><td>hash2']) assert.ok(html.includes(row),row);
assert.match(html,/ops-badge red/);assert.match(html,/未留存原始决策特征，规则重放不可用/);
assert.match(html,/发布者声明，未独立认证真实结果/);
assert.match(html,/标签成熟度依据发布者声明及可得时间检查/);
assert.doesNotMatch(html,/此工具未提供统一检查表|当前无监控工具输出/);
const missing=evidenceHtml({...value,tool_output:{overall_level:'not_available',score_psi:null,economic_outcome:null}});
assert.ok(missing.includes('坏账率</td><td>未知 / 无可用值'));
assert.ok(missing.includes('实际决策数</td><td>未知 / 无可用值'));
assert.match(missing,/标签成熟认证状态未知/);
const invalid=evidenceHtml({...value,tool_output:{population_count:'2',bad_rate:2,score_psi:Infinity,economic_outcome:NaN}});
assert.ok(invalid.includes('实际决策数</td><td>未知 / 无可用值'));
assert.ok(invalid.includes('坏账率</td><td>未知 / 无可用值'));
""")


def test_native_portfolio_tool_results_show_actual_loss_migration_and_null_profit():
    _node({
        "schema_version": "operations.periodic_portfolio.v1", "level": "red",
        "source_dataset_id": "source", "source_content_hash": "source-hash",
        "coverage_assurance": "publisher_declared", "window_row_count": 11,
        "current_window_row_count": 3, "red_flag_count": 1,
        "business_acceptance": "not_established",
        "tool_results": {
            "flow_rate": {"output": {"net_flows": [{"month": "2026-08", "into_bad": 1000, "out_of_bad": 0}]}},
            "bucket_migration": {"output": {"to_states": ["current", "bad"], "heat_table": [{"from": "current", "current": .5, "bad": .5}]}},
            "segment_profile": {"output": {"segments": [{"segment": "<north>", "count": 6, "pop_pct": .5, "bad_rate": None, "avg_score": None, "net_profit": None}], "red_flags": [{"message": "集中度风险"}]}},
            "expected_loss_estimate": {"output": {"total_el": 1000, "assumptions": {"reference_snapshot": "2026-09", "lgd": .5, "horizon_months": 1}}},
        },
    }, r"""
const html=evidenceHtml(value);
assert.ok(html.includes('预期损失</td><td>1000'));
assert.ok(html.includes('损失比例假设</td><td>50%'));
assert.ok(html.includes('2026-08</td><td>1000</td><td>0'));
assert.ok(html.includes('current</td><td>50%</td><td>50%'));
assert.ok(html.includes('&lt;north&gt;</td><td>6</td><td>50%</td><td>未知 / 无可用值</td><td>未知 / 无可用值</td><td>未知 / 无可用值'));
assert.match(html,/集中度风险/);assert.match(html,/业务验收尚未建立/);
assert.match(html,/预期损失估计不是实际经济结果/);
assert.doesNotMatch(html,/当前无监控工具输出/);
assert.match(html,/展开完整确定性证据/);
const missing=evidenceHtml({...value,tool_results:{expected_loss_estimate:{output:{total_el:null}}}});
assert.ok(missing.includes('预期损失</td><td>未知 / 无可用值'));
""")


def test_monthly_calendar_and_each_native_binding_retain_exact_context():
    _node({}, r"""
const base={schedule_id:'monthly',revision:2,enabled:true,monitoring_ref:'analysis.portfolio_run',
 calendar:{schema_version:'operations.calendar.months.v1',interval_months:1,timezone:'UTC'},
 portfolio_binding:{source:{task_id:'task',target_id:'portfolio_snapshots',complete_through:'2026-10-01T00:00:00Z'}}};
const html=scheduleHtml({contract:base},[],false);
assert.match(html,/每 1 个月 · UTC/);assert.match(html,/任务<\/dt><dd>task/);
assert.match(html,/portfolio_snapshots/);assert.doesNotMatch(html,/undefined 秒/);
const feedback=scheduleHtml({contract:{...base,portfolio_binding:null,
 calendar:{interval_seconds:60},feedback_binding:{task_id:'feedback-task',package_hash:'package',complete_through:'2026-10-01T00:00:00Z'}}},[],false);
assert.match(feedback,/60 秒/);assert.match(feedback,/feedback-task/);assert.match(feedback,/目标<\/dt><dd>package/);
assert.match(scheduleHtml({contract:{...base,calendar:{}}},[],false),/周期未知/);
""")
