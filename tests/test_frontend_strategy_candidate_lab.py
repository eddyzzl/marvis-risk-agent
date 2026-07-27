from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_cross_rule_launchers_use_authenticated_fields_and_exact_rule_pointers():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          STRATEGY_CANDIDATE_LAB_WORKFLOWS,
          collectStrategyCandidateLabRequest,
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function projectionOption(value, dataset = {}) {
          return {
            value,
            dataset: { candidateLabProjection: "1", ...dataset },
          };
        }
        function makeForm(workflow, values = {}, fields = {}) {
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              if (!match) return null;
              return fields[match[1]] || { value: values[match[1]] ?? "" };
            },
            querySelectorAll() { return []; },
          };
        }

        const age = projectionOption("age", { feature: "age" });
        const score = projectionOption("score", { feature: "score" });
        const search = collectStrategyCandidateLabRequest(makeForm(
          "cross_rule_search",
          {
            cross_rule_dimension: "2",
            cross_rule_min_lift: "1.5",
            cross_rule_min_bad_count: "20",
            cross_rule_max_hit_share: "0.3",
            cross_rule_min_amount_lift: "",
            cross_rule_max_trials: "500",
          },
          {
            cross_rule_features: { selectedOptions: [age, score] },
          },
        ));
        assert.deepEqual(search, {
          request_kind: "standard_workflow",
          workflow: "cross_rule_search",
          workflow_inputs: {
            features: ["age", "score"],
            dimension: 2,
            constraints: {
              min_lift: 1.5,
              min_bad_count: 20,
              max_hit_share: 0.3,
              min_amount_lift: null,
            },
            max_trials: 500,
          },
        });

        const searchId = `cross-rule-search-${"a".repeat(32)}`;
        const ruleId = `cross-rule-${"b".repeat(32)}`;
        const build = collectStrategyCandidateLabRequest(makeForm(
          "cross_rule_candidate_build_from_search",
          { cross_rule_selection_reason: "人工风险评审。" },
          {
            cross_rule_build_search_id: {
              value: searchId,
              selectedOptions: [
                projectionOption(searchId, { searchId }),
              ],
            },
            cross_rule_build_rule_id: {
              value: ruleId,
              selectedOptions: [
                projectionOption(ruleId, { searchId, ruleId }),
              ],
            },
          },
        ));
        assert.deepEqual(build, {
          request_kind: "standard_workflow",
          workflow: "cross_rule_candidate_build_from_search",
          workflow_inputs: {
            search_id: searchId,
            rule_id: ruleId,
            selection_reason: "人工风险评审。",
          },
        });
        assert.ok(STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes("cross_rule_search"));

        const html = strategyCandidateLabResultsHtml({
          candidates: {
            cross_rule_search: {
              all: [{
                search_id: searchId,
                dimension: 2,
                features: [],
                constraints: { min_lift: 1.5 },
                max_trials: 500,
                search_space: 1000,
                evaluated: 500,
                eligible: 3,
                truncated: true,
                rules_truncated: false,
                rules: [{
                  rule_id: ruleId,
                  rank: 1,
                  conditions: [],
                  metrics: { lift: 2.1, hit_share: 0.1 },
                  eligible: true,
                  constraint_failures: [],
                }],
              }],
              total: 1,
            },
          },
        });
        assert.match(html, /不会把第一名当成冠军/);
        assert.match(html, new RegExp(ruleId));
      """
    )


def test_candidate_lab_api_uses_task_owned_get_and_existing_agent_message_envelope():
    run_node(
        """
        import assert from "node:assert/strict";

        globalThis.document = {
          body: { dataset: { marvisLocalToken: "local-token" } },
        };
        const {
          getStrategyCandidateLab,
          submitStrategyCandidateLabRequest,
        } = await import("./marvis/static/js/v2/api_v2.js");

        const calls = [];
        globalThis.fetch = async (url, options = {}) => {
          calls.push({ url, options });
          return {
            ok: true,
            status: 200,
            headers: { get: () => "application/json" },
            json: async () => ({ task_id: "task / 1", messages: [] }),
            text: async () => "",
          };
        };

        await getStrategyCandidateLab("task / 1");
        assert.equal(
          calls.at(-1).url,
          "/api/tasks/task%20%2F%201/strategy-candidate-lab",
        );
        assert.equal(calls.at(-1).options.method, "GET");

        const strategyRequest = {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_analysis",
          workflow_inputs: { features: ["score"] },
        };
        await submitStrategyCandidateLabRequest(
          "task / 1",
          strategyRequest,
          "启动单变量候选分析",
        );
        const post = calls.at(-1);
        assert.equal(post.url, "/api/tasks/task%20%2F%201/agent/messages");
        assert.equal(post.options.method, "POST");
        assert.equal(post.options.headers["X-Marvis-Token"], "local-token");
        assert.deepEqual(JSON.parse(post.options.body), {
          content: "启动单变量候选分析",
          strategy_request: strategyRequest,
        });
        assert.equal("model_id" in JSON.parse(post.options.body), false);
        """
    )


def test_primary_launchers_emit_only_user_owned_inputs_and_omit_empty_methods():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function makeForm(workflow, values = {}, checked = {}) {
          const fields = new Map(
            Object.entries(values).map(([key, value]) => [key, { value }]),
          );
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? fields.get(match[1]) || null : null;
            },
            querySelectorAll(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match
                ? (checked[match[1]] || []).map((value) => ({ value }))
                : [];
            },
          };
        }

        const univariate = collectStrategyCandidateLabRequest(makeForm(
          "univariate_candidate_analysis",
          {
            features: "score，age",
            bin_count: "5",
            min_bin_pct: "0.02",
            loan_amount_col: "loan_amount",
            overdue_amount_col: "",
            sentinel_values: "number:-9999, text:UNKNOWN",
            manual_breakpoints: "",
          },
          { methods: [] },
        ));
        assert.deepEqual(univariate, {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_analysis",
          workflow_inputs: {
            features: ["score", "age"],
            bin_count: 5,
            min_bin_pct: 0.02,
            loan_amount_col: "loan_amount",
            sentinel_values: [-9999, "UNKNOWN"],
          },
        });
        assert.equal("methods" in univariate.workflow_inputs, false);

        const cross = collectStrategyCandidateLabRequest(makeForm(
          "cross_matrix_analysis",
          {
            x_feature: "age",
            x_method: "equal_frequency",
            y_feature: "score",
            y_method: "equal_width",
            bin_count: "4",
            min_bin_pct: "0.03",
            x_manual_breakpoints: "",
            y_manual_breakpoints: "",
            loan_amount_col: "",
            overdue_amount_col: "",
            sentinel_values: "",
          },
        ));
        assert.deepEqual(cross.workflow_inputs, {
          x_feature: "age",
          x_method: "equal_frequency",
          y_feature: "score",
          y_method: "equal_width",
          bin_count: 4,
          min_bin_pct: 0.03,
        });
        assert.equal("features" in cross.workflow_inputs, false);
        assert.equal("methods" in cross.workflow_inputs, false);

        const tree = collectStrategyCandidateLabRequest(makeForm(
          "automatic_tree_candidate_build",
          {
            features: "score;income",
            directions: "score=decreasing；income=unordered",
            max_depth: "3",
            min_leaf_count: "20",
            min_weight_fraction_leaf: "",
            seed: "42",
            sample_weight_col: "",
            loan_amount_col: "loan_amount",
            overdue_amount_col: "",
          },
        ));
        assert.deepEqual(tree.workflow_inputs, {
          features: ["score", "income"],
          directions: { score: "decreasing", income: "unordered" },
          max_depth: 3,
          min_leaf_count: 20,
          seed: 42,
          loan_amount_col: "loan_amount",
        });

        const platformOwned = new Set([
          "artifact_id",
          "content_hash",
          "dataset_id",
          "expected_content_hash",
          "sample_design_ref",
          "target_col",
          "workspace_revision",
        ]);
        for (const request of [univariate, cross, tree]) {
          for (const key of Object.keys(request.workflow_inputs)) {
            assert.equal(platformOwned.has(key), false, key);
            assert.equal(key.startsWith("expected_"), false, key);
          }
        }
        """
    )


def test_v2_workflow_spine_launchers_emit_only_explicit_user_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          STRATEGY_CANDIDATE_LAB_WORKFLOWS,
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function projectionOption(value) {
          return {
            value,
            dataset: {
              candidateLabProjection: "1",
              strategyType: value,
            },
          };
        }
        function makeForm(workflow, values = {}, fields = {}, checked = {}) {
          const controls = new Map(
            Object.entries(values).map(([key, value]) => [key, { value }]),
          );
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match
                ? fields[match[1]] || controls.get(match[1]) || null
                : null;
            },
            querySelectorAll(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? (checked[match[1]] || []) : [];
            },
          };
        }

        for (const workflow of [
          "strategy_sample_design_v2",
          "strategy_pool_impact",
          "strategy_impact_cube",
          "strategy_report_bundle_v2",
        ]) {
          assert.ok(STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes(workflow));
        }

        const sampleValues = {
          sample_target_bad_value: "1",
          sample_relationship: "nested_same_cohort",
          approval_population_column: "",
          approval_population_operator: "",
          approval_population_value: "",
          risk_population_column: "",
          risk_population_operator: "",
          risk_population_value: "",
          sample_time_field: "apply_date",
          sample_development_start: "2026-01-01",
          sample_development_end: "2026-03-31",
          sample_validation_start: "2026-04-01",
          sample_validation_end: "2026-04-30",
          sample_oot_start: "2026-05-01",
          sample_oot_end: "2026-05-31",
          sample_maturity_status: "unknown",
          sample_maturity_days: "",
          sample_maturity_cutoff: "",
          sample_maturity_reason: "暂未确认成熟度",
          sample_performance_status: "unavailable",
          sample_performance_days: "",
          sample_observation_status: "unavailable",
          sample_observation_start: "",
          sample_observation_end: "",
          sample_entity_field: "",
          sample_group_field: "",
          sample_month_field: "",
          sample_weight_field: "",
          sample_loan_amount_field: "",
          sample_overdue_amount_field: "",
          sample_historical_score_status: "unavailable",
          sample_historical_score_column: "",
          sample_historical_score_direction: "",
          sample_historical_score_reason: "暂未提供历史分",
        };
        const sampleDesign = collectStrategyCandidateLabRequest(makeForm(
          "strategy_sample_design_v2",
          sampleValues,
          { sample_drop_nan_labels: { checked: true } },
        ));
        assert.equal(sampleDesign.workflow_inputs.partitioning.column, "apply_date");
        assert.deepEqual(sampleDesign.workflow_inputs.maturity, {
          status: "unknown",
          performance_window_days: null,
          cutoff_date: null,
          reason: "暂未确认成熟度",
        });
        assert.deepEqual(sampleDesign.workflow_inputs.performance_window, {
          status: "unavailable",
          days: null,
        });
        assert.deepEqual(sampleDesign.workflow_inputs.observation_window, {
          status: "unavailable",
          start: null,
          end: null,
        });
        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm(
            "strategy_sample_design_v2",
            {
              ...sampleValues,
              sample_validation_start: "",
              sample_validation_end: "",
            },
            { sample_drop_nan_labels: { checked: true } },
          )),
          /验证集.*至少填写一个时间边界/,
        );

        const poolOption = projectionOption("approval");
        const poolImpact = collectStrategyCandidateLabRequest(makeForm(
          "strategy_pool_impact",
          {
            pool_impact_comparison_mode: "absolute",
            pool_impact_baseline_strategy_id: "",
            pool_impact_month_col: "apply_month",
            pool_impact_loan_amount_col: "",
            pool_impact_overdue_amount_col: "",
          },
          {
            pool_impact_strategy_type: {
              value: "approval",
              selectedOptions: [poolOption],
            },
            pool_impact_drop_nan_labels: { checked: true },
          },
        ));
        assert.deepEqual(poolImpact.workflow_inputs, {
          strategy_type: "approval",
          comparison_mode: "absolute",
          drop_nan_labels: true,
          month_col: "apply_month",
        });

        const impactCube = collectStrategyCandidateLabRequest(makeForm(
          "strategy_impact_cube",
          {
            impact_cube_month_col: "apply_month",
            impact_cube_group_col: "channel",
            impact_cube_segment_col: "",
            impact_cube_current_strategy_id: "strategy-current",
          },
          {
            impact_cube_strategy_type: {
              value: "approval",
              selectedOptions: [poolOption],
            },
          },
          {
            impact_cube_partitions: [
              { value: "development" },
              { value: "validation" },
            ],
          },
        ));
        assert.deepEqual(impactCube.workflow_inputs, {
          strategy_type: "approval",
          partitions: ["development", "validation"],
          month_col: "apply_month",
          group_col: "channel",
          current_strategy_id: "strategy-current",
        });

        const report = collectStrategyCandidateLabRequest(makeForm(
          "strategy_report_bundle_v2",
          {
            strategy_report_title: "风险策略迭代评审",
            strategy_report_status: "partial",
          },
        ));
        assert.deepEqual(report.workflow_inputs, {
          title: "风险策略迭代评审",
          status: "partial",
        });

        for (const request of [
          sampleDesign,
          poolImpact,
          impactCube,
          report,
        ]) {
          for (const forbidden of [
            "artifact_id",
            "pool_ref",
            "sample_design_ref",
            "dataset_id",
            "target_col",
            "revision",
            "snapshot_hash",
          ]) {
            assert.equal(forbidden in request.workflow_inputs, false);
          }
        }
        """
    )


def test_candidate_lab_renders_seven_stage_dual_population_and_report_spine():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const stages = [
          ["current_context", "项目现状", "complete"],
          ["history", "历史版本", "missing"],
          ["sample_design", "样本设计", "complete"],
          ["candidate_analysis", "单变量/模型", "complete"],
          ["strategy_combination", "交叉组合/策略", "complete"],
          ["impact", "影响测算", "stale"],
          ["report", "形成报告", "complete"],
        ].map(([id, label, status]) => ({ id, label, status }));
        const sample = {
          source_mode: "native_active_dataset",
          relationship: "parallel_time_cohorts",
          freshness: "current",
          analysis_universe_count: 120,
          target: { column: "bad", bad_value: 1, good_value: 0 },
          relationship_counts: {
            approval_and_risk: 10,
            approval_only: 30,
            risk_only: 70,
            neither: 10,
          },
          diagnostics: { overall_status: "warn" },
          populations: {
            approval: {
              total_count: 40,
              partitions: { development: 30, validation: 5, oot: 5 },
              maturity: { status: "not_applicable" },
            },
            risk: {
              total_count: 80,
              partitions: { development: 60, validation: 10, oot: 10 },
              maturity: {
                status: "confirmed_matured",
                performance_window_days: 30,
                cutoff_date: "2026-05-31",
                eligible_count: 80,
                labeled_count: 80,
              },
            },
          },
          artifact: {
            download_url: "/api/tasks/task-1/task-artifacts/sample/download",
          },
          membership_token: "NEVER_RENDER_RAW_MEMBERSHIP",
        };
        const reportArtifacts = Object.fromEntries(
          ["json", "markdown", "xlsx", "docx"].map((format) => [
            format,
            {
              download_url:
                `/api/tasks/task-1/task-artifacts/report-${format}/download`,
            },
          ]),
        );
        const html = strategyCandidateLabResultsHtml({
          workflow: {
            stages,
            sample_design: sample,
            latest_evidence: {
              pool_stability: {
                freshness: "stale",
                strategy_type: "approval",
                pool_revision: 2,
                artifact: {
                  download_url:
                    "/api/tasks/task-1/task-artifacts/stability/download",
                },
              },
              pool_impact: null,
              impact_cube: null,
              pool_validation: { validation: null, oot: null },
            },
            report: {
              report_id: "strategy-report-1",
              revision: 3,
              status: "partial",
              title: "风险策略迭代评审",
              freshness: "current",
              artifacts: reportArtifacts,
            },
          },
          candidates: {},
          pools: {},
        });

        assert.equal(
          (html.match(/class="candidate-lab-workflow-stage"/g) || []).length,
          7,
        );
        for (const expected of [
          "策略开发全流程",
          "审批人群",
          "风险表现人群",
          "parallel_time_cohorts",
          "confirmed_matured",
          "需刷新",
          "策略迭代评审报告",
          "JSON",
          "Markdown",
          "Excel",
          "Word",
          "report-docx/download",
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.equal(html.includes("NEVER_RENDER_RAW_MEMBERSHIP"), false);

        const missing = strategyCandidateLabResultsHtml({
          workflow: {
            stages,
            sample_design: null,
            latest_evidence: {
              pool_validation: { validation: null, oot: null },
            },
            report: null,
          },
          candidates: {},
          pools: {},
        });
        assert.match(missing, /尚无当前受认证 SampleDesign V2/);
        assert.match(missing, /尚未形成报告/);
        """
    )


def test_candidate_lab_renders_project_context_history_and_missing_information():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const html = strategyCandidateLabResultsHtml({
          workflow: {
            stages: [],
            project_context: {
              revision_id: "strategy-project-context-revision-123",
              revision: 2,
              as_of: "2026-07-27",
              freshness: "current",
              scope: { availability: "present", value: "存量复借策略" },
              current: {
                snapshot_id: "current-project-snapshot-123",
                status_fields: {
                  volume: {
                    availability: "present",
                    value: [{ metric_key: "application_count", value: 1000 }],
                  },
                  approval: { availability: "unavailable", value: null },
                  risk: {
                    availability: "present",
                    value: [{ metric_key: "bad_rate", value: 0.031 }],
                  },
                  economics: { availability: "unavailable", value: null },
                },
                maturity_summary: {
                  availability: "present",
                  value: { status: "confirmed_matured" },
                },
                red_flags: [],
              },
              historical_versions: [{
                review_id: "historical-strategy-review-123",
                version: 3,
                effective_period: {
                  availability: "present",
                  value: { start: "2026-01-01", end: "2026-03-31" },
                },
                asset_status: {
                  availability: "present",
                  value: "adopted_local",
                },
                scope: { availability: "present", value: "复借客群" },
                traffic_allocation: {
                  availability: "unavailable",
                  value: null,
                },
                availability: "present",
                effect_stages: ["backtested", "oot_validated"],
                external_source_count: 1,
                red_flags: [],
              }],
              history_resolution: "present",
              missing_information: [{
                field_path: "current.status_fields.economics",
                status: "pending",
                blocking: "report_optional",
                question: "如有收益口径，请提供。",
                reason: "No governed economics evidence is available.",
                asked_count: 1,
              }],
              red_flags: [],
              artifact: {
                download_url:
                  "/api/tasks/task-1/task-artifacts/project-context/download",
              },
            },
            sample_design: null,
            latest_evidence: { pool_validation: {} },
            report: null,
          },
          candidates: {},
          pools: {},
        });

        assert.match(html, /项目现状与历史版本/);
        assert.match(html, /存量复借策略/);
        assert.match(html, /版本 3/);
        assert.match(html, /backtested/);
        assert.match(html, /oot_validated/);
        assert.match(html, /如有收益口径，请提供。/);
        assert.match(html, /下载项目上下文/);
        assert.doesNotMatch(html, /No governed economics evidence/);
        """
    )


def test_workbench_context_materialization_and_delivery_collect_only_user_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function makeForm(workflow, fields = {}, lists = {}) {
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? (fields[match[1]] || null) : null;
            },
            querySelectorAll(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              const values = match ? (lists[match[1]] || []) : [];
              return selector.includes(":checked")
                ? values.filter((item) => item.checked)
                : values;
            },
          };
        }

        const context = collectStrategyCandidateLabRequest(makeForm(
          "strategy_project_context",
          {
            project_context_as_of: { value: "2026-07-27" },
            project_context_scope: { value: "存量复借策略" },
            project_context_business_context: {
              value: "project.channel=自营\\nproject.product=云闪付",
            },
            project_context_external_reports: {
              value: "历史策略.xlsx\\n历史复盘.pdf",
            },
          },
          {
            project_context_unavailable: [
              { checked: true, value: "current.status_fields.economics" },
              { checked: false, value: "historical_strategy_reviews" },
            ],
          },
        ));
        assert.deepEqual(context, {
          request_kind: "standard_workflow",
          workflow: "strategy_project_context",
          workflow_inputs: {
            as_of: "2026-07-27",
            scope: "存量复借策略",
            business_context: {
              "project.channel": "自营",
              "project.product": "云闪付",
            },
            explicit_unavailable: ["current.status_fields.economics"],
            external_report_filenames: ["历史策略.xlsx", "历史复盘.pdf"],
          },
        });

        const materializeOption = {
          value: "approval",
          dataset: {
            candidateLabProjection: "1",
            poolId: "strategy-pool-1",
          },
        };
        const materialize = collectStrategyCandidateLabRequest(makeForm(
          "strategy_pool_materialize",
          {
            pool_materialize_strategy_type: {
              value: "approval",
              selectedOptions: [materializeOption],
            },
          },
        ));
        assert.deepEqual(materialize.workflow_inputs, {
          strategy_type: "approval",
        });

        const strategyOption = {
          value: "strategy-current-v2",
          dataset: {
            candidateLabProjection: "1",
            strategyId: "strategy-current-v2",
          },
        };
        const delivery = collectStrategyCandidateLabRequest(makeForm(
          "strategy_dsl_delivery",
          {
            dsl_delivery_strategy_id: {
              value: "strategy-current-v2",
              selectedOptions: [strategyOption],
            },
          },
        ));
        assert.deepEqual(delivery.workflow_inputs, {
          strategy_id: "strategy-current-v2",
        });

        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm(
            "strategy_project_context",
            {
              project_context_as_of: { value: "2026-07-27" },
              project_context_scope: { value: "" },
              project_context_business_context: {
                value: "artifact_id=forged",
              },
              project_context_external_reports: { value: "" },
            },
            { project_context_unavailable: [] },
          )),
          /字段路径|不允许/,
        );
      """
    )


def test_scorecard_launchers_submit_only_visible_projection_ids_and_user_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function makeForm(workflow, values = {}, fields = {}) {
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              if (!match) return null;
              return fields[match[1]] || { value: values[match[1]] || "" };
            },
            querySelectorAll() { return []; },
          };
        }

        const equalFrequency = collectStrategyCandidateLabRequest(makeForm(
          "scorecard_band_build",
          {
            scorecard_banding_mode: "equal_frequency",
            scorecard_bin_count: "10",
            raw_pd_band_edges: "",
          },
        ));
        assert.deepEqual(equalFrequency, {
          request_kind: "standard_workflow",
          workflow: "scorecard_band_build",
          workflow_inputs: { bin_count: 10 },
        });

        const manualEdges = collectStrategyCandidateLabRequest(makeForm(
          "scorecard_band_build",
          {
            scorecard_banding_mode: "raw_pd_edges",
            scorecard_bin_count: "10",
            raw_pd_band_edges: "0, 0.2, 0.75, 1",
          },
        ));
        assert.deepEqual(manualEdges.workflow_inputs, {
          raw_pd_band_edges: [0, 0.2, 0.75, 1],
        });
        for (const forbidden of [
          "artifact_id",
          "asset_hash",
          "content_hash",
          "score_evidence_ref",
          "sample_design_ref",
        ]) {
          assert.equal(forbidden in manualEdges.workflow_inputs, false);
        }

        const assetId = `scorecard-band-asset-${"a".repeat(32)}`;
        const cutoffId = `scorecard-cutoff-${"b".repeat(32)}`;
        const assetOption = {
          value: assetId,
          dataset: { candidateLabProjection: "1" },
        };
        const cutoffOption = {
          value: cutoffId,
          dataset: {
            candidateLabProjection: "1",
            sourceAssetId: assetId,
          },
        };
        const selection = collectStrategyCandidateLabRequest(makeForm(
          "scorecard_cutoff_selection",
          { scorecard_selection_reason: "业务评审选择该观测切点" },
          {
            scorecard_asset_id: {
              value: assetId,
              selectedOptions: [assetOption],
              options: [assetOption],
            },
            scorecard_cutoff_id: {
              value: cutoffId,
              selectedOptions: [cutoffOption],
              options: [cutoffOption],
            },
          },
        ));
        assert.deepEqual(selection, {
          request_kind: "standard_workflow",
          workflow: "scorecard_cutoff_selection",
          workflow_inputs: {
            asset_id: assetId,
            cutoff_id: cutoffId,
            reason: "业务评审选择该观测切点",
          },
        });
        assert.deepEqual(Object.keys(selection.workflow_inputs).sort(), [
          "asset_id",
          "cutoff_id",
          "reason",
        ]);

        const forgedAsset = {
          scorecard_asset_id: {
            value: assetId,
            selectedOptions: [{ value: assetId, dataset: {} }],
            options: [],
          },
          scorecard_cutoff_id: {
            value: cutoffId,
            selectedOptions: [cutoffOption],
            options: [cutoffOption],
          },
        };
        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm(
            "scorecard_cutoff_selection",
            {},
            forgedAsset,
          )),
          /受认证投影/,
        );
        """
    )


def test_voting_launchers_submit_only_user_controls_and_authenticated_pointers():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function projectionOption(value, dataset = {}) {
          return {
            value,
            dataset: { candidateLabProjection: "1", ...dataset },
          };
        }

        function makeForm(workflow, values = {}, fields = {}) {
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              if (!match) return null;
              return fields[match[1]] || { value: values[match[1]] || "" };
            },
            querySelectorAll() { return []; },
          };
        }

        const includeId = `candidate-rule-${"a".repeat(32)}`;
        const excludeId = `candidate-rule-${"b".repeat(32)}`;
        const strategyType = projectionOption("approval");
        const include = projectionOption(includeId);
        const exclude = projectionOption(excludeId);
        const search = collectStrategyCandidateLabRequest(makeForm(
          "voting_candidate_search",
          {
            voting_member_count: "3",
            voting_n: "2",
            voting_objective_metric: "bad_rate",
            voting_objective_direction: "maximize",
            voting_constraints: "hit_share >= 0.10\\nbad_rate <= 0.30",
            voting_max_combinations: "250",
          },
          {
            voting_strategy_type: {
              value: "approval",
              selectedOptions: [strategyType],
            },
            voting_include_rule_ids: {
              selectedOptions: [include],
            },
            voting_exclude_rule_ids: {
              selectedOptions: [exclude],
            },
          },
        ));
        assert.deepEqual(search, {
          request_kind: "standard_workflow",
          workflow: "voting_candidate_search",
          workflow_inputs: {
            strategy_type: "approval",
            member_count: 3,
            n: 2,
            objective: { metric: "bad_rate", direction: "maximize" },
            constraints: [
              { metric: "bad_rate", operator: "lte", value: 0.3 },
              { metric: "hit_share", operator: "gte", value: 0.1 },
            ],
            include_rule_ids: [includeId],
            exclude_rule_ids: [excludeId],
            max_combinations: 250,
          },
        });

        const searchId = `voting-search-${"c".repeat(32)}`;
        const comboId = `voting-combo-${"d".repeat(32)}`;
        const searchOption = projectionOption(searchId, {
          strategyType: "approval",
        });
        const comboOption = projectionOption(comboId, {
          sourceSearchId: searchId,
        });
        const build = collectStrategyCandidateLabRequest(makeForm(
          "voting_candidate_build_from_search",
          {},
          {
            voting_search_id: {
              value: searchId,
              selectedOptions: [searchOption],
            },
            voting_combo_id: {
              value: comboId,
              selectedOptions: [comboOption],
            },
          },
        ));
        assert.deepEqual(build, {
          request_kind: "standard_workflow",
          workflow: "voting_candidate_build_from_search",
          workflow_inputs: {
            search_id: searchId,
            combo_id: comboId,
            strategy_type: "approval",
          },
        });

        for (const forbidden of [
          "pool_ref",
          "artifact_id",
          "content_hash",
          "dataset_id",
          "target_col",
          "hit_matrix",
          "score_vector",
        ]) {
          assert.equal(forbidden in search.workflow_inputs, false);
          assert.equal(forbidden in build.workflow_inputs, false);
        }

        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm(
            "voting_candidate_build_from_search",
            {},
            {
              voting_search_id: {
                value: searchId,
                selectedOptions: [{ value: searchId, dataset: {} }],
              },
              voting_combo_id: {
                value: comboId,
                selectedOptions: [comboOption],
              },
            },
          )),
          /受认证投影/,
        );
        """
    )


def test_candidate_lab_renders_voting_search_as_unselected_development_evidence():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const searchId = `voting-search-${"a".repeat(32)}`;
        const comboId = `voting-combo-${"b".repeat(32)}`;
        const ruleId = `candidate-rule-${"c".repeat(32)}`;
        const item = {
          search_id: searchId,
          strategy_type: "approval",
          pool_revision: 7,
          member_count: 3,
          n: 2,
          objective: { metric: "bad_rate", direction: "minimize" },
          constraints: [{ metric: "hit_share", operator: "gte", value: 0.1 }],
          include_rule_ids: [],
          exclude_rule_ids: [],
          max_combinations: 500,
          search_space: 50,
          evaluated: 20,
          eligible: 12,
          truncated: true,
          combinations: [{
            combo_id: comboId,
            members: [ruleId],
            eligible: false,
            failures: [{ metric: "hit_share", operator: "gte", value: 0.1 }],
            metrics: { hit_share: 0.08, bad_rate: 0.2 },
          }],
          artifact: {
            artifact_id: "task-artifact-voting",
            created_at: "2026-07-25T00:00:00+00:00",
            download_url: "/api/tasks/task-1/task-artifacts/artifact/download",
          },
        };
        const html = strategyCandidateLabResultsHtml({
          candidates: {
            voting_search: {
              latest: item,
              all: [item],
              total: 1,
              limit: 20,
              truncated: false,
            },
          },
          pools: { latest: null, all: [], total: 0, truncated: false },
        });

        assert.match(html, /Voting 组合搜索/);
        assert.ok(html.includes(searchId));
        assert.ok(html.includes(comboId));
        assert.ok(html.includes(ruleId));
        assert.match(html, /未构建\\/未入池/);
        assert.match(html, /不表达最佳、冠军或平台选择/);
        for (const secret of [
          "hit_matrix",
          "score_vector",
          "dataset_binding",
          "content_hash",
          "raw_provenance",
        ]) {
          assert.equal(html.includes(secret), false, secret);
        }
        """
    )


def test_scorecard_projection_has_dedicated_band_and_cutoff_evidence_tables():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const artifact = {
          artifact_id: "task-artifact-band",
          content_hash: "a".repeat(64),
          created_at: "2026-07-25T10:00:00Z",
          download_url: "/api/tasks/task-1/task-artifacts/band/download",
          path: "/private/workspace/never-render.json",
        };
        const assetId = `scorecard-band-asset-${"a".repeat(32)}`;
        const cutoffId = `scorecard-cutoff-${"b".repeat(32)}`;
        const band = {
          kind: "scorecard_band",
          artifact,
          candidate_id: assetId,
          lifecycle: { candidate_stage: "development" },
          detail: {
            asset_id: assetId,
            performance: { auc: 0.73, ks: 0.34 },
            sample: {
              row_count: 120,
              development_count: 100,
              labeled_count: 96,
              bad_count: 12,
            },
            directions: {
              raw_pd: {
                direction: "higher_is_riskier",
                meaning: "higher_raw_pd_means_higher_risk",
              },
              scorecard_points: {
                direction: "higher_is_better",
                meaning: "higher_points_mean_safer",
              },
            },
          },
          risks: { red_flags: [], report_info_gaps: [] },
          pointers: {
            bands: [{
              ordinal: 0,
              bin_id: "scorecard-band-0",
              lower_bound: 0,
              upper_bound: 0.2,
              count: 40,
              share: 0.3333,
              labeled_count: 38,
              bad_count: 1,
              bad_rate: 0.0263,
              average_pd: 0.11,
            }],
            cutoffs: [{
              ordinal: 0,
              cutoff_id: cutoffId,
              execution_pd: 0.2,
              display_points: 612.3,
              lower_risk: { count: 40, bad_rate: 0.0263 },
              higher_risk: { count: 80, bad_rate: 0.1375 },
            }],
          },
          total: 2,
          truncated: false,
        };
        const selection = {
          kind: "scorecard_cutoff_selection",
          artifact: { ...artifact, artifact_id: "task-artifact-selection" },
          candidate_id: `scorecard-cutoff-selection-${"c".repeat(32)}`,
          lifecycle: { candidate_stage: "selected" },
          detail: {
            selection_id: `scorecard-cutoff-selection-${"c".repeat(32)}`,
            asset_id: assetId,
            cutoff_id: cutoffId,
            reason: "业务评审选择",
            directions: band.detail.directions,
            effect: band.pointers.cutoffs[0],
          },
          risks: { red_flags: [], report_info_gaps: [] },
          pointers: {},
          total: 1,
          truncated: false,
        };
        const collection = (item) => ({
          latest: item,
          all: [item],
          total: 1,
          truncated: false,
        });
        const html = strategyCandidateLabResultsHtml({
          candidates: {
            scorecard_band: collection(band),
            scorecard_cutoff_selection: collection(selection),
          },
          pools: {},
        });

        for (const expected of [
          "评分卡分档",
          "Cutoff 选择记录",
          "分档证据",
          "Cutoff 观测",
          cutoffId,
          "612.3",
          "原始 PD 越高表示风险越高",
          "评分卡分数越高表示更安全",
          "不等于通过或拒绝动作",
          "不会自动进入 Strategy Pool",
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.equal(html.includes("最佳 Cutoff"), false);
        assert.equal(html.includes("/private/workspace"), false);
        assert.equal(html.includes('{"'), false);
        """
    )


def test_scorecard_band_exposes_collapsible_points_table_without_private_fields():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const band = {
          kind: "scorecard_band",
          artifact: {
            artifact_id: "task-artifact-band",
            download_url: "/api/tasks/task-1/task-artifacts/band/download",
          },
          candidate_id: `scorecard-band-asset-${"a".repeat(32)}`,
          lifecycle: { candidate_stage: "development" },
          detail: {
            asset_id: `scorecard-band-asset-${"a".repeat(32)}`,
            performance: { auc: 0.73, ks: 0.34 },
            sample: { row_count: 120, labeled_count: 96, bad_count: 12 },
          },
          risks: { red_flags: [], report_info_gaps: [] },
          pointers: {
            bands: [],
            cutoffs: [],
            scorecard_points: [
              {
                feature: "__base__",
                bin_index: -999,
                bin_label: "base_points",
                lower: null,
                upper: null,
                count: null,
                bad_count: null,
                good_count: null,
                bad_rate: null,
                woe: null,
                iv_contribution: null,
                coefficient: null,
                monotonic_direction: null,
                points: 320,
                base_score: 600,
                pdo: 50,
                base_odds: 50,
                factor: 72.1348,
                offset: 317.8072,
                asset_hash: "NEVER_RENDER_HASH",
                source_ref: "NEVER_RENDER_REF",
                private_note: "NEVER_RENDER_PRIVATE",
              },
              {
                feature: "income",
                bin_index: 0,
                bin_label: "[-inf, 10)",
                lower: null,
                upper: 10,
                count: 3,
                bad_count: 2,
                good_count: 1,
                bad_rate: 0.6667,
                woe: 0.4,
                iv_contribution: 0.08,
                coefficient: 0.5,
                monotonic_direction: "increasing",
                points: -14,
              },
              {
                feature: "empty_feature",
                bin_index: 1,
                bin_label: "MISSING_VALUE_ROW",
                lower: null,
                upper: null,
                count: null,
                bad_count: null,
                good_count: null,
                bad_rate: null,
                woe: null,
                iv_contribution: null,
                coefficient: null,
                monotonic_direction: null,
                points: null,
              },
            ],
          },
          total: 999,
          truncated: true,
        };
        const html = strategyCandidateLabResultsHtml({
          candidates: {
            scorecard_band: {
              latest: band,
              all: [band],
              total: 1,
              truncated: false,
            },
          },
          pools: {},
        });

        assert.match(
          html,
          /<details[^>]*data-candidate-lab-scorecard-points[^>]*>[\\s\\S]*?<summary>[\\s\\S]*?评分卡分值明细/,
        );
        for (const expected of [
          "评分卡分值越高，代表风险越低",
          "字段",
          "分箱标签",
          "区间",
          "样本数",
          "好样本",
          "坏样本",
          "坏率",
          "WOE",
          "IV 贡献",
          "系数",
          "单调方向",
          "分值",
          "基础分与刻度",
          "基准分",
          "PDO",
          "基准赔率",
          "Factor",
          "Offset",
          "320",
          "600",
          "-∞ ～ 10",
          "当前仅显示前 3 行",
        ]) {
          assert.ok(html.includes(expected), expected);
        }

        const missingRow = html.match(
          /<tr[^>]*>(?:(?!<\\/tr>)[\\s\\S])*MISSING_VALUE_ROW(?:(?!<\\/tr>)[\\s\\S])*<\\/tr>/,
        )?.[0];
        assert.ok(missingRow, "missing-value scorecard row");
        assert.equal(missingRow.includes(">0<"), false);
        assert.ok((missingRow.match(/>-<\\/td>/g) || []).length >= 8);
        for (const forbidden of [
          "NEVER_RENDER_HASH",
          "NEVER_RENDER_REF",
          "NEVER_RENDER_PRIVATE",
          ">null<",
          ">undefined<",
        ]) {
          assert.equal(html.includes(forbidden), false, forbidden);
        }
        """
    )


def test_controller_populates_scorecard_cutoff_selects_from_current_projection():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        class FakeSelect {
          constructor(fieldName) {
            this.options = [];
            this._value = "";
            this.disabled = false;
            this.dataset = { candidateLabField: fieldName };
            this.form = null;
          }
          set innerHTML(html) {
            this.options = Array.from(
              html.matchAll(/<option value="([^"]*)"([^>]*)>(.*?)<\\/option>/g),
            ).map((match) => {
              const attrs = match[2];
              const projection = attrs.match(/data-candidate-lab-projection="([^"]*)"/);
              const source = attrs.match(/data-source-asset-id="([^"]*)"/);
              return {
                value: match[1],
                selected: false,
                dataset: {
                  ...(projection ? { candidateLabProjection: projection[1] } : {}),
                  ...(source ? { sourceAssetId: source[1] } : {}),
                },
              };
            });
            this.value = this.options[0]?.value || "";
          }
          get innerHTML() { return ""; }
          set value(value) {
            this._value = value;
            for (const option of this.options) option.selected = option.value === value;
          }
          get value() { return this._value; }
          get selectedOptions() {
            return this.options.filter((option) => option.selected);
          }
          closest(selector) {
            if (selector === "[data-candidate-lab-field]") return this;
            if (
              selector
              === '[data-candidate-lab-workflow="scorecard_cutoff_selection"]'
            ) return this.form;
            return null;
          }
        }

        const assetId = `scorecard-band-asset-${"a".repeat(32)}`;
        const cutoffId = `scorecard-cutoff-${"b".repeat(32)}`;
        const otherAssetId = `scorecard-band-asset-${"c".repeat(32)}`;
        const otherCutoffId = `scorecard-cutoff-${"d".repeat(32)}`;
        const assetSelect = new FakeSelect("scorecard_asset_id");
        const cutoffSelect = new FakeSelect("scorecard_cutoff_id");
        const reason = { value: "", disabled: false, closest() { return null; } };
        const empty = { textContent: "" };
        const fields = new Map([
          ["scorecard_asset_id", assetSelect],
          ["scorecard_cutoff_id", cutoffSelect],
          ["scorecard_selection_reason", reason],
        ]);
        const selectionForm = {
          dataset: { candidateLabWorkflow: "scorecard_cutoff_selection" },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-scorecard-empty]") return empty;
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? fields.get(match[1]) || null : null;
          },
          querySelectorAll() { return []; },
          reset() {},
        };
        assetSelect.form = selectionForm;
        cutoffSelect.form = selectionForm;
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector(selector) {
            if (
              selector
              === '[data-candidate-lab-workflow="scorecard_cutoff_selection"]'
            ) return selectionForm;
            return null;
          },
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") return [selectionForm];
            if (selector === "[data-candidate-lab-retry]") return [];
            return [assetSelect, cutoffSelect, reason];
          },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => ({ id: "strategy-1", task_type: "strategy" }),
          getSelectedTaskId: () => "strategy-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => {
            const band = (asset, cutoff, pd, points) => ({
              detail: { asset_id: asset },
              pointers: {
                cutoffs: [{
                  cutoff_id: cutoff,
                  execution_pd: pd,
                  display_points: points,
                }],
              },
            });
            const first = band(assetId, cutoffId, 0.42, 608.5);
            const second = band(otherAssetId, otherCutoffId, 0.61, 571.2);
            return {
              task_id: "strategy-1",
              can_start: true,
              blocked_reason: null,
              candidates: {
                scorecard_band: {
                  latest: first,
                  all: [first, second],
                  total: 2,
                  truncated: false,
                },
              },
              pools: {},
            };
          },
        });

        await controller.selectTask({ id: "strategy-1", task_type: "strategy" });
        assert.equal(assetSelect.value, "");
        assert.equal(cutoffSelect.value, "");
        assert.ok(assetSelect.options.some((option) => option.value === assetId));
        assert.ok(assetSelect.options.some((option) => option.value === otherAssetId));
        assert.ok(empty.textContent.includes("先明确选择"));

        assetSelect.value = assetId;
        assert.equal(controller.handleChange({ target: assetSelect }), true);
        assert.equal(
          assetSelect.selectedOptions[0].dataset.candidateLabProjection,
          "1",
        );
        assert.equal(cutoffSelect.value, "");
        assert.ok(cutoffSelect.options.some((option) => option.value === cutoffId));
        assert.throws(
          () => collectStrategyCandidateLabRequest(selectionForm),
          /受认证投影/,
        );

        cutoffSelect.value = cutoffId;
        assert.equal(
          cutoffSelect.selectedOptions[0].dataset.sourceAssetId,
          assetId,
        );
        assert.deepEqual(
          collectStrategyCandidateLabRequest(selectionForm).workflow_inputs,
          { asset_id: assetId, cutoff_id: cutoffId },
        );

        assetSelect.value = otherAssetId;
        assert.equal(controller.handleChange({ target: assetSelect }), true);
        assert.equal(cutoffSelect.value, "");
        assert.ok(
          cutoffSelect.options.some((option) => option.value === otherCutoffId),
        );
        assert.equal(
          cutoffSelect.options.some((option) => option.value === cutoffId),
          false,
        );
        assert.throws(
          () => collectStrategyCandidateLabRequest(selectionForm),
          /受认证投影/,
        );
        """
    )


def test_sentinel_types_are_explicit_and_preserve_textual_leading_zeroes():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function formWithSentinels(sentinelValues) {
          const values = {
            features: "score",
            bin_count: "",
            min_bin_pct: "",
            loan_amount_col: "",
            overdue_amount_col: "",
            sentinel_values: sentinelValues,
            manual_breakpoints: "",
          };
          return {
            dataset: { candidateLabWorkflow: "univariate_candidate_analysis" },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? { value: values[match[1]] || "" } : null;
            },
            querySelectorAll() { return []; },
          };
        }

        const request = collectStrategyCandidateLabRequest(
          formWithSentinels(
            "text:001, number:1, text:1, number:-9999",
          ),
        );
        assert.deepEqual(request.workflow_inputs.sentinel_values, [
          "001",
          1,
          "1",
          -9999,
        ]);
        assert.equal(typeof request.workflow_inputs.sentinel_values[0], "string");
        assert.equal(typeof request.workflow_inputs.sentinel_values[1], "number");

        assert.throws(
          () => collectStrategyCandidateLabRequest(formWithSentinels("001, 1")),
          /必须显式标注类型/,
        );
        """
    )


def test_candidate_stability_launcher_submits_only_current_projection_pointer():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const entryId = `pool-entry-${"a".repeat(32)}`;
        const assetId = `candidate-asset-${"b".repeat(32)}`;
        function form(mode, option) {
          const fields = {
            stability_source_mode: { value: mode },
            stability_pool_entry: {
              value: mode === "pool_entry" ? option.value : "",
              selectedOptions: mode === "pool_entry" ? [option] : [],
            },
            stability_asset_id: {
              value: mode === "univariate_asset" ? option.value : "",
              selectedOptions: mode === "univariate_asset" ? [option] : [],
            },
          };
          return {
            dataset: { candidateLabWorkflow: "candidate_monthly_stability" },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? fields[match[1]] || null : null;
            },
            querySelectorAll() { return []; },
          };
        }

        const poolOption = {
          value: entryId,
          dataset: {
            candidateLabProjection: "1",
            strategyType: "approval",
          },
        };
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form("pool_entry", poolOption)),
          {
            request_kind: "standard_workflow",
            workflow: "candidate_monthly_stability",
            workflow_inputs: {
              strategy_type: "approval",
              entry_id: entryId,
            },
          },
        );

        const assetOption = {
          value: assetId,
          dataset: { candidateLabProjection: "1" },
        };
        assert.deepEqual(
          collectStrategyCandidateLabRequest(
            form("univariate_asset", assetOption),
          ).workflow_inputs,
          { asset_id: assetId },
        );

        assert.throws(
          () => collectStrategyCandidateLabRequest(
            form("pool_entry", {
              value: entryId,
              dataset: { strategyType: "approval" },
            }),
          ),
          /受认证投影/,
        );
        assert.throws(
          () => collectStrategyCandidateLabRequest(
            form("pool_entry", {
              value: entryId,
              dataset: {
                candidateLabProjection: "1",
                strategyType: "",
              },
            }),
          ),
          /策略类型/,
        );
        """
    )


def test_refinement_fresh_and_existing_payloads_match_manual_api_contract():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function makeForm(values, fields = {}) {
          return {
            dataset: {
              candidateLabWorkflow: "univariate_candidate_refinement",
            },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              if (!match) return null;
              return fields[match[1]] || { value: values[match[1]] || "" };
            },
            querySelectorAll() { return []; },
          };
        }

        const fresh = collectStrategyCandidateLabRequest(makeForm({
          refinement_mode: "fresh",
          refinement_feature: "score",
          refinement_method: "manual",
          risk_operator: ">=",
          risk_value: "0.5",
          bin_count: "5",
          min_bin_pct: "0.02",
          loan_amount_col: "loan_amount",
          overdue_amount_col: "",
          sentinel_values: "text:001, number:-9999",
          refinement_manual_breakpoints: "500, 700",
          selection_reason: "保留高风险区间",
        }));
        assert.deepEqual(fresh, {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_refinement",
          workflow_inputs: {
            feature: "score",
            method: "manual",
            selection: {
              risk_threshold: { operator: ">=", value: 0.5 },
            },
            bin_count: 5,
            min_bin_pct: 0.02,
            loan_amount_col: "loan_amount",
            sentinel_values: ["001", -9999],
            manual_breakpoints: { score: [500, 700] },
            selection_reason: "保留高风险区间",
          },
        });

        const candidateId = `candidate-${"a".repeat(32)}`;
        const sourceOption = {
          value: candidateId,
          dataset: { candidateLabProjection: "1" },
        };
        const pairOption = {
          value: "pair-0",
          dataset: {
            candidateLabProjection: "1",
            sourceCandidateId: candidateId,
            feature: "score",
            method: "manual",
          },
        };
        const bin = (value, selected = true) => ({
          value,
          selected,
          dataset: {
            candidateLabProjection: "1",
            sourceCandidateId: candidateId,
            feature: "score",
            method: "manual",
          },
        });
        const bins = [bin("regular:0"), bin("regular:1"), bin("regular:2", false)];
        const fields = {
          source_candidate_id: {
            value: candidateId,
            selectedOptions: [sourceOption],
            options: [sourceOption],
          },
          source_feature_method: {
            value: "pair-0",
            selectedOptions: [pairOption],
            options: [pairOption],
          },
          source_bin_ids: {
            selectedOptions: bins.filter((option) => option.selected),
            options: bins,
          },
        };
        const existing = collectStrategyCandidateLabRequest(makeForm({
          refinement_mode: "existing",
          merge_groups: "regular:0+regular:1",
          selection_reason: "合并并保留两个风险箱",
        }, fields));
        assert.deepEqual(existing, {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_refinement",
          workflow_inputs: {
            feature: "score",
            method: "manual",
            source_candidate_id: candidateId,
            selection: {
              source_bin_ids: ["regular:0", "regular:1"],
            },
            merge_groups: [["regular:0", "regular:1"]],
            selection_reason: "合并并保留两个风险箱",
          },
        });
        for (const forbidden of [
          "artifact_id",
          "content_hash",
          "evidence_hash",
          "expected_content_hash",
        ]) {
          assert.equal(forbidden in existing.workflow_inputs, false);
        }

        const forgedSource = {
          ...fields,
          source_candidate_id: {
            value: candidateId,
            selectedOptions: [{ value: candidateId, dataset: {} }],
            options: [],
          },
        };
        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm({
            refinement_mode: "existing",
          }, forgedSource)),
          /受认证投影/,
        );
        """
    )


def test_candidate_lab_renders_authenticated_collections_without_raw_json_or_paths():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const artifact = (suffix) => ({
          artifact_id: `task-artifact-${suffix}`,
          content_hash: suffix.repeat(64).slice(0, 64),
          created_at: "2026-07-24T10:00:00Z",
          download_url: `/api/tasks/task-1/task-artifacts/${suffix}/download`,
          path: "/private/workspace/never-render-this.json",
        });
        const collection = (item, truncated = false) => ({
          latest: item,
          all: [item],
          total: 1,
          truncated,
        });
        const candidate = (kind, pointerName, pointer, detail) => ({
          kind,
          artifact: artifact(kind.slice(0, 1)),
          candidate_id: `candidate-${kind}`,
          evidence_hash: `evidence-${kind}`,
          lifecycle: {
            candidate_stage: "development",
            validation_status: "unvalidated",
          },
          detail,
          risks: { red_flags: ["仅开发样本"], report_info_gaps: [] },
          pointers: { [pointerName]: [pointer] },
          total: 3,
          truncated: true,
        });
        const univariate = candidate(
          "univariate",
          "bins",
          { feature: "score", method: "equal_width", bin_id: "bin-1" },
          { metrics: { iv: 0.21, ks: 0.33 }, rankings: ["score"] },
        );
        const cross = candidate(
          "cross_matrix",
          "cells",
          {
            cell_id: "cross-cell-1",
            row_bin_id: "age-bin-1",
            column_bin_id: "score-bin-1",
            effect: { bad_rate: 0.08 },
          },
          {
            asset_id: "candidate-asset-cross",
            asset_hash: "cross-hash",
            axes: ["age", "score"],
            summary: { count: 120 },
          },
        );
        const tree = candidate(
          "automatic_tree",
          "leaves",
          {
            leaf_id: "leaf-1",
            fragment_id: "fragment-1",
            rule_id: "rule-1",
            effect_id: "effect-1",
            condition: { op: "lt", field: "score", value: 600 },
            metrics: { bad_rate: 0.12 },
          },
          {
            asset_id: "candidate-asset-tree",
            asset_hash: "tree-hash",
            tree_id: "tree-1",
            tree_result_hash: "tree-result-hash",
            summary: { count: 120 },
          },
        );
        const pool = {
          pool_id: "pool-1",
          strategy_type: "approval",
          revision: 2,
          revision_id: "pool-revision-2",
          snapshot_hash: "pool-hash",
          status: "draft",
          validation_status: "unvalidated",
          default_action: { type: "approval" },
          entries: [{
            position: 0,
            rule_id: "rule-1",
            source: { asset_id: "candidate-asset-tree" },
            action: { type: "reject" },
            execution: { condition: "score < 600" },
            enabled: true,
          }],
          artifact: artifact("p"),
          total: 4,
          truncated: true,
        };
        const html = strategyCandidateLabResultsHtml({
          candidates: {
            univariate: collection(univariate, true),
            cross_matrix: collection(cross),
            automatic_tree: collection(tree),
          },
          pools: collection(pool),
        });

        for (const expected of [
          "单变量候选",
          "Cross Matrix",
          "自动规则树",
          "Strategy Pool",
          "candidate-univariate",
          "evidence-cross_matrix",
          "tree-result-hash",
          "cross-cell-1",
          "leaf-1",
          "pool-revision-2",
          "下载受认证产物",
          "已由服务端截断",
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.equal(html.includes("/private/workspace"), false);
        assert.equal(html.includes('{"'), false);
        assert.equal(html.includes("undefined"), false);
        """
    )


def test_controller_drops_stale_task_response_and_uses_server_blocking_state():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        class ClassList {
          constructor() { this.values = new Set(["hidden"]); }
          toggle(name, force) {
            if (force) this.values.add(name);
            else this.values.delete(name);
          }
        }
        const controls = [{ disabled: false }, { disabled: false }];
        const panel = {
          classList: new ClassList(),
          dataset: {},
          attrs: {},
          setAttribute(name, value) { this.attrs[name] = value; },
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") return [];
            if (selector === "[data-candidate-lab-retry]") return [];
            return controls;
          },
        };
        const results = { innerHTML: "" };
        const status = { textContent: "", dataset: {} };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: results,
          strategyCandidateLabStatus: status,
        };
        const deferred = new Map();
        let currentTask = { id: "task-1", task_type: "strategy" };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => currentTask,
          getSelectedTaskId: () => currentTask?.id || "",
          getStrategyCandidateLab: (taskId) => new Promise((resolve) => {
            deferred.set(taskId, resolve);
          }),
        });

        const taskOneLoad = controller.selectTask(currentTask);
        currentTask = { id: "task-2", task_type: "strategy" };
        const taskTwoLoad = controller.selectTask(currentTask);
        deferred.get("task-1")({
          task_id: "task-1",
          can_start: true,
          candidates: {},
          pools: {},
        });
        await taskOneLoad;
        assert.equal(controller.getState().taskId, "task-2");
        assert.equal(controller.getState().payload, null);

        deferred.get("task-2")({
          task_id: "task-2",
          can_start: false,
          blocked_reason: "open_gate",
          candidates: {},
          pools: {},
        });
        await taskTwoLoad;
        assert.equal(controller.getState().payload.task_id, "task-2");
        assert.equal(panel.dataset.candidateLabBlockedReason, "open_gate");
        assert.ok(controls.every((control) => control.disabled));
        assert.ok(status.textContent.includes("待处理确认门"));

        currentTask = { id: "task-3", task_type: "validation" };
        await controller.selectTask(currentTask);
        assert.equal(controller.getState().taskId, "");
        assert.ok(panel.classList.values.has("hidden"));
        """
    )


def test_failed_typed_submit_preserves_inputs_and_never_requires_llm_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const featureField = { value: "score, age", disabled: false };
        const fields = new Map([
          ["features", featureField],
          ["bin_count", { value: "5", disabled: false }],
          ["min_bin_pct", { value: "0.02", disabled: false }],
          ["loan_amount_col", { value: "", disabled: false }],
          ["overdue_amount_col", { value: "", disabled: false }],
          ["sentinel_values", { value: "", disabled: false }],
          ["manual_breakpoints", { value: "", disabled: false }],
        ]);
        const errorTarget = { textContent: "" };
        const form = {
          dataset: { candidateLabWorkflow: "univariate_candidate_analysis" },
          resetCount: 0,
          reset() { this.resetCount += 1; },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") return errorTarget;
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? fields.get(match[1]) || null : null;
          },
          querySelectorAll(selector) {
            if (selector.includes(":checked")) return [];
            return [];
          },
        };
        const controls = Array.from(fields.values());
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") return [form];
            return controls;
          },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const calls = [];
        const statuses = [];
        let currentTask = { id: "strategy-1", task_type: "strategy" };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => currentTask,
          getSelectedTaskId: () => currentTask.id,
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => ({
            task_id: "strategy-1",
            can_start: true,
            blocked_reason: null,
            candidates: {},
            pools: {},
          }),
          submitStrategyCandidateLabRequest: async (...args) => {
            calls.push(args);
            throw new Error("server rejected request");
          },
          setActionStatus: (...args) => statuses.push(args),
        });
        await controller.selectTask(currentTask);
        // reset on task selection is expected; the operator fills the form after it.
        featureField.value = "score, age";
        const resetCountBeforeSubmit = form.resetCount;
        await controller.submit(form);

        assert.equal(calls.length, 1);
        assert.equal(calls[0][0], "strategy-1");
        assert.equal(calls[0][1].workflow, "univariate_candidate_analysis");
        assert.equal(calls[0][1].workflow_inputs.methods, undefined);
        assert.equal(featureField.value, "score, age");
        assert.equal(form.resetCount, resetCountBeforeSubmit);
        assert.equal(errorTarget.textContent, "server rejected request");
        assert.ok(statuses.some(([message, tone]) => (
          message === "server rejected request" && tone === "error"
        )));
        """
    )


def test_clarification_response_is_not_success_and_keeps_form_for_retry():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const featureField = { value: "score", disabled: false };
        const values = new Map([
          ["features", featureField],
          ["bin_count", { value: "5", disabled: false }],
          ["min_bin_pct", { value: "0.02", disabled: false }],
          ["loan_amount_col", { value: "", disabled: false }],
          ["overdue_amount_col", { value: "", disabled: false }],
          ["sentinel_values", { value: "text:001", disabled: false }],
          ["manual_breakpoints", { value: "", disabled: false }],
        ]);
        const errorTarget = { textContent: "" };
        const form = {
          dataset: { candidateLabWorkflow: "univariate_candidate_analysis" },
          resetCount: 0,
          reset() { this.resetCount += 1; },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") return errorTarget;
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? values.get(match[1]) || null : null;
          },
          querySelectorAll(selector) {
            if (selector.includes(":checked")) return [];
            return [];
          },
        };
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector() { return null; },
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") return [form];
            return Array.from(values.values());
          },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const statuses = [];
        const messagesSeen = [];
        let responseStatus = "clarification_required";
        const payload = {
          task_id: "strategy-1",
          can_start: true,
          blocked_reason: null,
          candidates: {},
          pools: {},
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => ({ id: "strategy-1", task_type: "strategy" }),
          getSelectedTaskId: () => "strategy-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => payload,
          submitStrategyCandidateLabRequest: async () => (
            responseStatus === "clarification_required"
              ? {
                  status: "clarification_required",
                  code: "strategy_workspace_required",
                  messages: [{
                    role: "assistant",
                    content: "请先选择活动数据集并确认目标列。",
                  }],
                }
              : { status: "accepted", messages: [] }
          ),
          setActionStatus: (...args) => statuses.push(args),
          setAgentMessages: (messages) => messagesSeen.push(messages),
        });
        await controller.selectTask({ id: "strategy-1", task_type: "strategy" });
        featureField.value = "score";
        const resetCountBefore = form.resetCount;

        const clarified = await controller.submit(form);
        assert.equal(clarified.status, "clarification_required");
        assert.equal(featureField.value, "score");
        assert.equal(form.resetCount, resetCountBefore);
        assert.equal(errorTarget.textContent, "请先选择活动数据集并确认目标列。");
        assert.ok(statuses.some(([message, tone]) => (
          message === "请先选择活动数据集并确认目标列。" && tone === "info"
        )));
        assert.equal(statuses.some(([, tone]) => tone === "success"), false);
        assert.equal(messagesSeen.length, 1);

        responseStatus = "accepted";
        const accepted = await controller.submit(form);
        assert.equal(accepted.status, "accepted");
        assert.ok(statuses.some(([, tone]) => tone === "success"));
        """
    )


def test_candidate_lab_refresh_is_single_flight_and_aborts_on_task_switch():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        let selectedTask = { id: "strategy-1", task_type: "strategy" };
        const calls = [];
        const pending = new Map();
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector() { return null; },
          querySelectorAll() { return []; },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => selectedTask,
          getSelectedTaskId: () => selectedTask.id,
          getBlockedReason: () => "",
          getStrategyCandidateLab(taskId, { signal } = {}) {
            calls.push({ taskId, signal });
            return new Promise((resolve, reject) => {
              pending.set(taskId, { resolve, reject });
              signal?.addEventListener("abort", () => {
                const error = new Error("aborted");
                error.name = "AbortError";
                reject(error);
              }, { once: true });
            });
          },
        });

        const first = controller.selectTask(selectedTask);
        const duplicate = controller.refresh("strategy-1");
        assert.equal(calls.length, 1);

        selectedTask = { id: "strategy-2", task_type: "strategy" };
        const second = controller.selectTask(selectedTask);
        assert.equal(calls.length, 2);
        assert.equal(calls[0].signal.aborted, true);

        pending.get("strategy-2").resolve({
          task_id: "strategy-2",
          can_start: true,
          blocked_reason: null,
          candidates: {},
          pools: {},
        });
        await Promise.all([first, duplicate, second]);
        assert.equal(controller.getState().taskId, "strategy-2");
        assert.equal(controller.getState().error, "");
        """
    )


def test_candidate_lab_refreshes_after_settle_once_but_never_per_poll_tick():
    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "marvis/static/index.html").read_text(encoding="utf-8")
    workbench_css = (
        ROOT / "marvis/static/css/v2-workbench.css"
    ).read_text(encoding="utf-8")

    assert (
        'from "./js/v2/strategy_candidate_lab_controller.js"' in app_js
    )
    assert "strategyCandidateLabController.selectTask(task)" in app_js
    assert (
        'strategyCandidateLabController.refresh(taskId, { silent: true })'
        in app_js
    )
    poll_body = app_js[
        app_js.index("async function pollValidationProgress(") :
        app_js.index("async function validateCurrentTask(")
    ]
    assert "strategyCandidateLabController.refresh(" not in poll_body
    assert poll_body.count(
        "refreshStrategyCandidateLabAfterSettled(taskId, polledTask)"
    ) == 1
    settled_refresh_body = app_js[
        app_js.index("async function refreshStrategyCandidateLabAfterSettled(") :
        app_js.index("async function pollValidationProgress(")
    ]
    assert settled_refresh_body.count(
        "strategyCandidateLabController.refresh(taskId, { silent: true })"
    ) == 1
    assert "try {" in settled_refresh_body
    assert "catch (_error)" in settled_refresh_body
    assert "strategyCandidateLabController.bind(document)" in app_js
    assert 'id="strategyCandidateLabPanel"' in index_html
    assert 'id="strategyCandidateLabResults"' in index_html
    for workflow in (
        "strategy_project_context",
        "strategy_sample_design_v2",
        "univariate_candidate_analysis",
        "univariate_candidate_refinement",
        "cross_matrix_analysis",
        "cross_matrix_candidate_search",
        "cross_matrix_candidate_build_from_search",
        "automatic_tree_candidate_build",
        "scorecard_band_build",
        "scorecard_cutoff_selection",
        "candidate_monthly_stability",
        "voting_candidate_search",
        "voting_candidate_build_from_search",
        "strategy_pool_impact",
        "strategy_impact_cube",
        "strategy_pool_materialize",
        "strategy_lifecycle_adopt",
        "strategy_dsl_delivery",
        "strategy_report_bundle_v2",
    ):
        assert f'data-candidate-lab-workflow="{workflow}"' in index_html
    refinement_start = index_html.index(
        'data-candidate-lab-workflow="univariate_candidate_refinement"'
    )
    refinement_end = index_html.index("</form>", refinement_start)
    refinement_html = index_html[refinement_start:refinement_end]
    assert 'data-candidate-lab-field="source_candidate_id"' in refinement_html
    assert '<select data-candidate-lab-field="source_candidate_id">' in refinement_html
    for forbidden in ("artifact_id", "content_hash", "evidence_hash"):
        assert forbidden not in refinement_html
    scorecard_selection_start = index_html.index(
        'data-candidate-lab-workflow="scorecard_cutoff_selection"'
    )
    scorecard_selection_end = index_html.index("</form>", scorecard_selection_start)
    scorecard_selection_html = index_html[
        scorecard_selection_start:scorecard_selection_end
    ]
    assert 'data-candidate-lab-field="scorecard_asset_id"' in (
        scorecard_selection_html
    )
    assert 'data-candidate-lab-field="scorecard_cutoff_id"' in (
        scorecard_selection_html
    )
    for forbidden in (
        "artifact_id",
        "asset_hash",
        "content_hash",
        "score_evidence_ref",
        "sample_design_ref",
    ):
        assert forbidden not in scorecard_selection_html
    stability_start = index_html.index(
        'data-candidate-lab-workflow="candidate_monthly_stability"'
    )
    stability_end = index_html.index("</form>", stability_start)
    stability_html = index_html[stability_start:stability_end]
    assert 'data-candidate-lab-field="stability_pool_entry"' in stability_html
    assert 'data-candidate-lab-field="stability_asset_id"' in stability_html
    for forbidden in (
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "dataset_id",
        "sample_design_ref",
        "target_col",
        "month_col",
    ):
        assert forbidden not in stability_html
    voting_search_start = index_html.index(
        'data-candidate-lab-workflow="voting_candidate_search"'
    )
    voting_search_end = index_html.index("</form>", voting_search_start)
    voting_search_html = index_html[voting_search_start:voting_search_end]
    for field in (
        "voting_strategy_type",
        "voting_member_count",
        "voting_n",
        "voting_objective_metric",
        "voting_objective_direction",
        "voting_constraints",
        "voting_include_rule_ids",
        "voting_exclude_rule_ids",
        "voting_max_combinations",
    ):
        assert f'data-candidate-lab-field="{field}"' in voting_search_html
    voting_build_start = index_html.index(
        'data-candidate-lab-workflow="voting_candidate_build_from_search"'
    )
    voting_build_end = index_html.index("</form>", voting_build_start)
    voting_build_html = index_html[voting_build_start:voting_build_end]
    assert 'data-candidate-lab-field="voting_search_id"' in voting_build_html
    assert 'data-candidate-lab-field="voting_combo_id"' in voting_build_html
    for forbidden in (
        "pool_ref",
        "artifact_id",
        "content_hash",
        "dataset_id",
        "target_col",
        "hit_matrix",
        "score_vector",
        "objective_value",
        "rank",
        "best",
        "champion",
    ):
        assert forbidden not in voting_search_html
        assert forbidden not in voting_build_html
    assert "原始 PD 越高表示风险越高" in index_html
    assert "评分卡分数越高表示更安全" in index_html
    assert "不等于通过或拒绝动作" in index_html
    assert "不会自动进入 Strategy Pool" in index_html
    assert "最佳 Cutoff" not in index_html
    assert "不会自动构建、选择、入池或部署" in index_html
    cross_search_start = index_html.index(
        'data-candidate-lab-workflow="cross_matrix_candidate_search"'
    )
    cross_search_end = index_html.index("</form>", cross_search_start)
    cross_search_html = index_html[cross_search_start:cross_search_end]
    assert 'data-candidate-lab-field="cross_search_features"' in (
        cross_search_html
    )
    assert 'data-candidate-lab-field="cross_search_max_pairs"' in (
        cross_search_html
    )
    cross_build_start = index_html.index(
        'data-candidate-lab-workflow="cross_matrix_candidate_build_from_search"'
    )
    cross_build_end = index_html.index("</form>", cross_build_start)
    cross_build_html = index_html[cross_build_start:cross_build_end]
    assert 'data-candidate-lab-field="cross_build_search_id"' in (
        cross_build_html
    )
    assert 'data-candidate-lab-field="cross_build_pair_id"' in (
        cross_build_html
    )
    for forbidden in (
        "method",
        "candidate_id",
        "artifact_id",
        "content_hash",
        "eligible",
        "rank",
        "interaction_gain_iv",
    ):
        assert f'data-candidate-lab-field="{forbidden}"' not in cross_search_html
        assert f'data-candidate-lab-field="{forbidden}"' not in cross_build_html
    assert "不会自动构建 Cross Matrix、加入 Pool、采纳或部署" in (
        cross_search_html
    )
    assert "不会自动入 Pool、采纳或部署" in cross_build_html
    interactive_tree_start = index_html.index(
        'data-candidate-lab-workflow="interactive_tree_revision"'
    )
    interactive_tree_end = index_html.index("</form>", interactive_tree_start)
    interactive_tree_html = index_html[
        interactive_tree_start:interactive_tree_end
    ]
    for field in (
        "interactive_tree_operation",
        "interactive_tree_source_id",
        "interactive_tree_node_id",
        "interactive_tree_threshold",
        "interactive_tree_reason",
    ):
        assert f'data-candidate-lab-field="{field}"' in interactive_tree_html
    for forbidden in (
        "feature",
        "current_threshold",
        "asset_hash",
        "metrics",
        "base_threshold",
    ):
        assert f'data-candidate-lab-field="{forbidden}"' not in (
            interactive_tree_html
        )
    assert "不会写回来源树、物化 frontier、入池、采纳或部署" in (
        interactive_tree_html
    )
    adoption_start = index_html.index(
        'data-candidate-lab-workflow="strategy_lifecycle_adopt"'
    )
    adoption_end = index_html.index("</form>", adoption_start)
    adoption_html = index_html[adoption_start:adoption_end]
    for field in (
        "lifecycle_adopt_strategy_id",
        "lifecycle_adoption_reason",
        "lifecycle_adopt_pd_mode",
        "lifecycle_adopt_pd_column",
        "lifecycle_adopt_pd_value",
        "lifecycle_adopt_lgd_mode",
        "lifecycle_adopt_lgd_column",
        "lifecycle_adopt_lgd_value",
        "lifecycle_adopt_utilization_mode",
        "lifecycle_adopt_ead_mode",
        "lifecycle_adopt_funding_rate_mode",
        "lifecycle_adopt_term_months_mode",
        "lifecycle_adopt_operating_cost_per_loan_mode",
    ):
        assert f'data-candidate-lab-field="{field}"' in adoption_html
    for forbidden in (
        "asset_status",
        "status",
        "content_hash",
        "validation_metrics",
        "deployment_status",
    ):
        assert f'data-candidate-lab-field="{forbidden}"' not in adoption_html
    assert "重新回测并等待人工确认" in adoption_html
    assert "本地采纳不是生产部署" in adoption_html
    assert "text:001" in index_html
    assert "number:-9999" in index_html
    assert ".candidate-lab-layout" in workbench_css
    assert (
        "grid-template-columns: minmax(260px, 0.72fr) minmax(0, 1.55fr)"
        in workbench_css
    )
    assert "@media" not in workbench_css[
        workbench_css.index("/* Candidate Lab") :
        workbench_css.index("/* Modeling setup gate controls.")
    ]


def test_local_strategy_adoption_collector_emits_only_governed_user_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function strategyOption(strategyId, strategyType, assetStatus = "draft") {
          return {
            value: strategyId,
            dataset: {
              candidateLabProjection: "1",
              strategyId,
              strategyType,
              assetStatus,
            },
          };
        }

        function makeForm(strategyType, fields = {}, optionStatus = "draft") {
          const strategyId = `strategy-${strategyType}-v2`;
          const option = strategyOption(strategyId, strategyType, optionStatus);
          const values = {
            lifecycle_adopt_strategy_id: {
              value: strategyId,
              selectedOptions: [option],
            },
            lifecycle_adoption_reason: {
              value: "已复核回测、影响测算与报告证据，同意进入本地采纳确认",
            },
            ...fields,
          };
          return {
            dataset: { candidateLabWorkflow: "strategy_lifecycle_adopt" },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? (values[match[1]] || null) : null;
            },
            querySelectorAll() { return []; },
          };
        }

        const limit = collectStrategyCandidateLabRequest(makeForm("limit", {
          lifecycle_adopt_pd_mode: { value: "column" },
          lifecycle_adopt_pd_column: { value: "pd_12m" },
          lifecycle_adopt_pd_value: { value: "0.01" },
          lifecycle_adopt_lgd_mode: { value: "value" },
          lifecycle_adopt_lgd_column: { value: "forged_lgd" },
          lifecycle_adopt_lgd_value: { value: "0.45" },
          lifecycle_adopt_utilization_mode: { value: "column" },
          lifecycle_adopt_utilization_column: { value: "utilization" },
          lifecycle_adopt_utilization_value: { value: "0.8" },
        }));
        assert.deepEqual(limit, {
          request_kind: "strategy_lifecycle",
          operation: "adopt",
          strategy_type: "limit",
          strategy_id: "strategy-limit-v2",
          adoption_reason: "已复核回测、影响测算与报告证据，同意进入本地采纳确认",
          economics_inputs: {
            pd_col: "pd_12m",
            lgd_value: 0.45,
            utilization_col: "utilization",
          },
        });

        const pricing = collectStrategyCandidateLabRequest(makeForm("pricing", {
          lifecycle_adopt_ead_mode: { value: "column" },
          lifecycle_adopt_ead_column: { value: "ead" },
          lifecycle_adopt_pd_mode: { value: "column" },
          lifecycle_adopt_pd_column: { value: "pd_12m" },
          lifecycle_adopt_lgd_mode: { value: "value" },
          lifecycle_adopt_lgd_value: { value: "0.5" },
          lifecycle_adopt_funding_rate_mode: { value: "value" },
          lifecycle_adopt_funding_rate_value: { value: "0.04" },
          lifecycle_adopt_term_months_mode: { value: "value" },
          lifecycle_adopt_term_months_value: { value: "12" },
          lifecycle_adopt_operating_cost_per_loan_mode: { value: "value" },
          lifecycle_adopt_operating_cost_per_loan_value: { value: "12" },
        }));
        assert.deepEqual(pricing.economics_inputs, {
          ead_col: "ead",
          pd_col: "pd_12m",
          lgd_value: 0.5,
          funding_rate_value: 0.04,
          term_months_value: 12,
          operating_cost_per_loan_value: 12,
        });

        for (const strategyType of ["approval", "reject", "segmentation"]) {
          const request = collectStrategyCandidateLabRequest(makeForm(
            strategyType,
            {
              lifecycle_adopt_pd_mode: { value: "value" },
              lifecycle_adopt_pd_value: { value: "0.01" },
            },
          ));
          assert.equal("economics_inputs" in request, false);
          for (const forbidden of [
            "asset_status",
            "status",
            "content_hash",
            "validation_metrics",
            "deployment_status",
          ]) {
            assert.equal(forbidden in request, false);
          }
        }

        assert.throws(
          () => collectStrategyCandidateLabRequest(
            makeForm("approval", {}, "adopted_local"),
          ),
          /draft|草稿|受认证投影/,
        );
        """
    )


def test_strategy_history_renders_local_champions_blockers_and_safe_artifacts():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const html = strategyCandidateLabResultsHtml({
          strategies: {
            latest: null,
            total: 2,
            truncated: false,
            current_local_champions: [{
              strategy_id: "strategy-approval-v1",
              strategy_type: "approval",
              version: 1,
            }],
            all: [
              {
                strategy_id: "strategy-limit-v2",
                strategy_type: "limit",
                version: 2,
                status: "draft",
                asset_status: "draft",
                created_at: "2026-07-27T02:00:00+00:00",
                adopted_at: null,
                parent_strategy_id: "strategy-limit-v1",
                rule_count: 3,
                strategy_spec_hash:
                  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                materialization: {
                  materialization_id: "materialization-limit-v2",
                  pool_id: "strategy-pool-limit",
                  pool_revision_id: "pool-revision-limit-4",
                  pool_revision: 4,
                  requirements_count: 2,
                  runtime_blockers: [
                    "缺少运行列 income",
                    "<img src=x onerror=alert(1)>",
                  ],
                },
                artifacts: {
                  all: [{
                    artifact_id: "strategy-artifact-limit-json",
                    kind: "strategy_json",
                    filename: "limit-strategy.json",
                    created_at: "2026-07-27T02:01:00+00:00",
                    content_size: 1200,
                    download_url:
                      "/api/tasks/task-1/strategy-artifacts/strategy-artifact-limit-json/download",
                  }, {
                    artifact_id: "strategy-artifact-forged",
                    kind: "strategy_sql",
                    filename: "forged.sql",
                    download_url: "javascript:alert(1)",
                  }],
                  total: 2,
                  truncated: false,
                },
              },
              {
                strategy_id: "strategy-approval-v1",
                strategy_type: "approval",
                version: 1,
                status: "adopted",
                asset_status: "adopted_local",
                created_at: "2026-07-27T01:00:00+00:00",
                adopted_at: "2026-07-27T01:30:00+00:00",
                parent_strategy_id: null,
                rule_count: 4,
                strategy_spec_hash:
                  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                materialization: null,
                artifacts: { all: [], total: 0, truncated: false },
              },
            ],
          },
          workflow: {},
          candidates: {},
          pools: {},
        });

        for (const expected of [
          "策略版本历史",
          "当前本地策略",
          "strategy-limit-v2",
          "strategy-approval-v1",
          "额度策略",
          "审批策略",
          "draft",
          "本地已采纳",
          "materialization-limit-v2",
          "pool-revision-limit-4",
          "运行阻塞",
          "缺少运行列 income",
          "下载 limit-strategy.json",
          "本地采纳",
          "不是生产部署",
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.ok(
          html.includes(
            "/api/tasks/task-1/strategy-artifacts/strategy-artifact-limit-json/download",
          ),
        );
        assert.equal(html.includes("javascript:alert"), false);
        assert.equal(html.includes("<img src=x"), false);
        assert.equal(
          html.includes(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          ),
          false,
        );
        assert.equal(
          html.includes(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          ),
          false,
        );
        """
    )


def test_strategy_workbench_accepts_only_known_canonical_hex_strategy_ids():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const strategyId = "0123456789abcdef0123456789abcdef";
        const option = {
          value: strategyId,
          dataset: {
            candidateLabProjection: "1",
            strategyId,
            strategyType: "approval",
            assetStatus: "draft",
          },
        };
        const fields = {
          lifecycle_adopt_strategy_id: {
            value: strategyId,
            selectedOptions: [option],
          },
          lifecycle_adoption_reason: {
            value: "已复核回测和风险影响，同意提交本地采纳确认",
          },
        };
        const form = (workflow) => ({
          dataset: { candidateLabWorkflow: workflow },
          querySelector(selector) {
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            if (!match) return null;
            if (match[1] === "dsl_delivery_strategy_id") {
              return {
                value: strategyId,
                selectedOptions: [{
                  value: strategyId,
                  dataset: {
                    candidateLabProjection: "1",
                    strategyId,
                  },
                }],
              };
            }
            return fields[match[1]] || null;
          },
          querySelectorAll() { return []; },
        });

        assert.equal(
          collectStrategyCandidateLabRequest(
            form("strategy_lifecycle_adopt"),
          ).strategy_id,
          strategyId,
        );
        assert.equal(
          collectStrategyCandidateLabRequest(
            form("strategy_dsl_delivery"),
          ).workflow_inputs.strategy_id,
          strategyId,
        );
        const html = strategyCandidateLabResultsHtml({
          strategies: {
            latest: null,
            all: [{
              strategy_id: strategyId,
              strategy_type: "approval",
              version: 3,
              status: "draft",
              asset_status: "draft",
              artifacts: { all: [], total: 0, truncated: false },
            }],
            total: 1,
            truncated: false,
            current_local_champions: [],
          },
          workflow: {},
          candidates: {},
          pools: {},
        });
        assert.ok(html.includes(strategyId));

        const forged = {
          ...option,
          value: "arbitrary strategy id",
          dataset: {
            ...option.dataset,
            strategyId: "arbitrary strategy id",
          },
        };
        fields.lifecycle_adopt_strategy_id = {
          value: forged.value,
          selectedOptions: [forged],
        };
        assert.throws(
          () => collectStrategyCandidateLabRequest(
            form("strategy_lifecycle_adopt"),
          ),
          /draft|草稿|受认证投影/,
        );
        """
    )


def test_local_strategy_adoption_uses_candidate_lab_submit_settle_and_refresh():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const strategyOption = {
          value: "strategy-approval-v2",
          dataset: {
            candidateLabProjection: "1",
            strategyId: "strategy-approval-v2",
            strategyType: "approval",
            assetStatus: "draft",
          },
        };
        const fields = new Map([
          ["lifecycle_adopt_strategy_id", {
            value: "strategy-approval-v2",
            selectedOptions: [strategyOption],
          }],
          ["lifecycle_adoption_reason", {
            value: "已复核回测和风险影响，同意提交本地采纳人工确认",
          }],
        ]);
        const errorTarget = { textContent: "" };
        const form = {
          dataset: { candidateLabWorkflow: "strategy_lifecycle_adopt" },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") {
              return errorTarget;
            }
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? (fields.get(match[1]) || null) : null;
          },
          querySelectorAll() { return []; },
        };
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector() { return null; },
          querySelectorAll() { return []; },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const payload = {
          task_id: "task-1",
          can_start: true,
          blocked_reason: null,
          workflow: {},
          candidates: {},
          pools: {},
          strategies: {
            latest: null,
            all: [{
              strategy_id: "strategy-approval-v2",
              strategy_type: "approval",
              version: 2,
              status: "draft",
              asset_status: "draft",
              materialization: { runtime_blockers: [] },
            }],
            total: 1,
            truncated: false,
            current_local_champions: [],
          },
        };
        const submits = [];
        const polls = [];
        const settles = [];
        let refreshes = 0;
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => ({ id: "task-1", task_type: "strategy" }),
          getSelectedTaskId: () => "task-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => {
            refreshes += 1;
            return payload;
          },
          submitStrategyCandidateLabRequest: async (...args) => {
            submits.push(args);
            return { status: "accepted", messages: [] };
          },
          pollAgentMessagesUntilSettled: (...args) => {
            polls.push(args);
          },
          settleCandidateLabSubmission: async (...args) => {
            settles.push(args);
          },
        });

        await controller.selectTask({ id: "task-1", task_type: "strategy" });
        const result = await controller.submit(form);

        assert.equal(result.status, "accepted");
        assert.equal(submits.length, 1);
        assert.deepEqual(submits[0], [
          "task-1",
          {
            request_kind: "strategy_lifecycle",
            operation: "adopt",
            strategy_type: "approval",
            strategy_id: "strategy-approval-v2",
            adoption_reason: "已复核回测和风险影响，同意提交本地采纳人工确认",
          },
          "提交策略本地采纳确认",
        ]);
        assert.equal(polls.length, 1);
        assert.equal(settles.length, 1);
        assert.equal(refreshes, 2);
        assert.equal(controller.getState().submitting, false);
        assert.equal(errorTarget.textContent, "");
        """
    )


def test_cross_auto_search_collectors_emit_only_explicit_governed_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function makeForm(workflow, fields) {
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? (fields[match[1]] || null) : null;
            },
            querySelectorAll() { return []; },
          };
        }

        const featureOption = (feature, projection = "1") => ({
          value: feature,
          dataset: {
            candidateLabProjection: projection,
            feature,
            method: "forged-method-must-not-submit",
            candidateId: "forged-candidate-must-not-submit",
            contentHash: "forged-hash-must-not-submit",
          },
        });
        const search = collectStrategyCandidateLabRequest(makeForm(
          "cross_matrix_candidate_search",
          {
            cross_search_features: {
              selectedOptions: [
                featureOption("age"),
                featureOption("score"),
                featureOption("income"),
              ],
            },
            cross_search_max_pairs: { value: "12" },
            artifact_id: { value: "forged-artifact" },
          },
        ));
        assert.deepEqual(search, {
          request_kind: "standard_workflow",
          workflow: "cross_matrix_candidate_search",
          workflow_inputs: {
            features: ["age", "score", "income"],
            max_pairs: 12,
          },
        });
        for (const forbidden of [
          "method",
          "methods",
          "candidate_id",
          "artifact_id",
          "content_hash",
        ]) {
          assert.equal(forbidden in search.workflow_inputs, false);
        }

        const searchId = "cross-search-0123456789abcdef0123456789abcdef";
        const pairId = "cross-pair-fedcba9876543210fedcba9876543210";
        const build = collectStrategyCandidateLabRequest(makeForm(
          "cross_matrix_candidate_build_from_search",
          {
            cross_build_search_id: {
              value: searchId,
              selectedOptions: [{
                value: searchId,
                dataset: {
                  candidateLabProjection: "1",
                  searchId,
                  eligible: "false",
                  rank: "7",
                },
              }],
            },
            cross_build_pair_id: {
              value: pairId,
              selectedOptions: [{
                value: pairId,
                dataset: {
                  candidateLabProjection: "1",
                  searchId,
                  pairId,
                  eligible: "false",
                  rank: "7",
                  interactionGainIv: "0.31",
                },
              }],
            },
          },
        ));
        assert.deepEqual(build, {
          request_kind: "standard_workflow",
          workflow: "cross_matrix_candidate_build_from_search",
          workflow_inputs: { search_id: searchId, pair_id: pairId },
        });

        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm(
            "cross_matrix_candidate_search",
            {
              cross_search_features: {
                selectedOptions: [
                  featureOption("age"),
                  featureOption("age"),
                ],
              },
              cross_search_max_pairs: { value: "1" },
            },
          )),
          /重复|独立/,
        );
        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm(
            "cross_matrix_candidate_search",
            {
              cross_search_features: {
                selectedOptions: [
                  featureOption("age"),
                  featureOption("score", "0"),
                ],
              },
              cross_search_max_pairs: { value: "191" },
            },
          )),
          /受认证投影|1 到 190/,
        );
        """
    )


def test_cross_auto_search_renders_aggregate_pairs_budget_and_safe_download():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const html = strategyCandidateLabResultsHtml({
          workflow: {},
          pools: {},
          strategies: {},
          candidates: {
            cross_search: {
              latest: null,
              total: 1,
              truncated: false,
              all: [{
                search_id:
                  "cross-search-0123456789abcdef0123456789abcdef",
                features: [{
                  feature: "age",
                  method: "equal_frequency",
                  axis_iv: 0.08,
                  bin_count: 5,
                }, {
                  feature: "score",
                  method: "tree",
                  axis_iv: 0.24,
                  bin_count: 6,
                }, {
                  feature: "income",
                  method: "chimerge",
                  axis_iv: 0.12,
                  bin_count: 4,
                }],
                max_pairs: 20,
                search_space: 190,
                evaluated: 20,
                eligible: 7,
                truncated: true,
                pairs: [{
                  pair_id:
                    "cross-pair-0123456789abcdef0123456789abcdef",
                  x_feature: "age",
                  x_method: "equal_frequency",
                  y_feature: "score",
                  y_method: "tree",
                  x_axis_iv: 0.08,
                  y_axis_iv: 0.24,
                  cross_total_iv: 0.51,
                  interaction_gain_iv: 0.19,
                  cell_count: 30,
                  empty_cell_count: 3,
                  empty_cell_share: 0.1,
                  min_nonempty_cell_count: 42,
                  eligible: true,
                  rank: 1,
                }, {
                  pair_id:
                    "cross-pair-fedcba9876543210fedcba9876543210",
                  x_feature: "age",
                  x_method: "equal_frequency",
                  y_feature: "<img src=x onerror=alert(1)>",
                  y_method: "chimerge",
                  x_axis_iv: 0.08,
                  y_axis_iv: 0.12,
                  cross_total_iv: 0.2,
                  interaction_gain_iv: 0,
                  cell_count: 20,
                  empty_cell_count: 12,
                  empty_cell_share: 0.6,
                  min_nonempty_cell_count: 2,
                  eligible: false,
                  rank: 2,
                }],
                artifact: {
                  artifact_id: "artifact-cross-search-1",
                  created_at: "2026-07-27T03:00:00+00:00",
                  download_url:
                    "/api/tasks/task-1/task-artifacts/artifact-cross-search-1/download",
                },
              }],
            },
          },
        });

        for (const expected of [
          "Cross 自动搜索",
          "cross-search-0123456789abcdef0123456789abcdef",
          "搜索参数与预算",
          "190",
          "20",
          "7",
          "Top Pairs",
          "cross-pair-0123456789abcdef0123456789abcdef",
          "Interaction Gain IV",
          "空单元格占比",
          "0.6",
          "预算截断",
          "不会自动构建、入池、采纳或部署",
          "下载受认证产物",
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.ok(html.includes(
          "/api/tasks/task-1/task-artifacts/artifact-cross-search-1/download",
        ));
        assert.equal(html.includes("<img src=x"), false);
        assert.equal(html.includes("最佳"), false);
        assert.equal(html.includes("冠军"), false);
        """
    )


def test_cross_auto_search_submits_only_current_projection_and_refreshes():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const searchId = "cross-search-0123456789abcdef0123456789abcdef";
        const pairId = "cross-pair-0123456789abcdef0123456789abcdef";
        const featureOption = (feature) => ({
          value: feature,
          dataset: { candidateLabProjection: "1", feature },
        });
        const projectedOption = (value, dataset) => ({
          value,
          dataset: { candidateLabProjection: "1", ...dataset },
        });
        const errorTarget = { textContent: "" };
        const form = (workflow, fields) => ({
          dataset: { candidateLabWorkflow: workflow },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") {
              return errorTarget;
            }
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? (fields[match[1]] || null) : null;
          },
          querySelectorAll() { return []; },
        });
        const searchForm = form("cross_matrix_candidate_search", {
          cross_search_features: {
            selectedOptions: [featureOption("age"), featureOption("score")],
          },
          cross_search_max_pairs: { value: "1" },
        });
        const buildForm = (selectedPairId) => form(
          "cross_matrix_candidate_build_from_search",
          {
            cross_build_search_id: {
              value: searchId,
              selectedOptions: [projectedOption(
                searchId,
                { searchId },
              )],
            },
            cross_build_pair_id: {
              value: selectedPairId,
              selectedOptions: [projectedOption(
                selectedPairId,
                { searchId, pairId: selectedPairId },
              )],
            },
          },
        );

        const payload = {
          task_id: "task-1",
          can_start: true,
          blocked_reason: null,
          workflow: {},
          pools: {},
          strategies: {},
          candidates: {
            univariate: {
              latest: null,
              total: 1,
              truncated: false,
              all: [{
                candidate_id: "candidate-source",
                artifact: { artifact_id: "artifact-source" },
                pointers: {
                  bins: [
                    { feature: "age", method: "equal_frequency" },
                    { feature: "score", method: "tree" },
                  ],
                },
              }],
            },
            cross_search: {
              latest: null,
              total: 1,
              truncated: false,
              all: [{
                search_id: searchId,
                evaluated: 1,
                search_space: 1,
                eligible: 1,
                artifact: { artifact_id: "artifact-cross-search" },
                pairs: [{
                  pair_id: pairId,
                  x_feature: "age",
                  x_method: "equal_frequency",
                  y_feature: "score",
                  y_method: "tree",
                  eligible: true,
                  rank: 1,
                }],
              }],
            },
          },
        };
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector() { return null; },
          querySelectorAll() { return []; },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const submits = [];
        const polls = [];
        const settles = [];
        let refreshes = 0;
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => ({ id: "task-1", task_type: "strategy" }),
          getSelectedTaskId: () => "task-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => {
            refreshes += 1;
            return payload;
          },
          submitStrategyCandidateLabRequest: async (...args) => {
            submits.push(args);
            return { status: "accepted", messages: [] };
          },
          pollAgentMessagesUntilSettled: (...args) => {
            polls.push(args);
          },
          settleCandidateLabSubmission: async (...args) => {
            settles.push(args);
          },
        });

        await controller.selectTask({ id: "task-1", task_type: "strategy" });
        await controller.submit(searchForm);
        await controller.submit(buildForm(pairId));

        assert.equal(submits.length, 2);
        assert.deepEqual(submits.map((call) => call[1]), [{
          request_kind: "standard_workflow",
          workflow: "cross_matrix_candidate_search",
          workflow_inputs: { features: ["age", "score"], max_pairs: 1 },
        }, {
          request_kind: "standard_workflow",
          workflow: "cross_matrix_candidate_build_from_search",
          workflow_inputs: { search_id: searchId, pair_id: pairId },
        }]);
        assert.equal(polls.length, 2);
        assert.equal(settles.length, 2);
        assert.equal(refreshes, 3);

        const stalePairId =
          "cross-pair-fedcba9876543210fedcba9876543210";
        const stale = await controller.submit(buildForm(stalePairId));
        assert.equal(stale, null);
        assert.equal(submits.length, 2);
        assert.match(errorTarget.textContent, /已过期|受认证投影/);
        """
    )


def test_cross_build_projection_never_auto_selects_single_search_or_pair():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        class FakeSelect {
          constructor(fieldName) {
            this.dataset = { candidateLabField: fieldName };
            this.options = [];
            this.selectedOptions = [];
            this._value = "";
            this._html = "";
            this.form = null;
          }
          set innerHTML(value) {
            this._html = value;
            this.options = [...value.matchAll(/<option value="([^"]*)"/g)]
              .map((match) => ({ value: match[1], selected: false }));
            this.value = "";
          }
          get innerHTML() { return this._html; }
          set value(value) {
            this._value = String(value);
            for (const option of this.options) {
              option.selected = option.value === this._value;
            }
            this.selectedOptions = this.options.filter(
              (option) => option.selected,
            );
          }
          get value() { return this._value; }
          closest(selector) {
            if (selector === "[data-candidate-lab-field]") return this;
            return selector.includes(
              'cross_matrix_candidate_build_from_search',
            )
              ? this.form
              : null;
          }
        }

        const searchId = "cross-search-0123456789abcdef0123456789abcdef";
        const pairId = "cross-pair-0123456789abcdef0123456789abcdef";
        const searchSelect = new FakeSelect("cross_build_search_id");
        const pairSelect = new FakeSelect("cross_build_pair_id");
        const help = { textContent: "" };
        const buildForm = {
          querySelector(selector) {
            if (selector.includes('cross_build_search_id')) return searchSelect;
            if (selector.includes('cross_build_pair_id')) return pairSelect;
            if (selector === "[data-candidate-lab-cross-build-help]") {
              return help;
            }
            return null;
          },
          querySelectorAll() { return []; },
        };
        searchSelect.form = buildForm;
        pairSelect.form = buildForm;
        const featureSelect = new FakeSelect("cross_search_features");
        const searchHelp = { textContent: "" };
        const searchForm = {
          querySelector(selector) {
            if (selector.includes('cross_search_features')) {
              return featureSelect;
            }
            if (selector === "[data-candidate-lab-cross-search-help]") {
              return searchHelp;
            }
            return null;
          },
          querySelectorAll() { return []; },
        };
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector(selector) {
            if (selector.includes(
              'cross_matrix_candidate_build_from_search',
            )) {
              return buildForm;
            }
            if (selector.includes('cross_matrix_candidate_search')) {
              return searchForm;
            }
            return null;
          },
          querySelectorAll(selector) {
            return selector === "[data-candidate-lab-form]"
              ? [searchForm, buildForm]
              : [];
          },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const payload = {
          task_id: "task-1",
          can_start: true,
          blocked_reason: null,
          workflow: {},
          pools: {},
          strategies: {},
          candidates: {
            univariate: {
              latest: null,
              total: 1,
              truncated: false,
              all: [{
                candidate_id: "candidate-source",
                artifact: { artifact_id: "artifact-source" },
                pointers: {
                  bins: [
                    { feature: "age", method: "equal_frequency" },
                    { feature: "score", method: "tree" },
                  ],
                },
              }],
            },
            cross_search: {
              latest: null,
              total: 1,
              truncated: false,
              all: [{
                search_id: searchId,
                evaluated: 1,
                search_space: 1,
                eligible: 1,
                artifact: { artifact_id: "artifact-search" },
                pairs: [{
                  pair_id: pairId,
                  x_feature: "age",
                  x_method: "equal_frequency",
                  y_feature: "score",
                  y_method: "tree",
                  cell_count: 20,
                  empty_cell_count: 2,
                  empty_cell_share: 0.1,
                  eligible: true,
                  rank: 1,
                }],
              }],
            },
          },
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => ({ id: "task-1", task_type: "strategy" }),
          getSelectedTaskId: () => "task-1",
          getStrategyCandidateLab: async () => payload,
        });

        await controller.selectTask({ id: "task-1", task_type: "strategy" });
        assert.ok(searchSelect.innerHTML.includes(searchId));
        assert.equal(searchSelect.value, "");
        assert.equal(pairSelect.innerHTML.includes(pairId), false);

        searchSelect.value = searchId;
        controller.handleChange({ target: searchSelect });
        assert.ok(pairSelect.innerHTML.includes(pairId));
        assert.equal(pairSelect.value, "");
        assert.match(help.textContent, /不会自动代选/);
        assert.ok(featureSelect.innerHTML.includes("age"));
        assert.ok(featureSelect.innerHTML.includes("score"));
        assert.equal(featureSelect.selectedOptions.length, 0);
        """
    )


def test_interactive_tree_threshold_collector_is_pointer_only_and_keeps_prune():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const sourceTreeId =
          "candidate-asset-0123456789abcdef0123456789abcdef";
        const nodeId = "node-0123456789abcdef0123";
        const sourceOption = {
          value: sourceTreeId,
          dataset: {
            candidateLabProjection: "1",
            sourceTreeId,
          },
        };
        const nodeOption = (operation, currentThreshold = "600") => ({
          value: nodeId,
          dataset: {
            candidateLabProjection: "1",
            sourceTreeId,
            nodeId,
            operation,
            feature: "score",
            currentThreshold,
            assetHash: "forged-hash-must-not-submit",
            metrics: "forged-metrics-must-not-submit",
          },
        });
        function form(operation, threshold, option = nodeOption(operation)) {
          const fields = {
            interactive_tree_operation: { value: operation },
            interactive_tree_source_id: {
              value: sourceTreeId,
              selectedOptions: [sourceOption],
            },
            interactive_tree_node_id: {
              value: nodeId,
              selectedOptions: [option],
            },
            interactive_tree_threshold: { value: threshold },
            interactive_tree_reason: {
              value: "业务希望把 score 分界从 600 调整到 575.5",
            },
          };
          return {
            dataset: { candidateLabWorkflow: "interactive_tree_revision" },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? (fields[match[1]] || null) : null;
            },
            querySelectorAll() { return []; },
          };
        }

        const adjustment = collectStrategyCandidateLabRequest(
          form("adjust_split_threshold", "575.5"),
        );
        assert.deepEqual(adjustment, {
          request_kind: "standard_workflow",
          workflow: "interactive_tree_revision",
          workflow_inputs: {
            source_tree_id: sourceTreeId,
            node_id: nodeId,
            operation: "adjust_split_threshold",
            threshold: 575.5,
            reason: "业务希望把 score 分界从 600 调整到 575.5",
          },
        });
        for (const forbidden of [
          "feature",
          "current_threshold",
          "asset_hash",
          "metrics",
          "threshold_delta",
        ]) {
          assert.equal(forbidden in adjustment.workflow_inputs, false);
        }

        const prune = collectStrategyCandidateLabRequest(
          form("prune_subtree", "999", nodeOption("prune_subtree")),
        );
        assert.deepEqual(prune.workflow_inputs, {
          source_tree_id: sourceTreeId,
          node_id: nodeId,
          operation: "prune_subtree",
          reason: "业务希望把 score 分界从 600 调整到 575.5",
        });

        assert.throws(
          () => collectStrategyCandidateLabRequest(
            form("adjust_split_threshold", "600"),
          ),
          /不同|当前阈值/,
        );
        assert.throws(
          () => collectStrategyCandidateLabRequest(
            form("adjust_split_threshold", "Infinity"),
          ),
          /有限|finite/,
        );
        assert.throws(
          () => collectStrategyCandidateLabRequest(
            form(
              "adjust_split_threshold",
              "575.5",
              nodeOption("prune_subtree"),
            ),
          ),
          /受认证|阈值调整/,
        );
        """
    )


def test_interactive_tree_render_exposes_authenticated_threshold_adjustments():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const sourceTreeId =
          "candidate-asset-0123456789abcdef0123456789abcdef";
        const nodeId = "node-0123456789abcdef0123";
        const tree = {
          kind: "automatic_tree",
          candidate_id: "candidate-tree",
          detail: {
            source_tree_id: sourceTreeId,
            asset_id: sourceTreeId,
            summary: { node_count: 3 },
          },
          pointers: {
            nodes: [{
              node_id: nodeId,
              kind: "split",
              depth: 0,
              feature: "score",
              threshold: 600,
              missing_child: "left",
              is_visible: true,
              is_frontier: false,
              can_prune: true,
              metrics: { count: 1000, bad_rate: 0.12 },
            }],
            eligible_prunes: [{
              source_tree_id: sourceTreeId,
              node_id: nodeId,
              operation: "prune_subtree",
            }],
            eligible_threshold_adjustments: [{
              source_tree_id: sourceTreeId,
              node_id: nodeId,
              operation: "adjust_split_threshold",
              feature: "score",
              current_threshold: 600,
            }],
            leaves: [],
          },
          artifact: {
            artifact_id: "artifact-tree",
            download_url:
              "/api/tasks/task-1/task-artifacts/artifact-tree/download",
          },
        };
        const html = strategyCandidateLabResultsHtml({
          workflow: {},
          pools: {},
          strategies: {},
          candidates: {
            automatic_tree: {
              latest: null,
              all: [tree],
              total: 1,
              truncated: false,
            },
          },
        });

        for (const expected of [
          "score ≤ 600",
          "剪枝到此节点",
          "调整 score 阈值",
          'data-candidate-lab-interactive-tree-threshold="1"',
          `data-source-tree-id="${sourceTreeId}"`,
          `data-node-id="${nodeId}"`,
          'data-current-threshold="600"',
          "每次剪枝或阈值调整都会创建新 revision",
          "不会写回来源树、物化 frontier、入池、采纳或部署",
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.equal(html.includes("最佳"), false);
        assert.equal(html.includes("排名"), false);
        """
    )


def test_interactive_tree_threshold_controller_requires_explicit_pointer_and_refreshes():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        class FakeSelect {
          constructor(fieldName, value = "") {
            this.dataset = { candidateLabField: fieldName };
            this.options = [];
            this._value = value;
            this.form = null;
          }
          set innerHTML(html) {
            this.options = Array.from(
              html.matchAll(/<option value="([^"]*)"([^>]*)>(.*?)<\\/option>/g),
            ).map((match) => {
              const attrs = match[2];
              const data = {};
              for (const [attribute, key] of [
                ["candidate-lab-projection", "candidateLabProjection"],
                ["source-tree-id", "sourceTreeId"],
                ["node-id", "nodeId"],
                ["operation", "operation"],
                ["feature", "feature"],
                ["current-threshold", "currentThreshold"],
              ]) {
                const value = attrs.match(
                  new RegExp(`data-${attribute}="([^"]*)"`),
                );
                if (value) data[key] = value[1];
              }
              return {
                value: match[1],
                selected: false,
                dataset: data,
              };
            });
            this.value = "";
          }
          set value(value) {
            this._value = String(value);
            for (const option of this.options) {
              option.selected = option.value === this._value;
            }
          }
          get value() { return this._value; }
          get selectedOptions() {
            return this.options.filter((option) => option.selected);
          }
          closest(selector) {
            if (selector === "[data-candidate-lab-field]") return this;
            return selector.includes("interactive_tree_revision")
              ? this.form
              : null;
          }
        }

        const sourceTreeId =
          "candidate-asset-0123456789abcdef0123456789abcdef";
        const nodeId = "node-0123456789abcdef0123";
        const operation = new FakeSelect(
          "interactive_tree_operation",
          "prune_subtree",
        );
        operation.options = [
          { value: "prune_subtree", selected: true, dataset: {} },
          {
            value: "adjust_split_threshold",
            selected: false,
            dataset: {},
          },
        ];
        const source = new FakeSelect("interactive_tree_source_id");
        const node = new FakeSelect("interactive_tree_node_id");
        const threshold = {
          value: "",
          dataset: { candidateLabField: "interactive_tree_threshold" },
          closest(selector) {
            if (selector === "[data-candidate-lab-field]") return this;
            return selector.includes("interactive_tree_revision")
              ? form
              : null;
          },
        };
        const reason = {
          value: "人工确认 score 新阈值",
          dataset: { candidateLabField: "interactive_tree_reason" },
          closest() { return null; },
        };
        const feature = { textContent: "" };
        const currentThreshold = { textContent: "" };
        const help = { textContent: "" };
        const errorTarget = { textContent: "" };
        const hiddenClasses = new Set(["hidden"]);
        const thresholdPanel = {
          classList: {
            contains: (name) => hiddenClasses.has(name),
            toggle(name, force) {
              if (force) hiddenClasses.add(name);
              else hiddenClasses.delete(name);
            },
          },
        };
        const launcher = { open: false };
        const fields = new Map([
          ["interactive_tree_operation", operation],
          ["interactive_tree_source_id", source],
          ["interactive_tree_node_id", node],
          ["interactive_tree_threshold", threshold],
          ["interactive_tree_reason", reason],
        ]);
        const form = {
          dataset: { candidateLabWorkflow: "interactive_tree_revision" },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") {
              return errorTarget;
            }
            if (selector === "[data-candidate-lab-tree-threshold-panel]") {
              return thresholdPanel;
            }
            if (selector === "[data-candidate-lab-tree-threshold-feature]") {
              return feature;
            }
            if (selector === "[data-candidate-lab-tree-current-threshold]") {
              return currentThreshold;
            }
            if (selector === "[data-candidate-lab-tree-help]") return help;
            const match = selector.match(
              /data-candidate-lab-field="([^"]+)"/,
            );
            return match ? (fields.get(match[1]) || null) : null;
          },
          querySelectorAll() { return []; },
          closest(selector) {
            return selector === ".candidate-lab-launcher" ? launcher : null;
          },
          reset() {},
        };
        operation.form = form;
        source.form = form;
        node.form = form;
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector(selector) {
            return selector.includes("interactive_tree_revision")
              ? form
              : null;
          },
          querySelectorAll() { return []; },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const tree = {
          kind: "automatic_tree",
          candidate_id: "candidate-tree",
          detail: {
            source_tree_id: sourceTreeId,
            asset_id: sourceTreeId,
          },
          pointers: {
            nodes: [{
              node_id: nodeId,
              kind: "split",
              feature: "score",
              threshold: 600,
              is_visible: true,
              is_frontier: false,
              can_prune: true,
            }],
            eligible_prunes: [{
              source_tree_id: sourceTreeId,
              node_id: nodeId,
              operation: "prune_subtree",
            }],
            eligible_threshold_adjustments: [{
              source_tree_id: sourceTreeId,
              node_id: nodeId,
              operation: "adjust_split_threshold",
              feature: "score",
              current_threshold: 600,
            }],
          },
        };
        const payload = {
          task_id: "task-1",
          can_start: true,
          blocked_reason: null,
          workflow: {},
          pools: {},
          strategies: {},
          candidates: {
            automatic_tree: {
              latest: null,
              total: 1,
              truncated: false,
              all: [tree],
            },
          },
        };
        const submits = [];
        let refreshes = 0;
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => ({ id: "task-1", task_type: "strategy" }),
          getSelectedTaskId: () => "task-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => {
            refreshes += 1;
            return payload;
          },
          submitStrategyCandidateLabRequest: async (...args) => {
            submits.push(args);
            return { status: "accepted", messages: [] };
          },
          pollAgentMessagesUntilSettled() {},
          settleCandidateLabSubmission: async () => {},
        });

        await controller.selectTask({ id: "task-1", task_type: "strategy" });
        assert.equal(source.value, "");
        assert.equal(node.value, "");
        assert.ok(source.options.some(
          (option) => option.value === sourceTreeId,
        ));

        const thresholdButton = {
          dataset: {
            sourceTreeId,
            nodeId,
            feature: "score",
            currentThreshold: "600",
          },
          closest(selector) {
            return selector
              === "[data-candidate-lab-interactive-tree-threshold]"
              ? this
              : null;
          },
        };
        assert.equal(controller.handleClick({
          target: thresholdButton,
          preventDefault() {},
        }), true);
        assert.equal(operation.value, "adjust_split_threshold");
        assert.equal(source.value, sourceTreeId);
        assert.equal(node.value, nodeId);
        assert.equal(feature.textContent, "score");
        assert.equal(currentThreshold.textContent, "600");
        assert.equal(hiddenClasses.has("hidden"), false);
        assert.match(help.textContent, /不会自动|明确选择/);

        threshold.value = "575.5";
        const result = await controller.submit(form);
        assert.equal(result.status, "accepted");
        assert.equal(submits.length, 1);
        assert.deepEqual(submits[0][1], {
          request_kind: "standard_workflow",
          workflow: "interactive_tree_revision",
          workflow_inputs: {
            source_tree_id: sourceTreeId,
            node_id: nodeId,
            operation: "adjust_split_threshold",
            threshold: 575.5,
            reason: "人工确认 score 新阈值",
          },
        });
        assert.equal(refreshes, 2);

        tree.pointers.nodes[0].threshold = 575.5;
        tree.pointers.eligible_threshold_adjustments[0].current_threshold =
          575.5;
        source.value = sourceTreeId;
        node.value = nodeId;
        threshold.value = "575.5";
        const stale = await controller.submit(form);
        assert.equal(stale, null);
        assert.equal(submits.length, 1);
        assert.match(errorTarget.textContent, /已过期|当前阈值|受认证投影/);
        """
    )
