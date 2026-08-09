from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

from tests.static_stylesheets import read_browser_stylesheets


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_labeling_panel_binds_workspace_builds_proposal_and_confirms_without_llm():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createLabelingSetupPanel } from "./marvis/static/js/v2/labeling_setup_panel.js";

        function element(initial = {}) {
          const listeners = new Map();
          return {
            hidden: false, disabled: false, textContent: "", className: "", value: "",
            ...initial,
            addEventListener(type, listener) { listeners.set(type, listener); },
            dispatch(type) { return listeners.get(type)?.({ preventDefault() {} }); },
          };
        }
        const ids = [
          "labelingSetupPanel", "labelingSetupForm", "labelingProposalPanel",
          "labelingSetupStatus", "labelingSetupSubmit", "labelingProposalSummary",
          "labelingProposalConfirm", "labelingIdCol", "labelingMobCol",
          "labelingCohortCol", "labelingDateCol", "labelingAsOfDate",
          "labelingTargetCol", "labelingObservationWindow", "labelingPerformanceWindow",
          "labelingAtMob", "labelingRuleKind", "labelingDpdFields", "labelingDpdCol",
          "labelingThresholdDpd", "labelingStatusFields", "labelingStatusCol",
          "labelingThresholdStatus", "labelingStates",
        ];
        const elements = Object.fromEntries(ids.map((id) => [id, element()]));
        Object.assign(elements.labelingIdCol, { value: "loan_id" });
        Object.assign(elements.labelingMobCol, { value: "mob" });
        Object.assign(elements.labelingCohortCol, { value: "cohort" });
        Object.assign(elements.labelingDateCol, { value: "snapshot_date" });
        Object.assign(elements.labelingAsOfDate, { value: "2026-06-30" });
        Object.assign(elements.labelingTargetCol, { value: "bad_m3" });
        Object.assign(elements.labelingObservationWindow, { value: "0" });
        Object.assign(elements.labelingPerformanceWindow, { value: "6" });
        Object.assign(elements.labelingAtMob, { value: "3" });
        Object.assign(elements.labelingRuleKind, { value: "dpd" });
        Object.assign(elements.labelingDpdCol, { value: "dpd" });
        Object.assign(elements.labelingThresholdDpd, { value: "30" });

        const workspaceState = {
          dirty: false, loading: false, saving: false,
          serverSnapshot: {
            task_id: "task-j", revision: 4, analysis_generation: 2,
            active_dataset_id: "dataset-1",
            active_dataset_content_hash: "a".repeat(64),
          },
        };
        const workspaceController = {
          getState: () => workspaceState,
          subscribe(listener) { this.listener = listener; return () => {}; },
        };
        let messages = [];
        const calls = [];
        const panel = createLabelingSetupPanel({
          getElementById: (id) => elements[id],
          api: async (path, options) => {
            calls.push({ path, body: JSON.parse(options.body) });
            if (calls.length === 1) {
              return { messages: [{ role: "assistant", metadata: {
                kind: "labeling_preplan_confirmation",
                labeling_proposal: {
                  proposal_hash: "b".repeat(64), source_dataset_name: "source.parquet",
                  rows_at_as_of: 120, rows_excluded_after_as_of: 3,
                  rule_summary: "dpd >= 30",
                  maturity: { all_matured: true, immature_cohorts: [] },
                },
              } }] };
            }
            return { messages: [{ role: "assistant", metadata: {
              labeling_proposal_resolution: "confirmed",
            } }] };
          },
          workspaceController,
          getSelectedTask: () => ({ id: "task-j", task_type: "data_join" }),
          getAgentMessages: () => messages,
          onMessages: (next) => { messages = next; },
        });

        panel.renderAvailability();
        assert.equal(elements.labelingSetupPanel.hidden, false);
        assert.equal(elements.labelingSetupForm.hidden, false);
        assert.equal(elements.labelingSetupSubmit.disabled, false);
        await elements.labelingSetupForm.dispatch("submit");

        assert.equal(calls[0].path, "api/tasks/task-j/agent/messages");
        assert.deepEqual(calls[0].body, {
          content: "提交已确认的标签构造口径",
          labeling_request: {
            dataset_id: "dataset-1",
            expected_content_hash: "a".repeat(64),
            workspace_revision: 4,
            analysis_generation: 2,
            id_col: "loan_id", mob_col: "mob", cohort_col: "cohort",
            date_col: "snapshot_date", as_of_date: "2026-06-30",
            target_col: "bad_m3", observation_window: 0,
            performance_window: 6, at_mob: 3, rule_kind: "dpd",
            dpd_col: "dpd", threshold_dpd: 30,
          },
        });
        assert.equal(elements.labelingSetupForm.hidden, true);
        assert.equal(elements.labelingProposalPanel.hidden, false);
        assert.match(elements.labelingProposalSummary.textContent, /120/);
        assert.match(elements.labelingProposalSummary.textContent, /dpd >= 30/);

        await elements.labelingProposalConfirm.dispatch("click");
        assert.deepEqual(calls[1].body, { content: "确认" });
        assert.equal(elements.labelingSetupPanel.hidden, true);
        """
    )


def test_labeling_panel_fails_closed_for_dirty_workspace_and_mixed_status_rule():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createLabelingSetupPanel } from "./marvis/static/js/v2/labeling_setup_panel.js";

        function element() {
          const listeners = new Map();
          return {
            hidden: false, disabled: false, textContent: "", className: "", value: "",
            addEventListener(type, listener) { listeners.set(type, listener); },
            dispatch(type) { return listeners.get(type)?.({ preventDefault() {} }); },
          };
        }
        const ids = [
          "labelingSetupPanel", "labelingSetupForm", "labelingProposalPanel",
          "labelingSetupStatus", "labelingSetupSubmit", "labelingProposalSummary",
          "labelingProposalConfirm", "labelingIdCol", "labelingMobCol",
          "labelingCohortCol", "labelingDateCol", "labelingAsOfDate",
          "labelingTargetCol", "labelingObservationWindow", "labelingPerformanceWindow",
          "labelingAtMob", "labelingRuleKind", "labelingDpdFields", "labelingDpdCol",
          "labelingThresholdDpd", "labelingStatusFields", "labelingStatusCol",
          "labelingThresholdStatus", "labelingStates",
        ];
        const elements = Object.fromEntries(ids.map((id) => [id, element()]));
        for (const id of ["labelingIdCol", "labelingMobCol", "labelingCohortCol", "labelingDateCol", "labelingAsOfDate", "labelingTargetCol"]) {
          elements[id].value = id;
        }
        elements.labelingObservationWindow.value = "3";
        elements.labelingPerformanceWindow.value = "3";
        elements.labelingAtMob.value = "3";
        elements.labelingRuleKind.value = "status";
        elements.labelingStatusCol.value = "status";
        elements.labelingThresholdStatus.value = "M2";
        elements.labelingStates.value = "current,M1,M1,M2";

        let state = {
          dirty: true, loading: false, saving: false,
          serverSnapshot: {
            task_id: "task-j", revision: 1, analysis_generation: 0,
            active_dataset_id: "dataset-1", active_dataset_content_hash: "c".repeat(64),
          },
        };
        let calls = 0;
        const panel = createLabelingSetupPanel({
          getElementById: (id) => elements[id], api: async () => { calls += 1; },
          workspaceController: { getState: () => state, subscribe() { return () => {}; } },
          getSelectedTask: () => ({ id: "task-j", task_type: "data_join" }),
          getAgentMessages: () => [],
        });
        panel.renderAvailability();
        assert.equal(elements.labelingSetupSubmit.disabled, true);
        assert.match(elements.labelingSetupStatus.textContent, /保存或丢弃/);

        state = { ...state, dirty: false };
        panel.renderAvailability();
        await elements.labelingSetupForm.dispatch("submit");
        assert.equal(calls, 0);
        assert.match(elements.labelingSetupStatus.textContent, /重复/);
        """
    )


def test_labeling_panel_is_mounted_next_to_data_workspace():
    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "marvis/static/index.html").read_text(encoding="utf-8")
    css = read_browser_stylesheets(ROOT / "marvis/static")

    assert 'id="labelingSetupPanel"' in index_html
    assert 'id="labelingSetupForm"' in index_html
    assert 'id="labelingProposalConfirm"' in index_html
    assert "createLabelingSetupPanel," in app_js
    assert 'from "./js/v2/labeling_setup_panel.js";' in app_js
    assert "const labelingSetupPanel = createLabelingSetupPanel({" in app_js
    assert "labelingSetupPanel.renderAvailability();" in app_js
    assert ".labeling-setup-panel" in css
