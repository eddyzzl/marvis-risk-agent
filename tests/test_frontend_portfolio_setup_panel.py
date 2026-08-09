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


def test_portfolio_setup_panel_submits_strict_typed_contract_and_closes_on_gate():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createPortfolioSetupPanel } from "./marvis/static/js/v2/portfolio_setup_panel.js";

        function element(initial = {}) {
          const listeners = new Map();
          return {
            hidden: false,
            disabled: false,
            textContent: "",
            className: "",
            value: "",
            ...initial,
            addEventListener(type, listener) { listeners.set(type, listener); },
            dispatch(type) {
              return listeners.get(type)?.({ preventDefault() {}, stopPropagation() {} });
            },
          };
        }

        const elements = Object.fromEntries([
          "portfolioSetupPanel",
          "portfolioSetupForm",
          "portfolioSetupStatus",
          "portfolioSetupSubmit",
          "portfolioIdCol",
          "portfolioSnapshotCol",
          "portfolioBucketCol",
          "portfolioBalanceCol",
          "portfolioSegmentCol",
          "portfolioLossState",
          "portfolioLgd",
          "portfolioHorizonMonths",
          "portfolioScoreCol",
          "portfolioExperimentId",
        ].map((id) => [id, element()]));

        elements.portfolioLgd.value = "0.45";
        elements.portfolioHorizonMonths.value = "18";
        const values = {
          portfolioIdCol: "loan_id",
          portfolioSnapshotCol: "snapshot_month",
          portfolioBucketCol: "bucket",
          portfolioBalanceCol: "balance",
          portfolioSegmentCol: "segment",
          portfolioLossState: "charged_off",
        };
        Object.entries(values).forEach(([id, value]) => { elements[id].value = value; });

        let task = { id: "task-p", task_type: "portfolio" };
        let messages = [{
          role: "assistant",
          metadata: { kind: "portfolio_setup_required" },
        }];
        const calls = [];
        const panel = createPortfolioSetupPanel({
          getElementById: (id) => elements[id],
          getSelectedTask: () => task,
          getAgentMessages: () => messages,
          api: async (path, options) => {
            calls.push({ path, options });
            return {
              messages: [{
                role: "assistant",
                metadata: {
                  kind: "gate",
                  portfolio_states: { proposed_states: ["current", "charged_off"] },
                },
              }],
            };
          },
          onMessages: (next) => { messages = next; },
        });

        panel.renderAvailability();
        assert.equal(elements.portfolioSetupPanel.hidden, false);
        await elements.portfolioSetupForm.dispatch("submit");

        assert.equal(calls.length, 1);
        assert.equal(calls[0].path, "api/tasks/task-p/agent/messages");
        assert.equal(calls[0].options.method, "POST");
        assert.deepEqual(JSON.parse(calls[0].options.body), {
          content: "提交已确认的组合分析口径",
          portfolio_request: {
            id_col: "loan_id",
            snapshot_col: "snapshot_month",
            bucket_col: "bucket",
            balance_col: "balance",
            segment_col: "segment",
            loss_state: "charged_off",
            lgd: 0.45,
            horizon_months: 18,
          },
        });
        assert.equal(elements.portfolioSetupPanel.hidden, true);
        assert.equal(elements.portfolioSetupSubmit.disabled, true);
        """
    )


def test_portfolio_setup_panel_requires_complete_trend_pair_and_valid_economics():
    run_node(
        """
        import assert from "node:assert/strict";
        import { createPortfolioSetupPanel } from "./marvis/static/js/v2/portfolio_setup_panel.js";

        function element() {
          const listeners = new Map();
          return {
            hidden: false,
            disabled: false,
            textContent: "",
            className: "",
            value: "",
            addEventListener(type, listener) { listeners.set(type, listener); },
            dispatch(type) { return listeners.get(type)?.({ preventDefault() {} }); },
          };
        }
        const ids = [
          "portfolioSetupPanel", "portfolioSetupForm", "portfolioSetupStatus",
          "portfolioSetupSubmit", "portfolioIdCol", "portfolioSnapshotCol",
          "portfolioBucketCol", "portfolioBalanceCol", "portfolioSegmentCol",
          "portfolioLossState", "portfolioLgd", "portfolioHorizonMonths",
          "portfolioScoreCol", "portfolioExperimentId",
        ];
        const elements = Object.fromEntries(ids.map((id) => [id, element()]));
        for (const id of [
          "portfolioIdCol", "portfolioSnapshotCol", "portfolioBucketCol",
          "portfolioBalanceCol", "portfolioSegmentCol", "portfolioLossState",
        ]) elements[id].value = id;
        elements.portfolioLgd.value = "1.2";
        elements.portfolioHorizonMonths.value = "0";
        elements.portfolioScoreCol.value = "score";

        let calls = 0;
        const panel = createPortfolioSetupPanel({
          getElementById: (id) => elements[id],
          getSelectedTask: () => ({ id: "task-p", task_type: "portfolio" }),
          getAgentMessages: () => [],
          api: async () => { calls += 1; },
        });
        panel.renderAvailability();
        await elements.portfolioSetupForm.dispatch("submit");

        assert.equal(calls, 0);
        assert.match(elements.portfolioSetupStatus.textContent, /LGD/);

        elements.portfolioLgd.value = "0.5";
        elements.portfolioHorizonMonths.value = "12";
        await elements.portfolioSetupForm.dispatch("submit");
        assert.equal(calls, 0);
        assert.match(elements.portfolioSetupStatus.textContent, /分数列和实验 ID/);
        """
    )


def test_portfolio_open_gate_uses_deterministic_turn_without_llm():
    run_node(
        """
        import assert from "node:assert/strict";
        import { portfolioTurnUsesDeterministicRoute } from "./marvis/static/js/v2/portfolio_setup_panel.js";

        const task = { id: "task-p", task_type: "portfolio", run_mode: "agent" };
        assert.equal(portfolioTurnUsesDeterministicRoute(task, [{
          role: "assistant",
          metadata: { kind: "portfolio_setup_required" },
        }]), true);
        assert.equal(portfolioTurnUsesDeterministicRoute(task, [{
          role: "assistant",
          metadata: { kind: "gate", portfolio_states: { proposed_states: ["C", "M1"] } },
        }]), true);
        assert.equal(portfolioTurnUsesDeterministicRoute(task, [{
          role: "assistant",
          metadata: { kind: "gate" },
        }]), true);
        assert.equal(portfolioTurnUsesDeterministicRoute(task, [{
          role: "assistant",
          metadata: { kind: "result" },
        }]), false);
        assert.equal(portfolioTurnUsesDeterministicRoute(
          { task_type: "strategy", run_mode: "agent" },
          [{ role: "assistant", metadata: { kind: "gate" } }],
        ), false);
        """
    )


def test_portfolio_entry_and_setup_panel_are_mounted_in_product_ui():
    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "marvis/static/index.html").read_text(encoding="utf-8")
    task_types = (ROOT / "marvis/static/js/task-types.js").read_text(encoding="utf-8")
    plan_rail = (ROOT / "marvis/static/js/v2/plan_rail_controller.js").read_text(
        encoding="utf-8"
    )

    assert 'id="welcomePortfolioAnalysisCard"' in index_html
    assert 'data-task-kind="portfolio"' in index_html
    assert 'id="portfolioSetupPanel"' in index_html
    assert 'id="portfolioSetupForm"' in index_html
    assert "createPortfolioSetupPanel," in app_js
    assert "portfolioTurnUsesDeterministicRoute," in app_js
    assert 'from "./js/v2/portfolio_setup_panel.js";' in app_js
    assert "const portfolioSetupPanel = createPortfolioSetupPanel({" in app_js
    assert "portfolioSetupPanel.renderAvailability();" in app_js
    assert "portfolioTurnUsesDeterministicRoute(task, messages)" in app_js
    assert "portfolio:" in task_types
    assert '"portfolio"' in plan_rail
