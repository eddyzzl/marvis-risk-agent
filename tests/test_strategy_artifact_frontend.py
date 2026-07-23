from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def _run_node_json(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_strategy_artifact_api_wrapper_uses_task_scoped_endpoint():
    payload = _run_node_json(
        """
        const calls = [];
        globalThis.fetch = async (url, options = {}) => {
          calls.push([url, options.method]);
          return {
            ok: true,
            status: 200,
            headers: { get: () => "application/json" },
            json: async () => ({ task_id: "task / 1", artifacts: [] }),
            text: async () => "",
          };
        };
        const { listStrategyArtifacts, listTaskArtifacts } = await import(
          "./marvis/static/js/v2/api_v2.js"
        );
        await listStrategyArtifacts("task / 1");
        await listTaskArtifacts("task / 1");
        process.stdout.write(JSON.stringify({ calls }));
        """
    )

    assert payload["calls"] == [
        ["/api/tasks/task%20%2F%201/strategy-artifacts", "GET"],
        ["/api/tasks/task%20%2F%201/task-artifacts", "GET"],
    ]


def test_done_strategy_plan_lazily_loads_escaped_artifacts_once_per_plan():
    module_url = (STATIC / "js" / "v2" / "plan_rail_controller.js").as_uri()
    payload = _run_node_json(
        f"""
        const {{ createPlanRailController }} = await import({json.dumps(module_url)});
        const classes = {{ add() {{}}, remove() {{}}, toggle() {{}} }};
        const workflowStepper = {{ innerHTML: "", dataset: {{}} }};
        const planDriverActions = {{
          innerHTML: "",
          dataset: {{}},
          classList: classes,
          setAttribute() {{}},
          querySelector() {{ return null; }},
        }};
        const elements = {{
          progressRail: {{ setAttribute() {{}} }},
          workflowStepper,
          planDriverActions,
        }};
        function $(id) {{ return elements[id] || null; }}
        globalThis.document = {{
          querySelector() {{ return {{ textContent: "" }}; }},
        }};
        let plan = {{
          id: "plan-1",
          revision: 1,
          status: "running",
          steps: [{{
            id: "step-1",
            index: 0,
            phase: "策略",
            title: "生成策略",
            status: "done",
            tool_ref: {{ plugin: "strategy", tool: "adopt_strategy" }},
            depends_on: [],
          }}],
        }};
        globalThis.fetch = async () => ({{
          ok: true,
          json: async () => ({{ plans: [plan] }}),
        }});
        let artifactCalls = 0;
        let taskArtifactCalls = 0;
        const listStrategyArtifactsClient = async () => {{
          artifactCalls += 1;
          return {{ artifacts: [{{
            id: "artifact-1",
            filename: "<img src=x onerror=alert(1)>.md",
            kind: "strategy<script>",
            version: 2,
            asset_status: "adopted_local",
            available: true,
            download_url: "/api/tasks/task-A/strategy-artifacts/artifact-1/download",
          }}] }};
        }};
        const listTaskArtifactsClient = async () => {{
          taskArtifactCalls += 1;
          return {{ artifacts: [
            {{
              id: "task-duplicate",
              filename: "<img src=x onerror=alert(1)>.md",
              kind: "strategy<script>",
              origin_tool: "strategy.duplicate",
              available: true,
              download_url: "/api/tasks/task-A/task-artifacts/task-duplicate/download?expected_content_hash={"d" * 64}",
            }},
            {{
              id: "task-analysis-1",
              filename: "profit.csv",
              kind: "profit_csv",
              origin_tool: "strategy.profit_calc",
              created_at: "2026-07-23T08:00:00Z",
              available: true,
              download_url: "/api/tasks/task-A/task-artifacts/task-analysis-1/download?expected_content_hash={"a" * 64}",
            }},
            ...[
              ["delivery-python-old", "strategy.py", "strategy_delivery_python", "{"5" * 64}"],
              ["delivery-sql-old", "strategy.sql", "strategy_delivery_sql", "{"6" * 64}"],
              ["delivery-json-old", "strategy.json", "strategy_delivery_json", "{"7" * 64}"],
              ["delivery-equivalence-old", "equivalence.json", "strategy_delivery_equivalence_json", "{"8" * 64}"],
            ].map(([id, filename, kind, contentHash]) => ({{
              id,
              filename,
              kind,
              origin_tool: "strategy.export_strategy_delivery",
              created_at: "2026-07-23T08:01:00Z",
              available: true,
              download_url: `/api/tasks/task-A/task-artifacts/${{id}}/download?expected_content_hash=${{contentHash}}`,
            }})),
            ...[
              ["delivery-python", "strategy.py", "strategy_delivery_python", "{"1" * 64}"],
              ["delivery-sql", "strategy.sql", "strategy_delivery_sql", "{"2" * 64}"],
              ["delivery-json", "strategy.json", "strategy_delivery_json", "{"3" * 64}"],
              ["delivery-equivalence", "equivalence.json", "strategy_delivery_equivalence_json", "{"4" * 64}"],
            ].map(([id, filename, kind, contentHash]) => ({{
              id,
              filename,
              kind,
              origin_tool: "strategy.export_strategy_delivery",
              created_at: "2026-07-23T08:02:00Z",
              available: true,
              download_url: `/api/tasks/task-A/task-artifacts/${{id}}/download?expected_content_hash=${{contentHash}}`,
            }})),
          ] }};
        }};
        const controller = createPlanRailController({{
          $,
          getSelectedTask: () => ({{ task_type: "strategy" }}),
          getSelectedTaskId: () => "task-A",
          getAgentMessages: () => [],
          isAgentMode: () => true,
          renderWorkflowStepper: () => {{}},
          listStrategyArtifactsClient,
          listTaskArtifactsClient,
        }});
        const signatures = {{}};
        controller.render({{ force: true, renderSignatures: signatures }});
        await new Promise((resolve) => setTimeout(resolve, 10));
        controller.render({{ force: true, renderSignatures: signatures }});
        const preDoneCalls = artifactCalls;
        const preDoneTaskCalls = taskArtifactCalls;

        plan = {{ ...plan, status: "done" }};
        controller.resetFetchThrottle("task-A");
        controller.render({{ force: true, renderSignatures: signatures }});
        await new Promise((resolve) => setTimeout(resolve, 10));
        controller.render({{ force: true, renderSignatures: signatures }});
        await new Promise((resolve) => setTimeout(resolve, 10));
        controller.render({{ force: true, renderSignatures: signatures }});
        const firstHtml = planDriverActions.innerHTML;
        const firstCalls = artifactCalls;

        plan = {{ ...plan, id: "plan-2", revision: 2 }};
        controller.resetFetchThrottle("task-A");
        controller.render({{ force: true, renderSignatures: signatures }});
        await new Promise((resolve) => setTimeout(resolve, 10));
        controller.render({{ force: true, renderSignatures: signatures }});
        await new Promise((resolve) => setTimeout(resolve, 10));
        controller.render({{ force: true, renderSignatures: signatures }});
        process.stdout.write(JSON.stringify({{
          preDoneCalls,
          preDoneTaskCalls,
          firstCalls,
          finalCalls: artifactCalls,
          finalTaskCalls: taskArtifactCalls,
          firstHtml,
          finalHtml: planDriverActions.innerHTML,
        }}));
        """
    )

    assert payload["preDoneCalls"] == 0
    assert payload["preDoneTaskCalls"] == 0
    assert payload["firstCalls"] == 1
    assert payload["finalCalls"] == 2
    assert payload["finalTaskCalls"] == 2
    assert "本地采纳" in payload["firstHtml"]
    assert "策略产物" in payload["firstHtml"]
    assert "不代表生产环境已上线" in payload["firstHtml"]
    assert "生产已部署" not in payload["firstHtml"]
    assert "已在生产环境上线" not in payload["firstHtml"]
    assert "artifact-1/download" in payload["firstHtml"]
    assert (
        f"task-artifacts/task-analysis-1/download?expected_content_hash={'a' * 64}"
        in payload["firstHtml"]
    )
    assert "task-artifacts/task-duplicate/download" not in payload["firstHtml"]
    assert "任务分析" in payload["firstHtml"]
    assert "Python" in payload["firstHtml"]
    assert "DuckDB SQL" in payload["firstHtml"]
    assert "Strategy JSON" in payload["firstHtml"]
    assert "Equivalence JSON" in payload["firstHtml"]
    assert payload["firstHtml"].count("离线交付") == 4
    for content_hash in ("1" * 64, "2" * 64, "3" * 64, "4" * 64):
        assert f"?expected_content_hash={content_hash}" in payload["firstHtml"]
    for content_hash in ("5" * 64, "6" * 64, "7" * 64, "8" * 64):
        assert f"?expected_content_hash={content_hash}" not in payload["firstHtml"]
    assert "delivery-python-old/download" not in payload["firstHtml"]
    assert "<img" not in payload["firstHtml"]
    assert "<script>" not in payload["firstHtml"]
    assert "&lt;img" in payload["firstHtml"]
    assert "&lt;script&gt;" in payload["firstHtml"]


def test_strategy_artifact_card_styles_live_in_driver_actions_family():
    source = (STATIC / "js" / "v2" / "plan_rail_controller.js").read_text(
        encoding="utf-8"
    )
    css = (STATIC / "css" / "v2-workbench.css").read_text(encoding="utf-8")

    assert "生产已部署" not in source
    assert "已在生产环境上线" not in source
    assert "不代表生产环境已上线" in source
    assert ".strategy-artifacts-notice" in css
    assert ".strategy-artifact-list" in css
    assert ".strategy-artifact-row" in css
    assert ".strategy-artifact-download" in css
