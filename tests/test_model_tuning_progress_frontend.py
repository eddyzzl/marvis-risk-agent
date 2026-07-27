import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def _node_json(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_tuning_progress_renderer_has_compact_and_glass_card_variants():
    module_url = (STATIC / "js" / "v2" / "model_tuning_progress.js").as_uri()
    script = "\n".join(
        [
            f"import {{ renderModelTuningProgress, renderModelTuningMessageProgress }} from {json.dumps(module_url)};",
            "const progress = { kind: 'model_tuning', algorithm: 'xgb', algorithm_index: 2, algorithm_total: 3, trial: 17, trial_total: 40, stage: 'fine', completed_trials: 57, total_trials: 120, percent: 47.5, selection_score: 0.3987, test_ks: 0.4058, best_selection_score: 0.412345, best_test_ks: 0.4172, best_by_algorithm: { lgb: { selection_score: 0.4012, test_ks: 0.4091 }, xgb: { selection_score: 0.412345, test_ks: 0.4172 } } };",
            "const compact = renderModelTuningProgress(progress, { compact: true });",
            "const card = renderModelTuningMessageProgress({ role: 'assistant', metadata: { kind: 'tool_progress', progress } });",
            "const coarseCard = renderModelTuningProgress({ ...progress, stage: 'coarse' });",
            "const unsafeCard = renderModelTuningProgress({ ...progress, algorithm: '<custom-model>', stage: 'CV <fold> with a deliberately long stage description' });",
            "const ordinaryStep = renderModelTuningProgress({ kind: 'data_join', percent: 50 }, { compact: true });",
            "const ordinaryMessage = renderModelTuningMessageProgress({ role: 'assistant', metadata: { kind: 'gate', progress } });",
            "process.stdout.write(JSON.stringify({ compact, card, coarseCard, unsafeCard, ordinaryStep, ordinaryMessage }));",
        ]
    )
    payload = _node_json(script)

    compact = payload["compact"]
    assert 'data-model-tuning-progress="compact"' in compact
    assert 'aria-valuenow="48"' in compact
    assert "XGBoost" in compact
    assert "算法 2 / 3" in compact
    assert "17 / 40 轮" in compact
    assert "57 / 120" in compact
    assert "最佳 0.4123" in compact
    assert "button" not in compact

    card = payload["card"]
    assert 'data-model-tuning-progress="card"' in card
    assert 'data-status="running"' in card
    assert "正在执行 · 调参" in card
    assert "最佳选择分" in card
    assert "细搜" in card
    assert "当前轮选择分" in card
    assert "0.3987" in card
    assert "最佳 Test KS" in card
    assert "LightGBM" in card
    assert "button" not in card
    assert "粗搜" in payload["coarseCard"]
    assert "&lt;custom-model&gt;" in payload["unsafeCard"]
    assert "CV &lt;fold&gt;" in payload["unsafeCard"]

    assert payload["ordinaryStep"] == ""
    assert payload["ordinaryMessage"] == ""


def test_tuning_progress_terminal_messages_keep_last_metrics_and_render_static_state():
    module_url = (STATIC / "js" / "v2" / "model_tuning_progress.js").as_uri()
    script = "\n".join(
        [
            f"import {{ normalizeModelTuningProgress, renderModelTuningMessageProgress }} from {json.dumps(module_url)};",
            "const progress = { kind: 'model_tuning', algorithm: 'lgb', trial: 9, trial_total: 40, completed_trials: 89, total_trials: 120, percent: 74.17, selection_score: 0.421, test_ks: 0.428, best_selection_score: 0.4321, best_test_ks: 0.4402 };",
            "const statuses = ['succeeded', 'failed', 'cancelled', 'interrupted'];",
            "const output = Object.fromEntries(statuses.map((status) => { const message = { metadata: { kind: 'tool_progress', status, streaming: false, progress } }; return [status, { normalized: normalizeModelTuningProgress(message), html: renderModelTuningMessageProgress(message) }]; }));",
            "process.stdout.write(JSON.stringify(output));",
        ]
    )
    payload = _node_json(script)
    labels = {
        "succeeded": "调参已完成",
        "failed": "调参失败",
        "cancelled": "调参已取消",
        "interrupted": "调参已中断",
    }

    for status, label in labels.items():
        normalized = payload[status]["normalized"]
        html = payload[status]["html"]
        assert normalized["status"] == status
        assert normalized["statusLabel"] == label
        assert normalized["terminal"] is True
        assert f'data-status="{status}"' in html
        assert label in html
        assert "89 / 120 轮" in html
        assert "0.4210" in html
        assert "0.4280" in html
        assert "0.4321" in html
        assert "0.4402" in html
        assert 'style="width:74.17%"' in html


def test_tuning_progress_replaces_only_the_superseded_thinking_placeholder():
    module_url = (STATIC / "js" / "v2" / "model_tuning_progress.js").as_uri()
    script = "\n".join(
        [
            f"import {{ hideSupersededTuningThinking }} from {json.dumps(module_url)};",
            "const oldThinking = { id: 'thinking-old', role: 'assistant', content: '', metadata: { optimistic: true, streaming: true } };",
            "const progress = { id: 'progress', role: 'assistant', content: '', metadata: { kind: 'tool_progress', progress: { kind: 'model_tuning', algorithm: 'lgb', trial_total: 40, total_trials: 120 } } };",
            "const newUser = { id: 'user-new', role: 'user', content: '另一个问题', metadata: { optimistic: true } };",
            "const newThinking = { id: 'thinking-new', role: 'assistant', content: '', metadata: { optimistic: true, streaming: true } };",
            "const first = hideSupersededTuningThinking([oldThinking, progress]).map((item) => item.id);",
            "const laterTurn = hideSupersededTuningThinking([oldThinking, progress, newUser, newThinking]).map((item) => item.id);",
            "process.stdout.write(JSON.stringify({ first, laterTurn }));",
        ]
    )
    payload = _node_json(script)

    assert payload["first"] == ["progress"]
    assert payload["laterTurn"] == ["progress", "user-new", "thinking-new"]


def test_plan_rail_keeps_running_subtask_spinner_and_adds_compact_progress():
    module_url = (STATIC / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = "\n".join(
        [
            f"import {{ createPlanRailController }} from {json.dumps(module_url)};",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper: { innerHTML: '' } };",
            "function $(id) { return elements[id] || null; }",
            "globalThis.document = { querySelector() { return { textContent: '' }; } };",
            "const plan = { id: 'plan-1', status: 'running', steps: [{ id: 'tune', index: 5, phase: '建模', title: '调参', status: 'running', tool_ref: { plugin: 'modeling', tool: 'tune_hyperparameters' }, progress: { kind: 'model_tuning', algorithm: 'catboost', algorithm_index: 3, algorithm_total: 3, trial: 8, trial_total: 40, completed_trials: 88, total_trials: 120, percent: 73.33, selection_score: 0.4231 } }] };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({ $, stepCheckerHtml: (status) => `<span class=\"check-icon ${status}\"></span>`, getSelectedTask: () => ({ task_type: 'modeling' }), getSelectedTaskId: () => 'task-A', getAgentMessages: () => [], isAgentMode: () => true, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "controller.render({ force: true, renderSignatures: {} });",
            "await new Promise((resolve) => setTimeout(resolve, 20));",
            "controller.render({ force: true, renderSignatures: {} });",
            "process.stdout.write(JSON.stringify({ html: elements.workflowStepper.innerHTML }));",
        ]
    )
    html = _node_json(script)["html"]

    assert "子任务 · 1" in html
    assert 'class="check-icon running"' in html
    assert 'data-model-tuning-progress="compact"' in html
    assert "CatBoost" in html
    assert "8 / 40 轮" in html
    assert "88 / 120" in html
    assert "button" not in html


def test_app_wires_tool_progress_into_timeline_and_css_prevents_rail_overflow():
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    plan_js = (STATIC / "js" / "v2" / "plan_rail_controller.js").read_text(encoding="utf-8")
    css = (STATIC / "css" / "v2-workbench.css").read_text(encoding="utf-8")

    assert 'from "./js/v2/model_tuning_progress.js"' in app_js
    assert "hideSupersededTuningThinking(" in app_js
    assert "renderModelTuningMessageProgress(message)" in app_js
    assert "if (tuningProgress) return tuningProgress.statusLabel;" in app_js
    assert "!options.hideMeta || isTuningProgress" in app_js
    assert 'tool_progress: metadata.kind === "tool_progress"' in app_js
    assert "progress: metadata.progress || metadata" in app_js
    assert 'status: metadata.status || ""' in app_js
    assert 'from "./model_tuning_progress.js"' in plan_js
    assert "renderModelTuningProgress(step?.progress, { compact: true })" in plan_js

    assert ".model-tuning-progress--compact" in css
    assert ".model-tuning-progress--card" in css
    assert "backdrop-filter: blur(18px)" in css
    assert ".model-tuning-progress {" in css
    progress_rule = css.split(".model-tuning-progress {", 1)[1].split("}", 1)[0]
    assert "min-width: 0" in progress_rule
    assert "max-width: 100%" in progress_rule
    assert "overflow: hidden" in progress_rule
    assert '.model-tuning-progress[data-status="running"] .model-tuning-progress__track > span' in css
    track_rule = css.split(".model-tuning-progress__track > span {", 1)[1].split("}", 1)[0]
    assert "animation:" not in track_rule
