"""F13: execute creation events and inert-fragment handling in Chromium.

The app and browser DOM are real. Only service responses are controlled; these
checks establish frontend behavior, not successful backend task execution.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.javascript_source import slice_function
from tests.test_frontend_playwright_smoke import _serve_smoke_page, _smoke_html

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("MARVIS_RUN_PLAYWRIGHT_SMOKE") != "1",
                       reason="Set MARVIS_RUN_PLAYWRIGHT_SMOKE=1 for real DOM behavior."),
]


def _wait_for_app_boot(page, errors, pending):
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    try:
        page.wait_for_selector("body:not(.app-booting)", timeout=20000)
    except PlaywrightTimeoutError:
        diagnostics = page.evaluate("({bodyClass: document.body.className, status: document.getElementById('actionStatus')?.textContent || ''})")
        pytest.fail(f"app boot failed: pageerrors={errors}; state={diagnostics}; pending={pending}")


def test_create_dialog_keyboard_validation_payload_and_backdrop_execute_in_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    posts, errors, uploads = [], [], []
    known = {"/api/branding", "/api/tasks", "/api/settings/execution-environment/options",
             "/api/settings/llm", "/api/settings/memory-policy"}
    with _serve_smoke_page(_smoke_html()) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.set_default_timeout(8000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            pending = {}
            page.on("request", lambda request: pending.update({request.url: request.resource_type}))
            page.on("requestfinished", lambda request: pending.pop(request.url, None))
            page.on("requestfailed", lambda request: errors.append(f"request failed {request.url}: {request.failure}"))
            def route(request):
                path = urlparse(request.request.url).path
                if path == "/api/material-uploads":
                    uploads.append(request.request.post_data_buffer)
                    request.fulfill(json={"source_dir": "/uploaded/bundle"})
                elif path == "/api/tasks" and request.request.method == "POST":
                    posts.append(request.request.post_data_json)
                    request.fulfill(status=422, json={"detail": "材料需要重新核对"})
                elif path in known:
                    request.continue_()
                else:
                    request.fulfill(json={})
            page.route("**/api/**", route)
            page.add_init_script("localStorage.clear(); localStorage.setItem('marvis_ui_locale','zh');")
            page.goto(address.rsplit("/", 1)[0] + "/")
            _wait_for_app_boot(page, errors, pending)
            page.click("#welcomeModelValidationCard")
            page.press("#homeComposerTemplate", "Enter")
            page.click("#homeDraft [data-home-draft-advanced]")
            page.wait_for_selector("#taskDialog[open]")
            assert page.locator('input[name="runMode"]:checked').count() == 1
            page.fill("#modelName", "用户模型名称")
            page.fill("#validator", "用户负责人")
            page.fill("#sourceDir", "/declared/materials")
            page.locator("#createTaskReportFields").evaluate("element => element.open = true")
            page.fill("#createModelOverview", "用户背景")
            page.press("#createModelOverview", "Enter")
            assert page.input_value("#createModelOverview").endswith("\n")
            assert posts == []
            page.dispatch_event("#modelName", "keydown", {"key": "Enter", "isComposing": True})
            assert posts == []

            # The controller must require an explicit selected mode, even when
            # a restored form or browser extension leaves all radios unchecked.
            page.locator('input[name="runMode"]').evaluate_all("items => items.forEach(item => item.checked = false)")
            page.click("#createTaskButton")
            page.get_by_text("请选择执行模式。", exact=True).wait_for(state="visible")
            assert posts == []
            page.locator("label:has(#runModeAgent)").click()
            with page.expect_response(lambda response: urlparse(response.url).path == "/api/tasks" and response.request.method == "POST"):
                page.press("#modelName", "Enter")
            page.get_by_text("材料需要重新核对", exact=False).wait_for(state="visible")
            assert len(posts) == 1
            payload = posts[0]
            assert payload["model_name"] == "用户模型名称"
            assert payload["model_version"] == ""
            assert payload["run_mode"] == "agent"
            assert payload["source_dir"] == "/declared/materials"
            assert payload["report_values"]["TEXT:model_overview"] == "用户背景"
            assert "TEXT:model_training_description" not in payload["report_values"]
            assert not set(payload).intersection({"algorithm", "notebook_path", "sample_path", "pmml_path",
                                                  "dictionary_path", "target_col", "score_col", "split_col", "time_col", "feature_columns"})
            assert page.locator("#taskDialog").evaluate("dialog => dialog.open")
            assert page.input_value("#modelName") == "用户模型名称"
            assert page.locator("#modelVersion").count() == 0
            page.click("#materialSourceUploadTab")
            page.locator("#materialUploadInput").set_input_files([
                {"name": "samples.csv", "mimeType": "text/csv", "buffer": b"x,y\n1,0\n"},
                {"name": "model.pmml", "mimeType": "application/xml", "buffer": b"<PMML/>"},
            ])
            page.click("#createTaskButton")
            page.wait_for_function("document.getElementById('createTaskButton').disabled === false")
            assert len(uploads) == 1
            assert len(posts) == 2
            assert posts[1]["source_dir"] == "/uploaded/bundle"
            assert b'name="files"' in uploads[0]
            assert b'name="relative_paths"' in uploads[0]
            assert b"samples.csv" in uploads[0] and b"model.pmml" in uploads[0]
            page.locator("#taskDialogTitle").click()
            assert page.locator("#taskDialog").evaluate("dialog => dialog.open")
            page.mouse.click(4, 4)
            page.wait_for_selector("#taskDialog[open]", state="detached")
            assert errors == []
        finally:
            browser.close()


def test_frozen_snapshot_sanitizer_uses_real_dom_and_keeps_safe_content():
    playwright = pytest.importorskip("playwright.sync_api")
    app = (Path(__file__).resolve().parents[1] / "marvis/static/app.js").read_text()
    sanitize = slice_function(app, "function stripIdsFromHtml(")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.set_content("<main id='live'></main>")
            outcome = page.evaluate("(source) => {" + sanitize + """
                const raw = `<section id="stale" onclick="window.executed=1"><script>window.executed=2</script>
                  <button id="stale-button" onfocus="window.executed=3">用户内容</button><table><tr><td>42</td></tr></table></section>`;
                const safe = stripIdsFromHtml(raw);
                document.querySelector('main').innerHTML = safe;
                document.querySelector('button').click(); document.querySelector('button').focus();
                return {scripts:document.querySelectorAll('script').length,
                  ids:[...document.querySelectorAll('main [id]')].map(x=>x.id),
                  events:[...document.querySelectorAll('main *')].flatMap(x=>[...x.attributes].filter(a=>/^on/i.test(a.name))),
                  text:document.querySelector('button').textContent, cell:document.querySelector('td').textContent,
                  executed:window.executed || null};
            }""", "")
            assert outcome == {"scripts": 0, "ids": [], "events": [], "text": "用户内容", "cell": "42", "executed": None}
        finally:
            browser.close()


def test_step_rail_keeps_scroll_and_nodes_while_elapsed_text_updates_in_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    app = (Path(__file__).resolve().parents[1] / "marvis/static/app.js").read_text()
    functions = "\n".join(slice_function(app, signature) for signature in (
        "function renderWorkflowStepper(", "function renderNotebookStepRail(",
        "function refreshWorkflowStepperElapsedTimes(", "function stepElapsedSeconds(", "function formatStepElapsed(",
    ))
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.set_content("""<style>#workflowStepper{height:120px;overflow:auto}.step{min-height:150px}</style>
                <aside id="progressRail"><div class="step-rail-head"><h3></h3></div><div id="workflowStepper"></div></aside>""")
            result = page.evaluate("() => {" + functions + """
                const $=id=>document.getElementById(id), renderSignatures={}, expandedValidationStages=new Map();
                let task={id:'a',status:'running'}, now=Date.parse('2026-11-04T00:00:10Z');
                Date.now=()=>now;
                const step={id:'cell',title:'用户步骤',status:'running',started_at:'2026-11-04T00:00:00Z',cell_count:2};
                const workflowSteps=['scan','notebook','metrics','report'].map(id=>({id,title:id,hint:id,target:id,action:id}));
                const renderTaskArtifacts=()=>{},syncRailStatus=()=>{},workflowIndex=()=>1;
                const workbenchTask=()=>task,workbenchTaskId=()=>task.id,workflowStepperSignature=()=>task.id;
                const planRailController={render:()=>false,clearRetryPanel(){},clearDriverActionsPanel(){}};
                const setTextIfChanged=(element,text)=>element.textContent=text;
                const workflowStepForTask=value=>value,workflowStepStatus=i=>i===1?'running':'pending';
                const stepCheckerHtml=tone=>'<i data-tone="'+tone+'"></i>',stepActionButtonHtml=()=>'';
                const usesPmmlScoringWorkflow=()=>false,localizeHtml=value=>value;
                const notebookStepsForRail=()=>[step],metricStepsForRail=()=>[];
                const notebookStepToneForRail=value=>value.status,stepAfterInLatestNotebookSteps=()=>null;
                const escapeHtml=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
                renderWorkflowStepper();
                const rail=$('workflowStepper'), old=rail.firstElementChild, elapsed=rail.querySelector('[data-step-elapsed-key]');
                const first=elapsed.textContent;rail.scrollTop=80;
                now+=5000;renderWorkflowStepper();
                const second=elapsed.textContent, same=rail.firstElementChild===old, scroll=rail.scrollTop;
                renderWorkflowStepper({force:true});
                const forcedSame=rail.firstElementChild===old;
                task={id:'b',status:'running'};renderWorkflowStepper();
                return {first,second,same,scroll,forcedSame,newScroll:rail.scrollTop,task:rail.dataset.taskId,
                  spans:rail.querySelectorAll('[data-step-elapsed-key="notebook:cell"]').length,
                  cells:rail.querySelector('.step-cells').textContent};
            }""")
            assert result == {"first": "10s", "second": "15s", "same": True, "scroll": 80,
                              "forcedSame": True, "newScroll": 0, "task": "b", "spans": 1, "cells": "2 cells"}
        finally:
            browser.close()


def test_workspace_wheel_glass_state_and_rebuilt_metric_tooltips_use_real_dom():
    playwright = pytest.importorskip("playwright.sync_api")
    app_js = (Path(__file__).resolve().parents[1] / "marvis/static/app.js").read_text()
    metrics_js = (Path(__file__).resolve().parents[1] / "marvis/static/js/metric-tables.js").read_text()
    functions = "\n".join(slice_function(app_js, signature) for signature in (
        "function scrollTargetIsWithin(", "function routeWorkspaceWheelToResult(",
        "function setTaskHeroGlassActive(", "function updateTaskHeroGlassState(",
    )) + "\n" + slice_function(metrics_js, "export function attachMetricTooltip(").replace("export ", "", 1)
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.set_content('''<div id="appShell"><div id="resultWorkspace"><div id="taskHero">标题</div>
            <div id="blank">空白</div><div id="progressRail">阶段</div><input id="input">
            <div id="resultScrollContent" style="height:100px;overflow:auto"><div style="height:500px">结果</div></div>
            <div id="metricPreview"><span data-tip="第一项说明">1</span></div></div></div>
            <div id="metricTooltip" hidden></div>''')
            page.evaluate("source => { const $=id=>document.getElementById(id); let taskHeroCanScroll=false,taskHeroGlassActive=null,metricTooltipAttached=false; eval(source + `\n document.addEventListener('wheel',routeWorkspaceWheelToResult,{passive:false});window.refreshGlass=()=>updateTaskHeroGlassState({measureScroll:true}); attachMetricTooltip($('metricPreview'));`); }", functions)
            result = page.evaluate('''() => {
              const scroll=document.getElementById('resultScrollContent');
              const wheel=id=>{const event=new WheelEvent('wheel',{deltaY:30,bubbles:true,cancelable:true});document.getElementById(id).dispatchEvent(event);return event.defaultPrevented;};
              const blank=wheel('blank');window.refreshGlass();const glass=document.getElementById('taskHero').classList.contains('is-glass-active');
              const before=scroll.scrollTop;const rail=wheel('progressRail'),input=wheel('input'),result=wheel('resultScrollContent');
              const after=scroll.scrollTop;scroll.scrollTop=0;window.refreshGlass();
              return {blank,glass,before,after,rail,input,result,cleared:!document.getElementById('resultWorkspace').classList.contains('is-glass-active')};
            }''')
            assert result == {"blank": True, "glass": True, "before": 30, "after": 30,
                              "rail": False, "input": False, "result": False, "cleared": True}
            page.locator("#metricPreview span").hover()
            assert page.locator("#metricTooltip").inner_text() == "第一项说明"
            page.evaluate("document.getElementById('metricPreview').outerHTML='<div id=metricPreview><span data-tip=刷新后说明>2</span></div>'")
            page.locator("#metricPreview span").hover()
            assert page.locator("#metricTooltip").inner_text() == "刷新后说明"
            assert page.locator("#metricTooltip").is_visible()
            page.evaluate("document.dispatchEvent(new Event('scroll'))")
            assert page.locator("#metricTooltip").is_hidden()
        finally:
            browser.close()


def test_sidebar_preferences_settings_navigation_and_model_test_execute_in_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    requests, errors = [], []
    known = {"/api/branding", "/api/tasks", "/api/settings/execution-environment/options",
             "/api/settings/llm", "/api/settings/memory-policy"}
    with _serve_smoke_page(_smoke_html()) as address, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.set_default_timeout(8000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            pending = {}
            page.on("request", lambda request: pending.update({request.url: request.resource_type}))
            page.on("requestfinished", lambda request: pending.pop(request.url, None))
            page.on("requestfailed", lambda request: errors.append(f"request failed {request.url}: {request.failure}"))
            def route(request):
                path = urlparse(request.request.url).path
                if path == "/api/settings/llm/test":
                    requests.append(request.request.post_data_json)
                    request.fulfill(json={"ok": True, "latency_ms": 3, "model_echo": "test-model"})
                elif path in known:
                    request.continue_()
                else:
                    request.fulfill(json={})
            page.route("**/api/**", route)
            page.add_init_script("if(!localStorage.getItem('f13_initialized')){localStorage.clear();localStorage.setItem('marvis_ui_locale','zh');localStorage.setItem('f13_initialized','1');}")
            page.goto(address.rsplit("/", 1)[0] + "/")
            _wait_for_app_boot(page, errors, pending)
            page.click("#sidebarCollapseButton")
            assert page.locator("#appShell").evaluate("element=>element.classList.contains('sidebar-collapsed')")
            page.focus("#sidebarBrandTrigger")
            page.press("#sidebarBrandTrigger", "Enter")
            assert not page.locator("#appShell").evaluate("element=>element.classList.contains('sidebar-collapsed')")
            page.click("#sidebarSettings summary")
            page.select_option("#settingsThemeSelect", "dark")
            assert page.locator("body").get_attribute("data-theme") == "dark"
            page.select_option("#settingsGroupSelect", "created_month")
            page.select_option("#settingsSortSelect", "name_desc")
            assert page.locator("#sidebarSettings").evaluate("element=>element.open")
            page.click("#homeComposerInput")
            assert not page.locator("#sidebarSettings").evaluate("element=>element.open")
            page.reload()
            _wait_for_app_boot(page, errors, pending)
            page.click("#sidebarSettings summary")
            assert page.input_value("#settingsThemeSelect") == "dark"
            assert page.input_value("#settingsGroupSelect") == "created_month"
            assert page.input_value("#settingsSortSelect") == "name_desc"
            page.click("#openGovernanceSettingsButton")
            page.wait_for_selector("#governanceSettingsDialog[open]")
            assert not page.locator("#sidebarSettings").evaluate("element=>element.open")
            page.click('[data-governance-nav="llm"]')
            page.click("#addLLMModelButton")
            page.wait_for_selector("#llmEngineEditDialog[open]")
            page.click("#testLLMEngineConnectionButton")
            assert "请先填写 API 地址与模型名称" in page.locator("#llmEngineTestResult").inner_text()
            assert requests == []
            page.fill("#llmEngineBaseUrl", "https://example.invalid/v1")
            page.fill("#llmEngineModelName", "test-model")
            page.fill("#llmEngineApiKey", "synthetic-test-value")
            page.click("#testLLMEngineConnectionButton")
            page.wait_for_function("document.getElementById('llmEngineTestResult').textContent.includes('连接成功')")
            assert requests == [{"api_base_url": "https://example.invalid/v1", "model_name": "test-model", "api_key": "synthetic-test-value"}]
            assert errors == []
        finally:
            browser.close()


def test_task_hero_toggle_updates_inert_details_and_ignores_controls_and_selection():
    playwright = pytest.importorskip("playwright.sync_api")
    app_js = (Path(__file__).resolve().parents[1] / "marvis/static/app.js").read_text()
    functions = "\n".join(slice_function(app_js, signature) for signature in (
        "function setTaskHeroCollapsed(", "function handleTaskHeroToggle(",
    ))
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.set_content('<section id="taskHero"><span id="title">用户任务标题</span><button id="taskHeroToggle">详情</button><button id="action">操作</button><div id="taskHeroDetails">详情内容</div></section>')
            page.evaluate("source=>{const $=id=>document.getElementById(id);const syncTaskHeroGlassLayout=()=>{};eval(source+`\n$('taskHero').addEventListener('click',handleTaskHeroToggle);`);}", functions)
            page.click("#title")
            assert page.locator("#taskHeroDetails").get_attribute("aria-hidden") == "true"
            assert page.locator("#taskHeroDetails").evaluate("element=>element.inert")
            assert page.locator("#taskHeroToggle").get_attribute("aria-expanded") == "false"
            page.click("#action")
            assert page.locator("#taskHeroDetails").evaluate("element=>element.inert")
            page.click("#taskHeroToggle")
            assert not page.locator("#taskHeroDetails").evaluate("element=>element.inert")
            page.evaluate("const range=document.createRange();range.selectNodeContents(document.getElementById('title'));getSelection().removeAllRanges();getSelection().addRange(range);document.getElementById('title').dispatchEvent(new MouseEvent('click',{bubbles:true}));")
            assert page.locator("#taskHeroDetails").get_attribute("aria-hidden") == "false"
        finally:
            browser.close()
