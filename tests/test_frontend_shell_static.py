import json
import subprocess
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _css_rule(css: str, selector: str) -> str:
    start = css.index(f"{selector} {{")
    end = css.index("}", start)
    return css[start:end]


def test_browser_chrome_uses_public_default_branding():
    index_html = _read_static("index.html")

    assert "<title>MARVIS-全能风控智能体</title>" in index_html
    assert '<meta name="color-scheme" content="light dark" />' in index_html
    assert '<meta id="appThemeColor" name="theme-color" content="#ffffff" />' in index_html
    assert '<meta name="apple-mobile-web-app-capable" content="yes" />' in index_html
    assert '<meta name="apple-mobile-web-app-title" content="MARVIS" />' in index_html
    assert '<link id="brandFavicon" rel="icon" type="image/png" media="(prefers-color-scheme: light)" href="static/brand/marvis-favicon.png' in index_html
    assert '<link id="brandFaviconDark" rel="icon" type="image/png" media="(prefers-color-scheme: dark)" href="static/brand/marvis-favicon-dark.png' in index_html
    assert '<link id="brandAppleTouchIcon" rel="apple-touch-icon" media="(prefers-color-scheme: light)" href="static/brand/marvis-apple-touch-icon.png' in index_html
    assert '<link id="brandAppleTouchIconDark" rel="apple-touch-icon" media="(prefers-color-scheme: dark)" href="static/brand/marvis-apple-touch-icon-dark.png' in index_html
    assert '<link rel="manifest" href="static/manifest.webmanifest" />' in index_html
    assert 'id="brandLogo"' in index_html
    assert 'class="brand-mark"' in index_html
    brand_logo_start = index_html.index('id="brandLogo"')
    brand_logo_end = index_html.index("/>", brand_logo_start)
    brand_logo_markup = index_html[brand_logo_start:brand_logo_end]
    assert 'src="static/brand/marvis-logo.png' in brand_logo_markup
    assert 'id="workspaceBrandLogo"' in index_html
    workspace_logo_start = index_html.index('id="workspaceBrandLogo"')
    workspace_logo_end = index_html.index("/>", workspace_logo_start)
    workspace_logo_markup = index_html[workspace_logo_start:workspace_logo_end]
    assert 'src="static/brand/marvis-workspace-logo.png' in workspace_logo_markup
    assert 'id="platformName"' in index_html
    assert "MARVIS-全能风控智能体" in index_html
    assert "private-logo.svg" not in index_html
    assert 'href="data:,"' not in index_html


def test_runtime_branding_hooks_exist():
    app_js = _read_static("app.js")
    branding_js = _read_static("js/branding.js")
    state_js = _read_static("js/state.js")

    assert "export const defaultBranding" in state_js
    assert 'import { applyBranding, normalizeBranding } from "./js/branding.js";' in app_js
    assert 'fetch("api/branding")' in app_js
    assert "async function loadBranding()" in app_js
    assert "function applyBranding(branding)" not in app_js
    assert "export function normalizeBranding" in branding_js
    assert "export function applyBranding" in branding_js
    assert "document.title = branding.browserTitle" in branding_js
    assert '$("platformName").textContent = branding.platformName' in branding_js
    assert '$("brandLogo").src = branding.logoUrl' in branding_js
    assert '$("brandLogo").alt = `${branding.platformName} logo`' in branding_js
    assert '$("workspaceBrandLogo").src = branding.workspaceLogoUrl || branding.logoUrl' in branding_js
    assert '$("workspaceBrandLogo").alt = `${branding.platformName} logo`' in branding_js
    assert 'favicon.href = branding.faviconUrl' in branding_js
    assert 'const darkFavicon = $("brandFaviconDark")' in branding_js
    assert 'const appleTouchIcon = $("brandAppleTouchIcon")' in branding_js
    assert 'const darkAppleTouchIcon = $("brandAppleTouchIconDark")' in branding_js
    assert 'document.documentElement.style.setProperty("--brand-primary", branding.primaryColor)' in branding_js
    assert 'document.documentElement.style.setProperty("--brand-primary-hover"' in branding_js
    assert "loadBranding();" in app_js


def test_app_entry_is_split_into_frontend_modules():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")

    assert '<script type="module" src="static/app.js?v=__MARVIS_STATIC_VERSION__"></script>' in index_html
    for module_name in [
        "api.js",
        "agent-memory-panel.js",
        "branding.js",
        "dialogs.js",
        "draft-tools-panel.js",
        "polling.js",
        "render-agent.js",
        "render-metrics.js",
        "state.js",
        "theme.js",
        "ui-utils.js",
    ]:
        assert (STATIC_DIR / "js" / module_name).exists()

    assert 'from "./js/api.js"' in app_js
    assert 'from "./js/agent-memory-panel.js"' in app_js
    assert 'from "./js/branding.js"' in app_js
    assert 'from "./js/dialogs.js"' in app_js
    assert 'from "./js/draft-tools-panel.js"' in app_js
    assert 'from "./js/polling.js"' in app_js
    assert 'from "./js/render-agent.js"' in app_js
    assert 'from "./js/render-metrics.js"' in app_js
    assert 'from "./js/state.js"' in app_js
    assert 'from "./js/theme.js"' in app_js
    assert 'from "./js/ui-utils.js"' in app_js


def test_unselected_workspace_shows_centered_welcome_only():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    workspace_view_js = _read_static("js/task-workspace-view.js")
    styles_css = _read_static("styles.css")
    welcome_css = _read_static("css/welcome.css")

    assert 'id="workspaceWelcome"' in index_html
    assert 'id="workspaceBrandLogo"' in index_html
    assert 'class="workspace-brand-logo"' in index_html
    assert 'id="workspaceGreetingTitle"' in index_html
    assert 'id="workspaceGreetingText"' in index_html
    assert 'id="workspaceGreetingCursor"' not in index_html
    assert 'class="workspace-greeting-cursor"' not in index_html
    assert 'class="workspace-greeting-nowrap"' in index_html
    # Greeting is a short time-of-day word ("早上好") + the fixed professional suffix,
    # not a long cutesy wellness line.
    assert ">早上好</span>" in index_html
    assert "我来帮您完成信贷风控工作" in index_html
    assert "我来帮您完成模型验证工作" not in index_html
    assert "欢迎，我来帮您完成信贷风控工作" not in index_html
    welcome_start = index_html.index('id="workspaceWelcome"')
    welcome_end = index_html.index('<div class="workspace-body">', welcome_start)
    welcome_markup = index_html[welcome_start:welcome_end]
    assert "创建任务或从左侧选择已有任务" not in welcome_markup
    assert "选择一个任务类型，补充材料和目标后进入 Agent 工作流。" not in welcome_markup
    assert 'class="workspace-welcome-copy"' not in welcome_markup
    assert 'id="welcomeTaskCards"' in welcome_markup
    assert 'id="welcomeDataJoinCard"' in welcome_markup
    assert 'id="welcomeVintageAnalysisCard"' in welcome_markup
    assert 'id="welcomeModelDevelopmentCard"' in welcome_markup
    assert 'id="welcomeModelValidationCard"' in welcome_markup
    assert 'id="welcomeStrategyDevelopmentCard"' in welcome_markup
    assert "自动识别主键，关联各种XY数据，诊断数据情况" in welcome_markup
    assert "上传多表、识别主键、诊断膨胀和确认 join" not in welcome_markup
    assert "模型开发" in welcome_markup
    vintage_card_start = welcome_markup.index('id="welcomeVintageAnalysisCard"')
    vintage_card_end = welcome_markup.index("</button>", vintage_card_start)
    vintage_card_markup = welcome_markup[vintage_card_start:vintage_card_end]
    assert "vintage-calendar-binding" in vintage_card_markup
    assert 'class="ln vintage-calendar-binding"' in vintage_card_markup
    assert "M7.2 4.8v2.8M16.8 4.8v2.8" in vintage_card_markup
    assert "cut-line" not in vintage_card_markup
    model_card_start = welcome_markup.index('id="welcomeModelDevelopmentCard"')
    model_card_end = welcome_markup.index("</button>", model_card_start)
    model_card_markup = welcome_markup[model_card_start:model_card_end]
    # modeling icon is now a layered terminal window (T1): window body + a
    # light title bar with three dots + a centered ">_" prompt.
    assert "model-code-mark" not in model_card_markup
    assert "model-code-slash" not in model_card_markup
    assert "M12 2.4 19.9 6.95v10.1L12 21.6 4.1 17.05V6.95Z" not in model_card_markup
    assert 'width="18.8" height="14.8"' in model_card_markup
    assert "M2.6 8 V7 Q2.6 4.6 5 4.6 H19 Q21.4 4.6 21.4 7 V8 Z" in model_card_markup
    assert "M8.2 11.2 11 13.8 8.2 16.4" in model_card_markup
    assert 'class="mid"' in model_card_markup
    assert "model-training-curve" not in model_card_markup
    assert 'class="back"' not in model_card_markup
    assert 'class="welcome-task-sheen"' in model_card_markup
    assert 'clip-path="url(#welcomeModelingIconClip)"' in model_card_markup
    assert '<use href="#welcomeModelingIconGlyph">' not in model_card_markup
    assert "L6.5 12" not in model_card_markup
    assert "L17.5 12" not in model_card_markup
    assert "模型验证" in welcome_markup
    assert "可复现性、稳定性、效果验证，压力测试" in welcome_markup
    assert "压力测试和编写报告" not in welcome_markup
    assert "一致性、稳定性、效果验证，压力测试和编写报告" not in welcome_markup
    assert "材料扫描、Notebook 复现与验证报告生成" not in welcome_markup
    assert "策略开发" in welcome_markup
    assert "数据处理" in welcome_markup
    assert "数据拼接" not in welcome_markup
    assert "风险分析" in welcome_markup
    assert "资产Vintage&滚动率分析、FPD、入催回收率分析" in welcome_markup
    assert "Vintage、FPD、营利性测算" not in welcome_markup
    assert "Vintage分析" not in welcome_markup
    assert "Vintage 分析" not in welcome_markup
    expected_card_titles = ["数据处理", "特征分析", "风险分析", "模型开发", "模型验证", "策略开发"]
    title_offsets = [welcome_markup.index(f"<strong>{title}</strong>") for title in expected_card_titles]
    assert title_offsets == sorted(title_offsets)
    assert 'data-task-kind="validation"' in welcome_markup
    assert 'data-task-kind="modeling"' in welcome_markup
    assert 'data-task-kind="strategy"' in welcome_markup
    assert 'disabled aria-disabled="true"' not in welcome_markup
    for card_id in ("welcomeVintageAnalysisCard", "welcomeStrategyDevelopmentCard"):
        card_start = welcome_markup.index(f'id="{card_id}"')
        card_end = welcome_markup.index("</button>", card_start)
        card_markup = welcome_markup[card_start:card_end]
        assert 'class="welcome-task-card available"' in card_markup
        assert 'aria-describedby="welcomeComingSoonHint"' not in card_markup
    assert "该任务暂未开放,点击后会显示敬请期待提示。" not in welcome_markup
    assert "validationWorkspace" in app_js
    assert "const hasTaskContext = Boolean(selectedTask || selectedTaskId);" in workspace_view_js
    assert 'classList.toggle("is-empty", !hasTaskContext)' in workspace_view_js
    assert "function openTaskTypeWelcome" in app_js
    welcome_entry_start = app_js.index("function openTaskTypeWelcome")
    welcome_entry_end = app_js.index("function closeTaskDialog", welcome_entry_start)
    welcome_entry = app_js[welcome_entry_start:welcome_entry_end]
    assert "deselectCurrentTask();" in welcome_entry
    assert "rememberSelectedTaskId(null);" in welcome_entry
    assert "showModal" not in welcome_entry
    assert '$("createTaskOpenButton").onclick = openTaskTypeWelcome;' in app_js
    assert '$("welcomeTaskCards").onclick = openTaskDialogFromCard;' in app_js
    assert '$("createTaskOpenButton").onclick = openTaskDialog;' not in app_js
    assert 'class="validation-workspace region is-empty"' in index_html

    assert 'href="static/styles.css?v=__MARVIS_STATIC_VERSION__"' in index_html
    assert 'href="static/css/welcome.css?v=__MARVIS_STATIC_VERSION__"' in index_html
    assert ".workspace-welcome" not in styles_css
    assert ".workspace-welcome" in welcome_css
    assert ".workspace-brand-logo" in welcome_css
    assert ".welcome-task-cards" in welcome_css
    assert ".welcome-task-card" in welcome_css
    assert ".workspace-greeting-nowrap" in welcome_css
    assert "white-space: nowrap" in welcome_css
    assert ".workspace-greeting-cursor" not in styles_css
    assert "workspace-greeting-cursor-blink" not in welcome_css
    title_start = welcome_css.index(".workspace-welcome h2 {")
    title_end = welcome_css.index("}", title_start)
    title_rule = welcome_css[title_start:title_end]
    assert "white-space: nowrap" in title_rule
    assert "max-width: none" in title_rule
    assert "word-break: keep-all" in title_rule
    logo_start = welcome_css.index(".workspace-brand-logo {")
    logo_end = welcome_css.index("}", logo_start)
    logo_rule = welcome_css[logo_start:logo_end]
    assert "width: 128px" in logo_rule
    assert "height: 128px" in logo_rule
    assert "object-fit: contain" in logo_rule
    assert "margin: 0 0 28px" in logo_rule
    assert "display: none" in welcome_css
    assert ".validation-workspace.is-empty .workspace-welcome" in welcome_css
    assert "display: grid" in welcome_css
    assert ".validation-workspace.is-empty .workspace-head" in styles_css
    assert ".validation-workspace.is-empty .workspace-body" in styles_css
    root_rule = _css_rule(styles_css, ":root")
    assert "--radius: 16px" in root_rule
    assert "--radius-sm: 16px" in root_rule
    assert "--radius-md: 16px" in root_rule
    assert "--radius-lg: 16px" in root_rule
    assert "--radius-control: 10px" in root_rule
    assert "--surface-subtle:" in root_rule
    assert "--border-color:" in root_rule
    assert "--tone-modeling:" in root_rule
    assert "--tone-strategy:" in root_rule
    cards_rule = _css_rule(welcome_css, ".welcome-task-cards")
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in cards_rule
    assert "max-width: 1040px" in cards_rule
    assert "margin-top: clamp(24px, 4vh, 36px)" in cards_rule
    card_rule = _css_rule(welcome_css, ".welcome-task-card")
    assert "border-radius: var(--radius)" in card_rule
    assert "text-align: left" in card_rule
    icon_rule = _css_rule(welcome_css, ".welcome-task-icon")
    assert "overflow: hidden" not in icon_rule
    assert "border-radius: 10px" not in icon_rule
    legacy_icon_sheen_rule = _css_rule(welcome_css, ".welcome-task-icon svg .icon-sheen")
    assert "display: none" in legacy_icon_sheen_rule
    icon_sheen_rule = _css_rule(welcome_css, ".welcome-task-sheen")
    assert "-webkit-mask-image: var(--welcome-icon-mask-image)" in icon_sheen_rule
    assert "mask-image: var(--welcome-icon-mask-image)" in icon_sheen_rule
    icon_sheen_beam_rule = _css_rule(welcome_css, ".welcome-task-sheen::before")
    assert "left: -18px" in icon_sheen_beam_rule
    assert "transform: translateX(0) skewX(-18deg)" in icon_sheen_beam_rule
    assert "will-change: transform, opacity" in icon_sheen_beam_rule
    assert "rgba(255, 255, 255, 0.98)" in icon_sheen_beam_rule
    assert "welcome-icon-sheen 560ms ease-out" in welcome_css
    assert "background-position: 140% 0" not in welcome_css
    expected_icon_sizes = {
        "feature_analysis": "41px",
        "data_join": "40px",
        "modeling": "36px",
        "validation": "39px",
        "strategy": "43px",
        "vintage": "41px",
    }
    for task_kind, size in expected_icon_sizes.items():
        icon_svg_rule = _css_rule(welcome_css, f'.welcome-task-card[data-task-kind="{task_kind}"] .welcome-task-icon svg')
        assert f"width: {size}" in icon_svg_rule
        assert f"height: {size}" in icon_svg_rule
        icon_mask_rule = _css_rule(welcome_css, f'.welcome-task-card[data-task-kind="{task_kind}"] .welcome-task-icon')
        assert f"--welcome-icon-mask-size: {size} {size}" in icon_mask_rule
    model_icon_rule = _css_rule(welcome_css, '.welcome-task-card[data-task-kind="modeling"] .welcome-task-icon')
    assert "--welcome-icon-mask-image" in model_icon_rule
    assert "width%3D%2718.8%27%20height%3D%2714.8%27" in model_icon_rule
    vintage_icon_rule = _css_rule(welcome_css, '.welcome-task-card[data-task-kind="vintage"] .welcome-task-icon')
    assert "M7.2%204.8v2.8M16.8%204.8v2.8" in vintage_icon_rule
    assert ".welcome-task-icon svg .vintage-calendar-binding" in welcome_css
    assert ".welcome-task-icon svg .cut-line" not in welcome_css
    assert ".welcome-task-icon svg .model-code-mark" not in welcome_css
    assert ".welcome-task-icon svg .model-code-slash" not in welcome_css
    assert ".welcome-task-icon svg .model-training-curve" not in welcome_css
    # the validation seal check uses a thinner carved stroke
    assert ".welcome-task-icon svg .cst" in welcome_css
    cst_rule = _css_rule(welcome_css, ".welcome-task-icon svg .cst")
    assert "stroke-width: 1.6" in cst_rule
    assert "mix-blend-mode" not in welcome_css
    assert "body[data-theme=\"dark\"] .welcome-task-icon" not in welcome_css
    assert "body[data-theme=\"dark\"] .welcome-task-card.available:hover" not in welcome_css
    assert "--tone: var(--tone-modeling)" in welcome_css
    assert "--tone: var(--tone-vintage)" in welcome_css


def test_empty_workspace_greeting_changes_by_local_time():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
    workspace_state_js = _read_static("js/task-workspace-state.js")
    workspace_view_js = _read_static("js/task-workspace-view.js")

    assert "export function workspaceGreetingForHour(hour)" in workspace_state_js
    assert 'from "./task-workspace-state.js"' in workspace_view_js
    assert "export function updateWorkspaceGreeting" in workspace_view_js
    assert "function updateWorkspaceGreeting(now = new Date())" in app_js
    assert "updateWorkspaceGreetingView({ now, getElementById: $ });" in app_js
    assert "workspaceGreetingForHour(now.getHours())" in workspace_view_js
    assert 'get("workspaceGreetingText")' in workspace_view_js
    assert "${greeting}，我来帮您完成信贷风控工作" not in app_js
    assert "我来帮您完成模型验证工作" not in app_js
    assert "updateWorkspaceGreeting();" in app_js

    assert 'return "早上好"' in workspace_state_js
    assert 'return "上午好"' in workspace_state_js
    assert 'return "下午好"' in workspace_state_js
    assert 'return "晚上好"' in workspace_state_js
    assert 'document.getElementById("workspaceGreetingText")' in index_html
    assert 'new Date().getHours()' in index_html
    assert index_html.index('id="workspaceGreetingText"') < index_html.index('new Date().getHours()')
    assert index_html.index('new Date().getHours()') < index_html.index('id="welcomeTaskCards"')

    greeting_start = workspace_state_js.index("export function workspaceGreetingForHour(hour)")
    greeting_logic = workspace_state_js[greeting_start:]
    assert "hour >= 5 && hour < 9" in greeting_logic
    assert "hour >= 9 && hour < 12" in greeting_logic
    assert "hour >= 12 && hour < 18" in greeting_logic
    assert 'return "晚上好"' in greeting_logic


def test_workspace_greeting_logic_runs_under_node():
    module_url = (STATIC_DIR / "js" / "task-workspace-state.js").as_uri()
    script = "\n".join(
        [
            f"import {{ workspaceGreetingForHour }} from {module_url!r};",
            "console.log([",
            "  workspaceGreetingForHour(7),",
            "  workspaceGreetingForHour(10),",
            "  workspaceGreetingForHour(15),",
            "  workspaceGreetingForHour(22),",
            "].join('|'));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "早上好|上午好|下午好|晚上好"


def test_task_workspace_view_shell_logic_runs_under_node():
    module_url = (STATIC_DIR / "js" / "task-workspace-view.js").as_uri()
    script = "\n".join(
        [
            f"import {{ renderCurrentTaskWorkspace }} from {module_url!r};",
            "function node() {",
            "  return {",
            "    textContent: '',",
            "    classList: {",
            "      values: new Set(),",
            "      toggle(cls, on) { if (on) this.values.add(cls); else this.values.delete(cls); },",
            "      contains(cls) { return this.values.has(cls); },",
            "    },",
            "  };",
            "}",
            "const nodes = {",
            "  validationWorkspace: node(),",
            "  currentTaskTitle: node(),",
            "  currentTaskSubtitle: node(),",
            "  workspaceGreetingText: node(),",
            "};",
            "let snapshots = 0;",
            "const statuses = [];",
            "let syncs = 0;",
            "const common = {",
            "  getElementById: (id) => nodes[id],",
            "  renderTaskSnapshot: () => { snapshots += 1; },",
            "  setActionStatus: (...args) => statuses.push(args),",
            "  syncTaskHeroGlassLayout: () => { syncs += 1; },",
            "  requestFrame: (cb) => cb(),",
            "};",
            "renderCurrentTaskWorkspace({ ...common, selectedTask: null, selectedTaskId: '', updateGreeting: ({ getElementById }) => { getElementById('workspaceGreetingText').textContent = '上午好'; } });",
            "const empty = {",
            "  title: nodes.currentTaskTitle.textContent,",
            "  subtitle: nodes.currentTaskSubtitle.textContent,",
            "  greeting: nodes.workspaceGreetingText.textContent,",
            "  isEmpty: nodes.validationWorkspace.classList.contains('is-empty'),",
            "};",
            "renderCurrentTaskWorkspace({ ...common, selectedTask: { id: 'task-1', model_name: 'Demo' }, selectedTaskId: 'task-1', taskDisplayName: (task) => task.model_name, taskActionStatusSnapshot: () => ({ message: '就绪', kind: 'success' }) });",
            "const selected = {",
            "  title: nodes.currentTaskTitle.textContent,",
            "  subtitle: nodes.currentTaskSubtitle.textContent,",
            "  isEmpty: nodes.validationWorkspace.classList.contains('is-empty'),",
            "  latestStatus: statuses.at(-1),",
            "  snapshots,",
            "  syncs,",
            "};",
            "process.stdout.write(JSON.stringify({ empty, selected }));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["empty"] == {
        "title": "验证任务",
        "subtitle": "创建任务或从左侧选择已有任务",
        "greeting": "上午好",
        "isEmpty": True,
    }
    assert payload["selected"]["title"] == "Demo"
    assert payload["selected"]["subtitle"] == ""
    assert payload["selected"]["isEmpty"] is False
    assert payload["selected"]["latestStatus"] == ["就绪", "success"]
    assert payload["selected"]["snapshots"] == 2
    assert payload["selected"]["syncs"] == 2


def test_task_workspace_snapshot_renderer_runs_under_node():
    module_url = (STATIC_DIR / "js" / "task-workspace-view.js").as_uri()
    script = "\n".join(
        [
            f"import {{ renderTaskSnapshot }} from {module_url!r};",
            "const snapshot = { className: '', textContent: '', innerHTML: '' };",
            "const getElementById = (id) => id === 'taskSnapshot' ? snapshot : null;",
            "renderTaskSnapshot({ getElementById });",
            "const empty = { className: snapshot.className, textContent: snapshot.textContent, innerHTML: snapshot.innerHTML };",
            "renderTaskSnapshot({",
            "  getElementById,",
            "  selectedTask: { run_mode: 'agent', source_dir: '/tmp/a&b', task_type: 'modeling' },",
            "  taskTypeLabel: () => '模型<开发>',",
            "  taskKindIconHtml: () => '<svg class=\"kind\"></svg>',",
            "  runModeLabel: () => 'Agent 模式',",
            "});",
            "const selected = { className: snapshot.className, textContent: snapshot.textContent, innerHTML: snapshot.innerHTML };",
            "process.stdout.write(JSON.stringify({ empty, selected }));",
        ]
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["empty"] == {
        "className": "workspace-task-meta empty",
        "textContent": "核心任务信息",
        "innerHTML": "",
    }
    assert payload["selected"]["className"] == "workspace-task-meta"
    assert "模型&lt;开发&gt;" in payload["selected"]["innerHTML"]
    assert "Agent 模式" in payload["selected"]["innerHTML"]
    assert 'class="task-snapshot-copy"' in payload["selected"]["innerHTML"]
    assert 'data-copy="/tmp/a&amp;b"' in payload["selected"]["innerHTML"]
