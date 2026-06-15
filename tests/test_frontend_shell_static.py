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
    assert '<link id="brandFavicon" rel="icon" type="image/png" href="static/brand/marvis-favicon.png' in index_html
    assert 'id="brandLogo"' in index_html
    assert 'class="brand-mark"' in index_html
    assert 'src="static/brand/marvis-logo.png' in index_html
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
    assert '$("workspaceBrandLogo").src = branding.logoUrl' in branding_js
    assert '$("workspaceBrandLogo").alt = `${branding.platformName} logo`' in branding_js
    assert 'favicon.href = branding.faviconUrl' in branding_js
    assert 'document.documentElement.style.setProperty("--brand-primary", branding.primaryColor)' in branding_js
    assert 'document.documentElement.style.setProperty("--brand-primary-hover"' in branding_js
    assert "loadBranding();" in app_js


def test_app_entry_is_split_into_frontend_modules():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")

    assert '<script type="module" src="static/app.js?v=__MARVIS_STATIC_VERSION__"></script>' in index_html
    for module_name in [
        "api.js",
        "branding.js",
        "dialogs.js",
        "polling.js",
        "render-agent.js",
        "render-metrics.js",
        "state.js",
        "ui-utils.js",
    ]:
        assert (STATIC_DIR / "js" / module_name).exists()

    assert 'from "./js/api.js"' in app_js
    assert 'from "./js/branding.js"' in app_js
    assert 'from "./js/dialogs.js"' in app_js
    assert 'from "./js/polling.js"' in app_js
    assert 'from "./js/render-agent.js"' in app_js
    assert 'from "./js/render-metrics.js"' in app_js
    assert 'from "./js/state.js"' in app_js
    assert 'from "./js/ui-utils.js"' in app_js


def test_unselected_workspace_shows_centered_welcome_only():
    index_html = _read_static("index.html")
    app_js = _read_static("app.js")
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
    assert "早上好，开启活力一天" in index_html
    assert "我来帮您完成信贷风控工作" in index_html
    assert "我来帮您完成模型验证工作" not in index_html
    assert "欢迎，我来帮您完成信贷风控工作" not in index_html
    welcome_start = index_html.index('id="workspaceWelcome"')
    welcome_end = index_html.index('<div class="workspace-body">', welcome_start)
    welcome_markup = index_html[welcome_start:welcome_end]
    assert "创建任务或从左侧选择已有任务" not in welcome_markup
    assert 'id="welcomeTaskCards"' in welcome_markup
    assert 'id="welcomeModelDevelopmentCard"' in welcome_markup
    assert 'id="welcomeModelValidationCard"' in welcome_markup
    assert 'id="welcomeStrategyDevelopmentCard"' in welcome_markup
    assert "模型开发" in welcome_markup
    assert "模型验证" in welcome_markup
    assert "一致性、稳定性、效果验证，压力测试和编写报告" in welcome_markup
    assert "材料扫描、Notebook 复现与验证报告生成" not in welcome_markup
    assert "策略开发" in welcome_markup
    assert 'data-task-kind="model_validation"' in welcome_markup
    assert 'disabled aria-disabled="true"' in welcome_markup
    assert "暂未开放" in welcome_markup
    assert "validationWorkspace" in app_js
    assert 'classList.toggle("is-empty", !selectedTask)' in app_js
    assert "function openTaskTypeWelcome" in app_js
    welcome_entry_start = app_js.index("function openTaskTypeWelcome")
    welcome_entry_end = app_js.index("function closeTaskDialog", welcome_entry_start)
    welcome_entry = app_js[welcome_entry_start:welcome_entry_end]
    assert "deselectCurrentTask();" in welcome_entry
    assert "rememberSelectedTaskId(null);" in welcome_entry
    assert "showModal" not in welcome_entry
    assert '$("createTaskOpenButton").onclick = openTaskTypeWelcome;' in app_js
    assert '$("welcomeModelValidationCard").onclick = openTaskDialog;' in app_js
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
    cards_rule = _css_rule(welcome_css, ".welcome-task-cards")
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in cards_rule
    assert "max-width: 760px" in cards_rule
    assert "margin-top: clamp(24px, 4vh, 36px)" in cards_rule
    card_rule = _css_rule(welcome_css, ".welcome-task-card")
    assert "border-radius: var(--radius)" in card_rule
    assert "text-align: left" in card_rule


def test_empty_workspace_greeting_changes_by_local_time():
    app_js = _read_static("app.js")

    assert "function workspaceGreetingForHour(hour)" in app_js
    assert "function updateWorkspaceGreeting(now = new Date())" in app_js
    assert "workspaceGreetingForHour(now.getHours())" in app_js
    assert "$(\"workspaceGreetingText\").textContent = greeting" in app_js
    assert "${greeting}，我来帮您完成信贷风控工作" not in app_js
    assert "我来帮您完成模型验证工作" not in app_js
    assert "updateWorkspaceGreeting();" in app_js

    assert "早上好，开启活力一天" in app_js
    assert "上午好，记得多补充水份" in app_js
    assert "下午好，记得起来活动一下" in app_js
    assert "晚上好，工作辛苦了" in app_js

    greeting_start = app_js.index("function workspaceGreetingForHour(hour)")
    greeting_end = app_js.index("function updateWorkspaceGreeting", greeting_start)
    greeting_logic = app_js[greeting_start:greeting_end]
    assert "hour >= 5 && hour < 9" in greeting_logic
    assert "hour >= 9 && hour < 12" in greeting_logic
    assert "hour >= 12 && hour < 18" in greeting_logic
    assert "return \"晚上好，工作辛苦了\"" in greeting_logic


def test_workspace_greeting_logic_runs_under_node():
    app_js = _read_static("app.js")
    start = app_js.index("function workspaceGreetingForHour(hour)")
    end = app_js.index("function updateWorkspaceGreeting", start)
    script = "\n".join(
        [
            app_js[start:end],
            "console.log([",
            "  workspaceGreetingForHour(7),",
            "  workspaceGreetingForHour(10),",
            "  workspaceGreetingForHour(15),",
            "  workspaceGreetingForHour(22),",
            "].join('|'));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "早上好，开启活力一天|上午好，记得多补充水份|下午好，记得起来活动一下|晚上好，工作辛苦了"
