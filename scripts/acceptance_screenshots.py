"""Browser acceptance screenshots: main screens x 1440/1600/1920 x light/dark.

Starts the real app on a fresh temporary workspace (no model configured, no
outbound calls), creates one demo modeling task through the public API, and
captures home, task workbench and system settings with Playwright Chromium.
Output is developer evidence for the user's own review, not a sign-off.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marvis.orchestrator.eval.perf_gate import _request, _Server  # noqa: E402

WIDTHS = (1440, 1600, 1920)
THEMES = ("light", "dark")
HEIGHT = 900


def _overflow(page):
    return page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")


def capture(port: int, out: Path) -> dict:
    from playwright.sync_api import sync_playwright

    screens = out / "screens"
    screens.mkdir(parents=True, exist_ok=True)
    index = []
    with tempfile.TemporaryDirectory(prefix="marvis-shots-") as tmp:
        tmp = Path(tmp)
        source = tmp / "workspace" / "source"  # material roots must sit under the workspace
        source.mkdir(parents=True)
        server = _Server(tmp / "workspace", port, out / "server.log")
        if server.start(120) is None:
            raise RuntimeError("acceptance_server_failed_to_start")
        try:
            status, _, _ = _request(server.base + "/api/tasks", method="POST", body={
                "task_type": "modeling", "model_name": "验收演示模型", "validator": "acceptance",
                "source_dir": str(source), "run_mode": "manual"})
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                for theme in THEMES:
                    for width in WIDTHS:
                        context = browser.new_context(viewport={"width": width, "height": HEIGHT},
                                                      color_scheme=theme, locale="zh-CN")
                        context.add_init_script(f"try{{localStorage.setItem('marvis_theme','{theme}')}}catch(e){{}}")
                        page = context.new_page()
                        errors = []
                        page.on("console", lambda msg, errors=errors: msg.type == "error" and errors.append(msg.text[:200]))
                        page.on("pageerror", lambda exc, errors=errors: errors.append(str(exc)[:200]))
                        page.goto(server.base + "/", wait_until="networkidle")
                        page.wait_for_timeout(800)

                        def shot(name, errors=errors, page=page, theme=theme, width=width):
                            path = screens / f"{name}-{width}-{theme}.png"
                            page.screenshot(path=str(path))
                            index.append({"screen": name, "width": width, "theme": theme,
                                          "file": f"screens/{path.name}",
                                          "body_theme": page.evaluate("document.body.dataset.theme || null"),
                                          "horizontal_overflow": _overflow(page),
                                          "console_errors": list(errors)})

                        shot("home")
                        try:
                            page.locator("#taskList [data-task-id], #taskList [role=listitem]").first.click(timeout=5000)
                            page.wait_for_timeout(1200)
                            shot("task-workbench")
                        except Exception as exc:  # recorded, not hidden
                            index.append({"screen": "task-workbench", "width": width, "theme": theme,
                                          "error": type(exc).__name__})
                        try:
                            page.locator("#sidebarSettings > summary").click(timeout=5000)
                            page.locator("#openGovernanceSettingsButton").click(timeout=5000)
                            page.wait_for_timeout(800)
                            shot("system-settings")
                        except Exception as exc:
                            index.append({"screen": "system-settings", "width": width, "theme": theme,
                                          "error": type(exc).__name__})
                        context.close()
                browser.close()
        finally:
            server.stop()
    report = {"schema": "marvis.acceptance-screenshots.v1", "label": "开发者截图，供人工核对，非验收签字",
              "demo_task_created": status == 200, "widths": WIDTHS, "themes": THEMES, "height": HEIGHT,
              "shots": index}
    (out / "index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or ROOT / ".omc/acceptance" / date.today().isoformat() / "browser"
    report = capture(args.port, out)
    taken = [s for s in report["shots"] if "file" in s]
    print(f"{len(taken)} screenshots, {len(report['shots']) - len(taken)} failures -> {out}")
    return 0 if len(taken) == len(report["shots"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
