"""Pet companion (2026-10-14 redesign, batch 4).

The companion is opt-in: with no stored choice there is no pet, and an explicit
pick still shows it. On the welcome screen it keeps its free, draggable
placement; in a task it docks in-flow as the first child of the sidebar footer,
so it covers no task row, footer entry or settings menu, and the docked rect is
never dragged, clamped or saved over the free placement.
"""

import json
import subprocess
from pathlib import Path

from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"
DOCK_SELECTOR = "body:has(.validation-workspace:not(.is-empty)) .pet-companion:not(.dragging)"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _run_node(script: str) -> object:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _bootstrap_script() -> str:
    index_html = _read("index.html")
    marker = 'const pet = document.getElementById("petCompanion");'
    start = index_html.rindex("<script>", 0, index_html.index(marker)) + len("<script>")
    return index_html[start:index_html.index("</script>", start)]


def test_pet_is_off_without_a_stored_choice_and_keeps_explicit_picks():
    data = _run_node(
        """
await import("./marvis/static/js/pet_catalog.js");
const catalog = globalThis.MarvisPetCatalog;
process.stdout.write(JSON.stringify({
  nothing: catalog.resolveStoredPreference(null, false),
  empty: catalog.resolveStoredPreference("", false),
  legacyNone: catalog.resolveStoredPreference("none", false),
  explicitNone: catalog.resolveStoredPreference("none", true),
  picked: catalog.resolveStoredPreference("xiaojiu", false),
  alias: catalog.resolveStoredPreference("danhuang", false),
  // A retired id still means the user once chose a pet.
  retired: catalog.resolveStoredPreference("retired-pet", false),
  defaultId: catalog.defaultId,
}));
"""
    )
    assert data == {
        "nothing": "none",
        "empty": "none",
        "legacyNone": "none",
        "explicitNone": "none",
        "picked": "xiaojiu",
        "alias": "naitang",
        "retired": "auditbot",
        "defaultId": "auditbot",
    }
    assert "export const defaultPetPreference = petCatalog.noneId;" in _read("js/state.js")


def test_boot_script_hides_the_pet_for_a_first_visit_and_shows_a_stored_pick():
    script = _bootstrap_script()
    harness = """
await import("./marvis/static/js/pet_catalog.js");
function run(store) {
  const classes = new Set();
  const pet = {
    dataset: {}, style: { setProperty() {} }, attrs: {},
    classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c) },
    setAttribute(name, value) { this.attrs[name] = value; },
  };
  const sprite = { style: {}, setAttribute() {} };
  const sticker = { dataset: {}, querySelector: () => sprite, replaceChildren() {} };
  globalThis.document = { getElementById: (id) => ({ petCompanion: pet, petSticker: sticker })[id] || null };
  globalThis.window = globalThis;
  globalThis.localStorage = { getItem: (key) => (key in store ? store[key] : null) };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
  globalThis.innerWidth = 1440;
  globalThis.innerHeight = 900;
  BOOT
  return { hidden: classes.has("hidden"), petId: pet.dataset.petId };
}
process.stdout.write(JSON.stringify([run({}), run({ marvis_pet: "auditbot" }), run({ marvis_pet: "none" })]));
""".replace("BOOT", script)
    assert _run_node(harness) == [
        {"hidden": True, "petId": "none"},
        {"hidden": False, "petId": "auditbot"},
        {"hidden": True, "petId": "none"},
    ]


def test_pet_floats_over_the_app_outside_the_sidebar():
    # 2026-10-19 user feedback: inside the sidebar the fixed pet was trapped
    # by the glass and could not move; it now lives after <main>.
    index_html = _read("index.html")
    footer_start = index_html.index('<footer class="sidebar-footer">')
    footer_end = index_html.index("</footer>", footer_start)
    pet = index_html.index('id="petCompanion"')
    assert not footer_start < pet < footer_end
    assert index_html.index("</main>") < pet
    # The welcome-screen bootstrap still runs after the markup it reads.
    assert pet < index_html.index("catalog.resolveStoredPreference(stored, explicitNone)")


def test_task_view_keeps_the_pet_free_and_animated():
    css = _read("css/app-shell.css")
    assert "position: static;" not in css[css.index(".pet-companion {"):css.index(".pet-sticker {")]
    assert DOCK_SELECTOR not in css
    base = css[css.index(".pet-companion {"):]
    base = base[: base.index("}")]
    assert "position: fixed;" in base
    assert "bottom: 28px;" in base
    # Moods change the face only: no shaking, bouncing or nodding.
    for keyframes in ("pet-worry-shake", "pet-playful-bounce", "pet-proud-nod"):
        assert f"animation: {keyframes}" not in css


def test_docked_pet_is_never_dragged_clamped_pinned_or_saved():
    app_js = _read("app.js")
    names = (
        "function petIsDocked(",
        "function petIsPinnedToWorkspaceLeftEdge(",
        "function pinPetToWorkspaceLeftEdge(",
        "function ensurePetWithinViewport(",
        "function startPetDrag(",
    )
    for name in names[2:]:
        body = slice_function(app_js, name)
        assert "if (petIsDocked()) return;" in body, name
    assert "if (petIsDocked()) return false;" in slice_function(app_js, names[1])
    functions = "\n".join(slice_function(app_js, name) for name in names)
    script = "\n".join(
        [
            "let position = 'static';",
            "const saved = [];",
            "const classes = new Set();",
            "const pet = {",
            "  style: { setProperty() { saved.push('style'); } },",
            "  classList: { contains: (c) => classes.has(c), add: (c) => classes.add(c) },",
            "  getBoundingClientRect: () => ({ left: 180, top: 780 }),",
            "};",
            "const workspace = { getBoundingClientRect: () => ({ left: 248, top: 0 }) };",
            "function $(id) { return ({ petCompanion: pet, validationWorkspace: workspace })[id] || null; }",
            "globalThis.getComputedStyle = () => ({ position, getPropertyValue: () => '-2px' });",
            "function petCssPx() { return -2; }",
            "function petWorkspaceOffset() { return -2; }",
            "function clampPetPosition(left, top) { return { left: Math.max(left, 246), top }; }",
            "function applyPetPosition() { saved.push('apply'); }",
            "function savePetPosition() { saved.push('save'); }",
            "let petDragState = null;",
            "globalThis.window = { addEventListener() { saved.push('listen'); } };",
            functions,
            "const event = { button: 0, clientX: 190, clientY: 790, preventDefault() {} };",
            "ensurePetWithinViewport({ persist: true });",
            "pinPetToWorkspaceLeftEdge({ persist: true });",
            "startPetDrag(event);",
            "const docked = { saved: [...saved], dragging: classes.has('dragging'), pinned: petIsPinnedToWorkspaceLeftEdge(), isDocked: petIsDocked() };",
            "position = 'fixed';",
            "ensurePetWithinViewport({ persist: true });",
            "const free = { saved: [...saved], pinned: petIsPinnedToWorkspaceLeftEdge(), isDocked: petIsDocked() };",
            "process.stdout.write(JSON.stringify({ docked, free }));",
        ]
    )
    data = _run_node(script)
    assert data["docked"] == {"saved": [], "dragging": False, "pinned": False, "isDocked": True}
    # Free placement on the welcome screen still clamps and persists.
    assert data["free"] == {"saved": ["apply", "save"], "pinned": True, "isDocked": False}
