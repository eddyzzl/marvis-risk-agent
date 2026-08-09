"""Characterize the ordered stylesheet interface exposed by index.html."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from tests.static_stylesheets import (
    browser_stylesheet_hrefs,
    browser_stylesheet_paths,
    read_browser_stylesheets,
)


STATIC = Path(__file__).resolve().parents[1] / "marvis" / "static"


def test_browser_stylesheet_bundle_is_ordered_unique_and_complete() -> None:
    hrefs = browser_stylesheet_hrefs(STATIC)

    assert tuple(urlparse(href).path for href in hrefs) == (
        "static/styles.css",
        "static/css/app-shell.css",
        "static/css/dialogs-governance.css",
        "static/css/task-shell.css",
        "static/css/execution-workspace.css",
        "static/css/agent-conversation.css",
        "static/css/result-artifacts.css",
        "static/css/welcome.css",
        "static/css/v2-governance.css",
        "static/css/v2-workbench.css",
        "static/css/v2-strategy.css",
        "static/css/v2-workflow-gates.css",
        "static/css/v2-task-setup.css",
        "static/css/validation-batch.css",
        "static/css/validation-batch-create.css",
    )
    assert len(hrefs) == len(set(hrefs))
    assert all("?v=__MARVIS_STATIC_VERSION__" in href for href in hrefs)
    assert all(path.is_file() for path in browser_stylesheet_paths(STATIC))


def test_browser_stylesheet_bundle_exposes_all_existing_semantic_surfaces() -> None:
    css = read_browser_stylesheets(STATIC)

    assert ":root {" in css
    assert ".workspace-welcome" in css
    assert ".governance-settings-dialog" in css
    assert ".candidate-lab" in css
    assert ".feature-binning-option" in css
    assert "#metricPreview" in css
    assert ".validation-batch-panel" in css
    assert ".validation-batch-create-dialog" in css


def test_candidate_lab_field_grids_share_a_shrink_safe_layout() -> None:
    css = read_browser_stylesheets(STATIC)

    field_grid_start = css.index(
        ".candidate-lab-form-grid,\n.candidate-lab-field-grid {"
    )
    field_grid_rule = css[field_grid_start : css.index("}", field_grid_start)]
    assert "display: grid;" in field_grid_rule
    assert "grid-template-columns: repeat(" in field_grid_rule
    assert "minmax(" in field_grid_rule
    assert "min-width: 0;" in field_grid_rule


def test_candidate_lab_workflow_styles_wrap_without_losing_status_semantics() -> None:
    css = read_browser_stylesheets(STATIC)

    stages_start = css.index(".candidate-lab-workflow-stages {")
    stages_rule = css[stages_start : css.index("}", stages_start)]
    assert "display: grid;" in stages_rule
    assert "grid-template-columns: repeat(" in stages_rule
    assert "minmax(" in stages_rule
    assert "min-width: 0;" in stages_rule
    assert "list-style: none;" in stages_rule

    stage_start = css.index(".candidate-lab-workflow-stage {")
    stage_rule = css[stage_start : css.index("}", stage_start)]
    assert "display: grid;" in stage_rule
    assert "min-width: 0;" in stage_rule
    assert ".candidate-lab-workflow-stage[data-status=\"complete\"]" in css
    assert ".candidate-lab-workflow-stage[data-status=\"stale\"]" in css
    assert ".candidate-lab-workflow-stage[data-status=\"missing\"]" in css

    actions_start = css.index(".candidate-lab-form-actions {")
    actions_rule = css[actions_start : css.index("}", actions_start)]
    assert "display: flex;" in actions_rule
    assert "flex-wrap: wrap;" in actions_rule
    assert "gap:" in actions_rule
    assert "min-width: 0;" in actions_rule

    action_button_start = css.index(".candidate-lab-form-actions > .button {")
    action_button_rule = css[
        action_button_start : css.index("}", action_button_start)
    ]
    assert "max-width: 100%;" in action_button_rule
    assert "white-space: normal;" in action_button_rule
    assert "overflow-wrap: anywhere;" in action_button_rule


def test_candidate_lab_stacks_when_its_workspace_column_is_narrow() -> None:
    css = read_browser_stylesheets(STATIC)

    lab_start = css.index(".strategy-candidate-lab {")
    lab_rule = css[lab_start : css.index("}", lab_start)]
    assert "container-type: inline-size;" in lab_rule

    container_start = css.index("@container (max-width: 720px) {")
    container_rule = css[container_start:]
    assert ".candidate-lab-layout" in container_rule
    assert "grid-template-columns: minmax(0, 1fr);" in container_rule
    assert ".candidate-lab-evidence" in container_rule
    assert "border-left: 0;" in container_rule
    assert "border-top:" in container_rule
