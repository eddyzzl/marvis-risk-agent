import { escapeHtml } from "../ui-utils.js";
import {
  listSkills as listSkillsApi,
  reloadSkills as reloadSkillsApi,
  validateSkill as validateSkillApi,
} from "./api_v2.js";

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function rejectedEntries(report) {
  return (report?.rejected || []).map((item) => {
    if (Array.isArray(item)) {
      return [item[0], item[1] || []];
    }
    return [item.id || item.name || "unknown", item.problems || []];
  });
}

export function skillRowHtml(id, status, problems = []) {
  const problemList = problems.length
    ? `<ul class="skill-problems">${problems.map((problem) => `<li>${escapeHtml(problem)}</li>`).join("")}</ul>`
    : "";
  return `<section class="skill skill-${escapeHtml(status)}">
    <strong>${escapeHtml(id)}</strong>
    <span class="skill-badge ${escapeHtml(status)}">${escapeHtml(status)}</span>
    ${problemList}
  </section>`;
}

export function skillValidationResultHtml(result = {}) {
  const problems = result.problems || [];
  if (!problems.length) {
    return '<div class="skill-validation valid">Valid skill</div>';
  }
  const items = problems
    .map((problem) => `<li>${escapeHtml(problem)}</li>`)
    .join("");
  return `<ul class="skill-validation problems">${items}</ul>`;
}

export function skillManagerHtml(report = {}) {
  const active = (report.active || []).map((id) => skillRowHtml(id, "active")).join("");
  const disabled = (report.disabled || []).map((id) => skillRowHtml(id, "disabled")).join("");
  const rejected = rejectedEntries(report)
    .map(([id, problems]) => skillRowHtml(id, "rejected", problems))
    .join("");
  return `<section class="skill-manager">
    <button id="reloadSkills" type="button" data-reload-skills>Reload skills</button>
    <div class="skill-list">
      ${active}
      ${disabled}
      ${rejected}
    </div>
    <label class="skill-validator">
      <textarea data-validate-skill spellcheck="false"></textarea>
    </label>
    <div data-skill-validation-result></div>
  </section>`;
}

export function renderSkillManagerShell(container, report = {}) {
  if (!container) {
    throw new Error("renderSkillManagerShell requires a container");
  }
  if (container.dataset) {
    container.dataset.v2SkillManager = "true";
  }
  container.innerHTML = skillManagerHtml(report);
  return () => {};
}

export async function renderSkillManager(container, deps = {}) {
  if (!container) {
    throw new Error("renderSkillManager requires a container");
  }
  if (container.dataset) {
    container.dataset.v2SkillManager = "true";
  }
  const actions = {
    listSkills: listSkillsApi,
    ...deps,
  };
  const report = await actions.listSkills();
  container.innerHTML = skillManagerHtml(report);
  return report;
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

function skillActionErrorMessage(error, fallback) {
  if (error?.status === 403) {
    return "Skill template management is available from the local workspace only.";
  }
  return error?.message || fallback;
}

export function attachSkillHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachSkillHandlers requires a stable event root");
  }
  const actions = {
    refreshSkills: async () => {},
    reloadSkills: reloadSkillsApi,
    showError: defaultShowError,
    validateSkill: validateSkillApi,
    ...deps,
  };

  const validationSlot = () => root.querySelector?.("[data-skill-validation-result]");

  const clickHandler = async (event) => {
    const reloadButton = closest(event.target, "#reloadSkills")
      || closest(event.target, "[data-reload-skills]");
    if (!reloadButton) {
      return;
    }
    event.preventDefault?.();
    try {
      await actions.reloadSkills();
      await actions.refreshSkills();
    } catch (error) {
      actions.showError(skillActionErrorMessage(error, "skill reload failed"));
    }
  };

  const inputHandler = async (event) => {
    const input = closest(event.target, "[data-validate-skill]");
    if (!input) {
      return;
    }
    const slot = validationSlot();
    let skill;
    try {
      skill = JSON.parse(input.value || "{}");
    } catch (_error) {
      if (slot) {
        slot.innerHTML = '<div class="skill-validation invalid">Invalid JSON</div>';
      }
      return;
    }
    try {
      const result = await actions.validateSkill(skill);
      if (slot) {
        slot.innerHTML = skillValidationResultHtml(result);
      }
    } catch (error) {
      if (slot) {
        slot.innerHTML = skillValidationResultHtml({
          problems: [skillActionErrorMessage(error, "skill validation failed")],
        });
      }
    }
  };

  root.addEventListener("click", clickHandler);
  root.addEventListener("input", inputHandler);
  return () => {
    root.removeEventListener?.("click", clickHandler);
    root.removeEventListener?.("input", inputHandler);
  };
}
