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

function statusLabel(status) {
  return {
    active: "启用",
    disabled: "停用",
    rejected: "已拒绝",
  }[status] || String(status || "未知");
}

export function skillRowHtml(id, status, problems = []) {
  const problemList = problems.length
    ? `<ul class="skill-problems">${problems.map((problem) => `<li>${escapeHtml(problem)}</li>`).join("")}</ul>`
    : "";
  return `<section class="skill skill-${escapeHtml(status)}">
    <strong>${escapeHtml(id)}</strong>
    <span class="skill-badge ${escapeHtml(status)}">${escapeHtml(statusLabel(status))}</span>
    ${problemList}
  </section>`;
}

export function skillValidationResultHtml(result = {}) {
  const problems = result.problems || [];
  if (!problems.length) {
    return '<div class="skill-validation valid">模板有效</div>';
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
    <button id="reloadSkills" type="button" data-reload-skills>重新加载模板</button>
    <div class="skill-list">
      ${active}
      ${disabled}
      ${rejected}
    </div>
    <label class="skill-validator">
      <textarea data-validate-skill spellcheck="false" placeholder="粘贴 Workflow 模板 JSON 进行校验"></textarea>
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
    return "Workflow 模板管理仅支持在本地工作区使用。";
  }
  return error?.message || fallback;
}

function defaultScheduleValidation(fn, delay) {
  return setTimeout(fn, delay);
}

function defaultCancelValidation(handle) {
  clearTimeout(handle);
}

export function attachSkillHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachSkillHandlers requires a stable event root");
  }
  const actions = {
    cancelValidation: defaultCancelValidation,
    refreshSkills: async () => {},
    reloadSkills: reloadSkillsApi,
    scheduleValidation: defaultScheduleValidation,
    showError: defaultShowError,
    validateSkill: validateSkillApi,
    validationDelayMs: 250,
    ...deps,
  };

  const validationSlot = () => root.querySelector?.("[data-skill-validation-result]");
  let pendingValidation = null;
  let validationVersion = 0;

  const cancelPendingValidation = () => {
    if (pendingValidation !== null) {
      actions.cancelValidation(pendingValidation);
      pendingValidation = null;
    }
  };

  const runValidation = async (skill, version) => {
    if (version !== validationVersion) {
      return;
    }
    const slot = validationSlot();
    try {
      const result = await actions.validateSkill(skill);
      if (version === validationVersion && slot) {
        slot.innerHTML = skillValidationResultHtml(result);
      }
    } catch (error) {
      if (version === validationVersion && slot) {
        slot.innerHTML = skillValidationResultHtml({
          problems: [skillActionErrorMessage(error, "模板校验失败")],
        });
      }
    }
  };

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
        actions.showError(skillActionErrorMessage(error, "模板重新加载失败"));
    }
  };

  const inputHandler = async (event) => {
    const input = closest(event.target, "[data-validate-skill]");
    if (!input) {
      return;
    }
    const slot = validationSlot();
    cancelPendingValidation();
    const version = validationVersion + 1;
    validationVersion = version;
    let skill;
    try {
      skill = JSON.parse(input.value || "{}");
    } catch (_error) {
      if (slot) {
        slot.innerHTML = '<div class="skill-validation invalid">JSON 格式无效</div>';
      }
      return;
    }
    pendingValidation = actions.scheduleValidation(() => {
      pendingValidation = null;
      return runValidation(skill, version);
    }, actions.validationDelayMs);
  };

  root.addEventListener("click", clickHandler);
  root.addEventListener("input", inputHandler);
  return () => {
    cancelPendingValidation();
    validationVersion += 1;
    root.removeEventListener?.("click", clickHandler);
    root.removeEventListener?.("input", inputHandler);
  };
}
