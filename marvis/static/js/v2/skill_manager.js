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

// The /api/skills payload is {skills: [{id, status, problems, title}], builtin: [...]};
// tests and older callers still hand in {active, disabled, rejected}. Accept both.
function normalizeSkillReport(report = {}) {
  const titles = {};
  if (Array.isArray(report.skills)) {
    const active = [];
    const disabled = [];
    const rejected = [];
    for (const item of report.skills) {
      const id = item?.id || item?.name || "unknown";
      if (item?.title) titles[id] = item.title;
      if (item?.status === "disabled") {
        disabled.push(id);
      } else if (item?.status === "rejected") {
        rejected.push([id, item?.problems || []]);
      } else {
        active.push(id);
      }
    }
    return { active, disabled, rejected, titles, builtin: report.builtin || [] };
  }
  return {
    active: report.active || [],
    disabled: report.disabled || [],
    rejected: report.rejected || [],
    titles,
    builtin: report.builtin || [],
  };
}

function statusLabel(status) {
  return {
    active: "启用",
    disabled: "停用",
    rejected: "已拒绝",
  }[status] || String(status || "未知");
}

export function skillRowHtml(id, status, problems = [], title = "") {
  const problemList = problems.length
    ? `<ul class="skill-problems">${problems.map((problem) => `<li>${escapeHtml(problem)}</li>`).join("")}</ul>`
    : "";
  const idLine = title && title !== id ? `<span class="skill-id">${escapeHtml(id)}</span>` : "";
  return `<section class="skill skill-${escapeHtml(status)}">
    <div class="skill-row-head">
      <strong>${escapeHtml(title || id)}</strong>
      <span class="skill-badge ${escapeHtml(status)}">${escapeHtml(statusLabel(status))}</span>
    </div>
    ${idLine}
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
  const normalized = normalizeSkillReport(report);
  const active = normalized.active.map((id) => skillRowHtml(id, "active", [], normalized.titles[id])).join("");
  const disabled = normalized.disabled.map((id) => skillRowHtml(id, "disabled", [], normalized.titles[id])).join("");
  const rejected = rejectedEntries(normalized)
    .map(([id, problems]) => skillRowHtml(id, "rejected", problems, normalized.titles[id]))
    .join("");
  const customRows = active + disabled + rejected;
  const customList = customRows
    || `<div class="v2-empty" data-v2-empty="skills">还没有自定义模板。把模板 JSON 文件放进工作区的 skills/ 目录，再点右侧「重新加载」即可生效。</div>`;
  const builtinChips = (normalized.builtin || [])
    .map((item) => {
      const id = item?.id || "";
      const title = item?.title || id;
      return `<span class="skill-builtin-chip" title="${escapeHtml(id)}">${escapeHtml(title)}</span>`;
    })
    .join("");
  const builtinSection = builtinChips
    ? `<section class="skill-section">
      <p class="skill-section-label">内置 Workflow</p>
      <p class="skill-section-hint">随平台发布、开箱即用，Agent 会按任务目标自动套用，无需配置。</p>
      <div class="skill-builtin-list">${builtinChips}</div>
    </section>`
    : "";
  return `<section class="skill-manager">
    ${builtinSection}
    <section class="skill-section">
      <div class="skill-toolbar">
        <div class="skill-toolbar-text">
          <p class="skill-section-label">自定义模板</p>
          <p class="skill-section-hint">用 JSON 编写自己的 Workflow 模板，放入工作区 skills/ 目录；加载时自动校验，有问题的模板会被拒绝并标注原因。</p>
        </div>
        <button id="reloadSkills" type="button" class="button primary compact" data-reload-skills>重新加载</button>
      </div>
      <div class="skill-list">${customList}</div>
    </section>
    <details class="skill-validator-details">
      <summary>
        <span class="skill-validator-summary-text">
          <strong>校验模板 JSON</strong>
          <span>发布前先粘贴到这里检查格式和步骤定义。</span>
        </span>
        <svg class="skill-validator-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      </summary>
      <div class="skill-validator-body">
        <label class="skill-validator">
          <textarea data-validate-skill spellcheck="false" placeholder="粘贴 Workflow 模板 JSON，输入即校验"></textarea>
        </label>
        <div data-skill-validation-result></div>
      </div>
    </details>
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
