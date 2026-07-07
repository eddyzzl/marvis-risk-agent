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

const WORKFLOW_TEMPLATE_EXAMPLE = {
  id: "custom_echo_review",
  title: "自定义 Echo 复核",
  goal_patterns: ["echo", "测试编排"],
  default_autonomy: 1,
  slots: [
    {
      name: "message",
      required: true,
      source: "user",
      description: "需要传给工具的文本",
    },
  ],
  steps: [
    {
      title: "Echo",
      tool: { plugin: "_sample", tool: "echo" },
      inputs: { message: "{slot:message}" },
      depends_on: [],
      needs_confirmation: false,
      post_checks: [
        { kind: "nonempty", spec: { field: "echoed" } },
      ],
    },
  ],
};

function codeBlockHtml(code) {
  return `<pre><code>${escapeHtml(code)}</code></pre>`;
}

function jsonBlockHtml(value) {
  return codeBlockHtml(JSON.stringify(value ?? {}, null, 2));
}

function workflowTemplateGuideHtml() {
  return `<details class="extension-format-guide workflow-format-guide">
    <summary>
      <span>
        <strong>模板 JSON 示例</strong>
        <span>保存为 workspace/skills/*.json 后点「重新加载」；也可以先粘贴到下方校验框。</span>
      </span>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
    </summary>
    <div class="extension-format-guide-body">
      <div>
        <strong>custom_echo_review.json</strong>
        ${codeBlockHtml(JSON.stringify(WORKFLOW_TEMPLATE_EXAMPLE, null, 2))}
      </div>
    </div>
  </details>`;
}

function toolLabel(tool = {}) {
  return [tool.plugin, tool.tool].filter(Boolean).join(".") || "未声明工具";
}

function workflowStepFlags(step = {}) {
  return [
    step.needs_confirmation ? "需要确认" : "",
    step.decision_point ? "决策点" : "",
    step.sub_agent_scope ? `子 Agent：${step.sub_agent_scope}` : "",
    step.phase ? `阶段：${step.phase}` : "",
  ].filter(Boolean);
}

function workflowPostChecksHtml(checks = []) {
  if (!checks.length) return "";
  return `<div class="skill-workflow-checks">
    <strong>后置检查</strong>
    <ul>
      ${checks.map((check) => `<li>${escapeHtml(check.kind || "")} ${escapeHtml(JSON.stringify(check.spec || {}))}</li>`).join("")}
    </ul>
  </div>`;
}

function workflowSlotListHtml(slots = []) {
  if (!slots.length) {
    return '<div class="skill-workflow-empty">无槽位要求。</div>';
  }
  return `<dl class="skill-workflow-slots">
    ${slots.map((slot) => `<div>
      <dt>${escapeHtml(slot.name || "")}${slot.required === false ? "" : '<span class="skill-required">*</span>'}</dt>
      <dd>${escapeHtml(slot.description || slot.source || "无说明")}</dd>
    </div>`).join("")}
  </dl>`;
}

function workflowStepListHtml(steps = []) {
  if (!steps.length) {
    return '<div class="skill-workflow-empty">暂无步骤明细。</div>';
  }
  return `<ol class="skill-workflow-steps">
    ${steps.map((step) => {
      const flags = workflowStepFlags(step);
      const dependencies = Array.isArray(step.depends_on) && step.depends_on.length
        ? `<span>依赖：${escapeHtml(step.depends_on.join(" / "))}</span>`
        : "";
      return `<li>
        <div class="skill-workflow-step-head">
          <strong>${escapeHtml(step.title || "未命名步骤")}</strong>
          <code>${escapeHtml(toolLabel(step.tool))}</code>
        </div>
        <div class="skill-workflow-step-meta">
          ${dependencies}
          ${flags.map((flag) => `<span>${escapeHtml(flag)}</span>`).join("")}
        </div>
        <div class="skill-workflow-step-inputs">
          <strong>输入模板</strong>
          ${jsonBlockHtml(step.inputs || {})}
        </div>
        ${workflowPostChecksHtml(step.post_checks || [])}
      </li>`;
    }).join("")}
  </ol>`;
}

function builtinWorkflowHtml(item = {}) {
  const id = item.id || "";
  const title = item.title || id || "未命名 Workflow";
  const goals = Array.isArray(item.goal_patterns) ? item.goal_patterns : [];
  const steps = Array.isArray(item.steps) ? item.steps : [];
  const slots = Array.isArray(item.slots) ? item.slots : [];
  return `<details class="skill-builtin-workflow">
    <summary>
      <span class="skill-builtin-workflow-title">
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(id)} · ${steps.length} 步 · L${escapeHtml(item.default_autonomy ?? 1)}</span>
      </span>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
    </summary>
    <div class="skill-builtin-workflow-body">
      ${goals.length ? `<p class="skill-workflow-goals">触发意图：${escapeHtml(goals.join(" / "))}</p>` : ""}
      <section>
        <h4>槽位</h4>
        ${workflowSlotListHtml(slots)}
      </section>
      <section>
        <h4>步骤</h4>
        ${workflowStepListHtml(steps)}
      </section>
      ${Array.isArray(item.success_criteria) && item.success_criteria.length ? `<section>
        <h4>成功标准</h4>
        ${jsonBlockHtml(item.success_criteria)}
      </section>` : ""}
    </div>
  </details>`;
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
    .map((item) => builtinWorkflowHtml(item))
    .join("");
  const builtinSection = builtinChips
    ? `<section class="skill-section">
      <p class="skill-section-label">内置 Workflow</p>
      <p class="skill-section-hint">随平台发布、开箱即用；点击任一 Workflow 查看槽位、步骤、工具和输入模板。</p>
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
      ${workflowTemplateGuideHtml()}
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
