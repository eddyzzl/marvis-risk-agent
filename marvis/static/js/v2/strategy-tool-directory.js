import { escapeHtml } from "../ui-utils.js";

const STAGES = [
  ["current_context", "项目现状"], ["history", "历史版本"], ["sample_design", "样本设计"],
  ["candidate_analysis", "单变量 / 模型"], ["strategy_combination", "交叉组合 / 策略"],
  ["impact", "影响测算"], ["report", "形成报告"],
];
export function strategyToolStage(workflow) {
  if (workflow === "strategy_project_context") return "current_context";
  if (workflow === "strategy_sample_design_v2") return "sample_design";
  if (/report_bundle|dsl_delivery|lifecycle_adopt/.test(workflow)) return "report";
  if (/impact|stability|validation/.test(workflow)) return "impact";
  if (/^strategy_pool_|^cross_|^voting_/.test(workflow)) return "strategy_combination";
  return "candidate_analysis";
}

const mounted = new WeakMap();
export function updateStrategyToolDirectory(root, payload, taskId) {
  // Small presenter tests may supply a non-DOM host; real mounting requires a document.
  if (!root?.ownerDocument) return;
  const launchers = root.querySelector(".candidate-lab-launchers");
  if (!launchers) return;
  let state = mounted.get(root);
  if (!state) {
    const directory = root.ownerDocument.createElement("div");
    directory.className = "strategy-tool-directory";
    directory.innerHTML = '<label>工作阶段<select data-strategy-tool-stage>'
      + STAGES.map(([id, title], i) => `<option value="${id}">${i + 1}. ${escapeHtml(title)}</option>`).join("")
      + '<option value="all">全部工具</option></select></label>'
      + '<label>查找工具<input type="search" data-strategy-tool-search placeholder="搜索名称或分析方法" /></label>'
      + '<span data-strategy-tool-count role="status"></span>';
    launchers.before(directory);
    state = { taskId: "", directory, launchers, touched: false };
    mounted.set(root, state);
    const select = directory.querySelector("select");
    const search = directory.querySelector("input");
    const filter = () => {
      const term = search.value.trim().toLocaleLowerCase();
      let count = 0;
      for (const details of launchers.querySelectorAll(".candidate-lab-launcher")) {
        const workflow = details.querySelector("[data-candidate-lab-workflow]")?.dataset.candidateLabWorkflow || "";
        const group = strategyToolStage(workflow);
        const match = term ? details.textContent.toLocaleLowerCase().includes(term) || workflow.includes(term)
          : select.value === "all" || select.value === group || (select.value === "history" && workflow === "strategy_project_context");
        details.hidden = !match;
        if (match) count += 1;
      }
      directory.querySelector("[data-strategy-tool-count]").textContent = `${count} 项操作${term ? " · 搜索全部工具" : ""}`;
    };
    state.filter = filter;
    directory.addEventListener("input", () => { state.touched = true; filter(); });
    launchers.addEventListener("toggle", (event) => {
      // A candidate's existing contextual action may open its corresponding form.
      // Reveal that form without altering its binding, controls or submission.
      if (event.target.open && event.target.hidden) {
        select.value = "all"; search.value = ""; state.touched = true; filter();
      }
    }, true);
  }
  if (state.taskId !== taskId) {
    state.taskId = taskId;
    state.touched = false;
    state.directory.querySelector("input").value = "";
  }
  if (!state.touched) {
    const nextStage = payload?.workflow?.stages?.find((stage) => stage.status !== "complete");
    state.directory.querySelector("select").value = nextStage?.id || "current_context";
  }
  state.filter();
}
