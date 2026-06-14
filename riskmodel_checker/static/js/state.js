export const defaultPetPreference = "naitang";
export const explicitPetNoneStorageKey = "riskmodel_checker_pet_none_explicit";
export const agentComposerPreferenceStorageKey = "riskmodel_checker_agent_composer_preferences";
export const selectedTaskStorageKey = "riskmodel_checker_selected_task_id";
// Per-task composer overrides: `{ [taskId]: { model_id, effort, acceptance_mode } }`.
// Each task remembers its own mode/model/effort. The global preferences above
// are only the seed value applied when a task has no override yet.
export const agentTaskComposerStorageKey = "riskmodel_checker_agent_task_composer_preferences";

export const defaultBranding = {
  platformName: "MARVIS-全能风控智能体",
  browserTitle: "MARVIS-全能风控智能体",
  primaryColor: "#000000",
  logoUrl: "static/brand/marvis-logo.png",
  faviconUrl: "static/brand/marvis-favicon.png",
  // Real validator name -> display alias, supplied per-workspace by brand.json.
  validatorAliases: {},
};

export const defaultExecutionEnvironment = {
  execution_mode: "jupyter_kernel",
  kernel_name: "python3",
  conda_env_name: "",
  python_executable: "",
};

export function createRenderSignatures() {
  return {
    actionStatus: "",
    currentTask: "",
    taskList: "",
    workflowStepper: "",
    metricPreview: "",
    metricPreviewTaskId: "",
    // Reproducibility precision-bar chart lives in a second highly-animated
    // region. We track its structural signature here (instead of on the DOM
    // dataset) and gate the CSS entry animation so it only plays for the
    // first populated render of a given task - not every time a transient
    // empty evidence payload arrives between populated ones.
    reproducibilityEvidence: "",
    reproducibilityTaskId: "",
    reproducibilityAnimatedTaskId: "",
  };
}

export const activeValidationStatuses = new Set([
  "created",
  "scanned",
  "running",
  "executed",
  "computing_metrics",
  "writing_artifacts",
]);

export const terminalTaskStatuses = new Set([
  "succeeded",
  "failed",
  "review_required",
]);

export const notebookReproducibilityCompleteStatuses = new Set([
  "executed",
  "computing_metrics",
  "writing_artifacts",
  "succeeded",
  "review_required",
]);

export const metricOverviewCompleteStatuses = new Set([
  "writing_artifacts",
  "succeeded",
  "review_required",
]);

export const workflowSteps = [
  { id: "scan", title: "模型材料完备性验证", hint: "巡检材料内容", target: "scanSection", action: "scan", actionLabel: "重新扫描" },
  { id: "notebook", title: "模型可复现性验证", hint: "执行建模代码", target: "notebookSection", action: "notebook", actionLabel: "运行" },
  { id: "metrics", title: "模型效果&稳定性验证", hint: "指标概览", target: "metricSection", action: "metrics", actionLabel: "生成" },
  { id: "report", title: "报告输出", hint: "Word报告与Excel分析", target: "reportSection", action: "report", actionLabel: "生成" },
];

export const statusLabels = {
  created: "已创建",
  scanned: "已扫描",
  running: "运行中",
  executed: "已执行",
  computing_metrics: "计算指标",
  writing_artifacts: "写入产物",
  succeeded: "已出报告",
  failed: "失败",
  review_required: "待复核",
};

export const roleLabels = {
  notebook: "Notebook",
  sample: "样本数据",
  model_pmml: "PMML 模型",
  data_dictionary: "数据字典",
  unknown: "未识别文件",
};

export const requiredMaterialRoles = [
  { role: "notebook", label: "Notebook" },
  { role: "sample", label: "样本数据" },
  { role: "model_pmml", label: "PMML 模型" },
  { role: "data_dictionary", label: "数据字典" },
];

export const scanFailurePrefix = "材料扫描失败：";
