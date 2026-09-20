// The headline presents workflow state. Action text remains separate detail;
// it must not reinterpret a structured plan failure as a validation result.
export function actionStatusPresentation(message, kind, {
  task = null,
  stopped = false,
  usesPlanWorkflow = false,
  workflowSnapshot = null,
} = {}) {
  if (!message) return null;
  if (stopped) return { label: "停止", tone: "neutral" };
  if (usesPlanWorkflow && workflowSnapshot?.label) {
    const tones = { danger: "fail", success: "ok", run: "run" };
    return {
      label: workflowSnapshot.label,
      tone: tones[workflowSnapshot.tone] || "neutral",
    };
  }
  if (kind === "stopped") return { label: "停止", tone: "neutral" };
  if (kind === "error") {
    return task?.status === "review_required"
      ? { label: "需复核", tone: "ok" }
      : { label: usesPlanWorkflow ? "执行失败" : "验证失败", tone: "fail" };
  }
  if (kind === "busy") return { label: "进行中", tone: "run" };
  if (kind === "success") {
    const wholeTaskComplete = task
      ? !usesPlanWorkflow && ["succeeded", "review_required"].includes(String(task.status || ""))
      : true;
    if (wholeTaskComplete) return { label: "已完成", tone: "ok" };
    if (/(?:等待|请).*确认|待确认/.test(message)) {
      return { label: "待确认", tone: "review" };
    }
    return { label: "待继续", tone: "review" };
  }
  if (kind === "info" && /(?:等待|待|请).*确认/.test(message)) {
    return { label: "待确认", tone: "review" };
  }
  return { label: "待处理", tone: "neutral" };
}
