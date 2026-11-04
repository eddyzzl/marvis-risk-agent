// Canonical display labels for fixed headers authored in English. Only call
// this for platform labels; user values, identifiers and payload keys stay raw.
import { localizeText } from "./i18n.js";

const ENGLISH_PLATFORM_LABELS = Object.freeze({
  Rank: "排名",
  Feature: "特征",
  Category: "类别",
  "Combo ID": "组合 ID",
  "Rule ID": "规则 ID",
  Conditions: "规则条件",
  "Hit Share": "命中占比",
  "Bad Rate": "坏账率",
  "Amount Lift": "额度提升倍数",
  Eligible: "满足约束",
  "Constraint Failures": "未满足的约束",
  "X source bin": "X 来源分箱",
  "Y source bin": "Y 来源分箱",
  "Artifact ID": "产物 ID",
  Kind: "类型",
  "Content Hash": "内容哈希",
  Population: "总体",
  action: "操作",
  "evidence status": "证据状态",
  coverage_rate: "覆盖率",
  KS_baseline: "基准 KS",
  KS_after: "压力后 KS",
  KS_delta: "KS 变化",
  "PSI vs baseline": "相对基准 PSI",
});

export function platformLabelSource(value) {
  const source = String(value ?? "");
  return Object.prototype.hasOwnProperty.call(ENGLISH_PLATFORM_LABELS, source)
    ? ENGLISH_PLATFORM_LABELS[source] : source;
}

export function localizePlatformLabel(value) {
  return localizeText(platformLabelSource(value));
}
