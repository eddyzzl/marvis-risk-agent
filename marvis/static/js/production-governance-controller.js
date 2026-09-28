import { projectedStrategyItems } from "./v2/strategy_candidate_lab_contracts.js";
import {
  packageBuildHtml,
  packageSignature,
  readinessHtml,
  collectPackage,
  selectOptions,
  rawFieldsHtml,
  packageRawFields,
} from "./production-package-form.js";
import { api } from "./api.js";
import {
  shellHtml,
  identityHtml,
  headsHtml,
  packagesHtml,
  requestsHtml,
  installationsHtml,
  packageHtml,
  requestHtml,
  featuresFormHtml,
  rollbackHtml,
  decisionHtml,
  jsonDetails,
} from "./production-governance-view.js";
const path = encodeURIComponent;
const text = (v, label) => {
  if (!String(v ?? "").trim()) throw new Error(`请填写${label}`);
  return String(v).trim();
};
export function collectProbe(values, schema) {
  return Object.fromEntries(
    schema.map((field, i) => {
      const raw = values[`feature_${i}`];
      if (raw === "" || raw == null) {
        if (field.nullable) return [field.name, null];
        throw new Error(`请填写 ${field.name}`);
      }
      if (field.type === "boolean") {
        if (!["true", "false"].includes(raw))
          throw new Error(`${field.name} 必须为 true 或 false`);
        return [field.name, raw === "true"];
      }
      if (["number", "integer"].includes(field.type)) {
        const n = Number(raw);
        if (
          !Number.isFinite(n) ||
          (field.type === "integer" && !Number.isSafeInteger(n))
        )
          throw new Error(`${field.name} 的数值类型无效`);
        return [field.name, n];
      }
      return [field.name, String(raw)];
    }),
  );
}
export function promotionPayload(values, record) {
  const c = record.manifest.configuration,
    seconds = Number(text(values.expires_in_seconds, "审批有效期"));
  if (!Number.isInteger(seconds) || seconds < 1 || seconds > 86400)
    throw new Error("审批有效期须为 1 至 86400 秒");
  if (!["production", "shadow"].includes(values.deployment_slot))
    throw new Error("请选择服务位置");
  return {
    environment: "local-reference",
    deployment_slot: values.deployment_slot,
    strategy_id: c.strategy_id,
    strategy_version: c.strategy_version,
    decision_package_hash: record.package_hash,
    reason: text(values.reason, "发布理由"),
    expires_in_seconds: seconds,
  };
}
export function createProductionGovernanceController({
  root,
  isVisible = () => true,
  apiClient = api,
  openIdentity = () => {},
} = {}) {
  let epoch = 0,
    selection = 0,
    principal = null,
    busy = false,
    packages = [],
    requests = [],
    installations = [],
    heads = [],
    environment = null,
    context = null;
  const q = (s) => root.querySelector(s),
    valid = (t) => t === epoch && isVisible(),
    post = (url, body) =>
      apiClient(url, { method: "POST", body: JSON.stringify(body) });
  const message = (text = "", kind = "info") => {
    const element = q("[data-production-error]");
    element.textContent = text;
    element.dataset.status = kind;
  };
  const error = (e) =>
    e.status === 409
      ? `状态或证据已变化，本次操作未被接受；表单已保留，请刷新并复核。${e.detail?.next_action || e.message || ""}`
      : e.detail?.next_action || e.message || "请求失败，请重试";
  const lock = () =>
    root
      .querySelectorAll("[data-production-form] button[type=submit]")
      .forEach((b) => (b.disabled = busy));
  function initialize() {
    if (!root.dataset.productionMounted) {
      root.dataset.productionMounted = "true";
      root.innerHTML = shellHtml();
    }
  }
  async function list(url, key) {
    let rows = [],
      offset = 0;
    do {
      const payload = await apiClient(
        `${url}${url.includes("?") ? "&" : "?"}limit=100&offset=${offset}`,
      );
      rows.push(...payload[key]);
      offset = payload.next_offset;
    } while (offset != null);
    return rows;
  }
  async function refresh() {
    initialize();
    if (busy) return;
    const ticket = ++epoch;
    message("正在读取角色与治理状态…");
    try {
      const identity = await apiClient("/api/production-governance/me");
      if (!valid(ticket)) return;
      const changed = principal?.id !== identity.id;
      principal = identity;
      q("[data-production-identity]").innerHTML = identityHtml(principal);
      q('[data-production-action="build"]').hidden = principal.role !== "maker";
      if (changed) {
        context = null;
        ++selection;
        q("[data-production-editor]").innerHTML = "";
        q("[data-production-detail]").innerHTML = "";
      }
      const result = await Promise.all([
        list("/api/reference-decision/packages", "packages"),
        list(
          "/api/production-governance/promotion-requests?environment=local-reference",
          "requests",
        ),
        list("/api/reference-decision/installations", "installations"),
        apiClient("/api/reference-decision/status?slot=production"),
        apiClient("/api/reference-decision/status?slot=shadow"),
        apiClient("/api/production-governance/environments/local-reference"),
      ]);
      if (!valid(ticket)) return;
      [packages, requests, installations] = result;
      heads = result.slice(3, 5);
      environment = result[5];
      q("[data-production-heads]").innerHTML = headsHtml(
        heads,
        environment,
        principal.role,
      );
      q("[data-production-packages]").innerHTML = packagesHtml(packages);
      q("[data-production-requests]").innerHTML = requestsHtml(requests);
      q("[data-production-installations]").innerHTML =
        installationsHtml(installations);
      message("");
      return true;
    } catch (e) {
      if (!valid(ticket)) return;
      if (e.status === 403) {
        principal = null;
        context = null;
        ++selection;
        root.innerHTML = shellHtml();
        q("[data-production-identity]").innerHTML = identityHtml(null);
        q('[data-production-action="build"]').hidden = true;
      } else message(error(e), "error");
    }
  }
  function selectContext(data) {
    context = data;
    ++selection;
    q("[data-production-editor]").innerHTML = "";
    q("[data-production-detail]").innerHTML = "";
    message("");
    return { ticket: epoch, version: selection };
  }
  const selected = (owner) =>
    valid(owner.ticket) && selection === owner.version;
  async function showPackage(id) {
    const owner = selectContext(null);
    try {
      const result = await apiClient(
        `/api/reference-decision/packages/${path(id)}`,
      );
      if (!selected(owner)) return;
      context = { kind: "package", record: result };
      q("[data-production-detail]").innerHTML = packageHtml(
        result,
        principal.role,
      );
      return true;
    } catch (e) {
      if (selected(owner)) message(error(e), "error");
    }
  }
  async function showRequest(id, { readInstallation = false } = {}) {
    const owner = selectContext(null);
    try {
      const request = await apiClient(
        `/api/production-governance/promotion-requests/${path(id)}`,
      );
      if (!selected(owner)) return;
      let installation = null;
      const exists = installations.some((i) => i.promotion_request_id === id);
      if (readInstallation || exists)
        installation = await apiClient(
          `/api/reference-decision/installations/${path(id)}`,
        );
      if (!selected(owner)) return;
      context = { kind: "request", request, installation };
      q("[data-production-detail]").innerHTML =
        requestHtml(request, principal, installation) +
        (readInstallation ? jsonDetails("本次安装回执读回", installation) : "");
      return true;
    } catch (e) {
      if (selected(owner)) message(error(e), "error");
    }
  }
  async function prepareFeatures(kind, id) {
    const request = context?.request,
      head = kind === "decision" ? heads[id === "shadow" ? 1 : 0] : null,
      hash = head?.package_hash || request?.decision_package_hash;
    const owner = selectContext(null);
    if (!hash) {
      message("当前记录没有可校验的冻结包，请刷新");
      return;
    }
    try {
      const record = await apiClient(
        `/api/reference-decision/packages/${path(hash)}`,
      );
      if (!selected(owner)) return;
      context = { kind, record, request, head, slot: id };
      q("[data-production-editor]").innerHTML = featuresFormHtml(
        kind,
        record.manifest.configuration.raw_schema,
        {
          slot: id,
          requestId: kind === "decision" ? `local-${crypto.randomUUID()}` : "",
        },
      );
    } catch (e) {
      if (selected(owner)) message(error(e), "error");
    }
  }
  async function startBuild() {
    const state = {
      kind: "build",
      strategies: [],
      artifacts: [],
      sourceVersion: 0,
      sourceVersions: {},
      readiness: null,
      checkedSignature: null,
    };
    const owner = selectContext(state);
    try {
      const tasks = await apiClient("/api/tasks");
      if (selected(owner) && context === state)
        q("[data-production-editor]").innerHTML = packageBuildHtml(tasks);
    } catch (e) {
      if (selected(owner)) message(error(e), "error");
    }
  }
  async function buildSource(form, name) {
    const state = context;
    if (state?.kind !== "build") return;
    const owner = { ticket: epoch, version: selection },
      version = (state.sourceVersions[name] || 0) + 1;
    state.sourceVersions[name] = version;
    ++state.sourceVersion;
    state.readiness = null;
    state.checkedSignature = null;
    q("[data-production-readiness]").innerHTML = "";
    const modeling = name === "model_task_id",
      target = modeling ? "model_artifact_id" : "strategy_id",
      task = form.elements[name].value;
    form.elements[target].innerHTML = selectOptions([]);
    if (modeling) state.artifacts = [];
    else state.strategies = [];
    if (!task) return;
    try {
      const data = await apiClient(
        `/api/tasks/${path(task)}/${modeling ? "experiments" : "strategy-candidate-lab"}`,
      );
      if (
        !selected(owner) ||
        context !== state ||
        version !== state.sourceVersions[name]
      )
        return;
      if (modeling) {
        state.artifacts = (data.experiments || [])
          .filter((e) => ["trained", "selected"].includes(e.status))
          .flatMap((e) =>
            (e.artifacts || []).filter((a) => a.id === e.artifact_id),
          );
        form.elements[target].innerHTML = selectOptions(
          state.artifacts.map((a) => ({
            id: a.id,
            label: `${a.algorithm} · ${a.id}`,
          })),
        );
      } else {
        state.strategies = projectedStrategyItems(data);
        form.elements[target].innerHTML = selectOptions(
          state.strategies.map((s) => ({
            id: s.strategy_id,
            label: `${s.strategy_id} · v${s.version} · ${s.asset_status || s.status}`,
          })),
        );
      }
      message("");
    } catch (e) {
      if (selected(owner) && version === state.sourceVersions[name])
        message(error(e), "error");
    }
  }
  async function checkReadiness() {
    const state = context,
      form = q('[data-production-form="build"]');
    if (state?.kind !== "build" || !form) return;
    const kind = form.elements.package_kind.value,
      artifact = form.elements.model_artifact_id.value,
      strategy = state.strategies.find(
        (s) => s.strategy_id === form.elements.strategy_id.value,
      );
    if (
      !["model", "rule_only"].includes(kind) ||
      !strategy ||
      (kind === "model" && !state.artifacts.some((a) => a.id === artifact))
    ) {
      message("请选择决策包类型、物化策略；模型决策另需当前任务发现的模型产物");
      return;
    }
    const owner = { ticket: epoch, version: selection },
      version = ++state.sourceVersion,
      signature = packageSignature(
        { package_kind: kind, model_artifact_id: artifact },
        strategy,
      );
    state.readiness = null;
    state.checkedSignature = null;
    message(
      kind === "rule_only"
        ? "正在核对策略表达式与原始字段…"
        : "正在核对原生训练与预处理来源…",
    );
    try {
      const result = await apiClient(
        `/api/reference-decision/readiness?${kind === "rule_only" ? "package_kind=rule_only" : `model_artifact_id=${path(artifact)}`}&strategy_id=${path(strategy.strategy_id)}&strategy_version=${strategy.version}`,
      );
      if (
        !selected(owner) ||
        context !== state ||
        version !== state.sourceVersion
      )
        return;
      state.readiness = result;
      state.checkedSignature = signature;
      q("[data-production-readiness]").innerHTML = readinessHtml(result, kind);
      message("");
    } catch (e) {
      if (selected(owner) && version === state.sourceVersion)
        message(error(e), "error");
    }
  }
  async function mutate(form) {
    if (busy || !context) return;
    const owner = { ticket: epoch, version: selection },
      bound = context,
      values = Object.fromEntries(
        Array.from(form.querySelectorAll("[name]")).map((i) => [
          i.name,
          i.value,
        ]),
      ),
      kind = form.dataset.productionForm;
    let url, body;
    try {
      if (kind === "build") {
        url = "/api/reference-decision/packages";
        body = collectPackage(values, bound);
      } else if (kind === "promotion") {
        url = "/api/production-governance/promotion-requests";
        body = promotionPayload(values, bound.record);
      } else if (kind === "approve") {
        url = `/api/production-governance/promotion-requests/${path(bound.request.id)}/approvals`;
        if (!["approve", "reject"].includes(values.decision))
          throw new Error("请选择复核决定");
        body = {
          decision: values.decision,
          reason: text(values.reason, "本次复核理由"),
        };
      } else if (kind === "activate") {
        const evidence = bound.installation?.activation_evidence;
        if (!evidence?.receipt_id || !evidence?.verifier_id)
          throw new Error("请重新读回安装证据");
        url = `/api/production-governance/environments/local-reference/promotion-requests/${path(bound.request.id)}/activate`;
        body = {
          receipt_id: evidence.receipt_id,
          verifier_id: evidence.verifier_id,
          reason: text(values.reason, "激活理由"),
        };
      } else if (kind === "install") {
        url = "/api/reference-decision/installations";
        body = {
          promotion_id: bound.request.id,
          probe_features: collectProbe(
            values,
            bound.record.manifest.configuration.raw_schema,
          ),
        };
      } else if (kind === "decision") {
        url = `/api/reference-decision/decisions?slot=${path(bound.slot)}`;
        body = {
          request_id: text(values.request_id, "请求唯一标识"),
          decision_node: bound.record.manifest.configuration.decision_node,
          expected_package_hash: bound.head.package_hash,
          features: collectProbe(
            values,
            bound.record.manifest.configuration.raw_schema,
          ),
        };
      } else if (kind === "rollback") {
        url = `/api/production-governance/environments/local-reference/deployments/${path(bound.head.deployment_id)}/rollback`;
        body = {
          reason: text(values.reason, "回滚理由"),
          expected_active_deployment_id: bound.head.deployment_id,
          deployment_slot: bound.slot,
        };
      } else return;
    } catch (e) {
      message(error(e), "error");
      return;
    }
    busy = true;
    lock();
    message("正在提交并验证当前绑定…");
    try {
      const result = await post(url, body);
      if (!selected(owner)) return;
      if (kind === "decision") {
        q("[data-production-detail]").innerHTML = decisionHtml(
          result,
          bound.record.manifest.configuration.package_kind,
        );
        message("本地决策回执已返回；请求标识和输入已保留，可用于幂等重试。");
      } else {
        busy = false;
        const previousSelection = selection;
        if (
          !(await refresh()) ||
          !isVisible() ||
          selection !== previousSelection
        )
          return;
        const refreshedEpoch = epoch,
          expectedSelection = selection + 1;
        let completed;
        if (kind === "build")
          completed = await showPackage(result.package_hash);
        else if (kind === "promotion") completed = await showRequest(result.id);
        else if (["approve", "install", "activate"].includes(kind))
          completed = await showRequest(bound.request.id);
        else {
          ++selection;
          completed = true;
          context = null;
          q("[data-production-editor]").innerHTML = "";
          q("[data-production-detail]").innerHTML = jsonDetails(
            "回滚回执",
            result,
          );
        }
        if (
          completed &&
          valid(refreshedEpoch) &&
          selection === expectedSelection
        )
          message("操作已完成，已重新读取治理状态。", "success");
      }
    } catch (e) {
      if (selected(owner)) message(error(e), "error");
    } finally {
      busy = false;
      if (selected(owner)) lock();
    }
  }
  function handle(event) {
    if (!root.contains(event.target)) return false;
    if (event.type === "input" || event.type === "change") {
      const form = event.target.closest('[data-production-form="build"]');
      if (!form || context?.kind !== "build") return false;
      if (
        event.target.name === "score_field" &&
        context.readiness?.state === "authenticated"
      ) {
        const base = context.readiness.raw_requirements.length;
        q("[data-production-extra-fields]").innerHTML = rawFieldsHtml(
          packageRawFields(context.readiness, event.target.value).slice(base),
          base,
        );
        return true;
      }
      if (event.type !== "change") return false;
      if (event.target.name === "package_kind") {
        const model = event.target.value === "model";
        q("[data-production-model-source]").hidden = !model;
        for (const name of ["model_task_id", "model_artifact_id"]) {
          form.elements[name].disabled = !model;
          form.elements[name].required = model;
        }
      }
      if (["model_task_id", "strategy_task_id"].includes(event.target.name))
        void buildSource(form, event.target.name);
      else if (
        ["package_kind", "model_artifact_id", "strategy_id"].includes(
          event.target.name,
        )
      ) {
        ++context.sourceVersion;
        context.readiness = null;
        context.checkedSignature = null;
        q("[data-production-readiness]").innerHTML = "";
      }
      return true;
    }
    if (event.type === "submit") {
      const form = event.target.closest("[data-production-form]");
      if (!form) return false;
      event.preventDefault();
      void mutate(form);
      return true;
    }
    const control = event.target.closest("[data-production-action]");
    if (!control) return false;
    event.preventDefault();
    if (busy) {
      message("操作正在提交，请等待完成");
      return true;
    }
    const { productionAction: action, productionId: id } = control.dataset;
    if (action === "identity") openIdentity();
    if (action === "refresh") void refresh();
    if (action === "package") void showPackage(id);
    if (action === "request") void showRequest(id);
    if (action === "installation")
      void showRequest(id, { readInstallation: true });
    if (action === "prepare-install") void prepareFeatures("install", id);
    if (action === "probe") void prepareFeatures("decision", id);
    if (action === "rollback") {
      const slot = id === "shadow" ? "shadow" : "production";
      const head = heads[slot === "shadow" ? 1 : 0];
      selectContext({ kind: "rollback", head, slot });
      q("[data-production-editor]").innerHTML = rollbackHtml(head, slot);
    }
    if (action === "build") void startBuild();
    if (action === "readiness") void checkReadiness();
    return true;
  }
  function leave() {
    ++epoch;
    ++selection;
    context = null;
    if (root.dataset.productionMounted) {
      q("[data-production-editor]").innerHTML = "";
      q("[data-production-detail]").innerHTML = "";
    }
  }
  return {
    refresh,
    handle,
    leave,
  };
}
