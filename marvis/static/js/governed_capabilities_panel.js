import { api as defaultApi } from "./api.js";

const BASE = "/api/plugins/governance";
const labels = {
  capability_head_conflict: "版本已变化。请刷新并重新核对提案。",
  capability_mode_conflict: "安装模式已变化。请刷新后再提交。",
  publisher_not_trusted: "发布者尚未受信任。请先由管理员核验公钥。",
  publisher_revoked: "发布者已撤销，不能安装、启用或回滚其版本。",
  publisher_signature_not_verified: "签名验证失败，材料没有注册或执行。",
  publisher_signature_dependency_unavailable: "运行环境缺少签名验证组件，请管理员安装 audit 依赖后重试。材料没有注册或执行。",
  capability_signed_content_changed: "签名与材料内容不一致，材料没有注册或执行。",
  capability_permissions_not_approved: "请逐项确认本次声明的权限。",
  unsigned_enabled_plugins_must_be_disabled: "请先停用未签名插件，再启用受治理模式。",
  unsigned_active_workflows_must_be_disabled: "请先停用未签名 Workflow，再启用受治理模式。",
  capability_proposal_actor_changed: "提案属于另一个会话，请在创建提案的会话确认。",
  capability_proposal_expired: "提案已过期，请重新上传材料并核对。",
  capability_version_identity_reused: "已保留版本不能覆盖，请发布新的版本号。",
  signed_capability_required: "此模式要求签名能力，请使用受治理安装入口。",
  workflow_tool_declaration_changed: "Workflow 引用的工具声明已变化，请重新发布明确版本。",
};

function el(document, tag, text = "", className = "") {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function input(document, type, name, label, parent) {
  const wrapper = el(document, "label", label, "capability-field");
  const node = el(document, "input");
  node.type = type;
  node.name = name;
  node.dataset.capabilityField = name;
  wrapper.append(node);
  parent.append(wrapper);
  return node;
}

function select(document, name, choices, parent) {
  const node = el(document, "select");
  node.name = name;
  node.dataset.capabilityField = name;
  for (const [value, label] of choices) {
    const option = el(document, "option", label);
    option.value = value;
    node.append(option);
  }
  parent.append(node);
  return node;
}

export function createGovernedCapabilitiesPanel({
  container, api = defaultApi, getAdminToken = null, onChanged = () => {},
}) {
  if (!container) throw new Error("governed capability container required");
  const document = container.ownerDocument;
  let overview = null;
  let proposal = null;
  let destroyed = false;
  let busy = false;
  container.replaceChildren();
  const status = el(document, "p", "正在读取安装状态…", "capability-status");
  status.setAttribute("role", "status");
  status.dataset.capabilityStatus = "true";
  const controls = el(document, "div", "", "capability-controls");
  const admin = input(document, "password", "admin", "管理员确认凭据", controls);
  admin.autocomplete = "off";
  const refreshButton = el(document, "button", "刷新状态", "button secondary");
  refreshButton.type = "button";
  refreshButton.dataset.capabilityAction = "refresh";
  controls.append(refreshButton);
  const modeBox = el(document, "section", "", "capability-mode");
  modeBox.append(el(document, "h4", "安装模式"));
  const mode = select(document, "mode", [["trusted_local", "可信本地"], ["governed", "受治理签名安装"]], modeBox);
  const modeReason = input(document, "text", "modeReason", "变更原因", modeBox);
  modeReason.maxLength = 2000;
  const modeButton = el(document, "button", "确认变更模式", "button secondary");
  modeButton.type = "button";
  modeButton.dataset.capabilityAction = "mode";
  modeBox.append(modeButton, el(document, "p", "受治理模式只接受可信发布者签名和已确认权限；可信本地模式保留现有工作流。此处不提供操作系统隔离。", "muted"));
  const trustBox = el(document, "section", "", "capability-trust");
  trustBox.append(el(document, "h4", "可信发布者"));
  const publisherId = input(document, "text", "publisherId", "发布者标识", trustBox);
  const publisherName = input(document, "text", "publisherName", "名称", trustBox);
  const publisherKey = input(document, "text", "publisherKey", "Ed25519 公钥（Base64）", trustBox);
  const trustButton = el(document, "button", "核验后加入信任库", "button secondary");
  trustButton.type = "button";
  trustButton.dataset.capabilityAction = "publisher";
  trustBox.append(trustButton, el(document, "p", "公钥身份由管理员核验，平台签名校验不代替机构身份核验。", "muted"));
  const publishers = el(document, "div", "", "capability-publishers");
  trustBox.append(publishers);
  const installBox = el(document, "section", "", "capability-install");
  installBox.append(el(document, "h4", "签名材料与权限提案"));
  const kind = select(document, "kind", [["plugin", "Plugin"], ["workflow", "Workflow 模板"]], installBox);
  const target = select(document, "target", [["", "新增能力"]], installBox);
  const file = input(document, "file", "file", "签名 ZIP 材料", installBox);
  file.accept = ".zip,application/zip";
  const proposeButton = el(document, "button", "验证并形成提案", "button primary");
  proposeButton.type = "button";
  proposeButton.dataset.capabilityAction = "propose";
  installBox.append(proposeButton, el(document, "p", "验证不会执行包代码。签名覆盖精确材料、工具、Hook、权限和版本；确认后才注册。", "muted"));
  const proposalBox = el(document, "section", "", "capability-proposal");
  const versions = el(document, "section", "", "capability-versions");
  container.append(status, controls, modeBox, trustBox, installBox, proposalBox, versions);

  function headers() {
    const token = getAdminToken ? getAdminToken() : admin.value;
    return { "x-marvis-plugin-admin": String(token || "") };
  }
  function request(url, options = {}) {
    const body = options.body;
    const serialized = body !== undefined && !(body instanceof FormData) && typeof body !== "string"
      ? JSON.stringify(body) : body;
    return api(url, { ...options, ...(body !== undefined ? { body: serialized } : {}) });
  }
  function setStatus(text, tone = "") {
    if (destroyed) return;
    status.textContent = text;
    status.dataset.tone = tone;
  }
  async function perform(action) {
    if (destroyed || busy) return;
    busy = true;
    container.setAttribute("aria-busy", "true");
    container.inert = true;
    const disabled = actionButtons.map((button) => [button, button.disabled]);
    actionButtons.forEach((button) => { button.disabled = true; });
    try {
      await action();
    } catch (error) {
      const code = typeof error?.detail === "string" ? error.detail : error?.message;
      setStatus(labels[code] || "操作未完成。请核对材料、权限与当前版本后重试。", "error");
    } finally {
      busy = false;
      container.setAttribute("aria-busy", "false");
      container.inert = false;
      disabled.forEach(([button, previous]) => { button.disabled = previous; });
    }
  }
  function currentHead(entryKind, name) {
    return overview?.versions.find((entry) => entry.kind === entryKind && entry.name === name && entry.active);
  }
  function renderTargets() {
    const previous = target.value;
    target.replaceChildren();
    for (const [value, label] of [["", "新增能力"], ...(overview?.versions || [])
      .filter((entry) => entry.kind === kind.value && entry.active)
      .map((entry) => [entry.name, `${entry.name} · 当前 ${entry.version}`])]) {
      const option = el(document, "option", label);
      option.value = value;
      target.append(option);
    }
    if ([...target.options].some((option) => option.value === previous)) target.value = previous;
  }
  function renderProposal() {
    proposalBox.replaceChildren();
    if (!proposal) return;
    proposalBox.append(el(document, "h4", `${proposal.name} · ${proposal.version} 待确认`),
      el(document, "p", `发布者 ${proposal.publisher_id} · 当前版本序号 ${proposal.expected_revision}`),
      el(document, "p", `材料 ${proposal.content_hash}`, "capability-hash"),
      el(document, "p", `新增权限：${proposal.delta.added_permissions.join("、") || "无"}；移除权限：${proposal.delta.removed_permissions.join("、") || "无"}`));
    const declarations = el(document, "details");
    declarations.append(el(document, "summary", "核对完整能力声明"), el(document, "pre", JSON.stringify(proposal.declaration, null, 2)));
    proposalBox.append(declarations);
    const checkedPermissions = [];
    for (const permission of proposal.permissions) {
      const check = input(document, "checkbox", `permission:${permission}`, permission, proposalBox);
      check.value = permission;
      checkedPermissions.push(check);
    }
    const reviewed = input(document, "checkbox", "reviewed", "我已核对材料版本、能力声明和权限差量", proposalBox);
    const confirmButton = el(document, "button", "确认权限并切换版本", "button primary");
    confirmButton.type = "button";
    confirmButton.dataset.capabilityAction = "confirm";
    confirmButton.disabled = true;
    const validate = () => { confirmButton.disabled = !reviewed.checked || checkedPermissions.some((check) => !check.checked); };
    reviewed.addEventListener("change", validate);
    for (const check of checkedPermissions) check.addEventListener("change", validate);
    confirmButton.addEventListener("click", () => perform(async () => {
      if (confirmButton.disabled) return;
      const binding = proposal;
      const receipt = await request(`${BASE}/proposals/${encodeURIComponent(binding.id)}/confirm`, {
        method: "POST", headers: headers(), body: {
          proposal_hash: binding.proposal_hash, expected_revision: binding.expected_revision,
          approved_permissions: checkedPermissions.map((check) => check.value),
        },
      });
      proposal = null;
      renderProposal();
      await refresh();
      setStatus(`${receipt.name} 已切换到 ${receipt.version}。既有计划保留其固定版本。`, "success");
      await onChanged(receipt);
    }));
    proposalBox.append(confirmButton);
    setStatus("签名已验证。请核对完整声明并逐项确认权限；材料尚未注册或执行。", "pending");
  }
  function renderOverview() {
    mode.value = overview.settings.mode;
    renderTargets();
    publishers.replaceChildren();
    for (const publisher of overview.publishers) {
      const row = el(document, "div", "", "capability-version-row");
      row.append(el(document, "span", `${publisher.display_name} · ${publisher.id} · ${publisher.revoked_at ? "已撤销" : "可信"}`));
      if (!publisher.revoked_at) {
        const reason = input(document, "text", `revoke:${publisher.id}`, "撤销原因", row);
        const button = el(document, "button", "撤销信任", "button secondary");
        button.type = "button";
        button.addEventListener("click", () => perform(async () => {
          if (!reason.value.trim()) { setStatus("请填写撤销原因。", "error"); return; }
          await request(`${BASE}/publishers/${encodeURIComponent(publisher.id)}/revoke`, {
            method: "POST", headers: headers(), body: { reason: reason.value },
          });
          await refresh();
          setStatus("发布者已撤销，保留版本停止接受新调用。", "success");
        }));
        row.append(button);
      }
      publishers.append(row);
    }
    versions.replaceChildren(el(document, "h4", "保留版本与回滚"));
    if (!overview.versions.length) versions.append(el(document, "p", "尚无已确认的签名版本。"));
    for (const version of overview.versions) {
      const row = el(document, "div", "", "capability-version-row");
      row.append(el(document, "span", `${version.kind === "plugin" ? "Plugin" : "Workflow"} · ${version.name} · ${version.version} · ${version.active ? (version.enabled ? "当前启用" : "当前停用") : "保留"}${version.publisher_revoked ? " · 发布者已撤销" : ""}`));
      row.append(el(document, "code", version.content_hash.slice(0, 16)));
      const button = el(document, "button", version.active ? (version.enabled ? "停用" : "启用") : "回滚到此版本", "button secondary");
      button.type = "button";
      button.disabled = Boolean(version.publisher_revoked && (!version.active || !version.enabled));
      button.addEventListener("click", () => perform(async () => {
        const head = currentHead(version.kind, version.name);
        if (!head) { setStatus("当前版本已变化，请刷新。", "error"); return; }
        if (version.active) {
          await request(`${BASE}/enabled`, { method: "POST", headers: headers(), body: {
            kind: version.kind, name: version.name, enabled: !version.enabled, expected_revision: head.head_revision,
          } });
          await refresh();
          setStatus("启用状态已更新。", "success");
        } else {
          proposal = await request(`${BASE}/rollback-proposals`, { method: "POST", headers: headers(), body: {
            kind: version.kind, name: version.name, version: version.version, expected_revision: head.head_revision,
          } });
          renderProposal();
        }
      }));
      row.append(button);
      versions.append(row);
    }
  }
  async function refresh() {
    const response = await request(BASE);
    if (destroyed) return;
    overview = response;
    renderOverview();
    if (!proposal) setStatus(`当前为${overview.settings.mode === "governed" ? "受治理签名安装" : "可信本地"}模式。`, "ready");
  }
  refreshButton.addEventListener("click", () => perform(refresh));
  kind.addEventListener("change", renderTargets);
  modeButton.addEventListener("click", () => perform(async () => {
    if (!overview || !modeReason.value.trim()) { setStatus("请填写模式变更原因。", "error"); return; }
    await request(`${BASE}/settings`, { method: "PUT", headers: headers(), body: {
      mode: mode.value, expected_revision: overview.settings.revision, reason: modeReason.value,
    } });
    await refresh();
    setStatus("安装模式已更新。", "success");
  }));
  trustButton.addEventListener("click", () => perform(async () => {
    await request(`${BASE}/publishers`, { method: "POST", headers: headers(), body: {
      id: publisherId.value, display_name: publisherName.value, public_key: publisherKey.value,
    } });
    publisherKey.value = "";
    await refresh();
    setStatus("可信发布者已保存。", "success");
  }));
  proposeButton.addEventListener("click", () => perform(async () => {
    if (!file.files?.[0]) { setStatus("请选择签名 ZIP 材料。", "error"); return; }
    const head = target.value ? currentHead(kind.value, target.value) : null;
    const form = new FormData();
    form.append("file", file.files[0]);
    const params = new URLSearchParams({ kind: kind.value, expected_revision: String(head?.head_revision || 0) });
    if (target.value) params.set("expected_name", target.value);
    proposal = await request(`${BASE}/proposals?${params}`, { method: "POST", headers: headers(), body: form });
    renderProposal();
  }));
  const actionButtons = [refreshButton, modeButton, trustButton, proposeButton];
  const ready = perform(refresh);
  return { refresh: () => perform(refresh), ready, destroy() { destroyed = true; admin.value = ""; container.replaceChildren(); } };
}
