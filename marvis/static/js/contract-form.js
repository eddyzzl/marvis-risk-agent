// Forms consume installed, server-issued schemas. They never execute schema text.
const labels = {
  profile: "数据服务配置", profile_id: "服务编号", mode: "接入方式", endpoint: "本机参考服务地址", token: "服务凭据（仅本机保存）",
  task_id: "任务编号", source_id: "事件来源编号", source_contract_hash: "来源合同指纹", grant_id: "授权编号", grantee_id: "被授权身份编号",
  subject_namespace: "主体标识命名空间", subject_token: "脱敏主体指纹", purpose: "使用目的", basis_id: "授权依据编号", reference: "依据引用", declaration: "依据性质",
  basis_artifact_id: "已登记授权依据产物编号", starts_at: "生效时间（含时区）", expires_at: "失效时间（含时区）", request_id: "本次请求编号",
  historical: "历史记录定位", dataset: "数据快照绑定", dataset_id: "已登记快照编号", expected_content_hash: "快照内容指纹", row_id_column: "记录编号列", row_id: "记录编号", response_column: "完整服务响应列",
  expires_in_seconds: "请求有效秒数", requests_per_minute: "每分钟请求上限", timeout_seconds: "单次超时秒数", breaker_threshold: "熔断失败次数", breaker_seconds: "熔断恢复秒数",
  entity_namespaces: "实体标识命名空间", event_types: "事件类型", numeric_fields: "整数值字段与单位", publisher_ref: "来源声明引用", permissions: "授权权限", json_column: "完整事件记录列", kind: "导入内容",
  contract: "事件计算口径", decision_at: "决策时间（含时区）", knowledge_cutoff: "信息可得截止时间（含时区）", availability_mode: "可得时间来源", window_seconds: "回看窗口秒数", include_current_event: "包含当前事件", current_event_id: "当前事件编号",
  focus_kind: "分析实体类型", focus: "脱敏分析实体", namespace: "命名空间", token: "脱敏指纹", window_features: "时间窗特征", relation_features: "关系特征", name: "特征名称", operation: "计算方式", field: "计算字段", target_kind: "关联实体类型", via_kind: "中间实体类型", policy_ref: "计算口径依据",
  raw_schema: "输入字段合同", operation_id: "操作编号", model_artifact_id: "已登记模型产物编号", promotion_id: "已批准晋级编号", install_operation_id: "已确认安装操作编号", deployment_operation_id: "已确认部署操作编号", expected_current_ref: "当前远端版本引用", probe_features: "健康检查输入", query: "数据查询", limit: "最大记录数",
};
const values = { historical_file: "已登记历史快照", reference_http: "本机参考 HTTP 服务", events: "事件记录", coverage: "覆盖声明", read: "读取", write: "写入", subject: "主体", device: "设备", account: "账户", platform_observed: "平台当时已观察", retrospective_declared: "来源回溯声明", count: "计数", sum: "整数求和", distinct: "实体去重计数", neighbor_count: "直接关联数", shared_neighbor_count: "共享实体关联数", kyc: "身份核验", credit_review: "信用审查", fraud_review: "反欺诈审查", synthetic_reference_test: "合成参考验收", deidentified_history_review: "脱敏历史复核", documented_authorization: "已留存正式授权", complete: "完整", partial: "部分", unknown: "未知" };
const absent = Symbol("absent");
const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const element = (document, tag, text) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  return node;
};

export function createContractForm({ document = globalThis.document, schema, initial = {}, choices = {}, title = "填写合同", submitLabel = "提交", onSubmit = async () => {} }) {
  if (!schema || typeof schema !== "object") throw new Error("本机合同版本缺失");
  const form = element(document, "form");
  form.className = "contract-form";
  const definitions = schema.$defs || {};
  function resolve(input) {
    if (!input.$ref) return input;
    if (!input.$ref.startsWith("#/$defs/")) throw new Error("合同包含不受支持的引用");
    const name = input.$ref.slice(8);
    if (!own(definitions, name)) throw new Error("合同引用缺失");
    return { ...definitions[name], ...input, $ref: undefined };
  }
  function control(raw, key, value, path, required = true, depth = 0) {
    if (depth > 16) throw new Error("合同层级超出表单支持范围");
    let spec = resolve(raw);
    const wrapper = element(document, "div");
    wrapper.className = "contract-field";
    wrapper.dataset.contractPath = path;
    const label = element(document, "label", key === "" ? title : labels[key] || spec.title || key);
    wrapper.append(label);
    let enabled = true, toggle = null;
    const branches = spec.anyOf || spec.oneOf;
    const nullable = spec.type === "null" || branches?.some((s) => resolve(s).type === "null");
    let unionVariants = null;
    if (branches) {
      const variants = branches.filter((s) => resolve(s).type !== "null");
      if (variants.length === 1) spec = { ...resolve(variants[0]), ...Object.fromEntries(Object.entries(spec).filter(([k]) => !["anyOf", "oneOf", "type"].includes(k))) };
      else unionVariants = variants.map(resolve);
    }
    if (!required || nullable) {
      toggle = element(document, "input");
      toggle.type = "checkbox";
      toggle.dataset.contractEnabled = path;
      enabled = value !== undefined && value !== null;
      toggle.checked = enabled;
      label.prepend(toggle);
      label.append(element(document, "small", nullable ? "（未勾选保留为空）" : "（可选）"));
    }
    const body = element(document, "fieldset");
    body.className = "contract-input";
    wrapper.append(body);
    const activate = () => { enabled = toggle ? toggle.checked : true; body.disabled = !enabled; body.hidden = !enabled; };
    if (toggle) toggle.addEventListener("change", activate);
    let read;
    if (unionVariants) {
      wrapper.classList.add("contract-complex");
      const select = element(document, "select"), slot = element(document, "div"); body.append(select, slot);
      unionVariants.forEach((variant, index) => { const option = element(document, "option", { number: "数值", integer: "整数", string: "文本", boolean: "是 / 否", object: "对象", array: "列表" }[variant.type] || variant.title || `类型 ${index + 1}`); option.value = String(index); select.append(option); });
      const initialIndex = unionVariants.findIndex((variant) => variant.type === typeof value || variant.type === "integer" && Number.isInteger(value));
      select.value = String(Math.max(0, initialIndex));
      let child;
      const change = (seed) => { child = control(unionVariants[Number(select.value)], key, seed, path, true, depth + 1); slot.replaceChildren(child.node); };
      change(value); select.addEventListener("change", () => change(undefined)); read = () => child.read();
    } else if (spec.type === "object" || spec.properties) {
      wrapper.classList.add("contract-complex"); body.classList.add("contract-object");
      const children = [], properties = spec.properties || {};
      const declared = spec.required || [];
      for (const [name, child] of Object.entries(properties)) {
        const seed = value && own(value, name) ? value[name] : own(child, "default") ? child.default : undefined;
        const item = control(child, name, seed, path ? `${path}.${name}` : name, declared.includes(name), depth + 1);
        children.push([name, item]); body.append(item.node);
      }
      const rows = [];
      if (spec.additionalProperties === true || spec.additionalProperties && typeof spec.additionalProperties === "object") {
        const list = element(document, "div"); body.append(list);
        function add(name = "", seed) {
          const maximum = spec.maxProperties || 1000;
          if (rows.length >= maximum) throw new Error(`最多允许 ${maximum} 项`);
          const row = element(document, "div"); row.className = "contract-map-row";
          const nameInput = element(document, "input"); nameInput.placeholder = "字段名称"; nameInput.value = name; nameInput.required = true;
          const childSchema = spec.additionalProperties === true ? { anyOf: [{ type: "number" }, { type: "string" }, { type: "boolean" }, { type: "null" }] } : spec.additionalProperties;
          const child = control(childSchema, "值", seed, `${path}.${name || rows.length}`, true, depth + 1);
          row.append(nameInput, child.node);
          const remove = element(document, "button", "移除"); remove.type = "button"; row.append(remove);
          const item = { row, nameInput, child }; rows.push(item); list.append(row);
          remove.addEventListener("click", () => { rows.splice(rows.indexOf(item), 1); row.remove(); });
        }
        Object.entries(value || {}).filter(([k]) => !own(properties, k)).forEach(([k, v]) => add(k, v));
        const button = element(document, "button", "添加字段"); button.type = "button"; body.append(button);
        button.addEventListener("click", () => { try { add(); } catch (e) { status.textContent = e.message; } });
      }
      read = () => {
        const result = Object.create(null);
        for (const [name, child] of children) { const resultValue = child.read(); if (resultValue !== absent) result[name] = resultValue; }
        for (const item of rows) {
          const name = item.nameInput.value.trim();
          if (!name || own(result, name) || ["__proto__", "constructor", "prototype"].includes(name)) throw new Error("字段名称为空、重复或无效");
          result[name] = item.child.read();
        }
        return result;
      };
    } else if (spec.type === "array") {
      wrapper.classList.add("contract-complex");
      const list = element(document, "div"), rows = []; body.append(list);
      const add = (seed) => {
        const maximum = spec.maxItems || 1000;
        if (rows.length >= maximum) throw new Error(`最多允许 ${maximum} 项`);
        const row = element(document, "div"); row.className = "contract-array-row";
        const child = control(spec.items || {}, key, seed, `${path}.${rows.length}`, true, depth + 1);
        row.append(child.node);
        const remove = element(document, "button", "移除"); remove.type = "button"; row.append(remove);
        const item = { row, child }; rows.push(item); list.append(row);
        remove.addEventListener("click", () => { rows.splice(rows.indexOf(item), 1); row.remove(); });
      };
      (Array.isArray(value) ? value : []).forEach(add);
      const button = element(document, "button", "添加一项"); button.type = "button"; body.append(button);
      button.addEventListener("click", () => { try { add(); } catch (e) { status.textContent = e.message; } });
      read = () => { if (rows.length < (spec.minItems || 0)) throw new Error(`${labels[key] || key}至少需要 ${spec.minItems} 项`); return rows.map((item) => item.child.read()); };
    } else {
      let input;
      const enumValues = own(spec, "const") ? [spec.const] : spec.enum;
      const suppliedChoices = choices[path] || choices[key];
      if (enumValues || suppliedChoices) {
        input = element(document, "select");
        input.append(element(document, "option", "请选择"));
        for (const option of enumValues ? enumValues.map((v) => ({ value: v, label: values[v] || String(v) })) : suppliedChoices) {
          const node = element(document, "option", option.label); node.value = String(option.value); input.append(node);
        }
        if (own(spec, "const")) value = spec.const;
      } else { input = element(document, "input"); input.type = spec.type === "boolean" ? "checkbox" : ["number", "integer"].includes(spec.type) ? "number" : key === "token" && path === "token" ? "password" : "text"; }
      input.dataset.contractInput = path;
      input.autocomplete = "off";
      if (spec.type === "boolean") input.checked = value === true;
      else if (value !== undefined && value !== null) input.value = String(value);
      // JSON Schema searches for a match; HTML pattern requires a full match.
      // Preserve the server schema's search semantics for unanchored patterns.
      if (spec.pattern) input.pattern = `(?:[\\s\\S]*(?:${spec.pattern})[\\s\\S]*)`;
      if (spec.maxLength) input.maxLength = spec.maxLength;
      if (spec.minLength) input.minLength = spec.minLength;
      if (spec.minimum !== undefined) input.min = spec.minimum;
      if (spec.maximum !== undefined) input.max = spec.maximum;
      input.step = spec.type === "integer" ? "1" : "any";
      input.required = spec.type !== "boolean";
      label.htmlFor = input.id = `contract-${createContractForm.sequence++}`;
      body.append(input);
      read = () => {
        if (spec.type === "boolean") return input.checked;
        if (input.value === "") throw new Error(`请填写${labels[key] || key}`);
        if (["number", "integer"].includes(spec.type)) {
          const number = Number(input.value);
          if (!Number.isFinite(number) || (spec.type === "integer" && !Number.isSafeInteger(number))) throw new Error("数值超出可精确输入范围，请使用已登记数据快照");
          return number;
        }
        if (enumValues) return enumValues.find((v) => String(v) === input.value);
        return input.value;
      };
    }
    activate();
    return { node: wrapper, read: () => enabled ? read() : nullable ? null : absent };
  }
  const root = control(schema, "", initial, "", true);
  form.append(root.node);
  const status = element(document, "p"); status.setAttribute("role", "status");
  const button = element(document, "button", submitLabel); button.type = "submit"; button.className = "button compact primary";
  form.append(button, status);
  let pending = false;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pending) return;
    try {
      if (!form.reportValidity()) return;
      const payload = root.read(); pending = true; button.disabled = true; status.textContent = "正在提交…";
      await onSubmit(payload); status.textContent = "已保存，可查看回执。";
    } catch (error) { status.textContent = error?.detail?.code || error.message || "提交失败，已填写内容保留。"; }
    finally { pending = false; button.disabled = false; }
  });
  return { element: form, read: root.read, status, submitButton: button };
}
createContractForm.sequence = 1;
