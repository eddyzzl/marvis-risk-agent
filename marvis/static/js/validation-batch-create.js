export const MIN_VALIDATION_BATCH_ROWS = 2;
export const MAX_VALIDATION_BATCH_ROWS = 10;

const FILE_ROLE_ORDER = ["notebook", "sample", "pmml", "dictionary"];

export const VALIDATION_BATCH_FILE_RULES = {
  notebook: {
    label: "Notebook",
    accept: ".ipynb",
    extensions: [".ipynb"],
    hint: "Jupyter Notebook（.ipynb）",
  },
  sample: {
    label: "验证样本",
    accept: ".csv,.parquet,.feather,.xlsx,.xls",
    extensions: [".csv", ".parquet", ".feather", ".xlsx", ".xls"],
    hint: "CSV、Parquet、Feather 或 Excel",
  },
  pmml: {
    label: "PMML 模型",
    accept: ".pmml",
    extensions: [".pmml"],
    hint: "PMML 模型文件（.pmml）",
  },
  dictionary: {
    label: "数据字典",
    accept: ".csv,.parquet,.xlsx,.xls",
    extensions: [".csv", ".parquet", ".xlsx", ".xls"],
    hint: "CSV、Parquet 或 Excel",
  },
};

function normalizedText(value) {
  return String(value || "").trim();
}

function fileExtension(file) {
  const name = normalizedText(file?.name).toLowerCase();
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot) : "";
}

const DICTIONARY_NAME_PATTERN = /字典|dict|dictionary|metadata|meta|feature/i;
const SAMPLE_NAME_PATTERN = /样本|sample|train|oot|data/i;

// One file's likely role, from its extension and (for tables) its name. A
// table that names neither a dictionary nor a sample counts as a sample,
// the same fallback classifyValidationBatchFiles uses.
export function validationFileRole(file) {
  const ext = fileExtension(file);
  if (VALIDATION_BATCH_FILE_RULES.notebook.extensions.includes(ext)) return "notebook";
  if (VALIDATION_BATCH_FILE_RULES.pmml.extensions.includes(ext)) return "pmml";
  const name = normalizedText(file?.name);
  if (VALIDATION_BATCH_FILE_RULES.dictionary.extensions.includes(ext) && DICTIONARY_NAME_PATTERN.test(name)) {
    return "dictionary";
  }
  if (VALIDATION_BATCH_FILE_RULES.sample.extensions.includes(ext)) return "sample";
  return "";
}

export function classifyValidationBatchFiles(files = []) {
  const list = Array.from(files || []).filter(Boolean);
  const assigned = {
    notebook: null,
    sample: null,
    pmml: null,
    dictionary: null,
  };
  const isAssigned = (file) => Object.values(assigned).includes(file);
  const take = (role, predicate) => {
    if (assigned[role]) return;
    const match = list.find((file) => !isAssigned(file) && predicate(file));
    if (match) assigned[role] = match;
  };

  take("notebook", (file) => fileExtension(file) === ".ipynb");
  take("pmml", (file) => fileExtension(file) === ".pmml");
  take("dictionary", (file) => {
    const ext = fileExtension(file);
    return VALIDATION_BATCH_FILE_RULES.dictionary.extensions.includes(ext)
      && DICTIONARY_NAME_PATTERN.test(normalizedText(file.name));
  });
  take("sample", (file) => {
    const ext = fileExtension(file);
    return VALIDATION_BATCH_FILE_RULES.sample.extensions.includes(ext)
      && SAMPLE_NAME_PATTERN.test(normalizedText(file.name));
  });
  take("sample", (file) => (
    VALIDATION_BATCH_FILE_RULES.sample.extensions.includes(fileExtension(file))
  ));
  take("dictionary", (file) => (
    VALIDATION_BATCH_FILE_RULES.dictionary.extensions.includes(fileExtension(file))
  ));
  if (!assigned.notebook || !assigned.sample || !assigned.pmml || !assigned.dictionary) {
    return null;
  }
  if (assigned.sample === assigned.dictionary) return null;
  return assigned;
}

function errorMessage(error, fallback = "请求失败") {
  const message = normalizedText(error?.message || error?.detail || error);
  return message || fallback;
}

function safeUploadFilename(file, role) {
  const rawName = normalizedText(file?.name) || `${role}.bin`;
  const basename = rawName.split(/[\\/]/).pop() || `${role}.bin`;
  const safeName = basename.replace(/[\u0000-\u001f\u007f]/g, "_").replace(/^\.+$/, "file");
  return safeName || `${role}.bin`;
}

function validationRowKey(row, index) {
  return normalizedText(row?.id) || `row-${index + 1}`;
}

export function validationBatchRelativePaths(files = {}) {
  return Object.fromEntries(
    FILE_ROLE_ORDER.map((role) => [
      role,
      `${role}/${safeUploadFilename(files[role], role)}`,
    ]),
  );
}

export function validateValidationBatchDraft(draft = {}) {
  const formErrors = [];
  const rowErrors = {};
  const rows = Array.isArray(draft.rows) ? draft.rows : [];

  if (!normalizedText(draft.batchName)) formErrors.push("请填写批次名称。");
  if (!normalizedText(draft.validator)) formErrors.push("请填写验证人员。");
  if (rows.length < MIN_VALIDATION_BATCH_ROWS || rows.length > MAX_VALIDATION_BATCH_ROWS) {
    formErrors.push("每个批次必须包含 2 至 10 个模型。");
  }

  rows.forEach((row, index) => {
    const errors = [];
    const rowKey = validationRowKey(row, index);
    if (!normalizedText(row?.modelName)) errors.push("请填写模型名称。");
    if (normalizedText(row?.sourceDir)) {
      if (errors.length) rowErrors[rowKey] = errors;
      return;
    }
    FILE_ROLE_ORDER.forEach((role) => {
      const rule = VALIDATION_BATCH_FILE_RULES[role];
      const file = row?.files?.[role];
      if (!file) {
        errors.push(`请上传${rule.label}。`);
        return;
      }
      if (!rule.extensions.includes(fileExtension(file))) {
        errors.push(`${rule.label}仅支持 ${rule.extensions.join("、")}。`);
      }
      if (Number(file.size) === 0) errors.push(`${rule.label}不能为空文件。`);
    });
    if (errors.length) rowErrors[rowKey] = errors;
  });

  return {
    valid: formErrors.length === 0 && Object.keys(rowErrors).length === 0,
    formErrors,
    rowErrors,
  };
}

export function buildValidationBatchUploadFormData(row) {
  const formData = new FormData();
  const relativePaths = validationBatchRelativePaths(row?.files);
  FILE_ROLE_ORDER.forEach((role) => {
    const file = row.files[role];
    formData.append("files", file, file.name);
    formData.append("relative_paths", relativePaths[role]);
  });
  return formData;
}

function uploadFingerprint(row) {
  return FILE_ROLE_ORDER.map((role) => {
    const file = row?.files?.[role];
    return [role, file?.name || "", file?.size ?? "", file?.lastModified ?? ""].join(":");
  }).join("|");
}

function batchItemFromUpload(row, upload, relativePaths) {
  const sourceDir = normalizedText(upload?.source_dir);
  if (!sourceDir) throw new Error("材料上传响应缺少 source_dir，尚未创建批次。");
  const uploadToken = normalizedText(upload?.upload_token);
  if (!uploadToken) throw new Error("材料上传响应缺少 upload_token，尚未创建批次。");
  return {
    model_name: normalizedText(row.modelName),
    model_version: normalizedText(row.modelVersion),
    source_dir: sourceDir,
    upload_token: uploadToken,
    notebook_path: relativePaths.notebook,
    sample_path: relativePaths.sample,
    pmml_path: relativePaths.pmml,
    dictionary_path: relativePaths.dictionary,
  };
}

export async function submitValidationBatchDraft(
  draft,
  {
    uploadMaterials,
    createBatch,
    cleanupMaterials,
    onRowStatus = () => {},
  } = {},
) {
  const validation = validateValidationBatchDraft(draft);
  if (!validation.valid) {
    const error = new Error("请先修正批次创建表单中的必填项或文件类型。");
    error.name = "ValidationBatchDraftError";
    error.validation = validation;
    throw error;
  }
  if (typeof uploadMaterials !== "function" || typeof createBatch !== "function") {
    throw new Error("批次创建服务尚未初始化。");
  }

  const items = [];
  const acquiredUploads = [];
  try {
    for (let index = 0; index < draft.rows.length; index += 1) {
      const row = draft.rows[index];
      const rowId = validationRowKey(row, index);
      if (normalizedText(row.sourceDir) && !row.files?.notebook) {
        onRowStatus(rowId, "uploaded", "已使用材料目录，等待创建批次。");
        items.push({
          model_name: normalizedText(row.modelName),
          model_version: normalizedText(row.modelVersion) || "v1",
          source_dir: normalizedText(row.sourceDir),
          notebook_path: "",
          sample_path: "",
          pmml_path: "",
          dictionary_path: "",
        });
        continue;
      }
      const relativePaths = validationBatchRelativePaths(row.files);
      const fingerprint = uploadFingerprint(row);
      let upload = row.uploadCache?.fingerprint === fingerprint
        ? row.uploadCache.payload
        : null;

      if (!upload) {
        onRowStatus(
          rowId,
          "uploading",
          `正在上传 4 份材料（第 ${index + 1}/${draft.rows.length} 个模型）…`,
        );
        try {
          upload = await uploadMaterials({
            row,
            index,
            total: draft.rows.length,
            relativePaths,
          });
          if (!normalizedText(upload?.source_dir)) {
            throw new Error("材料上传响应缺少 source_dir");
          }
          if (!normalizedText(upload?.upload_token)) {
            throw new Error("材料上传响应缺少 upload_token");
          }
          row.uploadCache = { fingerprint, payload: upload };
        } catch (error) {
          onRowStatus(rowId, "error", `材料上传失败：${errorMessage(error)}`);
          throw error;
        }
      }
      acquiredUploads.push(upload);
      onRowStatus(rowId, "uploaded", "4 份材料已上传，等待创建批次。");
      items.push(batchItemFromUpload(row, upload, relativePaths));
    }

    const result = await createBatch({
      batch_name: normalizedText(draft.batchName),
      validator: normalizedText(draft.validator),
      run_mode: "agent",
      items,
    });
    draft.rows.forEach((row) => { row.uploadCache = null; });
    return result;
  } catch (error) {
    const tokens = [...new Set(
      acquiredUploads
        .map((upload) => normalizedText(upload?.upload_token))
        .filter(Boolean),
    )];
    if (typeof cleanupMaterials === "function") {
      await Promise.allSettled(tokens.map((token) => cleanupMaterials(token)));
    }
    draft.rows.forEach((row) => { row.uploadCache = null; });
    throw error;
  }
}
