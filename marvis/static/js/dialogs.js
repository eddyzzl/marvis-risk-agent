function isDialogLike(value) {
  if (!value) return false;
  if (typeof HTMLDialogElement !== "undefined") return value instanceof HTMLDialogElement;
  return typeof value.close === "function";
}

export function closeDialogOnBackdropClick(event) {
  const dialog = event.currentTarget;
  if (!isDialogLike(dialog)) return;
  if (event.target !== dialog || !dialog.open) return;
  dialog.close();
}

export function bindDialogBackdropDismissal({ root = document } = {}) {
  root.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", closeDialogOnBackdropClick);
  });
}

export function materialUploadSelectionText(files = []) {
  if (!files.length) return "请选择文件或文件夹。";
  const names = files
    .slice(0, 3)
    .map((file) => file?.name || file?.relativePath || "未命名文件")
    .join("、");
  const suffix = files.length > 3 ? ` 等 ${files.length} 个文件` : "";
  const folderCount = new Set(
    files
      .map((file) => (file?.relativePath || "").split("/").slice(0, -1).join("/"))
      .filter(Boolean),
  ).size;
  const folderText = folderCount > 0 ? `，包含 ${folderCount} 个目录` : "";
  return `已选择 ${names}${suffix}${folderText}。`;
}

export function renderMaterialUploadSelection({
  files = [],
  getElementById = (id) => document.getElementById(id),
} = {}) {
  const status = getElementById("materialUploadStatus");
  if (!status) return;
  status.textContent = materialUploadSelectionText(files);
}

export function createMaterialSourceController({ $, onFilesChanged }) {
  let mode = "path";
  let files = [];

  function setMode(nextMode) {
    mode = nextMode === "upload" ? "upload" : "path";
    const isPath = mode === "path";
    const pathTab = $("materialSourcePathTab");
    const uploadTab = $("materialSourceUploadTab");
    const pathPanel = $("materialSourcePathPanel");
    const uploadPanel = $("materialSourceUploadPanel");
    if (pathTab) {
      pathTab.classList.toggle("selected", isPath);
      pathTab.setAttribute("aria-selected", isPath ? "true" : "false");
    }
    if (uploadTab) {
      uploadTab.classList.toggle("selected", !isPath);
      uploadTab.setAttribute("aria-selected", isPath ? "false" : "true");
    }
    if (pathPanel) {
      pathPanel.hidden = !isPath;
      pathPanel.classList.toggle("hidden", !isPath);
    }
    if (uploadPanel) {
      uploadPanel.hidden = isPath;
      uploadPanel.classList.toggle("hidden", isPath);
    }
    if (!isPath && uploadPanel) {
      uploadPanel.scrollIntoView({ block: "nearest" });
    }
  }

  function reset() {
    files = [];
    setMode("path");
    notifyFilesChanged();
  }

  function fileItem(file, relativePath = "") {
    const fallbackName = file?.name || "未命名文件";
    return {
      file,
      name: fallbackName,
      relativePath: (relativePath || file?.webkitRelativePath || fallbackName).replace(/^\/+/, ""),
      size: Number(file?.size || 0),
    };
  }

  function captureFileItems(fileItems) {
    files = fileItems.map((item) => fileItem(item.file, item.relativePath));
    notifyFilesChanged();
  }

  function captureFiles(fileList) {
    captureFileItems(Array.from(fileList || []).map((file) => ({ file })));
  }

  function fileFromEntry(entry) {
    return new Promise((resolve, reject) => {
      entry.file(resolve, reject);
    });
  }

  function readDirectoryEntries(reader) {
    return new Promise((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
  }

  async function walkDroppedEntry(entry, parentPath = "") {
    if (!entry) return [];
    if (entry.isFile) {
      const file = await fileFromEntry(entry);
      return [{ file, relativePath: `${parentPath}${file.name || entry.name}` }];
    }
    if (!entry.isDirectory) return [];
    const reader = entry.createReader();
    const children = [];
    while (true) {
      const batch = await readDirectoryEntries(reader);
      if (!batch.length) break;
      children.push(...batch);
    }
    const nextPath = `${parentPath}${entry.name}/`;
    const nested = await Promise.all(children.map((child) => walkDroppedEntry(child, nextPath)));
    return nested.flat();
  }

  async function droppedFileItems(dataTransfer) {
    const entries = Array.from(dataTransfer?.items || [])
      .map((item) => (
        typeof item.webkitGetAsEntry === "function" ? item.webkitGetAsEntry() : null
      ))
      .filter(Boolean);
    if (entries.length) {
      const nested = await Promise.all(entries.map((entry) => walkDroppedEntry(entry)));
      const entryFiles = nested.flat();
      if (entryFiles.length) return entryFiles;
    }
    return Array.from(dataTransfer?.files || []).map((file) => ({ file }));
  }

  function selectedFiles() {
    return [...files];
  }

  function notifyFilesChanged() {
    if (typeof onFilesChanged === "function") {
      onFilesChanged(selectedFiles());
    }
  }

  function bindTabs() {
    const pathTab = $("materialSourcePathTab");
    const uploadTab = $("materialSourceUploadTab");
    if (!pathTab || !uploadTab) return;
    pathTab.onclick = () => setMode("path");
    uploadTab.onclick = () => setMode("upload");
  }

  function bindDropzone() {
    const input = $("materialUploadInput");
    const dropzone = document.querySelector(".material-upload-dropzone");
    if (!input || !dropzone) return;
    const openFilePicker = () => {
      input.click();
    };
    dropzone.addEventListener("click", openFilePicker);
    dropzone.addEventListener("keydown", (event) => {
      if (!["Enter", " "].includes(event.key)) return;
      event.preventDefault();
      openFilePicker();
    });
    input.onchange = () => captureFiles(input.files);
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, () => {
        dropzone.classList.remove("is-dragover");
      });
    });
    dropzone.ondrop = async (event) => {
      event.preventDefault();
      captureFileItems(await droppedFileItems(event.dataTransfer));
    };
  }

  return {
    bindDropzone,
    bindTabs,
    captureFiles,
    mode: () => mode,
    reset,
    selectedFiles,
    setMode,
  };
}
