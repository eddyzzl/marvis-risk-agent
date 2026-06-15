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
    if (pathPanel) pathPanel.hidden = !isPath;
    if (uploadPanel) uploadPanel.hidden = isPath;
  }

  function reset() {
    files = [];
    setMode("path");
    notifyFilesChanged();
  }

  function captureFiles(fileList) {
    files = Array.from(fileList || []).map((file) => ({
      name: file.name || "未命名文件",
      size: Number(file.size || 0),
    }));
    notifyFilesChanged();
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
    dropzone.onclick = () => input.click();
    dropzone.onkeydown = (event) => {
      if (!["Enter", " "].includes(event.key)) return;
      event.preventDefault();
      input.click();
    };
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
    dropzone.ondrop = (event) => {
      event.preventDefault();
      captureFiles(event.dataTransfer?.files);
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
