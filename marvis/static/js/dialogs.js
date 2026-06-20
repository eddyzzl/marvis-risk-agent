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
    if (!isPath && uploadPanel) {
      uploadPanel.scrollIntoView({ block: "nearest" });
    }
  }

  function reset() {
    files = [];
    setMode("path");
    notifyFilesChanged();
  }

  function captureFiles(fileList) {
    files = Array.from(fileList || []).map((file) => {
      const relativePath = (file.webkitRelativePath || file.name || "未命名文件")
        .replace(/^\/+/, "");
      return {
        file,
        name: file.name || "未命名文件",
        relativePath,
        size: Number(file.size || 0),
      };
    });
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
    const folderInput = $("materialFolderUploadInput");
    const fileButton = $("materialUploadFileButton");
    const folderButton = $("materialUploadFolderButton");
    const dropzone = document.querySelector(".material-upload-dropzone");
    if (!input || !dropzone) return;
    dropzone.onclick = (event) => {
      if (event.target.closest("button")) return;
      input.click();
    };
    if (fileButton) {
      fileButton.onclick = (event) => {
        event.stopPropagation();
        input.click();
      };
    }
    if (folderButton && folderInput) {
      folderButton.onclick = (event) => {
        event.stopPropagation();
        folderInput.click();
      };
    }
    input.onchange = () => captureFiles(input.files);
    if (folderInput) {
      folderInput.onchange = () => captureFiles(folderInput.files);
    }
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
