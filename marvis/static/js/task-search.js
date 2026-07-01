export function createTaskSearchController({
  getElementById,
  documentRef = document,
  windowRef = window,
  getQuery = () => "",
  setQuery = () => {},
  renderTaskList = () => {},
}) {
  let active = false;

  function openTaskSearch() {
    if (active) return;
    active = true;
    documentRef.body.classList.add("search-active");
    getElementById("taskSearchToggle").setAttribute("aria-expanded", "true");
    const input = getElementById("taskSearchInput");
    windowRef.requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
  }

  function closeTaskSearch({ focusToggle = false } = {}) {
    if (!active) return;
    active = false;
    documentRef.body.classList.remove("search-active");
    getElementById("taskSearchToggle").setAttribute("aria-expanded", "false");
    const input = getElementById("taskSearchInput");
    if (input.value || getQuery()) {
      input.value = "";
      setQuery("");
      renderTaskList();
    }
    if (focusToggle) getElementById("taskSearchToggle").focus();
  }

  function toggleTaskSearch() {
    if (active) {
      closeTaskSearch({ focusToggle: true });
    } else {
      openTaskSearch();
    }
  }

  return {
    closeTaskSearch,
    isActive: () => active,
    openTaskSearch,
    toggleTaskSearch,
  };
}
