import { $ } from "./ui-utils.js";

const browserChromeThemeColors = {
  light: "#ffffff",
  dark: "#181818",
};

function systemTheme() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function syncBrowserChromeTheme(theme) {
  const resolvedTheme = theme === "dark" ? "dark" : "light";
  const isDark = resolvedTheme === "dark";
  const themeColor = $("appThemeColor") || document.querySelector('meta[name="theme-color"]');
  if (themeColor) themeColor.setAttribute("content", browserChromeThemeColors[resolvedTheme]);
  $("brandFavicon")?.setAttribute("media", isDark ? "not all" : "all");
  $("brandFaviconDark")?.setAttribute("media", isDark ? "all" : "not all");
  $("brandAppleTouchIcon")?.setAttribute("media", isDark ? "not all" : "all");
  $("brandAppleTouchIconDark")?.setAttribute("media", isDark ? "all" : "not all");
}

export function createThemeController({ onChange } = {}) {
  let preference = "light";
  let current = "light";

  const notify = () => {
    if (typeof onChange === "function") {
      onChange({ preference, current });
    }
  };

  const applyTheme = (theme) => {
    preference = ["light", "dark", "system"].includes(theme) ? theme : "light";
    current = preference === "system" ? systemTheme() : preference;
    document.body.dataset.theme = current;
    syncBrowserChromeTheme(current);
    $("themeModeLabel").textContent = "设置";
    try {
      localStorage.setItem("marvis_theme", preference);
    } catch (_) {
      // Local storage can be unavailable in restricted notebook browsers.
    }
    notify();
  };

  const restoreTheme = () => {
    try {
      applyTheme(localStorage.getItem("marvis_theme") || "light");
    } catch (_) {
      applyTheme("light");
    }
  };

  const watchSystemTheme = () => {
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!media?.addEventListener) return;
    media.addEventListener("change", () => {
      if (preference === "system") applyTheme("system");
    });
  };

  return {
    applyTheme,
    restoreTheme,
    watchSystemTheme,
    get current() {
      return current;
    },
    get preference() {
      return preference;
    },
  };
}
