import { escapeHtml } from "./ui-utils.js";

export function renderAgentMarkdown(content) {
  const lines = String(content || "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let listType = "";
  let codeLines = null;
  let codeLanguage = "";

  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = "";
  };
  const openList = (type, start = "") => {
    if (listType === type) return;
    closeList();
    listType = type;
    const startNumber = Number.parseInt(start, 10);
    const startAttr = type === "ol" && Number.isFinite(startNumber) && startNumber > 1
      ? ` start="${startNumber}"`
      : "";
    html.push(`<${type}${startAttr}>`);
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim().startsWith("```")) {
      const language = line.trim().slice(3).trim().split(/\s+/)[0] || "";
      if (codeLines) {
        html.push(renderMarkdownCodeBlock(codeLines, codeLanguage));
        codeLines = null;
        codeLanguage = "";
      } else {
        closeList();
        codeLines = [];
        codeLanguage = normalizeMarkdownCodeLanguage(language);
      }
      continue;
    }
    if (codeLines) {
      codeLines.push(line);
      continue;
    }
    if (!line.trim()) {
      closeList();
      continue;
    }
    if (splitMarkdownTableRow(line).length > 1 && isMarkdownTableDivider(lines[index + 1])) {
      closeList();
      const headerCells = splitMarkdownTableRow(line);
      const bodyRows = [];
      index += 2;
      while (index < lines.length && splitMarkdownTableRow(lines[index]).length > 1) {
        if (isMarkdownTableDivider(lines[index])) break;
        bodyRows.push(splitMarkdownTableRow(lines[index]));
        index += 1;
      }
      index -= 1;
      html.push(renderMarkdownTable(headerCells, bodyRows));
      continue;
    }
    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      closeList();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderMarkdownInline(heading[2])}</h${level}>`);
      continue;
    }
    const unordered = /^\s*[-*]\s+(.+)$/.exec(line);
    if (unordered) {
      openList("ul");
      html.push(`<li>${renderMarkdownInline(unordered[1])}</li>`);
      continue;
    }
    const ordered = /^\s*(\d+)\.\s+(.+)$/.exec(line);
    if (ordered) {
      openList("ol", ordered[1]);
      html.push(`<li>${renderMarkdownInline(ordered[2])}</li>`);
      continue;
    }
    const quote = /^\s*>\s+(.+)$/.exec(line);
    if (quote) {
      closeList();
      html.push(`<blockquote>${renderMarkdownInline(quote[1])}</blockquote>`);
      continue;
    }
    closeList();
    html.push(`<p>${renderMarkdownInline(line)}</p>`);
  }
  if (codeLines) html.push(renderMarkdownCodeBlock(codeLines, codeLanguage));
  closeList();
  return `<div class="agent-markdown">${html.join("")}</div>`;
}

export function normalizeMarkdownCodeLanguage(language) {
  const normalized = String(language || "").trim().toLowerCase().replace(/[^a-z0-9_+.-]/g, "");
  return {
    py: "python",
    js: "javascript",
    ts: "typescript",
    sh: "bash",
    shell: "bash",
    yml: "yaml",
  }[normalized] || normalized;
}

export function renderMarkdownCodeBlock(codeLines, language = "") {
  const normalizedLanguage = normalizeMarkdownCodeLanguage(language);
  const languageClass = normalizedLanguage ? ` class="language-${escapeHtml(normalizedLanguage)}"` : "";
  return `<pre><code${languageClass}>${highlightMarkdownCode(codeLines.join("\n"), normalizedLanguage)}</code></pre>`;
}

export function highlightMarkdownCode(code, language = "") {
  return String(code || "")
    .split("\n")
    .map((line) => highlightMarkdownCodeLine(line, language))
    .join("\n");
}

export function highlightMarkdownCodeLine(line, language = "") {
  const keywordSet = markdownCodeKeywordSet(language);
  const commentMarkers = markdownCodeCommentMarkers(language);
  const segments = [];
  let index = 0;
  while (index < line.length) {
    const commentMarker = commentMarkers.find((marker) => line.startsWith(marker, index));
    if (commentMarker) {
      segments.push(markdownCodeToken("comment", line.slice(index)));
      break;
    }
    const char = line[index];
    if (char === '"' || char === "'" || (char === "`" && ["javascript", "typescript"].includes(language))) {
      const endIndex = findMarkdownCodeStringEnd(line, index, char);
      segments.push(markdownCodeToken("string", line.slice(index, endIndex)));
      index = endIndex;
      continue;
    }
    const number = /^\d+(?:\.\d+)?\b/.exec(line.slice(index));
    if (number) {
      segments.push(markdownCodeToken("number", number[0]));
      index += number[0].length;
      continue;
    }
    const word = /^[A-Za-z_][A-Za-z0-9_]*/.exec(line.slice(index));
    if (word) {
      const value = word[0];
      const nextChar = line.slice(index + value.length).trimStart()[0] || "";
      if (keywordSet.has(value)) {
        segments.push(markdownCodeToken("keyword", value));
      } else if (nextChar === "(") {
        segments.push(markdownCodeToken("function", value));
      } else {
        segments.push(escapeHtml(value));
      }
      index += value.length;
      continue;
    }
    segments.push(escapeHtml(char));
    index += 1;
  }
  return segments.join("");
}

export function findMarkdownCodeStringEnd(line, startIndex, quote) {
  let index = startIndex + 1;
  while (index < line.length) {
    if (line[index] === "\\" && index + 1 < line.length) {
      index += 2;
      continue;
    }
    if (line[index] === quote) return index + 1;
    index += 1;
  }
  return line.length;
}

export function markdownCodeToken(type, value) {
  return `<span class="agent-code-token ${type}">${escapeHtml(value)}</span>`;
}

export function markdownCodeKeywordSet(language = "") {
  const common = [
    "async", "await", "break", "case", "catch", "class", "const", "continue", "default",
    "else", "export", "false", "finally", "for", "from", "function", "if", "import",
    "in", "let", "new", "null", "return", "throw", "true", "try", "undefined", "var",
    "while",
  ];
  const python = [
    "and", "as", "def", "elif", "except", "False", "for", "from", "if", "import", "in",
    "is", "lambda", "None", "not", "or", "pass", "raise", "return", "True", "with", "yield",
  ];
  const sql = [
    "and", "as", "by", "case", "desc", "else", "end", "from", "group", "having", "in",
    "inner", "insert", "join", "left", "limit", "not", "null", "on", "or", "order",
    "outer", "right", "select", "then", "update", "when", "where",
  ];
  const yaml = ["false", "null", "true"];
  if (language === "python") return new Set([...common, ...python]);
  if (language === "sql") return new Set([...common, ...sql, ...sql.map((word) => word.toUpperCase())]);
  if (language === "yaml") return new Set([...common, ...yaml]);
  return new Set(common);
}

export function markdownCodeCommentMarkers(language = "") {
  if (["python", "bash", "yaml"].includes(language)) return ["#"];
  if (language === "sql") return ["--"];
  return ["//"];
}

export function splitMarkdownTableRow(line) {
  const value = String(line || "").trim();
  if (!value.includes("|")) return [];
  const trimmed = value.replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

export function isMarkdownTableDivider(line) {
  const cells = splitMarkdownTableRow(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, "")));
}

export function normalizeMarkdownTableCells(cells, width) {
  const normalized = [...cells];
  while (normalized.length < width) normalized.push("");
  return normalized.slice(0, width);
}

export function renderMarkdownTable(headerCells, bodyRows) {
  const width = headerCells.length;
  const header = normalizeMarkdownTableCells(headerCells, width)
    .map((cell) => `<th>${renderMarkdownInline(cell)}</th>`)
    .join("");
  const rows = bodyRows.map((row) => {
    const cells = normalizeMarkdownTableCells(row, width)
      .map((cell) => `<td>${renderMarkdownInline(cell)}</td>`)
      .join("");
    return `<tr>${cells}</tr>`;
  });
  return [
    '<div class="agent-markdown-table-wrap">',
    "<table>",
    `<thead><tr>${header}</tr></thead>`,
    `<tbody>${rows.join("")}</tbody>`,
    "</table>",
    "</div>",
  ].join("");
}

export function renderMarkdownInline(content) {
  return String(content || "")
    .split(/(`[^`]*`)/g)
    .map((segment) => {
      if (segment.startsWith("`") && segment.endsWith("`") && segment.length >= 2) {
        return `<code>${escapeHtml(segment.slice(1, -1))}</code>`;
      }
      return renderMarkdownInlineText(segment);
    })
    .join("");
}

export function renderMarkdownInlineText(content) {
  const text = String(content || "");
  const linkPattern = /\[([^\]\n]+)\]\(([^)\s]+)\)/g;
  const parts = [];
  let cursor = 0;
  for (const match of text.matchAll(linkPattern)) {
    const href = match[2] || "";
    parts.push(renderMarkdownEmphasisText(text.slice(cursor, match.index)));
    if (isSafeMarkdownHref(href)) {
      parts.push(markdownAnchorHtml(match[1], href));
    } else {
      parts.push(renderMarkdownEmphasisText(match[1]));
    }
    cursor = match.index + match[0].length;
  }
  parts.push(renderMarkdownEmphasisText(text.slice(cursor)));
  return parts.join("");
}

export function renderMarkdownEmphasisText(content) {
  return escapeHtml(content)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n]+?)__/g, (match, value, offset, source) =>
      hasMarkdownBoundaries(source, offset, match.length) ? `<strong>${value}</strong>` : match
    )
    .replace(/\*([^*\n]+?)\*/g, (match, value, offset, source) =>
      hasMarkdownBoundaries(source, offset, match.length) ? `<em>${value}</em>` : match
    )
    .replace(/_([^_\n]+?)_/g, (match, value, offset, source) =>
      hasMarkdownBoundaries(source, offset, match.length) ? `<em>${value}</em>` : match
    );
}

export function markdownAnchorHtml(label, href) {
  return `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
}

export function isSafeMarkdownHref(href) {
  // Reject protocol-relative URLs ("//evil.com") — browsers resolve them to an
  // external https origin, so they must not pass as same-origin "/" links.
  if (href.startsWith("//")) return false;
  return /^https?:\/\//i.test(href) || href.startsWith("/") || href.startsWith("#");
}

export function hasMarkdownBoundaries(source, offset, length) {
  return isMarkdownBoundary(source[offset - 1] || "") && isMarkdownBoundary(source[offset + length] || "");
}

export function isMarkdownBoundary(value) {
  return !value || /[\s([{\u3000)\]},.;:!?，。；：！？]/.test(value);
}
