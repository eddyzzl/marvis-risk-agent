from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from docx import Document
from docx.document import Document as DocumentType
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.table import CT_Tbl, CT_Tc
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph


_BROWSER_PREVIEW_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
}

_IMAGE_TYPE_LABELS = {
    "image/emf": "EMF",
    "image/x-emf": "EMF",
    "image/wmf": "WMF",
    "image/x-wmf": "WMF",
    "image/vnd.ms-photo": "WDP",
    "image/x-wdp": "WDP",
    "image/tiff": "TIFF",
}


def docx_to_html_preview(docx_path: Path) -> str:
    """Render a DOCX report as self-contained HTML for in-platform preview."""
    document = Document(docx_path)
    counters = {"paragraph": 0, "table": 0, "image": 0, "unsupported_image": 0}
    converter = _WordPreviewConverter(counters)
    article_html = "".join(
        part
        for block in converter.iter_block_items(document)
        if (part := converter.block_html(block))
    )
    css = _preview_css()
    image_summary = f'{counters["image"]} 张图片'
    if counters["unsupported_image"]:
        image_summary += f'，{counters["unsupported_image"]} 张图片仅在 Word 中保留'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(docx_path.name)} - HTML 预览</title>
  <style>
{css}
  </style>
</head>
<body>
  <main class="shell">
    <div class="toolbar">
      <strong>{html.escape(docx_path.name)}</strong>
      <span>由 DOCX 解析生成：{counters["paragraph"]} 个段落，{counters["table"]} 个表格，{image_summary}。已保留 Word 表格的横向/纵向合并；此预览用于看内容结构，非 Word 精确排版。</span>
    </div>
    <article>{article_html}</article>
  </main>
</body>
</html>
"""


class _WordPreviewConverter:
    def __init__(self, counters: dict[str, int]) -> None:
        self.counters = counters
        self.image_rid_to_data_uri: dict[str, str] = {}
        self.unsupported_image_rids: set[str] = set()

    def iter_block_items(self, parent: DocumentType | _Cell):
        if isinstance(parent, DocumentType):
            parent_elm = parent.element.body
            parent_obj: Any = parent
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
            parent_obj = parent
        else:
            raise TypeError(type(parent))

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent_obj)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent_obj)

    def block_html(self, block: Paragraph | Table) -> str:
        if isinstance(block, Paragraph):
            return self.paragraph_html(block)
        return self.table_html(block)

    def paragraph_html(self, paragraph: Paragraph, *, cell: bool = False) -> str:
        self.counters["paragraph"] += 1
        content = "".join(self.run_html(run, paragraph) for run in paragraph.runs)
        if not content.strip():
            return '<p class="cell-p">&nbsp;</p>' if cell else ""

        tag = "p"
        style_name = paragraph.style.name if paragraph.style is not None else ""
        if not cell:
            if style_name.startswith("Heading 1"):
                tag = "h1"
            elif style_name.startswith("Heading 2"):
                tag = "h2"
            elif style_name.startswith("Heading 3"):
                tag = "h3"

        attrs: list[str] = []
        if cell:
            attrs.append('class="cell-p"')
        if align_style := self.paragraph_alignment_style(paragraph):
            attrs.append(f'style="{align_style}"')
        attr_text = " " + " ".join(attrs) if attrs else ""
        return f"<{tag}{attr_text}>{content}</{tag}>"

    def paragraph_alignment_style(self, paragraph: Paragraph) -> str:
        if paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER:
            return "text-align:center"
        if paragraph.alignment == WD_ALIGN_PARAGRAPH.RIGHT:
            return "text-align:right"
        if paragraph.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY:
            return "text-align:justify"
        if paragraph.alignment == WD_ALIGN_PARAGRAPH.LEFT:
            return "text-align:left"
        return ""

    def run_html(self, run, paragraph: Paragraph) -> str:
        parts: list[str] = []
        attrs = self.run_attrs(run)
        if run.text:
            text = html.escape(run.text).replace("\n", "<br>")
            parts.append(f"<span{attrs}>{text}</span>")

        for blip in run._element.xpath('.//*[local-name()="blip"]'):
            rid = blip.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
            )
            if not rid:
                continue
            parts.append(self.image_html(paragraph, rid))
        return "".join(parts)

    def run_attrs(self, run) -> str:
        classes = []
        styles = []
        if run.bold:
            classes.append("bold")
        if run.italic:
            classes.append("italic")
        if run.underline:
            classes.append("underline")
        if run.font.size is not None:
            styles.append(f"font-size:{run.font.size.pt:.1f}pt")
        if run.font.color is not None and run.font.color.rgb is not None:
            styles.append(f"color:#{run.font.color.rgb}")

        attrs = []
        if classes:
            attrs.append(f'class="{html.escape(" ".join(classes), quote=True)}"')
        if styles:
            attrs.append(f'style="{html.escape(";".join(styles), quote=True)}"')
        return " " + " ".join(attrs) if attrs else ""

    def image_html(self, paragraph: Paragraph, rid: str) -> str:
        image_part = paragraph.part.related_parts.get(rid)
        if image_part is None:
            return self.unsupported_image_html(rid, "missing")

        content_type = getattr(image_part, "content_type", "application/octet-stream")
        if content_type not in _BROWSER_PREVIEW_IMAGE_TYPES:
            return self.unsupported_image_html(rid, content_type)

        return (
            f'<img class="doc-image" src="{self.image_data_uri(rid, image_part)}" '
            'alt="Word embedded image">'
        )

    def unsupported_image_html(self, rid: str, content_type: str) -> str:
        if rid not in self.unsupported_image_rids:
            self.unsupported_image_rids.add(rid)
            self.counters["unsupported_image"] += 1
        label = html.escape(_image_type_label(content_type))
        return (
            '<span class="unsupported-image-note" role="note">'
            f"预览暂不支持 {label} 图片，已在 Word 报告中保留。"
            "</span>"
        )

    def image_data_uri(self, rid: str, image_part: Any) -> str:
        if rid in self.image_rid_to_data_uri:
            return self.image_rid_to_data_uri[rid]
        content_type = getattr(image_part, "content_type", "application/octet-stream")
        data = base64.b64encode(image_part.blob).decode("ascii")
        uri = f"data:{html.escape(content_type, quote=True)};base64,{data}"
        self.image_rid_to_data_uri[rid] = uri
        self.counters["image"] += 1
        return uri

    def table_html(self, table: Table) -> str:
        self.counters["table"] += 1
        rows = self.table_grid_rows(table)
        row_html = []
        for row_index, row in enumerate(rows):
            cell_html = []
            for item in row:
                if item["vmerge"] == "continue":
                    continue
                attrs = []
                if item["colspan"] > 1:
                    attrs.append(f'colspan="{item["colspan"]}"')
                if item["vmerge"] == "restart":
                    rowspan = self.compute_rowspan(rows, row_index, item["col"], item["colspan"])
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                if width_style := self.tc_col_width_style(item["tc"]):
                    attrs.append(f'style="{width_style}"')
                tag = "th" if row_index == 0 else "td"
                cell = _Cell(item["tc"], table)
                attr_text = " " + " ".join(attrs) if attrs else ""
                cell_html.append(f"<{tag}{attr_text}>{self.cell_html(cell)}</{tag}>")
            row_html.append("<tr>" + "".join(cell_html) + "</tr>")
        return '<div class="table-wrap"><table>' + "".join(row_html) + "</table></div>"

    def cell_html(self, cell: _Cell) -> str:
        parts = [
            part
            for block in self.iter_block_items(cell)
            if (part := self.paragraph_html(block, cell=True) if isinstance(block, Paragraph) else self.table_html(block))
        ]
        return "".join(parts) or "&nbsp;"

    def table_grid_rows(self, table: Table) -> list[list[dict[str, Any]]]:
        rows = []
        for tr in table._tbl.tr_lst:
            col = 0
            items = []
            for tc in tr.tc_lst:
                colspan = self.tc_grid_span(tc)
                items.append(
                    {
                        "tc": tc,
                        "col": col,
                        "colspan": colspan,
                        "vmerge": self.tc_vmerge_val(tc),
                    }
                )
                col += colspan
            rows.append(items)
        return rows

    def tc_grid_span(self, tc: CT_Tc) -> int:
        tc_pr = tc.tcPr
        if tc_pr is None or tc_pr.gridSpan is None:
            return 1
        return int(tc_pr.gridSpan.val)

    def tc_vmerge_val(self, tc: CT_Tc) -> str | None:
        tc_pr = tc.tcPr
        if tc_pr is None or tc_pr.vMerge is None:
            return None
        return "continue" if tc_pr.vMerge.val is None else str(tc_pr.vMerge.val)

    def compute_rowspan(
        self,
        rows: list[list[dict[str, Any]]],
        row_index: int,
        col: int,
        colspan: int,
    ) -> int:
        span = 1
        for next_index in range(row_index + 1, len(rows)):
            matches = [self.find_cell_at_col(rows[next_index], c) for c in range(col, col + colspan)]
            if any(match is None or match.get("vmerge") != "continue" for match in matches):
                break
            span += 1
        return span

    def find_cell_at_col(self, row: list[dict[str, Any]], col: int) -> dict[str, Any] | None:
        for item in row:
            start = item["col"]
            if start <= col < start + item["colspan"]:
                return item
        return None

    def tc_col_width_style(self, tc: CT_Tc) -> str:
        tc_pr = tc.tcPr
        if tc_pr is None or tc_pr.tcW is None or tc_pr.tcW.w is None:
            return ""
        try:
            width = int(tc_pr.tcW.w)
        except (TypeError, ValueError):
            return ""
        if width <= 0:
            return ""
        return f"width:{width / 20:.1f}pt"


def _image_type_label(content_type: str) -> str:
    if content_type == "missing":
        return "缺失"
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized in _IMAGE_TYPE_LABELS:
        return _IMAGE_TYPE_LABELS[normalized]
    if normalized.startswith("image/"):
        subtype = normalized.split("/", 1)[1]
        if subtype.startswith("x-"):
            subtype = subtype[2:]
        return subtype.upper() or "未知格式"
    return "未知格式"


def _preview_css() -> str:
    return """
    :root { color-scheme: light; --ink:#18212f; --muted:#667085; --line:#d8dee9; --paper:#ffffff; --bg:#f4f6f9; --accent:#275eb8; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif; line-height:1.62; }
    .shell { max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }
    .toolbar { position: sticky; top: 0; z-index: 2; margin: -32px -20px 24px; padding: 14px 20px; background: #f4f6f9; border-bottom: 1px solid var(--line); }
    .toolbar strong { display:block; font-size:15px; }
    .toolbar span { display:block; color:var(--muted); font-size:12px; margin-top:2px; }
    article { background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 42px 54px; box-shadow: 0 18px 50px rgba(31,41,55,.08); }
    h1 { font-size: 28px; line-height: 1.32; margin: 0 0 18px; text-align:center; }
    h2 { font-size: 21px; margin: 34px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); color:#163b73; }
    h3 { font-size: 17px; margin: 24px 0 8px; color:#1c4e8a; }
    p { margin: 8px 0; }
    .bold { font-weight: 700; } .italic { font-style: italic; } .underline { text-decoration: underline; }
    .table-wrap { width:100%; overflow:auto; margin: 14px 0 22px; border: 1px solid var(--line); border-radius: 8px; }
    table { border-collapse: collapse; width: 100%; min-width: 680px; table-layout: fixed; font-size: 13px; }
    th, td { border: 1px solid var(--line); padding: 7px 9px; vertical-align: middle; overflow-wrap: anywhere; }
    th { background:#edf3fb; font-weight:700; text-align:center; }
    td { background:#fff; }
    .cell-p { margin: 0; } .cell-p + .cell-p { margin-top: 4px; }
    .doc-image { display:block; max-width:100%; height:auto; margin: 12px auto; border: 1px solid var(--line); border-radius: 6px; }
    .unsupported-image-note { display:block; margin: 12px auto; padding: 10px 12px; border: 1px dashed var(--line); border-radius: 6px; background:#f8fafc; color:var(--muted); font-size:12px; text-align:center; }
    @media (max-width: 760px) { article { padding: 26px 18px; border-radius: 0; } .shell { padding-left:0; padding-right:0; } .toolbar { margin-left:0; margin-right:0; } }
""".strip()
