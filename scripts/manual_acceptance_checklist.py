"""Generate the printable 54-item manual acceptance checklist from the handbook.

The output is blank: results, evidence, reasons and every signature field stay
empty for the user/independent acceptor. Nothing here signs or closes anything.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import html
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = ROOT / "docs/development/手动验收手册.md"
ITEM_COUNT = 54
_ROW = re.compile(r"^\| (S\d{2}) \| (.+?) \| (.+?) \|\s*$")
CONCLUSIONS = ("软件操作检查", "正式软件冻结门", "历史业务 H", "机构生产 P")
RESULTS = ("通过", "失败", "未执行", "不适用")


def parse_items(text: str) -> list[dict]:
    section = text.split("## 3. 软件操作检查", 1)[-1].split("## 4.", 1)[0]
    items = [{"id": m[1], "action": m[2].strip(), "expected": m[3].strip()}
             for m in (_ROW.match(line) for line in section.splitlines()) if m]
    ids = [item["id"] for item in items]
    if len(items) != ITEM_COUNT or ids != [f"S{n:02d}" for n in range(1, ITEM_COUNT + 1)]:
        raise ValueError(f"handbook_must_list_S01_to_S54 (found {len(items)})")
    return items


def render_markdown(items, digest, generated):
    lines = [f"# MARVIS 手动验收清单（{ITEM_COUNT} 项）", "",
             f"来源：`docs/development/手动验收手册.md`（sha256 `{digest}`）｜生成：{generated}",
             "", "候选版本/源码摘要：________ 环境：________ 浏览器/分辨率：________ 验收人：________", "",
             "结果填 通过 / 失败 / 未执行 / 不适用；先写预期再操作，失败照实保留。", ""]
    for item in items:
        lines += [f"## {item['id']}", f"- 操作：{item['action']}", f"- 必须看到：{item['expected']}",
                  "- 结果：□通过 □失败 □未执行 □不适用", "- 证据位置：", "- 原因/备注：", "- 检查人签字/日期：", ""]
    lines += ["## 结论（全部留空，由验收人填写）", ""]
    lines += [f"- {name}：□通过 □失败 □未执行　签字：　　　日期：" for name in CONCLUSIONS]
    lines += ["", "验收人姓名：　　　　 实际责任：　　　　 日期：　　　　 证据索引：", ""]
    return "\n".join(lines)


def render_html(items, digest, generated):
    esc = html.escape
    boxes = "".join(f'<span class="box"></span>{r}　' for r in RESULTS)
    rows = "".join(
        f"<tr><td class=id>{esc(i['id'])}</td><td>{esc(i['action'])}</td><td>{esc(i['expected'])}</td>"
        f"<td class=res>{boxes}</td><td class=blank></td><td class=blank></td><td class=sign></td></tr>"
        for i in items)
    conclusions = "".join(f"<tr><td>{esc(c)}</td><td class=res>{boxes}</td><td class=sign></td><td class=sign></td></tr>"
                          for c in CONCLUSIONS)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>MARVIS 手动验收清单</title><style>
body{{font:12px/1.45 -apple-system,"PingFang SC",sans-serif;color:#111;background:#fff;margin:24px}}
h1{{font-size:18px;margin:0 0 4px}} .meta{{color:#444;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%;margin-bottom:16px}} th,td{{border:1px solid #888;padding:4px 6px;vertical-align:top}}
th{{background:#eee}} tr{{page-break-inside:avoid}} .id{{white-space:nowrap;font-weight:600}}
.res{{white-space:nowrap}} .blank{{width:12%}} .sign{{width:10%;height:28px}}
.box{{display:inline-block;width:10px;height:10px;border:1px solid #333;margin-right:2px;vertical-align:-1px}}
.fill{{display:inline-block;min-width:160px;border-bottom:1px solid #333}}
@page{{size:A4 landscape;margin:12mm}}
</style></head><body>
<h1>MARVIS 手动验收清单（{ITEM_COUNT} 项）</h1>
<div class=meta>来源 docs/development/手动验收手册.md · sha256 {digest} · 生成 {generated} · 本表空白即未验收</div>
<p>候选版本/源码摘要 <span class=fill></span> 环境 <span class=fill></span> 浏览器/分辨率 <span class=fill></span> 验收人 <span class=fill></span></p>
<table><thead><tr><th>编号</th><th>操作</th><th>必须看到的结果</th><th>结果</th><th>证据位置</th><th>原因</th><th>签字/日期</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2 style="font-size:14px">结论与签字（由验收人填写）</h2>
<table><thead><tr><th>层</th><th>结论</th><th>签字</th><th>日期</th></tr></thead><tbody>{conclusions}</tbody></table>
<p>验收人姓名 <span class=fill></span> 实际责任 <span class=fill></span> 日期 <span class=fill></span> 证据索引 <span class=fill></span></p>
</body></html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handbook", type=Path, default=HANDBOOK)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    raw = args.handbook.read_bytes()
    items = parse_items(raw.decode("utf-8"))
    digest, generated = hashlib.sha256(raw).hexdigest(), date.today().isoformat()
    out = args.out or ROOT / ".omc/acceptance" / generated
    out.mkdir(parents=True, exist_ok=True)
    (out / "manual-checklist.md").write_text(render_markdown(items, digest, generated), encoding="utf-8")
    (out / "manual-checklist.html").write_text(render_html(items, digest, generated), encoding="utf-8")
    print(f"{len(items)} items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
