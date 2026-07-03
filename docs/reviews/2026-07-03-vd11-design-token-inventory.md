# VD-11 设计 Token 收敛清单（第一轮：radius-pill）

- **日期**：2026-07-03
- **分支**：`fix/vd11-tokens`
- **范围声明**：本轮是**纯收敛重构**，不改任何视觉呈现。已存在的 `:root` token 层（`marvis/static/styles.css:1-189`，共 172 个变量 + `body[data-theme="dark"]` 覆盖块）在此之前就相当成熟；本轮只新增 1 个 token 并替换其对应的重复硬编码值，同时给后续"值调整"轮次留一份完整现状清单。

## 1. 本轮已收敛

| Token | 值 | 语义 | 替换处数 | 文件分布 |
|---|---|---|---|---|
| `--radius-pill` | `999px` | 胶囊形圆角（pill / badge / switch / scrollbar thumb 等） | 37 | `styles.css` 25、`v2-workbench.css` 10、`welcome.css` 2 |

新增位置：`marvis/static/styles.css:53`（`:root` 块内，紧跟 `--radius-control`）。不需要 dark 主题覆盖 —— 999px 是形状值不是颜色，和同层的 `--radius` / `--radius-control` 一样只在 `:root` 定义一次，两个主题共用。

选择 `999px` 单独收敛的原因：37 处出现的值和用途完全一致（都是 `border-radius: 999px;`，做圆角/胶囊形状），是全仓库唯一一个"重复值 + 零上下文分歧"的硬编码模式，风险最低、收益最直接。

## 2. 已盘点但本轮未动（留给后续拍板轮）

以下类别在 3 个 CSS 文件（`styles.css` 7478 行 / `v2-workbench.css` 2352 行 / `welcome.css` 436 行）中确认有重复，但因为语义上下文分歧较大（同一数值在不同组件里代表不同设计意图），或属于"数值调整"性质，按用户红线本轮不擅自合并，仅记录供拍板：

### 2.1 圆角（radius）—— 非 999px 的其他重复值

| 值 | 出现次数（3 文件合计） | 典型用途 |
|---|---|---|
| `8px` | 4（`v2-workbench.css`） | 卡片/面板圆角 |
| `6px` | 5（`styles.css` 3 + `v2-workbench.css` 2） | 小控件圆角 |
| `5px` | 4（`styles.css`） | 徽标/小块圆角 |
| `4px` | 5（`styles.css` 4 + `welcome.css` 1） | 细节圆角 |
| `1px` | 4（`styles.css`） | 极细圆角（近似直角修饰） |
| `3px` | 1（`styles.css`） | — |
| `12px` | 1（`welcome.css`） | — |
| `50%` | 多处（未精确计数，语义是"圆形"非"圆角"，与上面数值型 token 不同类） | 头像/圆点 |

未收敛原因：这些值分散在卡片、控件、徽标等不同组件语境里，且已有 `--radius`（16px）/`--radius-sm/md/lg`（均为 16px，彼此已经重复但值相同——这是既存状态，非本轮引入）/`--radius-control`（10px）三档体系。要不要让 8px/6px/5px/4px 归入新档位或复用现有档位，属于"取值调整"范畴，需要出对比稿给用户拍板。

### 2.2 颜色（hex）

`v2-workbench.css` 与 `welcome.css` 两个文件**零硬编码 hex**——已经完全走 `var()`。`styles.css` 在 `:root`/`dark` 块之外仍有 **18 个不同的 hex 值、共 20 处出现**，但全部是"只出现一次"的组件专属强调色（例如模型/效率 composer chip 悬停色 `#8b5cf6`/`#7c3aed`/`#eab308`/`#b45309` 及其 dark 变体 `#a78bfa`/`#c4b5fd`/`#facc15`/`#fde047`；run-mode 卡片 tone 色 `#b87912`/`#fff7e8`/`#54a867`/`#edf8f0` 及 dark 变体 `#322513`/`#162d1d`）。因为没有重复出现（每个值只用一次），不构成"legacy 重复硬编码"，本轮判定为不在收敛范围内，原样保留。

### 2.3 间距（spacing：gap / padding / margin）

**目前整个 token 层没有任何 `--space-*` 变量**——间距是全仓库唯一完全未 token 化的类别，重复量也最大：

`gap` 字面值重复次数最高的几档（3 文件合计）：

| 值 | 次数 |
|---|---|
| `8px` | 58 |
| `10px` | 38 |
| `6px` | 38 |
| `12px` | 33 |
| `7px` | 14 |
| `4px` | 13 |
| `5px` | 7 |
| `3px` | 7 |
| `2px` | 6 |
| `14px` | 6 |

`padding`（单值/多值 shorthand）重复次数最高的几档：`6px 12px`×10、`10px 12px`×9、`10px`×8、`9px 12px`×7、`7px 8px`×7、`7px 9px`×6、`12px`×6、`11px 13px`×6，另有 20+ 种更低频组合。

未收敛原因：这是最大的收敛机会，但也是风险最高的一类——间距数值分布很密（几乎每 1-2px 一档），且当前 `gap`/`padding` 混用大量非规整值（7px、9px、11px、13px 这类"非 4/8 倍数"数值），暗示这批数值本身可能就是需要"调整"而非"原样收敛"的候选（例如统一到 4/8 基准网格）。这不符合本轮"值不变、只做 token 化"的红线执行条件——若原样收敛出 10+ 个 `--space-N` token，会锁死一批可能本就不够规整的历史数值，增加后续调整的心智负担。建议单独出一轮"spacing scale 提案"，附带视觉对比稿，由用户决定基准网格（4px/8px）后再做 token 化 + 收敛。

### 2.4 阴影（box-shadow）

`:root` 层已有 10 个具名阴影 token（`--shadow`、`--shadow-lift`、`--shadow-floating`、`--settings-menu-shadow`、`--progress-rail-shadow`、`--button-solid-shadow`/`-hover`、`--button-secondary-shadow`/`-hover`、`--agent-send-stop-shadow`、`--agent-user-message-shadow`、`--toast-shadow`），且都有 dark 主题覆盖。文件体内散落的多行 `box-shadow` 字面值（如 `0 12px 28px rgba(0,0,0,0.08)`、`0 18px 48px rgba(0,0,0,0.2)` 等）经排查基本是**单次出现**，与已有具名阴影不完全相同（层数/扩散/透明度均有细微差异），因此判定为组件专属阴影而非遗留重复，本轮不动。`box-shadow: none;` 出现 21 次，属于状态重置而非"值"，不构成 token 候选。

### 2.5 字号（font-size）—— 高重复但判定为"类型系统"而非本轮范围

| 值 | 次数（3 文件合计） |
|---|---|
| `12px` | 121 |
| `13px` | 40 |
| `14px` | 30 |
| `11px` | 29 |
| `12.5px` | 9 |
| `15px` | 8 |
| `18px` | 5 |
| `10px` | 3 |
| `22px` / `19px` | 各 2 |
| `13.5px` / `11.5px` | 各 2 |
| `32px` / `28px` / `25px` / `17px` / `16px` | 各 1 |

未收敛原因：字号是任务描述中明确点名"取值调整需先出对比稿"的三类之一（radius / spacing / type scale）。虽然重复量很大（12px 出现 121 次），但字号属于 type scale 范畴，红线要求先由用户拍板取值再落地，本轮不擅自建 `--font-size-*` token 层。

### 2.6 字重（font-weight）—— 参考数据，非本轮范围

`600`×54、`700`×47、`650`×14、`800`×6、`500`×5、`750`×2、`400`×1。同属 type scale，处理原则同 2.5。

## 3. 建议的下一步

1. **Spacing scale 提案**（2.3）：先决定 4px 还是 8px 基准网格，出一版对比稿（含当前非规整值 7px/9px/11px/13px 归并到哪个最近档位的建议），拍板后再做 token 化 + 收敛，预计能消掉 300+ 处重复。
2. **Type scale 提案**（2.5 + 2.6）：字号/字重同理，先出对比稿。
3. **Radius 第二轮**（2.1）：8px/6px/5px/4px 四档是否归并、归并到几档，出对比稿。
4. 颜色（2.2）与阴影（2.4）当前判定为组件专属值，无重复，暂不建议强行 token 化（除非未来复用增多）。

## 4. 等价性验证方法与结果

**方法**：写一次性 Python 脚本（未入库，路径见下），对替换前（`git show HEAD:<file>`）与替换后（工作区文件）做如下比对：
1. 从替换后 `styles.css` 的 `:root` 块解析出 `--radius-pill` 的字面值。
2. 把替换后文件中所有 `var(--radius-pill)` 逐字符串替换回该字面值（`999px`），并去掉新增的 token 定义那一行。
3. 对比"还原后的替换后文件"与"替换前文件"是否逐字节相同（`==` 比较，非近似 diff）。

对 `styles.css`、`v2-workbench.css`、`welcome.css` 三个文件分别执行，三者均 **PASS（byte-identical）**——证明本次改动是纯字面量替换，没有引入任何选择器、属性或其他数值的变化。

脚本执行输出：
```
--radius-pill resolves to: '999px'
[styles.css] resolved-after == before: OK (byte-identical)
[v2-workbench.css] resolved-after == before: OK (byte-identical)
[welcome.css] resolved-after == before: OK (byte-identical)

PASS: all files are byte-identical after resolving --radius-pill back to 999px.
```

辅助交叉检查（人工抽样）：`git diff --stat` 显示 `styles.css` 51 insertions / 37 deletions（25 处替换的 25 组 -/+ 配对 + 1 行纯新增 token 定义 = 26 insertions + 25 deletions = 51/25，核对一致）；`v2-workbench.css` 20 insertions / 10 deletions（10 组 -/+ = 20/10，一致）；`welcome.css` 4 insertions / 2 deletions（2 组 -/+ = 4/2，一致）。逐条 `git diff` 抽查确认每一处 hunk 只涉及 `border-radius` 一行的字面量替换，无相邻行变化。

## 5. 遗留不敢动清单

- 内联 SVG（`welcome.css` 中 `--welcome-icon-mask-image` 的 data URI，`marvis/static/index.html` 等处的 logo 相关 SVG）——用户红线，未触碰、未审查。
- JS 内颜色字符串——未扫描修改，任务要求"纯重复常量且有把握"才动，本轮聚焦 CSS 未涉及 JS。
- 2.1-2.6 列出的所有类别——按红线，取值调整需先出对比稿，本轮保持原样。
