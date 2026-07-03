# JOIN 阶段详细 Spec（数据拼接）

> V2 完善总计划的第 1 份阶段 spec（见 [v2-completion-plan.md](v2-completion-plan.md)）。
> JOIN 是最先建、且含**最高风险**的阶段（join 安全不变量：样本表锚定 + 左连接 + 键字典 + 强制确认）。作为可组合模块被 FEATURE / MODELING 复用。
> 核心原则:**样本是锚,只贴列、不改行(样本永远 1:1)。**

---

## 0. 地面真值（已核对代码）
- `data_ops` 工具:`infer_schema(dataset_id)→columns/has_target/target_col`、`align_columns(anchor_id,feature_ids)→alignments`、`propose_join(anchor_id,feature_ids)→join_plan_id/joins/status`(只读,写 join_plan)、`execute_join(join_plan_id)→result_dataset_id/anchor_rows/joined_rows/fan_out/warnings`(写 dataset)、`dedup_rows(dataset_id,keys,strategy)→removed_rows`、`clean_format(dataset_id,ops)`。
- `JoinEngine`(`marvis/data/join_engine.py`):
  - `propose_join_plan` 每张特征表产 `diagnostics`:`match_rate`、`fan_out_detected`、`shrink_detected`(< `SHRINK_WARN_THRESHOLD`)、`new_columns_null_rate`;每个 `key_pair` 带 `match_rate` + `fingerprint`(raw vs md5)。
  - `confirm_join_spec` → `spec.confirmed=True` + 审计事件 `join.confirmed`。
  - `execute_join_plan` **引擎层硬阻断**:任一 join 未 `confirmed` 即抛 `JoinNotConfirmedError`(L205)。强制确认是引擎不变量,不是 UI 礼貌。

---

## 1. 跳过判据
- 输入 **≤1 张表** → JOIN 跳过。右栏显示"已跳过:单一样本,无需拼接"。
- 但 **目标列与"这张是样本"仍在入口确认一次**(见 C1 单表退化),目标列向 FEATURE/MODELING 复用。

## 2. 步骤序列与门
```
infer_schema(每张)  →  C1 文件角色分配(确认)  →  propose_join(只读,备诊断)
   →  C2 锚定+诊断门(二次强制确认 + 调整)  →  execute_join  →  [dedup 兜底]  →  阶段完成摘要 → 交 FEATURE
```
- PlanStep 标注:`propose_join` = `decision_point=True`;C2 对应步骤 `needs_confirmation=True`(引擎停在 AWAITING_CONFIRM)。
- C1 是阶段入口的结构化交互(角色/目标),C2 是执行前的强制门。**两次确认**,因为这步最重要。

---

## 3. C1 — 文件角色分配（对话流富表 / 弹窗）
对每张输入文件先跑 `infer_schema`,然后在对话流内联一张富表(手动=控件;agent=LLM 提议+你确认):

| 文件 | 行数 | 列数 | 含目标列? | 候选目标列 | **角色(选择框)** |
|---|---|---|---|---|---|
| A.csv | 209,245 | 4,865 | 是 | long_y | ●样本主表 ○特征表 ○忽略 |
| B.csv | 530,110 | 88 | 否 | — | ○样本主表 ●特征表 ○忽略 |
| C.csv | … | … | 否 | — | ○样本主表 ●特征表 ○忽略 |

- **锚自动提议**:含目标列者 = 提议样本主表;其余 = 特征表。歧义(多/无目标)→ 留给你点选。
- **目标列显式确认**(对锚表):"样本主表 = A,目标列 = `long_y`,对吗?"——最一开始必须确认一次。
- 点"确认" → C1 完成。
- **单表退化**:只有一张表时 C1 = "确认这张是样本 + 目标列",随后 JOIN 跳过。

## 4. propose_join（C1 与 C2 之间,只读）—— 含动态键识别
- 输入 C1 定的 `anchor_id` + `feature_ids` → 产 `join_plan`(draft)。
- **键识别(不强求三要素)**:
  - 先**语义识别**各表的身份要素列(手机号 / 身份证号 / 姓名)与日期列——列名各表不同,靠语义 + 键字典映射到样本列。
  - **要素数因表而异**:可能 二要素+日期、一要素+日期……都有;领域保证 **样本要素 ⊇ 特征要素**,故特征的要素总能映射回样本。
  - **匹配键 = 去重键 = 该特征表「可用身份要素(映射到样本)+ 日期」**,默认取**全部可用要素**(最唯一 → 最不膨胀)。
  - 目标(你的判据):**都 join 上(match 高)且不膨胀(fan-out 无)**。
    - 全要素键 `match_rate` 过低(某要素脏/缺)→ C2 提示"可减一个要素换更高 match",但**减后必须重检不膨胀**;
    - 键太弱致 `fan_out_detected` → C2 提示"加要素 / 特征侧去重"。
    - 取舍由你/LLM 在 C2 拍,系统**只提议+亮 match/fanout 两个指标**,不静默改键。
  - 每张算 `match_rate / fan_out_detected / shrink_detected / new_columns_null_rate / fingerprint(raw vs md5)`。

## 5. C2 — 锚定 + 诊断门（二次强制确认,执行前最后一关）
对话流内联,复述意图 + 一并摊诊断:

> 锚 = `A`(目标 `long_y`)、特征表左连接、**行群以样本为准(1:1)**。

| 特征表 | 识别要素 | 匹配/去重键(可改) | 命中率 | 行膨胀 | 新列空值率 | 指纹 raw=md5? | 状态 |
|---|---|---|---|---|---|---|---|
| B | 身份证+姓名+日期(二要素+时间) | id_no+name+stat_date | 96.2% | 否 | 3.8% | ✓一致 | 待确认 |
| C | 身份证+日期(一要素+时间) | idcard+dt | 71.4% | **是→特征侧去重** | 28.6% | ✗(raw≠md5,键格式不一致) | 待确认 |
| D | 手机+身份证+姓名+日期(三要素+时间) | mobile+id_no+name+dt | 99.1% | 否 | 0.9% | ✓一致 | 待确认 |

- **调整项(走 Q5 重跑本步)**:改匹配/去重键、去掉某张特征表、设特征去重策略、对指纹不一致的键 `clean_format` 标准化后重 `propose_join`。
- **告警**:`shrink_detected`(命中过低)/`fan_out_detected`(行膨胀)/指纹 raw≠md5(键类型或格式不一致,如 "0123" vs 123)。
- **二次确认** → 每张 `confirm_join_spec`(引擎强制,未确认即 `JoinNotConfirmedError`)→ 进 execute。

## 6. 特征表去重（两级,样本永远 1:1）
键 = 匹配键(识别出的身份要素 + 日期,要素数因表而异;与 §4 同一把键)。
- **一级·安全去重(自动)**:按键分组,**整行完全相同**(键 + 所有特征值都等)的重复 → 直接删,无损(同人同天特征本应一致)。
- **二级·冲突检测(告警,不自动删)**:一级后若同键仍多行(**同人同天、特征值不一致**)→ 数据质量红旗 → 告警:列出冲突键 + 不一致的特征,建议排查/排除该特征。
- **裁决**:排除问题特征,或接受 → 选策略(随机一行 / 保留首条;有更细业务列可按其取)。
- 结果:特征表每键至多 1 行 → 左连接后样本 1:1,只贴列。

## 7. execute_join + 兜底 dedup
- `execute_join`:左连接,样本行数不变,只新增特征列;返回 `anchor_rows/joined_rows/fan_out/warnings`。**断言 `joined_rows == anchor_rows`**(否则说明仍有膨胀,回 C2)。
- `dedup_rows`:仅当**样本主表自身**主键重复才触发(罕见),会删行时给小确认。

## 8. 阶段完成 + 交接
- 内联摘要:最终 行数(=样本行数)/ 列数 / 新增特征列数 / 整体空值率 / 各特征表贡献列数与命中率。
- 产出 `result_dataset_id`(锚 + 已拼特征)→ 作为 FEATURE 阶段输入。

---

## 9. 手动 vs agent 模式（同一底座,只换门的脸）
| 门 | 手动模式 | agent 模式 |
|---|---|---|
| C1 角色分配 | 富表 + 角色选择框/目标列下拉 + 确认按钮 | LLM 读 schema 提议角色/目标,你点或一句话改 |
| C2 诊断确认 | 诊断富表 + 改键/去表控件 + 去重策略下拉 + 二次确认按钮 | LLM 摘要风险("C 命中仅71%且键格式不一致,建议先标准化")+ 你确认/自由文本调整 |
| 冲突告警 | 冲突列表 + 排除勾选 / 策略下拉 | LLM 点出冲突特征 + 建议,你拍板 |
- 两模式共用 `propose_join → confirm_join_spec → execute_join` 与"重跑本步换参数"原语;只差谁提议、怎么喂参数。

## 10. 映射到 V2 原语
- PlanSteps(phase=`数据拼接`):`infer_schema`(每张,连续不停)→ `propose_join`(decision_point)→ `execute_join`(其前置确认门 needs_confirmation)→ `dedup_rows`(条件)。
- 确认/调整:`confirm_step`/新"调整指令"端点;改键=重跑 propose_join;去表/换锚=`replan` 带约束。
- skip:单表时该 phase 步骤置 `SKIPPED`,显式显示。

## 11. 新建 vs 复用
- **复用**:全部 `data_ops` 工具 + `JoinEngine`(锚定/指纹/强制确认/match_rate 全有)+ 既有 join_review 前端可借鉴。
- **新建**:**身份要素语义识别**(per 表识别 手机/身份证/姓名/日期,映射到样本列,要素数可变、择键以 match 高+不膨胀为准——`align_columns`/键字典可借力但"动态择键"逻辑是新的);C1 文件角色分配 UI;C2 诊断内联富表 + 调整控件;两级特征去重(一级 drop_duplicates 全列;二级同键冲突检测——`dedup_rows` 需扩出"冲突报告"而非静默删);execute 后 `joined_rows==anchor_rows` 断言;phase skip 显示。

---

## 12. 待你确认的小项（我已给默认,审时可改）
1. **冲突默认策略**:二级冲突若你选"接受",默认 = 有业务时间列按最新、否则保留首条。**随机一行**仅在你明确选时用(随机不可复现,与确定性不变量有张力——建议默认不用随机)。
2. **指纹 raw≠md5**:默认在 C2 提示并提供 `clean_format` 标准化选项(去空格/去前导零/统一大小写),不自动改键。
3. **shrink 阈值**:命中率低于 `SHRINK_WARN_THRESHOLD`(沿用引擎现值)只告警不阻断,你可在 C2 决定是否仍拼。
4. **clean_format 时机**:只在 C2 检出指纹不一致时按需触发,不进默认步骤序列。
