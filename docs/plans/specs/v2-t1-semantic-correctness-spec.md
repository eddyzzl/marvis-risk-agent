# T1 Spec：语义正确性修复包

> **计划归属**：[v2-trust-first-plan.md](../v2-trust-first-plan.md) Phase T 第一份出审 spec。**依据**：[2026-07-04 全量读码报告](../../reviews/2026-07-04-full-read-and-owner-qa.md) A2/A4/A5。
> **颗粒度**：函数级 + 内部伪代码（按用户偏好）。每项含现状机理、关键代码位置、补丁伪代码、失败形状测试、风险/口径变化、待确认设计点。
> **状态**：待审。审定后按项实施，每项一个 commit，每项补丁必须先落"失败形状测试"（喂脏形状断言正确行为）再改实现（TDD）。
> **上下文来源**：14 项均由 opus 逐项读码产出（workflows wj6vnlp63 + w33032n7d），非文档推断。

---

## 0. 这份 spec 修什么

报告 A2/A4/A5 的 8 条确认 bug + join 二审 2 条 high + 方法学残留 4 条，共 14 项，全部是同一物种：**代码按设计运行，但设计对真实数据的语义假设错了，或呈现给人的证据与计算事实不一致**。分四组：

| 组 | 项 | 一句话 |
|---|---|---|
| A 数据语义 | A1–A6 | vintage 快照重复计数、EL 跨月求和、sentinel 不进重放、slice NULL 当好、join 空白键错配、float64 证件号静默不匹配 |
| B join 一致性 | B7–B8 | 诊断/执行读取器分裂、键跨文件 dtype 不一致 |
| C 证据/交互 | C9–C10 | 冠军证据错指标、"先别开始验证"→开跑 |
| D 方法学残留 | D11–D14 | 精选特征 train+test 泄漏、time_col 不触发 OOT、screen 无 NaN 门、refit test_ks 进 headline |

**贯穿原则**：所有语义修复遵循 NaN 标签门先例（typed error + 显式 flag + 强制确认），不做静默默认；所有指标口径变化直接修正、不做旧值兼容、在第 3 节统一登记进发布说明。

---

## 1. 待你拍板的设计决策点（审 spec 时一次性定夺）

补丁里每个设计点都给了推荐默认+理由，可直接采纳。以下 5 个影响产品语义或涉及"诚实声明"，单独提请确认：

**D1. vintage `label_semantics` 门的严格度（A1）** — 推荐：内核默认 `incremental`（保证 validation/modeling 现状全绿），但**策略工具在未声明 label_semantics 时硬报错**（typed error → gate 问用户"你的 bad 列是快照还是增量"，附两种语义示例），与 NaN 标签门完全对称。备选：给策略工具一个静默默认（不推荐——正是这个静默造成了 12 倍虚高）。
→ 需你确认：**采纳"策略工具强制声明"**？

**D2. `approval_rate` 未识别决策 token 的处理（A4）** — 推荐：未识别 token（如自由文本 "pending"）算 **unlabeled 排除**（而非当 non-approve），并在输出暴露 `unlabeled_count` 让用户看到覆盖度。这会改变自由文本决策列的 approval_rate。备选：保持"未识别=非批准"。
→ 需你确认：这取决于你的决策列**实际怎么填**——是规整的 approve/reject 编码，还是可能有自由文本？

**D3. refit 指标的诚实声明（D14）** — 推荐 option A：refit 用 train+test 全量重训只为**换部署 artifact**，报告的 test_ks **保留 pre-refit 模型在真实 held-out test 上的值**（因为 refit 后原 test 已进训练集，没有真正的 held-out 可评）。代价：模型卡必须写明"报告的 KS 是 pre-refit 模型的留出测试值；部署的 refit artifact 为最大化数据用 train+test 训练、未独立再评"。
→ 需你确认：**接受这个"部署模型≠报告指标模型"的诚实声明**，还是希望 refit 也在某个口径上重评？

**D4. 空白键语义（A5）** — 推荐：空白/纯空格键 = **missing（nullif 到处，join 不到任何行）**，与 pandas fallback 和诊断口径一致。这会让"以前空白键错配"的行变成正确的未匹配（特征列 NULL，行数不变）。
→ 需你确认：**接受空白键=缺失**？（除非你有真实数据把空字符串当合法类别 join。）

**D5. 口径变化整体接受（A1/A2/A4/A5/D11/D14）** — 你在计划拍板 #5 已同意"口径变化不做兼容、发布说明列明"。第 3 节是完整清单。
→ 需你确认：**清单无遗漏、可照此发布**？

其余技术性设计点（B7 是否删 `_duckdb_text_rel`、B8 红旗只对 text↔float 还是也含 int↔float、C10 英文否定是否覆盖、D12 大小写敏感性等）已在各项补丁按"最安全默认"给出，除非你有异议即照此实施。

---

## 2. 实施顺序与依赖

- **A5 → A6 → B7 → B8** 有共享代码面（`marvis/data/backend.py` 的键规范化），建议连续实施、一次性回归，避免互相踩。**注意**：这正是 T0 期间某只读 agent 越界改到的 `_duckdb_rel` 区域——实施 B7 时以本 spec 为准，不复用那次被还原的改动。
- **A4** 与 **T2-1**（标签 0/1 强转收敛到 labels.py）同源，A4 先按本 spec 局部修，T2 再统一收编。
- **D11/D13** 与 select/screen 的 holdout 语义相关，建议同批；**D14** 与监控的 dev 基线读取相关（monitor 读 train_ks/test_ks），修 D14 时顺带确认监控读的是原始 test。
- **A1/A2/A3/C9/C10/D12** 相互独立，可并行。

每项 TDD：先加失败形状测试（该测试在旧代码上必须失败），再改实现至通过，再跑该模块回归。

---

## 3. 口径变化清单（发布说明附录）

以下数字在修复后会与历史报告不一致——这是**修正**不是回归，接入真实数据前必须让使用者知晓：

| 项 | 什么数字变了 | 变化方向 |
|---|---|---|
| A1 | 快照型 bad 列的 vintage 累计率 | 从虚高（可达 N 倍）降到真实值；策略工具现在会先要求声明语义 |
| A2 | 组合报告 headline `total_el` | 从跨月求和（~N 倍）降到参考快照口径 |
| A4 | slice `bad_rate` / `approval_rate`（数据含 NULL/非法标签时） | bad_rate 升高（分母剔除未标注行）；approval_rate 语义随 D2 定 |
| A5 | 含空白键的 join，空白键行的特征值 | 从错误挂载变为正确的 NULL（行数不变） |
| D11 | 默认建模（有 OOT 时）的精选特征集与其报告的 test 指标 | 精选不再看 test 标签，选出的特征集与 test KS 会变（更诚实） |
| D14 | refit 后冠军的 headline test_ks | 从随机 5% carve 的 KS 回到原始 test 留出 KS |

---

## 4. 逐项修复规格

（每项：现状机理 → 关键代码 → 调用面 → 现有测试与缺口 → 补丁伪代码 → 失败形状测试 → 风险/口径 → 设计点）


### A1 · vintage 快照标签重复计数 → label_semantics 参数 + 强制确认门

**现状（bug 机理）**：The strategy-facing vintage curve kernel `compute_vintage_curve` (marvis/validation/vintage.py:35-138) ALWAYS treats the target column as an INCREMENTAL per-MOB event: it computes per-(cohort,mob) marginal bad_count/bad_rate (first pass, L69-99), fixes a cohort-level denominator = max observed sample_count (or balance) across MOBs (L101-108), then ACCUMULATES the bad numerator across MOBs in ascending order to produce `cum_bad_rate` (second pass, L110-137), clipping to 1.0 and emitting a `data_quality_warnings` string when the raw ratio exceeds 1.0 (L118-124). There is no `label_semantics` parameter and no way to declare the alternative SNAPSHOT semantics where each MOB column already means "bad as of this MOB" (monotonic per loan) — in that case accumulation double-counts. The modeling report caller `compute_vintage_report` (marvis/packs/modeling/report_compute.py:113-162) self-protects against exactly this: its `mob_observe_cols` are cumulative-by-construction snapshot flags, so it deliberately calls `vintage_curve_wide(points, metric="bad_rate")` (L160) instead of `cum_bad_rate`, with an explicit comment (L155-159) documenting that using cum_bad_rate would "re-accumulate and double-count the already-cumulative bads". The strategy wrapper `vintage_curve` (marvis/packs/strategy/vintage.py:9-39) hardcodes `metric="cum_bad_rate"` (L26) — so any strategy user whose bad_col is a snapshot flag silently gets a double-counted, wrong curve with NO way to declare snapshot semantics and NO forced confirmation. `VintageCurve` (marvis/packs/strategy/contracts.py:7-14) has NO `warnings` field even though the kernel produces `data_quality_warnings` per point and `vintage_summary_payload` (vintage.py:202-213) aggregates them — the sibling `RollRateMatrix` (contracts.py:17-25) DOES carry `data_quality_warnings` and `tool_roll_rate` surfaces them (tools.py:97), so vintage is asymmetric: its warnings are computed but dropped on the floor by `tool_vintage_curve` (tools.py:51-77, which never reads point warnings). The tool `tool_vintage_curve` accepts `drop_nan_labels` via manifest and calls `resolve_labeled_frame` (tools.py:58-62) but the VINTAGE_ANALYSIS template (orchestrator/templates/strategy.py:86-100) never threads `drop_nan_labels` and never passes a `label_semantics` slot.

**关键代码位置**：
```
### marvis/validation/vintage.py:35-138 (compute_vintage_curve — the kernel to change)
```
35 def compute_vintage_curve(
36     dataframe: pd.DataFrame,
37     *,
38     cohort_col: str,
39     mob_col: str,
40     target_col: str,
41     balance_col: str | None = None,
42     denominator: str = "count",
43 ) -> list[VintagePoint]:
...   # first pass builds per-(cohort,mob) rows with bad_numerator (L71-99)
101         # Fixed cohort-level denominator: max observed base across the cohort's MOBs.
102         if denominator == "balance":
103             cohort_denominator = max((row["balance_sum"] or 0.0 for row in rows), default=0.0)
104         else:
108             cohort_denominator = float(max((row["sample_count"] for row in rows), default=0))
110         # Second pass: accumulate the bad numerator across MOBs in ascending order.
111         cumulative_bad = 0.0
112         for row in rows:
113             cumulative_bad += row["bad_numerator"]
114             if cohort_denominator == 0:
115                 raw_cum_ratio = 0.0
116             else:
117                 raw_cum_ratio = cumulative_bad / cohort_denominator
118             cum_bad_rate = min(raw_cum_ratio, 1.0)
119             warnings: tuple[str, ...] = ()
120             if raw_cum_ratio > 1.0 + 1e-9:
121                 warnings = (f"cum_bad_rate clipped for cohort {cohort} at mob {row['mob']}: ...",)
125             points.append(VintagePoint(... cum_bad_rate=cum_bad_rate, data_quality_warnings=warnings))
```

### marvis/packs/strategy/vintage.py:9-39 (strategy wrapper — hardcodes cum_bad_rate, no semantics)
```
9  def vintage_curve(df, *, cohort_col, mob_col, bad_col, mob_max=12) -> VintageCurve:
20     points = compute_vintage_curve(df, cohort_col=cohort_col, mob_col=mob_col, target_col=bad_col)
26     wide = vintage_curve_wide(points, metric="cum_bad_rate")
32     return VintageCurve(cohort_col=..., mob_max=..., cohorts=..., curves=curves, counts=..., mob_axis=mob_axis)
```

### marvis/packs/modeling/report_compute.py:154-160 (the self-protecting snapshot caller — the semantic precedent)
```
155         # mob_observe_cols are cumulative-by-construction snapshot flags: each
156         # column already means "bad as of this MOB" and is monotonic per loan, so
157         # the kernel's per-MOB bad_rate IS the true cumulative rate here. Using
158         # cum_bad_rate would re-accumulate and double-count the already-cumulative bads
160         "curves": vintage_curve_wide(points, metric="bad_rate"),
```

### marvis/packs/strategy/contracts.py:7-25 (VintageCurve lacks warnings; RollRateMatrix has it — the asymmetry)
```
7  @dataclass(frozen=True)
8  class VintageCurve:
9      cohort_col: str
10     mob_max: int
11     cohorts: tuple[str, ...]
12     curves: dict[str, list[float | None]]
13     counts: dict[str, int]
14     mob_axis: tuple[int, ...] = ()
17 @dataclass(frozen=True)
18 class RollRateMatrix:
25     data_quality_warnings: tuple[dict, ...] = ()
```

### marvis/data/labels.py:55-78 (resolve_labeled_frame — typed-error + explicit-flag precedent to mirror)
```
55 def resolve_labeled_frame(frame, target_col, *, drop_nan_labels, scope="dataset"):
71     if not drop_nan_labels:
72         raise NanLabelNotConfirmedError(target_col=..., n_total=..., n_nan=n_nan, scope=scope)
78     return frame.loc[~mask], n_nan
```

### marvis/data/errors.py:28-66 (NanLabelNotConfirmedError — the typed-error shape to copy for a new LabelSemanticsNotDeclaredError)
```
28 class NanLabelNotConfirmedError(DataLayerError):
37     def __init__(self, *, target_col, n_total, n_nan, scope="dataset", by_split=None): ...
57     def to_detail(self) -> dict:
59         return {"kind": "nan_label_not_confirmed", "target_col": ..., "n_total": ..., "n_nan": ..., "scope": ..., "by_split": ...}
```
```

**受影响调用面**：
- marvis/validation/vintage.py:35 — compute_vintage_curve kernel signature (add label_semantics param + snapshot branch)
- marvis/validation/vintage.py:141 — vintage_curve_wide (metric selection; snapshot path must pick bad_rate, incremental picks cum_bad_rate)
- marvis/validation/vintage.py:202 — vintage_summary_payload (already aggregates point.data_quality_warnings into payload['warnings'])
- marvis/packs/strategy/vintage.py:9 — vintage_curve wrapper: add label_semantics param, thread to kernel + pick metric, populate VintageCurve.warnings
- marvis/packs/strategy/vintage.py:26 — hardcoded metric='cum_bad_rate' must become semantics-driven
- marvis/packs/strategy/vintage.py:42 — vintage_summary (unchanged, reads curve.curves)
- marvis/packs/strategy/tools.py:51 — tool_vintage_curve: read inputs['label_semantics'], pass to vintage_curve, add 'warnings' to output dict (mirror tool_roll_rate:97), raise/propagate the typed error
- marvis/packs/strategy/tools.py:63 — vintage_curve() call site inside the tool
- marvis/packs/strategy/contracts.py:7 — VintageCurve dataclass: add warnings field
- marvis/packs/strategy/manifest.json:12-89 — vintage_curve tool: add label_semantics to input_schema properties (additionalProperties:false REQUIRES this), add 'warnings' to output_schema properties (additionalProperties:false REQUIRES this)
- marvis/orchestrator/templates/strategy.py:73-104 — VINTAGE_ANALYSIS template: add label_semantics SlotSpec + thread {slot:label_semantics} and drop_nan_labels into the vintage_curve step inputs_template
- marvis/agent/vintage_setup.py:37-56 — VintageProposal + template_slots(): add label_semantics field so setup can propose/heuristically default it
- marvis/agent/renderers.py:1146 — _render_vintage_curve: surface o['warnings'] (currently ignored)
- marvis/agent/gates/contracts.py:299,327 — vintage_curve already mapped to strategy_direction_approval risk flag; the new label-semantics gate must be routed here too
- marvis/plugins/subprocess_worker.py:65-68,639-658 — _structured_error_detail already converts any to_detail() typed error across the subprocess boundary (new LabelSemanticsNotDeclaredError just needs to_detail())
- marvis/errors.py:45-49 — ErrorKind: add LABEL_SEMANTICS_NOT_DECLARED constant next to the data-layer kinds
- marvis/packs/modeling/report_compute.py:142-160 — compute_vintage_report: pass label_semantics='snapshot' explicitly (its self-protection becomes the canonical snapshot path), can then drop its metric='bad_rate' workaround comment

**现有测试与缺口**：
- tests/validation/test_vintage.py — full kernel coverage: test_vintage_curve_uses_snapshot_cohort_denominator_without_pooling (L74) and test_cum_bad_rate_is_genuinely_cumulative_across_two_cohorts_and_three_mobs (L107) BOTH assert the current INCREMENTAL accumulation semantics as correct; test_cum_bad_rate_clips_to_one_and_records_data_quality_warning (L159) asserts the clip warning; test_vintage_summary_payload_is_json_serializable (L237) asserts payload keys {vintage,roll_rate,warnings}. These lock in incremental behavior — the new default MUST stay backward-compatible (label_semantics='incremental' default) or these break.
- tests/test_strategy_vintage.py — test_vintage_curve_wraps_phase4v_wide_curve_and_counts (L9) asserts vintage_curve wraps cum_bad_rate wide; test_strategy_package_exports_vintage_functions (L134). Adding a param with a default keeps these green; a new snapshot-path test is needed.
- GAP: NO existing test feeds a snapshot-flag bad_col to the STRATEGY vintage_curve tool and asserts it does NOT double-count — this is the exact silent-corruption bug. report_compute's snapshot protection has no direct unit test asserting the double-count avoidance either (only the comment).
- GAP: NO test asserts VintageCurve.warnings threading or that tool_vintage_curve surfaces point-level data_quality_warnings (unlike tool_roll_rate which is covered).
- GAP: NO test for the forced-confirmation gate on the strategy path when label_semantics is undeclared.

**补丁方案（伪代码）**：
Follow the NaN-label gate precedent (typed error in marvis/data/errors.py + explicit flag threaded from tool → template, structured to_detail() crossing the subprocess boundary, risk-flag routing in gates/contracts.py).

(a) KERNEL — marvis/validation/vintage.py compute_vintage_curve:
  add param `label_semantics: str = "incremental"`; validate ∈ {"incremental","snapshot"} else ValueError.
  Keep the existing first pass (per-MOB marginal rows) unchanged.
  if label_semantics == "incremental":  run the EXISTING second pass (accumulate bad_numerator over cohort_denominator → cum_bad_rate + clip warning). [current behavior, byte-identical]
  if label_semantics == "snapshot":  do NOT accumulate. For each MOB row set `cum_bad_rate = row["bad_rate"]` (the per-MOB marginal rate IS the true cumulative rate — exactly report_compute's metric='bad_rate' read). Add a red-flag heuristic: compute per-cohort per-MOB bad_count sequence; if for EVERY cohort bad_count is non-decreasing across ascending MOB (monotone), that is consistent with snapshot; if the caller declared 'incremental' but the data is globally monotone (or vice-versa: declared snapshot but bad_count ever DECREASES, impossible for true snapshot flags), append a `data_quality_warnings` entry ("per-MOB bad_count non-decreasing across all cohorts — data looks like a SNAPSHOT flag; declared=<semantics>").
  Keep VintagePoint.data_quality_warnings as the carrier (already exists).

(b) TYPED ERROR + GATE — new `LabelSemanticsNotDeclaredError(DataLayerError)` in marvis/data/errors.py mirroring NanLabelNotConfirmedError: fields target_col, n_cohorts, monotone_heuristic (bool), to_detail() → {"kind":"label_semantics_not_declared", target_col, monotone_heuristic, "examples": {"incremental":"每行=该MOB当期新发生的坏 (会累加)", "snapshot":"每行/列=截至该MOB累计是否坏 (已单调, 不再累加)"}}. Add ErrorKind.LABEL_SEMANTICS_NOT_DECLARED="label_semantics_not_declared" in marvis/errors.py. In marvis/packs/strategy/tools.py tool_vintage_curve: read `label_semantics = _optional_str(inputs.get("label_semantics"))`; if None → raise LabelSemanticsNotDeclaredError (with the monotone heuristic computed on the labeled frame) so the driver pauses at a gate asking the user to pick, with the two concrete examples. subprocess_worker._structured_error_detail (already generic) carries it across; gates/contracts._HIGH_RISK_GATE_SOURCE_TOOLS already maps "vintage_curve"→strategy_direction_approval so the gate is non-auto-confirmable — no change needed there, but the gate reply must write label_semantics into the step inputs (band_edges/selection override precedent).

(c) CONTRACT + OUTPUT + RENDERER:
  contracts.py: `VintageCurve` gains `warnings: tuple[str, ...] = ()` (default keeps existing constructors green).
  strategy/vintage.py vintage_curve: accept `label_semantics`; pick `metric = "bad_rate" if label_semantics=="snapshot" else "cum_bad_rate"`; pass label_semantics to compute_vintage_curve; collect `warnings = tuple(w for p in points for w in p.data_quality_warnings)` into VintageCurve.warnings.
  tools.py tool_vintage_curve: add `"warnings": list(curve.warnings)` to the returned dict (mirror tool_roll_rate's data_quality_warnings at tools.py:97).
  renderers.py _render_vintage_curve: append `o.get("warnings")` as 🚩 lines (mirror _render_slice_aggregate:1222-1224).

(d) MANIFEST — marvis/packs/strategy/manifest.json vintage_curve tool (additionalProperties:false on BOTH schemas is load-bearing): add `"label_semantics": {"type":"string","enum":["incremental","snapshot"]}` to input_schema.properties (NOT required — its absence is what triggers the gate on the strategy path); add `"warnings": {"type":"array","items":{"type":"string"}}` to output_schema.properties (add to required list alongside cohorts/mob_axis/... so it is always emitted).

(e) TEMPLATE + SETUP — orchestrator/templates/strategy.py VINTAGE_ANALYSIS: add SlotSpec("label_semantics", False, "user", "...") and SlotSpec("drop_nan_labels", False, ...); add "label_semantics":"{slot:label_semantics}" (baked literal-null default like band_edges so the gate override reaches it) + "drop_nan_labels":"{slot:drop_nan_labels}" to the vintage_curve step inputs_template. agent/vintage_setup.py VintageProposal: add `label_semantics: str | None = None` + heuristic detection (monotone bad flags → propose "snapshot") and include it in template_slots(). report_compute.compute_vintage_report: pass label_semantics="snapshot" explicitly to make its self-protection the canonical snapshot code path.

**新增失败形状测试**：
- tests/validation/test_vintage.py::test_snapshot_semantics_does_not_accumulate_bad — feed a cohort where the same loans are flagged bad at consecutive MOBs (snapshot flags: bad at mob0 stays bad at mob1), call compute_vintage_curve(label_semantics='snapshot'); assert cum_bad_rate == per-MOB bad_rate (NOT accumulated) i.e. equals the report_compute metric='bad_rate' values, and is NOT >1.0 / not double-counted.
- tests/validation/test_vintage.py::test_incremental_semantics_is_backward_compatible — same fixtures as existing tests but explicitly pass label_semantics='incremental'; assert identical cum_bad_rate values to the current tests (accumulation preserved).
- tests/validation/test_vintage.py::test_monotone_bad_count_across_all_cohorts_flags_snapshot_red_flag — feed data whose per-MOB bad_count is non-decreasing in every cohort while declared incremental; assert a data_quality_warnings entry naming 'snapshot'/'看起来是快照'.
- tests/test_strategy_vintage.py::test_vintage_curve_snapshot_picks_bad_rate_metric — vintage_curve(..., label_semantics='snapshot') produces curves equal to vintage_curve_wide(points, metric='bad_rate'); label_semantics='incremental' equals metric='cum_bad_rate'.
- tests/test_strategy_vintage.py::test_vintage_curve_threads_warnings_into_contract — a snapshot-flag frame declared incremental yields non-empty VintageCurve.warnings.
- tests/test_strategy_pack.py (or new)::test_tool_vintage_curve_raises_label_semantics_not_declared — call tool_vintage_curve WITHOUT label_semantics; assert LabelSemanticsNotDeclaredError.to_detail()['kind']=='label_semantics_not_declared' and it carries both incremental/snapshot examples.
- tests/test_strategy_pack.py::test_tool_vintage_curve_surfaces_warnings_in_output — output dict contains a 'warnings' list (schema additionalProperties:false compliance).
- tests/test_orch_templates.py::test_vintage_template_threads_label_semantics_and_drop_nan_labels — VINTAGE_ANALYSIS vintage step inputs_template contains label_semantics + drop_nan_labels keys.

**风险 / 口径变化**：口径变化 (metric-basis change): This changes what `cum_bad_rate` MEANS for snapshot-flag inputs on the strategy path — snapshot curves will now be numerically DIFFERENT (lower, un-accumulated) than the current silently-wrong output. Any adopted strategy or saved artifact computed from a snapshot bad_col was previously wrong; recomputation will shift the reported vintage curve. Backward-compat is preserved ONLY if label_semantics defaults to 'incremental' in the kernel (so validation tests and modeling paths that don't pass it are unchanged) — but the STRATEGY tool must make it REQUIRED-via-gate (typed error when undeclared), which is a behavior change: existing callers/tests that invoke tool_vintage_curve without label_semantics will now hit the gate/error. The manifest additionalProperties:false on both input and output schemas means the new fields MUST be added to the schema or the plugin runner's SchemaValidationError will reject the call/output at runtime (hard failure, not silent) — this is the single highest-risk mechanical detail. gates/contracts.py already maps vintage_curve to a high-risk flag so AUTO cannot silently confirm the new gate — good, but the gate-reply override channel (apply_adjust) must be wired to write label_semantics into step inputs exactly like band_edges/selection, or the user's choice never reaches the tool. Blast radius: modeling report_compute.compute_vintage_report should switch to explicit label_semantics='snapshot' (its comment already documents this) — low risk since its metric='bad_rate' read already gives the identical numbers. Determinism invariant (INV: deterministic metrics) is preserved: both paths are pure count-based reductions. The monotone heuristic is advisory-only (a warning), never mutates the curve, mirroring RollRateMatrix.data_quality_warnings — so it cannot corrupt results.

**设计决策点（提案已在补丁中给出，待审确认）**：
- Default label_semantics on the KERNEL vs the TOOL: the safest design is kernel default 'incremental' (keeps validation/modeling green) but the strategy TOOL raises when undeclared. Confirm the spec wants the strategy path to HARD-gate (typed error) vs soft-default to incremental with only a red flag — the fix text says 'typed error when strategy path called without declared semantics', which I've assumed.
- Should the monotone-bad_count heuristic also proactively raise (block) when declared='incremental' but data is globally monotone, or only emit a warning? The NaN-gate precedent blocks; the direction-conflict precedent blocks-unless-confirmed. I assumed warning-only for the heuristic and hard-gate only for the undeclared case, to avoid false-positive blocks on genuinely-incremental-but-coincidentally-monotone small cohorts.
- vintage_setup.detect_setup / build_vintage_proposal could auto-detect snapshot vs incremental from column names (mob_observe cumulative flags vs event columns) to pre-fill the slot; the spec mentions vintage_setup.detect_setup but the current detect_setup only resolves target/cohort/mob columns — confirm whether auto-heuristic proposal is in scope for T1-A1 or deferred.
- The 'warnings' output field: add to manifest output_schema.required (always emitted, even empty list) vs optional. I recommend required+always-present to match the deterministic-output contract, but that is a schema-tightening the reviewer should confirm against any golden-output snapshot tests (e.g. tests/test_agent_autodrive.py, test_e2e_journey.py) that assert on the vintage tool payload shape.

---

### A2 · expected_loss 跨快照月求和虚高 → 参考快照口径

**现状（bug 机理）**：expected_loss_estimate computes a point-in-time EL for EVERY snapshot month in the dataset and sums them into total_el, giving ~12x inflation on a 12-month panel (a loan destined to default is counted once per month it appears).

Chain of custody:
- marvis/packs/analysis/loss.py:98-100 — expected_loss_estimate calls `_el_by_month(df, contract, state_order, p_to_loss, lgd=lgd)` which returns (el_by_month rows, total_el, months_available).
- loss.py:174-199 — `_el_by_month` groups the WHOLE frame by month; for each month it sums balance*P(loss)*lgd into a MonthEL row AND accumulates the same value into `total_el` (line 198 `total_el += el`). So total_el = Σ over all snapshot months. Per-month rows (line 197) are individually correct; only the cross-month sum is misleading.
- loss.py:123-138 — `assumptions` dict is built with {lgd, horizon_months, matrix_window, loss_state}; there is NO entry describing the summation basis of total_el. ExpectedLossResult (loss.py:39-48) carries total_el and assumptions.
- marvis/packs/analysis/tools.py:122-144 — tool_expected_loss_estimate returns total_el verbatim (`float(result.total_el)`), plus el_by_month, chain, assumptions, red_flags.
- marvis/packs/analysis/report.py:44-49 — gate_summary_payload promotes `expected_loss.get("total_el")` into highlights['total_el'] (the headline gate number). No basis annotation.
- marvis/output/portfolio_report.py:71-76 — _write_overview writes total_el and each assumptions.{key} into the 组合概览 sheet; el_by_month + chain into the 预期损失 sheet (portfolio_report.py:119-127).
- marvis/agent/renderers.py:1683-1703 — _render_el_estimate headlines "合计 EL {total_el}" and renders the per-month table; it does NOT surface assumptions at all today.

Review doc docs/reviews/2026-07-04-full-read-and-owner-qa.md:42-45 (A2.2, HIGH, CONFIRMED) and the fix directive docs/plans/v2-trust-first-plan.md:70 both specify: change total_el to a reference-snapshot basis (default latest snapshot), keep per-month rows, annotate basis in gate/xlsx highlights and assumptions.

**关键代码位置**：
```
loss.py:174-199 (_el_by_month — the summation site):
```
174	def _el_by_month(
175	    df: pd.DataFrame,
176	    contract,
177	    states: tuple[str, ...],
178	    p_to_loss: dict[str, float],
179	    *,
180	    lgd: float,
181	) -> tuple[list[MonthEL], float, int]:
182	    frame = df[[contract.snapshot_col, contract.bucket_col, contract.balance_col]].copy()
183	    frame["_month"] = frame[contract.snapshot_col].map(parse_snapshot_month)
184	    frame["_bucket"] = frame[contract.bucket_col].astype(str)
185	    frame["_balance"] = pd.to_numeric(frame[contract.balance_col], errors="coerce").fillna(0.0).astype(float)
186	    frame = frame[frame["_month"].notna()]
187	    months = sorted({str(month) for month in frame["_month"].tolist()})
188	    rows: list[MonthEL] = []
189	    total_el = 0.0
190	    for month in months:
191	        month_frame = frame[frame["_month"] == month]
192	        balance = float(month_frame["_balance"].sum())
193	        el = 0.0
194	        for bucket, group in month_frame.groupby("_bucket", sort=False):
195	            probability = p_to_loss.get(str(bucket), 0.0)
196	            el += float(group["_balance"].sum()) * probability * float(lgd)
197	        rows.append(MonthEL(month=month, balance=balance, expected_loss=el))
198	        total_el += el
199	    return rows, total_el, len(months)
```

loss.py:98-138 (caller + assumptions assembly + result):
```
98	    el_by_month, total_el, months_available = _el_by_month(
99	        df, contract, state_order, p_to_loss, lgd=lgd
100	    )
...
123	    assumptions = {
124	        "lgd": float(lgd),
125	        "horizon_months": int(horizon_months),
126	        "matrix_window": list(migration.window_months),
127	        "loss_state": resolved_loss,
128	    }
129	    return ExpectedLossResult(
130	        loss_state=resolved_loss,
131	        lgd=float(lgd),
132	        horizon_months=int(horizon_months),
133	        chain=chain,
134	        el_by_month=el_by_month,
135	        total_el=float(total_el),
136	        assumptions=assumptions,
137	        red_flags=red_flags,
138	    )
```

report.py:44-49 (highlights promotion):
```
44	    highlights: dict = {}
45	    if expected_loss:
46	        highlights["total_el"] = expected_loss.get("total_el")
```

portfolio_report.py:71-76 (xlsx overview):
```
71	    if payload.expected_loss:
72	        rows.append(("预期损失", ""))
73	        rows.append(("total_el", _cell(payload.expected_loss.get("total_el"))))
74	        assumptions = payload.expected_loss.get("assumptions") or {}
75	        for key, value in assumptions.items():
76	            rows.append((f"假设.{key}", _cell(value)))
```

renderers.py:1687 (headline text):
```
1687	    text = f"**预期损失估计完成**:损失态 `{o.get('loss_state', '')}`，合计 EL {_fmt(o.get('total_el'))}。"
```
```

**受影响调用面**：
- marvis/packs/analysis/loss.py:98-100 — expected_loss_estimate unpacks (el_by_month, total_el, months_available) from _el_by_month; the total_el semantics change lives here + inside _el_by_month (loss.py:174-199)
- marvis/packs/analysis/loss.py:123-128 — assumptions dict; must gain a total_el_basis / reference_snapshot annotation key
- marvis/packs/analysis/loss.py:135 — ExpectedLossResult.total_el set from the new reference-snapshot value
- marvis/packs/analysis/tools.py:137-143 — tool_expected_loss_estimate returns total_el + assumptions verbatim (JSON contract to gate/report/renderer); no code change needed if assumptions carries the new key, but the returned total_el value changes
- marvis/packs/analysis/report.py:44-49 — gate_summary_payload highlights['total_el']; add a basis annotation into highlights (e.g. highlights['total_el_basis'] / reference_snapshot) so the gate headline states the口径
- marvis/output/portfolio_report.py:71-76 — _write_overview writes total_el + assumptions.* into 组合概览 sheet; new assumptions key auto-flows, but consider an explicit basis label row
- marvis/output/portfolio_report.py:119-127 — _write_expected_loss writes el_by_month + chain into 预期损失 sheet (per-month rows retained; unaffected)
- marvis/agent/renderers.py:1683-1703 — _render_el_estimate headline '合计 EL {total_el}' + per-month table; should annotate the basis (currently ignores assumptions entirely)
- marvis/packs/analysis/manifest.json:141-153 — expected_loss_estimate output_schema: total_el (number), el_by_month/chain (open objects), assumptions (open object). New assumptions keys and any per-month reference-flag field are schema-compatible (objects are unconstrained); no schema edit strictly required unless a top-level field is added

**现有测试与缺口**：
- tests/test_analysis_pack.py:258-288 test_expected_loss_absorbing_chain_hand_computed — asserts ONLY chain probabilities (current .25 / M1 .75 / bad 1.0) and red_flags kinds; does NOT assert total_el or el_by_month values, so it will not catch the basis change (safe)
- tests/test_analysis_pack.py:291-316 test_expected_loss_matrix_not_absorbing_warns — kernel-level red-flag assertion only; no total_el assertion
- tests/test_portfolio_api.py:134-145 test_gate_summary_aggregates_red_flags — asserts highlights['total_el'] == 123.0 from a hand-built el dict (passes total_el straight through; not tied to the kernel). Adding a basis key to highlights must not break this exact-value assertion
- tests/test_portfolio_api.py:148-169 test_report_carries_numbers_not_recompute — feeds a hand-built el dict with total_el/el_by_month/assumptions and asserts the xlsx 预期损失 sheet echoes 999.0 then 111.0; only checks pass-through, not computation basis
- tests/test_portfolio_api.py:179-211 test_portfolio_report_tool_registers_artifact_audit — runs the real EL tool then the report; asserts sheet list + artifact audit, NOT total_el value
- GAP: NO test locks the total_el computation basis. The review (docs/reviews/2026-07-04...md:45) explicitly notes 'none found'. A 12-month stable panel would currently yield ~12x the single-snapshot EL and no test would fail.

**补丁方案（伪代码）**：
Goal: total_el becomes EL of ONE reference snapshot (default = latest month), per-month rows unchanged, basis annotated in assumptions + gate highlights + xlsx + renderer.

1) loss.py `_el_by_month` (174-199): split the per-month row build from the total.
   - Keep the loop producing `rows: list[MonthEL]` (per-month EL) exactly as today.
   - Stop accumulating `total_el += el`. Instead, after building `rows`, pick a reference month:
     `reference_month = months[-1]` (latest; `months` is already sorted asc at line 187). Guard empty: if not months -> reference_month=None, total_el=0.0.
     `total_el = next((r.expected_loss for r in rows if r.month == reference_month), 0.0)`.
   - Return `(rows, float(total_el), len(months), reference_month)` — extend the tuple with reference_month so the caller can annotate. (Update the return type hint.)

2) loss.py expected_loss_estimate (98-100): unpack the extra value:
   `el_by_month, total_el, months_available, reference_month = _el_by_month(...)`.

3) loss.py assumptions (123-128): add basis keys:
   `assumptions["total_el_basis"] = "reference_snapshot"`
   `assumptions["reference_snapshot"] = reference_month`  (the month string, or None)
   (These are the machine-readable口径 that flow verbatim to gate/xlsx/renderer.)

4) (Optional, recommended) Add an explicit MonthEL flag so consumers can mark the reference row: either add `is_reference: bool` to the MonthEL dataclass (loss.py:32-36) set True for reference_month, OR leave rows unchanged and rely on assumptions.reference_snapshot. Prefer the assumptions-only route to keep MonthEL/asdict JSON stable unless the renderer wants to star the row.

5) report.py gate_summary_payload (44-49): annotate the headline口径:
   `highlights["total_el"] = expected_loss.get("total_el")`
   `assumptions = expected_loss.get("assumptions") or {}`
   `highlights["total_el_basis"] = assumptions.get("total_el_basis")`
   `highlights["reference_snapshot"] = assumptions.get("reference_snapshot")`
   (Pure pass-through of already-present assumptions fields — INV-1 presentation only.)

6) portfolio_report.py: assumptions.* already auto-written by _write_overview (74-76), so the new keys appear as 假设.total_el_basis / 假设.reference_snapshot rows automatically. Optionally relabel the total_el row to e.g. "total_el（参考快照口径）". Minimal: no code change needed beyond relying on the new assumptions keys.

7) renderers.py _render_el_estimate (1683-1703): surface the basis in the headline:
   `assumptions = o.get("assumptions") or {}`
   `ref = assumptions.get("reference_snapshot")`
   `basis_note = f"（参考快照 {ref} 口径）" if ref else ""`
   `text = f"**预期损失估计完成**:损失态 ...，合计 EL {_fmt(o.get('total_el'))}{basis_note}。"`
   Per-month table unchanged (optionally mark the reference month row).

8) manifest.json: no required change — assumptions/el_by_month are open objects. If MonthEL gains is_reference or a top-level reference_snapshot is added, they remain schema-valid under the existing `{"type":"object"}` items.

**新增失败形状测试**：
- tests/test_analysis_pack.py — test_expected_loss_total_el_is_reference_snapshot_not_sum: build a multi-month stable panel (e.g. reuse _absorbing_hand_frame or a 3+ month frame with identical balances each month). Assert total_el == the LATEST month's el_by_month row expected_loss (NOT the sum). Concretely: total_el == pytest.approx(el_by_month[-1]['expected_loss']) and total_el < sum(r['expected_loss'] for r in el_by_month) when months>1 — this is the anti-inflation lock the review says is missing.
- tests/test_analysis_pack.py — test_expected_loss_assumptions_annotate_basis: assert result assumptions['total_el_basis'] == 'reference_snapshot' and assumptions['reference_snapshot'] == the latest snapshot month string (dirty shape: multi-month panel; assertion: the口径 is machine-readable).
- tests/test_analysis_pack.py — test_expected_loss_per_month_rows_unchanged: assert el_by_month still has one row per snapshot month with the same per-month EL as before the change (guards that only the sum changed, not the rows).
- tests/test_analysis_pack.py — test_expected_loss_single_month_total_equals_month: dirty shape = single-snapshot dataset; assert total_el == that month's EL and reference_snapshot == that month (edge: no regression for 1-month panels).
- tests/test_portfolio_api.py — test_gate_summary_promotes_el_basis: feed el dict with assumptions={'total_el_basis':'reference_snapshot','reference_snapshot':'2025-12'}; assert payload['highlights']['total_el_basis']=='reference_snapshot' and ['reference_snapshot']=='2025-12' while keeping existing highlights['total_el'] assertion intact.
- tests/test_portfolio_api.py — extend test_report_carries_numbers_not_recompute or add test_report_writes_el_basis: assert the 组合概览 sheet contains a '假设.reference_snapshot' (or 假设.total_el_basis) cell so the xlsx口径 annotation is locked.

**风险 / 口径变化**：口径变化 (metric-basis change) — this is a deliberate, breaking change to the headline number: total_el drops from Σ-over-months to a single reference snapshot (~1/12 on a 12-month panel). Blast radius:
- Gate headline highlights['total_el'] and the xlsx 组合概览 total_el cell will show a materially smaller number; any downstream doc/expectation of the old value is invalidated (intended). The added assumptions.total_el_basis / reference_snapshot + gate/renderer annotations are what make the new口径 self-documenting.
- Backward-compat of JSON contract: total_el stays a number; new assumptions keys are additive; el_by_month/chain unchanged. manifest output_schema stays valid (open objects). Adding is_reference to MonthEL would change the asdict() JSON of el_by_month rows — only do it if the renderer needs it, and confirm no strict consumer rejects extra keys (none seen; renderer reads by key, xlsx _write_dict_table derives headers dynamically).
- Reference-month choice: 'latest snapshot' assumes months sort lexicographically correct (they do — parse_snapshot_month normalizes to YYYY-MM strings, sorted asc at loss.py:187). If a caller passes a `window` that filters the migration matrix but not the EL frame, note that _el_by_month uses the FULL df (not window) — the reference month is the latest month present in df, which may differ from the matrix window; call this out (possible open question: should reference snapshot be constrained to the window?).
- Existing tests: test_gate_summary_aggregates_red_flags asserts exact total_el==123.0 via hand-built dict (unaffected, pass-through). test_report_carries_numbers_not_recompute similarly hand-builds — unaffected. Only the kernel tests (which currently avoid asserting total_el) are truly affected; new tests fill the gap.

**设计决策点（提案已在补丁中给出，待审确认）**：
- Reference-snapshot default: spec says 'default latest snapshot' — confirm 'latest' = max(months) present in the EL frame (current sketch) vs latest of the migration `window`. When window is passed, the matrix is windowed but _el_by_month scans the full df; should the reference month be clamped to the window's last month for consistency?
- Should the reference snapshot be user-selectable via a new tool input (e.g. reference_snapshot / as_of_month) surfaced in manifest input_schema, or is default-latest sufficient for T1-A2? The directive only mandates 'default latest', implying a param is optional/future.
- Should the per-month EL table also mark which row is the reference (is_reference flag / ★ in renderer) so the user visually ties total_el to a specific month? Sketch keeps it optional to avoid changing MonthEL JSON.
- Does any other consumer (docs, notebooks, monitoring, or a strategy tool) read highlights['total_el'] or the xlsx total_el cell and compare against a stored/expected value that would need updating for the new口径? A repo-wide grep for 'total_el' outside the files in scope was not exhaustively run for non-analysis consumers.

---

### A3 · sentinel 掩码不进预处理重放链 → 新增 sentinel step

**现状（bug 机理）**：Sentinel masking (e.g. -999) is applied at fit/derive time by four FEATURE tools but is NEVER recorded in the persisted preprocessing chain, so scoring-time replay (apply_preprocessing_steps) and the handoff notebook treat -999 as a genuine value -> train/serve skew.

Concretely, in marvis/packs/feature/tools.py each tool resolves per-column sentinels via _sentinel_values_for (L678-694) and masks them to NaN via mask_sentinel_values BEFORE fitting, then registers the derived frame + preprocessing step via _register_frame:
- tool_impute_missing (L579-631): masks (L599 masked=mask_sentinel_values(out[column],...)), builds the __was_missing indicator from the MASKED NaN mask (L600 masked.isna()), fills NaN with a train-only value, but emits ONLY {kind:"missing_indicator"} + {kind:"impute", params:fill_values} (L610-616). The sentinel set is in neither step.
- tool_cap_outliers (L634-675): masks (L653), fits bounds on masked fit-values, emits ONLY {kind:"cap", params:bounds} (L668).
- tool_normalize (L534-576): masks fit and full series (L546/L548), emits ONLY {kind:"normalize", params:scaler_params} (L569).
- tool_woe_encode (L385-435): masks out[feature] (L396-397), fits edges/WOE on the masked frame (so na_woe absorbs sentinel rows), emits ONLY {kind:"woe", params:woe_maps} (L426). tool_woe_encode_categorical (L458-510) does NOT accept sentinel_values today (no _sentinel_values_for call), so it is out of scope.
- tool_bin_feature (L299-347) masks (L314-317) but never registers a derived frame/chain (report-only), so no replay concern.

The sidecar write in _register_frame (L805-847) persists exactly whatever steps the tool passed (L836-844) plus the source chain read first (L835), so it inherits the gap. train_tools._preprocessing_steps_for_training (L575-585) reads that sidecar chain straight onto artifact.params["preprocessing_steps"]; that flows to scoring._ModelArtifactScorer._replay_preprocessing (scoring.py L305-317) and to the handoff notebook (handoff.py L580-581, L595, L636).

At replay, none of the step appliers re-mask sentinels, so skew appears in ALL kinds:
- _apply_impute (preprocessing.py L141-150) does out[column].fillna(value) -> a raw -999 is finite, not NaN, so it is NOT filled: -999 survives as a genuine value (train filled it to the median).
- _apply_missing_indicator (L153-166) computes out[column].isna() -> a raw -999 gives indicator 0, but at fit the masked -999 gave indicator 1: flipped.
- _apply_cap (L169-186) clips only np.isfinite values, so -999 is clipped to [lower,upper] instead of being treated as NaN (fit excluded it): a -999 becomes the lower bound value, not NaN.
- _apply_normalize (L189-198) scales the finite -999 with the fitted min/max -> a large-magnitude negative z/minmax value, not NaN.
- _apply_woe (L220-240) calls woe_encode; assign_bins (binning.py L220-233) does np.clip(assigned,0,edges.size-2) so the finite -999 is clipped into bin 0 and gets bin-0 WOE, NOT na_woe (fit put sentinel rows into the NaN/na_woe bucket): direct WOE skew.

**关键代码位置**：
```
marvis/packs/feature/tools.py L588-616 (tool_impute_missing core loop):
588	    sentinel_values = _sentinel_values_for(inputs, columns)
589	    add_indicators = bool(inputs.get("add_indicators"))
590	    indicator_columns: list[str] = []
591	    for column in columns:
592	        column_sentinels = sentinel_values.get(column)
593	        _filled_fit, value = impute_missing(
594	            out.loc[fit_mask, column], strategy=str(inputs["strategy"]),
595	            fill_value=inputs.get("fill_value"), sentinel_values=column_sentinels)
599	        masked = mask_sentinel_values(out[column], column_sentinels)
600	        if add_indicators and masked.isna().any():
604	            indicator_name = _unique_column_name(f"{column}__was_missing", out.columns)
605	            out[indicator_name] = masked.isna().astype(int)
606	            indicators[column] = indicator_name
608	        out[column] = masked.fillna(value)
609	        fill_values[column] = value
610	    preprocessing_steps = []
611	    if indicators:
613	        preprocessing_steps.append({"kind": "missing_indicator", "columns": list(indicators), "params": _jsonable(indicators)})
616	    preprocessing_steps.append({"kind": "impute", "columns": columns, "params": _jsonable(fill_values)})

marvis/packs/feature/tools.py L678-694 (_sentinel_values_for; the per-column resolver every fix site reuses):
684	    raw = inputs.get("sentinel_values")
685	    if not raw: return {}
687	    if isinstance(raw, dict):
688	        return {str(column): [float(v) for v in values] for column, values in raw.items() if str(column) in columns and values}
693	    flat = [float(v) for v in raw]
694	    return {column: flat for column in columns} if flat else {}

marvis/feature/preprocessing.py L107-138 (apply_preprocessing_steps dispatch - where a new 'sentinel' kind must be handled, and ordering enforced):
118	    for step in steps:
119	        kind = str(step.get("kind") or "")
122	        if kind == "impute": out = _apply_impute(out, columns, params)
124	        elif kind == "cap": out = _apply_cap(out, columns, params)
126	        elif kind == "normalize": out = _apply_normalize(out, columns, params)
128	        elif kind == "onehot": out = _apply_onehot(out, columns, params)
130	        elif kind == "missing_indicator": out = _apply_missing_indicator(out, columns, params)
132	        elif kind == "woe": out = _apply_woe(out, columns, params)
134	        elif kind == "categorical_woe": out = _apply_categorical_woe_step(out, columns, params)
136	        else: raise FeatureError(f"unsupported preprocessing step kind: {kind!r}")

marvis/feature/transform.py L133-141 (mask_sentinel_values - the exact fit-time masking a 'sentinel' replay step must reproduce):
137	    if not sentinel_values: return series
139	    numeric = pd.to_numeric(series, errors="coerce")
140	    mask = numeric.isin([float(value) for value in sentinel_values])
141	    return series.mask(mask)

marvis/feature/binning.py L226-233 (why a raw sentinel skews WOE if not pre-masked - it is clipped into bin 0, never na):
226	    arr = np.asarray(values, dtype=float)
227	    out = np.full(arr.shape, -1, dtype=int)
228	    mask = np.isfinite(arr)
231	    assigned = np.searchsorted(edges[1:-1], arr[mask], side="right")
232	    out[mask] = np.clip(assigned, 0, edges.size - 2)
```

**受影响调用面**：
- marvis/packs/feature/tools.py:610-624 tool_impute_missing - must prepend a {kind:'sentinel'} step (before missing_indicator) into preprocessing_steps when sentinel_values were used
- marvis/packs/feature/tools.py:662-668 tool_cap_outliers - must emit a sentinel step ahead of the cap step
- marvis/packs/feature/tools.py:563-570 tool_normalize - must emit a sentinel step ahead of the normalize step
- marvis/packs/feature/tools.py:420-427 tool_woe_encode - must emit a sentinel step ahead of the woe step
- marvis/packs/feature/tools.py:805-847 _register_frame - already persists whatever steps list is passed + source chain; supports both preprocessing_step (single) and preprocessing_steps (list). cap/normalize/woe currently pass single preprocessing_step and must switch to the list form to carry the extra sentinel step
- marvis/feature/preprocessing.py:107-138 apply_preprocessing_steps - add elif kind=='sentinel' branch dispatching to a new _apply_sentinel
- marvis/feature/preprocessing.py:14-41 module docstring - the PreprocessingStep kind enumeration comment must add 'sentinel'
- marvis/packs/modeling/train_tools.py:575-585 _preprocessing_steps_for_training - no change needed; reads chain onto artifact automatically, but confirms the sentinel step will reach the artifact
- marvis/packs/modeling/scoring.py:305-317 _ModelArtifactScorer._replay_preprocessing - no change needed; apply_preprocessing_steps handles the new kind
- marvis/packs/modeling/handoff.py:557-646 _scoring_notebook_source / _preprocessing_steps (L366-372) - no change needed; notebook already calls apply_preprocessing_steps(dataframe, RMC_PREPROCESSING_STEPS) which will include the sentinel step
- marvis/packs/feature/tools.py:299-347 tool_bin_feature - report-only, registers no chain; no change
- marvis/packs/feature/tools.py:458-510 tool_woe_encode_categorical - does not accept sentinel_values today; out of scope unless the spec extends sentinel support to it

**现有测试与缺口**：
- tests/test_feature_pack.py:1328 test_impute_cap_normalize_bin_woe_accept_sentinel_values (@slow) - asserts fit/derive-time correctness ONLY (fill=3.5, bounds 1..6, minmax min/max 1..6, na_bin count, na_woe not None); it NEVER reads the preprocessing chain and NEVER replays via apply_preprocessing_steps, so the train/serve skew is completely uncaught - this is the core gap
- tests/test_feature_pack.py:1020 test_impute_missing_add_indicators_emits_was_missing_columns_and_sidecar_step - asserts chain kinds ['missing_indicator','impute'] and replays, but with NO sentinel_values, so it locks in the current (sentinel-free) chain shape and would need updating if a sentinel step is unconditionally prepended (it should only be added when sentinels are used)
- tests/test_feature_pack.py:1097 test_impute_missing_without_add_indicators_omits_indicator_columns - asserts chain kinds ['impute'] with no sentinel; must remain green (no sentinels -> no sentinel step)
- tests/test_feature_pack.py:1138 test_woe_encode_and_woe_encode_categorical_persist_preprocessing_chain_sidecar - asserts chain kinds ['woe'] / ['categorical_woe'] with no sentinels; must remain green
- tests/test_feature_pack.py:983-1017 (PREP-2 chain accumulation test) - asserts ['impute','cap','normalize'] and onehot chain with no sentinels; must remain green
- tests/test_modeling_pack.py:3429 test_train_model_persists_combined_impute_cap_woe_chain_and_exports_pmml_consistently (@slow) - the canonical round-trip proof (chain onto artifact, replay raw==already-transformed), but uses NO sentinels; the new coverage should mirror this shape WITH a sentinel column
- tests/test_feature_pack.py:1300 test_screen_features_reports_sentinel_columns_notice - only covers detection/notice, not replay

**补丁方案（伪代码）**：
1) marvis/feature/preprocessing.py:
- Extend module docstring kind list (L14-41) to include 'sentinel'.
- Add branch in apply_preprocessing_steps (L118-137): `elif kind == "sentinel": out = _apply_sentinel(out, columns, params)`.
- New `_apply_sentinel(frame, columns, params)`: for each column present in frame, read `values = params.get(column)` (a list[float]); `out[column] = mask_sentinel_values(out[column], values)` reusing marvis.feature.transform.mask_sentinel_values (import it). Skip missing columns per-column like the other appliers. This masks raw sentinels -> NaN so every downstream step (_apply_missing_indicator sees NaN=1, _apply_impute fills the NaN, _apply_cap treats it non-finite, _apply_normalize yields NaN, _apply_woe -> assign_bins returns -1 -> na_woe) exactly reproduces fit-time behavior.
- CRITICAL ordering: the 'sentinel' step must be applied FIRST for its columns, i.e. emitted into the chain BEFORE the paired missing_indicator/impute/cap/normalize/woe step. Since apply_preprocessing_steps iterates in list order, correctness is guaranteed purely by emit order in tools.py.

2) marvis/packs/feature/tools.py - add a small helper `_sentinel_step(sentinel_values: dict[str,list[float]]) -> dict|None` returning `{"kind":"sentinel","columns":list(sentinel_values),"params":_jsonable(sentinel_values)}` when non-empty else None. Then at each tool:
- tool_impute_missing (L610-616): build `preprocessing_steps=[]`; if sentinel_values: prepend the sentinel step; then existing missing_indicator (if any) then impute. Order: [sentinel, missing_indicator, impute].
- tool_cap_outliers (L662-669): switch from `preprocessing_step={kind:cap,...}` to `preprocessing_steps=[*sentinel_step_if_any, {kind:cap,...}]`.
- tool_normalize (L563-570): same -> [*sentinel, {kind:normalize,...}].
- tool_woe_encode (L420-427): same -> [*sentinel, {kind:woe,...}].
_register_frame already accepts preprocessing_steps (list) and appends onto the source chain (L820-847), so no _register_frame change is required; only the tools switch to the list form and prepend the sentinel step.

3) No change to train_tools/_preprocessing_steps_for_training, scoring._replay_preprocessing, or handoff notebook - they consume artifact.params["preprocessing_steps"] and call apply_preprocessing_steps, which now handles the sentinel kind. The handoff notebook string (handoff.py L595/L636) already serializes and replays the full chain.

4) Interaction notes to encode in the design:
- __was_missing indicator: because sentinel runs first (masks -999->NaN), _apply_missing_indicator's out[column].isna() now flags sentinel rows as 1, matching fit (which used masked.isna()). Correct only if sentinel precedes missing_indicator in the chain.
- na_woe: sentinel-first ensures the raw -999 becomes NaN -> assign_bins -1 -> woe.na_woe, matching the fit that put sentinel rows in the NaN bucket. Without it, np.clip in assign_bins would give it bin-0 WOE (skew).
- categorical_woe: tool_woe_encode_categorical does not take sentinel_values today; leave out of scope (or explicitly extend if spec wants it).

**新增失败形状测试**：
- tests/test_feature_pack.py: extend/replace test_impute_cap_normalize_bin_woe_accept_sentinel_values to ALSO read_preprocessing_chain on each result and assert the sentinel step is present and ordered first, e.g. impute chain == ['sentinel','impute'] (or ['sentinel','missing_indicator','impute'] with add_indicators), cap == ['sentinel','cap'], normalize == ['sentinel','normalize'], woe == ['sentinel','woe']; assert chain[0]['params'] == {'amount':[-999.0]}
- tests/test_feature_pack.py new: feed a fresh raw frame containing -999 rows through apply_preprocessing_steps(raw, chain) for the impute chain and assert replayed['amount'] fills the -999 rows with the SAME train median (3.5) as the derived frame - i.e. replayed equals the derived column, proving no -999 survives (dirty shape: raw sentinel at serve time)
- tests/test_feature_pack.py new: WOE replay skew - raw frame with a -999 row, replay the woe chain, assert replayed['amount_woe'] for the sentinel row == na_woe (not the bin-0 WOE); contrast against a control chain WITHOUT the sentinel step to show it would otherwise get a bin value
- tests/test_feature_pack.py new: __was_missing interaction - impute with add_indicators + sentinel_values, replay on raw -999 data, assert replayed[col__was_missing]==1 for the sentinel row (dirty shape: sentinel present, indicator must flip to 1)
- tests/test_feature_pack.py new: no-sentinel regression - impute/cap/normalize/woe WITHOUT sentinel_values must NOT emit a sentinel step (chain kinds unchanged), guarding the existing PREP-2/PREP-8 tests
- tests/test_modeling_pack.py new (mirror test_train_model_persists_combined_impute_cap_woe_chain_and_exports_pmml_consistently @3429): train a model off a chain built WITH sentinel_values, assert artifact.params['preprocessing_steps'][0]['kind']=='sentinel', then _ModelArtifactScorer(replay_preprocessing=True).score(raw_frame_with_-999) == already_transformed_scorer.score(chained_frame) - end-to-end proof the sentinel step round-trips through train->artifact->replay

**风险 / 口径变化**：Blast radius is small and additive: a new 'sentinel' step kind + emit-order change in 4 FEATURE tools; consumers (train_tools, scoring, handoff notebook) are untouched because they already delegate to apply_preprocessing_steps.

Backward-compat: pre-fix artifacts/sidecars have no sentinel step -> apply_preprocessing_steps still fine (unknown-kind path only triggers on the literal string 'sentinel', which old chains never contain). Old handoff notebooks embed apply_preprocessing_steps by import (not vendored), so a redeployed marvis picks up the new kind automatically; already-exported notebooks won't retroactively fix but that matches the existing PREP-2 posture.

口径变化 (metric-basis change): NONE at fit/derive time - the derived parquet is byte-identical (sentinels were already masked). The change is purely that serve-time scores on RAW data now match train, which is the intended fix, not a silent metric shift on existing already-transformed scoring paths (replay is opt-in via replay_preprocessing=True; default report/stress/calibration paths that score the already-transformed frame are unaffected because a sentinel step on already-NaN'd data is a no-op).

Ordering hazard: the sentinel step MUST precede missing_indicator/impute/cap/normalize/woe for the same column; a wrong emit order silently reintroduces the skew (esp. the __was_missing flip and na_woe). This is the single behavioral invariant the new tests must lock.

Determinism: mask_sentinel_values is pure/deterministic (isin), preserving the platform's determinism invariant.

Edge: mask_sentinel_values on non-numeric columns coerces via pd.to_numeric(errors='coerce'); consistent with fit-time. tool_woe_encode_categorical is intentionally left without sentinel support (no behavior change) - flag as open question if the spec wants parity.

**设计决策点（提案已在补丁中给出，待审确认）**：
- Should the fix also add sentinel_values support to tool_woe_encode_categorical (L458-510)? Today it accepts no sentinel_values, so there is no skew to fix, but a sentinel-value NaN mask before categorical WOE fit would route sentinel rows to na_woe/default_woe consistently. Spec says 'consider interaction with na_woe' - recommend scoping this fix to the four numeric tools that already mask and adding categorical only if explicitly wanted.
- Emit shape: a single combined sentinel step per tool call carrying {column:[values]} for all columns (recommended, matches _sentinel_values_for output and _apply_sentinel loop), vs per-column steps. Combined is simpler and order-safe since one sentinel step precedes all paired transform steps in the same tool call.
- When chaining multiple tools (impute->cap->woe) each with the SAME sentinel_values, each tool emits its own sentinel step, so the accumulated chain has repeated sentinel steps. That is harmless (masking an already-NaN value is a no-op) but slightly verbose; acceptable, or optionally dedupe identical consecutive sentinel steps. Recommend keeping it simple (no dedupe) for auditability.
- params serialization: sentinel values are floats; _jsonable already handles them. Confirm the sidecar's sort_keys/default=str JSON round-trips a list under the column key (it does - list of floats).

---

### A4 · slice_aggregate 把 NULL 标签当好客户 → 剔除分母

**现状（bug 机理）**：In marvis/packs/data_ops/tools.py, tool_slice_aggregate compiles a whitelisted group-by aggregate into a single DuckDB SQL. Two of the aggregate ops encode credit-risk ratios with a CASE-over-avg pattern that puts every scanned row into the denominator, silently treating NULL and any non-matching label as "good"/"not-approved":

- bad_rate (L527): `avg(CASE WHEN try_cast({ident} AS DOUBLE) = 1 THEN 1.0 ELSE 0.0 END)` — a row whose label column is NULL, empty, non-castable ("N/A"), or any value other than exactly 1 (e.g. 2, -1, or an unlabeled application) lands in the ELSE branch and contributes 0.0 to the average. Because `avg()` over a CASE that never returns NULL divides by the full group count, unlabeled/unresolved rows deflate the reported bad_rate. Example: a group of 100 rows with 10 confirmed bad (label==1), 40 confirmed good (label==0), and 50 NULL/unlabeled reports 10/100 = 0.10 instead of the true labeled bad_rate 10/50 = 0.20 — understated by 2x.
- approval_rate (L532): `avg(CASE WHEN lower(trim(CAST({ident} AS VARCHAR))) = 'approve' THEN 1.0 ELSE 0.0 END)` — same denominator flaw. A row whose decision is NULL, blank, "pending", "review", or any non-"approve"/"reject" value counts as a non-approval, understating approval_rate. There is no way to distinguish "genuinely rejected" from "decision unknown".

Contrast with the sibling ops that are already correct-by-construction: mean (L518) is `avg(try_cast({ident} AS DOUBLE))`, and DuckDB's avg ignores NULLs, so try_cast failures (bad casts -> NULL) are correctly excluded from the mean denominator. bad_rate/approval_rate break this convention only because of the `ELSE 0.0` that converts "not affirmatively bad/approve" into a concrete 0.0 rather than NULL.

This also violates the platform's established finite/labeled convention used everywhere else in the codebase:
- marvis/feature/metrics.py `_finite_binary_pairs` (L346-356) masks rows to `np.isfinite(scores) & np.isfinite(target)` and asserts the surviving target is strictly binary 0/1 before any KS/AUC/bad-rate computation — NULL/NaN labels never enter the denominator.
- marvis/packs/modeling/report_tools.py L532-534 computes `bad_rate = bad_count / labeled_count if labeled_count else None` where labeled_count counts only non-NULL labels; report_compute.py L89-97 does the same (`labeled_count = int(target.notna().sum())`).

slice_aggregate is the one surface that diverges, so an analyst doing ad-hoc problem-solving ("即席问数") on a partially-labeled slice gets a silently understated bad_rate with no unlabeled_count to flag it.

No other CASE-pattern aggregate exists in the file (verified: only L527 and L532 use CASE WHEN; the _AGG_COMPARATORS dict at L370 is for filter operators, not metrics). count/sum/min/max/distinct do not have the flaw (count(*) is intentional total; distinct/min/max/sum ignore NULL by DuckDB semantics; sum wraps in coalesce(...,0) which is intended so an all-NULL group sums to 0 not NULL).

**关键代码位置**：
```
marvis/packs/data_ops/tools.py L523-532 (the two flawed metric exprs):

    if op == "bad_rate":
        if not col:
            raise ValueError("metric op 'bad_rate' requires the target column")
        ident = sql_identifier(col, allowed_columns)
        return f"avg(CASE WHEN try_cast({ident} AS DOUBLE) = 1 THEN 1.0 ELSE 0.0 END)"
    if op == "approval_rate":
        if not col:
            raise ValueError("metric op 'approval_rate' requires the decision column")
        ident = sql_identifier(col, allowed_columns)
        return f"avg(CASE WHEN lower(trim(CAST({ident} AS VARCHAR))) = 'approve' THEN 1.0 ELSE 0.0 END)"

Correct sibling for contrast, marvis/packs/data_ops/tools.py L515-522:

        numeric = f"try_cast({ident} AS DOUBLE)"
        return {
            "sum": f"coalesce(sum({numeric}), 0)",
            "mean": f"avg({numeric})",
            ...

Row-building mechanism that constrains how a new output column can be exposed, marvis/packs/data_ops/tools.py L430-434:

    columns = [*group_by, *metric_labels]
    rows = [
        {column: _jsonable_cell(value) for column, value in zip(columns, record, strict=True)}
        for record in frame.itertuples(index=False, name=None)
    ]

zip(..., strict=True) means every SELECT column MUST have exactly one matching entry in `columns`; adding an SQL select without a matching label raises ValueError. So unlabeled_count must flow through the same metric_labels list, or through a parallel select list whose labels are appended to `columns`.

Platform labeled-denominator convention, marvis/packs/modeling/report_tools.py L532-534:

            labeled_count = int(np.sum(label_mask))
            bad_count = int(np.sum(labels[label_mask] == 1)) if labeled_count else 0
            bad_rate = (bad_count / labeled_count) if labeled_count else None

Finite/binary gate, marvis/feature/metrics.py L351-355:

    mask = np.isfinite(scores_arr) & np.isfinite(target_arr)
    ...
    if target_arr.size and not np.all(np.isin(target_arr, [0, 1])):
        raise FeatureError("target must be binary 0/1")
```

**受影响调用面**：
- marvis/packs/data_ops/tools.py L503 _metric_selects() builds `{expr} AS {label}` per metric using _metric_expr — the single caller of _metric_expr; labels come from _metric_label (L536-537)
- marvis/packs/data_ops/tools.py L400 tool_slice_aggregate calls `_metric_selects(metrics, allowed_columns)` -> (metric_selects, metric_labels); metric_labels feeds L410 _order_clause (sort_by can name a metric label) and L430 `columns = [*group_by, *metric_labels]`
- marvis/packs/data_ops/tools.py L430-434 row dict built by zip(columns, record, strict=True) — HARD constraint: SELECT arity must equal len(columns)
- marvis/agent/renderers.py L1201 _render_slice_aggregate consumes o['columns']/o['rows'] generically (lays every column into a table via _fmt) and echoes spec_echo + red_flags; registered at L1876 dispatch `"slice_aggregate": _render_slice_aggregate`. Any new numeric column renders automatically; a new red_flag message renders automatically at L1223-1224
- marvis/packs/data_ops/manifest.json L504-653 slice_aggregate manifest: input_schema metrics.op enum L528-537 (bad_rate/approval_rate live here), output_schema L609-646 (rows are free-form objects, columns is string array — additive output needs no rows schema change)
- tests/test_data_ops_slice_aggregate.py L63-101 test_slice_aggregate_hand_calculated_group_metrics asserts exact bad_rate_bad/approval_rate_decision values on a fully-labeled frame (all bad in {0,1}, all decision in {approve,reject}) — must stay green since fully-labeled denominators are unchanged

**现有测试与缺口**：
- tests/test_data_ops_slice_aggregate.py::test_slice_aggregate_hand_calculated_group_metrics (L63) — fully-labeled frame; asserts rows['A']['bad_rate_bad']==0.5, rows['B']['bad_rate_bad']==1.0, approval_rate 0.5 each. Fix must keep these identical (no NULL/non-binary labels present, so labeled_count==group count and the ratio is unchanged).
- tests/test_data_ops_slice_aggregate.py::test_slice_aggregate_rejects_injected_column_name (L103) — whitelist rejection, unaffected.
- tests/test_data_ops_slice_aggregate.py::test_slice_aggregate_hallucinated_metric_column_rejected (L118) — unknown metric col rejected, unaffected.
- tests/test_data_ops_slice_aggregate.py::test_slice_aggregate_truncated_and_between_in_filters (L132) — top_k truncation + between/in filters + empty_result flag, unaffected.
- tests/test_data_ops_slice_aggregate.py::test_slice_aggregate_writes_audit_row (L175) — audit row, unaffected.
- NOTE: _frame() (L53-60) has bad in {0,1} and decision in {approve,reject} with zero NULLs, so it exercises none of the buggy path — a new fixture with NULL/other labels is required.

**补丁方案（伪代码）**：
Goal: (1) exclude NULL/non-finite/non-binary labels from bad_rate and non-approve/reject-or-NULL decisions from approval_rate denominators, matching report_tools.py labeled_count semantics; (2) expose a per-group unlabeled_count so the deflation is visible; (3) keep fully-labeled results byte-identical.

FILE marvis/packs/data_ops/tools.py — _metric_expr (L508-533): make the CASE return NULL for "not labeled", not 0.0, so avg() drops it (mirroring the mean op at L518).

  def _metric_expr(op, col, allowed_columns):
      ...
      if op == "bad_rate":
          if not col: raise ValueError("metric op 'bad_rate' requires the target column")
          ident = sql_identifier(col, allowed_columns)
          num = f"try_cast({ident} AS DOUBLE)"
          # bad=1, good=0 count; everything else (NULL, non-castable, not-in-{0,1}) -> NULL -> dropped by avg
          return (f"avg(CASE WHEN {num} = 1 THEN 1.0 "
                  f"WHEN {num} = 0 THEN 0.0 ELSE NULL END)")
      if op == "approval_rate":
          if not col: raise ValueError("metric op 'approval_rate' requires the decision column")
          ident = sql_identifier(col, allowed_columns)
          norm = f"lower(trim(CAST({ident} AS VARCHAR)))"
          # approve=1, reject=0; NULL/blank/pending/other -> NULL -> dropped by avg
          return (f"avg(CASE WHEN {norm} = 'approve' THEN 1.0 "
                  f"WHEN {norm} = 'reject' THEN 0.0 ELSE NULL END)")

  Rationale: DuckDB avg() ignores NULL rows, so denominator == count of affirmatively-labeled rows == report_tools.labeled_count semantics. A fully-labeled group (all in {0,1} / {approve,reject}) yields the identical average — the existing hand-calc test stays green. An all-unlabeled group yields NULL (rendered n/a by _fmt), which is the honest answer.

  DESIGN CHOICE for approval_rate denominator: the fix intentionally puts only approve+reject in the denominator (unknown decisions dropped). Confirm with spec author whether "pending" should be denominator (approval_rate = approvals / all-decided-or-not) — this pseudocode uses decided-only, consistent with bad_rate labeled-only. See open_questions.

EXPOSE unlabeled_count — auto-derive it alongside every bad_rate/approval_rate metric rather than as a user-requestable op (keeps the op enum stable). In _metric_selects (L489-505), after appending the metric's own select, if op in {bad_rate, approval_rate} also append a companion count select and its label so the zip(strict=True) at L432 stays balanced:

  def _metric_selects(metrics, allowed_columns):
      selects, labels, seen = [], [], set()
      for metric in metrics:
          op = str(metric.get("op") or ""); col = _optional_str(metric.get("col"))
          label = _metric_label(op, col)
          if label in seen: raise ValueError(f"duplicate metric label: {label}")
          seen.add(label)
          selects.append(f"{_metric_expr(op, col, allowed_columns)} AS {_quote(label)}")
          labels.append(label)
          if op in {"bad_rate", "approval_rate"}:
              comp_label = f"unlabeled_count_{col}"      # deterministic, mirrors _metric_label
              if comp_label not in seen:
                  seen.add(comp_label)
                  selects.append(f"{_unlabeled_count_expr(op, col, allowed_columns)} AS {_quote(comp_label)}")
                  labels.append(comp_label)
      return selects, labels

  def _unlabeled_count_expr(op, col, allowed_columns):
      ident = sql_identifier(col, allowed_columns)
      if op == "bad_rate":
          num = f"try_cast({ident} AS DOUBLE)"
          return f"count(*) - count(CASE WHEN {num} IN (0,1) THEN 1 END)"
      norm = f"lower(trim(CAST({ident} AS VARCHAR)))"
      return f"count(*) - count(CASE WHEN {norm} IN ('approve','reject') THEN 1 END)"

  Because metric_labels now includes unlabeled_count_* labels, L430 `columns = [*group_by, *metric_labels]` and the L432 zip stay consistent automatically, _order_clause (L580-596) still works (unlabeled_count_* is a valid metric label if someone sorts by it), and renderers.py _render_slice_aggregate lays the extra column into the table with no change.

  Guard: if a group is fully labeled, unlabeled_count_* == 0 (harmless extra column). Test author may treat a nonzero unlabeled_count as the red-flag trigger (see new_tests) — optionally add a red_flag in tool_slice_aggregate after rows are built (L436) scanning rows for unlabeled_count_* > 0 to emit an amber "N 行标签缺失，坏率仅基于已标注样本" flag, consistent with existing empty_result/truncated flags at L437-448.

FILE marvis/packs/data_ops/manifest.json — no input_schema change (op enum unchanged; unlabeled_count is auto-derived, not user-selected). output_schema rows are already free-form `{"type":"object"}` (L618-623) so the extra key needs no schema edit. RECOMMENDED: update the slice_aggregate summary (L505) / add a note that bad_rate/approval_rate denominators exclude unlabeled rows and emit companion unlabeled_count_<col> columns, so the LLM's tool description reflects the new contract. If the spec prefers unlabeled_count be a formal output field, add it to output_schema.properties as an integer, but per-group placement inside rows makes a top-level field ambiguous for multi-group results — keep it per-row.

**新增失败形状测试**：
- tests/test_data_ops_slice_aggregate.py::test_bad_rate_excludes_null_and_nonbinary_labels — new fixture with a 'bad' column mixing 1, 0, NULL (NaN), and a non-binary value (e.g. 2 or 'N/A'); assert bad_rate_bad == confirmed_bad / (count of rows where label in {0,1}), NOT / total group size. Concretely a group of [1,1,0,NULL,'x'] must report 2/3, not 2/5.
- tests/test_data_ops_slice_aggregate.py::test_bad_rate_exposes_unlabeled_count — same fixture; assert row['unlabeled_count_bad'] == number of NULL/non-binary rows (e.g. 2), and that it is 0 for a fully-labeled group.
- tests/test_data_ops_slice_aggregate.py::test_approval_rate_excludes_unknown_decisions — decision column with 'approve','reject',NULL,'pending'; assert approval_rate_decision == approvals / (approve+reject count) and unlabeled_count_decision == count of NULL+pending.
- tests/test_data_ops_slice_aggregate.py::test_all_unlabeled_group_reports_null_rate — a group whose label column is entirely NULL: assert bad_rate_bad is None (rendered n/a) and unlabeled_count_bad == group size, and the tool does not raise.
- REGRESSION GUARD: extend/keep test_slice_aggregate_hand_calculated_group_metrics — its fully-labeled frame must still yield bad_rate_bad 0.5/1.0 and approval_rate 0.5/0.5 unchanged, and now also unlabeled_count_bad==0 / unlabeled_count_decision==0 for both channels.
- OPTIONAL if the amber red_flag is added: test that a slice with any unlabeled rows emits a red_flag with a stable code (e.g. 'unlabeled_present') while a fully-labeled slice does not.

**风险 / 口径变化**：["Output-shape change: every bad_rate/approval_rate query now returns an extra unlabeled_count_<col> column in columns[] and each row dict. Any downstream consumer that assumes a fixed column set (snapshot tests, the agent's numeric-citation guard, memory/context serializers) sees a new key. renderers.py handles it generically (lays all columns into the table), but confirm no other consumer hard-codes the column list. Alternative if this is unacceptable: expose unlabeled_count only as a red_flag/aggregate metadata field, not a per-row column — but that loses per-group granularity for multi-group slices.", "Determinism (INV-1): unlabeled_count_<col> is a deterministic count(*) - count(CASE...) so it is stable; but adding it changes ORDER BY tie-break surface only if someone sort_by='unlabeled_count_<col>' — it is a valid metric label so _order_clause handles it; no nondeterminism introduced.", "Semantic decision for approval_rate: the sketch drops unknown decisions from BOTH numerator and denominator (approval_rate = approvals / decided). If the business definition is approvals / all-applications (unknown = not-yet-approved), the denominator should stay count(*) and only the numerator excludes nothing — that is the OPPOSITE of this fix. Must be confirmed against the credit-risk convention before implementing; bad_rate labeled-only is unambiguous, approval_rate is not.", "empty_result vs unlabeled: a group with rows but all-unlabeled currently is NOT empty (has rows) yet reports NULL rate — ensure the empty_result red_flag logic (L437) still keys off `not rows` (row count) and does not confuse all-NULL-rate with empty.", "Backward-compat of stored/cited numbers: prior conversations may have cited the old (understated) bad_rate. This fix changes the number for any partially-labeled slice — intended, but note it is a behavior change, not purely additive. count/sum/min/max/distinct are untouched, so their citations are stable.", "The float NaN -> None mapping already exists in _jsonable_cell (L603-615, math.isfinite guard), so a NULL avg surfaces as None cleanly; no new NaN-leak risk."]

**设计决策点（提案已在补丁中给出，待审确认）**：
- approval_rate denominator semantics: should unknown/pending/NULL decisions be EXCLUDED from the denominator (approval_rate = approvals / decided, symmetric with bad_rate labeled-only, as the patch_sketch assumes) or KEPT (approval_rate = approvals / all-applications, where 'not approved yet' legitimately counts against the rate)? These give different numbers for partially-decided slices and the spec must pin one. bad_rate labeled-only is unambiguous; approval_rate is the genuine judgment call.
- Should unlabeled_count be auto-derived alongside every bad_rate/approval_rate metric (patch_sketch approach — keeps the op enum stable, always visible), or a separately-requestable metric op the LLM must opt into (adds 'unlabeled_count' to the manifest enum at L528-537 and _metric_expr)? Auto-derive guarantees the deflation is never silently hidden; opt-in keeps output narrower.
- Should the tool additionally emit an amber red_flag when any group has unlabeled_count > 0 (consistent with empty_result/truncated flags), or is the per-row column sufficient? A red_flag surfaces the caveat in the rendered summary text; the column alone requires the reader to notice a nonzero cell.
- Non-binary-but-non-null labels (e.g. a target coded 2 for 'indeterminate', or -1): the patch treats anything not in {0,1} as unlabeled (excluded + counted). Confirm the platform never uses a third label value that should count as good/bad — _finite_binary_pairs raises on non-{0,1} targets, so excluding is consistent, but slice_aggregate silently excludes rather than raising (correct for an exploratory tool, but worth confirming it should not hard-error like the feature path).

---

### A5 · join 空白键执行/诊断/pandas 三方不一致 → nullif 统一

**现状（bug 机理）**：In marvis/data/backend.py the executed LEFT JOIN and the match-rate diagnostics normalize join keys through DIFFERENT code paths, so blank/whitespace-only keys are treated inconsistently. The executed join builds its ON condition via _join_condition (L1140-1195) which for text keys emits _sql_transform (L1103-1138): trim(...) and (numeric-looking) strip trailing '.0', but does NOT convert empty-string/whitespace to NULL — so two rows with key '' or '   ' compare EQUAL in the join, producing matches. The diagnostics path (match-rate + uniqueness) uses _sql_normalized_key (L1043-1076) which wraps the same normalization in nullif(trim(...),'') so blank keys become NULL and are EXCLUDED. The pandas fallback _pandas_left_join (L1245-1310) via _normalize_key_series (L1210-1243) does .str.strip() then ''->NA (also excludes blank). THREE inconsistent behaviors: SQL join (blank matches blank), SQL diagnostics (blank=NULL excluded), pandas fallback (blank=NA excluded). Net: a blank-keyed anchor row silently joins to a blank-keyed feature row (wrong attach) while the gate's match-rate says those rows are unmatched.

**关键代码位置**：
```
marvis/data/backend.py:1103-1138 _sql_transform (executed-join normalizer, NO nullif):
    def _sql_transform(expr):
        return ("CASE WHEN regexp_matches(trim(CAST("+expr+" AS VARCHAR)), '^[0-9]+\\.0+$') "
                "THEN regexp_replace(trim(CAST("+expr+" AS VARCHAR)), '\\.0+$','') "
                "ELSE trim(CAST("+expr+" AS VARCHAR)) END")

marvis/data/backend.py:1043-1076 _sql_normalized_key (diagnostics, HAS nullif):
    def _sql_normalized_key(col):
        return "nullif("+_sql_transform(_quote_ident(col))+", '')"

marvis/data/backend.py:1210-1243 _normalize_key_series (pandas, empty->NA):
    s = series.astype('string').str.strip()
    s = s.mask(s.eq(''), pd.NA)

The executed join calls _join_condition -> _sql_transform per key WITHOUT the nullif wrapper, so blank keys survive as '' and match each other.
```

**受影响调用面**：
- marvis/data/backend.py:1140-1195 _join_condition — builds ON a.k=b.k via _sql_transform; must wrap each side in nullif(...,'') (blank->NULL, NULL never equals NULL in join)
- marvis/data/backend.py:1103-1138 _sql_transform — shared normalizer; decision: add nullif here (affects join AND diagnostics uniformly)
- marvis/data/backend.py:1043-1076 _sql_normalized_key — becomes redundant if nullif moves into _sql_transform
- marvis/data/backend.py:900-1041 uniqueness/dedup exprs — must stay consistent with join normalization
- marvis/data/backend.py:1245-1310 _pandas_left_join / _normalize_key_series — already excludes blank; reference behavior
- marvis/data/join_engine.py diagnose_join — match-rate improves automatically once join excludes blanks

**现有测试与缺口**：
- tests/test_data_backend.py — join + uniqueness tests; NO test feeds '' or '   ' keys
- tests/test_join_engine.py — fan-out/1:1 tests; no blank-key case
- NO test asserts a blank-key anchor row is LEFT UNMATCHED (feature cols NULL) rather than wrongly attached — core regression to add

**补丁方案（伪代码）**：
Preferred: move nullif into the shared normalizer so all three paths agree blank=missing. 1) _sql_transform: wrap final result in nullif(<result>,'') so blank/whitespace -> NULL everywhere (join ON + diagnostics). SQL NULL=NULL is false, so blank-keyed rows stop matching in LEFT JOIN and fall through to NULL feature columns — matching pandas .str.strip()+mask(''->NA) and the diagnostics. 2) _sql_normalized_key's own nullif becomes redundant (harmless double-nullif; simplify to just _sql_transform). 3) Verify _transformed_key_exprs (uniqueness/dedup) inherits the same _sql_transform so a blank key isn't counted as a duplicate value. 4) Confirm 1:1 row-count assertion still holds (blank anchor rows remain in LEFT result with NULL features — count preserved). Alternative (narrower): wrap only _join_condition sides in nullif — rejected, leaves _sql_transform callers inconsistent.

**新增失败形状测试**：
- anchor rows with key '' and '   '; feature has key '' -> those anchor rows have NULL feature columns (unmatched), NOT the blank feature row's values
- 1:1 preservation: blank-key anchor rows still present in output (row count == anchor count) after fix
- match-rate diagnostic and executed-join matched-count AGREE on a dataset containing blank keys (previously diverged)
- uniqueness: a feature table with two '' keys is NOT reported as a duplicate-key collision (blanks are missing) — lock chosen semantics
- pandas fallback vs SQL parity: same blank-key dataset yields identical joined result under both paths

**风险 / 口径变化**：口径变化: rows with blank keys that PREVIOUSLY matched (blank=blank) will now be unmatched (feature cols NULL) — correct behavior but changes outputs; in 口径 release notes. Row COUNT unchanged (left join keeps anchor rows), so 1:1 invariant and downstream row-count asserts unaffected — only feature values for blank-key rows change (wrong-attach -> NULL). Ensure no COALESCE/ifnull downstream turns NULL keys back into '' before compare. Confirm _sql_transform is used everywhere keys are compared (join, uniqueness, dedup, fingerprint). Very low perf impact.

**设计决策点（提案已在补丁中给出，待审确认）**：
- Blank/whitespace keys as MISSING (nullif -> never match, recommended, matches pandas+diagnostics) vs a legitimate joinable value (current SQL-join behavior)? Recommend MISSING; confirm no real dataset joins on empty-string as a real category.
- Does marvis/data/fingerprint.py also need the same nullif so a blank isn't fingerprinted as a matchable value? Parallel check in the same PR.
- Surface a count of blank-keyed (now-unmatched) rows so the user knows WHY match-rate < 100%? Recommend blank_key_row_count in diagnostics as a follow-up, not blocking.

---

### A6 · float64 长 ID 科学计数法静默不匹配 → 键规范化统一 + 精度红旗

**现状（bug 机理）**：JOIN key normalization has two independent code paths that disagree on float64-stored long-integer IDs, and both silently mis-match once the id reaches >=16 significant digits.

SQL path (DuckDB JOIN condition + every match-rate/dedup/conflict SQL): marvis/data/backend.py:1135 `_sql_value_text` wraps every key expression as `trim(CAST({expression} AS VARCHAR))`, then a CASE that strips a trailing `.0` ONLY when the text matches regex `^-?[0-9]+\.0+$`. When the underlying column is DOUBLE/FLOAT (what a float64-promoted id column becomes in the persisted parquet, or what read_csv_auto infers when a null forces float promotion), DuckDB renders large integral doubles in scientific notation. Verified empirically: `CAST(CAST(123456789012345678 AS DOUBLE) AS VARCHAR)` = `1.2345678901234568e+17`; `CAST(CAST(9999999999999999 AS DOUBLE) AS VARCHAR)` = `1e+16`. The regex does NOT match scientific notation, so the .0-strip CASE is a no-op and the SQL key stays `1.2345678901234568e+17`. The flip to scientific notation starts at magnitude >=1e16 (17-digit `1eN`, or 16-digit all-nines); below that (e.g. 5.0, 1000000000000000.0) the existing regex correctly strips to the bare integer -- which is why the bug is silent and the existing test test_left_join_exact_normalizes_integral_float_keys_like_diagnostics (backend tests L299, ids 5.0/6.0) passes.

Python path (pandas fallback for match-rate; dedup/uniqueness/conflict computed in transformed key space): backend.py:1079 `_value_text` renders a Real that is_integer() via `str(int(number))`, producing a precision-lost integer WITHOUT exponent: `str(int(float('123456789012345678')))` = `123456789012345680`.

So for a real 17-18 digit id 123456789012345678 stored as float64: true = ...678, SQL key = 1.2345678901234568e+17, Python key = 123456789012345680 -- all three differ. Effects: (a) a float-stored id JOINed against the string-stored ground-truth id never matches -> 100% silent no-match; left_join at backend.py:527-533 then raises `row loss (shrink)` (loud) OR, more insidiously, the pandas match-rate / align key-selection paths (which never trip the shrink guard) silently report match_rate=0 and the wrong key gets picked. (b) When BOTH sides are float-stored, the SQL path and Python path DISAGREE with each other (`...e+17` vs `...680`), so a uniqueness/dedup check computed in Python (backend.py:452 with_transformed_key_columns, join_engine.py:170-177 two_level_dedup on transformed keys) disagrees with what the DuckDB JOIN actually does -- the exact class of SQL-vs-Python divergence the surrounding code comments repeatedly promise cannot happen. Note the fingerprint itself is also corrupted: fingerprint.py:32 does non_null.astype(str) on a float64 series, yielding `1.2345678901234568e+17`, so even value_kind classification sees the mangled text.

No red flag is raised anywhere when precision loss is possible; JoinDiagnostics (contracts.py:120-137) has no field for it and join_engine.diagnose_join (join_engine.py:103-217) never checks id-column digit magnitude.

**关键代码位置**：
```
backend.py:1079-1086 (Python path):
```
def _value_text(value: Any) -> str:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if number.is_integer():
            return str(int(number))
    return str(value).strip()
```

backend.py:1135-1142 (SQL path -- the broken regex):
```
def _sql_value_text(expression: str) -> str:
    trimmed = f"trim(CAST({expression} AS VARCHAR))"
    return (
        "CASE "
        f"WHEN regexp_matches({trimmed}, '^-?[0-9]+\\.0+$') "
        f"THEN regexp_replace({trimmed}, '\\.0+$', '') "
        f"ELSE {trimmed} END"
    )
```

backend.py:1113-1116 + 958-969 (callers into _sql_value_text):
```
def _sql_transform(method, expression, *, side, pair) -> str:
    trimmed = _sql_value_text(expression)
    if method == "exact":
        return trimmed
...
def _join_condition(self, pair, anchor_columns, feature_columns) -> str:
    anchor_col = "a." + sql_identifier(pair.anchor_col, anchor_columns)
    feature_col = "b." + sql_identifier(pair.feature_col, feature_columns)
    return (f"{_sql_transform(pair.match_method, anchor_col, side='anchor', pair=pair)} = "
            f"{_sql_transform(pair.match_method, feature_col, side='feature', pair=pair)}")
```

backend.py:1145-1146 (_sql_normalized_key -- the match-rate/dedup SQL entrypoint, also routes through _sql_value_text):
```
def _sql_normalized_key(method, expression, *, fingerprint) -> str:
    text = f"nullif({_sql_value_text(expression)}, '')"
```

csv_ingest.py:22-27 (existing threshold vocabulary to reuse):
```
LONG_ID_DIGIT_THRESHOLD = 15
```
excel_ingest.py:22 also defines `LONG_ID_FLOAT_THRESHOLD = 1e15`.
```

**受影响调用面**：
- marvis/data/backend.py:1114 _sql_transform -> _sql_value_text (JOIN condition + dedup partition + transformed-key exprs)
- marvis/data/backend.py:1146 _sql_normalized_key -> _sql_value_text (all match-rate SQL: _duckdb_match_rate_for_method L777/L782, _duckdb_match_rates_for_methods L732/L740)
- marvis/data/backend.py:967-968 _join_condition (the actual LEFT JOIN ON clause used by left_join L503-506)
- marvis/data/backend.py:1048 _normalize_value -> _value_text (Python match-rate fallback, L610/L629)
- marvis/data/backend.py:1200 _transformed_key_value -> _value_text (with_transformed_key_columns L1006, used by dedup/uniqueness in transformed key space)
- marvis/data/backend.py:452-453 distinct_count / 464 is_key_unique (Python branch calls with_transformed_key_columns)
- marvis/data/join_engine.py:131 is_key_unique, :137 match_rate_for_method, :170-177 two_level_dedup on transformed keys, :204-217 JoinDiagnostics construction (where a red-flag field would surface)
- marvis/data/align.py:117-129 _resolve_by_data match_rates_for_methods (silent wrong-key selection when match_rate collapses to 0)
- marvis/data/registry.py:333-337 parquet upload copied verbatim (float64 id survives, no long-id guard) and :357-360 feather (pd.read_feather preserves float64) -- the ingest sources that reach the join still float64
- marvis/routers/data.py:352-359 long_id_columns already surfaced to the API response on CSV ingest (the existing red-flag surface to extend for join-time detection)
- marvis/data/contracts.py:120-137 JoinDiagnostics dataclass (add precision-loss red-flag field here)

**现有测试与缺口**：
- tests/test_data_backend.py:299 test_left_join_exact_normalizes_integral_float_keys_like_diagnostics -- ONLY existing float-int key test; uses tiny ids (5.0/6.0) that stay below the 1e16 scientific-notation threshold, so it passes today and does NOT cover the bug
- tests/test_data_backend.py:486 test_python_match_rate_date_fallback_uses_same_formats_as_duckdb_join -- pattern for asserting SQL/Python path agreement (date), the shape to copy for a float-id agreement test
- tests/test_join_engine.py:528/564/595 test_join_engine_*_uniqueness_and_dedup_use_transformed_key_space -- assert transformed-key-space consistency for exact_lower/date/hash but never for float-int coercion
- tests/test_perf_regressions.py:302 test_join_match_rate_smoke_bounded_queries_and_deterministic -- match-rate smoke, no long-id case
- GAP: no test feeds a >=16-digit float64-stored id through left_join, match_rate_for_method (both DuckDB and pandas branches), distinct_count/is_key_unique, or conflict_report
- GAP: no test asserts _sql_value_text and _value_text produce the SAME normalized string for an integral float
- GAP: no test asserts a precision-loss red flag is raised when an id column exceeds 15 digits

**补丁方案（伪代码）**：
1. Unify integral-float rendering in SQL (_sql_value_text, backend.py:1135). Replace the regexp-.0-strip CASE with a branch that handles scientific notation. Because a float-integer's precision is already lost at parse, the goal is to make SQL render the SAME rounded integer string Python produces. Concretely, when the expression's runtime value is an integral floating type, cast through an integer type before VARCHAR so no exponent appears:
```
def _sql_value_text(expression):
    v = f"CAST({expression} AS VARCHAR)"
    # integral floating value -> render as integer (no sci-notation, no trailing .0)
    # try_cast to HUGEINT covers up to 38 digits; guard isfinite to avoid inf/nan.
    return (
      "trim(CASE "
      f"WHEN typeof({expression}) IN ('DOUBLE','FLOAT','REAL') "
      f"     AND {expression} IS NOT NULL AND isfinite({expression}) "
      f"     AND {expression} = floor({expression}) "
      f"  THEN CAST(TRY_CAST({expression} AS HUGEINT) AS VARCHAR) "
      # keep the existing textual .0 strip for already-VARCHAR '123.0' shaped inputs
      f"WHEN regexp_matches({v}, '^-?[0-9]+\\.0+$') THEN regexp_replace({v}, '\\.0+$', '') "
      f"ELSE {v} END)"
    )
```
Verified: `CAST(TRY_CAST(CAST('123456789012345678' AS DOUBLE) AS HUGEINT) AS VARCHAR)` = `123456789012345680`, which equals Python `str(int(float('123456789012345678')))` -> the two paths now AGREE. (typeof() is only correct on a bare column ref; since _sql_value_text always receives a quoted identifier like `a."id"` or `b."id"`, typeof works. If a wrapped/derived expression is ever passed, fall back to the regex branch.)

2. Keep Python _value_text (backend.py:1079) as-is for the integral-float branch (it already yields str(int(number)) with no exponent), so after step 1 SQL == Python. Optionally extract a shared module-level helper `_normalize_integral_float_text(number) -> str` so both paths reference one definition and a future change can't desync them.

3. Red flag when precision loss is possible (>15 digits). Add a detector reusing LONG_ID_DIGIT_THRESHOLD (csv_ingest.py:27). In join_engine.diagnose_join (join_engine.py:103), after resolving key columns, for each anchor/feature key column check whether the stored dtype is floating AND sampled values have >=16 significant digits (or |value| >= 1e15, matching excel_ingest.LONG_ID_FLOAT_THRESHOLD). Surface via a new optional field on JoinDiagnostics (contracts.py:120), e.g. `precision_loss_columns: tuple[str, ...] = ()`, set when detected. This mirrors the existing csv-ingest long-id surface (routers/data.py:352-359) so the C2 gate can warn: `列 id 以 float 存储且超过 15 位，join 键可能已丢失精度，无法可靠匹配 -- 请以字符串重导入 (see item B8)`. The red flag is REPORTED, never silently swallowed.

4. Scope boundary vs B8: this item makes the two JOIN key paths agree AND warns; it does NOT fix the root cause (id already float64 in the persisted parquet -- true digits unrecoverable). The full dtype-consistency fix at ingest (parquet/feather uploads copied verbatim at registry.py:333-360; sniff_long_id_columns misses beyond row 2000 / <90% coverage) is item B8. Cross-reference B8 in the red-flag message.

**新增失败形状测试**：
- test_sql_and_python_value_text_agree_on_long_integral_float: for id 123456789012345678 stored as float64, assert the DuckDB _sql_value_text expression evaluated in duckdb equals _value_text(float(...)) -- both '123456789012345680'; parametrize across 16/17/18/19 digits and all-nines boundary (9999999999999999)
- test_left_join_matches_long_float_ids_both_sides_float: anchor+feature parquet both with float64 id column of 18-digit ids, exact method; assert joined_rows == anchor_rows and feature values land (today: silent no-match -> shrink error)
- test_match_rate_for_method_long_float_ids_duckdb_and_python_agree: run match_rate_for_method with a feature CSV (DuckDB all_varchar path) vs a feature feather (Python fallback path) for the same 18-digit ids; assert both return full match (today they disagree and both under-count)
- test_distinct_count_and_is_key_unique_long_float_ids: float64 id column where two ids round to the SAME float (e.g. ...678 and ...679 both -> ...680); assert distinct_count reflects the collision consistently in SQL and Python branches
- test_diagnose_join_flags_precision_loss_for_over_15_digit_float_id: build anchor/feature where the key column is float64 with 18-digit ids; assert JoinDiagnostics.precision_loss_columns contains the key column and a short-digit float id (5.0) does NOT flag
- test_value_text_below_threshold_unchanged: ids like 5.0, 1000000000000000.0 (15 digits) still normalize to '5' / '1000000000000000' and do NOT raise the red flag (regression guard so the existing L299 test semantics hold)

**风险 / 口径变化**：口径变化 (metric-basis change): after the fix, match_rate / matched_rows / distinct_count / is_key_unique for any dataset that currently has a float64-stored long-id key will CHANGE (typically from ~0 to correct, or dedup counts shift) -- intended correction but it moves numbers that may live in prior JoinDiagnostics/plans. Blast radius touches every consumer of _sql_value_text (JOIN condition, dedup partition, transformed-key exprs, all match-rate SQL) and _value_text (Python match-rate, uniqueness, conflict, feature_derive) -- broad but all within the normalize-key concern. TRY_CAST to HUGEINT caps at 38 digits (fine for ids); out-of-range TRY_CAST returns NULL, so guard with isfinite and fall through to the textual branch. The typeof()-based branch assumes _sql_value_text receives a bare column identifier (true for all current callers via sql_identifier) -- a future caller passing a computed sub-expression would re-evaluate it twice; acceptable for column refs, flag at review. Critical non-fix: precision is already lost at float parse, so the fix makes float-vs-float and SQL-vs-Python CONSISTENT and warns, but a float-stored id still never matches a string-stored ground-truth id -- only item B8 (dtype-consistency at ingest) restores true values, so the red flag must direct users to re-import as string. Backward-compat: the new JoinDiagnostics field defaults to () and the new CASE branch is additive; sub-1e16 behavior is preserved (regression test #6).

**设计决策点（提案已在补丁中给出，待审确认）**：
- Does the SQL fix need to also cover the case where a key expression passed to _sql_value_text is NOT a bare column (e.g. a wrapped hash/date sub-expression)? Current callers only pass column identifiers before applying method transforms, so typeof() is safe, but confirm no future path double-wraps.
- Preferred red-flag surface: extend JoinDiagnostics with precision_loss_columns (cleanest, matches diagnose_join flow) vs. reuse the csv-ingest long_id_columns channel in routers/data.py -- the latter only fires at CSV ingest, not for parquet/feather uploads that reach the join, so a join-time detector in diagnose_join is needed regardless.
- Should detection key off the stored dtype being floating (definitive precision-loss risk) or also off digit magnitude of string-stored ids (>15 digits but safe as strings)? Recommend: flag only when dtype is floating AND magnitude >= 1e15, to avoid false alarms on correctly string-stored 18-digit id cards (value_kind == raw_idcard).
- Whether to align the red-flag threshold constant across csv_ingest (LONG_ID_DIGIT_THRESHOLD=15) and excel_ingest (LONG_ID_FLOAT_THRESHOLD=1e15) by introducing one shared constant, or leave both and reference the digit form in the join detector.

---

### B7 · 匹配率诊断 all_varchar vs 执行 typed 读取分裂 → 读取器统一

**现状（bug 机理）**：The match-rate diagnostics and the executed join read CSV features through TWO DIFFERENT DuckDB readers, so their key typing can disagree.

- Execution (`left_join`, marvis/data/backend.py:466-534) reads BOTH sides via `_duckdb_rel` (backend.py:818-824), which for CSV is `csv_rel` = `read_csv_auto(<path>)` — DuckDB type-sniffs each column. Anchor at L516 `FROM {self._duckdb_rel(anchor_path)}`; feature via `_dedup_feature_rel` (L855-885) whose `rel = self._duckdb_rel(feature_path)` at L864.
- Match-rate diagnostics: the FEATURE side is read via `_duckdb_text_rel` (backend.py:826-832) = `read_csv_auto(<path>, all_varchar=true)` — every column forced to VARCHAR — at exactly two sites: `_duckdb_match_rate_for_method` (L794) and `_duckdb_match_rates_for_methods` (L725). The ANCHOR side of these helpers is NOT a DuckDB scan at all: it is a registered pandas DataFrame (`conn.register("anchor_sample", anchor_frame)` L754/L810) produced by `sample_rows` (L389-411) → `read_frame` (L357-387) → for CSV `read_csv_with_fallback_encoding` (marvis/data/csv_ingest.py:72-101). That pandas read preserves a zero-padded/leading-zero id column as string ONLY when it is >=15 digits (LONG_ID_DIGIT_THRESHOLD, csv_ingest.py:27,36-69); a short zero-padded id (e.g. "007") is promoted to int and loses the padding.

Net asymmetry: the feature key is always VARCHAR (padding preserved: "007"), the anchor key is typed-by-pandas (padding possibly stripped to 7), and at EXECUTION time both sides are typed-by-DuckDB (`read_csv_auto` typed, both become 7). The SQL normalizer `_sql_value_text` (backend.py:1135-1142) only strips a trailing `.0` from float-like ints (5.0 -> "5"); it does NOT re-pad or strip leading zeros. So a zero-padded id can show one match rate under the mixed-reader diagnostic and a different actual join result under the uniformly-typed execution — the spec's "high match rate then join to nothing" (and the mirror: low diagnostic rate but execution matches). All OTHER diagnostic consumers already use the typed `_duckdb_rel` and thus already agree with execution: row_count (L201), numeric_columns (L220/L349/column_names), conflict_report (L260,L297,L327), distinct_count (L446). Only the two match-rate feature scans diverge.

IMPORTANT scope: `_resolve_path`+registry — in the normal product flow paths come from `registry.resolve_path` (marvis/data/registry.py:313-315) which returns the normalized PARQUET product (`_normalize_to_parquet`, registry.py:319-356 converts every CSV/xlsx upload to parquet). For a `.parquet` path BOTH `_duckdb_rel` and `_duckdb_text_rel` return the identical `parquet_rel` (backend.py:822-823 vs 830-831), so there is NO divergence in the product flow. The bug is a latent contract split on the raw-CSV public seam: `backend.left_join`/`match_rate_for_method`/`match_rates_for_methods` accept `.csv` paths directly (SUPPORTED_DUCKDB_SUFFIXES includes ".csv", backend.py:26) and are called with raw CSVs in tests (tests/test_data_backend.py) and by any caller that skips the registry.

**关键代码位置**：
```
backend.py:818-832 (the two readers — the split):
```
818	    def _duckdb_rel(self, path: Path) -> str:
819	        suffix = path.suffix.lower()
820	        if suffix == ".csv":
821	            return csv_rel(path)          # read_csv_auto(path)  -- TYPED
822	        if suffix == ".parquet":
823	            return parquet_rel(path)
824	        raise DataBackendError(f"unsupported DuckDB dataset format: {path.suffix}")
826	    def _duckdb_text_rel(self, path: Path) -> str:
827	        suffix = path.suffix.lower()
828	        if suffix == ".csv":
829	            return f"read_csv_auto({sql_string_literal(path.as_posix())}, all_varchar=true)"  # VARCHAR
830	        if suffix == ".parquet":
831	            return parquet_rel(path)       # identical to _duckdb_rel for parquet
832	        raise DataBackendError(f"unsupported DuckDB dataset format: {path.suffix}")
```

backend.py:794-808 (`_duckdb_match_rate_for_method` — feature via text_rel, anchor via registered pandas frame):
```
794	        feature_rel = self._duckdb_text_rel(feature_path)
795	        query = (
796	            "WITH anchor_keys AS ("
797	            f"SELECT {anchor_projection} FROM anchor_sample a"
798	            "), feature_keys AS ("
799	            f"SELECT DISTINCT {feature_projection} FROM {feature_rel} b"
...
809	        with self._connect() as conn:
810	            conn.register("anchor_sample", anchor_frame)
811	            matched = conn.execute(query).fetchone()[0]
```

backend.py:725,736 (`_duckdb_match_rates_for_methods` — same text_rel feature scan):
```
725	        feature_rel = self._duckdb_text_rel(feature_path)
...
736	        ctes = [f"feature_keys AS (SELECT {feature_exprs} FROM {feature_rel} b)"]
```

backend.py:513-517 + 864 (execution reads both sides TYPED):
```
513	        query = (
514	            "COPY ("
515	            f"SELECT a.*{feature_select} "
516	            f"FROM {self._duckdb_rel(anchor_path)} a "
517	            f"LEFT JOIN ({feature_rel}) b ON {on_sql}"
...
864	        rel = self._duckdb_rel(feature_path)   # inside _dedup_feature_rel, the feature side of the join
```

backend.py:590 (anchor sample is a typed pandas read, NOT all_varchar):
```
590	        anchor_frame = self.sample_rows(anchor_path, sample_n, seed=seed)
```

csv_ingest.py:27,36-69,90-92 (why anchor typing is not all_varchar and only >=15-digit ids stay string):
```
27	LONG_ID_DIGIT_THRESHOLD = 15
...
90	            long_id_columns = sniff_long_id_columns(path, encoding=encoding)
91	            dtype_overrides = {column: str for column in long_id_columns} or None
92	            frame = pd.read_csv(path, encoding=encoding, dtype=dtype_overrides, **read_csv_kwargs)
```

backend.py:1135-1142 (the shared SQL normalizer does not touch leading zeros):
```
1135	def _sql_value_text(expression: str) -> str:
1136	    trimmed = f"trim(CAST({expression} AS VARCHAR))"
1137	    return ("CASE WHEN regexp_matches(...'^-?[0-9]+\\.0+$') THEN regexp_replace(...) ELSE {trimmed} END")
```
```

**受影响调用面**：
- marvis/data/backend.py:794 — _duckdb_match_rate_for_method feature scan via _duckdb_text_rel (CHANGE target)
- marvis/data/backend.py:725 — _duckdb_match_rates_for_methods feature scan via _duckdb_text_rel (CHANGE target)
- marvis/data/backend.py:826-832 — _duckdb_text_rel definition (only two callers; candidate for deletion after unification)
- marvis/data/backend.py:516 — left_join anchor scan via _duckdb_rel (execution reference reader)
- marvis/data/backend.py:864 — _dedup_feature_rel feature scan via _duckdb_rel (execution reference reader)
- marvis/data/backend.py:590 + 754/810 — anchor_sample = sample_rows(...) registered as pandas frame; the anchor side of every match-rate query (typed pandas, not all_varchar)
- marvis/data/align.py:117 — Aligner._resolve_by_data calls match_rates_for_methods (empirical key discovery; consumes the diagnostic rate)
- marvis/data/join_engine.py:137,241 — JoinEngine.diagnose_join / _relaxation_alternatives call match_rate_for_method (multi-key and relaxation match rates)
- marvis/data/join_engine.py:131,255 — is_key_unique (already _duckdb_rel typed — no change)
- marvis/data/join_engine.py:154,259 — distinct_count (already _duckdb_rel typed — no change)
- marvis/data/join_engine.py:180 — conflict_report (already _duckdb_rel typed — no change)
- marvis/data/join_engine.py:355 — left_join execution (already _duckdb_rel typed)
- marvis/data/join_engine.py:64,68,351 + marvis/data/registry.py:313-315 — resolve_path returns the normalized PARQUET product, so the product flow never hits the CSV divergence; the seam is raw-CSV callers
- marvis/data/registry.py:319-356 — _normalize_to_parquet converts every CSV/xlsx upload to parquet on ingest (why product flow is parquet-only)
- marvis/packs/data_ops/tools.py:413 — separate top_k analytics use of _duckdb_rel (unaffected by this fix; noted only because it also touches _duckdb_rel)

**现有测试与缺口**：
- tests/test_data_backend.py:299 test_left_join_exact_normalizes_integral_float_keys_like_diagnostics — asserts execution normalizes 5.0-vs-"5" like diagnostics, but uses PARQUET inputs so it does NOT exercise the CSV reader split
- tests/test_data_backend.py:329 test_match_rate_pushes_feature_key_scan_to_duckdb — CSV inputs, values A1/B2 (no leading zeros), asserts (2,3) and that the feature scan stays in DuckDB (guards against reverting to read_frame)
- tests/test_data_backend.py:360 test_match_rate_duckdb_path_resolves_relative_dataset_paths — CSV inputs, relative paths, exact_lower, (2,3)
- tests/test_data_backend.py:380 test_match_rate_falls_back_for_hash_methods_not_supported_by_duckdb — CSV inputs, sha1 forces the Python fallback (not the DuckDB text_rel path)
- tests/test_data_backend.py:411 test_match_rate_normalizes_hash_case_and_dates — CSV inputs, md5/sha256/date normalization
- tests/test_data_backend.py:77,162,262 left_join preservation / shrink-guard / collision-alias tests — establish the 1:1 execution invariant
- GAP: NO test pairs a zero-padded/leading-zero CSV key (e.g. anchor "007" typed to int 7 by pandas vs feature "007" kept VARCHAR by all_varchar) and asserts the diagnostic match rate equals the actual left_join match — this is exactly the T1-B7 failure and is currently uncovered
- GAP: NO test asserts diagnostics and execution use the SAME reader for a CSV feature (nothing pins the two paths together for CSV; the one that does — L299 — is parquet-only)
- tests/test_join_engine.py and tests/test_feature_derive.py reference these methods but (per grep) do not cover the zero-padded CSV reader-split case

**补丁方案（伪代码）**：
Direction: make DIAGNOSTICS adopt the EXECUTION reader so both sides of every comparison use identical typing and identical key transforms. Two viable shapes; recommend Option A.

Option A (minimal, recommended) — unify the reader:
1. In `_duckdb_match_rate_for_method` (backend.py:794) and `_duckdb_match_rates_for_methods` (backend.py:725), replace `feature_rel = self._duckdb_text_rel(feature_path)` with `feature_rel = self._duckdb_rel(feature_path)` so the feature scan is TYPED exactly like `left_join`/`_dedup_feature_rel`.
2. Also make the ANCHOR side typed the same way. Today the anchor is a pandas frame from `sample_rows` whose CSV typing (via read_csv_with_fallback_encoding) can differ from DuckDB's `read_csv_auto`. For the DuckDB match-rate helpers, prefer sampling the anchor INSIDE DuckDB from `self._duckdb_rel(anchor_path)` (mirror the existing reservoir-sample SQL at backend.py:404-408) instead of registering a pandas frame — so anchor and feature are both `read_csv_auto`-typed and match `left_join` bit-for-bit. If that is too invasive, at minimum register the anchor sample but CAST/normalize it through the same `_sql_value_text` path (already applied via `_sql_normalized_key`), and add a regression test proving parity with execution for zero-padded keys.
3. Delete `_duckdb_text_rel` (backend.py:826-832) once its only two callers are converted, OR keep it and document it is unused, to avoid a future accidental reintroduction of the split. Deletion is cleaner.
4. Because `read_csv_auto` type-sniffs, a wide/heterogeneous feature column that previously forced all_varchar could now be sniffed to a numeric/date type — but `_sql_normalized_key`/`_sql_value_text` already `CAST(... AS VARCHAR)` before normalizing, so downstream key comparison is unaffected; the only behavior change is the TYPING of the raw scan, which is the intended fix (it now equals execution). Confirm DuckDB does not error on the wide-payload columns from test_data_backend.py:333 ("x"*1000) — read_csv_auto handles long VARCHAR fine; the all_varchar flag was not required for correctness there.

Option B (heavier) — route diagnostics and execution through ONE shared private reader helper (e.g. `_join_scan_rel(path)`) that both `left_join`/`_dedup_feature_rel` and the two match-rate helpers call, guaranteeing they can never drift again. Same runtime behavior as Option A; better long-term coupling; larger diff.

Regardless of option: keep the key-transform path identical. Note the diagnostic normalizer is `_sql_normalized_key` (backend.py:1145-1164) while execution uses `_sql_transform` (backend.py:1113-1132) — these are already intended to agree, but the spec asks to verify "same key transform"; if any divergence exists between `_sql_normalized_key` and `_sql_transform` (e.g. the `nullif(...,'')` empty-string handling at L1146 that `_sql_transform` lacks), it must be reconciled or explicitly justified as diagnostic-only. Flag in open_questions.

**新增失败形状测试**：
- test_match_rate_matches_left_join_for_zero_padded_csv_keys: write anchor.csv and feature.csv where the join key is a SHORT zero-padded id below the 15-digit long-id threshold (e.g. anchor ['007','012'] , feature ['007','012','099']); call match_rate_for_method(method='exact') AND left_join with the same KeyPair; assert the diagnostic matched-count and the actual joined non-null-feature count agree (the value the diagnostic predicts is realized by execution). This is the direct T1-B7 regression.
- test_match_rate_feature_scan_uses_typed_reader_like_execution: for a CSV feature, monkeypatch/assert that _duckdb_text_rel is no longer used by the match-rate helpers (or that _duckdb_rel is the reader), pinning diagnostics and execution to the same reader so the split cannot silently return.
- test_match_rates_for_methods_zero_padded_parity: batched-path (align._resolve_by_data shape) counterpart — one anchor/feature column, methods=[exact, exact_lower], zero-padded CSV keys; assert each method's diagnostic rate equals the realized left_join match under that method.
- test_match_rate_wide_varchar_feature_still_works_under_typed_reader: reuse the wide 'x'*1000 payload from test_data_backend.py:333 but with the typed reader, asserting the switch away from all_varchar does not break wide-column CSV feature scans (guards the one concrete reason all_varchar was introduced).
- test_diagnostic_and_execution_agree_on_leading_zero_via_join_engine: integration at JoinEngine.diagnose_join level — feed a CSV feature (or, if registry-normalized, document that parquet neutralizes it) with a leading-zero key and assert diagnostics.match_rate is consistent with the executed left_join row/non-null counts.

**风险 / 口径变化**：Blast radius is narrow: `_duckdb_text_rel` has exactly two callers (backend.py:725, 794), both feature-side scans in the DuckDB match-rate helpers; every other diagnostic already uses the typed `_duckdb_rel`, so unifying to `_duckdb_rel` moves match-rate INTO agreement with the rest of the pipeline and with execution.

Metric-basis change (口径变化): the empirically-computed key-match RATE for RAW-CSV features can change after this fix — specifically for datasets with typing-sensitive keys (leading zeros, integer-vs-string ids). The new rate is the CORRECT one (it equals what the executed join realizes), but any snapshot/golden that recorded the old mixed-reader rate for a CSV feature must be re-baselined. This feeds `align._resolve_by_data` (align.py:117) key selection and `join_engine.diagnose_join` gate payloads, so a key that previously passed/failed MIN_KEY_MATCH_RATE could now flip — this is the intended correction but is a behavior change at the C2 gate.

Product-flow safety: because `registry.resolve_path` returns the normalized PARQUET product and both readers are identical for parquet, the LIVE product path sees NO change — the fix only affects raw-CSV callers (tests, and any future/public caller that passes .csv directly). This means the fix is low-risk in production but its regression tests MUST use raw .csv inputs (parquet inputs will not exercise the bug — see the already-parquet test at test_data_backend.py:299).

Backward-compat: switching the feature scan from all_varchar to read_csv_auto changes the raw column TYPES DuckDB infers; downstream key normalization already CASTs to VARCHAR before comparing (_sql_value_text L1136, _sql_normalized_key L1145), so key semantics are preserved. The only latent risk is a pathological CSV where read_csv_auto's type sniff errors or mis-parses a column that all_varchar tolerated; verify against the wide-payload test. If `_duckdb_text_rel` is deleted, confirm no test imports/patches it (grep shows tests reference match_rate/left_join, and test_data_backend.py:397 patches `_duckdb_match_rate_for_method`, not the reader).

Determinism (INV-1): unchanged — both readers are deterministic; the memoized cache keys already include file identity and method/fingerprint tuples, so switching readers does not require cache-key changes.

**设计决策点（提案已在补丁中给出，待审确认）**：
- Unification direction: the spec suggests diagnostics adopt the execution reader (typed _duckdb_rel). Confirm this is desired vs the opposite (make execution all_varchar) — typed is recommended because execution's 1:1 invariant and all other diagnostics already rely on typed reads, and all_varchar on the join would change join output column types.
- Anchor-side typing: the anchor in the DuckDB match-rate helpers is a REGISTERED PANDAS FRAME (sample_rows -> read_csv_with_fallback_encoding), not a DuckDB read_csv_auto scan. Even after switching the FEATURE reader to _duckdb_rel, the anchor could still be typed differently (pandas vs DuckDB) for a zero-padded key <15 digits. Should the fix also sample the anchor inside DuckDB from _duckdb_rel(anchor_path) (mirroring backend.py:404-408) so anchor+feature+execution are all read_csv_auto-typed? Recommended yes; needs a decision because it enlarges the change.
- Normalizer parity: execution uses _sql_transform (L1113) via _join_condition; diagnostics use _sql_normalized_key (L1145). They are meant to agree but differ in at least one place — _sql_normalized_key wraps in nullif(...,'') (L1146) and _sql_transform does not. Verify empty-string / whitespace-only keys are treated identically by both, or the reader unification alone will not fully close the diagnostics-vs-execution gap. This may be a second latent split worth reconciling in the same spec.
- Was all_varchar=true required for any real feature CSV (e.g. a column DuckDB's read_csv_auto mis-sniffs or errors on)? The introducing commit (0b54507b) gives no rationale in the diff. If some production CSV relied on all_varchar to avoid a sniff error, switching to read_csv_auto could regress it; the wide-payload test suggests all_varchar was not strictly needed, but confirm against any real-data fixtures.
- Should _duckdb_text_rel be deleted or retained-but-unused after unification? Deletion prevents future accidental reintroduction of the split; retention is safer if any out-of-tree caller exists (none found in-repo).

---

### B8 · join 键跨文件 dtype 不一致静默 miss → 一致性检查 + 红旗

**现状（bug 机理）**：The CSV long-id guard decides per-file, independently, whether each column is stored as string (dtype=object) or left to pandas' default type inference. `sniff_long_id_columns` (marvis/data/csv_ingest.py:36-69) samples up to 2000 rows with dtype=str, flags a column only when `long_digit_like.sum() / max(digit_like.sum(),1) >= 0.9 and long_digit_like.sum() > 0` where "long" means `>= LONG_ID_DIGIT_THRESHOLD (15)` digits (csv_ingest.py:27,61-67); `read_csv_with_fallback_encoding` (csv_ingest.py:72-101) then applies `dtype_overrides = {col: str for col in long_id_columns} or None`. Each upload runs this in isolation (registry.py:_write_upload_as_parquet, lines 338-356: `frame, report = read_csv_with_fallback_encoding(source_path)` then `frame.to_parquet(...)`), so the SAME logical join key can end up dtype=object (string) in one file and float64/int64 in another. The stored parquet dtype flows into `ColumnProfile.dtype = str(series.dtype)` (schema_infer.py:96 via infer_column_profile) and into the fingerprint via `fingerprint_column` (fingerprint.py:20-75), which does `non_null.astype(str)` — so a float64 id `1.1010119900101e17` fingerprints as value_kind "numeric" while the string side `"110101199001011234"` fingerprints as "raw_idcard".

Two concrete silent-miss mechanisms result: (1) FINGERPRINT DIVERGENCE — `candidate_match_methods(numeric, raw_idcard)` (fingerprint.py:78-98) falls through every branch and returns `[]`, so `_resolve_by_data` (align.py:98-141) proposes NO KeyPair for that column and the aligner may emit an empty/wrong key set — the feature silently contributes all-null columns or the join is proposed on a weaker key. (2) PRECISION LOSS + PARTIAL MATCH — even when both sides tokenize to VARCHAR at join time (backend.py:_sql_value_text lines 1135-1142 casts to VARCHAR and strips a trailing `\.0+$`, so `12345.0`→`12345`), any id that already lost trailing digits under float64 promotion (>=16 significant digits) is permanently corrupted before the cast, so those rows miss. If, say, 60% of keys are short enough to survive and 40% are corrupted, the sampled `match_rate` can land ~0.6, above `MIN_KEY_MATCH_RATE = 0.5` (contracts.py:9), so the KeyPair passes the gate and 40% of rows silently fail to match with no warning. The aligner and diagnostics never compare the two sides' dtypes; `diagnose_join` (join_engine.py:103-217) surfaces match_rate/uniqueness/fan-out/conflict but has no dtype-consistency signal, and `_render_propose_join` (renderers.py:1228-1320) has no dtype red flag. The guard's rigid `>=15 digits AND >=90% share` also entirely misses zero-padded SHORT codes (e.g. 6-digit org codes like "000123") where the leading-zero loss under int/float promotion is the failure, not float64 mantissa truncation.

**关键代码位置**：
```
marvis/data/csv_ingest.py:36-69 (sniff_long_id_columns — the per-file guard):
```
def sniff_long_id_columns(path, *, encoding, sample_rows=2000) -> tuple[str, ...]:
    try:
        sample = pd.read_csv(path, encoding=encoding, dtype=str, nrows=sample_rows, keep_default_na=True)
    except (UnicodeDecodeError, pd.errors.ParserError, csv.Error):
        return ()
    flagged: list[str] = []
    for column in sample.columns:
        values = sample[column].dropna()
        if values.empty:
            continue
        digit_like = values.str.fullmatch(r"\d+")
        long_digit_like = digit_like & (values.str.len() >= LONG_ID_DIGIT_THRESHOLD)
        if digit_like.sum() == 0:
            continue
        if long_digit_like.sum() / max(int(digit_like.sum()), 1) >= 0.9 and long_digit_like.sum() > 0:
            flagged.append(str(column))
    return tuple(flagged)
```

marvis/data/align.py:98-141 (_resolve_by_data — where key pairs are built; dtypes are in scope as anchor_col.dtype / feature_col.dtype but never compared):
```
def _resolve_by_data(self, anchor_col, feature_cols, anchor_path, feature_path, seed, *, resolved_by):
    best = None
    for feature_col in feature_cols:
        methods = candidate_match_methods(anchor_col.fingerprint, feature_col.fingerprint)
        if not methods:
            continue
        ...
        for method, (matched, sampled) in zip(methods, rates):
            rate = matched / sampled if sampled else 0.0
            if rate < MIN_KEY_MATCH_RATE:
                continue
            candidate = KeyPair(anchor_col=anchor_col.name, feature_col=feature_col.name, match_method=method,
                                transform_side=_raw_side(anchor_col, feature_col, method),
                                match_rate=round(rate, 4), resolved_by=resolved_by)
            if best is None or candidate.match_rate > best.match_rate:
                best = candidate
    return best
```

marvis/data/fingerprint.py:78-98 (candidate_match_methods — returns [] for numeric-vs-raw_idcard → silent no-pair):
```
def candidate_match_methods(a, b):
    if a.value_kind == b.value_kind and a.value_kind in {"raw_phone", "raw_idcard"}:
        return ["exact", "exact_lower"]
    if a.value_kind == "hash" and b.value_kind == "hash":
        return ["exact_lower"] if a.length_mode == b.length_mode else []
    kinds = {a.value_kind, b.value_kind}
    if "hash" in kinds and ("raw_phone" in kinds or "raw_idcard" in kinds): ...
    if a.value_kind == "date" and b.value_kind == "date":
        return ["date"]
    if a.value_kind == b.value_kind:
        return ["exact", "exact_lower"]
    return []
```

marvis/data/contracts.py:74-82 (KeyPair — carries no dtype) and 120-137 (JoinDiagnostics — the natural carrier for a dtype-divergence flag; currently has conflict_report and key_alternatives but no dtype signal).

marvis/data/schema_infer.py:92-100 (dtype source): `ColumnProfile(... dtype=str(series.dtype) ...)`.

marvis/packs/data_ops/tools.py:679-720 (payload serialization — _key_pair_payload omits dtype; _diagnostics_payload = asdict(diagnostics); _join_plan_payload assembles joins[].key_pairs/diagnostics).

marvis/agent/renderers.py:1264-1320 (_render_propose_join — builds the 指纹（raw=md5?） cell and inline ⚠️ warnings but has no dtype red flag).
```

**受影响调用面**：
- marvis/data/csv_ingest.py:36 sniff_long_id_columns — the guard to relax for zero-padded short codes / threshold
- marvis/data/csv_ingest.py:90-91 read_csv_with_fallback_encoding — applies dtype_overrides per-file
- marvis/data/registry.py:339-355 _write_upload_as_parquet — per-file CSV read that establishes each file's stored parquet dtype independently
- marvis/data/profiler.py:19-20 profile_dataset → infer_dataset_schema — where ColumnProfile.dtype is set from the parquet sample
- marvis/data/schema_infer.py:96 infer_column_profile — dtype=str(series.dtype)
- marvis/data/align.py:98-141 ColumnAligner._resolve_by_data — has anchor_col/feature_col ColumnProfile (dtype in scope) when building KeyPair; natural place to compute divergence
- marvis/data/align.py:70-96 ColumnAligner.align — returns list[KeyPair]; would need to thread dtype info out (via KeyPair field or a side-channel)
- marvis/data/join_engine.py:103-217 JoinEngine.diagnose_join — assembles JoinDiagnostics gate payload; where a dtype-divergence flag would be attached
- marvis/data/join_engine.py:69-83 propose_join_plan — calls aligner.align then diagnose_join per feature
- marvis/data/contracts.py:74-82 KeyPair / 120-137 JoinDiagnostics — dataclasses to extend with dtype/divergence fields
- marvis/packs/data_ops/tools.py:679-687 _key_pair_payload — serializes KeyPair (add dtype fields here)
- marvis/packs/data_ops/tools.py:690-691 _diagnostics_payload=asdict(diagnostics) — auto-carries any new JoinDiagnostics field
- marvis/packs/data_ops/tools.py:705-720 _join_plan_payload — joins[].key_pairs/diagnostics assembly
- marvis/packs/data_ops/tools.py:119-136 tool_propose_join — final payload the renderer consumes
- marvis/agent/renderers.py:1228-1320 _render_propose_join — surface the red flag / forced-confirmation text + table cell
- marvis/agent/gate_payloads.py:24-70 build_dedup_payload / gate wiring — reads propose joins[].diagnostics; forced-confirmation gate for dtype mismatch would mirror the dedup gate
- marvis/agent/gates/contracts.py:292 propose_join gate controls tuple — where an irreversible/forced-confirm control id is declared

**现有测试与缺口**：
- tests/test_data_repository_registry.py:765 test_dataset_registry_ingests_gbk_encoded_csv_with_long_id_column — 18-digit id + null → kept as string; asserts report.long_id_columns contains 'id_card' (single-file only)
- tests/test_data_repository_registry.py:802 test_dataset_registry_keeps_utf8_csv_encoding_and_skips_short_ids — 11-digit mobile → NOT flagged (long_id_columns == ())
- tests/test_data_repository_registry.py:824 test_dataset_registry_reads_scientific_notation_ids_as_strings — 19-digit card + null → kept as string (single-file)
- tests/test_align.py:34-133 — align resolves phone→md5, sha256 fallback, hash-case/date-format normalization, rejects name-only, fuzzy fallback; NONE feed a dtype-divergent key pair (string one side, float64 the other)
- GAP: no test where the SAME logical key is dtype=object in file A and float64 in file B and asserts a divergence red flag / forced confirmation
- GAP: no test that a partially-corrupted float64 id column yields match_rate just above MIN_KEY_MATCH_RATE (0.5) yet is flagged as dtype-divergent rather than silently accepted
- GAP: no test for candidate_match_methods(numeric, raw_idcard) == [] causing a silently-dropped key pair
- GAP: no test for zero-padded SHORT code (e.g. '000123') losing leading zeros under int/float promotion (guard's >=15-digit rule ignores it)
- GAP: no renderer test asserting _render_propose_join surfaces a dtype-mismatch red flag row/text
- GAP: no gate test that a dtype-divergent proposed key forces confirmation before execute_join

**补丁方案（伪代码）**：
Three coordinated layers; keep INV-1 (presentation only where possible) and never auto-swap/auto-drop a key.

1) DETECTION at key-pair build time (marvis/data/align.py). In `_resolve_by_data`, when a candidate KeyPair is selected, compute a coarse dtype-family for each side from ColumnProfile.dtype and record divergence. Add helper `_dtype_family(profile) -> str`:
```
def _dtype_family(profile) -> str:  # "text" | "float" | "int" | "date" | "other"
    dt = str(profile.dtype).lower()
    if dt in {"object", "string"} or dt.startswith("string"): return "text"
    if "float" in dt: return "float"
    if "int" in dt: return "int"
    if "datetime" in dt or dt.startswith("date"): return "date"
    return "other"
```
Extend contracts.KeyPair with `anchor_dtype: str = ""`, `feature_dtype: str = ""`, `dtype_divergent: bool = False` (defaulted so existing constructors/tests keep working). In `_resolve_by_data` set them on the candidate: `dtype_divergent = _dtype_family(anchor_col) != _dtype_family(feature_col)` and the raw dtype strings. Divergence that couples text↔float is the dangerous precision-loss case; treat text↔int and int↔float as WARN (lossless at VARCHAR-cast time but still worth surfacing), text↔float as RED.

2) GATE aggregation (marvis/data/join_engine.py::diagnose_join). Extend contracts.JoinDiagnostics with `key_dtype_divergences: tuple[KeyDtypeDivergence, ...] = ()` (new frozen dataclass: anchor_col, feature_col, anchor_dtype, feature_dtype, level ["red"|"warn"]). Build it from `key_pairs`:
```
divergences = tuple(
    KeyDtypeDivergence(p.anchor_col, p.feature_col, p.anchor_dtype, p.feature_dtype,
                       level=_divergence_level(p.anchor_dtype, p.feature_dtype))
    for p in key_pairs if p.dtype_divergent)
```
Return it in JoinDiagnostics(...). No new口径 for match_rate (INV-1); this is a metadata flag derived from already-computed dtypes.

3) FORCED CONFIRMATION + RED FLAG. In confirm_join_spec (join_engine.py:278-319) add: if any spec.diagnostics.key_dtype_divergences has level=="red" and the caller has not passed an explicit acknowledgement flag, raise a new typed error (e.g. `KeyDtypeMismatchError` in marvis/data/errors.py) mirroring DedupRequiredError, so execute cannot proceed silently. Thread an `ack_dtype_mismatch: bool` param through tool_confirm_join.

4) SERIALIZATION (marvis/packs/data_ops/tools.py). Add anchor_dtype/feature_dtype/dtype_divergent to `_key_pair_payload`; `_diagnostics_payload=asdict(diagnostics)` auto-carries key_dtype_divergences.

5) RENDERER (marvis/agent/renderers.py::_render_propose_join). Aggregate `any_dtype_mismatch` over joins[].diagnostics.key_dtype_divergences, add a "键类型" cell to the 拼接诊断 table (✓ / ✗ text≠float), append a ⚠️ block "检测到**键类型不一致**（一侧文本、一侧浮点/整型）：可能已发生精度丢失/前导零丢失导致静默漏配，请确认是否为同一标识后再拼接。" mirroring the existing any_fp_mismatch block at lines 1297-1301, and (for red level) mark it as forcing confirmation.

6) THRESHOLD relaxation for zero-padded short codes (marvis/data/csv_ingest.py::sniff_long_id_columns). Add a second flagging branch: a column is also long-id-protected when a majority of its non-null sampled values are purely-digit AND share a leading zero (regex `^0\d+$`) at a stable width — i.e. zero-padded codes where int/float promotion would strip the leading zero. Pseudocode inside the loop:
```
zero_padded = digit_like & values.str.match(r"^0\d+$")
if zero_padded.sum() / max(int(digit_like.sum()),1) >= 0.9 and zero_padded.sum() > 0:
    flagged.append(str(column)); continue
```
This keeps the existing >=15-digit rule and adds a parallel leading-zero rule; both force dtype=str so a zero-padded code cannot silently diverge across files.

**新增失败形状测试**：
- test_align_flags_dtype_divergent_key_pair: anchor id_card as strings ('110...234'), feature id_card read as float64 (values with a null so pandas promotes) → aligner still proposes a pair OR none; assert the surfaced KeyPair/diagnostics has dtype_divergent True with level 'red' (text vs float)
- test_candidate_match_methods_numeric_vs_idcard_returns_empty: assert candidate_match_methods(numeric_fp, raw_idcard_fp) == [] so the silent no-pair path is pinned, motivating the diagnostics-level dtype check
- test_diagnose_join_reports_key_dtype_divergence: construct anchor/feature ColumnProfiles with matching key name but dtype 'object' vs 'float64'; assert JoinDiagnostics.key_dtype_divergences is non-empty and level=='red'
- test_partial_precision_loss_passes_match_rate_but_flagged: feature key float64 where ~40% of ids are >=16 digits (corrupted); assert match_rate > MIN_KEY_MATCH_RATE (so the old gate accepts it) AND key_dtype_divergences flags it red
- test_confirm_join_forces_ack_on_dtype_mismatch: a red divergence with ack_dtype_mismatch=False raises KeyDtypeMismatchError; with ack=True it proceeds
- test_render_propose_join_surfaces_dtype_mismatch: feed a joins payload with diagnostics.key_dtype_divergences level red; assert the returned text contains the 键类型不一致 warning and the table has the ✗ text≠float cell
- test_sniff_long_id_flags_zero_padded_short_codes: CSV column of '000123','000456','000789' (6-digit, leading zero) → assert the column is in long_id_columns (relaxed rule) and the stored parquet keeps it as string with leading zeros intact
- test_sniff_long_id_does_not_flag_plain_short_numerics: column of '1','22','333' (no leading zero, short) → assert NOT flagged (relaxation must not over-trigger)
- test_cross_file_same_key_consistent_when_both_zero_padded: two files each with the same zero-padded code column both kept as string → assert no dtype divergence flag in diagnose_join

**风险 / 口径变化**：Blast radius: KeyPair and JoinDiagnostics are frozen dataclasses widely constructed in tests and serialized via asdict; new fields MUST be defaulted (anchor_dtype='', feature_dtype='', dtype_divergent=False; key_dtype_divergences=()) or every existing KeyPair(...) call site and stored-plan deserialization breaks. _diagnostics_payload uses asdict so a new nested dataclass auto-serializes — confirm the frontend propose_join renderer tolerates the extra keys (it reads by key, so additive is safe). Metric-basis (口径): match_rate, fan_out, uniqueness, conflict_report are UNCHANGED — the dtype divergence is a NEW metadata flag, not a change to any existing number, preserving INV-1 determinism. Forced-confirmation is a behavior change: previously a dtype-divergent key could execute silently; now a red divergence blocks execute until acknowledged — this can break existing auto_drive/end-to-end flows that don't pass an ack flag, so the ack path must be threaded through tool_confirm_join, gate_payloads, and auto_drive, and the gate contract (gates/contracts.py:292) updated; scope the forced block to level=='red' (text↔float) only to avoid false positives on lossless int↔text. Backward-compat: plans persisted before this field exists will deserialize with key_dtype_divergences=() (no flag) — acceptable (matches None-for-legacy pattern already used for Dataset.content_hash). The csv_ingest zero-padded relaxation risks over-flagging legitimately-numeric columns that happen to start with 0 in the sample (e.g. a fee column '0','012'); mitigate with the >=90% share + purely-digit + leading-zero regex and a min-length guard so single-digit '0' rows don't trip it, and note the guard is per-file so it only ensures within-file string storage — the cross-file consistency check in diagnose_join is the real safety net. False-negative residual: divergence is inferred from ColumnProfile.dtype which is sampled (SMALL_SAMPLE_N rows via profiler); a column that is object in the sample but effectively numeric is possible — acceptable since the check is conservative (only flags true dtype-family mismatch).

**设计决策点（提案已在补丁中给出，待审确认）**：
- Should the forced-confirmation gate fire only for text↔float (precision-loss) divergences, or also for int↔float / text↔int (lossless under the VARCHAR-cast join)? Recommendation: red-block only text↔float; warn (surface, no block) for the rest, but confirm with the spec owner given the memory note on 'forced confirmation' preference.
- The dtype divergence must be carried out of align() to diagnose_join — extend KeyPair with dtype fields (cleaner, but touches the frozen dataclass and every construction site incl. relaxation alternatives) vs recompute from ColumnProfile inside diagnose_join by name lookup (no KeyPair change, but diagnose_join must re-fetch profiles like _key_fps already does at join_engine.py:517-534). Which is preferred?
- For zero-padded short-code relaxation: is stripping-leading-zero the only short-code failure mode in scope, or should the guard also cover fixed-width all-digit codes without leading zeros that某些 files store as int and others as string (no precision loss but dtype divergence)? The cross-file check catches the latter regardless; only the within-file string-forcing needs the leading-zero heuristic.
- Where should the ack flag live end-to-end — a per-feature ack in JoinSpec (persisted) or a transient tool input on confirm_join? This determines whether an acknowledged mismatch survives a page reload / re-propose.

---

### C9 · 冠军证据渲染错指标 + 方向词写死 → 读真实 selection_metric

**现状（bug 机理）**：marvis/agent/renderers.py::_render_train_models (L364-417) builds the champion evidence line by HARD-CODING the selection metric per target_type instead of reading the tool's emitted selection_metric field. For binary it sets selector_label="按 OOT KS", selector_key="oot_ks", higher_is_better=True (renderers.py L381-383), then passes these into _champion_evidence_text (renderers.py L335-361), which reads metrics.get("oot_ks") for both champion and runner-up. But marvis/packs/modeling/train_tools.py::_pick_best_experiment (L504-551) for binary does NOT select on oot_ks: it selects via _binary_selection_score_and_metric (train_tools.py L493-501), which returns either _overfit_penalized_test_ks (score = test_ks - 0.5*max(0, train_ks - test_ks), preferring weighted_test_ks/weighted_train_ks when present; train_tools.py L453-477) with label BINARY_SELECTION_METRIC = "test_ks(overfit-penalized)" (train_tools.py L443), or, when eval_metric=="response_lift", _response_lift_score (reads test_lift_head_10; train_tools.py L480-490) with label RESPONSE_LIFT_SELECTION_METRIC = "test_lift_head_10" (train_tools.py L450). That chosen label is returned as tool output field selection_metric (train_tools.py L421-431), which _render_train_models NEVER reads (o.get("selection_metric") is ignored in that renderer). Two concrete defects: (1) WRONG METRIC DISPLAYED — the evidence line claims the champion was chosen "按 OOT KS" and prints OOT KS values, when the real basis was overfit-penalized test KS (or top-decile test lift for response_lift scenarios). (2) DIRECTION/MAGNITUDE-WRONG COMPARISON — _champion_evidence_text (renderers.py L355-360) recomputes its own runner-up as the max oot_ks among non-champions and prints abs(gap) with a fixed "高", so on the OOT-KS axis (which the champion did NOT win) the champion can have a LOWER oot_ks than the printed runner-up yet the text still asserts it beat them by abs(gap) "高". The champion genuinely leads on overfit-penalized test KS, but the sentence can claim it beat a runner-up it actually lost to on the axis being shown. NOTE: continuous (selector_key="oot_rmse", higher_is_better=False, L373-375) and multiclass (selector_key="oot_macro_auc", higher_is_better=True, L377-379) happen to match _pick_best_experiment's returned selection_metric ("oot_rmse" L527, "oot_macro_auc" L544), so ONLY the binary branch is mismatched; the response_lift sub-case is doubly wrong (wrong metric family entirely).

**关键代码位置**：
```
renderers.py L335-361 (_champion_evidence_text):
```
335 def _champion_evidence_text(experiments, best_id, selector_key, selector_label, higher_is_better) -> str:
340     def _val(exp):
341         metrics = exp.get("metrics") or {}
342         value = metrics.get(selector_key)
343         return float(value) if isinstance(value, (int, float)) else None
345     champion = next((e for e in experiments if e.get("experiment_id") == best_id), None)
346     champion_value = _val(champion) if champion is not None else None
347     if champion_value is None:
348         return ""
349     others = [
350         (e, _val(e)) for e in experiments
351         if e.get("experiment_id") != best_id and _val(e) is not None
352     ]
353     if not others:
354         return f"（依据：{selector_label}={champion_value:.4f}，为唯一可比算法）"
355     runner_up, runner_value = max(others, key=lambda item: item[1]) if higher_is_better else min(others, key=lambda item: item[1])
356     gap = champion_value - runner_value
357     return (
358         f"（依据：{selector_label}={champion_value:.4f}，"
359         f"较次优 {runner_up.get('recipe', '?')}（{runner_value:.4f}）"
360         f"{'高' if higher_is_better else '低'} {abs(gap):.4f}）"
361     )
```
renderers.py L380-405 (binary branch + evidence call inside _render_train_models):
```
380     else:
381         metric_columns = ["train_ks", "test_ks", "oot_ks", "test_auc", "oot_auc"]
382         selector_label = "按 OOT KS"
383         selector_key, higher_is_better = "oot_ks", True
...
393     if len(experiments) > 1:
399         evidence = _champion_evidence_text(
400             experiments, best_id, selector_key, selector_label, higher_is_better
401         )
402         text = (
403             f"**训练完成**:对比 {len(experiments)} 个算法，"
404             f"最优 **{best_recipe}**（★；{selector_label}）{evidence}。"
405         )
```
train_tools.py L453-501 (real binary selection functions + label resolver):
```
453 def _overfit_penalized_test_ks(metrics: dict) -> float:
468     test_ks = metrics.get("weighted_test_ks")
469     if not isinstance(test_ks, (int, float)):
470         test_ks = metrics.get("test_ks")
471     if not isinstance(test_ks, (int, float)):
472         return float("-inf")
473     train_ks = metrics.get("weighted_train_ks")
474     if not isinstance(train_ks, (int, float)):
475         train_ks = metrics.get("train_ks")
476     gap = float(train_ks) - float(test_ks) if isinstance(train_ks, (int, float)) else 0.0
477     return float(test_ks) - _CHAMPION_OVERFIT_PENALTY * max(0.0, gap)
480 def _response_lift_score(metrics: dict) -> float:
487     value = metrics.get("test_lift_head_10")
488     if not isinstance(value, (int, float)):
489         return float("-inf")
490     return float(value)
493 def _binary_selection_score_and_metric(eval_metric: str) -> tuple[Callable[[dict], float], str]:
499     if str(eval_metric or "").strip() == "response_lift":
500         return _response_lift_score, RESPONSE_LIFT_SELECTION_METRIC
501     return _overfit_penalized_test_ks, BINARY_SELECTION_METRIC
```
train_tools.py L436-450 (constants): _CHAMPION_OVERFIT_PENALTY=0.5 (L438); BINARY_SELECTION_METRIC="test_ks(overfit-penalized)" (L443); RESPONSE_LIFT_SELECTION_METRIC="test_lift_head_10" (L450).
train_tools.py L421-433 (emitted output includes selection_metric):
```
421     best, selection_metric = _pick_best_experiment(experiments, target_type=target_type, eval_metric=eval_metric)
424     return {
425         "experiments": experiments,
...
431         "selection_metric": selection_metric,
432         "failed": failed,
433     }
```
tests/test_strategy_development.py L786-800 (locks the WRONG label — feeds only oot_ks, asserts "按 OOT KS" evidence, no selection_metric field):
```
786 def test_render_train_models_champion_carries_evidence_vs_runner_up():
789     text, _ = _render_train_models({
790         "target_type": "binary",
791         "best_experiment_id": "exp-1", "best_recipe": "lgb",
792         "experiments": [
793             {"experiment_id": "exp-1", "recipe": "lgb", "metrics": {"oot_ks": 0.43}},
794             {"experiment_id": "exp-2", "recipe": "xgb", "metrics": {"oot_ks": 0.39}},
795         ],
796     })
798     assert "最优 **lgb**" in text
799     assert "依据：按 OOT KS=0.4300" in text
800     assert "较次优 xgb（0.3900）高 0.0400" in text
```
```

**受影响调用面**：
- marvis/agent/renderers.py:1856 — _RENDERERS["train_models"] = _render_train_models registers the renderer
- marvis/agent/renderers.py:1903-1905 — render_tool_output(tool, output) dispatches to _render_train_models
- marvis/agent/gate_adapters.py:67-70 — AUTO/gate consumer: render_tool_output(dep.tool_ref.tool, output) for a train_models dep; the returned text (champion evidence line) is appended to result.parts (surfaced as gate confirmation prose) and dep_tables to result.tables — the false '高' claim propagates into AUTO-mode messages, not just interactive display
- marvis/agent/plan_message_composer.py:117 — render_tool_output(terminal.tool_ref.tool, output) for terminal-step display (interactive)
- marvis/agent/gate_payloads.py:240 — builds model_delivery.selection_metric from o.get('selection_metric') for compare_experiments/select_experiment/post_training_action (NOT train_models); this path already shows the real field
- marvis/static/js/v2/model_delivery_panel.js:53,59 — frontend reads delivery.selection_metric and shows it as the '指标' chip; confirms selection_metric is the intended display source that _render_train_models ignores
- marvis/packs/modeling/select_tools.py:381-382 & 385-401 — _pick_best_comparison_row / _selection_metric_basis mirror the same binary basis (test_ks / test_lift_head_10) for the select/compare path; the train_models renderer is the sole place that diverges to oot_ks

**现有测试与缺口**：
- tests/test_strategy_development.py:786 test_render_train_models_champion_carries_evidence_vs_runner_up — LOCKS THE BUG: feeds only oot_ks metrics (no selection_metric field), asserts the evidence text says '依据：按 OOT KS=0.4300' and '较次优 xgb（0.3900）高 0.0400'. Must be rewritten to feed a selection_metric field + the real basis metrics and assert the correct label/value/direction.
- tests/test_modeling_pack.py:1562 — asserts tool out['selection_metric'] == 'test_ks(overfit-penalized)' (default binary). Confirms the field the renderer must consume.
- tests/test_modeling_pack.py:1595,1599 test_train_models_wires_scenario_eval_metric_into_champion_selection — asserts out['selection_metric'] == 'test_lift_head_10' and champion has max test_lift_head_10 for response_lift.
- tests/test_modeling_pack.py:1658 — asserts selection_metric == 'test_ks(overfit-penalized)' in another train path.
- tests/test_modeling_pack.py:2565,2597,2631,2637,2664,2668,2698,2768,2932,3092 — unit tests of _pick_best_experiment / _binary_selection_score_and_metric confirming the returned metric labels; these constrain what the renderer must map from.
- GAPS: no test asserts _render_train_models actually reads o['selection_metric']; no test covers the direction-wrong case where the champion has a LOWER value than the printed runner-up on the shown axis; no test covers the response_lift champion evidence line; frontend fixtures (test_frontend_playwright_smoke.py:246/325/339, test_frontend_screen_table.py:331/531, test_plan_driver.py:655/849) set selection_metric:'oot_ks' for the delivery panel — fixture-only for compare/select, confirms none assert on the train_models champion sentence.

**补丁方案（伪代码）**：
Goal: make _render_train_models render the ACTUAL selection metric (from o['selection_metric']) and produce a direction-correct comparison on the axis the champion actually won.

1) In renderers.py, add a resolver mapping the tool's selection_metric LABEL to (display_label, per-experiment value extractor, higher_is_better). selection_metric is a presentation label, NOT a metrics dict key, so a key/scorer is needed to fetch per-experiment values:
   def _selection_axis(o):
       sel = str(o.get("selection_metric") or "")
       target_type = str(o.get("target_type") or "binary")
       if sel == "test_ks(overfit-penalized)":   # BINARY_SELECTION_METRIC
           return ("按 test KS(过拟合惩罚)", _penalized_ks_value, True)   # value = test_ks - 0.5*max(0,train_ks-test_ks), weighted-aware
       if sel == "test_lift_head_10":              # RESPONSE_LIFT_SELECTION_METRIC
           return ("按 test 头部10%提升", _key_value("test_lift_head_10"), True)
       if sel == "oot_rmse":       return ("按 OOT RMSE", _key_value("oot_rmse"), False)
       if sel == "oot_macro_auc":  return ("按 OOT macro-AUC", _key_value("oot_macro_auc"), True)
       if sel == "oot_logloss":    return ("按 OOT logloss", _key_value("oot_logloss"), False)
       # fallback: preserve today's per-target_type defaults when selection_metric absent (old cached outputs)
       ... existing target_type switch (binary->oot_ks etc.) ...
   Subtlety for the penalized-KS axis: per-experiment value is NOT a single column; it is recomputed as test_ks - 0.5*max(0, train_ks - test_ks) (weighted-aware). Reuse the SAME scoring function as train_tools._overfit_penalized_test_ks (import it or a shared helper) so the rendered champion value == the value that drove selection. Do NOT invent a new formula.

2) Generalize _champion_evidence_text to take a value-extractor callable (value_of(exp)->float|None) instead of a bare selector_key, plus display_label + higher_is_better. Its runner-up pick + gap logic (L355-360) is already direction-aware via higher_is_better; because the axis is now the SAME axis the champion won, the champion is >= (higher_is_better) / <= the runner-up by construction, so the '高'/'低' word becomes truthful. Keep abs(gap) magnitude. Defensive guard: if the champion is somehow not the extreme on this axis, emit a neutral phrasing instead of asserting '高'.

3) In _render_train_models binary branch (L380-405): stop hard-coding selector_label/selector_key/higher_is_better; call _selection_axis(o) for (display_label, value_of, higher_is_better) and thread display_label into both the champion sentence (L404 '★；{selector_label}') and _champion_evidence_text. Route continuous/multiclass through the same _selection_axis so all target types share one source of truth; their behavior is unchanged because selection_metric already equals oot_rmse/oot_macro_auc.

4) Leave the comparison TABLE columns (metric_columns L373/377/381) unchanged — they are the full metric grid; only the evidence SENTENCE metric changes.

5) Backward-compat: when o lacks 'selection_metric', fall back to today's per-target_type label+key so nothing crashes; new tool outputs always carry the field.

**新增失败形状测试**：
- REWRITE tests/test_strategy_development.py:786 — feed target_type='binary', selection_metric='test_ks(overfit-penalized)', per-experiment metrics with train_ks/test_ks (e.g. exp-1 test_ks=0.40 train_ks=0.44 -> penalized 0.40-0.5*0.04=0.38; exp-2 test_ks=0.42 train_ks=0.60 -> penalized 0.42-0.5*0.18=0.33). Assert champion=exp-1 line cites the penalized-test-KS label with champion value 0.3800 and '较次优 xgb（0.3300）高 0.0500'; assert the substring 'OOT KS' does NOT appear in the evidence sentence.
- ADD direction-correctness test: champion (won on penalized test KS) has a LOWER oot_ks than a runner-up. Assert the evidence sentence does NOT print oot_ks values and does NOT falsely claim '高' vs that runner-up on the OOT axis — i.e. it is computed on the penalized-KS axis where the champion truly leads.
- ADD response_lift test: selection_metric='test_lift_head_10', two experiments with differing test_lift_head_10; assert the evidence label mentions the lift metric and champion/runner-up values come from test_lift_head_10, not oot_ks.
- ADD parity test: rendered champion penalized-KS value == train_tools._overfit_penalized_test_ks(champion_metrics) exactly (guards formula drift between renderer and selector).
- ADD continuous/multiclass regression test: selection_metric='oot_rmse' and 'oot_macro_auc' still render identically to today (label + value + '低'/'高' direction unchanged), proving the refactor is behavior-preserving for the already-correct branches.
- ADD backward-compat test: output dict WITHOUT a 'selection_metric' key still renders without error via the per-target_type fallback (mirrors old cached outputs).

**风险 / 口径变化**：Blast radius is the champion evidence SENTENCE in _render_train_models only; the comparison table and all other renderers are untouched. This is a 口径变化 (metric-basis change) in user-visible text: the evidence line will now say '按 test KS(过拟合惩罚)' (or the lift metric) instead of '按 OOT KS', and the printed champion/runner-up numbers change from oot_ks to the true selection axis — expected and correct, but any downstream that string-matches the old '按 OOT KS' / 'OOT KS' wording (docs, snapshot tests, AUTO-mode gate transcripts) needs updating. INV-1 (presentation only) is preserved: no selection logic changes, only which already-computed numbers are shown. Reusing train_tools._overfit_penalized_test_ks for the penalized value creates a renderers->train_tools import edge — prefer a small shared helper (or duplicate the exact 3-line formula with a parity test) to avoid a layering cycle; do NOT re-derive a different formula. Continuous/multiclass are behavior-preserving because their selection_metric already equals oot_rmse/oot_macro_auc. Backward-compat: older outputs missing selection_metric must fall back to current per-target_type defaults so replayed/cached data does not regress. Frontend model_delivery_panel.js and the gate_payloads path already read the real selection_metric and are unaffected (they cover compare/select, not train_models).

**设计决策点（提案已在补丁中给出，待审确认）**：
- The tool emits selection_metric as a LABEL string ('test_ks(overfit-penalized)', 'test_lift_head_10', 'oot_rmse', 'oot_macro_auc', 'oot_logloss') — not a metrics dict key. The renderer must map label->underlying key(s)/scorer to look up per-experiment values. Recommended: export the (label, scorer) pairs (or the constants + _overfit_penalized_test_ks) from train_tools so renderer and tool cannot drift. Confirm whether to import from train_tools or duplicate with a parity test.
- For the penalized-test-KS axis the per-experiment value is a recomputation (test_ks - 0.5*max(0,train_ks-test_ks), weighted-aware). Confirm the evidence sentence should display this recomputed penalized value (matches selection) rather than the plain test_ks column. Spec says 'render the actual selection_metric', read as the penalized value. If product prefers showing plain test_ks with a '(过拟合惩罚后选优)' note, that is smaller but leaves a mismatch between shown number and ranking.
- test_strategy_development.py:786 currently passes no selection_metric field; after the fix, should the renderer fallback still emit an 'OOT KS' evidence line for outputs lacking the field (legacy-only), while all NEW tool outputs always carry selection_metric? Assumed keep a safe fallback but rewrite the test to exercise the real field.

---

### C10 · "先别开始验证"→开跑 → 否定标记 + 疑问句护栏

**现状（bug 机理）**：`is_start_validation_intent` (marvis/agent/service.py:90-129) lower-cases + whitespace-strips input into `compact`, then at L113-114 does a bare substring test: `if any(phrase in compact for phrase in direct_phrases): return True`. `direct_phrases` (L95-112) includes "开始验证","启动验证","执行验证","运行验证","跑验证","开始执行","开始运行","开始跑","跑一下","跑一遍","跑起来","启动任务","开始任务","startvalidation","runvalidation","validatethistask". Because the check is pure substring containment with NO negation guard and NO question guard, any message that merely CONTAINS one of these fragments launches the run: "先别开始验证" (先别 = "hold off, don't"), "不要开始验证", "什么时候开始验证?" (a question), "能不能先不开始验证" all return True. Only the second branch (L115-129, `direct_commands` bare-command set) uses `compact.strip("。.!！?？") in direct_commands` full-string matching + trailing punctuation stripping — but that branch is never reached for these strings because the substring branch fires first. This is the negation/interrogative false-positive. Contrast the two sibling matchers that already guard against this: `is_continue_validation_intent` (L132-193) checks a 12-entry `negation_markers` tuple and returns False on any hit BEFORE its own substring fallback; `is_stop_validation_intent` (L439-459) checks `negated_phrases` first. Neither `is_start_validation_intent` nor its callee has any question-mark guard; the precedent for a question guard is `plan_driver.py` `_QUESTION` (L50-53). Dispatch: `is_agent_advance_intent` (service.py:259-260) = `is_start_validation_intent(content) or is_continue_validation_intent(content)`, consumed at marvis/routers/validation_agent.py:219 `if not is_agent_advance_intent(content):` — when advance-intent is (falsely) True, the else-path at L258-273 records intent="advance" and calls `dispatch_agent_validation_job`, actually launching the validation run instead of routing to the chat branch (L219-257).

**关键代码位置**：
```
marvis/agent/service.py:90-129 (is_start_validation_intent — the function to fix):
 90 def is_start_validation_intent(content: str) -> bool:
 91     text = content.strip().lower()
 92     if not text:
 93         return False
 94     compact = "".join(text.split())
 95     direct_phrases = (
 96         "开始验证",
 97         "启动验证",
 98         "执行验证",
 99         "运行验证",
100         "跑验证",
101         "开始执行",
102         "开始运行",
103         "开始跑",
104         "跑一下",
105         "跑一遍",
106         "跑起来",
107         "启动任务",
108         "开始任务",
109         "startvalidation",
110         "runvalidation",
111         "validatethistask",
112     )
113     if any(phrase in compact for phrase in direct_phrases):
114         return True
115     direct_commands = {
116         "开始",
117         "开始吧",
118         "启动",
119         "启动吧",
120         "运行",
121         "运行吧",
122         "执行",
123         "执行吧",
124         "跑吧",
125         "start",
126         "run",
127         "validate",
128     }
129     return compact.strip("。.!！?？") in direct_commands

marvis/agent/service.py:132-159 (is_continue_validation_intent — the negation-marker precedent; quote its 12-marker list; note it strips interior punctuation into `compact` first):
132 def is_continue_validation_intent(content: str) -> bool:
133     text = content.strip().lower()
134     if not text:
135         return False
136     compact = "".join(text.split()).strip("。.!！?？")
137     ... (drops interior punctuation ，。、；：,;: into compact)
143     negation_markers = (
144         "不继续",
145         "不要继续",
146         "先不继续",
147         "暂不继续",
148         "暂时不继续",
149         "不用继续",
150         "别继续",
151         "无需继续",
152         "不需要继续",
153         "不想继续",
154         "没必要继续",
155         "不打算继续",
156         "不会继续",     # (13 entries actually present; item prompt says ~12)
157     )
158     if any(marker in compact for marker in negation_markers):
159         return False

marvis/agent/service.py:439-459 (is_stop_validation_intent — negated_phrases precedent):
439 def is_stop_validation_intent(content: str) -> bool:
440     text = content.strip().lower()
441     if not text:
442         return False
443     negated_phrases = ("不要停止", "不用停止", "无需停止", "别停止")
444     if any(phrase in text for phrase in negated_phrases):
445         return False
446     keywords = (... "停止","停下","终止","中止","取消","别跑","不用跑","stop","cancel","abort","terminate")
459     return any(keyword in text for keyword in keywords)

marvis/agent/plan_driver.py:47-76 (the _QUESTION guard precedent from AGT-1; is_confirm checks _QUESTION on the RAW text before compacting):
47 # Interrogative guard: any question mark, or a trailing/whole-string question particle,
50 _QUESTION = re.compile(
51     r"[?？]|吗|吧$|行不行|可不可以|能不能|好不好|对不对|是不是|呢$",
52     re.IGNORECASE,
53 )
...
65 def is_confirm(text: str) -> bool:
66     raw = text or ""
67     if _QUESTION.search(raw):
68         return False
...

marvis/routers/validation_agent.py:219 (dispatch site):
219     if not is_agent_advance_intent(content):
```

**受影响调用面**：
- marvis/agent/service.py:260 — is_agent_advance_intent returns is_start_validation_intent(content) or is_continue_validation_intent(content) (the only in-repo caller of is_start_validation_intent)
- marvis/routers/validation_agent.py:219 — `if not is_agent_advance_intent(content):` gate in post_agent_message; false-positive here launches dispatch_agent_validation_job at L266-273 (intent='advance') instead of the chat branch L219-257
- marvis/routers/validation_agent.py:132 — the DRIVER_AGENT_TASK_TYPES branch returns BEFORE line 219, so is_agent_advance_intent (and is_start_validation_intent) only ever fires for non-driver wired task types; per marvis/agent/validation_app_service.py:157-166 WIRED minus marvis/agent/turn_handlers.py:60-69 DRIVER = TASK_TYPE_VALIDATION only. Blast radius is the classic single-model validation chat flow.
- marvis/agent/plan_driver.py:50-53,67 — _QUESTION regex + its use in is_confirm; reuse the same particle set as the question-guard precedent (do not import cross-module unless refactoring; a local mirror is acceptable and matches how negation lists are already duplicated per-matcher)

**现有测试与缺口**：
- tests/test_agent_intent.py — the ONLY unit test for these matchers. It imports is_agent_advance_intent and is_continue_validation_intent (NOT is_start_validation_intent directly).
- tests/test_agent_intent.py:16-47 test_continue_intent_recognized — parametrized positive cases; asserts both is_continue_validation_intent AND is_agent_advance_intent are True. None of these strings contain a start-phrase, so they are unaffected by a start-side negation guard.
- tests/test_agent_intent.py:50-78 test_non_continue_inputs_stay_chat — parametrized negatives; asserts NOT is_continue_validation_intent only (does NOT assert is_agent_advance_intent). GAP: because it never asserts is_agent_advance_intent for the negatives, a start-side false positive on e.g. '不要开始验证' would NOT be caught here. Also every negation case is a 继续-family string; there is ZERO coverage of start-family negation ('先别开始验证','不要开始验证') or interrogative start ('什么时候开始验证?').
- GAP: is_start_validation_intent has NO direct unit test at all — it is only exercised transitively via is_agent_advance_intent, and only on continue-family inputs.

**补丁方案（伪代码）**：
In marvis/agent/service.py, harden `is_start_validation_intent` to mirror the sibling matchers' negation-first + question-guard pattern, WITHOUT weakening the legitimate positives (bare '开始' / '开始验证' / 'start').

1. Add module-level constants near the other intent constants (top of file):
   - START_VALIDATION_NEGATION_MARKERS = a start-family analogue of is_continue_validation_intent's negation_markers, e.g. ("不开始","不要开始","先别开始","别开始","暂不开始","暂时不开始","不用开始","无需开始","不需要开始","不想开始","没必要开始","不打算开始","不会开始","不启动","不要启动","别启动","不执行","不要执行","别执行","不运行","不要运行","不跑","不要跑","别跑"). Include the generic negators that plan_driver._NEGATED_CONFIRM already trusts ("先别","别","不要","不用","不需要","先不","暂不","暂停") so "先别开始验证"/"不要开始验证" are caught even for phrasings not enumerated. English: do\s*not|don't|dont|not\s+(start|run|validate) — keep parity with existing English direct_phrases.
   - _START_QUESTION = re.compile(r"[?？]|吗|吧$|呢$|什么时候|何时|要不要|需不需要|能不能|可不可以|是不是", re.IGNORECASE) — mirror plan_driver._QUESTION's particle set plus start-context interrogatives ('什么时候开始验证?','要不要开始').

2. Inside is_start_validation_intent, AFTER `text = content.strip().lower()` / empty guard and BEFORE any positive matching:
     if _START_QUESTION.search(text):        # guard on RAW text like is_confirm does
         return False
     compact = "".join(text.split())
     # drop interior punctuation into a matchable form for negation scan,
     # same as is_continue_validation_intent L141-142
     compact_np = compact
     for ch in "，。、；：,;:":
         compact_np = compact_np.replace(ch, "")
     if any(marker in compact_np for marker in START_VALIDATION_NEGATION_MARKERS):
         return False
   Then keep the existing direct_phrases substring branch (L113-114) and direct_commands full-string branch (L115-129) UNCHANGED. Order matters: question guard + negation guard must precede the substring branch so "先别开始验证"/"什么时候开始验证?" short-circuit to False before the substring test can return True.

Rationale for question-on-raw-text (not compact): plan_driver.is_confirm runs _QUESTION.search on `raw` before compacting (plan_driver.py:66-67); '吧$'/'呢$' anchors need the un-stripped trailing char. Since is_start's positive branch also strips trailing 。.!！?？ only in the direct_commands branch, guarding raw text is consistent and catches '开始验证?'.

Do NOT touch is_continue_validation_intent, is_stop_validation_intent, is_agent_advance_intent, or plan_driver.py. The fix is localized to is_start_validation_intent + new module constants.

**新增失败形状测试**：
- In tests/test_agent_intent.py, import is_start_validation_intent directly and add test_start_intent_recognized parametrized over the legitimate positives that MUST stay True: '开始验证','启动验证','执行验证','运行验证','开始','开始吧','启动','start','run','validate','开始执行','跑起来' — assert is_start_validation_intent AND is_agent_advance_intent True. Guards against over-broad negation markers eating real starts.
- Add test_start_negations_stay_chat parametrized over the bug strings: '先别开始验证','不要开始验证','别开始验证','暂不开始验证','不用开始验证','不需要开始验证','先不开始' — assert NOT is_start_validation_intent AND NOT is_agent_advance_intent (the second assertion is the one the current suite omits and is what actually protects the dispatch at validation_agent.py:219).
- Add test_start_questions_stay_chat parametrized over interrogatives: '什么时候开始验证?','什么时候开始验证？','要不要开始验证','能不能开始验证','开始验证吗？','现在可以开始验证吗' — assert NOT is_start_validation_intent AND NOT is_agent_advance_intent. Feeds the interrogative dirty shape; asserts the _START_QUESTION guard fires.
- Add a boundary case that must remain True to prove the guard is not too greedy: '开始吧' (ends in 吧 which _QUESTION treats as a particle via '吧$') — DECISION NEEDED: plan_driver._QUESTION treats trailing '吧' as a question, but '开始吧'/'继续吧' are legitimate affirmatives in the start/continue vocab (they are explicit entries in direct_commands/direct_phrases). So _START_QUESTION MUST NOT include bare '吧$' if '开始吧' is to stay True — either drop '吧$' from _START_QUESTION, or run the question guard only when no exact direct_commands/direct_phrase full-string match exists. Add test asserting is_start_validation_intent('开始吧') is True to lock this in. Record as open question.
- Extend the existing tests/test_agent_intent.py:50-78 negatives to also assert NOT is_agent_advance_intent (currently only asserts NOT is_continue_validation_intent), closing the transitive-coverage gap for the whole advance gate.

**风险 / 口径变化**：Blast radius is narrow: is_start_validation_intent feeds only is_agent_advance_intent → validation_agent.py:219, and that line is only reachable for TASK_TYPE_VALIDATION (all DRIVER_AGENT_TASK_TYPES short-circuit at validation_agent.py:132 via the driver branch, which has its own confirm gating in plan_driver.is_confirm). So the classic single-model validation chat flow is the only user-facing surface affected. Metric/口径 change: none — this only reclassifies borderline user messages from 'advance/launch run' to 'chat question'; it changes WHICH branch runs, not any computed validation metric. Backward-compat direction is safe-by-default: the fix only ADDS False returns (negations/questions no longer launch), it never adds new True returns, so it cannot newly auto-launch a run that previously stayed chat. Primary regression risk = over-broad markers swallowing legitimate starts: (a) generic negators like '别'/'不要' as bare substrings are safe because a legit start would never contain them, but a marker like '不用' could appear in unrelated chat — acceptable since such chat should route to chat anyway; (b) the '吧$' question-particle collision with the legitimate affirmative '开始吧'/'启动吧'/'运行吧' (all present in direct_commands) is the real trap — _START_QUESTION must exclude bare trailing '吧' or run the guard after checking exact-command membership, else these affirmatives break (covered by a new positive test). Keep negation lists as a local per-matcher tuple (consistent with the existing duplicated negation lists across the three matchers) rather than importing plan_driver internals, to avoid coupling the router-facing service to the orchestrator module.

**设计决策点（提案已在补丁中给出，待审确认）**：
- '吧$' collision: plan_driver._QUESTION treats trailing '吧' as interrogative, but '开始吧'/'启动吧'/'运行吧'/'跑吧' are legitimate start affirmatives explicitly listed in is_start_validation_intent's direct_commands (L117-124). If _START_QUESTION includes bare '吧$', these break. Recommended: either (a) omit '吧$' from _START_QUESTION, or (b) evaluate the exact direct_commands/direct_phrases match BEFORE the question guard for whole-string affirmatives. Needs a decision; a positive test on '开始吧' should lock whichever is chosen.
- Interior-punctuation normalization scope: is_continue_validation_intent strips ，。、；：,;: into compact before negation scan (L141-142); should is_start_validation_intent mirror this exactly, or is the simpler compact = ''.join(text.split()) sufficient for start-family negations? Recommend mirroring for parity, but confirm no start negation depends on a punctuation-bearing form.
- Should the fix also add is_start_validation_intent to the direct imports/assertions in tests/test_agent_intent.py (currently only is_agent_advance_intent + is_continue are imported)? Recommended yes, to get direct start-side coverage rather than only transitive-via-advance.
- English-negation parity: existing English positives are 'start'/'run'/'validate'/'startvalidation' etc.; should the negation guard also cover English ("do not start", "don't run") to match plan_driver._NEGATED_CONFIRM, or is Chinese-only sufficient given the product's primary user language? Recommend including the small English set for parity.

---

### D11 · 默认模板令精选特征在 train+test 拟合 → 恢复安全 holdout

**现状（bug 机理）**：The default modeling templates leak test-split labels into the feature-selection statistics fit whenever an OOT split exists. Both the single-table `modeling` template and the multi-table `modeling_with_join` template feed a holdout value of `['oot']` into the `精选特征` (select_features) step, overriding `select_features`'s safe default of `('test','oot')`.

Data-flow trace of where `['oot']` originates and how it reaches select:
- `marvis/agent/modeling_setup.py:304-312` — `build_modeling_proposal` computes `oot = split_values.get("oot")` and sets `ModelingProposal.holdout_values = [oot] if oot else []` (so `['oot']` when an OOT split exists, `[]` otherwise).
- `marvis/agent/modeling_setup.py:80` and `:103` — `ModelingProposal.template_slots()` binds that list into the `holdout_values` template slot for both `modeling_with_join` and `modeling`.
- `marvis/packs/modeling/prepare_tools.py:133` (`prepare_split` tool) and `:189` (`make_split` tool) — independently recompute `"holdout_values": ["oot"] if "oot" in counts else []` in their tool output, which the multi-table template reads via `$ref`.
- Single-table template `marvis/orchestrator/templates/modeling.py:293` binds `"holdout_values": "{slot:holdout_values}"` (= `['oot']`) into the `精选特征` step.
- Multi-table template `marvis/orchestrator/templates/modeling.py:581` binds `"holdout_values": "$ref:切分样本.output.holdout_values"` (make_split output = `['oot']`) into the `精选特征` step.
- `marvis/packs/modeling/feature_tools.py:25,41` — `tool_select_features` does `holdout = inputs.get("holdout_values")` then passes `holdout_values=tuple(str(v) for v in holdout) if holdout else ("test","oot")`. A truthy `['oot']` overrides the safe default, so it passes `("oot",)`.
- `marvis/packs/modeling/select.py:67-74` calls `_selection_fit_mask(..., holdout_values=("oot",))`; `_selection_fit_mask` (`select.py:144-148`) builds `mask = ~frame[split_col].isin(("oot",))`, i.e. fit rows = train + test. IV (`_select_features_raw` at select.py:167-190), collinearity dedup (`_drop_collinear`), VIF (`_drop_high_vif`), and top_k all fit on train+test instead of train-only.

Net effect: in EVERY default modeling run that has an OOT split, univariate IV filtering, correlation dedup, VIF, and top_k selection peek at the test-split labels — the exact FS-2 leakage the `('test','oot')` default was designed to prevent. When no OOT exists, the slot is `[]` (falsy), feature_tools.py:41 falls back to `('test','oot')`, and there is no leak — so the bug is specific to OOT-present runs.

Crucially, the SAME `holdout_values` value (`['oot']`) is ALSO fed to the `特征筛选` (screen_features) step (modeling.py:259 single, :560 multi), and for screen `['oot']` is CORRECT: `screen_features` intentionally screens on train+test as pooled-dev and only holds out OOT (its own tool default is `('oot',)` at feature_tools.py:94; `_dev_mask` docstring at screen.py:107 confirms "exclude the holdout (e.g. OOT)"). So the shared slot is right for screen and wrong for select — the fix must not change the screen binding.

**关键代码位置**：
```
marvis/packs/modeling/feature_tools.py:14-47 (tool_select_features — the leak entry point):
```
14  def tool_select_features(inputs: dict, ctx) -> dict:
15      runtime = _runtime(ctx)
16      dataset = runtime.registry.get(str(inputs["dataset_id"]))
...
24      split_col = _optional_str(inputs.get("split_col"))
25      holdout = inputs.get("holdout_values")
26      result = select_features(
...
39          split_col=split_col,
40          split_value=inputs.get("split_value"),
41          holdout_values=tuple(str(v) for v in holdout) if holdout else ("test", "oot"),
42          allow_full_fit=bool(inputs.get("allow_full_fit")),
...
47      )
```

marvis/orchestrator/templates/modeling.py:286-303 (single-table 精选特征 step — the offending binding at :293):
```
286  title="精选特征",
287  tool_ref=ToolRef("modeling", "select_features"),
288  inputs_template={
289      "dataset_id": "$ref:切分样本.output.result_dataset_id",
290      "features": "$ref:特征筛选.output.selected",
291      "target_col": "{slot:target_col}",
292      "split_col": "{slot:split_col}",
293      "holdout_values": "{slot:holdout_values}",     # <-- leaks ['oot'] into select
294      "target_type": "$ref:选择建模规格.output.target_type",
295      "space": "raw",
296      "iv_min": 0.02,
297      "corr_max": 0.95,
...
302      "vif_max": 1e9,
303  },
```

marvis/orchestrator/templates/modeling.py:574-588 (multi-table 精选特征 step — offending binding at :581):
```
574  title="精选特征",
575  tool_ref=ToolRef("modeling", "select_features"),
576  inputs_template={
577      "dataset_id": "$ref:切分样本.output.result_dataset_id",
578      "features": "$ref:特征筛选.output.selected",
579      "target_col": "{slot:target_col}",
580      "split_col": "$ref:切分样本.output.split_col",
581      "holdout_values": "$ref:切分样本.output.holdout_values",   # <-- leaks ['oot'] into select
582      "target_type": "$ref:选择建模规格.output.target_type",
583      "space": "raw",
...
587      "vif_max": 1e9,
588  },
```

marvis/packs/modeling/select.py:144-148 (fit-mask exclusion — where the wrong holdout takes effect):
```
144  holdout = tuple(str(value) for value in (holdout_values or ("test", "oot")))
145  mask = ~frame[str(split_col)].astype(str).isin(holdout)
146  if not mask.any():
147      raise FeatureError("select_features fit frame is empty after excluding holdout rows")
148  return mask, "train"
```
select_features signature default (select.py:46): `holdout_values: tuple[str, ...] = ("test", "oot")` — the correct, already-tested (FS-2) default that the template override defeats.
```

**受影响调用面**：
- marvis/orchestrator/templates/modeling.py:293 — single-table `modeling` template `精选特征` step binds `holdout_values: {slot:holdout_values}` (=['oot']) into select_features (THE FIX SITE)
- marvis/orchestrator/templates/modeling.py:581 — multi-table `modeling_with_join` template `精选特征` step binds `holdout_values: $ref:切分样本.output.holdout_values` (=['oot']) into select_features (THE FIX SITE)
- marvis/orchestrator/templates/modeling.py:259 — single-table `特征筛选` (screen_features) step binds the same slot; ['oot'] is CORRECT here, DO NOT touch
- marvis/orchestrator/templates/modeling.py:560 — multi-table `特征筛选` (screen_features) step binds the same $ref; ['oot'] is CORRECT here, DO NOT touch
- marvis/orchestrator/templates/modeling.py:199 & :467 — SlotSpec('holdout_values', ...) description is literally 'OOT split value(s) held out of the leakage SCREEN' — confirms the slot is the screen's holdout, not select's
- marvis/agent/modeling_setup.py:80 & :103 — ModelingProposal.template_slots() populates the holdout_values slot from proposal.holdout_values
- marvis/agent/modeling_setup.py:304-312 — build_modeling_proposal sets holdout_values=[oot] if oot else []
- marvis/packs/modeling/prepare_tools.py:133 (prepare_split) & :189 (make_split) — emit output holdout_values=['oot'] if 'oot' in counts else [] (the multi-table $ref source)
- marvis/packs/modeling/feature_tools.py:41 — tool_select_features applies the override; :94 (screen) & :209 (non-binary screen) apply ('oot',) default — screen semantics
- marvis/packs/feature/tools.py:157-165 & :450 — feature-pack select/woe consume holdout_values with ('test','oot')/('oot',) defaults but the FEATURE template (marvis/orchestrator/templates/feature.py) has NO select_features/woe_encode step (only screen_features at feature.py:187), so no template feeds them ['oot'] — feature template unaffected
- marvis/packs/modeling/select.py:144 — _selection_fit_mask consumes holdout_values

**现有测试与缺口**：
- tests/test_modeling_select.py:204 test_select_features_default_excludes_test_and_oot_from_iv_statistics — proves select_features's ('test','oot') DEFAULT excludes both splits from IV (uses inverted holdout labels + train-only oracle). This guards the tool-level default but NOT the template binding, so it passes today despite the leak.
- tests/test_modeling_select.py:251 test_select_features_auto_detects_standard_split_column — default-holdout path via SPLIT_COLUMN auto-detect; also asserts fit_split=='train', fit_rows==8 (train-only).
- tests/test_orch_templates.py:144 test_modeling_template_phases_gates_and_refs — instantiates the `modeling` template with holdout_values=['oot'] (line 160) and validates phases/gates/refs, but does NOT assert what holdout_values the 精选特征 step receives — so it currently accepts the leak.
- tests/test_orch_templates.py:275-276 — asserts holdout_values omitted from screen/refine inputs when the slot is dropped (optional-slot handling), relevant to the omit-when-absent behavior a fix relies on.
- tests/test_feature_pack.py:332 test_woe_encode_default_holdout_excludes_test_and_oot — parallel PREP-1 precedent: default holdout moved from ('oot',) to ('test','oot') for WOE fit; establishes the platform pattern the select fix must match.
- tests/test_plan_driver.py:1172,1180 — driver-level fixtures showing make_split output holdout_values=['oot'] (OOT present) and =[] (no OOT); confirms the [] falsy-fallback path.
- GAP: no test asserts the DEFAULT MODELING TEMPLATE feeds select_features a train-only fit when an OOT exists. No end-to-end/template-level test catches this leak — the tool default is tested but the template override that defeats it is not.
- GAP: no test asserts the two 精选特征 steps' resolved inputs exclude ['oot']-only holdout (i.e. that select gets ('test','oot') or no holdout_values key).

**补丁方案（伪代码）**：
Recommended fix (template binding; smallest, lowest-risk, matches WOE/PREP-1 precedent): remove the `holdout_values` key from BOTH `精选特征` (select_features) StepTemplate.inputs_template blocks so select_features falls back to its own safe default.

marvis/orchestrator/templates/modeling.py — single-table `精选特征` (currently :286-303):
```
inputs_template={
    "dataset_id": "$ref:切分样本.output.result_dataset_id",
    "features": "$ref:特征筛选.output.selected",
    "target_col": "{slot:target_col}",
    "split_col": "{slot:split_col}",
    # DELETE: "holdout_values": "{slot:holdout_values}",
    # (rationale comment) FS-2: selection must fit train-only. Do NOT forward the
    # screen's ['oot'] holdout slot here — let select_features apply its own
    # ('test','oot') default so IV/corr/VIF/top_k never see test-split labels.
    "target_type": "$ref:选择建模规格.output.target_type",
    "space": "raw",
    "iv_min": 0.02, "corr_max": 0.95, "vif_max": 1e9,
},
```

marvis/orchestrator/templates/modeling.py — multi-table `精选特征` (currently :574-588): identically delete the `"holdout_values": "$ref:切分样本.output.holdout_values",` line (:581) and add the same rationale comment.

Why deletion is safe and sufficient:
- `feature_tools.py:41` already reads `inputs.get("holdout_values")` and applies `("test","oot")` when the key is absent/falsy — so an omitted key yields the correct train-only fit for both OOT-present and OOT-absent runs.
- Do NOT change `feature_tools.py:41` default nor `select.py:46/144` default — they are already correct and independently tested (FS-2). Changing them is a no-op for the default and would not stop the template's explicit override, so it is the wrong lever.
- Do NOT touch the `特征筛选` (screen_features) bindings at modeling.py:259/:560 — `['oot']` is the intended screen holdout (screen pools train+test as dev; tool default is `('oot',)`). Leave the `holdout_values` slot (modeling.py:199/:467) and its `modeling_setup.py`/`prepare_tools.py` producers unchanged; they still serve the screen step.
- The `woe_encode` tool (feature/tools.py:450) shares the `('test','oot')` default but is never fed `holdout_values` by any template, so it is already safe — this fix makes select consistent with woe.

Alternative considered and rejected — a dedicated `holdout_values_select` slot bound to `['test','oot']`: unnecessary indirection. Since select's tool-level default is already `('test','oot')`, omitting the key achieves the identical result with less surface area. Only introduce a separate slot if a future requirement needs the modeling template to override select's holdout to something other than the safe default (not the case today).

Also update the stale SlotSpec doc at modeling.py:199 & :467 is optional — the slot remains a screen-only holdout after the fix; if desired, keep the description as-is ("held out of the leakage screen") since it is now literally accurate (only the screen consumes it).

**新增失败形状测试**：
- Template-level (single-table): instantiate get_template('modeling') via Planner.from_template with holdout_values=['oot'] and an OOT split; assert the resolved `精选特征` step inputs do NOT contain a `holdout_values` key equal to ['oot'] (either key absent, or == ['test','oot']). Dirty shape fed: OOT-present run where the slot is ['oot']. Assertion: select step's holdout != ['oot'].
- Template-level (multi-table): same for get_template('modeling_with_join') — with 切分样本.output.holdout_values=['oot'], assert the `精选特征` step's resolved holdout_values is absent/('test','oot'), while the `特征筛选` (screen) step still receives ['oot'] (regression guard that the screen binding is untouched). Dirty shape: joined modeling with OOT. Assertion: screen holdout==['oot'] AND select holdout omitted.
- End-to-end tool behavior via the template default: run select_features (or tool_select_features with the template's resolved inputs but no holdout_values key) on a frame where test/oot carry inverted labels (reuse the fixture shape from test_modeling_select.py:212-219); assert result.fit_split=='train' and result.fit_rows == train-count (test+oot excluded) and scores['signal']['iv'] == train-only oracle (not the train+test pooled IV). Dirty shape: OOT+test present with inverted holdout labels. Assertion: IV matches train-only, i.e. test rows excluded from the fit.
- Regression: OOT-absent run — instantiate modeling template with holdout_values=[] (no OOT); assert select still fits train-only (excludes test) via the ('test','oot') default, i.e. the [] falsy-fallback keeps working. Dirty shape: test-only split, no OOT. Assertion: fit_split=='train', test rows excluded.
- Guard against re-introduction: a static assertion over both templates that the select_features step's inputs_template has no 'holdout_values' key (or, if a select-specific slot is introduced instead, that it resolves to ('test','oot')). Prevents a future edit from re-binding the screen slot into select.

**风险 / 口径变化**：Blast radius: two lines in marvis/orchestrator/templates/modeling.py (the two `精选特征` steps). No pack code changes. The screen step, WOE, and the feature-pack templates are untouched.

Metric-basis change (口径变化): YES, intentional and desirable. After the fix, in any default modeling run WITH an OOT split, IV/correlation/VIF/top_k selection statistics are computed on TRAIN ONLY instead of TRAIN+TEST. This can change which features survive the funnel (and therefore downstream tuning/model metrics) versus historical runs. This is a leakage-fix correction, not a regression — but any golden/snapshot test or saved baseline that recorded selected-feature sets or scores from a leaking run will shift and must be re-baselined. Runs WITHOUT an OOT split are unaffected (they already fell back to ('test','oot')).

Backward-compat: The `holdout_values` slot still exists and still drives the screen step, so existing driver/adjust flows that set holdout_values continue to work for screening. Callers who deliberately passed holdout_values to influence SELECT (unlikely — no such flow found; the driver only sets it once and it was routed to both) lose that channel for select; if a future need arises, add a distinct select-holdout slot rather than reusing the screen slot.

Non-risks confirmed by trace: (1) feature-pack select_features/woe_encode are not fed holdout_values by any template (feature template only has screen_features), so they are unaffected. (2) select.py and feature_tools.py defaults are already ('test','oot'); omitting the key is exactly equivalent to the safe default, including the empty-OOT case. (3) screen_features default is ('oot',) and its binding is preserved, so screen behavior is byte-identical.

Edge case to keep in mind for tests: if a dataset had NO test split (only train+oot), excluding ('test','oot') simply excludes oot (isin over absent 'test' is a no-op), so the fit is train-only either way — no empty-frame risk beyond what _selection_fit_mask already guards (select.py:146 raises FeatureError on empty fit frame).

**设计决策点（提案已在补丁中给出，待审确认）**：
- Whether any driver/adjust UX intends users to widen select's holdout at runtime. Today the single `holdout_values` slot is routed to both screen and select; after the fix it drives only screen. If product wants a user-tunable select holdout, a separate slot (e.g. holdout_values_select defaulting to ['test','oot']) is the clean extension — but no current flow exercises this, so I recommend deferring it.
- Whether to update the SlotSpec descriptions at modeling.py:199/:467 ('held out of the leakage screen'). They become literally accurate after the fix (slot now screen-only), so no change is strictly required; a one-line clarification could note it is the screen holdout only.
- Whether existing regression/golden baselines (if any) capture selected-feature sets or KS/AUC from OOT-present default runs; those will legitimately shift once selection stops seeing test labels and will need re-baselining. I did not find such golden fixtures in tests/, but a repo-wide baseline artifact (outside tests/) could exist.

---

### D12 · task.time_col 从不触发时间 OOT → 传参 + 非别名列可触发

**现状（bug 机理）**：A user-supplied task.time_col (default 'apply_month', settable at task creation via CreateTaskRequest.time_col at marvis/api_schemas.py:16 -> marvis/routers/tasks.py:105 -> marvis/repositories/tasks.py:1026/1114 -> TaskRecord.time_col at marvis/domain.py:64/101) never reaches modeling setup. In marvis/agent/turn_handlers.py::_run_modeling_setup (:635-697) the call to build_modeling_proposal (:659-673) passes target_type/recipes/sample_weight_col/anchor_id/join_feature_ids/target_col/field_hints but NOT time_col. Contrast the sibling vintage flow _run_vintage_setup (:519-541), which DOES pass time_col=getattr(task,'time_col','') or None at :529.

Inside build_modeling_proposal (marvis/agent/modeling_setup.py:145-329) the time column driving time-extrapolated OOT is derived ONLY from business_columns.get('loan_month_col') (:267 joined branch, :292 single-file branch). business_columns = _infer_business_columns(available_columns) (:177; def :350-364), matching via _first_matching_column (:367-372) against the exact-lowercased alias tuple _BUSINESS_COLUMN_ALIASES['loan_month_col'] = ('loan_month','apply_month','book_month' + 3 CJK synonyms) at :135-142. So time-based OOT fires ONLY when a column name lowercased exactly equals one of those 6 aliases. A user whose date column is named stmt_date, observe_month, dt, or vintage_ym gets NO OOT even though they explicitly set time_col at task creation; downstream OOT metrics degrade to n/a (notes at :276-279).

Three split branches decide OOT: (1) :249-253 'if setup.split_col:' - an existing split column detected by detect_setup wins, no time OOT; (2) :254-279 'elif joined:' - builds auto_split_config with oot_by_time=loan_month_col if present, split runs in-plan later via make_split; (3) :280-294 'else:' (single file, no split col) - calls _generate_split(..., time_col=business_columns.get('loan_month_col')...) (def :567-623) which sets split_config['oot_by_time'] and runs prepare_modeling_frame now. All three paths ignore task.time_col.

**关键代码位置**：
```
## marvis/agent/turn_handlers.py:659-673 (call site - the fix must add time_col here)
659        proposal = build_modeling_proposal(
660            registry,
661            backend,
662            task.id,
663            task.source_dir,
664            target_type=_modeling_target_type(task),
665            recipes=_modeling_recipes(task),
666            sample_weight_col=getattr(task, "sample_weight_col", "") or None,
667            anchor_id=(c1_assignment or {}).get("anchor_id"),
668            join_feature_ids=(c1_assignment or {}).get("feature_ids"),
669            target_col=(c1_assignment or {}).get("target_col"),
670            field_hints=fetch_field_convention_hints(
671                runtime.settings,
672                keywords=_modeling_field_hint_keywords(task, c1_proposal),
673            ),
674        )

## turn_handlers.py:528-530 (vintage call - the pattern to mirror)
528        target_col=getattr(task, "target_col", "") or None,
529        time_col=getattr(task, "time_col", "") or None,
530    )

## modeling_setup.py:145-154 (signature - needs a new keyword-only time_col param)
145 def build_modeling_proposal(
146     registry, backend, task_id: str, source_dir, *, seed: int = DEFAULT_RANDOM_SEED,
147     recipe: str | None = None, recipes: list[str] | None = None,
148     target_type: str | None = None,
149     sample_weight_col: str | None = None,
150     anchor_id: str | None = None,
151     join_feature_ids: list[str] | None = None,
152     target_col: str | None = None,
153     field_hints: dict | None = None,
154 ) -> ModelingProposal:
...
176     available_columns = backend.column_names(path)
177     business_columns = _infer_business_columns(available_columns)

## modeling_setup.py:254-293 (the two branches that resolve the time column via alias-only lookup)
254     elif joined:
...
267         time_col = business_columns.get("loan_month_col")
268         if isinstance(time_col, str) and time_col:
269             auto_split_config["oot_by_time"] = time_col
270             auto_split_config["oot_size"] = DEFAULT_OOT_SIZE
...
280     else:
281         dataset_id, split_col, split_values, counts, note = _generate_split(
...
292             time_col=business_columns.get("loan_month_col") if isinstance(business_columns.get("loan_month_col"), str) else None,
293         )

## modeling_setup.py:249-253 (split-col branch takes precedence over any time OOT)
249     if setup.split_col:
250         dataset_id = dataset.id
251         split_col = setup.split_col
252         split_values = dict(setup.split_values)
253         counts = dict(setup.counts)

## marvis/packs/modeling/prepare.py:166-172 (missing-column guard - proves time_col MUST exist in the frame or make_split raises)
166     time_col = config.get("oot_by_time")
167     oot_size = float(config.get("oot_size", DEFAULT_OOT_SIZE))
168     expect_oot = bool(rule_mask.any() and (out.loc[rule_mask, SPLIT_COLUMN] == "oot").any())
169     if time_col:
170         time_col = str(time_col)
171         if time_col not in out.columns:
172             raise ModelingError(f"missing columns: {time_col}")

## marvis/agent/vintage_setup.py:116-128 (_resolve_named_col - the precedent for 'explicit requested col wins if present, else alias fallback')
116 def _resolve_named_col(columns: list[str], requested: str | None, hints: tuple[str, ...]) -> str:
117     requested = str(requested or "").strip()
118     if requested and requested in columns:
119         return requested
120     lowered = {column.lower(): column for column in columns}
121     for hint in hints:
122         if hint in lowered:
123             return lowered[hint]
124     for column in columns:
125         low = column.lower()
126         if any(hint in low for hint in hints):
127             return column
128     return ""
```

**受影响调用面**：
- marvis/agent/turn_handlers.py:659-673 - the ONLY production caller of build_modeling_proposal; must add time_col=getattr(task,'time_col','') or None (mirror :529 vintage)
- marvis/agent/modeling_setup.py:145-154 - build_modeling_proposal signature; add keyword-only param time_col: str | None = None
- marvis/agent/modeling_setup.py:254-279 - joined branch; resolve effective time col from user time_col (validated against available_columns) OR business_columns['loan_month_col'] before setting auto_split_config['oot_by_time']
- marvis/agent/modeling_setup.py:280-294 - single-file branch; pass the resolved effective time col into _generate_split(time_col=...)
- marvis/agent/modeling_setup.py:249-253 - split-col branch; decide whether user time_col should override an auto-detected (non-configured) split_col, since detect_setup runs WITHOUT configured_split (called at :196-202 with no configured_split) and may auto-pick a split column via _looks_like_split_name/object-col heuristics (sample_setup.py:154-163)
- marvis/agent/modeling_setup.py:196-202 - detect_setup() call; note it is invoked WITHOUT configured_split, so split-col detection is purely heuristic and could shadow a user's intended time OOT
- marvis/agent/sample_setup.py:73-165 detect_setup / :282-284 _looks_like_split_name - the heuristic that may auto-detect a split col that conflicts with the user's time_col intent
- tests/test_modeling_recipes.py:1408-1760 - ~15 unit callers of build_modeling_proposal, all keyword-arg style after 4 positionals; new keyword-only param is backward-compatible (none pass time_col today)
- marvis/agent/vintage_setup.py:60-88 build_vintage_proposal / :116-128 _resolve_named_col - reference implementation of the resolve-precedence to mirror

**现有测试与缺口**：
- tests/test_modeling_api.py:467-514 test_modeling_business_materials_without_split_survive_auto_split - single-file, column literally named 'loan_month' (an alias) so current alias path already time-extrapolates OOT; asserts '已按' + '时间外推 OOT' + 'loan_month' in opening and oot count > 0. Does NOT exercise a non-alias time_col.
- tests/test_modeling_api.py:839-892 test_modeling_multiple_files_with_time_column_auto_splits_oot_after_join - joined path, anchor column named 'loan_month' (alias); asserts split_step.inputs.split_config.oot_by_time == 'loan_month'. Task created WITHOUT any time_col param, relies on alias match. Does NOT exercise a user-provided non-alias time_col.
- tests/test_modeling_recipes.py:1408-1760 - build_modeling_proposal unit suite (target_type/recipe/weight/categorical); none pass time_col and none assert OOT-by-time, so no coverage of the time_col->oot_by_time wiring.
- GAP: No test creates a modeling task with time_col set to a NON-alias column name (e.g. 'stmt_date') and asserts oot_by_time picks it up. Both single-file (_generate_split) and joined (auto_split_config) paths are uncovered for the non-alias case.
- GAP: No test for the conflict cases - (a) user time_col equals a column detect_setup auto-selects as split_col; (b) user time_col equals a real existing split column; (c) user time_col names a column absent from the frame (must NOT set oot_by_time, else prepare.py:171-172 raises 'missing columns').
- GAP: No test asserts the default 'apply_month' value is inert when no such column exists (must not fabricate oot_by_time='apply_month' and crash make_split).

**补丁方案（伪代码）**：
Goal: let a user's task.time_col trigger time-extrapolated OOT for NON-alias column names, while preserving current alias behavior and not crashing on absent/default values.

1) marvis/agent/turn_handlers.py::_run_modeling_setup (:659-673): add argument to the build_modeling_proposal call, mirroring vintage :529:
     time_col=getattr(task, "time_col", "") or None,

2) marvis/agent/modeling_setup.py::build_modeling_proposal signature (:145-154): add keyword-only param
     time_col: str | None = None,

3) In build_modeling_proposal, AFTER available_columns is known (:176) and business_columns computed (:177), compute a single resolved effective time column with explicit-wins-then-alias precedence (mirror vintage_setup._resolve_named_col :116-128), reusing available_columns for existence validation so a non-existent value (incl. the inert default 'apply_month' when absent) never sets oot_by_time:
     def _resolve_effective_time_col(requested, available_columns, business_columns) -> str | None:
         req = str(requested or "").strip()
         if req:
             # exact match, then case-insensitive match against real columns
             if req in available_columns: return req
             lower = {str(c).strip().lower(): str(c) for c in available_columns}
             hit = lower.get(req.lower())
             if hit: return hit
             # requested but absent -> do NOT fabricate (avoids prepare.py:171-172 crash); fall through to alias
         alias = business_columns.get("loan_month_col")
         return alias if isinstance(alias, str) and alias else None
   effective_time_col = _resolve_effective_time_col(time_col, available_columns, business_columns)

4) Replace the two alias-only lookups with effective_time_col:
   - joined branch (:267-270): time_col = effective_time_col; if time_col: auto_split_config["oot_by_time"]=time_col; ["oot_size"]=DEFAULT_OOT_SIZE (keep the else note when None).
   - single-file branch (:292): time_col=effective_time_col (pass into _generate_split, which already guards None).

5) Split-col-conflict decision (:249-253): current logic gives an auto-detected split_col unconditional precedence over any time OOT. Since detect_setup is called WITHOUT configured_split (:196-202), the split_col may be a heuristic guess. Decision to record in spec (see open_questions): EITHER (a) keep split_col precedence but, when the user explicitly supplied a time_col that is NOT the detected split_col, append a note that the detected split column overrode the requested time-OOT; OR (b) when user explicitly set time_col AND it differs from setup.split_col, prefer time-extrapolated OOT (treat explicit user intent as authoritative). Recommend (a) as the conservative default (no behavior change for existing-split datasets) unless product wants explicit-time-col to win.

6) Guard the exact-match-is-the-split-col case: if effective_time_col == setup.split_col (or a real split column), do NOT set oot_by_time on top of an existing split (the split-col branch already returns early). If effective_time_col equals target_col, treat as invalid -> fall back to None (a time col that is the label is nonsensical), optionally add a note.

7) Update the note wording at :271-279 / :615-621 in _generate_split remains valid; when effective_time_col came from the explicit user field vs alias, wording is identical (column name is interpolated), so no message-shape change needed.

**新增失败形状测试**：
- test_modeling_single_file_user_time_col_non_alias_triggers_oot: create a single-file modeling task whose date column is 'stmt_date' (NOT in _BUSINESS_COLUMN_ALIASES), pass time_col='stmt_date' at task creation; assert the 切分样本 step's split_config.oot_by_time == 'stmt_date' and the derived split has oot count > 0 and the opening note says 时间外推 OOT with 'stmt_date'.
- test_modeling_joined_user_time_col_non_alias_sets_oot_by_time: multi-file (joined) task, anchor has non-alias date col 'observe_month', pass time_col='observe_month'; assert modeling_with_join split_step.inputs.split_config.oot_by_time == 'observe_month'.
- test_modeling_user_time_col_absent_column_does_not_crash: pass time_col='no_such_col' (and no alias column present); assert NO oot_by_time is set (split_config has no oot_by_time), the plan builds, and make_split does not raise (falls back to random train/test, note says OOT n/a).
- test_modeling_default_apply_month_inert_when_absent: create task WITHOUT setting time_col (default 'apply_month') on a frame that has no apply_month/alias column; assert oot_by_time is NOT set to 'apply_month' (proves the default is inert and does not fabricate a missing-column OOT).
- test_modeling_user_time_col_alias_case_insensitive: column named 'StmtDate' with time_col='stmtdate'; assert resolution is case-insensitive and oot_by_time == 'StmtDate' (real column name preserved).
- test_modeling_user_time_col_conflicts_with_detected_split_col: frame has an auto-detectable split col (e.g. 'split' with train/test values) AND user passes time_col naming a different date col; assert the chosen resolution matches the spec decision in patch_sketch step 5 (either split_col wins with an explanatory note, or time OOT wins) - lock the behavior with an explicit assertion + note check.
- test_modeling_user_time_col_equals_target_rejected_or_ignored: time_col == target_col; assert oot_by_time is NOT set to the label column (no leakage), falls back to None/alias.

**风险 / 口径变化**：Blast radius is small: build_modeling_proposal has exactly one production caller (turn_handlers.py:659) and ~15 keyword-style unit callers; adding a keyword-only param is backward-compatible. Metric-basis change (口径变化): datasets with a user-named non-alias date column that previously got NO OOT will now get a time-extrapolated OOT split, changing train/test/oot counts and therefore all OOT-dependent metrics (OOT KS/AUC move from n/a to real numbers) - this is the intended fix but it silently changes results for existing tasks re-run after the change; call it out in the spec. Determinism (INV: deterministic metrics) is preserved because prepare._make_split ranks+quantiles deterministically with a fixed seed. Key correctness risk: task.time_col defaults to 'apply_month' and is NEVER empty (api_schemas.py:16, no user-set-vs-default distinction), so a naive pass-through would set oot_by_time='apply_month' unconditionally and crash make_split (prepare.py:171-172 raises 'missing columns') whenever no apply_month column exists - the fix MUST validate the resolved column against available_columns before setting oot_by_time (patch step 3). Second risk: split-col precedence - detect_setup runs without configured_split, so a heuristic split_col can silently shadow the user's explicit time-OOT intent; behavior must be pinned by a test (new_tests item 6) and the chosen policy documented. Third: ensure effective_time_col is added to passthrough_cols so it survives into the derived frame (single-file path already does this in _generate_split :586; joined path relies on the column being an anchor column - verify the join carries it)."

**设计决策点（提案已在补丁中给出，待审确认）**：
- Conflict policy (split_col vs explicit time_col): when detect_setup auto-detects a split_col (heuristically, since it is called without configured_split at :196-202) AND the user explicitly supplied a differing time_col, should the detected split win (conservative, no change for existing-split data) or should explicit user time_col force time-extrapolated OOT? Patch recommends split_col-wins + explanatory note; needs product sign-off.
- Should the resolution distinguish a user-EXPLICIT time_col from the inert default 'apply_month'? There is no flag separating them (api_schemas.py:16 defaults it). Current plan treats any value uniformly and relies on existence-in-columns to gate, which means a real column literally named 'apply_month' would be picked up even if the user never set it - is that acceptable (matches today's alias behavior, since 'apply_month' is already an alias) or should the default be suppressed?
- Joined path passthrough: for the joined branch the split runs later in-plan on the joined frame; confirm the user's time_col is guaranteed to be an anchor (pre-join) column and survives into the joined frame. If the time col lives only on a feature table, oot_by_time would reference a column absent at split time - should the fix validate against anchor columns specifically (available_columns here = backend.column_names(anchor path)) and refuse/warn otherwise?
- Case/whitespace normalization: should matching be exact-only, or case-insensitive (vintage _resolve_named_col is case-insensitive via lowered map)? Patch assumes case-insensitive to match vintage precedent - confirm that is desired for modeling too.

---

### D13 · screen 无 NaN 标签门 → 接入确认门

**现状（bug 机理）**：screen_features (marvis/feature/screen.py:157-364) computes all label-dependent statistics on the labeled subset silently, with no NaN-label gate. It reads base_cols=[target_col,(split_col)] via backend.read_frame (screen.py:193-195) and does target = base[target_col].to_numpy(dtype=float) (L195), then target_dev = target[dev] (L197). It never checks for non-finite labels. Every KS is computed via feature_ks(v_dev, target_dev) (screen.py:237) and feature_ks(values[train_mask], target_train) (L256-257), which route through _finite_binary_pairs (marvis/feature/metrics.py:346-356): mask = np.isfinite(scores_arr) & np.isfinite(target_arr) (L351) silently keeps only labeled rows. IV enrichment for the selected set (screen.py:342, feature_metrics -> compute_woe_iv -> feature_ks) carries the same finite mask. So with 40% NaN labels, the leakage gate (ks>=leakage_ks, L284), leakage_watch band (L299), split_shift (L289-297), ks_decay (L307), the ranking clean.sort (L327) and top_k selection (L329) ALL silently run on the labeled 60% while the tool reports selected/n_screened as if the full sample were used -- the exact silent degradation INV-1/INV-2 forbid. Both entrypoints tool_screen_features (marvis/packs/feature/tools.py:117-193; marvis/packs/modeling/feature_tools.py:59-128) call screen_features with NO drop_nan_labels arg and emit NO nan_labels_dropped. Contrast the sibling label-dependent tools that DO gate: tool_compute_feature_metrics (feature/tools.py:62-66), tool_bin_feature (feature/tools.py:306-310) both call require_labels_confirmed(...drop_nan_labels=bool(inputs.get('drop_nan_labels'))) and return nan_labels_dropped; select_features (modeling/select.py:76-78) same; tune/recipes call resolve_modeling_splits. Neither screen_features manifest entry declares drop_nan_labels/nan_labels_dropped (feature manifest.json:76-124; modeling manifest.json:408-605). The screen step templates pass no drop_nan_labels (orchestrator/templates/feature.py:181-196; modeling.py:251-276).

**关键代码位置**：
```
marvis/feature/screen.py (screen_features, current, the block that must gain the gate):
192  feats = [f for f in dict.fromkeys(features) if f != target_col]
193  base_cols = [target_col] + ([split_col] if split_col else [])
194  base = backend.read_frame(dataset_path, columns=base_cols)
195  target = base[target_col].to_numpy(dtype=float)
196  dev = _dev_mask(base, split_col, holdout_values)
197  target_dev = target[dev]
...
237          ks = feature_ks(v_dev, target_dev)
...
284          if ks >= leakage_ks:
285              leakage.append((col, ks, f"univariate KS {ks:.3f} >= {leakage_ks} ..."))

marvis/data/labels.py (require_labels_confirmed -- the gate to wire; matches feature-pack sibling tools which already use it):
28  def require_labels_confirmed(frame, target_col, *, drop_nan_labels, scope="dataset") -> int:
43      mask = nan_label_mask(frame, target_col)
44      n_nan = int(mask.sum())
45      if n_nan and not drop_nan_labels:
46          raise NanLabelNotConfirmedError(target_col=target_col, n_total=int(len(frame)), n_nan=n_nan, scope=scope)
52      return n_nan
18  def nan_label_mask(frame, target_col) -> np.ndarray:
24      values = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
25      return ~np.isfinite(values)

marvis/packs/feature/tools.py (tool_screen_features -- binary path; NO gate today):
159      result = screen_features(
160          runtime.backend, runtime.registry.resolve_path(dataset.id),
161          features=features, target_col=str(inputs["target_col"]),
...          (no drop_nan_labels passed)
171      )
172      payload = { "selected": ..., "n_screened": ..., }   # no nan_labels_dropped

marvis/packs/modeling/feature_tools.py (tool_screen_features -- binary path; NO gate today):
88       result = screen_features(runtime.backend, runtime.registry.resolve_path(dataset.id),
                 features=features, target_col=str(inputs["target_col"]), ...)   # no drop_nan_labels
102      payload = { "selected": ..., "n_screened": ..., }   # no nan_labels_dropped

marvis/plugins/subprocess_worker.py (typed-error already crosses the subprocess boundary as structured data):
647  def _structured_error_detail(exc):  # calls exc.to_detail() -> {"kind":"nan_label_not_confirmed", ...}

marvis/data/errors.py (NanLabelNotConfirmedError.to_detail):
57  def to_detail(self): return {"kind": "nan_label_not_confirmed", "target_col":..., "n_total":..., "n_nan":..., "scope":..., "by_split":...}
```

**受影响调用面**：
- marvis/feature/screen.py:157 screen_features (add drop_nan_labels param + gate); L237/L256-257 feature_ks call sites run on the 60% labeled subset today
- marvis/feature/screen.py:380 screen_features_non_binary (shares silent-drop target read at L414; decide scope -- see open_questions)
- marvis/packs/feature/tools.py:117 tool_screen_features binary path -> screen_features call at L159 must pass drop_nan_labels and echo nan_labels_dropped in payload L172
- marvis/packs/feature/tools.py:242 _screen_features_non_binary -> screen_features_non_binary call at L259
- marvis/packs/modeling/feature_tools.py:59 tool_screen_features binary path -> screen_features call at L88 must pass drop_nan_labels + echo nan_labels_dropped in payload L102
- marvis/packs/modeling/feature_tools.py:187 _screen_features_non_binary -> screen_features_non_binary call at L202
- marvis/packs/feature/manifest.json:76-124 screen_features tool schema: add drop_nan_labels to input_schema.properties (L80-92) + nan_labels_dropped to output_schema.properties (L98-114)
- marvis/packs/modeling/manifest.json:408-605 screen_features tool schema: add drop_nan_labels to input_schema (L412-461) + nan_labels_dropped to output_schema (L472-585)
- marvis/orchestrator/templates/feature.py:181-196 特征筛选 step (feature pack screen) -- no needs_confirmation today; consider a drop_nan_labels slot in inputs_template
- marvis/orchestrator/templates/modeling.py:251-276 特征筛选 step (modeling screen, needs_confirmation=True at L274) -- gate re-invoke path already exists for this step
- marvis/agent/renderers.py:66 _render_screen (add a NaN-labels-dropped line to the 特征筛选完成 summary, mirroring the excluded_categorical block at ~L82)

**现有测试与缺口**：
- tests/test_feature_screen.py -- 15 unit tests on screen_features / screen_features_non_binary (split_shift, watch-band, ks_decay, coverage note, iv_binning, psi_split, non-binary ranking). NONE feed a NaN label; all assume fully-labeled targets. This is the core gap.
- tests/test_label_nan_gate.py -- gate primitives + the failure-shape pattern to copy: test_feature_metrics_gate_raises_without_confirmation (L287-300 asserts result.ok is False, result.error_kind=='nan_label_not_confirmed', error_detail['n_nan'], error_detail['n_total']) and test_feature_metrics_gate_drops_when_confirmed (L303-320 asserts result.ok, output['nan_labels_dropped']==1) via runner.invoke(ToolRef('feature','compute_feature_metrics'),...). Also test_nan_label_error_detail_survives_subprocess (L203) proves the typed error crosses the boundary. No screen_features case exists.
- tests/test_feature_pack.py -- exercises screen_features tool via the runner but with clean labels; no NaN case.
- tests/test_modeling_pack.py -- exercises modeling screen_features tool via the runner; no NaN case.
- GAP: no test asserts screen_features raises NanLabelNotConfirmedError by default when labels are NaN, nor that drop_nan_labels=True drops rows and reports nan_labels_dropped, nor that ranking/leakage on the 60% differs from the (wrong) silent behavior.

**补丁方案（伪代码）**：
1) marvis/feature/screen.py::screen_features -- add keyword param drop_nan_labels: bool = False. Right after reading base (L194), before computing target/dev, call the gate on the base frame:
   from marvis.data.labels import require_labels_confirmed  (module-level import)
   nan_labels_dropped = require_labels_confirmed(base, target_col, drop_nan_labels=drop_nan_labels, scope="screen")
   This raises NanLabelNotConfirmedError when NaN labels exist and not confirmed. When confirmed (drop_nan_labels=True), do NOT mutate base -- the deterministic core (_finite_binary_pairs) already excludes NaN-label rows from every KS/IV, exactly like require_labels_confirmed's documented "check-only, core drops" contract used by compute_feature_metrics/bin_feature. Thread nan_labels_dropped into ScreenResult (add field default 0) and return it. Rationale: keeping the drop implicit-in-core preserves per-column missing_rate/coverage semantics (computed over v_dev which still spans all dev rows) unchanged; only label-consuming stats already dropped NaN-label rows.
2) marvis/feature/screen.py::ScreenResult -- add nan_labels_dropped: int = 0 field.
3) marvis/packs/feature/tools.py::tool_screen_features (binary path) -- pass drop_nan_labels=bool(inputs.get("drop_nan_labels")) into screen_features (L159-171); add "nan_labels_dropped": result.nan_labels_dropped to payload (L172).
4) marvis/packs/modeling/feature_tools.py::tool_screen_features (binary path) -- same: pass drop_nan_labels into screen_features (L88-101); add nan_labels_dropped to payload (L102-111).
5) Both manifests: add "drop_nan_labels": {"type":"boolean"} to each screen_features input_schema.properties and "nan_labels_dropped": {"type":"integer","minimum":0} to each output_schema.properties (both keep additionalProperties:false so the field must be declared to echo).
6) marvis/agent/renderers.py::_render_screen -- if o.get("nan_labels_dropped"): append a 中文 line "已剔除 N 行缺失标签样本(不参与筛选)" to text, mirroring the excluded_categorical block.
7) Optional (open question): screen_features_non_binary + its two tool wrappers get the same gate for parity; require_labels_confirmed works for continuous targets (nan_label_mask parses numeric, ~isfinite). Templates: modeling 特征筛选 already needs_confirmation=True so the existing gate-adjust re-invoke path carries drop_nan_labels on re-run; feature template screen step (feature.py:181-196) has post_checks=() and no confirmation -- if that path can see NaN labels it now hard-errors until re-invoked with drop_nan_labels, which is the intended INV-1 behavior.

**新增失败形状测试**：
- test_screen_features_raises_on_nan_label_by_default: build a frame where target has ~40% NaN, call screen_features(backend, path, features=[...], target_col='y') with drop_nan_labels omitted -> pytest.raises(NanLabelNotConfirmedError); assert exc.n_nan and exc.scope=='screen'. Dirty shape: partially-NaN binary label column.
- test_screen_features_drops_and_reports_when_confirmed: same frame, drop_nan_labels=True -> returns ScreenResult with nan_labels_dropped == count of NaN rows; assert ranked/selected computed only over labeled rows (KS matches feature_ks on the labeled subset). Dirty shape: 40% NaN label + confirm flag.
- test_screen_features_clean_labels_reports_zero_dropped: fully-labeled target -> nan_labels_dropped == 0 and no raise (regression guard that the gate is inert on clean data).
- test_tool_screen_features_gate_raises_without_confirmation (feature pack, mirror test_label_nan_gate.py:287): runner.invoke(ToolRef('feature','screen_features'), {dataset_id, features, target_col:'y'}) on a NaN-label sample -> result.ok is False, result.error_kind=='nan_label_not_confirmed', result.error_detail['n_nan']>0. Dirty shape: registered CSV with NaN in label.
- test_tool_screen_features_gate_drops_when_confirmed (feature pack): same invoke with drop_nan_labels=True -> result.ok, result.output['nan_labels_dropped']>0. Dirty shape: NaN label + confirm flag.
- test_tool_screen_features_modeling_gate_raises_without_confirmation (modeling pack, mirror of the above via ToolRef('modeling','screen_features')): asserts same error_kind/error_detail. Dirty shape: NaN label sample.
- test_screen_features_non_binary_nan_gate (only if scope includes non-binary): continuous target with NaN entries -> raises by default / reports nan_labels_dropped when confirmed.

**风险 / 口径变化**：Blast radius is small and mirrors an already-shipped pattern (require_labels_confirmed is used by compute_feature_metrics/bin_feature/select_features), so the typed-error + drop-flag + subprocess to_detail() plumbing and the gate-adjust re-invoke path already exist and are proven by tests/test_label_nan_gate.py. Metric-basis (口径) note: this is NOT a口径 change on clean data -- with no NaN labels the gate returns 0 and every existing screen output is byte-identical (ranking, selected, scores unchanged), so tests/test_feature_screen.py passes unmodified. The口径 change is the intended one: on a NaN-labeled sample the tool now STOPS by default instead of silently ranking on the labeled subset. Backward-compat: default drop_nan_labels=False makes previously-passing runs on NaN-labeled samples fail until confirmed (the INV-1/INV-2 fix); modeling template screen step is already needs_confirmation=True so it fits the gate-adjust flow, but the feature template screen step is not a confirmation gate today, so a NaN-labeled feature-pack screen would hard-error and require re-invoke with drop_nan_labels. Manifests use additionalProperties:false, so payload MUST declare nan_labels_dropped or the runner rejects the output -- schema and tool must be edited together. No determinism risk: gate is a pure count/mask, no RNG.

**设计决策点（提案已在补丁中给出，待审确认）**：
- Scope of non-binary: should the gate also wire into screen_features_non_binary (screen.py:380) and its two wrappers? It reads target as float (L414) and silently drops NaN via association-score computation, so the same silent-degradation exists for regression/multiclass screens. The item title/read-list is framed around the binary KS path; recommend including non-binary for parity but flag as a scope decision.
- Feature-pack screen step (orchestrator/templates/feature.py:181-196) has post_checks=() and is NOT a needs_confirmation gate, unlike the modeling screen step (modeling.py:274 needs_confirmation=True). If a NaN-labeled dataset reaches the feature-pack screen, the new default will hard-error with no in-band confirm/adjust affordance until the caller re-invokes with drop_nan_labels. Confirm whether the feature template screen step should also carry a drop_nan_labels slot / gate, or whether hard-error-then-retry is acceptable there.
- The memory-write auto_distill paths are in marvis/pipeline.py (_capture_agent_memory_for_metrics_success/_failure, gated on load_memory_policy(...).auto_distill at L1202/L1238) and marvis/agent/memory_bridge.py (L87, join-outcome capture) -- NOT marvis/agent/api.py + orchestrator/pipeline.py as the memory note phrased it. Both run post-hoc on task metrics/join outcomes and never call screen_features; they are IRRELEVANT to this fix (no changes needed).
- scope string for the error: propose scope='screen' so the gate detail is distinguishable from 'dataset'/'train/test'; confirm the gate UI/renderer copy doesn't key off a specific scope literal (renderers.py:_render_screen currently renders no scope).

---

### D14 · refit 5% carve 的 test_ks 进 headline → 剔出并保留 pre-refit 指标

**现状（bug 机理）**：The TUNE-4 champion refit retrains on train+test but must satisfy split_modeling_frame's non-empty-test contract, so _refit_champion_on_train_plus_test (train_tools.py ~L664-681) carves a deterministic random 5% of the combined rows back out and labels it '__refit_holdout__', wiring scratch_split_values so that slice becomes the recipe's "test". Because that holdout is a random 5% drawn from the same train+test population (NOT a time-based OOT holdout), it is fully in-distribution with the training data, so any KS/AUC/PSI on it is optimistically biased and near-meaningless.

_train_recipe -> compute_model_metrics (recipes/common.py L404-479) then computes the FULL test_* family on that slice: test_ks, test_auc, psi_test_vs_train, test_ks_ci_low/high/std, test_lift_head/tail_5/10, weighted_test_ks/auc/psi_test_vs_train. The code comment at train_tools.py L666-667 claims this slice "is never fit on and never reported -- only train/OOT metrics from this refit are surfaced," but the "never reported" half is FALSE.

The refit result's ModelMetrics (carrying these optimistic test_* values) is persisted via runtime.experiments.attach_result (select_tools.py L262). _apply_champion_refit (select_tools.py L268-279) then builds post_metrics from runtime.experiments.compare([refit_experiment_id]) filtered ONLY by _is_metric_key, which accepts every 'train_'/'test_'/'oot_'/'psi_'/'weighted_' prefix plus 'overfit_flag' (_common.py L85-86), and returns ALL of them as the headline "metrics". select_experiment sets final_metrics = refit_info["metrics"] (select_tools.py L150) and returns it as the tool's headline "metrics" (L162).

Three real consumers surface these bogus refit test_* numbers as if they were a genuine held-out test evaluation:
(1) renderers.py _render_select_experiment L475-481 blindly renders EVERY key in the headline metrics dict into the "最终模型指标" table.
(2) The model card reads the refit experiment.metrics structurally (delivery_tools.py L789) and _model_card_key_metrics (L864-896) lists test_ks/test_auc/psi_test_vs_train/weighted_test_ks/weighted_test_auc.
(3) The monitoring-policy baseline_metrics (delivery_tools.py L356 -> _monitoring_check_payload L415-417) feeds the default psi_test_vs_train check (delivery_tools.py L30-36); for the refit that PSI is feature_psi(__refit_train__, __refit_holdout__) which is ~0 (same distribution), so the check reads green off a meaningless baseline.

What is SAFE and must be preserved: the monitor dev-reference read at monitor_tools.py L415-416 uses dev_metrics.train_ks / train_auc (NOT test_*), so train_* keys must remain. The champion-selection basis test_ks(overfit-penalized) is computed BEFORE refit on the real train-split champions (select_tools.py L139 pre_refit_metrics, and _overfit_penalized_test_ks in tune-time), so suppressing test_* from the REFIT metrics only does not change selection. metrics_before_refit already carries the honest pre-refit train/test/oot (select_tools.py L280) and must stay untouched.

**关键代码位置**：
```
train_tools.py L664-681 (the holdout carve + false comment):
```
# ...this slice is never fit on and never reported -- only train/
# OOT metrics from this refit are surfaced).
combined_idx = frame.index[train_mask | test_mask]
rng = np.random.RandomState(int(config.seed))
shuffled = combined_idx.to_numpy().copy()
rng.shuffle(shuffled)
holdout_n = max(1, min(len(shuffled) - 1, round(len(shuffled) * 0.05)))
scratch = frame.copy()
scratch[split_col] = scratch[split_col].astype(object)
scratch.loc[combined_idx, split_col] = "__refit_train__"
scratch.loc[shuffled[:holdout_n], split_col] = "__refit_holdout__"
scratch_split_values = {"train": "__refit_train__", "test": "__refit_holdout__", ...}
```

select_tools.py L272-279 (post_metrics returns ALL test_* keys as headline):
```
post_metrics = {k: v for k, v in refit_row.items() if _is_metric_key(k) and v is not None}
return {
    "applied": True, "requested": True,
    "reason": "已用冠军的定型超参在 train+test 全量重训,OOT 未参与训练,指标为重训后模型的 OOT/train 结果。",
    "experiment_id": refit_experiment_id,
    "artifact_id": refit_experiment.artifact_id,
    "metrics": post_metrics,
```

select_tools.py L150 / L162 (post_metrics becomes tool headline):
```
final_metrics = refit_info.get("metrics") or pre_refit_metrics
...
"metrics": final_metrics,
```

_common.py L85-86 (the over-broad filter that lets test_* through):
```
def _is_metric_key(key: str) -> bool:
    return key.startswith(("train_", "test_", "oot_", "psi_", "weighted_")) or key == "overfit_flag"
```

delivery_tools.py L789 + L864-896 (model card reads refit experiment.metrics structurally, surfaces test_ks/test_auc/psi_test_vs_train/weighted_test_*).
monitor_tools.py L415-416 (SAFE — reads dev_metrics.train_ks/train_auc, not test_*).
```

**受影响调用面**：
- marvis/packs/modeling/select_tools.py:150 select_experiment sets final_metrics = refit_info.get('metrics') then L162 returns it as headline 'metrics'
- marvis/packs/modeling/select_tools.py:272 _apply_champion_refit builds post_metrics via _is_metric_key over the refit compare() row
- marvis/agent/renderers.py:475-481 _render_select_experiment renders every key of headline metrics into '最终模型指标' table
- marvis/packs/modeling/delivery_tools.py:789 _model_card_payload reads metrics = _json_safe(experiment.metrics) (refit experiment) -> L828 _model_card_key_metrics
- marvis/packs/modeling/delivery_tools.py:864-896 _model_card_key_metrics lists test_ks/test_auc/psi_test_vs_train/weighted_test_ks/weighted_test_auc
- marvis/packs/modeling/delivery_tools.py:356 _monitoring_policy_payload baseline_metrics = _json_safe(experiment.metrics) -> default psi_test_vs_train check L30-36
- marvis/packs/modeling/experiment.py:113-177 _experiment_row/compare flattens ModelMetrics test_* fields (the dict post_metrics is filtered from)
- marvis/packs/modeling/monitor_tools.py:415-416 dev-reference read uses dev_metrics.train_ks/train_auc ONLY (must remain intact)

**现有测试与缺口**：
- tests/test_modeling_pack.py:1868 test_select_experiment_refits_champion_on_train_plus_test_by_default — asserts refit applied and (L1919-1920) that metrics_before_refit AND metrics_after_refit intersect {train_ks,test_ks,oot_ks}. This test WILL BREAK if we strip test_ks from metrics_after_refit; must be updated to expect refit_holdout_test_ks (or drop test_ks from the after set and keep train_ks/oot_ks)
- tests/test_modeling_pack.py:1931 test_select_experiment_refit_attach_failure_rolls_back_unattached_artifact_files — rollback path, unaffected by metric renaming
- tests/test_modeling_pack.py:2022 test_select_experiment_refit_on_train_plus_test_false_keeps_original_champion — refit disabled path, headline metrics = pre_refit_metrics (honest test_ks), must remain unchanged
- tests/test_modeling_pack.py:1562 / 1658 assert selection_metric == 'test_ks(overfit-penalized)' — selection basis unaffected (pre-refit)
- tests/test_modeling_pack.py:1294 / 1308 assert result.output['metrics']['test_ks'] is None and row['test_ks'] is None — normal (non-refit) train path; test_ks must stay a first-class key there

**补丁方案（伪代码）**：
Goal: keep the refit's holdout-derived metrics as INTERNAL diagnostics (renamed refit_holdout_*), suppress them from every headline surface, and preserve train_*/oot_* (honest for the refit) and the pre-refit honest test_*. Do NOT mutate ModelMetrics dataclass (its structural test_* fields are read by ~40 consumers). Fix at the dict/presentation boundary.

1) select_tools.py _apply_champion_refit (~L268-284): after building refit_row, transform the refit's test-family keys.
   - Add module-level constant set of "holdout-tainted" prefixes/keys:
     _REFIT_HOLDOUT_TAINTED = lambda k: k == "psi_test_vs_train" or k == "weighted_psi_test_vs_train" or k.startswith(("test_", "weighted_test_"))
     (this captures test_ks, test_auc, test_ks_ci_*, test_lift_*, weighted_test_ks/auc, and the two test-vs-train PSIs; leaves train_*, oot_*, weighted_train/oot_*, psi_oot_vs_train, overfit_flag intact).
   - Build post_metrics WITHOUT tainted keys:
       post_metrics = {k: v for k, v in refit_row.items() if _is_metric_key(k) and v is not None and not _REFIT_HOLDOUT_TAINTED(k)}
   - Capture the tainted ones under a renamed internal bucket:
       refit_holdout_diag = {f"refit_holdout_{k}": v for k, v in refit_row.items() if _is_metric_key(k) and v is not None and _REFIT_HOLDOUT_TAINTED(k)}
   - Return dict: add "refit_holdout_metrics": refit_holdout_diag (internal diagnostic, NOT merged into headline). Keep "metrics": post_metrics (now train/oot only), metrics_before_refit unchanged (honest pre-refit).
   - Update the "reason" string to state the reported metrics are OOT + train from the refit and that a random holdout slice's metrics are recorded separately as diagnostics only.

2) train_tools.py L666-667: fix the now-accurate comment to say the slice's test-family metrics ARE computed but are relabeled refit_holdout_* and excluded from headline/model-card/monitoring (remove the false "never reported").

3) Model card — delivery_tools.py: the model card reads experiment.metrics structurally (L789), so it does not go through post_metrics. Add refit-awareness: detect refit via config param REFIT_ON_TRAIN_PLUS_TEST_PARAM_KEY (already on refit_config.params, train_tools.py L699) — pass an is_refit flag into _model_card_payload / _model_card_key_metrics. When is_refit, drop test_ks/test_auc/psi_test_vs_train/weighted_test_ks/weighted_test_auc from the key-metrics list (they came from the random holdout) OR relabel them "refit_holdout_test_ks" with a footnote in _model_card_limitations explaining the holdout is in-distribution and not a genuine held-out evaluation. Simplest: filter them out of _model_card_key_metrics rows when is_refit.

4) Monitoring baseline — delivery_tools.py L356 _monitoring_policy_payload: when the experiment is a refit, the psi_test_vs_train baseline (L30-36 default check) is meaningless (~0, same distribution). Set that check to "missing"/skip by dropping psi_test_vs_train from baseline_metrics for refit experiments (its value flows via metrics.get(metric) at L417 -> "missing" status L437-438), or omit the psi_test_vs_train check from thresholds when is_refit. Prefer dropping the key so the existing "missing" handling reports it honestly.

5) renderers.py _render_select_experiment L475-481: no change needed if step 1 already removed test_* from headline metrics — the table iterates o["metrics"] which will now be train/oot only. Optionally add a small note that refit test metrics are diagnostic-only when refit.applied.

**新增失败形状测试**：
- Update tests/test_modeling_pack.py:1919-1920: change metrics_after_refit assertion to expect test_ks NOT in headline (assert 'test_ks' not in refit['metrics_after_refit']) while train_ks/oot_ks remain (assert {'train_ks','oot_ks'} & set(refit['metrics_after_refit']))
- New test test_refit_headline_metrics_exclude_holdout_test_family: assert select_experiment headline output['metrics'] for a refit contains train_ks/oot_ks but NOT test_ks/test_auc/psi_test_vs_train/weighted_test_ks; assert refit['refit_holdout_metrics'] contains refit_holdout_test_ks
- New test test_refit_model_card_excludes_holdout_test_metrics: deliver a refit artifact, assert _model_card_key_metrics rows contain oot_ks/train_ks but no bare test_ks/test_auc/psi_test_vs_train (or that they appear only under a refit_holdout_/diagnostic label)
- New test test_refit_monitoring_baseline_omits_psi_test_vs_train: assert the monitoring policy's psi_test_vs_train check for a refit experiment is status 'missing'/skipped rather than green off a ~0 same-distribution PSI
- New test test_non_refit_path_still_reports_test_metrics: guard against over-suppression — a normal (refit disabled, L2022 style) select_experiment still surfaces honest test_ks in headline metrics and model card
- New test test_refit_selection_metric_unchanged: assert selection_metric stays 'test_ks(overfit-penalized)' and champion choice is unchanged (selection uses pre-refit train-split metrics, not the refit holdout)

**风险 / 口径变化**：Blast radius is contained if the fix stays at the dict/presentation boundary. Do NOT rename ModelMetrics structural fields — experiment.py _experiment_row (L142-177), report_tools.py L348/L390, and the model card all read .test_ks/.psi_test_vs_train directly; renaming there would break dozens of non-refit call sites and persisted-metric readers.

Main risk = OVER-suppression: the same code path (_apply_champion_refit -> post_metrics) only runs for refits, and non-refit select_experiment returns pre_refit_metrics (honest test_*), so guard tests must confirm normal training/selection still surfaces test_ks (existing tests at L1294/L1308/L1562 protect this).

Consumer break to fix in-PR: existing test at L1919-1920 asserts test_ks IS in metrics_after_refit — it will fail and MUST be updated as part of this change (it currently encodes the buggy behavior).

Monitoring-baseline change (dropping psi_test_vs_train for refit) is a behavior change a monitoring consumer could notice — it flips that check from a spuriously-green baseline to 'missing', which is more honest but should be called out in the reason/limitations text.

Determinism preserved: the holdout carve stays deterministic (seed-derived), we only change how its metrics are labeled/surfaced, so no metric-value or artifact-hash changes — only the headline/model-card/monitoring dictionaries change key names. train_*/oot_* on the refit remain genuinely valid (refit was trained on train+test, OOT untouched) and stay in the headline, so the caller still sees a real before/after OOT comparison (oot_ks_before/after_refit unchanged).
