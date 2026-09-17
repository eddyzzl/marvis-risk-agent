# 文档-代码一致性核验（architecture.md / capability-status.md）

- 日期：2026-09-03
- 范围：`docs/architecture.md` 全文可机械核验的规模断言与代码语义断言；
  `docs/capability-status.md` 中按模块声称的单测数量
- 结论性质：**只读审计**。未改动任何代码，唯一产物是本文件
- 结论：34 项断言中 **32 项精确命中，2 项不准确**（均位于 `architecture.md` §5.1）

## 1. 核验基准：三态对比

核验前先确定「文档写作时看到的是哪份代码」，否则无法区分两种性质完全不同的偏差。

```text
git log --diff-filter=A -- docs/architecture.md
  → a3580695  (docs: record 90-day plan execution status, architecture, audit, capability matrix)
git branch --show-current
  → feat/validation-module-upgrade  (HEAD = 45c83aa2，在 main 之上 1 个 doc-only commit)
```

`a3580695` 既是两份文档的引入 commit，也正是 `main` 的 HEAD。因此：

| 对比 | 偏差性质 |
|---|---|
| 文档值 ≠ HEAD 值 | **文档当初就写错了** |
| HEAD 值 ≠ 工作区值 | 未提交改动导致的**代码漂移**，文档无过 |
| 文档值 = HEAD 值 = 工作区值 | 一致 |

本报告所有判定均按此基准。注意当前工作区有 69 个文件、+4477/−653 行未提交改动
（分支 `feat/validation-module-upgrade`），因此「工作区值」与「文档值」的差异不代表文档失真。

## 2. 精确命中清单（32 项）

### 2.1 文件规模断言（11/11 在 HEAD 上一致）

| 文件 | 文档称 | HEAD | 工作区 | 判定 |
|---|---:|---:|---:|---|
| `marvis/__main__.py` | 644 | 644 | 644 | 一致 |
| `marvis/app.py` | 903 | 903 | 903 | 一致 |
| `marvis/api.py` | 186 | 186 | 186 | 一致 |
| `marvis/db.py` | 47 | 47 | 47 | 一致 |
| `marvis/db_schema.py` | 4609 | 4609 | 4609 | 一致 |
| `marvis/pipeline.py` | 2324 | 2324 | **2351** | HEAD 一致，工作区 +27 |
| `marvis/plugins/runner.py` | 2175 | 2175 | 2175 | 一致 |
| `marvis/notebooks.py` | 1810 | 1810 | 1810 | 一致 |
| `marvis/notebook_worker.py` | 84 | 84 | 84 | 一致 |
| `marvis/notebook_contract.py` | 747 | 747 | 747 | 一致 |
| `marvis/static/app.js` | 8382 | 8382 | **8858** | HEAD 一致，工作区 +476 |

### 2.2 §6 拆分债务的车道行数（5/5，文档给的是约数，实测全部落在约数内）

| 车道文件 | 文档约数 | 实测 |
|---|---:|---:|
| `turn_handlers/strategy_turns.py` | ~3.0k | 3014 |
| `turn_handlers/strategy_candidates.py` | ~4.5k | 4551 |
| `strategy_request_compiler/core.py` | ~2.7k | 2704 |
| `strategy_request_compiler/pool.py` | ~2.5k | 2549 |
| `strategy_request_compiler/tree.py` | ~2.4k | 2401 |

### 2.3 前端与字节数（4/4）

| 项 | 文档称 | HEAD 实测 |
|---|---:|---:|
| `static/js/` 模块 / 行数 | 31 / 6409 | 31 / 6409 |
| `static/js/v2/` 模块 / 行数 | 32 / 18937 | 32 / 18937 |
| `llm_prompts.py` 字节 | 77,089 | 77,089 |
| `api_schemas.py` 字节 | 78,995 | 78,995 |

### 2.4 代码语义断言（12/12）

| 文档断言 § | 代码位置 | 判定 |
|---|---|---|
| §4.1 RMC 四契约（`RMC_SAMPLE_DF`/`RMC_TARGET_COL`/`RMC_ALGORITHM`/`RMC_SCORE_FN`） | `notebook_contract.py:15-18` | 一致 |
| §4.1 `RMC_FEATURES` 已废弃 | 契约列表中已无该项 | 一致 |
| §4.3 legacy live kernel 三重 opt-in | `pipeline.py:233-236`，三项逻辑等价且缺一不可 | 一致 |
| §5.1 `NumberProvenance` 四元组 | `provenance.py:73-79` | 一致 |
| §5.2 `CANONICAL_RESULT_TOOLS` = 8 个 presenter | `canonical_results.py:23-34`，恰 8 项 | 一致 |
| §5.3 `MEMORY_CANDIDATE_TEXT_MAX_CHARS = 12000` | `agent_memory/policy.py:122` | 一致 |
| §5.3 `PAYLOAD_FIELD_ALLOWLISTS` + 两个 classify 函数 | `policy.py:49/125/141` | 一致 |
| §5.4 `compute_platform_validation_results` | `validation/platform_metrics.py:445` | 一致 |
| §3.1 v1_compat 四个 tool 函数 | `packs/v1_compat/tools.py:23/37/52/60` | 一致 |
| §3.1 `MODEL_VALIDATION` 四步顺序 | `templates/validation.py:19-63`，scan → run_notebook → compute → render | 一致 |
| §3.1 模板 post_checks（status 值域 / ks,auc 0..1 / psi allow_null） | `templates/validation.py:46-49` | 一致 |
| §1.1 注册 27 个 `include_router` | `app.py:609-635`，恰 27 次 | 一致 |

### 2.5 测试数量断言（6/6）

| 模块 | 文档称 | AST 数函数 | `pytest --collect-only` | 判定 |
|---|---:|---:|---:|---|
| `sql_ingest.py` | 22 | 13 | **22** | 一致 |
| `historical_backtest.py` | 15 | 15 | — | 一致 |
| `leakage_diagnostics.py` | 12 | 12 | — | 一致 |
| `limit_pricing_impact.py` | 20 | 12 | **20** | 一致 |
| `fairness_evidence.py` | 14 | 14 | — | 一致 |
| `counterfactual.py` | 10 | 10 | — | 一致 |

> **参数化陷阱**：`sql_ingest` 与 `limit_pricing_impact` 分别含 3 处和 2 处
> `@pytest.mark.parametrize`。仅用 AST 统计 `test_*` 函数会得到 13 和 12，
> 误判为「文档多报 9 条 / 8 条」。经 `pytest --collect-only` 实际收集，
> 参数化展开后恰为 22 和 20，与文档完全一致。**核验含参数化的测试数量必须使用 collect-only。**

### 2.6 「12135 passed」量级交叉校验

全库 AST 统计：687 个测试文件、8427 个 `test_*` 函数、928 处 parametrize。
按每处 parametrize 平均展开 5 例反推：`8427 − 928 + 928 × 5 ≈ 12139`，与文档声称的
12135 passed 吻合。此为量级校验，二者不等价（skipped / failed / 参数化实际展开数均会影响）。

## 3. 不一致项（2 项，均在 `architecture.md` §5.1）

### 3.1 `plugins/runner.py` 并未使用 `hmac.compare_digest`

文档原文：

> 同样的「重哈希 + `hmac.compare_digest`」出现在 `routers/artifacts.py`、
> `download_snapshot.py`、`artifacts/model_score_vector.py`、**`plugins/runner.py`**
> （`_payload_hash` / `_manifest_receipt_hash` / `result_hash`）。

实测：全库 `hmac.compare_digest` 命中 100+ 个文件，但 `plugins/runner.py` **不在其中**；
该文件内既无 `import hmac` 也无任何 `compare_digest` 调用。

实际实现（`runner.py:2115-2133`）：

```python
def _result_payload_hash(output: dict) -> str: ...
def _manifest_receipt_hash(manifest: PluginManifest) -> str:
    return _result_payload_hash(manifest_to_dict(manifest))
```

性质判定：这是**评估不足（understatement）而非过度声明**。`runner.py` 做的是「计算并
记录哈希」（`_result_payload_hash`、`_manifest_receipt_hash`、`result_hash`），属于
**生成**证据；`compare_digest` 是**比对**两个哈希值时的恒定时间比较，防时序侧信道。
两者是不同性质的操作，Runner 只做前者，本不需要后者。文档把「重哈希」与
「`hmac.compare_digest`」合并成一种模式后套用到 Runner 上，属于归类不当。

**建议修法**：将 `plugins/runner.py` 从该句 `compare_digest` 的列举中移出，
或改写为「Runner 对每次调用计算并绑定 `_result_payload_hash` / `_manifest_receipt_hash` /
`result_hash`，供下游在比对时使用 `hmac.compare_digest`」。

### 3.2 函数名写作 `_payload_hash`，实际为 `_result_payload_hash`

同上引文。实际函数名是 `_result_payload_hash`（`runner.py:2115`）。
（`_manifest_receipt_hash` 与 `result_hash` 两处名称正确。）

**建议修法**：改为 `_result_payload_hash`。

## 4. 工作区漂移（文档无过，但影响 capability-status 结论）

| 文件 | HEAD | 工作区 | 漂移 |
|---|---:|---:|---:|
| `static/app.js` | 8382 | 8858 | +476 |
| `static/js/` | 31 模块 / 6409 行 | 33 模块 / 7540 行 | +2 模块 / +1131 行 |
| `marvis/pipeline.py` | 2324 | 2351 | +27 |
| `marvis/llm_prompts.py` | 77089 B | 77696 B | +607 B |

**需要单独指出**：`capability-status.md` 的「B-6 前端单体收敛 ✅ 完成」一条，
在当前工作区已被实质性推翻——

```text
B-6 拆分行动之前   app.js = 8,465
B-6 之后 (main)    app.js = 8,382
当前工作区          app.js = 8,858   （未提交改动 +653 / −177）
```

即本次未提交的验证模块升级让前端单体净增 476 行，**超过了收敛行动之前的 8,465 行**。
本报告不改文档，仅提示：该结论在提交这批改动前需要重新评估并更新。

## 5. 本轮核验的局限

1. 未跑全量 `pytest`（12135 用例）验证「完整本地 gate 退出码 0」，只做了量级交叉校验。
2. 未逐条核验 `capability-status.md` 中 `VERIFIED` 分层的**证据充分性**（那是判断问题，
   不是机械核验问题），仅核验了其中可机械核对的测试数量。
3. 未核验 `roadmap.md` 的产品范围断言——那是产品决定，不是代码事实。
4. 代码语义核验采用「符号存在 + 函数体阅读」，未做行为级运行验证。
5. `hmac.compare_digest` 全库 100+ 命中未逐一核对，仅核验了文档点名的 5 个文件中的
   `plugins/runner.py`（发现不符）；其余 4 个（`packs/strategy/report_bundle.py`、
   `routers/artifacts.py`、`download_snapshot.py`、`artifacts/model_score_vector.py`）
   均在全库命中列表内，未发现不符。

## 6. 方法论建议

1. **逻辑断言必须读完整函数体**。本轮曾据 `pipeline.py:236` 的
   `return _truthy_env(LEGACY_LIVE_NOTEBOOK_ENV_VAR)` 单行，误判三重 opt-in 只验了环境变量
   一项；读函数体后确认第 234 行的短路条件已覆盖前两项。grep 单行输出不足以判定逻辑。
2. **含参数化的测试数量用 `pytest --collect-only`**，AST 数函数会系统性低估。
3. **跑项目测试用绝对路径** `/opt/miniconda3/envs/py_313/bin/python -m pytest`。
   `conda run -n py_313 python` 会解析到 workbuddy managed python（无 pytest）。
