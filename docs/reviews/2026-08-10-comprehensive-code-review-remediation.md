# 2026-08-10 全仓代码复审与整改记录

**日期：** 2026-08-10

**方式：** 主流程整改，多个独立 subagent 只读复核；未暂存、未提交、未发布。
**范围：** HTTP/共享主机边界、Draft 生成与转正、插件运行时及历史迁移、URL 抓取、任务/输出事务、数据聚合、前端代理挂载路径、依赖与静态门禁。

## 结论

本轮发现的可证明 P0/P1/P2 已全部有代码修复和定向回归。整改重点不是继续累积字符串黑名单，而是把信任边界改为可验证的凭据、受限语言、持久身份收据和原子资源所有权。

最终全仓 pytest 已在本次工作树的生产代码快照上完成：**12,208 passed、11 skipped、32 warnings，耗时 1:55:07**。完成后收紧的自动规则树测试文件已另行整文件验证，详见下方证据。

## 已关闭的高优先级问题

### CR-01 / P0：共享主机上的 loopback 不再等同于已认证用户

设置 `MARVIS_LOCAL_TOKEN` 时，过去匿名 loopback 首页会把本地及插件管理 bearer token 写入 HTML；共享主机的另一进程能够继承私有读写和插件管理能力。

- [`marvis/app.py:199`](../../marvis/app.py#L199) 只把已配置的直接代理对端作为转发边界；XFF 本身不授予信任。
- [`marvis/app.py:497`](../../marvis/app.py#L497) 要求私有读取用 Basic（token 为密码）或 `X-Marvis-Token`，写请求仅接受显式 `X-Marvis-Token`，避免浏览器缓存 Basic 后形成跨站写入面。
- 已认证私有响应与首页使用 `no-store`；首页同时 `Vary: Authorization`，不会把 token 交给未认证或远端读取者。
- [`docs/runbook.md`](../runbook.md) 和 [`docs/deploy-linux-env-checklist.md`](../deploy-linux-env-checklist.md) 同步说明 Basic bootstrap、显式写 token、可信代理覆写 XFF 与 HTTPS 要求。

独立复核未发现私有 GET/HEAD、unsafe 请求、可信代理或缓存语义的高置信矛盾；`tests/test_app_security.py` 46 项通过，认证/API 组合为 75 项通过。

### CR-02 / P1：Draft 不再依赖 Python 黑名单，并在转正后保持限制

原 Draft AST gate 允许语言结构和已加载模块的全局对象绕出受限 builtins；转正后又会作为普通插件运行，丢失 Draft 限制。

- 新增 [`marvis/draft_language.py:181`](../../marvis/draft_language.py#L181)，以 Draft Language v1 显式验证语法、名称、属性、调用和有限能力表。
- Draft 源在移除 import 的受控 AST 上执行，使用最小 DraftContext；运行时不向其提供原始 importer 或完整宿主上下文。
- [`marvis/plugins/manifest.py:266`](../../marvis/plugins/manifest.py#L266) 的 `DraftPromotionReceipt` 将 draft、插件名称/版本/校验值、工具和 execution profile 绑定起来。
- [`marvis/plugins/registry.py:216`](../../marvis/plugins/registry.py#L216) 对历史工件按收据或可证明的时间线迁移；来源不明确的旧 Draft 标记为 `draft_repromotion_required`，在 resolve、planner 与 runner 中均不可执行。

这是一套严格的小语言与运行时能力边界，不应被表述为针对敌意任意 Python 的 OS/container 隔离。需要执行任意第三方 Python 时，仍应使用进程/容器级隔离与最小宿主权限。

### CR-03 / P1：验证“需要复核”不再被写为成功

[`marvis/pipeline.py:1399`](../../marvis/pipeline.py#L1399) 现在只有一致性 PASS 才会使终态成功；REVIEW 与 FAIL 都进入 `REVIEW_REQUIRED`。相应的可复现性回归覆盖了小幅不一致的 REVIEW 状态。

### CR-04 / P2：URL 抓取的 SSRF、重绑定与响应体上限收紧

[`marvis/drafts/web_search.py:51`](../../marvis/drafts/web_search.py#L51) 只接受 HTTP(S)，拒绝凭据和非全局/混合解析地址，并在重定向与连接阶段保持已验证地址的绑定；它禁用环境代理、限制跳转次数、限制声明及实际流式响应大小，并拒绝压缩内容以避免解压后限额失真。

定向 Draft/Web 测试覆盖私有地址、混合 DNS、重定向、重绑定、响应体上限和正常公开地址控制路径。

## 已关闭的核心正确性与一致性问题

- `slice_aggregate` 现在要求源列排序键同时属于 `group_by`，或是已选择指标；避免 DuckDB 在 `GROUP BY` 后接受非法 `ORDER BY`。见 [`marvis/agent/adhoc_analysis.py:219`](../../marvis/agent/adhoc_analysis.py#L219) 与 [`marvis/packs/data_ops/tools.py:2265`](../../marvis/packs/data_ops/tools.py#L2265)。
- 阶段失败后的记忆降级已经隔离：SQLite/记忆记录再失败也不会替换 notebook、指标或报告的原始失败原因。见 [`marvis/pipeline.py:2131`](../../marvis/pipeline.py#L2131)。
- 验证 Excel 的图片暂存目录纳入父 `ArtifactUnitOfWork`；父事务回滚不会留下 `.staging/excel_images`。见 [`marvis/pipeline.py:1283`](../../marvis/pipeline.py#L1283) 与 [`marvis/output/excel.py:45`](../../marvis/output/excel.py#L45)。
- 自动规则树异常路径的回归不再比较整个长寿 pytest 进程的 FD 总数，而是直接验证本测试创建的 source FD、snapshot FD 与 Parquet reader 都恰好关闭；成功、schema 失败、执行失败及 ABA 路径均覆盖。未改生产关闭逻辑，因为受控 500 次检查和 FD 清单没有证明生产泄漏。

## 前端与部署兼容性

前端此前把 `/api/...` 作为 origin-root URL，在 JupyterHub 的 `/user/.../proxy/.../` 挂载下会跳出应用前缀。

- API wrapper、XHR、下载链接、iframe 预览、Plan Rail 和 Markdown 中经验证的同源 API 链接都改为相对应用基址。
- [`marvis/static/js/url-safety.js`](../../marvis/static/js/url-safety.js) 先做同源 API 校验，再返回 app-relative 地址；[`marvis/static/js/render-agent.js:296`](../../marvis/static/js/render-agent.js#L296) 对 Markdown 使用同一规则。
- 受影响前端回归 418 项通过，并加入 `document.baseURI` 为 JupyterHub proxy 前缀的解析契约。

## 验证证据

- 最终全仓：`conda run --no-capture-output -n py_313 python -m pytest -q -x`：**12,208 passed、11 skipped、32 warnings，耗时 6,907.68s**。warning 为既有 LightGBM API 弃用、固定预算 MLP 未收敛和兼容采样 API 弃用；没有失败或 error。
- 安全、Draft、插件、pipeline、数据聚合、前端和审计相关的组合 pytest：**855 passed**。
- 自动规则树文件：**26 passed**；其中资源关闭目标用例与 ABA 用例均已单独通过。
- 受影响前端回归：**418 passed**；另有 candidate/static/artifact 组合 **372 passed**。
- Ruff（`marvis` 与 `tests`）、64 个静态 JS 的 `node --check`、`git diff --check`、`uv lock --check --offline`、Bandit 基线和 `scripts/check --fast --audit --skip-pytest`：均 exit 0。
- `pip-audit` 使用锁定依赖而非环境中任意 conda 包；本次报告无已知漏洞。

## 剩余边界与后续建议

1. **共享主机部署是配置契约。** 需要保密工作区时必须设置 `MARVIS_LOCAL_TOKEN`；JupyterHub/反向代理必须由可信代理覆写转发头并用 HTTPS。`MARVIS_ALLOW_REMOTE_READ` 是明确降低私有读取边界的运维选择，不适用于需要同机租户隔离的场景。
2. **Draft 兼容性是有意收紧。** 无法证明来源的历史 Draft 会要求重新转正；这是为了避免默认回退为 unrestricted plugin。上线前应盘点这些条目并安排重写/重新转正。
3. **校验值语义可继续硬化（P3）。** 转正路径的 receipt 校验值在 `manifest.json` 写入前计算，因而并非最终目录的全树哈希。它不构成本轮已验证的执行绕过（restricted loader 仍会验证源/profile），但若该字段要承担“完整部署工件哈希”的含义，应单独设计不可自指的 canonical manifest/source hash。
4. **仍需对应环境验收。** 本地回归不能代替真实 JupyterHub、真实反向代理、Windows 原生文件语义、真实材料与生产责任人签字。

## 判定

**PASS（本地代码与自动化门禁）。** 当前未留存已知未关闭的 P0/P1/P2；上述部署和隔离边界不应被误报为“已在本地自动验收”。
