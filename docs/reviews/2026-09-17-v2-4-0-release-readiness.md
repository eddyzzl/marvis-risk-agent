# V2.4.0 发布候选验证记录

> 日期：2026-09-17。结论：发布前完整本地门禁通过。
> 本文记录本地候选证据；远端发布与 CI 状态应以对应 tag、commit 和 Actions 结果为准。

## 范围

- 纳入已合并的 [issue #23 修复 PR #24](https://github.com/eddyzzl/marvis-risk-agent/pull/24)：认证快照先发布、后设为只读，保留 Windows 权限行为回归。
- 纳入本地尚未公开的只读 SQL 接入、确定性策略分析内核、受限反事实重放、Agent 模块拆分及其治理、兼容性测试和文档。
- 统一模型验证入口，支持 1–10 个模型；各模型保留独立进度、证据与可编辑报告草稿。单模型仅生成自身 Word/Excel，多个模型额外生成不重算指标的汇总 Excel。
- 更新 T/A 卡叙事、Lift/分箱解释、报告风险标记、任务悬浮信息与模型切换体验；确定性指标仍由平台代码计算。

## 本轮闭环的问题

- 报告草稿进入确认门时，批次不再被误判为失败；生成报告后同步父批次状态。
- 自动审查保留最终报告草稿确认门；全部确认在同一事务内校验修订号、输入合同并领取任务，冲突时整体回滚。
- 一个模型报告失败不阻断其余已确认模型；失败状态仍进入批次汇总。
- 修复批次自动运行 Promise 未释放的问题；创建表单更新默认叙事时保留用户已编辑内容。
- 增量 Bandit 统一基线与扫描文件路径，不改动原基线或降低检查级别；回归覆盖既有问题识别和新增问题拒绝。
- 更新缺少工作台上下文的前端测试夹具及旧的源码文本断言，保留原业务断言；公开示例使用泛化名称。

## 验证证据

| 验证层 | 结果 |
|---|---|
| 完整本地发布门禁 | `scripts/check -- -q` 退出码 0；12,415 passed、8 skipped、33 warnings，耗时 6,185.93 秒（1:43:05） |
| 静态检查 | Ruff、Node 语法、`git diff --check` 通过 |
| 增量安全检查 | 与 `origin/main` 比较的 Bandit 检查通过；不是完整安全审计结论 |
| 前端定向回归 | `tests/test_frontend_static_v2.py` 的 335 项通过；没有删除或跳过原失败用例 |
| 真实建模与交付 | 实际服务 JOIN→建模→PMML、业务材料交付，以及模型制品、验证交接、自然语言交付的 30 项定向测试通过；完整门禁中再次通过 |
| 真实浏览器创建页 | 单/多模型创建、Agent 选择、T/A 默认文案及保留用户编辑的 DOM 检查通过；控制台检查无错误 |
| 候选 wheel | 726 个包内文件逐字节匹配候选源树；新增模块、JS 与报告图片存在，无本地工作区或废弃模块混入；隔离安装后健康检查及六项 HTTP 资源检查通过 |
| 候选一致性 | 完整门禁前后改动指纹与新增检查脚本哈希一致；测试期间未修改候选代码 |

完整门禁启用了 `MARVIS_RUN_PLAYWRIGHT_SMOKE=1`，未使用 fast/affected 层级过滤。
8 个跳过项均为需要显式 DeepSeek 工作区配置的在线诊断，分别来自
`test_semantic_authorization_live_profile.py`、`test_semantic_intent_live_profile.py`
和 `test_strategy_sample_binding_semantic_delegation.py`；本轮未调用这些真实在线模型。

## 环境与边界

本机 `py_313` 中的 scikit-learn 1.9.1 超出当前 sklearn2pmml 0.131.0 转换器支持范围，
曾使真实 PMML 导出失败。仓库锁文件指定 scikit-learn 1.9.0；在由 `py_313` 派生、
复用其余依赖的临时隔离环境中安装该锁定版本后，真实流程与完整门禁通过。
未修改日常 conda 环境，也未放宽 Plugin worker 的环境隔离规则。
这不是宣称临时环境的每一个依赖都与锁文件完全相同。

示意命令（路径需替换为实际候选与隔离环境）：

```bash
MARVIS_RUN_PLAYWRIGHT_SMOKE=1 \
PYTHONPATH=/path/to/candidate \
PYTHON=/path/to/release-venv/bin/python \
CHECK_DIFF_RANGE=origin/main \
scripts/check -- -q
```

候选 wheel 在版本元数据提升前验证；正式版本由 `scripts/release_push.py` 创建，
正式 wheel 需在提升版本后重新构建并核对。原始开发快照和本地未跟踪材料保留在本地，
不进入公开发布历史或安装包。

本轮不证明原生 Windows/NTFS、Windows 安装包、真实机构材料验收、真实 LLM 全旅程、
生产部署或业务签字；这些边界不能用本地测试、浏览器创建页或发布 tag 替代。
