# FIN-2 全量审查 + FIN-3 修复循环收官报告（2026-07-03）

## 停机判定：达成

用户停机条件（backlog §11.5 / DoD-11）：修复→回归→再审循环，直至**一轮无新 critical/high**。

- **第一轮（FIN-2 全量审查）**：8 镜头并行 + 每条 critical/high 三席对抗表决（47 agents，288 万 tokens）。产出 2 确认 HIGH、11 条被表决驳杀、12 条 medium/low；另 2 个镜头（security/api）结构化输出失败后补跑，追加 1 确认 HIGH（插件魔法头）+ bandit 42 中危全裁误报 + 10 新工具 manifest 零漂移 + 12 条工厂一致性 low。
- **第一轮修复**：14 commits 双 agent 并行（并发/仓储 7 + 域/指标 7），全部带注入式并发测试或手算证据。
- **第二轮（复审）**：逐 commit 聚焦复审 14 个修复——**全部 correct，零新 critical/high**；第一轮验证时暴露的并发上传 FileNotFoundError（1/6 复现）根因定位为 `.staging` 目录级 TOCTOU，修复（`8c64c21d`：stage 即占位+元数据锁）经 40 连跑零失败+人工 diff 复核干净。
- **第二轮无新 critical/high → 循环停机。**

## 修复清单（FIN-3 全部 15 commits）

| 严重度 | 修复 | commit |
|---|---|---|
| HIGH | 采纳时兄弟退役无 rowcount 守卫→原子 CAS+ConflictError | `f4615df4` |
| HIGH | 门×旗矩阵系统复查：7 个漏网强制门接 AUTO halt（含 portfolio/execute_join/monitor_run 等） | `12c6a90c` |
| HIGH | 插件管理魔法头 `local-dev`→每工作区 secrets.token_hex(32)+0600+恒时比较 | `bc143c58` |
| HIGH | 并发上传 `.staging` 目录 TOCTOU→占位+元数据锁（40 连跑零失败） | `8c64c21d` |
| Med | CSI 期望改存真实训练 bin 占比（退化分布正确），老快照三层回退 | `c76c70eb` |
| Med | 状态读提进 BEGIN IMMEDIATE 写锁 | `ba85cde1` |
| Med | start_step_run 状态守卫（重试路径不受影响） | `fa15a8a2` |
| Med | EL 缺 pd_col 优雅降级 None+红旗（不给假 0） | `b302250a` |
| Med | mine_rules 列序确定性（INV-1，四种乱序逐字节相等） | `f429f780` |
| Med | 记忆锚点注入定界+控制字符剥离+截断（INV-4） | `899e4250` |
| Med | 分箱退化 opt-in 诊断 | `76959328` |
| Low | 前端死代码（renderLegacyTable 等）清除 | `6ff5e25b` |
| Low | 陈旧并发测试注释更新 | `9b76869f` |
| Low | 监控 due 边界测试钉死（UTC 免疫 DST，核实无需改码） | `660f89d9` |
| Low | 错误工厂扩展到助手/agent 服务层（12 处，guard 扫描扩面） | `ea2c22a5` |

## 表决驳杀与裁决（不修的部分及理由）
- 11 条 FIN-2 发现被三席对抗表决驳倒（如"退役策略无法再采纳""受信代理伪造提权"——后者实为降权方向）。
- bandit 42 中危全部逐类裁决误报（占位符计数拼接/白名单标识符/参数化值，报告含逐类理由）；B310/B301 同裁（配置文件 URL/框架内 pickle）。
- 已知残余（低风险记录在案）：TransactionalDirectoryStore 目录级并发暂存窗口未完全闭合（drafts 单用户手动路径，依赖 mkdir(exist_ok=False)，同锁护住 parent-mkdir）；materials 上传 symlink 窗口（uuid4 临时目录，窗口极小）。

## 质量轨迹
测试规模 1988 → 2742+（收官门禁另行记录）；三轮全量门禁零失败；FIN-1 落地核验 182/182。
