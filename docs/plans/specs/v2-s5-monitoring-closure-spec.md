# S5 监控闭环与定期报告 — 函数级 Spec

> 状态：待实现。依赖：**S2（采纳门产出 monitoring_plan_json 交付物）与 S3（trend/EL/portfolio_report 工具）先落地**。
> 范围校准：计划表 S5 列的「score/feature 稳定性趋势 + expected_loss_estimate + portfolio_report」已被 S3 spec 吸收实现；本批剩余职责 = **让监控计划不再是纸面 JSON**——采纳时落盘的计划被真实消费、告警门与策略上下文打通、逾期提醒可见。MONITORING_RUN 模板与 monitor_run 工具（S1b）是执行底座，本批不重写。

## 一、验收标准

对一个已采纳策略：按其监控计划跑一次监控→出告警报告→红灯给出处置建议（含"起新版本策略"动作）；计划有 due 语义且逾期在健康面板可见；全程审计。

## 二、Commit 1：监控计划的读取与执行闭环

### 计划契约固化（S2 落盘时的 monitoring_plan_json，本批给正式 schema）
```
{plan_version: 1, strategy_id, experiment_id?,          # 打分模型（可无：纯规则策略只监控通过率面）
 cadence_days: 30, last_run_at?: iso, 
 thresholds: {…MONITOR_RUN_THRESHOLDS 同构…},           # 采纳时可覆盖默认
 expectation_baseline: {approval_rate, approved_bad_rate,  # 来自采纳时点的回测
                        source_backtest_id}}
```
- `marvis/packs/strategy/monitoring_plan.py`：`load_monitoring_plan(artifact_path) -> MonitoringPlan`（dataclass，未知字段容忍、缺必填 typed error）+ `save_monitoring_plan`（S2 的写入点迁移到此模块，保持单一来源）。

### `tool_run_strategy_monitoring`（strategy 包）
```
入: strategy_id, dataset_id(新一期表现/申请数据), score_col?(有模型时)
出: checks=[…monitor_run 同构 check…] + 策略面新增:
    {id:"approval_rate_drift", value, level, baseline, actual}
    {id:"approved_bad_rate_drift", …}                    # 有标签时才出，无标签 level="n/a"
    overall_level(green|amber|red), plan_updated(last_run_at 刷新), red_flags
```
- 内部：读采纳策略的 monitoring_plan → 有 experiment_id 时委托 monitor_run 内核跑 PSI/CSI（同一函数，INV-1，阈值用计划覆盖值）→ 策略面漂移：对 expectation_baseline 比较（阈值：approval ±5pp=amber ±10pp=red，写常量可配）→ 合成 overall_level → **刷新计划 last_run_at（唯一的写回字段）** + 审计（kind='strategy.monitor', detail 含 overall_level）。
- 未采纳策略调用→typed error（"仅对已采纳策略执行监控"）。
- 测试：手算漂移分级三档；无标签 n/a；纯规则策略（无 experiment）跳过 PSI/CSI 只出策略面；last_run_at 写回+审计。

## 三、Commit 2：告警门处置闭环 + 逾期可见

### 模板 `STRATEGY_MONITORING`（templates/monitoring.py 内新增，与 MONITORING_RUN 并列）
```
slots: strategy_id(task_context), dataset_id(user, required)
steps: 1 执行策略监控 run_strategy_monitoring   **needs_confirmation + decision_point**
         （告警门：门文案分级列 checks；red 时 checklist 注入处置建议——
          「维持并观察 / 调阈值重跑 / 起新版本策略(new_version_from)」三选项，
          门回复解析三关键词，选"起新版本"→输出 next_action 字段供 driver 起后续
          STRATEGY_DEVELOPMENT 任务提示，不自动创建任务）
       2 生成监控报告 render_monitoring_report    post: nonempty report_path
```
- `tool_render_monitoring_report`：checks+趋势(有历史 run 时从审计聚合最近 N 次 overall_level 时间线)拼 Markdown/sheet，登记 strategy_artifacts(kind='monitoring_report_md')。
- goal_patterns: ("策略监控","跑监控","monitoring run 策略")；strategy 任务类型意图路由多认一组（S4 同款先例）。

### 逾期可见（不做守护进程，单机拍板：手动/agent 触发 + 被动提醒）
- `StrategyRepository.list_monitoring_due(now) -> [{strategy_id, due_at, overdue_days}]`（由 plan.cadence_days+last_run_at 推导；全 SQL/JSON 解析在仓储层）。
- `/api/health` 增 `monitoring_overdue_count`；`GET /api/strategies/monitoring-due` 明细端点（loopback 守卫同现有面）。
- 任务工作台策略卡（若有）或健康面板显示逾期徽标——前端最小改动：复用既有 stuck_jobs 徽标模式。
- 测试：due 推导边界（无 last_run→采纳时点起算）；health 计数；端点守卫。

## 四、非目标
cron/守护进程自动跑（单机拍板手动触发）、告警外发（邮件/webhook）、监控历史独立表（审计行聚合够用，量大再升表）、S3 组合报告的调度化。
