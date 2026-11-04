# 冻结性能配置的执行入口

`marvis.orchestrator.eval.perf_gate` 支持本机基线和按冻结配置执行的两种模式。仓库中的 [性能配置](performance-gate-profile.json) 仍为 `pending_user_freeze`：软件不会替用户填写硬件、负载或 SLO。

本机基线不需要批准阈值，也不产生达标结论：

```sh
python -m marvis.orchestrator.eval.perf_gate baseline --out /evidence/local-baseline
```

取得批准的配置和材料后，正式入口可执行固定的 `task-ingest-read-v1` 工作负载：

```sh
python -m marvis.orchestrator.eval.perf_gate formal \
  --profile /held/performance-profile.json \
  --binding /held/workload-binding.json \
  --out /evidence/frozen-performance-run
```

输出目录须尚不存在。启动前检查配置、批准日期、硬件指纹、源码清单哈希、CSV 原件哈希及实际行数。无效输入在创建输出和启动服务前拒绝。批准字段只是调用方提供的记录，不由此认定独立签字有效。

绑定文件顶层必须恰有以下字段：

| 字段 | 内容 |
| --- | --- |
| `schema` | `marvis.performance-workload-binding.v1` |
| `workload` | `task-ingest-read-v1` |
| `scope` | 本地合成材料用 `local_synthetic_fixture`；实际提供的验收环境用 `supplied_acceptance_environment` |
| `dataset` | `path` 为真实 CSV 的绝对路径；`sha256` 为文件字节哈希；`rows` 为不含表头的实际行数，须等于 profile 中批准的 `data_rows` |
| `source_inventory_sha256` | `perf_gate_formal.source_inventory()` 的规范 JSON SHA-256；执行前后均核对包代码和资产 |
| `execution` | `cold_start_runs`、`recovery_runs`、`requests_per_worker`、`request_timeout_seconds`、`startup_timeout_seconds` 五个整数；其支持范围由 `_EXECUTION` 合同检查 |

硬件指纹由 `perf_gate_formal.hardware_fingerprint()` 读取当前机器生成。源码清单使用按键排序、无空白分隔的 JSON 编码后计算 SHA-256。材料只支持 UTF-8 CSV、唯一非空表头和等宽数据行；当前上限为 128 MiB、1,000 万行。超出支持范围会明确拒绝。

实际执行包括：全新工作区冷启动、已摄取材料的持续读请求、批准数量的并发工作线程创建任务并注册材料、终止自有服务进程后在同一工作区重启。恢复检查比较任务和数据集身份；失败请求保留在计时及错误率分母中，不重试掩盖。

输出保留配置、绑定、材料副本、源码清单、逐次观测、服务日志及 JSON/Markdown 报告。报告的 `gate_verdict` 只判断该工作负载是否达到调用方提供的阈值，`acceptance_claim` 始终为 `none`。

本入口测量材料摄取和任务界面请求，不测模型训练吞吐量，不代表机构生产流量。隐藏案例的真实并发、预热和故障原件另见 [隐藏基准执行说明](hidden-formal-execution.md)。正式软件验收、历史数据和机构生产仍各自需要对应原件与独立复核。
