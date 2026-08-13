# Bandit 基线重建与分类审查（2026-08-13）

> 计划条目 B-5 的交付物。基线文件 `.bandit-baseline.json` 已于本日重建；
> `scripts/check` 新增了对变更文件的 bandit 增量检查。

## 结论

- 基线并非"过期的大清单"：以 CI 同款参数（bandit 1.9.4，`-r marvis -ll -ii`）
  重新扫描，当前 finding 总数为 **83 条**（旧基线同为 83 条，仅 1 处行号漂移）。
  原文件体积大（9,500 行 JSON）是因为每条 finding 的 JSON 展开约 110 行，不是
  finding 数量爆炸。
- 与旧基线的差异：移除 1 条（`packs/data_ops/tools.py:2029` B608，行号漂移）、
  新增 1 条（同文件 2030 行 B608）。净变化为零。
- 带新基线复跑 CI 同款命令 `bandit -r marvis -ll -ii -b .bandit-baseline.json`
  退出码 0：发布门通过。

## 83 条 finding 分类

| test id | 数量 | 严重级 | 置信度 | 分类说明 |
|---|---|---|---|---|
| B608（f-string SQL 注入） | 71 | MEDIUM | 71 MEDIUM | 主体。DuckDB/DataWorkspace 的受治理 SQL 构造；参数来自受控 identifier 校验与既有 DSL compiler，非用户自由 SQL 直连。基线接受，但属于"最值得专项复审"的一类（见下）。 |
| B103（宽松文件权限） | 3 | MEDIUM | 3 HIGH | 工作区/临时目录权限调用；本地单机工作台模型下接受。 |
| B314（xml 解析） | 3 | MEDIUM | 3 HIGH | PMML/文档解析路径。 |
| B310（urllib） | 2 | MEDIUM | 2 HIGH | 内部升级/下载辅助。 |
| B102（exec 检测触发点） | 2 | MEDIUM | 2 HIGH | 带 nosec 注释的受控执行边界（plugins/subprocess_worker 等），注释仍在生效。 |
| B301（pickle） | 1 | MEDIUM | 1 MEDIUM | 受控工件反序列化。 |
| B317（socket） | 1 | MEDIUM | 1 HIGH | 本地回环服务辅助。 |

- 12 条 HIGH 置信度条目集中在 B103/B314/B310/B102/B317——非 B608。
- 全部 83 条均已在基线中被明确接受；本次重建未引入任何新的安全信号。

## 后续动作

1. **B608 专项复审**（建议列入下个审查窗口，非本 90 天计划 P0/P1）：71 条
   f-string SQL 全部过一遍，确认每个构造点的输入都经过 identifier 白名单/类型
   校验；可验证的加 `# nosec B608` + 理由，不可验证的修复。
2. 保持增量门：`scripts/check` 现在对相对 `HEAD`（或 `CHECK_DIFF_RANGE`）变更的
   `marvis/*.py` 文件跑 `bandit -ll -ii -b .bandit-baseline.json`，新增 medium/high
   finding 会使门失败；`--skip-bandit` 可跳过，环境缺 bandit 时优雅跳过（与
   pip-audit 相同模式）。

## 验证记录

- `bash -n scripts/check`：通过
- `PYTHON=<conda py_313>/bin/python scripts/check --skip-pytest --skip-ruff --skip-node --skip-diff`：
  通过（"no changed marvis/*.py files"）
- `conda run -n py_313 python -m bandit -r marvis -ll -ii -b .bandit-baseline.json`：exit 0
