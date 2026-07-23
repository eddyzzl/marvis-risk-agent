# T4-2 公开数据 KS 基线验收

- 验收日期：2026-07-24（Asia/Shanghai）
- 结论：**BLOCKED**
- 原因：两个公开原始文件均不存在，两个地面真值基线均未记录。

## 只读数据查找

执行：

```bash
find <worktree> \
     <local-data-staging> \
  -type f \( \
    -iname 'cs-training.csv' -o \
    -iname 'application_train.csv' -o \
    -iname '*give*me*some*credit*' -o \
    -iname '*home*credit*default*risk*' \
  \) -print
```

结果：无输出。没有用合成数据、相似文件名或平台历史任务冒充公开数据。

## 机器门

```bash
python scripts/ks_baseline.py --status
```

结果：退出码 `2`。

| 数据集 | 原始文件 | 人工精调基线 | 状态 |
|---|---|---|---|
| Give Me Some Credit | 缺失 | `null` | BLOCKED |
| Home Credit Default Risk | 缺失 | `null` | BLOCKED |

## 精确补齐命令

Give Me Some Credit：

```bash
python scripts/ks_baseline.py \
  --dataset give_me_some_credit \
  --input /absolute/path/to/cs-training.csv \
  --params-json @/absolute/path/to/reviewed-lgb-params.json \
  --record \
  --tuned-by "<name/team>" \
  --tuning-note "<method and review>"

python scripts/ks_baseline.py \
  --dataset give_me_some_credit \
  --input /absolute/path/to/cs-training.csv
```

Home Credit：

```bash
python scripts/ks_baseline.py \
  --dataset home_credit \
  --input /absolute/path/to/application_train.csv \
  --params-json @/absolute/path/to/reviewed-lgb-params.json \
  --record \
  --tuned-by "<name/team>" \
  --tuning-note "<method and review>"

python scripts/ks_baseline.py \
  --dataset home_credit \
  --input /absolute/path/to/application_train.csv
```

`--record` 会保存原始文件 SHA-256、样本数、特征数、seed、split、配方、
完整参数、调优/复核责任人和时间。匿名 Agent 运行会被拒绝，不能成为地面真值。
