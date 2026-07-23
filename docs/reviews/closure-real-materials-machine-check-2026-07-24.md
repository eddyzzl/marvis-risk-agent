# 真实材料机器预检报告

- 生成时间：`2026-07-23T16:50:11.028447+00:00`
- 任务：`<governed-real-material-task>` · 受控真实材料建模任务
- 计划：`<completed-modeling-plan>` · `done`
- 机器判定：**FAIL**
- 收口判定：**BLOCKED_MACHINE**

> 本报告只核验仓库/SQLite/产物中可机器验证的证据，不等同于人工签字。
> 外部财务、拨备、风险报表口径及责任人签字绝不自动填充。

## A/B/C/D 核验

| 项 | 状态 | 结论 | 证据 | 人工动作 |
|---|---|---|---|---|
| A1 | FAIL | 检测到哨兵值，但冠军模型的预处理链不可追溯。 | screen_features 检测到 sentinel 列；精确列数仅保留在受控任务审计中；model_artifacts.params.preprocessing_chain_traceable=false | — |
| A2 | N/A | 当前任务是模型开发，未执行 vintage 计算。 | plan.template_id=modeling | — |
| A3 | PASS | 特征筛选未丢弃标签，未发现静默排除。 | screen_features 未报告标签丢弃；精确计数仅保留在受控任务审计中。 | — |
| A4 | N/A | 该已完成建模计划没有 join gate，不能由建模产物复核长 ID 键。 | plan steps do not contain data_join tools | — |
| A5 | N/A | 该已完成建模计划没有 join gate，不能由建模产物复核空白/零填充键。 | plan steps do not contain data_join tools | — |
| A6 | PASS | 已生成非空 OOT，切分证据可追溯。 | split_col 与 holdout_values 已记录；train/test/oot 均非空，精确业务行数仅保留在受控任务审计中。 | — |
| B1 | MANUAL | vintage 累计坏账率必须与外部业务口径对账，仓库内不存在可替代的地面真值。 | no external finance/risk ground truth supplied | 填写外部口径来源、实测 vs 口径并由责任人签字（B1）。 |
| B2 | MANUAL | 组合 EL必须与外部业务口径对账，仓库内不存在可替代的地面真值。 | no external finance/risk ground truth supplied | 填写外部口径来源、实测 vs 口径并由责任人签字（B2）。 |
| B3 | MANUAL | 分群 bad_rate必须与外部业务口径对账，仓库内不存在可替代的地面真值。 | no external finance/risk ground truth supplied | 填写外部口径来源、实测 vs 口径并由责任人签字（B3）。 |
| B4 | MANUAL | 平台内部 KS 在选择结果与模型卡间一致；与独立复算/历史模型的外部对账仍需人工完成。 | internal_consistency=True；train/test/OOT KS 已记录在受控任务审计中，未写入源码。 | 提供独立复算或历史同类模型 KS，并完成 B4 签字。 |
| B5 | N/A | 当前完成计划没有 join 步骤；join 匹配率需在数据处理任务单独验收。 | no data_join step in completed modeling plan | — |
| C1 | PASS | 训练输出记录了真实选择指标。 | train_models.selection_metric='test_ks(overfit-penalized)'; select_experiment.selection_reason='用户指定实验。' | — |
| C2 | PASS | 步骤证据信封包含代码清单哈希、参数哈希、数据引用与随机种子。 | manifest_hash=True; input_hash=True; source_dataset_refs=True; seed=True | — |
| C3 | PASS | 对账红旗由对抗形状/双路对账回归网验证，不依赖业务人员目测。 | tests/test_dirty_shape_regression.py + tests/test_reconcile_reference_numbers.py | — |
| D1 | PASS | 精选特征仅在 train 上拟合。 | select_features.fit_split='train'；训练行数已记录在受控任务审计中。 | — |
| D2 | PASS | 本任务无 NaN 标签；NaN 门另由对抗形状回归覆盖。 | 标签缺失计数仅保留在受控任务审计中；tests/test_dirty_shape_regression.py::test_nan_label_screen_requires_confirmation | — |
| D3 | PASS | 本任务未执行 train+test refit，不存在随机 5% headline 冒充问题。 | select_experiment.refit={'applied': False, 'requested': False, 'reason': '未请求全量重训(refit_on_train_plus_test=false)。'} | 若 refit=true，核对模型卡 pre-refit/部署差异说明。 |

## 产物

- report: PASS
- native_model: PASS
- pmml: PASS
- model_card: PASS

## 模型治理

- 监控状态：`fail` · 需模型风险复核后再交付
- 限制：预处理链不可追溯：训练数据集无预处理血缘记录，无法确认打分重放是否完整。
- 限制：选型策略警示：train-test KS 差距超过警戒阈值 0.1，存在过拟合风险（未阻断选择）；精确指标仅保留在受控任务审计中。
- 限制：需模型风险复核后再交付

## 仍需人工完成

- B1-B4 external ground-truth reconciliation and accountable signatures; the script cannot and will not fabricate them.
- 完成人应在 `docs/plans/v2-real-materials-reconciliation-checklist.md` 填写外部口径、实测值及签字。

## 对抗形状与双路对账回归

执行：

```bash
PYTHONPATH=.:tests /opt/miniconda3/envs/py_313/bin/python -m pytest -q \
  tests/test_dirty_shapes_generator.py \
  tests/test_dirty_shape_regression.py \
  tests/test_reconcile.py \
  tests/test_reconcile_reference_numbers.py
```

结果：`66 passed in 2.00s`。
