# 数据处理 / 特征分析 / 模型开发 Fresh-workspace Smoke

- 生成时间：`2026-07-23T20:17:19.416353+00:00`
- 测试版本：提交前最终工作树（即本次提交内容）
- 临时根目录：`<fresh-temporary-root>`
- 总体：**PASS**

| 流程 | 状态 | 用时（秒） | fresh workspace | 测试入口 |
|---|---|---:|---|---|
| data_join | PASS | 9.071 | `<fresh-temporary-root>/data_join` | `tests/test_data_join_api.py::test_data_join_conversation_end_to_end` |
| feature_analysis | PASS | 8.779 | `<fresh-temporary-root>/feature_analysis` | `tests/test_feature_analysis_api.py::test_feature_analysis_end_to_end` |
| modeling | PASS | 28.372 | `<fresh-temporary-root>/modeling` | `tests/test_modeling_api.py::test_modeling_business_materials_flow_into_report_and_delivery` |

## 原始 pytest 输出

### data_join

```text
.                                                                        [100%]
1 passed in 8.50s
```

### feature_analysis

```text
.                                                                        [100%]
1 passed in 8.22s
```

### modeling

```text
.                                                                        [100%]
1 passed in 27.84s
```
