# Compatibility symbol registry (F20)

Every compatibility re-export or alias that production code does not need must
be listed here with an owner, its users, why it is kept, and the condition for
deleting it. An alias with zero users is deleted, not registered. Re-run the
reference scan (attribute access, `from … import`, `monkeypatch.setattr` strings
and `setattr(module, "name")`) before adding or removing a row.

Last scan: 2026-11-02. Owner for all rows: validation-agent / platform maintainers
(`marvis/agent/validation_app_service.py`, `marvis/repositories/`).

## `marvis/api.py` (ARCH-1 facade)

Deleted on 2026-11-02 (zero users): `agent_rerun_stage`, `answer_chat_message`,
`is_agent_advance_intent`, `is_stop_validation_intent`, `summarize_stage`,
`generate_word_conclusions`, `_reject_if_task_has_active_job`,
`AGENT_STOP_ACK_CONTENT`, `AGENT_STOP_STATUS_MESSAGE`, `AGENT_ACCEPTANCE_*`,
`_handle_agent_stop_message`, `_require_agent_task`,
`_require_wired_agent_task_type`, `_DRIVER_AGENT_TASK_TYPES`,
`_WIRED_AGENT_TASK_TYPES`, `_VALID_EFFORTS` and 40 further `_vas.*` /
`_validation_service.*` aliases (`_repo`, `_task_tier`, `_open_agent_stage`,
`_normalize_effort`, `_agent_memory_context`, …), and the unused `logger`.

| Symbol | Defined in | Users | Reason kept | Delete when |
| --- | --- | --- | --- | --- |
| `router` | `marvis.api` (real) | `marvis/app.py`, tests | production route root | never (not an alias) |
| `AgentValidationCancelled` | `agent.orchestrator` | `tests/test_agent_api.py` | test imports via api | test imports from `marvis.agent.orchestrator` |
| `_agent_evidence_from_settings` | `agent.validation_evidence` | `tests/test_agent_api.py` | monkeypatch target | test patches the leaf module |
| `_format_conclusion_values` | `agent.validation_messages` | `tests/test_agent_api.py` | test import | test imports leaf |
| `_run_agent_validation_job_impl` | `agent.validation_runner` | `tests/test_agent_api.py` | test import | test imports leaf |
| `_run_agent_scan_stage_impl`, `_run_agent_metrics_stage_impl` | `agent.validation_stages` | `tests/test_agent_api.py` | test import | test imports leaf |
| `_run_stage_job` | `api_stage_helpers` | `tests/test_api_v2.py` | test import | test imports leaf |
| `_task_payload` | `api_task_payloads` | `tests/test_api_structure.py` | structure test | test imports leaf |
| `_agent_has_stop_ack_message`, `_clear_agent_cancellation` | `agent.validation_service` | `tests/test_api_v2.py`, `tests/test_agent_api.py` | test import | tests import leaf |
| `_request_agent_cancellation` | wrapper over `agent.orchestrator.request_agent_cancellation` | `tests/test_agent_api.py` | test import | test calls orchestrator |
| `_run_agent_validation_job`, `_run_agent_word_conclusion_stage` | `agent.validation_app_service` | `tests/test_agent_api.py`, `tests/test_report_narrative_context.py` | test import | tests import `validation_app_service` |
| `_dispatch_agent_validation_job` | shim translating `repo=` → `repo_=` | `tests/test_agent_api.py` | old keyword signature | test calls `validation_app_service.dispatch_agent_validation_job(repo_=…)` |
| `_add_streaming_agent_message`, `_stream_agent_message` | `agent.validation_app_service` | `tests/test_agent_api.py` | test import | test imports leaf |

## `marvis/db.py` (repository facade)

Deleted on 2026-11-02 (zero users): `SchemaDowngradeError`, `_write_audit_row`,
`DataWorkspaceDataError`, `DataWorkspaceDatasetMismatch`,
`DataWorkspaceDatasetNotFound`, `DataWorkspaceResetRequired`,
`DataWorkspaceRevisionConflict`, `AGENT_REPORT_CONCLUSION_KEYS`, and the unused
private `_now` with its `datetime` imports.

| Symbol | Defined in | Users (files) | Reason kept | Delete when |
| --- | --- | --- | --- | --- |
| `connect`, `init_db` | `db_schema` | 87 / 184 test files | the standard test fixture entry point | a mechanical import migration of the test suite lands |
| `SCHEMA_VERSION`, `_ensure_column`, `sqlite_health` | `db_schema` | 5 / 1 / 1 tests | test import | tests import `db_schema` |
| `_list_audit_rows` | `repositories.audit` | 3 tests | test import | tests import leaf |
| `llm_usage_summary`, `record_llm_call` | `repositories.llm_calls` | `tests/test_llm_usage.py` | test import | test imports leaf |
| `TaskRepository`, `PluginRepository`, `DatasetRepository`, `StrategyRepository`, `PlanRepository`, `DataWorkspaceRepository`, `ModelingRepository`, `TaskArtifactRepository`, `DraftRepository`, `ValidationContractRepository` | `repositories.*` | 6–138 test files each | the shared repository facade used by fixtures | the same mechanical test-import migration |
| `_normalize_task_type` | `repositories.tasks` | `tests/test_api_v2.py` | test import | test imports leaf |

## `marvis/validation/effectiveness.py` (stage kernels)

`compute_overall_ks`, `compute_overall_psi`, `compute_bin_tables`,
`compute_independent_quantile_bin_tables`, `compute_psi_stability_table`,
`compute_roc_ks_curves`, `compute_monthly_ks`, `compute_monthly_psi`,
`compute_auc`, `compute_head_tail_lift`, `build_effectiveness_result`,
`prepare_effectiveness_context`, `_should_reverse_eval_bins`, `_roc_ks_curve`:
not compatibility aliases. Each one is called by `EffectivenessComputation`
(and so by the Notebook metric cells), and the tests call it directly as a
deterministic kernel. Kept. A kernel becomes deletable only when
`EffectivenessComputation` stops calling it.
