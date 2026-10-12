# 机构通用适配器协议

本模块提供受配置约束的来源查询、原生模型登记和规则包部署接口。当前没有指定机构；两个独立本地 HTTP 参考实例的互操作与故障恢复测试，只证明软件协议在这些参考载体上可用。机构身份、证书、真实接口、SLO、流量批准及成熟业务结果仍需单独验收。

协议版本为 `marvis.institution-adapter.v1`。接口复用平台现有身份和 maker/checker/admin 审批链。普通任务请求不能设置主机、凭据、文件路径、任意 HTTP 路由、回调或验签密钥。用户材料、模型和来源响应不会进入 LLM；已认证的原生模型文件仅在用户配置的登记接口上传。

## 本地配置与应用挂载

配置只从受信的本地启动配置读取。`MARVIS_INSTITUTION_ADAPTERS_CONFIG` 指向 JSON 数组；未设置时能力接口返回 `status: unconfigured`，不连接网络。非法配置或重复 alias/environment 会明确失败，不静默跳过一个来源。

```json
[
  {
    "alias": "reference-a",
    "instance_id": "reference-a",
    "environment": "reference-a",
    "base_url": "http://127.0.0.1:19181",
    "credential_file": "/private/local/reference-a/access_token",
    "mode": "local_reference",
    "timeout_seconds": 3.0
  },
  {
    "alias": "reference-b",
    "instance_id": "reference-b",
    "environment": "reference-b",
    "base_url": "http://127.0.0.1:19182",
    "credential_file": "/private/local/reference-b/access_token",
    "mode": "local_reference",
    "timeout_seconds": 3.0
  }
]
```

凭据文件必须由当前用户拥有、为普通文件、只允许本人读取；参考服务生成权限为 `0600` 的随机凭据。凭据只在本机读取后用于协议 Authorization 头，不能复制到 API 请求或报告。`local_reference` 只允许显式端口的 `http://127.0.0.1`；机构模式为 `institution_https`，必须使用证书验证的 HTTPS，可在本地配置 `ca_file`。不跟随重定向，不使用环境代理。实际机构 HTTPS 联调尚未完成。内置环境 `local-reference` 保留，不能分配给通用接口。

应用挂载示例（由应用的统一组装层调用）：

```python
from marvis.institution_adapters.router import (
    configure_institution_adapters,
    router as institution_adapters_router,
)

# 先初始化 app.state.settings、已有身份、生产治理和 risk_sources。
configure_institution_adapters(app)
app.include_router(institution_adapters_router)
```

模块不新增中央数据库迁移；操作日志与本地授权记录保存在 workspace 下 `institution_adapters/operations.sqlite`，其完整性凭据为同目录私有 `authority.key`。这些私有文件不能当作公开归档或独立签名权威。

## 独立参考服务

用两个不同状态目录、实例 ID 和端口启动两个独立进程。以下命令用于公开 runbook；项目开发验证使用 `py_313` 环境。

```bash
python -m marvis.institution_adapters.reference_endpoint \
  --state-dir /private/local/reference-a \
  --instance-id reference-a --port 19181

python -m marvis.institution_adapters.reference_endpoint \
  --state-dir /private/local/reference-b \
  --instance-id reference-b --port 19182 \
  --drop-every-new-operation-response
```

第二个命令故意在每个新操作提交后丢弃响应，用于验证显式恢复。该选项是合成故障演练，正常配置应省略。来源查询可用 `--records /private/local/synthetic-source-records.json` 加载合成记录；每项必须符合既有 `SourceEnvelope`，没有匹配记录返回 `no_record`。不要把真实客户材料放入参考演练目录。

远端只实现以下固定路由：

| 方法 | 路由 | 用途 |
| --- | --- | --- |
| GET | `/v1/adapter/capabilities` | 验证配置实例及协议执行范围 |
| POST | `/v1/adapter/operations` | 一个固定类型操作 |
| GET | `/v1/adapter/operations/{operation_id}` | 读回确切请求的结果 |
| GET | `/v1/adapter/registry/{bundle_hash}` | 重读原生文件并复核内容和 schema |
| GET | `/v1/adapter/deployment/{install_operation_id}` | 重读规则文件、执行探针并读当前部署头 |

参考模型登记仅保存并校验不透明原生文件，不反序列化 pickle，也不提供模型评分服务。部署执行范围为 `rule_only`；当前不提供通用远端 `/decisions` 评分接口。

## 任务 API 与操作身份

以下路径统一以 `/api/tasks/{task_id}/institution-adapters` 为前缀。请求使用应用现有认证会话。调用前先 GET `/capabilities`，确认 alias 的配置范围。

每个远端实例的 `operation_id` 必须使用新的全局唯一 ID，例如 UUID。相同 ID 只能重复完全相同的请求。操作还绑定本地任务、调用身份、配置摘要和确切请求内容。不要在不同任务复用一个远端 ID。模型 schema 严格保留原生特征顺序；数值、类型、nullable 都需明确，不能通过自由文本描述推断。

| 方法 | alias 下的路径 | 请求 DTO / 身份 |
| --- | --- | --- |
| POST | `/{alias}/registry/uploads` | `RegistryUpload` / maker |
| GET | `/{alias}/registry/uploads/{operation_id}/readback` | 原操作调用者或 admin |
| POST | `/{alias}/registry/promotions` | `RegistryTransition` / admin |
| POST | `/{alias}/registry/revocations` | `RegistryTransition` / admin |
| POST | `/{alias}/deployments/install` | `DeploymentInstall` / admin |
| GET | `/{alias}/deployments/installations/{operation_id}/readback` | 原操作调用者或 admin |
| POST | `/{alias}/deployments/activate` | `DeploymentActivate` / admin |
| POST | `/{alias}/deployments/rollback` | `DeploymentRollback` / admin |
| GET | `/{alias}/operations/{operation_id}` | 本地只读状态 |
| POST | `/{alias}/operations/{operation_id}/recover` | 显式远端 GET 恢复 |
| POST | `/{alias}/sources/grants` | 既有 `SourceGrant` / admin |
| POST | `/{alias}/sources/grants/{grant_id}/revoke` | admin，可重复撤销 |
| POST | `/{alias}/sources/requests` | `SourceRequest` / maker |

### 原生模型登记

先通过 MARVIS 原生训练获得本任务的 `ModelArtifact` 和可认证 producer receipt。上传前认证所属实验、artifact closure、文件字节和特征顺序；读取文件总量最多 40 MB。任何闭包缺失、文件漂移、schema 顺序不一致都会在上传前拒绝。

`POST /{alias}/registry/uploads`：

```json
{
  "operation_id": "registry-upload-unique-id",
  "model_artifact_id": "native-model-artifact-id",
  "raw_schema": [
    {"name": "x1", "type": "number", "nullable": false},
    {"name": "x2", "type": "number", "nullable": false}
  ]
}
```

成功返回操作状态、确切 `bundle_hash`、`manifest.schema_hash`、文件 SHA/大小及 `storage_only: true`。随后 GET readback，远端重新读取实际文件并复算 SHA。升格/撤销前，需在已经过原生验证的策略 metadata 中绑定如下内容，再走既有生产治理的 maker 请求、checker 和 admin 两次独立批准（三个身份必须不同）：

```json
{
  "institution_registry": {
    "bundle_hash": "64位上传包摘要",
    "model_artifact_id": "native-model-artifact-id",
    "schema_hash": "64位有序schema摘要"
  }
}
```

原生批准入口为 `POST /api/production-governance/promotion-requests`，指定已验证 strategy ID/version、配置的 environment、deployment_slot 和 reason；两位批准者分别调用 `POST /api/production-governance/promotion-requests/{id}/approvals`。不能通过直接修改资产状态替代原生验证或批准。

`POST /{alias}/registry/promotions` 或 `/registry/revocations`：

```json
{
  "operation_id": "registry-transition-unique-id",
  "upload_operation_id": "registry-upload-unique-id",
  "promotion_id": "native-approved-promotion-id"
}
```

登记升格只改变登记状态，返回 `serving: false`。它不证明模型已部署或业务流量已批准。

### 规则部署与回滚

先在已验证策略的 metadata 明确绑定完整有序 schema 的规范 JSON 摘要；摘要使用平台 `digest()`。所有规则引用字段必须在 schema 中声明；平台派生字段不能伪装为原始输入。

```json
{
  "institution_deployment": {
    "package_kind": "rule_only",
    "schema_hash": "64位有序schema摘要"
  }
}
```

通过既有原生审批获得当前配置 environment 的 approved promotion。`POST /{alias}/deployments/install`：

```json
{
  "operation_id": "install-unique-id",
  "promotion_id": "native-approved-promotion-id",
  "raw_schema": [
    {"name": "x1", "type": "number", "nullable": false},
    {"name": "x2", "type": "number", "nullable": false}
  ],
  "probe_features": {"x1": 0.1, "x2": 0.2}
}
```

参考服务安装确切规则字节、校验 schema 并实际执行探针。GET installation readback，确认实际文件仍健康。激活请求：

```json
{"operation_id": "activate-unique-id", "install_operation_id": "install-unique-id"}
```

激活复核原批准与安装绑定，远端和本地分别使用预期部署头比较再写入。远端激活成功后，以实际观察到的 receipt 和探针证据调用既有原生治理仓库；不会另设任意签名密钥的批准旁路。

真实第二版本激活后，回滚请求：

```json
{
  "operation_id": "rollback-unique-id",
  "deployment_operation_id": "activate-second-unique-id",
  "expected_current_ref": "activate-second-unique-id"
}
```

只有真实前驱部署且当前头匹配时才允许回滚；首个部署没有前驱会明确拒绝。完成后需要读回远端 installation 当前 `active_ref` 和原生 environment 头；历史操作的 `completed` 只说明该操作曾完成，不代表它现在仍在服务。

### 来源查询与授权

先由 admin 在既有 `/api/tasks/{task_id}/risk-sources/authorization-bases` 登记本地授权依据；来源 basis 的原文件、hash、任务和原生签名都会重新认证。再在 configured alias 下 POST `/sources/grants`，使用完整既有 `SourceGrant`：

```json
{
  "grant_id": "source-grant-unique-id",
  "profile_id": "reference-a",
  "task_id": "current-task-id",
  "grantee_id": "existing-maker-principal-id",
  "subject_namespace": "reference.people",
  "subject_token": "64位上游脱敏token",
  "purpose": "credit_review",
  "basis_artifact_id": "registered-authorization-basis-artifact-id",
  "starts_at": "2026-10-10T00:00:00+00:00",
  "expires_at": "2026-10-10T01:00:00+00:00"
}
```

query 请求只包含操作和授权标识；不得包含原始身份证、电话或客户详情：

```json
{"operation_id": "source-query-unique-id", "grant_id": "source-grant-unique-id", "expires_in_seconds": 300}
```

返回严格 provider 协议和平台确定性 assessment。未知/部分/无记录保持未知；`automated_clearance` 始终为 false，`source_truth` 明确是未经独立事实认证的 provider claims。授权撤销/过期后，新查询以及该授权下原操作读回均被拒绝。

## 不确定状态与恢复

本地先持久保存 claim，再认证配置实例并发送一次 POST。失败后 `remote_effect_unknown`（409）不能解释为远端没有发生效果。原 ID 再次提交只读取原状态，不发送新 POST。必须显式 POST `/operations/{operation_id}/recover`；实现只发送远端 GET，验证相同 request hash 后返回结果。远端 404 继续保持不确定，不隐式重试操作。

远端已提交、本地治理尚未完成时，GET 操作返回 `status: remote_committed_local_pending` 和 next_action，保持只读。显式 recover 会读回远端并补齐本地治理；若原生提交已完成而日志尾写失败，会认证既有确切记录后补齐日志，不能第二次激活。批准过期或本地头漂移等无法安全补齐的情况持续明确失败，需要人工核查双方状态，不能重发同一操作或伪报成功。

## 软件验证覆盖

`tests/test_institution_adapters.py` 使用两个独立本地 HTTP 进程、独立 token/SQLite/文件目录，合成数据通过实际 MARVIS ToolRunner 训练产生 native artifact。覆盖七类操作正常与响应丢失后的 GET 恢复、客户端重启、远端文件漂移、native persistence 失败、native 已提交但日志尾写失败、权限/schema/配置实例/凭据错误，以及明确撤销。

这些测试不建立真实机构验收，也不建立任意原生模型远端执行能力。实际机构证书握手、SLO/负载、外部合法授权及真实来源真实性仍需对应责任人和机构接口证据。当前通用协议的本地日志尚未加入正式运行归档 carrier；不能把参考测试冒称为独立归档验收。
