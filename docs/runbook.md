# MARVIS 本地运行手册（V2.1.10）

MARVIS-Agent 的产品边界是面向数据处理、特征分析、建模、验证、策略、监控和组合分析的本地优先信贷风控智能体。当前 V2.1.10 公开版已经进入多工作流 Agent 平台主线；本手册的直接 CLI 流水线部分仍以模型验证兼容工作流为例，因为它是最稳定、最适合脚本化演示的路径。

## 本地部署要求

- Windows 个人电脑优先使用一键安装包；安装包内置私有 Python runtime 和 Java runtime。
- 源码安装需要 Python 3.11 或更高版本，推荐新环境使用 Python 3.12。
- 当前验证过的源码安装运行环境为 macOS / Linux。
- 源码安装如需执行 PMML 打分，请安装与 `pypmml` 兼容的 Java Runtime。
- Node.js 只用于前端静态语法检查；运行 Web 服务不依赖前端构建。

## Windows 一键安装

面向没有 Python/Java 的个人 Windows 电脑，默认发布物是：

```text
MARVIS-Setup-<version>-win-x64.exe
```

安装器按当前用户安装到 `%LOCALAPPDATA%\Programs\MARVIS-Agent`，运行数据放在 `%LOCALAPPDATA%\MARVIS-Agent\workspace`，日志放在 `%LOCALAPPDATA%\MARVIS-Agent\logs`。双击桌面或开始菜单里的 `MARVIS-Agent` 后，启动器会设置私有 `JAVA_HOME` 和 `PATH`，执行 `marvis serve --host 127.0.0.1 --port 8000`，并打开浏览器。

构建脚本和安装器模板位于 `packaging/windows/`。Docker 只作为后续服务器或 IT 管理部署选项，不作为个人 Windows 默认入口。

## 从 GitHub 安装

创建环境时不要依赖开发者本机的固定环境名。用户从 GitHub clone 后，在仓库目录内安装。任选 `venv` 或 conda 均可：

```bash
git clone https://github.com/eddyzzl/marvis-risk-agent.git
cd marvis-risk-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

或：

```bash
git clone https://github.com/eddyzzl/marvis-risk-agent.git
cd marvis-risk-agent
conda create -n marvis python=3.12
conda activate marvis
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## 启动 Web 服务

安装后可直接运行：

```bash
marvis
```

默认等价于：

```bash
marvis serve \
    --host 127.0.0.1 --port 8000 --workspace ./workspace
```

`marvis` 是 V1 为兼容当前验证运行时保留的 Python 模块名。以下旧入口仍可用：

```bash
python -m marvis serve \
    --host 127.0.0.1 --port 8000 --workspace ./workspace
```

```bash
marvis-risk-agent serve \
    --host 127.0.0.1 --port 8000 --workspace ./workspace
```

## 材料目录允许范围

创建任务时，后端只接受位于当前 `workspace` 或当前用户 home 目录下的材料目录。这样可以避免本地服务被误暴露时读取任意路径。

如果 Windows 部署时材料放在 D 盘、外接盘或其他不在 home 下的位置，先设置额外材料根目录再启动：

```powershell
$env:RMC_MATERIAL_ROOTS="D:\model_materials"
marvis serve --host 127.0.0.1 --port 8000 --workspace .\workspace
```

多个根目录可以按系统路径分隔符拼接；Windows 用分号，macOS/Linux/WSL 用冒号。

WSL2 中运行时，页面里的材料目录也要填写 Linux/WSL 路径，例如 `/mnt/c/Users/<you>/Downloads/project`，不要填写 `C:\Users\...`。

## 共享主机部署：JupyterHub / 多用户服务器

MARVIS 的默认信任模型是"回环地址（127.0.0.1）即可信"——这在单人笔记本上成立，但 CLI 本身建议的 JupyterHub 部署方式恰恰是一台**多用户共享服务器**：同一台机器上其他已登录用户的进程同样能连到 `127.0.0.1:8000`，默认会被判定为本地可信客户端，从而获得读取全部借款人明细、删除任务、安装插件（插件在服务账号下执行任意代码）等全部权限。

如果部署在共享主机 / 多账号 Linux 服务器上，按需组合下面三个环境变量：

### `MARVIS_LOCAL_TOKEN`：本地写操作令牌（推荐，最先配置）

设置后，所有非 `GET` 请求（包括来自 `127.0.0.1` 的请求）都必须在 `X-Marvis-Token` 请求头中带上该值，否则返回 403。未设置时行为不变（当前单人体验不受影响）。

```bash
export MARVIS_LOCAL_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
marvis serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

浏览器打开首页（`GET /`）时，服务端会把该令牌注入页面（`<body data-marvis-local-token>`），前端 `api()` 封装自动在后续所有非 `GET` 请求上回传 `X-Marvis-Token`。该令牌只在本机客户端访问首页时下发，即使同时开启了 `MARVIS_ALLOW_REMOTE_READ`，远程客户端也读不到它。

这只是一层轻量防护：任何能读到 `MARVIS_LOCAL_TOKEN` 环境变量或首页 HTML 源码的本机用户仍然可以拿到令牌。真正的隔离需要更彻底的方案（例如监听 Unix domain socket 并设置 0700 权限，目前尚未实现）或系统级用户隔离（容器/虚拟机）。

### `MARVIS_ALLOW_REMOTE_READ`：允许非本机客户端只读访问

默认情况下，非本机（非回环地址）客户端只能访问 `/`、`/api/health` 和 `/static/`。设置为 `1`/`true` 后，非本机客户端可以读取任务、数据集等只读 API，但系统设置（`/api/settings*`）、品牌配置（`/api/branding`、`/branding/*`）和 `MARVIS_LOCAL_TOKEN` 本身依然只对本机客户端可见。所有写操作（非 `GET`）无论是否开启该变量，始终只允许本机客户端发起。

```bash
export MARVIS_ALLOW_REMOTE_READ=1
```

### `MARVIS_TRUSTED_PROXY_HOSTS`：反向代理场景下的真实客户端识别

如果通过 JupyterHub 的 `/proxy/<port>/` 或其他同机反向代理访问，直连的 TCP 对端会是代理自身的回环地址，这会让每个远程请求都"看起来"像本机请求。将代理自身的地址加入该变量后，中间件会改为信任 `X-Forwarded-For` 头中的真实客户端地址来做本地/远程判定，避免代理误把所有远程访问放行为本地权限：

```bash
export MARVIS_TRUSTED_PROXY_HOSTS="127.0.0.1"
```

多个可信代理地址用逗号分隔。未在该列表中的回环对端携带的转发头会被直接忽略（fail closed），不会被当作本地请求。

### 三者的组合建议

- 单人笔记本：三者都不需要设置，维持当前行为。
- 团队共享的 JupyterHub 服务器：至少设置 `MARVIS_LOCAL_TOKEN`；如果通过反向代理访问，还需要设置 `MARVIS_TRUSTED_PROXY_HOSTS`；`MARVIS_ALLOW_REMOTE_READ` 按是否需要跨机器只读查看再决定，不开启也不影响本机通过代理正常使用。

## 多 worktree / 多版本同时启动

多个 worktree 同时启动时，端口和 workspace 都要分开，避免访问错版本或共用 SQLite/任务产物。profile 会自动选择默认值：

```bash
# 稳定 main 演示
marvis serve --profile main
# http://127.0.0.1:8000, workspace ./workspace-main

# V2 开发或对比
marvis serve --profile v2
# http://127.0.0.1:8200, workspace ./workspace-v2
```

显式参数优先：

```bash
marvis serve --profile v2 --port 8217 --workspace ./custom-workspace
```

## 升级

GitHub clone 安装的用户可以在干净的 `main` 分支运行：

```bash
marvis update
```

该命令会执行：

```text
git fetch origin
git pull --ff-only origin main
python -m pip install -e . --no-deps
```

默认升级只刷新 MARVIS 自身的 editable 安装，不重新解析整个 Python 环境依赖，避免 Windows/Anaconda `base` 中的 Spyder、Streamlit、conda-repo 等包被 pip 连带升级或降级。

如果从 Anaconda/conda `base` 中运行 `marvis update`，命令会自动创建或复用专用 `marvis` 环境，并把安装写到该环境：

```text
conda create -y -n marvis python=3.12 pip
conda run -n marvis python -m pip install -e .
```

更新完成后仍然使用单行命令启动：

```bash
marvis
```

`base` 中的 `marvis` 入口会自动把运行命令代理到专用环境。可用 `marvis update --env-name <name>` 指定其他专用环境名。如果新版本确实新增运行时依赖，请在专用 MARVIS 环境中运行 `marvis update --with-deps`，不要在 `base` 里刷新依赖树。

如果已跟踪文件有未提交改动，升级会被拒绝。先 commit、stash 或备份这些 tracked 改动后再重新运行。未跟踪的本地文件允许保留，除非 Git 判断本次 pull 会覆盖它们。

如果当前旧版本还没有 `marvis update`，第一次升级需要在仓库目录手动执行一次：

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps
```

如果旧版本安装在 Anaconda `base`，第一次手动升级先只安装轻量 MARVIS 入口，再让 `marvis update` 准备专用环境：

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps
marvis update
marvis
```

完成后，后续升级可使用 `marvis update`。

## 备份与迁移

所有任务、审计记录、实验与记忆都存放在单个 SQLite 文件（`workspace/marvis.sqlite`，WAL 模式）加上 `workspace/` 下的文件树中。**服务运行时直接 `cp -r workspace/` 是不安全的**：WAL 模式下最近提交的事务可能还没有被 checkpoint 回主数据库文件，naive 拷贝会得到一份「看起来完整、实际缺尾」的数据库副本。

使用内置的备份命令，它通过 SQLite 在线备份 API（`sqlite3.Connection.backup()`）生成一致性快照，可以在服务运行时安全执行：

```bash
marvis backup --workspace ./workspace --out marvis-backup-2026-07-02.tar.gz
```

默认不包含 `workspace/datasets`（体积较大且可从原始文件重新生成）；如需一并备份：

```bash
marvis backup --workspace ./workspace --out full-backup.tar.gz --include-datasets
```

恢复到新目录（目标目录必须为空，否则需要 `--force` 覆盖）：

```bash
marvis restore marvis-backup-2026-07-02.tar.gz --workspace ./workspace-restored
```

恢复后可直接 `marvis serve --workspace ./workspace-restored` 启动；已有的启动期 reconcile 逻辑会像处理一次非正常关机那样清理残留的未完成写入产物。

## 直接 CLI 跑流水线（无需 Web）

```bash
# 1. 在 Web 页面或 API 创建任务，拿到 task_id
# 2. CLI 跑当前内置的模型验证流水线
marvis validate <task_id> \
    --workspace ./workspace
```

兼容入口：`python -m marvis validate <task_id> --workspace ./workspace`。

## 当前内置模型验证流程

1. 创建任务：填写模型名/版本/验证人、material 目录、算法等项目信息
2. 平台 SCAN：识别 notebook、样本、PMML、数据字典
3. 平台 NOTEBOOK：执行 notebook 副本（注入 head + tail cell），提取运行时契约和内存模型分
4. 平台 COMPUTE：用内存模型分与提交 PMML 分数做一致性验证，并计算基本信息、效果、压力测试
5. 平台 ARTIFACTS：写出 `outputs/validation.xlsx` 和 `outputs/validation_report.docx`

## 产出物位置

```text
workspace/tasks/<task_id>/
  execution/
    prepared.ipynb       注入 head + tail cell 后的 notebook 副本
    executed.ipynb       执行完的 notebook
    runtime_contract.json 平台从 notebook 读取的 RMC 契约
    code_model_scores.csv RMC_SCORE_FN 生成的内存模型分
    feature_importance.csv 可选，平台从 RMC_FEATURE_IMPORTANCE 提取
    model_params.json      可选，平台从 RMC_MODEL_PARAMS 提取
    model_meta.json        报告兼容用的特征重要性 + 参数汇总
    notebook_steps.json    Markdown 标题步骤与 cell 执行证据
    notebook.log
  outputs/
    validation.xlsx      带格式的 Excel
    validation_report.docx
  images/                Word 用的 matplotlib PNG
```

## Word 模板位置

`workspace/report_templates/04_贷前评分卡MOB3验证模板_带占位符.docx`

模板里使用 `{{TEXT:key}}` 和 `{{IMAGE:key}}` 占位符，平台会自动替换。

## 真实样例手工验收

样例目录：`/path/to/sample-credit-risk-project`

步骤：
1. 启动平台 (`serve`)
2. 创建任务，source_dir 填上述路径，算法 lgb
3. 触发 validate (Web 上点击或 CLI)
4. 确认 outputs 里两份文件生成
5. 打开 Excel/Word，对比原 `04_验证数据汇总表.xlsx` 与 `04_*验证文档*.docx`，指标一致即通过

## 开发人员相关

如果你是建模人员，要让你的 notebook 能被当前模型验证工作流执行，请看 [docs/notebook_contract.md](notebook_contract.md)。
