# MARVIS 本地运行手册（V1.1.4）

MARVIS 的产品边界是面向建模、分析、策略和验证的全能信贷风控智能体。当前 V1.1.4 公开版已经稳定落地的是模型验证工作流，因此本手册的流水线部分以该工作流为例。

## 本地部署要求

- Python 3.11 或更高版本，推荐新环境使用 Python 3.12。
- 当前验证过的本地运行环境为 macOS / Linux。
- 如需执行 PMML 打分，请安装与 `pypmml` 兼容的 Java Runtime。
- Node.js 只用于前端静态语法检查；运行 Web 服务不依赖前端构建。

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

## 多 worktree / 多版本同时启动

多个 worktree 同时启动时，端口和 workspace 都要分开，避免访问错版本或共用 SQLite/任务产物。profile 会自动选择默认值：

```bash
# 稳定 main 演示
marvis serve --profile main
# http://127.0.0.1:8000, workspace ./workspace-main

# V1.1 开发或对比
marvis serve --profile v1-1
# http://127.0.0.1:8001, workspace ./workspace-v1-1
```

显式参数优先：

```bash
marvis serve --profile v1-1 --port 8017 --workspace ./custom-workspace
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
python -m pip install -e .
```

如果本地有未提交改动，升级会被拒绝。先 commit、stash 或备份后再重新运行。

如果当前旧版本还没有 `marvis update`，第一次升级需要在仓库目录手动执行一次：

```bash
git pull --ff-only origin main
python -m pip install -e .
```

完成后，后续升级可使用 `marvis update`。

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
