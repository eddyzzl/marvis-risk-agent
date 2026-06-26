<p align="center">
  <img src="marvis/static/brand/marvis-logo.png" alt="MARVIS-Agent logo" width="148" />
</p>

<h1 align="center">MARVIS-Agent</h1>

<p align="center">
  面向信贷风控、模型开发、模型验证、数据处理、特征分析和策略工作流的本地优先智能体平台。
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="README.zh-CN.md"><strong>中文</strong></a>
</p>

---

MARVIS-Agent 面向需要本地文件、本地运行环境和可审计证据的信贷风控工作，覆盖模型开发、模型验证、数据处理、特征衍生、特征分析、策略生成、策略验证、监控和受控任务自动化。

当前 V1.1.6 版本已经稳定落地第一个内置工作流：模型验证。平台可以执行基于 Notebook 的验证任务，生成结构化证据，并通过 Agent 模式辅助起草 Excel/Word 验证报告。模型验证只是第一个工作流，不是产品边界。

后续路线见 [docs/roadmap.md](docs/roadmap.md)：V1.1 已包含用于历史验证指标对比的 Agent Memory Foundation，V2 增加 Agent Plugin/Tool Runtime，后续版本在这个运行时之上扩展建模和策略能力包。

## 你可以获得什么

- **本地优先执行**：在自己的机器或服务器 workspace 中启动平台。
- **Agent 辅助工作流**：围绕信贷风控任务沉淀结构化证据并辅助报告起草。
- **Notebook 验证运行时**：执行验证 Notebook 和下游指标计算，生成可复现产物。
- **可配置品牌**：把私有客户或机构品牌配置留在源码之外。
- **适合开源的默认品牌**：删除本地 branding 配置后，应用会自动回退到公开 MARVIS 品牌。

## 核心文档

- [Roadmap](docs/roadmap.md)：V1/V1.1/V2/V3/V4 路线和 Plugin/Tool/Hook/Workflow 术语。
- [Versioning](docs/versioning.md)：发布 helper、tag、版本更新和 forward-port 规则。
- [Notebook contract](docs/notebook_contract.md)：当前模型验证 Notebook 运行契约。
- [Design](DESIGN.md)：产品体验和 UI/UX 决策来源。

## 公开默认品牌

- 平台名称：`MARVIS-全能风控智能体`
- 主题色：中性炭灰（`#343438`）
- 默认 logo 和 favicon：`marvis/static/brand/`

## Branding 配置

私有或客户专属 branding 不会提交到公开仓库。需要本地品牌时，创建一个已被忽略的 workspace 配置：

```text
workspace/branding/brand.json
```

示例：

```json
{
  "platform_name": "本地信贷风控智能体",
  "browser_title": "本地信贷风控工作台",
  "primary_color": "#1f6feb",
  "logo": "private-logo.svg",
  "favicon": "private-logo.svg"
}
```

把引用到的 logo 文件放在 `brand.json` 同级目录。缺少 `workspace/branding/` 时，应用会回退到公开 MARVIS 品牌。

更多细节见 `docs/branding.md`。

## 本地部署要求

- Python 3.11 或更新版本。全新本地安装推荐 Python 3.12。
- 当前已验证的本地工作流覆盖 macOS 和 Linux。
- 如果需要 PMML 打分，需要安装与 `pypmml` 兼容的 Java 运行时。
- Node.js 只用于前端语法检查；应用本身通过 FastAPI 提供静态 HTML/CSS/JS。

## 从 GitHub 安装

用户从 GitHub clone 后，在仓库目录内安装。可以使用任意环境名称。例如使用 `venv`：

```bash
git clone https://github.com/eddyzzl/marvis-risk-agent.git
cd marvis-risk-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

或者使用 conda：

```bash
git clone https://github.com/eddyzzl/marvis-risk-agent.git
cd marvis-risk-agent
conda create -n marvis python=3.12
conda activate marvis
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## 本地启动

安装后可以直接运行：

```bash
marvis
```

默认等价于：

```bash
marvis serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

然后打开 `http://127.0.0.1:8000/`。

V1 中仍保留 Python 模块名 `marvis`，用于兼容当前验证运行时。以下旧入口仍可用：

```bash
python -m marvis serve --host 127.0.0.1 --port 8000 --workspace ./workspace
marvis-risk-agent serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

## 材料目录

创建任务时，材料目录默认必须位于当前 `workspace` 或当前用户 home 目录下。Windows 用户如果把材料放在 D 盘、外接盘或其他不在 home 下的位置，需要启动前显式放行：

```powershell
$env:RMC_MATERIAL_ROOTS="D:\model_materials"
marvis serve --host 127.0.0.1 --port 8000 --workspace .\workspace
```

在 WSL2 中运行时，不要填写 `C:\...` 形式的 Windows 路径；请使用对应的 WSL 路径，例如 `/mnt/c/Users/<you>/Downloads/project`。

## 多 worktree / 多版本同时启动

多个 worktree 同时启动时必须使用不同端口和不同 workspace。可以使用 profile 自动选择默认值：

```bash
# 稳定 main 演示
marvis serve --profile main
# http://127.0.0.1:8000, workspace ./workspace-main

# V1.1 开发或对比
marvis serve --profile v1-1
# http://127.0.0.1:8001, workspace ./workspace-v1-1
```

显式参数优先级更高：

```bash
marvis serve --profile v1-1 --port 8017 --workspace ./custom-workspace
```

## 升级

如果安装来源是 GitHub clone，并且当前在干净的 `main` 分支上，可以运行：

```bash
marvis update
```

该命令会执行 `git fetch origin`、`git pull --ff-only origin main`，然后刷新 editable 安装：

```bash
python -m pip install -e .
```

如果已跟踪文件有未提交改动，`marvis update` 会拒绝继续。请先 `git commit`、`git stash` 或备份这些 tracked 改动后再升级。未跟踪的本地文件允许保留，除非 Git 判断本次 pull 会覆盖它们。

如果当前安装的旧版本还没有 `marvis update`，第一次升级需要在仓库目录手动执行一次：

```bash
git pull --ff-only origin main
python -m pip install -e .
```

完成后，后续升级就可以使用 `marvis update`。

## 测试

```bash
python -m pytest -q
ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
```

## 发布推送

发布新的公开版本时，使用 release helper，不要直接裸跑 `git push`。执行时机是：功能、修复或文档改动已经验证并 commit 之后。脚本要求工作区干净，并会单独创建一个版本 bump commit 和 annotated tag。

```bash
python scripts/release_push.py --bump patch
```

这个 helper 会更新版本元数据、创建发布 commit、创建带注释的 `Vx.y.z` tag，并推送 `main` 和 tag。完整发布顺序和版本规则见 `docs/versioning.md`。

## License

本项目使用 MIT License。详见 `LICENSE`。
