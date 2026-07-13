<p align="center">
  <img src="marvis/static/brand/marvis-workspace-logo.png" alt="MARVIS-Agent V2 logo" width="156" />
</p>

<h1 align="center">MARVIS-Agent V2</h1>

<p align="center">
  面向模型验证、数据处理、特征分析、模型开发、策略和 Vintage 工作流的本地优先信贷风控 Agent 工作台。
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="README.zh-CN.md"><strong>中文</strong></a>
</p>

---

MARVIS-Agent V2 是当前主线的可用 Agent 工作台。它保留本地文件、本地运行环境和可审计证据优先的边界，并在 V1.1 已稳定的模型验证工作流之外，继续扩展数据处理、特征分析、模型开发、策略和 Vintage 等信贷风控任务。

V2 不是只有运行时外壳：欢迎页露出的每个任务入口都是真实可用的端到端工作流，包含人在环确认、工具执行、结构化结果展示、下载或报告，以及可审计历史。截至 V2.0，这覆盖数据拼接、特征分析、模型开发与交付、打分与监控、策略开发（分数带/规则挖掘/版本化采纳）、组合分析、额度定价与即席问数——完整证据链见 `docs/plans/v2-master-backlog.md` 与 `docs/reviews/`。

当前 checkout 的状态：

- **模型验证**：继续保留 V1.1 已稳定的手动模式和 Agent 辅助验证路径。
- **数据处理、特征分析、模型开发**：是当前 V2 主线，围绕 Plugin/Tool/Workflow 运行时和任务级 Agent 流程推进。
- **策略、监控、组合分析与 Vintage 工作流**：已端到端接通（S1-S6 批次），每个关键门带红旗清单确认。

## 你可以获得什么

- **本地优先执行**：在自己的机器或服务器 workspace 中启动平台。
- **任务级 Agent 工作台**：围绕对话、确认门和右侧执行上下文推进信贷风控任务。
- **Plugin/Tool/Workflow 运行时**：以 schema、权限、执行日志和可审计产物管理内置或安装的能力包。
- **Notebook 验证运行时**：保留 V1.1 验证 Notebook 和下游指标的可复现能力，并让 V2 工作流围绕它继续扩展。
- **可配置品牌**：把私有客户或机构品牌配置留在源码之外。
- **适合开源的默认品牌**：删除本地 branding 配置后，应用会自动回退到公开 MARVIS 品牌。

## 核心文档

- [Roadmap](docs/roadmap.md)：当前 V2 平台地图、V1 兼容边界、未来 V3/V4 方向和 Plugin/Tool/Hook/Workflow 术语。
- [Versioning](docs/versioning.md)：发布 helper、tag、版本更新和 forward-port 规则。
- [Notebook contract](docs/notebook_contract.md)：当前模型验证 Notebook 运行契约。
- [Design](DESIGN.md)：产品体验和 UI/UX 决策来源。

## 公开默认品牌

- 平台名称：`MARVIS-全能风控智能体`
- 主题色：中性炭灰（`#303034`）
- 默认主 logo：`marvis/static/brand/marvis-workspace-logo.png`
- 默认 favicon：`marvis/static/brand/marvis-favicon.png`

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

- Windows 用户可以使用后续 release 附带的一键安装包；安装包内置私有 Python runtime 和 Java runtime。
- 源码安装需要 Python 3.11 或更新版本。全新本地安装推荐 Python 3.12。
- 当前已验证的源码安装工作流覆盖 macOS 和 Linux。
- 源码安装如果需要 PMML 打分，需要安装与 `pypmml` 兼容的 Java 运行时。
- Node.js 只用于前端语法检查；应用本身通过 FastAPI 提供静态 HTML/CSS/JS。

## Windows 一键安装包

个人 Windows 电脑优先使用 release 中的安装包：

```text
MARVIS-Setup-<version>-win-x64.exe
```

用户机器不需要预装 Python、Java、Git、conda、WSL 或 Docker。安装包按当前用户安装，双击后启动本地 MARVIS 服务，并打开 `http://127.0.0.1:8000/`。Windows 安装包构建资产位于 `packaging/windows/`。

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

# V2 开发 worktree
marvis serve --profile v2
# http://127.0.0.1:8200, workspace ./workspace-v2
```

显式参数优先级更高：

```bash
marvis serve --profile v2 --port 8217 --workspace ./custom-workspace
```

## 升级

如果安装来源是 GitHub clone，并且当前在干净的 `main` 分支上，可以运行：

```bash
marvis update
```

该命令会执行 `git fetch origin`、`git pull --ff-only origin main`，然后刷新 MARVIS 的 editable 安装，但不重新解析整个 Python 环境依赖：

```bash
python -m pip install -e . --no-deps
```

如果在 Anaconda/conda `base` 里运行 `marvis update`，MARVIS 会自动创建或复用专用的 `marvis` 环境，并把安装写入该环境，不修改 `base`。更新完成后仍然只需要输入一个命令启动：

```bash
marvis
```

`base` 里的 `marvis` 入口会自动把运行命令代理到专用环境。这个默认行为是为了兼容 Anaconda 和 Windows 机器上同一环境内其他包的严格依赖约束。可以用 `--env-name <name>` 指定其他专用 conda 环境。如果未来版本新增运行时依赖，请在专门的 MARVIS 环境里运行 `marvis update --with-deps`，不要在 Anaconda `base` 环境里刷新依赖树。

如果已跟踪文件有未提交改动，`marvis update` 会拒绝继续。请先 `git commit`、`git stash` 或备份这些 tracked 改动后再升级。未跟踪的本地文件允许保留，除非 Git 判断本次 pull 会覆盖它们。

如果当前安装的旧版本还没有 `marvis update`，第一次升级需要在仓库目录手动执行一次：

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps
```

如果当前在 Anaconda `base`，第一次手动升级先只安装轻量 MARVIS 入口，再让 `marvis update` 准备专用环境：

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps
marvis update
marvis
```

完成后，后续升级就可以使用 `marvis update`。

如果你明确运行的是 V2 分支或 worktree，请显式指定分支：

```bash
marvis update --branch <v2-branch>
```

## 测试

```bash
python -m pytest -q
ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
```

日常小改动可运行 `scripts/check --affected`，按 git diff 选择相关测试；无法安全映射时会自动退回 fast 层。需要覆盖全部非重型用例时运行 `scripts/check --fast`，它会排除 `slow`、`e2e` 和 `llm` 用例。发布前仍运行不带这两个参数的完整 `scripts/check`。

## 发布推送

发布新的公开版本时，使用 release helper，不要直接裸跑 `git push`。执行时机是：功能、修复或文档改动已经验证并 commit 之后。脚本要求工作区干净，并会单独创建一个版本 bump commit 和 annotated tag。

```bash
python scripts/release_push.py --bump patch
```

这个 helper 会更新版本元数据、创建发布 commit、创建带注释的 `Vx.y.z` tag，并推送 `main` 和 tag。完整发布顺序和版本规则见 `docs/versioning.md`。

## License

本项目使用 MIT License。详见 `LICENSE`。
