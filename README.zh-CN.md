<p align="center">
  <img src="riskmodel_checker/static/brand/marvis-logo.png" alt="MARVIS Risk Agent logo" width="148" />
</p>

<h1 align="center">MARVIS Risk Agent</h1>

<p align="center">
  面向建模、分析、策略和验证工作流的本地优先信贷风控智能体平台。
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="README.zh-CN.md"><strong>中文</strong></a>
</p>

---

MARVIS Risk Agent 面向需要本地文件、本地运行环境和可审计证据的信贷风控工作。长期产品方向是一个全能信贷风控智能体，覆盖模型开发、组合分析、策略评估、监控、验证和受控任务自动化。

当前 V1.1.0 版本已经稳定落地第一个内置工作流：模型验证。平台可以执行基于 Notebook 的验证任务，生成结构化证据，并通过 Agent 模式辅助起草 Excel/Word 验证报告。模型验证只是第一个工作流，不是产品边界。

后续路线见 [docs/roadmap.md](docs/roadmap.md)：V1.1 增加用于历史验证指标对比的 Agent Memory Foundation，V2 增加 Agent Plugin/Tool Runtime，后续版本在这个运行时之上扩展建模和策略能力包。

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
- 主题色：黑色
- 默认 logo 和 favicon：`riskmodel_checker/static/brand/`

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

可以使用任意环境名称。例如使用 `venv`：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

或者使用 conda：

```bash
conda create -n marvis python=3.12
conda activate marvis
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## 本地启动

```bash
python -m riskmodel_checker serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

V1 中仍保留 Python 模块名 `riskmodel_checker`，用于兼容当前验证运行时。如果已经以 editable 模式安装，也可以使用面向产品的命令别名：

```bash
marvis-risk-agent serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

然后打开 `http://127.0.0.1:8000/`。

## 测试

```bash
python -m pytest -q
ruff check riskmodel_checker tests --extend-exclude '*.ipynb'
node --check riskmodel_checker/static/app.js
```

## 发布推送

发布新的公开版本时，使用 release helper，不要直接裸跑 `git push`。执行时机是：功能、修复或文档改动已经验证并 commit 之后。脚本要求工作区干净，并会单独创建一个版本 bump commit 和 annotated tag。

```bash
python scripts/release_push.py --bump patch
```

这个 helper 会更新版本元数据、创建发布 commit、创建带注释的 `Vx.y.z` tag，并推送 `main` 和 tag。完整发布顺序和版本规则见 `docs/versioning.md`。

## License

本项目使用 MIT License。详见 `LICENSE`。
