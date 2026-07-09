# 版本与发布规则

本文档定义版本命名、发布推送、tag 和 forward-port 规则。产品阶段和术语统一放在 `docs/roadmap.md`。

## 当前版本线

- **V2.x**：当前主线；本地优先、可治理的多工作流信贷风控 Agent 平台，覆盖数据处理、特征分析、模型开发、模型验证、策略、Vintage/风险分析、监控和组合分析等 workflow。
- **V1.1.x**：模型验证兼容基础能力，已包含 Agent Memory Foundation；记忆只辅助解释、建议、历史对比和审计，不改变确定性验证结果。
- **V1.0.x**：上一条稳定模型验证线。
- **V3**：后续平台治理与扩展深化，例如多用户审批、调度告警、第三方能力包治理、实时评分和更强执行隔离。
- **V4**：后续 Strategy and Portfolio Pack 深化，例如更完整的经营策略库、周期化组合监控、挑战者策略和收益风险复盘闭环。
- **公开默认版**：无私有 `workspace/branding/` 配置时的开源安全运行形态。

## 版本号格式

产品版本使用语义化版本号：

```text
V<MAJOR>.<MINOR>.<PATCH>
```

示例：

- `V1.1.1`：V1 兼容线的历史 patch 版本。
- `V1.2.0`：同一 major 内新增用户可见能力的 minor 版本。
- `V2.1.8`：V2 主线的稳定 patch 版本。
- `V3.0.0-alpha.1`：后续平台治理或扩展阶段的预发布版本。

含义：

- `MAJOR`：产品线或架构阶段变化，例如 V1 到 V2。
- `MINOR`：同一 major 内新增用户可见能力，同时保持既有流程兼容。
- `PATCH`：缺陷修复、兼容性修复、文档修正、发布工具修复、小体验修正。
- `alpha` / `beta` / `rc` 等预发布后缀用于尚不能作为稳定公开版本的节点。

## 什么时候更新版本

版本更新以“形成可识别、可回滚、可演示或可发布的节点”为边界。

需要更新版本或发布记录的情况：

- 对外发布到 GitHub 或其他远端。
- 给稳定演示线或交付版本打 tag。
- 用户可见 bugfix 要成为稳定 patch。
- 新增、删除或改变用户可见功能、报告口径、API 契约、Notebook 契约、Agent 行为、记忆行为、Plugin 行为或文档承诺。

不需要更新版本的情况：

- 本地试验。
- feature branch 中间提交。
- 未发布的临时调试。
- 纯格式化或小内部整理，除非它本身就是一个交付节点。

## 发布 helper

对外发布不要使用裸 `git push` 手工移动 tag。统一使用 `scripts/release_push.py`，让版本元数据、release commit、annotated tag 和远端推送保持一致。

默认 patch 发布：

```bash
python scripts/release_push.py --bump patch
```

minor / major 发布：

```bash
python scripts/release_push.py --bump minor
python scripts/release_push.py --bump major
```

指定版本：

```bash
python scripts/release_push.py --version V1.1.0
```

指定 V2 预发布版本：

```bash
python scripts/release_push.py --version V2.0.0-alpha.1
```

预发布后缀仅支持 `alpha.N`、`beta.N`、`rc.N`。tag 使用 SemVer 风格
（例如 `V2.0.0-alpha.1`），Python 包元数据会自动写成 PEP 440 风格
（例如 `2.0.0a1`）。

预览：

```bash
python scripts/release_push.py --bump patch --dry-run
```

只在本地创建 release commit 和 tag，不推送：

```bash
python scripts/release_push.py --bump patch --no-push
```

执行时机：先完成并验证普通功能、修复或文档改动，使用普通 commit 提交这些改动；确认 `main` 工作区干净后，再运行 release helper。不要在有未提交改动时运行 release helper，也不要先创建 release commit 再回头补业务 commit。

推荐顺序：

```bash
git status --short
git diff --check
git log -1 --oneline
python scripts/release_push.py --version V1.1.0
```

脚本会执行：

1. 从最新稳定 `V<MAJOR>.<MINOR>.<PATCH>` tag 计算下一版本（或使用 `--version` 指定稳定/预发布版本），并检查目标 tag 不存在。
2. 要求工作区干净并在目标分支上。
3. 更新 `pyproject.toml`、README、runbook、Notebook 要求等版本元数据。
4. 创建版本 bump commit。
5. 创建 annotated tag。
6. 除非使用 `--no-push`，否则 push `main` 和新 tag。

发布 tag 视为不可变。发布后发现问题时，修复后创建下一个 patch 版本，不移动旧 tag。

### Windows 安装包

发布 tag 推送完成后，Windows 个人电脑安装包作为 release 附加工件构建，不由 `scripts/release_push.py` 直接生成。构建机必须是 Windows x64，并安装 Python、micromamba 和 Inno Setup：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

产物为 `dist\windows\MARVIS-Setup-<version>-win-x64.exe` 和同名 `.sha256`。安装包内置私有 Python runtime 与 OpenJDK runtime，用户机器不需要预装 Python、Java、Git、conda、WSL 或 Docker。构建脚本会先运行 `marvis version`、核心 import 和内置 Java smoke check；上传 release 前还应在干净 Windows 用户环境中双击安装并确认 `/api/health` 和首页可打开。

## 并行维护与 forward-port

V1 可以保持稳定，同时 V2+ 在独立 worktree 或分支开发。V2 及后续版本不能丢失 V1 已确认行为。

规则：

1. **保持 V1 兼容行为稳定**

   V1.1 模型验证契约仍是兼容边界：Notebook 变量、PMML 对比、确定性指标、报告产出、手动模式和 Agent 辅助验证都不能被 V2 平台化改动破坏。

2. **V2 是当前主线平台**

   V2 可以改 Agent planner、Plugin/Tool/Hook runtime、Workflow 执行、扩展输出和新增业务 workflow。如果 V2 改动影响 V1 兼容行为，必须在设计/spec/路线文档中写清迁移理由，并补回归测试。

3. **V1 fix 必须 forward-port**

   只要 V1 修复的是稳定流程、报告输出、Notebook 契约、前端状态、下载、任务生命周期、branding、发布工具、记忆行为或部署兼容性问题，V2+ 存在时都应同步。

4. **优先使用 merge 或 cherry-pick**

   不要手动复制代码。

   ```bash
   # 在 V2 worktree 中同步一批 V1 修复
   git switch <v2-branch>
   git merge <v1-stable-branch>

   # 在 V2 worktree 中只同步某个 V1 修复
   git switch <v2-branch>
   git cherry-pick <v1-fix-commit>
   ```

5. **冲突时先保行为**

   - V1 用户可见行为和回归测试不能丢。
   - V2 可以用新架构实现同一行为。
   - 如果 V2 有意改变 V1 行为，必须写明理由并测试新契约。

6. **测试是冲突裁决依据**

   forward-port 后要跑对应回归测试。没有测试的 V1 fix，应先补最小有用测试。

## Worktree 规则

长期并行维护 major line 时使用独立 worktree，避免服务、缓存和未提交改动互相污染。

推荐形态：

```text
/path/to/marvis-v1   V1 稳定线和 V1.1 memory 线
/path/to/marvis-v2   V2 plugin/tool runtime 线
```

维护并行产品线时明确路径：

```text
只在 /path/to/marvis-v1 开发 V1.1 memory。
只在 /path/to/marvis-v2 开发 V2 plugin/tool runtime。
```

不要在一个工作区中来回切换 major line，除非任务本身就是 forward-port 或冲突处理。

## Branding 与公开版本

只改 branding 不创建新产品版本。Branding 从 workspace-local 文件运行时配置。

规则：

- 没有 branding 配置时使用公开 MARVIS 默认值。
- 本地 branding 配置路径为 `workspace/branding/brand.json`。
- 私有 logo、机构名、内网地址、客户样例、本地 branding 资产不得提交到公开仓库。
- 源码默认资产必须保持可公开。

## 发布前检查

代码发布前至少执行：

```bash
git status --short
scripts/check
```

如果只是文档修订，通常只需要 `scripts/check --skip-pytest --skip-ruff --skip-node`。最终说明中写明未运行代码测试的原因。

日常本地迭代可用 `scripts/check --fast` 只跑快层测试（`-m "not slow and not e2e"`，排除真训练/真子进程/浏览器 e2e 用例，明显更快）；发布前检查仍需跑不带 `--fast` 的全量。可选加 `--audit` 跑 `pip-audit` 依赖 CVE 扫描（未安装时打印跳过原因，不会失败）。

本机开发工作区可使用 `conda run -n py_313 python -m pytest ...` 和
`conda run -n py_313 python -m ruff check ...` 执行同一组检查；公开 README/runbook 示例仍使用普通
`python`，避免把个人 conda 环境写成用户安装前提。

本机如果需要使用指定环境，可以显式传入 Python：

```bash
PYTHON=/opt/miniconda3/envs/py_313/bin/python scripts/check
```
