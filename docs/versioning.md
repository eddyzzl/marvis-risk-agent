# 版本命名与并行维护规则

本文档定义平台版本、分支、worktree 和 forward-port 规则。目标是让 V1 可稳定演示和回滚，同时允许 V2/P2 继续开发，不因为并行开发造成修复丢失或行为口径混乱。MARVIS 的长期产品边界是全能信贷风控智能体，模型验证只是 V1 已落地的首个内置工作流。

## 当前版本线

- **V1**：当前已基本完成的 MARVIS 首个公开稳定版本。核心可演示能力是模型验证工作流；后续只做缺陷修复、兼容性修复、文档修正和不改变核心能力边界的小幅改进。
- **V2**：当前 P2 开发线，重点是 Agent Skill Runtime、Agent 自主编排，以及面向建模、分析、策略、验证等信贷风控任务的可插拔能力扩展。
- **公开开源默认版**：无私有 branding 配置时的默认运行形态。默认品牌为 `MARVIS-全能风控智能体`，默认主题色为黑色，使用内置 MARVIS logo/favicon。公开版不再作为单独功能版本线维护。

## 版本号规则

平台使用语义化版本号：

```text
V<MAJOR>.<MINOR>.<PATCH>
```

示例：

- `V1.0.0`：V1 首个可演示稳定版本。
- `V1.0.1`：V1 的缺陷修复版本。
- `V1.1.0`：V1 内新增小能力，但不改变主要架构和使用方式。
- `V2.0.0-alpha.1`：V2/P2 的第一个预发布验证版本。

版本含义：

- `MAJOR`：阶段性产品线变化，例如 V1 到 V2。新增大架构、大能力边界或重大使用方式变化时提升。
- `MINOR`：同一大版本内新增可见能力，但保持已有用户流程兼容。
- `PATCH`：缺陷修复、兼容性修复、文档修正、小范围体验修正，不引入新的主要能力。
- 预发布后缀：用于还不能作为稳定演示版本的开发节点，例如 `alpha`、`beta`、`rc`。

## 什么时候更新版本

不要求每次改代码都更新版本号。版本更新以“形成可识别、可回滚、可演示或可发布的节点”为边界。

需要更新版本或补充版本记录的情况：

- 合入稳定演示线之前，例如合入 `main`、`demo-stable` 或 V1 稳定分支。
- 对外 push、打 tag、发压缩包、给他人演示或交付之前。
- 修复了用户可见 bug，且这个修复会成为 V1/V2 的一个可回滚节点。
- 新增、删除或改变用户可见功能、报告口径、API 契约、Notebook 契约、Agent 行为或文档承诺。

不需要立即更新版本的情况：

- 本地试验性改动。
- feature branch 内的中间 commit。
- 未合入稳定线、未对外交付的临时调试提交。
- 纯格式化或无用户影响的小整理，除非它会作为单独交付节点。

建议发布动作：

```bash
git status --short
git diff --check
python scripts/release_push.py --bump patch
```

如后续增加 `CHANGELOG.md` 或包内 `__version__`，发布前应同步更新。

## 发布推送与 tag 自动更新

对外发布不要使用裸 `git push` 手工移动 tag。发布时统一使用 `scripts/release_push.py`，让版本元数据、release commit、annotated tag 和远端推送保持一致。

默认 patch 发布：

```bash
python scripts/release_push.py --bump patch
```

指定 minor / major：

```bash
python scripts/release_push.py --bump minor
python scripts/release_push.py --bump major
```

指定明确版本：

```bash
python scripts/release_push.py --version V1.1.0
```

脚本会执行以下动作：

1. 要求工作区干净，避免把未提交业务改动混进版本 bump。
2. 从最新 `V<MAJOR>.<MINOR>.<PATCH>` tag 计算下一个版本，或使用 `--version` 指定值。
3. 自动更新 `pyproject.toml`、README、runbook 和 Notebook 要求文档中的当前发布版本。
4. 创建一个版本 bump commit。
5. 创建 annotated tag。
6. push `main` 和新 tag 到远端。

发布前可先预览：

```bash
python scripts/release_push.py --bump patch --dry-run
```

如果只想在本地生成版本 commit 和 tag、不推送远端：

```bash
python scripts/release_push.py --bump patch --no-push
```

发布 tag 视为不可变。发布后发现问题时，不要移动旧 tag；修复后创建下一个 patch 版本。

## V1/V2 并行维护与 forward-port 规则

V1 和 V2 可以并行开发，但 V2 最终必须包含 V1 后续所有必要 bug 修复。规则如下：

1. **V1 是当前稳定演示线**

   V1 主要用于演示、回归、修 bug 和保持当前成果可用。不要把 V2 的未稳定架构改动直接混入 V1。

2. **V2 是未来能力开发线**

   V2 可以改 Agent、skill runtime、模型族抽象和任务编排，但不能丢失 V1 已确认的用户可见行为。V2 改动如果影响 V1 已有流程，必须有明确迁移理由和回归测试。

3. **V1 bugfix 必须 forward-port 到 V2**

   只要 V1 修复的是稳定流程、报告输出、Notebook 契约、前端状态、下载、任务生命周期或部署兼容性问题，修复后都要同步到 V2。否则 V2 开发完成后会重新带回旧 bug。

4. **优先用 git 合并或 cherry-pick，不手动复制代码**

   同步整批 V1 修复时优先用 merge；只同步单个修复时用 cherry-pick。

   ```bash
   # 在 V2 worktree 中执行：同步一批 V1 修复
   git switch <v2-branch>
   git merge <v1-stable-branch>

   # 在 V2 worktree 中执行：只同步某个 V1 修复提交
   git switch <v2-branch>
   git cherry-pick <v1-fix-commit>
   ```

5. **冲突时先保行为，再保架构**

   如果 V1 和 V2 都改了同一块代码，判断顺序是：
   - V1 已验证的用户可见行为和回归测试不能丢。
   - V2 可以用新的架构实现同一个行为，不要求逐行保留 V1 写法。
   - 如果 V2 有意改变 V1 行为，必须写入设计文档或版本记录，并补上迁移说明和测试。

6. **测试是冲突裁决依据**

   V1 修复同步到 V2 后，要在 V2 上跑对应回归测试。没有测试的 V1 bugfix，应先补最小回归测试，再 forward-port。

7. **每条长期维护线使用独立 worktree**

   同时维护 V1、V2 等功能开发线时，使用不同目录承载不同 worktree，避免在一个目录里频繁切分支导致运行中的服务、缓存和未提交改动互相污染。品牌差异默认通过本地 branding 配置解决，不再要求单独创建 OSS 去品牌 worktree。

   推荐形态：

   ```text
   /path/to/riskmodel_checker-v1   V1 稳定线
   /path/to/riskmodel_checker-v2   V2/P2 开发线
   ```

8. **Codex 会话必须绑定明确 worktree**

   要求 Codex 开发时，应明确指定路径，例如：

   ```text
   请在 /path/to/riskmodel_checker-v1 里修复这个 V1 bug。
   请在 /path/to/riskmodel_checker-v2 里开发 P2 skill runtime。
   ```

   不要在同一个任务中让 Codex 来回切换 V1/V2 worktree，除非任务本身就是 forward-port 或冲突处理。

## 品牌定制与公开版本

品牌定制只修改 logo、favicon/web logo、主题色、平台名称、浏览器页面标题等展示元素，功能不变时不需要做多租户，也不需要单独升一个产品大版本。推荐做法是通过运行时配置切换品牌资产，而不是改源码或维护一套去品牌分支。

默认规则：

- 无 branding 配置时展示公开默认品牌：平台名称和浏览器标题为 `MARVIS-全能风控智能体`，主题色为黑色，logo/favicon 使用内置 MARVIS 默认资产。
- 有 branding 配置时，运行时覆盖 logo、favicon/web logo、primary color、platform name 和 browser title；当前配置文件路径为 `workspace/branding/brand.json`，资产放在同一文件夹下。
- primary color 至少驱动两个创建任务按钮和 Agent 对话发送按钮；后续新增主操作应复用同一个 brand token。
- 私有 branding 配置和资产应放在 workspace-local 文件夹中，例如 `workspace/branding/`，并被 `.gitignore` 排除。
- 仓库可以提交安全的示例配置，但不能提交机构私有 logo、机构名、内网地址、样例数据或客户材料。
- 公开到 GitHub 前，删除本地 branding 配置文件夹后应用必须能退回 MARVIS 默认品牌；这只在源码默认资产本身不含私有元素时成立。

## 发布前检查

发布、演示或合入稳定线前，至少执行：

```bash
git status --short
git diff --check
python -m pytest -q
ruff check riskmodel_checker tests --extend-exclude '*.ipynb'
node --check riskmodel_checker/static/app.js
```

如果只是文档修订，可以只运行 `git diff --check`，并说明未运行代码测试的原因。
