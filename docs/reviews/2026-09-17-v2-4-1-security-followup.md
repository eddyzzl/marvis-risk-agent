# V2.4.1 依赖安全补丁

> 日期：2026-09-17。范围：Tornado 锁定版本与发布记录，不改变应用逻辑。

V2.4.0 发布后的 [CI 安全检查](https://github.com/eddyzzl/marvis-risk-agent/actions/runs/35185959598/job/105087970316)
发现锁文件中的 Tornado 6.5.7 命中 `PYSEC-2026-3928`、`GHSA-wwv5-g3v4-889x`
和 `GHSA-8423-8fgw-73vq`，并将 6.5.8 列为修复版本。
[上游 6.5.8 发布说明](https://www.tornadoweb.org/en/stable/releases/v6.5.8.html)
说明了表单参数数量、multipart 内存消耗和旧式 cookie 参数校验的安全修复。

通过 `uv lock --upgrade-package tornado==6.5.8` 更新该依赖及对应制品哈希，
其余依赖版本不变，不忽略漏洞、不降低检查级别、不移动已经公开的 V2.4.0 标签。
补丁版本仍由 `scripts/release_push.py` 创建和推送。

## 验证

- 与 CI 相同的锁文件导出与 pip-audit 2.10.1 检查退出码为 0，结果为
  `No known vulnerabilities found`。
- 与 CI 相同的完整递归 Bandit 1.9.4 基线检查退出码为 0；原基线未修改。
- `uv lock --check` 和 `git diff --check` 通过。
- [V2.4.0 完整门禁](2026-09-17-v2-4-0-release-readiness.md) 的 12,415 项通过结果
  来自已经安装 Tornado 6.5.8 的隔离运行环境；该环境中的 Tornado 在本轮测试前就已存在，
  不是在检查之后替换的版本。本补丁没有应用代码变化，因此未重复整套 1:43:05 的测试。
- 正式补丁 wheel 在版本提升后重新构建并核对；远端安全检查和其余 CI 状态以补丁提交的
  Actions 结果为准，不将本地扫描通过等同于远端 CI 全部完成。

本轮未修改日常 conda 环境。原生 Windows、真实 LLM、生产部署与业务签字的验证边界
保持不变。
