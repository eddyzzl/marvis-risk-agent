# MARVIS-Agent Linux 源码部署 — 环境准备清单

**产品**：MARVIS-Agent V2.x（信贷风控本地 Agent）
**部署方式**：源码安装 + 本机 Web 服务（FastAPI）
**分工**：运维准备 OS / 系统包 / Python / Java / 账号 / 磁盘 / 网络与权限；业务侧自行安装应用并启动。

不需要 Node.js、前端构建。Docker 不是个人/默认路径；若 IT 要用容器自行包装，仍须满足本文中的运行时依赖。

相关文档：

- 本地运行手册：[`docs/runbook.md`](runbook.md)
- 安装与启动说明：[`README.zh-CN.md`](../README.zh-CN.md)
- 依赖声明：[`pyproject.toml`](../pyproject.toml)
- 参考锁定版本：[`uv.lock`](../uv.lock)

---

## 1. 机器与 OS

| 项 | 要求 | 原因 |
|----|------|------|
| OS | Linux x86_64（推荐 Ubuntu 22.04/24.04 或 RHEL/Rocky 8/9） | 源码安装路径已验证 macOS/Linux；wheel 多为 `manylinux2014_x86_64` |
| 架构 | 优先 x86_64；若必须 aarch64 需提前说明 | 避免装包失败 |
| glibc | ≥ 2.17（`manylinux2014`） | 二进制 wheel 基线 |
| 时区/语言 | UTF-8 locale（如 `en_US.UTF-8` 或 `zh_CN.UTF-8`） | Notebook / 路径 / 日志避免编码问题 |
| 用户 | 专用普通用户（如 `marvis`），不要用 root 跑服务 | 安全与文件权限 |
| SSH | 部署人可登录该用户，对安装目录可写 | 业务侧自行部署 |

### 建议配置

| 资源 | 最低 | 推荐 |
|------|------|------|
| CPU | 4 核 | 8 核+ |
| 内存 | 16 GB | 32 GB+（建模 + DuckDB JOIN + PMML JVM 并存） |
| 系统盘 | 20 GB 可用（装环境） | 40 GB+ |
| 数据盘（workspace） | 100 GB | 200 GB+（任务产物、数据集、DuckDB 临时文件） |
| GPU | 不需要 | 训练默认 CPU；Linux 上 xgboost 可能拉取 `nvidia-nccl-cu12`，不代表要装 CUDA 驱动 |

---

## 2. 系统级软件（运维安装）

### 2.1 必装

| 组件 | 版本 | 用途 |
|------|------|------|
| Git | 任意较新 | clone / 更新代码 |
| Python | **3.12.x**（硬性 `>=3.11`，强烈推荐 3.12；不要用系统 3.8/3.9） | 应用运行时 |
| OpenJDK / Temurin | **17**（JRE 即可） | PMML 打分（`pypmml` / JPype / py4j） |
| ca-certificates / openssl | 系统默认最新 | HTTPS 拉 PyPI、调 LLM |
| curl / wget | 任意 | 连通性检查 |
| libgomp | 系统包（Debian: `libgomp1`） | LightGBM / XGBoost OpenMP |

也可用 Miniconda / Micromamba 创建专用环境（推荐环境名 `marvis`，Python 3.12）。**不要**把 MARVIS 装进 conda `base`。

#### Debian / Ubuntu 示例

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl ca-certificates \
  python3.12 python3.12-venv python3.12-dev \
  openjdk-17-jre-headless \
  libgomp1 \
  build-essential
```

`python3.12-dev` 与 `build-essential`：多数包有 wheel，但个别环境/架构若需源码编译会用到。

#### RHEL / Rocky 示例

```bash
sudo dnf install -y git curl ca-certificates \
  python3.12 python3.12-devel \
  java-17-openjdk-headless \
  libgomp gcc gcc-c++ make
```

### 2.2 Java 必须可被服务进程找到

以部署用户执行：

```bash
java -version    # 应显示 17.x
which java       # 必须在 PATH 中
```

建议同时配置：

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64   # 路径按实际发行版调整
export PATH="$JAVA_HOME/bin:$PATH"
```

写入该用户的 `~/.bashrc`，或 systemd unit 的 `Environment=`。

说明：应用子进程会透传 `PATH`。请确保 `java` 在 PATH 上，不要只设 `JAVA_HOME` 却不进 PATH。

### 2.3 建议安装（降低边缘失败）

| 包 | 用途 |
|----|------|
| `graphviz`（系统包） | CatBoost 相关 Python `graphviz` 画图时可能需要 `dot`；核心训练通常不依赖 |
| `fonts-dejavu-core` 或中文字体 | matplotlib 出图缺字警告（非必须） |

### 2.4 不需要运维准备

- Node.js / npm
- Docker / WSL（除非 IT 自行规定用容器）
- CUDA / nvidia-driver
- 前端构建工具

---

## 3. Python 包依赖

运维需保证：**部署用户能访问 PyPI（或内网镜像）**，并允许安装下列包。

业务侧部署时执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
# 不要装 .[dev]，那是开发测试用
```

### 3.1 直接依赖（`pyproject.toml`，生产必装）

| 包 | 版本约束 |
|----|----------|
| fastapi | `>=0.115,<1` |
| uvicorn | `>=0.30,<1` |
| python-multipart | `>=0.0.9,<1` |
| pydantic | `>=2.7,<3` |
| filelock | `>=3.13,<4` |
| packaging | `>=16.8,<27` |
| jsonschema | `>=4.0,<5` |
| psutil | `>=5.9,<8` |
| nbformat | `>=5.9,<6` |
| nbclient | `>=0.8,<0.12` |
| ipykernel | `>=6.23.2,<8` |
| ipython | `>=8.20,<10` |
| jupyter-client | `>=8.6,<9` |
| numpy | `>=1.26,<3` |
| pandas | `>=2.2,<4` |
| scipy | `>=1.12,<2` |
| scikit-learn | `>=1.4,<2` |
| statsmodels | `>=0.14,<1` |
| joblib | `>=1.3,<2` |
| pyarrow | `>=15,<25` |
| duckdb | `>=0.9,<2` |
| openpyxl | `>=3.1,<4` |
| xlrd | `>=2.0,<3` |
| python-docx | `>=1.1,<2` |
| rapidfuzz | `>=3,<4` |
| matplotlib | `>=3.8,<4` |
| seaborn | `>=0.13,<1` |
| xgboost | `>=2.0,<4` |
| lightgbm | `>=4.0,<5` |
| catboost | `>=1.2,<2` |
| pypmml | `>=1.5,<2` |
| sklearn2pmml | `>=0.131,<1` |

### 3.2 关键传递依赖（常被防火墙 / 审计卡住）

| 包 | 用途 |
|----|------|
| JPype1 | `pypmml` → JVM |
| py4j | `pypmml` |
| dill | `sklearn2pmml` |
| pillow / fonttools / kiwisolver / contourpy | matplotlib |
| lxml / et-xmlfile | docx / Excel |
| pyzmq / tornado / traitlets / jupyter-core | Notebook kernel |
| starlette / anyio / h11 / click | FastAPI / uvicorn |
| nvidia-nccl-cu12 | Linux 上 xgboost 的声明依赖（**不需要 GPU/CUDA 驱动**，但 PyPI 必须能下这个包） |
| plotly / graphviz（Python） | catboost 相关 |

### 3.3 参考锁定版本（仓库 `uv.lock`，Python ≥3.12）

便于对照镜像白名单或离线缓存。**以实际 `pip install` 解析为准**；完整解析树约 90+ 个运行时包。

```text
fastapi==0.138.0
uvicorn==0.49.0
pydantic==2.13.4
numpy==2.5.0
pandas==3.0.3
scipy==1.18.0
scikit-learn==1.9.0
pyarrow==24.0.0
duckdb==1.5.4
xgboost==3.3.0
lightgbm==4.6.0
catboost==1.2.10
pypmml==1.5.8
sklearn2pmml==0.131.0
jpype1==1.7.1
py4j==0.10.9.9
matplotlib==3.11.0
ipykernel==7.3.0
seaborn==0.13.2
statsmodels==0.14.6
openpyxl==3.1.5
python-docx==1.2.0
filelock==3.29.4
psutil==7.2.2
rapidfuzz==3.14.5
jsonschema==4.26.0
nbformat==5.10.4
nbclient==0.11.0
jupyter-client==8.9.1
ipython==9.14.1
joblib==1.5.3
python-multipart==0.0.32
xlrd==2.0.2
packaging==26.2
nvidia-nccl-cu12==2.30.7
```

若内网只能白名单包名：按 §3.1 + §3.2 放行，或提供 PyPI 镜像 / 离线 wheelhouse。

---

## 4. 网络与防火墙

### 4.1 安装阶段（部署用户）

| 方向 | 目标 | 说明 |
|------|------|------|
| 出站 HTTPS | `pypi.org` + `files.pythonhosted.org`（或公司 PyPI 镜像） | `pip install` |
| 出站 HTTPS | Git 源（GitHub 或内网 Git） | `git clone` / `git pull` |
| 出站 | LLM API 地址 | Agent 对话；安装阶段可不配，上线要用 |

### 4.2 运行阶段

| 项 | 要求 |
|----|------|
| 监听 | 默认 `127.0.0.1:8000`（本机回环） |
| JupyterHub / 反向代理 | 放行本机到代理端口；并配置 §6 安全变量 |
| LLM | 服务主机能访问 OpenAI Compatible 端点（`api_base_url`） |
| 可选外网 | 默认探测 `example.com`、搜索 DuckDuckGo；可关闭外网，不影响核心验证/建模 |

请运维提前确认并告知部署人：

1. PyPI 是否直连？镜像地址是什么？是否要代理（`HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`）？
2. LLM 的 `api_base_url`、是否要内网 DNS、是否要公司 CA（`SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`）？
3. 服务是仅本机用，还是经 JupyterHub proxy 暴露？

---

## 5. 目录与权限

示例路径可改，权限模型保持：

```text
/opt/marvis/app          # 代码目录，部署用户可写（或先 clone 再 chown）
/data/marvis/workspace   # 运行数据目录，部署用户读写
/data/marvis/materials   # 可选：模型材料/样本目录（若不在 home 下）
/var/log/marvis          # 可选：日志目录
```

| 路径 | 权限 |
|------|------|
| 代码目录 | 部署用户 rwx |
| workspace | 部署用户 rwx（会写 SQLite、任务、datasets、`.duckdb_tmp`） |
| 材料目录 | 至少可读；若需复制入库则最好可读 |

若材料不在 `$HOME` 且不在 workspace 下，启动前需设：

```bash
export RMC_MATERIAL_ROOTS="/data/marvis/materials"
```

多个根路径用 `:` 分隔。

---

## 6. 进程与安全相关环境变量

单人本机可不设；**多用户 Linux / JupyterHub 强烈要求**至少配置 `MARVIS_LOCAL_TOKEN`。

| 变量 | 建议 | 作用 |
|------|------|------|
| `MARVIS_LOCAL_TOKEN` | 随机 32+ 字节 | 本机写操作必须带 token，防同机其他用户滥用 |
| `MARVIS_TRUSTED_PROXY_HOSTS` | 如 `127.0.0.1` | 反向代理时用 `X-Forwarded-For` 识别真实客户端 |
| `MARVIS_ALLOW_REMOTE_READ` | 按需 `1` | 允许非本机只读；写操作仍仅本机 |
| `JAVA_HOME` + PATH | 必配 | PMML |
| `MARVIS_DUCKDB_MEMORY_LIMIT` | 默认 `4GB`，内存紧可改 `2GB` | DuckDB |
| `MARVIS_DUCKDB_THREADS` | 默认约 `cpu/2` | DuckDB |
| `MARVIS_MAX_CSV_UPLOAD_BYTES` | 默认 2GB | 上传上限 |
| `MARVIS_MAX_EXCEL_UPLOAD_BYTES` | 默认 500MB | 上传上限 |
| `MARVIS_MAX_EXCEL_ROWS` | 默认 200 万行 | Excel 行数上限 |
| `MARVIS_LOG_LEVEL` | 默认 INFO | 日志 |

工具默认 RSS 护栏约 4GB/进程树；整机内存建议 ≥16–32GB。

生成 token 示例：

```bash
export MARVIS_LOCAL_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

---

## 7. LLM（Agent）配置前提

平台使用 **OpenAI Compatible** HTTP API。运维保证网络可达；模型密钥由部署人在应用设置里配置：

- `api_base_url`
- `model_name`
- `api_key`（或环境变量名 `api_key_env`）

未配 LLM 时：确定性工具与验证仍可跑；自然语言 Agent 不可用。

---

## 8. 运维验收清单（交给部署人之前）

以**部署用户**执行并保留输出：

```bash
uname -m && cat /etc/os-release | head -5
python3.12 -V
java -version
which java
git --version
locale | grep -i utf
df -h
free -h
ulimit -n

# 网络
curl -I https://pypi.org/simple/ || curl -I "<你们的PyPI镜像>"
# curl -I "<LLM的api_base_url>"   # 若已确定
```

勾选确认：

- [ ] 部署用户对代码目录、workspace 可写
- [ ] `java` 在 PATH 中且为 17
- [ ] PyPI（或镜像）可装大 wheel（catboost ~100MB 级、pyarrow 等）
- [ ] 若走代理：已配置 pip/curl 代理与公司 CA
- [ ] 若 JupyterHub：已约定端口与 `MARVIS_LOCAL_TOKEN` / `MARVIS_TRUSTED_PROXY_HOSTS`
- [ ] 防火墙不拦出站 HTTPS（安装）与 LLM 地址（运行）
- [ ] UTF-8 locale 已启用

---

## 9. 业务侧部署步骤（环境备好后）

```bash
# 1. 拿代码
cd /opt/marvis
git clone <仓库URL> app
cd app

# 2. 环境
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .

# 3. 冒烟
java -version
python -c "import fastapi, pandas, lightgbm, xgboost, catboost, duckdb, pypmml; print('ok')"
marvis version

# 4. 启动（共享机务必带 token）
export MARVIS_LOCAL_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# export MARVIS_TRUSTED_PROXY_HOSTS=127.0.0.1   # 若经代理
# export RMC_MATERIAL_ROOTS=/data/marvis/materials

marvis serve --host 127.0.0.1 --port 8000 --workspace /data/marvis/workspace
```

浏览器或代理访问：`http://127.0.0.1:8000/`（或 JupyterHub proxy 路径）。

备份请使用内置命令，勿在服务运行时直接 `cp` SQLite：

```bash
marvis backup --workspace /data/marvis/workspace --out marvis-backup.tar.gz
```

---

## 10. 常见失败 → 对应运维缺口

| 现象 | 多半缺什么 |
|------|------------|
| `pip` 超时 / SSL 错 | PyPI 网络、代理、公司 CA |
| `No matching distribution` / 编译失败 | Python 版本不对，或缺 `python3.12-dev` / `gcc` |
| `pypmml` / JVM 相关报错 | 未装 JDK 17，或 `java` 不在 PATH |
| `libgomp.so.1: cannot open` | 未装 `libgomp1` |
| 导入 xgboost 拉 `nvidia-nccl-cu12` 失败 | PyPI 白名单未放行该包（不需要装 CUDA） |
| 启动后同机别人能乱写任务 | 未设 `MARVIS_LOCAL_TOKEN` |
| 经 proxy 后权限错乱 | 未设 `MARVIS_TRUSTED_PROXY_HOSTS` |
| 材料路径被拒 | 未设 `RMC_MATERIAL_ROOTS` |
| OOM / 进程被杀 | 内存不足；或需下调 DuckDB / 上传上限 |

---

## 11. 工单填空（发给运维时一并填写）

| 项 | 填写 |
|----|------|
| Linux 发行版与版本 | |
| CPU 架构（x86_64 / aarch64） | |
| 是否 JupyterHub / 反向代理 | |
| 监听端口约定 | 默认 8000 |
| 代码目录 | 例：`/opt/marvis/app` |
| workspace 目录 | 例：`/data/marvis/workspace` |
| 材料目录（如有） | |
| PyPI 镜像地址 | |
| HTTP(S) 代理 | |
| 公司 CA 证书路径 | |
| LLM `api_base_url` | |
| 部署用户名 | |
