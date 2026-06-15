<p align="center">
  <img src="marvis/static/brand/marvis-logo.png" alt="MARVIS Risk Agent logo" width="148" />
</p>

<h1 align="center">MARVIS Risk Agent</h1>

<p align="center">
  A local-first credit-risk agent platform for modeling, analysis, strategy, and validation workflows.
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a>
  ·
  <a href="README.zh-CN.md">中文</a>
</p>

---

MARVIS Risk Agent is built for governed credit-risk work that should stay close to local files, local runtimes, and auditable evidence. The long-term product direction is an all-purpose credit-risk agent for model building, portfolio analysis, strategy evaluation, monitoring, validation, and governed task automation.

The current V1.1.2 release ships model validation as the first stable built-in workflow. It can run notebook-based validation tasks, generate structured evidence, and draft Excel/Word validation reports through Agent mode. Model validation is the first workflow, not the product boundary.

The current V1.1.x line already ships the Agent Memory Foundation for historical validation metric comparison. Planned next steps are documented in [docs/roadmap.md](docs/roadmap.md): V2 adds the Agent Plugin/Tool Runtime, and later releases build modeling and strategy capability packs on that runtime.

## What You Get

- **Local-first execution**: serve the platform from your own machine or server workspace.
- **Agent-assisted workflows**: guide credit-risk tasks with structured evidence and report drafting.
- **Notebook validation runtime**: execute validation notebooks and downstream metrics with reproducible artifacts.
- **Configurable branding**: keep private customer or institution branding outside source code.
- **OSS-friendly defaults**: remove local branding config and the app falls back to the public MARVIS brand.

## Core Docs

- [Roadmap](docs/roadmap.md): V1/V1.1/V2/V3/V4 phases and Plugin/Tool/Hook/Workflow terminology.
- [Versioning](docs/versioning.md): release helper, tags, version bumps, and forward-port rules.
- [Notebook contract](docs/notebook_contract.md): the current model-validation notebook runtime contract.
- [Design](DESIGN.md): product experience and UI/UX decision source of truth.

## Public Default Brand

- Platform name: `MARVIS-全能风控智能体`
- Primary color: black
- Default logo and favicon: `marvis/static/brand/`

## Branding

Private or customer-specific branding is intentionally not committed. To apply a local brand, create an ignored workspace config:

```text
workspace/branding/brand.json
```

Example:

```json
{
  "platform_name": "本地信贷风控智能体",
  "browser_title": "本地信贷风控工作台",
  "primary_color": "#1f6feb",
  "logo": "private-logo.svg",
  "favicon": "private-logo.svg"
}
```

Put referenced logo files next to `brand.json`. When `workspace/branding/` is absent, the app falls back to the public MARVIS brand.

See `docs/branding.md` for details.

## Local Deployment Requirements

- Python 3.11 or newer. Python 3.12 is recommended for a new local install.
- macOS or Linux for the currently verified local workflow.
- A Java runtime compatible with `pypmml` if you need PMML scoring.
- Node.js is only needed for frontend syntax checks; the app itself serves static HTML/CSS/JS through FastAPI.

## Install From GitHub

Clone the repository, then install from the checkout. Create an environment with any name you prefer. For example, with `venv`:

```bash
git clone https://github.com/eddyzzl/marvis-risk-agent.git
cd marvis-risk-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Or with conda:

```bash
git clone https://github.com/eddyzzl/marvis-risk-agent.git
cd marvis-risk-agent
conda create -n marvis python=3.12
conda activate marvis
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Local Run

After installation, start MARVIS with:

```bash
marvis
```

By default, this is equivalent to:

```bash
marvis serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

Then open `http://127.0.0.1:8000/`.

The Python module name `marvis` is retained in V1 for compatibility with the current validation runtime. The older entrypoints still work:

```bash
python -m marvis serve --host 127.0.0.1 --port 8000 --workspace ./workspace
marvis-risk-agent serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

## Material Directories

When creating a task, the material directory must be under the current `workspace` or the current user's home directory by default. On Windows, allow another drive or local folder before startup:

```powershell
$env:RMC_MATERIAL_ROOTS="D:\model_materials"
marvis serve --host 127.0.0.1 --port 8000 --workspace .\workspace
```

When running under WSL2, enter the WSL path such as `/mnt/c/Users/<you>/Downloads/project`, not a `C:\...` Windows path.

## Multiple Worktrees / Versions

When running multiple worktrees at the same time, use different ports and different workspaces. Profiles choose safe defaults:

```bash
# Stable main demo
marvis serve --profile main
# http://127.0.0.1:8000, workspace ./workspace-main

# V1.1 development or comparison
marvis serve --profile v1-1
# http://127.0.0.1:8001, workspace ./workspace-v1-1
```

Explicit options override profile defaults:

```bash
marvis serve --profile v1-1 --port 8017 --workspace ./custom-workspace
```

## Update

If MARVIS was installed from a GitHub clone and the checkout is on a clean `main` branch, run:

```bash
marvis update
```

The command runs `git fetch origin`, `git pull --ff-only origin main`, then refreshes the editable install:

```bash
python -m pip install -e .
```

If local changes exist, `marvis update` refuses to continue. Commit, stash, or back up local changes before updating.

If your current older install does not have `marvis update` yet, run one manual upgrade from the repository directory:

```bash
git pull --ff-only origin main
python -m pip install -e .
```

After that, future upgrades can use `marvis update`.

## Tests

```bash
python -m pytest -q
ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
```

## Release Push

Use the release helper instead of raw `git push` when publishing a new public version. Run it **after** the feature, fix, or documentation changes have been verified and committed. The helper requires a clean worktree and creates a separate version bump commit plus an annotated tag.

```bash
python scripts/release_push.py --bump patch
```

The helper updates release metadata, creates a release commit, creates an annotated `Vx.y.z` tag, and pushes `main` plus the tag. See `docs/versioning.md` for the full release sequence and versioning rules.

## License

This project is released under the MIT License. See `LICENSE` for details.
