<p align="center">
  <img src="marvis/static/brand/marvis-workspace-logo.png" alt="MARVIS-Agent V2 logo" width="156" />
</p>

<h1 align="center">MARVIS-Agent V2</h1>

<p align="center">
  A local-first credit-risk Agent workbench for validation, data processing, feature analysis, modeling, strategy, and vintage workflows.
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a>
  ·
  <a href="README.zh-CN.md">中文</a>
</p>

---

MARVIS-Agent V2 is the current mainline for a usable credit-risk Agent workbench. It keeps governed work close to local files, local runtimes, and auditable evidence while expanding beyond the stable V1.1 model-validation workflow.

V2 is not just a runtime shell: every task entry shown on the welcome screen is a real end-to-end workflow with human-in-the-loop confirmation, tool execution, structured results, downloads or reports, and audit history. As of V2.0 this covers data join, feature analysis, model development and delivery, scoring and monitoring, strategy development (cutoff bands, rule mining, adoption with versioning), portfolio analysis, limit/pricing, and ad-hoc slice analytics — see `docs/plans/v2-master-backlog.md` and `docs/reviews/` for the full evidence trail.

Current status in this checkout:

- **Model validation** keeps the stable V1.1 manual and Agent-assisted validation path.
- **Data processing, feature analysis, and model development** are the active V2 build path, using the Plugin/Tool/Workflow runtime and task-level Agent flow.
- **Strategy, monitoring, portfolio analysis, and vintage workflows** are wired end to end (S1-S6 batches), each behind confirmation gates with red-flag checklists.

## What You Get

- **Local-first execution**: serve the platform from your own machine or server workspace.
- **Task-level Agent workbench**: drive credit-risk tasks through conversation, confirmation gates, and a persistent right-rail execution context.
- **Plugin/Tool/Workflow runtime**: install or ship governed capability packs with schemas, permissions, execution logs, and auditable outputs.
- **Notebook validation runtime**: keep V1.1 validation notebooks and downstream metrics reproducible while V2 workflows grow around them.
- **Configurable branding**: keep private customer or institution branding outside source code.
- **OSS-friendly defaults**: remove local branding config and the app falls back to the public MARVIS brand.

## Core Docs

- [Roadmap](docs/roadmap.md): current V2 platform map, V1 compatibility boundary, future V3/V4 directions, and Plugin/Tool/Hook/Workflow terminology.
- [Versioning](docs/versioning.md): release helper, tags, version bumps, and forward-port rules.
- [Notebook contract](docs/notebook_contract.md): the current model-validation notebook runtime contract.
- [Design](DESIGN.md): product experience and UI/UX decision source of truth.

## Public Default Brand

- Platform name: `MARVIS-全能风控智能体`
- Primary color: neutral charcoal (`#303034`)
- Default main logo: `marvis/static/brand/marvis-workspace-logo.png`
- Default favicon: `marvis/static/brand/marvis-favicon.png`

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

# V2 development worktree
marvis serve --profile v2
# http://127.0.0.1:8200, workspace ./workspace-v2
```

Explicit options override profile defaults:

```bash
marvis serve --profile v2 --port 8217 --workspace ./custom-workspace
```

## Update

If MARVIS was installed from a GitHub clone and the checkout is on a clean `main` branch, run:

```bash
marvis update
```

The command runs `git fetch origin`, `git pull --ff-only origin main`, then refreshes the editable MARVIS install without re-resolving the whole Python environment:

```bash
python -m pip install -e . --no-deps
```

If `marvis update` is run from Anaconda/conda `base`, MARVIS creates or reuses a dedicated `marvis` environment and installs there instead of modifying `base`. After the update, start the app with the same single command:

```bash
marvis
```

The `base` launcher automatically delegates runtime commands into the dedicated environment. This default is intentional for Anaconda and Windows machines where unrelated packages in the same environment may have strict pins. Use `--env-name <name>` to choose another dedicated conda environment. If a future release adds new runtime dependencies, run `marvis update --with-deps` from a dedicated MARVIS environment, not from Anaconda `base`.

If tracked local files have uncommitted changes, `marvis update` refuses to continue. Commit, stash, or back up those tracked changes before updating. Untracked local files are allowed unless Git itself detects that a pull would overwrite them.

If your current older install does not have `marvis update` yet, run one manual upgrade from the repository directory:

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps
```

From Anaconda `base`, install only the lightweight MARVIS launcher first, then let `marvis update` prepare the dedicated environment:

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps
marvis update
marvis
```

After that, future upgrades can use `marvis update`.

If you are deliberately running a V2 branch or worktree, pass the branch explicitly:

```bash
marvis update --branch <v2-branch>
```

## Tests

```bash
python -m pytest -q
ruff check marvis tests --extend-exclude '*.ipynb'
node --check marvis/static/app.js
```

Tests are tiered with pytest markers (`slow`, `e2e`, `llm`). For fast local
iteration, run only the fast tier (excludes real-training/real-subprocess
tests and browser e2e smoke tests):

```bash
python -m pytest -m "not slow and not e2e" -q
# or
scripts/check --fast
```

CI always runs the full, untiered suite; `--fast` is a local-only speedup.

## Release Push

Use the release helper instead of raw `git push` when publishing a new public version. Run it **after** the feature, fix, or documentation changes have been verified and committed. The helper requires a clean worktree and creates a separate version bump commit plus an annotated tag.

```bash
python scripts/release_push.py --bump patch
```

The helper updates release metadata, creates a release commit, creates an annotated `Vx.y.z` tag, and pushes `main` plus the tag. See `docs/versioning.md` for the full release sequence and versioning rules.

## License

This project is released under the MIT License. See `LICENSE` for details.
