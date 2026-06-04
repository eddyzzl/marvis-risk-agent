<p align="center">
  <img src="riskmodel_checker/static/brand/marvis-logo.png" alt="MARVIS Risk Agent logo" width="148" />
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

The current V1.0.1 release ships model validation as the first stable built-in workflow. It can run notebook-based validation tasks, generate structured evidence, and draft Excel/Word validation reports through Agent mode. Model validation is the first workflow, not the product boundary.

## What You Get

- **Local-first execution**: serve the platform from your own machine or server workspace.
- **Agent-assisted workflows**: guide credit-risk tasks with structured evidence and report drafting.
- **Notebook validation runtime**: execute validation notebooks and downstream metrics with reproducible artifacts.
- **Configurable branding**: keep private customer or institution branding outside source code.
- **OSS-friendly defaults**: remove local branding config and the app falls back to the public MARVIS brand.

## Public Default Brand

- Platform name: `MARVIS-全能风控智能体`
- Primary color: black
- Default logo and favicon: `riskmodel_checker/static/brand/`

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

Create an environment with any name you prefer. For example, with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Or with conda:

```bash
conda create -n marvis python=3.12
conda activate marvis
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Local Run

```bash
python -m riskmodel_checker serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

The Python module name `riskmodel_checker` is retained in V1 for compatibility with the current validation runtime. If installed in editable mode, the product-facing command alias is also available:

```bash
marvis-risk-agent serve --host 127.0.0.1 --port 8000 --workspace ./workspace
```

Then open `http://127.0.0.1:8000/`.

## Tests

```bash
python -m pytest -q
ruff check riskmodel_checker tests --extend-exclude '*.ipynb'
node --check riskmodel_checker/static/app.js
```

## Release Push

Use the release helper instead of raw `git push` when publishing a new public version. Run it **after** the feature, fix, or documentation changes have been verified and committed. The helper requires a clean worktree and creates a separate version bump commit plus an annotated tag.

```bash
python scripts/release_push.py --bump patch
```

The helper updates release metadata, creates a release commit, creates an annotated `Vx.y.z` tag, and pushes `main` plus the tag. See `docs/versioning.md` for the full release sequence and versioning rules.

## License

This project is released under the MIT License. See `LICENSE` for details.
