# MARVIS Risk Agent

MARVIS Risk Agent is a local-first credit-risk agent platform for modeling, analysis, strategy, and validation workflows.

The current V1.0.0 release ships model validation as the first built-in workflow: it can run notebook-based validation tasks, generate structured evidence, and draft Excel/Word validation reports through Agent mode. That validation workflow is one part of the product direction, not the product boundary. Future work should keep MARVIS oriented around broader credit-risk work such as model building, portfolio analysis, strategy evaluation, monitoring, and governed task automation.

The public default brand is MARVIS:

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

Use the release helper instead of raw `git push` when publishing a new public version:

```bash
python scripts/release_push.py --bump patch
```

The helper updates release metadata, creates a release commit, creates an annotated `Vx.y.z` tag, and pushes `main` plus the tag. See `docs/versioning.md` for the versioning rules.

## License

This project is released under the MIT License. See `LICENSE` for details.
