# Validation Execution Environment

This folder tracks the separate model-validation execution environment requested
for the Windows installer. This is not the MARVIS platform runtime.

Target shape:

- Platform runtime: `packaging/windows/environment.yml`, Python 3.12, runs the
  MARVIS FastAPI app and static frontend.
- Validation runtime: a separate Python environment registered as a Jupyter
  kernel so users can choose it from the execution-environment settings panel.

The validation runtime must not replace the platform runtime. It is selected
only for task/Notebook execution through MARVIS's existing execution environment
picker.

## Source Package List

`pkg.txt` is copied from the supplied company-server environment package list.

The original supplied file SHA256 at ingestion time was:

```text
662504e2702f7a0fb1771f3bd8f2c66ead73b63c77094d6bd9edf36afef08b7b
```

The committed copy is line-ending-normalized to LF. Its SHA256 is:

```text
8e5575b7f3e9980475ad7cbeb06c8694b233da2a539ebcd21cb0ce53bae6e4a3
```

The package versions in future validation-env work should remain traceable to
this file unless a conflict is explicitly documented.

## Windows Packaging Bridge

The supplied list is not directly installable as-is on native Windows:

- `pkg.txt` pins `python 3.7.6`.
- `pkg.txt` is a Linux Anaconda list and includes Linux-only packages such as
  `ld_impl_linux-64`, `libgcc-ng`, and `libstdcxx-ng`.
- `jpype1==1.5.0` is skipped in the Python 3.7 kernel; PMML/JVM-backed
  validation runs in the platform Python 3.12 runtime.

The Windows installer therefore builds a native Python 3.7 validation runtime
from `environment.yml`, installs core packages pinned from `pkg.txt` via
`requirements-core-win-py37.txt`, and attempts optional model packages from
`requirements-optional-win-py37.txt` best-effort. Any skipped Linux-only package
or failed optional package is written into
`validation-runtime\MARVIS_VALIDATION_ENV_REPORT.txt`.

The selected validation kernel only runs the user's Notebook code and scoring
function. MARVIS deterministic validation metrics, PMML scoring, Excel output,
and report generation run in the platform runtime.
