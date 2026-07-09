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

## Current Blocker

This package list cannot be bundled as a working native Windows validation
kernel yet.

Hard conflicts:

- `pkg.txt` pins `python 3.7.6`.
- Current MARVIS requires Python `>=3.11`.
- The validation Notebook pipeline injects cells that import current
  `marvis.*` modules inside the selected Jupyter kernel.
- Current MARVIS source uses syntax that Python 3.7 cannot parse.
- `pkg.txt` is a Linux Anaconda list and includes Linux-only packages such as
  `ld_impl_linux-64`, `libgcc-ng`, and `libstdcxx-ng`.
- A Windows cp37 wheel preflight also fails for `jpype1==1.5.0`; that matters
  for PMML/JVM-backed scoring paths.

Therefore the Windows build script refuses `-IncludeValidationEnvironment` for
this package list instead of shipping a selectable environment that fails at
runtime.

## Viable Paths

1. Build a MARVIS compatibility bridge: run the user's notebook code in the
   legacy Python 3.7 kernel, serialize `RMC_SAMPLE_DF`, code-model scores, and
   model metadata, then run MARVIS deterministic validation cells in the
   platform Python 3.12 runtime.
2. Rebuild the validation package list on Python 3.11 or 3.12 while preserving
   the business-critical package versions where compatible.
3. Use the company server as a managed validation execution target instead of
   packaging this exact Linux Python 3.7 environment into a native Windows
   installer.

Until one of those paths is implemented, the Windows installer should keep the
platform runtime and task execution environment separated but should not default
to this legacy validation package list.
