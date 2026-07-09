# Windows Installer

This folder builds the personal-computer Windows installer for MARVIS-Agent.
The installer is intentionally native Windows, not Docker-first: users should
not need Python, Java, Git, conda, WSL, or Docker before installing MARVIS.

## Output

The release artifact is:

```text
MARVIS-Setup-<version>-win-x64.exe
```

It installs per-user by default into:

```text
%LOCALAPPDATA%\Programs\MARVIS-Agent
```

Runtime data stays outside the install directory:

```text
%LOCALAPPDATA%\MARVIS-Agent\workspace
%LOCALAPPDATA%\MARVIS-Agent\logs
```

Uninstalling the app removes the runtime files but intentionally leaves the
workspace and logs for the user to review or delete.

## Runtime Shape

The installer payload contains:

- `runtime\python.exe`: private Python 3.12 runtime.
- `runtime\Library\bin\java.exe`: private OpenJDK 17 runtime for PMML scoring.
- `validation-runtime\python.exe`: optional separate task/Notebook execution
  runtime when a compatible validation environment is bundled.
- `MARVIS-Agent.cmd`: shortcut target.
- `bin\Start-MARVIS.ps1`: starts `marvis serve` and opens the browser.

The launcher sets `JAVA_HOME` and `PATH` only for the MARVIS child process. It
does not register Python, does not modify system Java, and does not require
administrator privileges.

If `validation-runtime\python.exe` exists, the launcher registers it as a
Jupyter kernel named `MARVIS Validation (pkg.txt)` and prepends the install
directory to `JUPYTER_PATH`. That makes the environment visible in MARVIS's
execution-environment picker without replacing the platform runtime.

## Build Prerequisites

Run this on Windows x64:

- Python 3.12 or newer for the build host.
- micromamba on `PATH`.
- Inno Setup 6 on `PATH` as `iscc.exe`.

## Build

From the repository root:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

To prepare and smoke-test the payload without compiling the final `.exe`:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1 -SkipInstaller
```

The script:

1. Builds the MARVIS wheel from the current checkout.
2. Creates a private runtime from `environment.yml`.
3. Installs MARVIS and its Python dependencies into that runtime.
4. Runs smoke checks for `marvis version`, core imports, and bundled Java.
5. Copies launchers and metadata into `dist\windows\build\payload`.
6. Compiles `MARVIS-Setup-<version>-win-x64.exe` with Inno Setup.
7. Writes a sidecar SHA256 file next to the installer.

To attempt bundling a separate validation execution environment:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1 -IncludeValidationEnvironment
```

For the current `packaging\windows\validation\pkg.txt`, this intentionally fails
with a conflict report: the supplied environment is a Linux Python 3.7.6
Anaconda environment, while current MARVIS injected validation cells require
Python 3.11+ in the selected kernel. The conflict is documented in
`packaging/windows/validation/README.md`.

## Manual Launcher Test

After `-SkipInstaller`, run:

```powershell
.\dist\windows\build\payload\MARVIS-Agent.cmd -NoBrowser
```

Then open:

```text
http://127.0.0.1:8000/
```

## Docker

Docker can still be offered later for server or IT-managed deployments. It is
not the default personal Windows path because Docker Desktop adds WSL2,
virtualization, volume-mapping, and licensing concerns that a native installer
does not impose on ordinary users.
