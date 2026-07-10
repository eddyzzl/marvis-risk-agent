from pathlib import Path
import re


WINDOWS_PACKAGING = Path("packaging/windows")


def test_windows_runtime_environment_bundles_python_pip_and_java():
    text = (WINDOWS_PACKAGING / "environment.yml").read_text(encoding="utf-8")
    constraints = (WINDOWS_PACKAGING / "constraints-runtime-win.txt").read_text(
        encoding="utf-8"
    )

    assert "python=3.12" in text
    assert "- pip" in text
    assert "openjdk=17" in text
    assert "numpy==1.26.4" in constraints
    assert "pandas==2.2.3" in constraints
    assert "scipy==1.13.1" in constraints
    assert "scikit-learn==1.5.2" in constraints
    assert "pyarrow==16.1.0" in constraints
    assert "X86_V2" in constraints


def test_windows_launcher_uses_private_runtime_and_user_workspace():
    text = (WINDOWS_PACKAGING / "launcher" / "Start-MARVIS.ps1").read_text(
        encoding="utf-8"
    )

    assert 'Join-Path $RuntimeRoot "python.exe"' in text
    assert 'Join-Path $RuntimeRoot "Library"' in text
    assert 'Join-Path $env:LOCALAPPDATA "MARVIS-Agent\\workspace"' in text
    assert '"marvis"' in text
    assert '"serve"' in text
    assert "/api/health" in text
    assert "$response.StatusCode -eq 200" in text


def test_windows_launcher_does_not_reuse_stale_or_foreign_port_8000_service():
    text = (WINDOWS_PACKAGING / "launcher" / "Start-MARVIS.ps1").read_text(
        encoding="utf-8"
    )

    assert "Get-MarvisListenerProcessInfo" in text
    assert "Get-NetTCPConnection -LocalPort $Port -State Listen" in text
    assert "Get-CimInstance Win32_Process" in text
    assert "Test-CurrentInstallProcess" in text
    assert "Test-MarvisServeProcess" in text
    assert "Stopping stale MARVIS process on port $Port" in text
    assert "not a MARVIS process from this installation" in text


def test_windows_launcher_registers_optional_validation_kernel():
    text = (WINDOWS_PACKAGING / "launcher" / "Start-MARVIS.ps1").read_text(
        encoding="utf-8"
    )

    assert 'Join-Path $InstallRoot "validation-runtime\\python.exe"' in text
    assert "marvis-validation-pkg" in text
    assert "MARVIS Validation (pkg.txt)" in text
    assert "System.Collections.Generic.List[string]" in text
    assert 'Join-Path $env:APPDATA "jupyter\\kernels\\marvis-validation-pkg"' in text
    assert "JUPYTER_PATH" in text
    assert "CONDA_PREFIX" in text
    assert "$ValidationRoot\\Library\\bin" in text
    assert "PYTHONNOUSERSITE" in text
    assert 'PYTHONPATH = ""' in text


def test_windows_cmd_launcher_uses_powershell_without_profile():
    text = (WINDOWS_PACKAGING / "launcher" / "MARVIS-Agent.cmd").read_text(
        encoding="utf-8"
    )

    assert "powershell.exe -NoProfile -ExecutionPolicy Bypass" in text
    assert "Start-MARVIS.ps1" in text
    assert "MARVIS-Agent failed to start" in text
    assert "%LOCALAPPDATA%\\MARVIS-Agent\\logs" in text
    assert 'if /I not "%CI%"=="true" pause' in text


def test_windows_inno_installer_is_per_user_and_creates_shortcuts():
    text = (WINDOWS_PACKAGING / "installer.iss").read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={localappdata}\Programs\MARVIS-Agent" in text
    assert 'Source: "{#PayloadDir}\\*"' in text
    assert "Compression=zip" in text
    assert "SolidCompression=no" in text
    assert "SetupIconFile={#IconFile}" in text
    assert "MARVIS-Agent.cmd" in text
    assert "{autodesktop}\\MARVIS-Agent" in text
    assert 'IconFilename: "{app}\\assets\\MARVIS-Agent.ico"' in text


def test_windows_build_script_produces_payload_before_compiling_installer():
    text = (WINDOWS_PACKAGING / "build-installer.ps1").read_text(encoding="utf-8")

    assert "python.exe" in text
    assert "$MicromambaExe create -y -p $RuntimeRoot" in text
    assert "$Python -m build" in text
    assert "pypmml" in text
    assert "constraints-runtime-win.txt" in text
    assert "-c $RuntimeConstraints" in text
    assert "assets\\MARVIS-Agent.ico" in text
    assert "/DIconFile=$IconFile" in text
    assert "Get-FileHash -Algorithm SHA256" in text
    assert "SkipInstaller" in text
    assert "IncludeValidationEnvironment" in text
    assert '$ValidationRuntimeRoot = Join-Path $PayloadRoot "validation-runtime"' in text
    assert "requirements-conda-core-win-py37.txt" in text
    assert "requirements-core-win-py37.txt" in text
    assert "requirements-optional-win-py37.txt" in text
    assert "MARVIS_VALIDATION_ENV_REPORT.txt" in text
    assert "validation core imports ok" in text
    assert "$RuntimePython -m marvis.kernel_probe" in text
    assert "Validation runtime Jupyter kernel handshake failed" in text
    assert "$ValidationRuntimeRoot\\Library\\bin" in text
    assert "-ConstraintPath $ValidationCoreRequirements" in text


def test_windows_workflow_smoke_launches_bundled_payload():
    text = Path(".github/workflows/windows-installer.yml").read_text(encoding="utf-8")

    assert "Smoke launch bundled payload" in text
    assert r".\dist\windows\build\payload\MARVIS-Agent.cmd" in text
    assert "/api/health" in text
    assert "Get-NetTCPConnection -LocalPort $port" in text


def test_validation_pkg_source_and_conflict_notes_are_kept_with_packaging():
    pkg_text = (WINDOWS_PACKAGING / "validation" / "pkg.txt").read_text(
        encoding="utf-8"
    )
    readme = (WINDOWS_PACKAGING / "validation" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "python                    3.7.6" in pkg_text
    assert "jpype1                    1.5.0" in pkg_text
    assert "Windows Packaging Bridge" in readme
    assert "requirements-conda-core-win-py37.txt" in readme
    assert "requirements-core-win-py37.txt" in readme
    assert "registered as a Jupyter" in readme


def test_validation_windows_requirements_are_pinned_from_pkg_source():
    pkg_text = (WINDOWS_PACKAGING / "validation" / "pkg.txt").read_text(
        encoding="utf-8"
    )
    core = (
        WINDOWS_PACKAGING / "validation" / "requirements-core-win-py37.txt"
    ).read_text(encoding="utf-8")
    conda_core = (
        WINDOWS_PACKAGING / "validation" / "requirements-conda-core-win-py37.txt"
    ).read_text(encoding="utf-8")
    optional = (
        WINDOWS_PACKAGING / "validation" / "requirements-optional-win-py37.txt"
    ).read_text(encoding="utf-8")

    for requirement in [
        "numpy==1.21.6",
        "pandas==1.3.3",
        "scikit-learn==1.0.2",
        "xgboost==1.2.0",
        "lightgbm==2.3.1",
    ]:
        name, version = requirement.split("==", 1)
        assert requirement in core + optional
        assert re.search(rf"^{re.escape(name)}\s+{re.escape(version)}\s", pkg_text, re.M)
    for requirement in [
        "cloudpickle=1.3.0",
        "ipython_genutils=0.2.0",
        "openpyxl=3.0.3",
        "xlrd=1.2.0",
    ]:
        name, version = requirement.split("=", 1)
        assert requirement in conda_core
        assert re.search(rf"^{re.escape(name)}\s+{re.escape(version)}\s", pkg_text, re.M)
