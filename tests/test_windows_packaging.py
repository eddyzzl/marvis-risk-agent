from pathlib import Path


WINDOWS_PACKAGING = Path("packaging/windows")


def test_windows_runtime_environment_bundles_python_pip_and_java():
    text = (WINDOWS_PACKAGING / "environment.yml").read_text(encoding="utf-8")

    assert "python=3.12" in text
    assert "- pip" in text
    assert "openjdk=17" in text


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


def test_windows_launcher_registers_optional_validation_kernel():
    text = (WINDOWS_PACKAGING / "launcher" / "Start-MARVIS.ps1").read_text(
        encoding="utf-8"
    )

    assert 'Join-Path $InstallRoot "validation-runtime\\python.exe"' in text
    assert "marvis-validation-pkg" in text
    assert "MARVIS Validation (pkg.txt)" in text
    assert "JUPYTER_PATH" in text


def test_windows_cmd_launcher_uses_powershell_without_profile():
    text = (WINDOWS_PACKAGING / "launcher" / "MARVIS-Agent.cmd").read_text(
        encoding="utf-8"
    )

    assert "powershell.exe -NoProfile -ExecutionPolicy Bypass" in text
    assert "Start-MARVIS.ps1" in text


def test_windows_inno_installer_is_per_user_and_creates_shortcuts():
    text = (WINDOWS_PACKAGING / "installer.iss").read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={localappdata}\Programs\MARVIS-Agent" in text
    assert 'Source: "{#PayloadDir}\\*"' in text
    assert "MARVIS-Agent.cmd" in text
    assert "{autodesktop}\\MARVIS-Agent" in text


def test_windows_build_script_produces_payload_before_compiling_installer():
    text = (WINDOWS_PACKAGING / "build-installer.ps1").read_text(encoding="utf-8")

    assert "python.exe" in text
    assert "$MicromambaExe create -y -p $RuntimeRoot" in text
    assert "$Python -m build" in text
    assert "pypmml" in text
    assert "Get-FileHash -Algorithm SHA256" in text
    assert "SkipInstaller" in text
    assert "IncludeValidationEnvironment" in text
    assert '$ValidationRuntimeRoot = Join-Path $PayloadRoot "validation-runtime"' in text
    assert "Assert-ValidationPackageListIsSupported" in text
    assert "Write-ValidationPackageInstallSpecs" in text
    assert "validation-requirements.txt" in text
    assert "validation-conda-specs.txt" in text


def test_validation_pkg_source_and_conflict_notes_are_kept_with_packaging():
    pkg_text = (WINDOWS_PACKAGING / "validation" / "pkg.txt").read_text(
        encoding="utf-8"
    )
    readme = (WINDOWS_PACKAGING / "validation" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "python                    3.7.6" in pkg_text
    assert "jpype1                    1.5.0" in pkg_text
    assert "Current MARVIS requires Python `>=3.11`" in readme
    assert "registered as a Jupyter" in readme
