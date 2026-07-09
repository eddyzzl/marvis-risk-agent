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
