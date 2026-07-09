[CmdletBinding()]
param(
    [string] $Version = "",
    [string] $OutputRoot = "",
    [string] $Python = "python",
    [string] $Micromamba = "micromamba",
    [string] $InnoSetupCompiler = "iscc.exe",
    [switch] $SkipInstaller,
    [switch] $SkipSmoke
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "dist\windows"
}

function Get-MarvisVersion {
    $pyproject = Join-Path $RepoRoot "pyproject.toml"
    foreach ($line in Get-Content $pyproject) {
        if ($line -match '^version\s*=\s*"([^"]+)"') {
            return $Matches[1]
        }
    }
    throw "Could not find project.version in pyproject.toml"
}

function Resolve-CommandPath {
    param(
        [string] $CommandName,
        [string] $InstallHint
    )
    $command = Get-Command $CommandName -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "$CommandName was not found on PATH. $InstallHint"
    }
    return $command.Source
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = Get-MarvisVersion
}

$MicromambaExe = Resolve-CommandPath `
    -CommandName $Micromamba `
    -InstallHint "Install micromamba first, then rerun this script."

if (-not $SkipInstaller) {
    $InnoExe = Resolve-CommandPath `
        -CommandName $InnoSetupCompiler `
        -InstallHint "Install Inno Setup 6 first, then rerun this script or pass -SkipInstaller."
}

$BuildRoot = Join-Path $OutputRoot "build"
$Wheelhouse = Join-Path $BuildRoot "wheelhouse"
$PayloadRoot = Join-Path $BuildRoot "payload"
$RuntimeRoot = Join-Path $PayloadRoot "runtime"
$BinRoot = Join-Path $PayloadRoot "bin"

Remove-Item -Recurse -Force $BuildRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Wheelhouse, $PayloadRoot, $BinRoot | Out-Null

Write-Host "Building MARVIS wheel for version $Version"
& $Python -m pip install --quiet --upgrade build
if ($LASTEXITCODE -ne 0) {
    throw "Installing the build backend failed"
}
& $Python -m build --wheel --outdir $Wheelhouse $RepoRoot
if ($LASTEXITCODE -ne 0) {
    throw "Wheel build failed"
}

$wheel = Get-ChildItem $Wheelhouse -Filter "marvis-$Version-*.whl" | Select-Object -First 1
if (-not $wheel) {
    throw "Expected wheel marvis-$Version-*.whl was not produced in $Wheelhouse"
}

Write-Host "Creating private runtime at $RuntimeRoot"
$env:MAMBA_ROOT_PREFIX = Join-Path $BuildRoot "mamba-root"
& $MicromambaExe create -y -p $RuntimeRoot -f (Join-Path $ScriptRoot "environment.yml")
if ($LASTEXITCODE -ne 0) {
    throw "Runtime environment creation failed"
}

$RuntimePython = Join-Path $RuntimeRoot "python.exe"
& $RuntimePython -m pip install --no-cache-dir --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Upgrading pip in the runtime failed"
}
& $RuntimePython -m pip install --no-cache-dir $($wheel.FullName)
if ($LASTEXITCODE -ne 0) {
    throw "Installing MARVIS into the runtime failed"
}

if (-not $SkipSmoke) {
    Write-Host "Running runtime smoke checks"
    & $RuntimePython -m marvis version
    if ($LASTEXITCODE -ne 0) {
        throw "marvis version smoke check failed"
    }
    & $RuntimePython -c "import fastapi, uvicorn, pandas, pypmml; print('runtime imports ok')"
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime import smoke check failed"
    }
    & (Join-Path $RuntimeRoot "Library\bin\java.exe") -version
    if ($LASTEXITCODE -ne 0) {
        throw "Bundled Java smoke check failed"
    }
}

Copy-Item -Force (Join-Path $ScriptRoot "launcher\MARVIS-Agent.cmd") (Join-Path $PayloadRoot "MARVIS-Agent.cmd")
Copy-Item -Force (Join-Path $ScriptRoot "launcher\Start-MARVIS.ps1") (Join-Path $BinRoot "Start-MARVIS.ps1")
Copy-Item -Force (Join-Path $RepoRoot "LICENSE") (Join-Path $PayloadRoot "LICENSE.txt")
Copy-Item -Force (Join-Path $RepoRoot "README.md") (Join-Path $PayloadRoot "README.md")
Set-Content -Encoding utf8 -Path (Join-Path $PayloadRoot "VERSION.txt") -Value $Version

if ($SkipInstaller) {
    Write-Host "Prepared installer payload at $PayloadRoot"
    exit 0
}

Write-Host "Compiling Inno Setup installer"
& $InnoExe `
    "/DAppVersion=$Version" `
    "/DPayloadDir=$PayloadRoot" `
    "/DOutputDir=$OutputRoot" `
    (Join-Path $ScriptRoot "installer.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed"
}

$Installer = Join-Path $OutputRoot "MARVIS-Setup-$Version-win-x64.exe"
if (-not (Test-Path $Installer)) {
    throw "Installer was not produced at $Installer"
}

$Hash = Get-FileHash -Algorithm SHA256 $Installer
$HashLine = "$($Hash.Hash)  $(Split-Path -Leaf $Installer)"
Set-Content -Encoding ascii -Path "$Installer.sha256" -Value $HashLine

Write-Host "Built $Installer"
Write-Host "SHA256 $HashLine"
