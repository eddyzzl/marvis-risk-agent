[CmdletBinding()]
param(
    [string] $Version = "",
    [string] $OutputRoot = "",
    [string] $Python = "python",
    [string] $Micromamba = "micromamba",
    [string] $InnoSetupCompiler = "iscc.exe",
    [string] $ValidationPackageList = "",
    [switch] $IncludeValidationEnvironment,
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
    if (Test-Path $CommandName) {
        return (Resolve-Path $CommandName).Path
    }
    $command = Get-Command $CommandName -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "$CommandName was not found on PATH. $InstallHint"
    }
    if ($command.Path) {
        return $command.Path
    }
    if ($command.Source -and (Test-Path $command.Source)) {
        return (Resolve-Path $command.Source).Path
    }
    throw "$CommandName resolved to '$($command.Source)', but that is not an executable path. $InstallHint"
}

function Read-CondaListFile {
    param([string] $Path)
    $rows = @{}
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed -split "\s+"
        if ($parts.Length -lt 2) {
            continue
        }
        $rows[$parts[0].ToLowerInvariant()] = @{
            Name = $parts[0]
            Version = $parts[1]
            Build = if ($parts.Length -ge 3) { $parts[2] } else { "" }
            Channel = if ($parts.Length -ge 4) { $parts[3] } else { "" }
        }
    }
    return $rows
}

function Write-ValidationCompatibilityReport {
    param(
        [string] $Path,
        [string] $PackageListPath,
        [string[]] $SkippedPackages,
        [string[]] $FailedOptionalPackages
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("MARVIS validation runtime compatibility report")
    $lines.Add("")
    $lines.Add("Source package list: $PackageListPath")
    $lines.Add("Runtime role: selectable Jupyter kernel for user model-validation notebooks only.")
    $lines.Add("Platform validation metrics run in the MARVIS platform runtime, not in this Python 3.7 kernel.")
    $lines.Add("")
    $lines.Add("Known source conflicts handled by the Windows packaging bridge:")
    $lines.Add("- Linux-only conda packages from pkg.txt are not installed into the native Windows runtime.")
    $lines.Add("- The current MARVIS package is not installed into this Python 3.7 runtime.")
    $lines.Add("- PMML/JVM-backed validation runs in the platform runtime, so jpype1 from pkg.txt is not required here.")
    $lines.Add("")
    if ($SkippedPackages.Count -gt 0) {
        $lines.Add("Skipped packages:")
        foreach ($entry in $SkippedPackages) {
            $lines.Add("- $entry")
        }
        $lines.Add("")
    }
    if ($FailedOptionalPackages.Count -gt 0) {
        $lines.Add("Optional packages that failed to install:")
        foreach ($entry in $FailedOptionalPackages) {
            $lines.Add("- $entry")
        }
        $lines.Add("")
    }
    $lines.Add("Core conda packages are installed from packaging/windows/validation/requirements-conda-core-win-py37.txt.")
    $lines.Add("Core pip packages are installed from packaging/windows/validation/requirements-core-win-py37.txt.")
    $lines.Add("Optional packages are attempted from packaging/windows/validation/requirements-optional-win-py37.txt.")
    Set-Content -Encoding utf8 -Path $Path -Value $lines.ToArray()
}

function Install-PipRequirementLines {
    param(
        [string] $PythonExe,
        [string] $RequirementsPath,
        [string] $ConstraintPath = "",
        [switch] $Required
    )
    $failed = New-Object System.Collections.Generic.List[string]
    foreach ($line in Get-Content $RequirementsPath) {
        $requirement = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($requirement) -or $requirement.StartsWith("#")) {
            continue
        }
        Write-Host "Installing validation package $requirement"
        if ([string]::IsNullOrWhiteSpace($ConstraintPath)) {
            & $PythonExe -m pip install --no-cache-dir --only-binary=:all: $requirement
        } else {
            & $PythonExe -m pip install --no-cache-dir --only-binary=:all: -c $ConstraintPath $requirement
        }
        if ($LASTEXITCODE -ne 0) {
            if ($Required) {
                throw "Installing required validation package failed: $requirement"
            }
            Write-Warning "Optional validation package failed and will be reported: $requirement"
            $failed.Add($requirement)
        }
    }
    return $failed.ToArray()
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
$ValidationRuntimeRoot = Join-Path $PayloadRoot "validation-runtime"
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

if ($IncludeValidationEnvironment) {
    if ([string]::IsNullOrWhiteSpace($ValidationPackageList)) {
        $ValidationPackageList = Join-Path $ScriptRoot "validation\pkg.txt"
    }
    $ValidationPackages = Read-CondaListFile -Path $ValidationPackageList
    $SkippedValidationPackages = New-Object System.Collections.Generic.List[string]
    foreach ($name in @(
        "ld_impl_linux-64",
        "libgcc-ng",
        "libgfortran-ng",
        "libstdcxx-ng",
        "dbus",
        "gst-plugins-base",
        "gstreamer",
        "jeepney",
        "jpype1"
    )) {
        if ($ValidationPackages.ContainsKey($name)) {
            $row = $ValidationPackages[$name]
            $SkippedValidationPackages.Add("$($row.Name)==$($row.Version)")
        }
    }
    Write-Host "Creating optional validation execution runtime at $ValidationRuntimeRoot"
    & $MicromambaExe create -y -p $ValidationRuntimeRoot -f (Join-Path $ScriptRoot "validation\environment.yml")
    if ($LASTEXITCODE -ne 0) {
        throw "Validation execution runtime creation failed"
    }
    $ValidationPython = Join-Path $ValidationRuntimeRoot "python.exe"
    $ValidationCondaRequirements = Join-Path $ScriptRoot "validation\requirements-conda-core-win-py37.txt"
    $ValidationCoreRequirements = Join-Path $ScriptRoot "validation\requirements-core-win-py37.txt"
    $ValidationOptionalRequirements = Join-Path $ScriptRoot "validation\requirements-optional-win-py37.txt"
    & $MicromambaExe install -y -p $ValidationRuntimeRoot --file $ValidationCondaRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Installing required validation conda packages failed"
    }
    $PreviousPath = $env:PATH
    try {
        $env:PATH = "$ValidationRuntimeRoot;$ValidationRuntimeRoot\Scripts;$ValidationRuntimeRoot\Library\bin;$env:PATH"
        & $ValidationPython -m pip install --disable-pip-version-check --no-cache-dir --upgrade "pip<25"
        if ($LASTEXITCODE -ne 0) {
            throw "Upgrading pip in the validation runtime failed"
        }
        [void](Install-PipRequirementLines `
            -PythonExe $ValidationPython `
            -RequirementsPath $ValidationCoreRequirements `
            -Required)
        $FailedOptionalPackages = Install-PipRequirementLines `
            -PythonExe $ValidationPython `
            -RequirementsPath $ValidationOptionalRequirements `
            -ConstraintPath $ValidationCoreRequirements
        if ($null -eq $FailedOptionalPackages) {
            $FailedOptionalPackages = @()
        }
        & $ValidationPython -m ipykernel --version
        if ($LASTEXITCODE -ne 0) {
            throw "Validation runtime ipykernel smoke check failed"
        }
        & $ValidationPython -c "import numpy, pandas, sklearn, scipy; print('validation core imports ok')"
        if ($LASTEXITCODE -ne 0) {
            throw "Validation runtime core import smoke check failed"
        }
    }
    finally {
        $env:PATH = $PreviousPath
    }
    Write-ValidationCompatibilityReport `
        -Path (Join-Path $ValidationRuntimeRoot "MARVIS_VALIDATION_ENV_REPORT.txt") `
        -PackageListPath $ValidationPackageList `
        -SkippedPackages $SkippedValidationPackages.ToArray() `
        -FailedOptionalPackages $FailedOptionalPackages
    # Start-MARVIS.ps1 registers validation-runtime\python.exe as a Jupyter
    # kernel at first launch using the final install path, so the kernel argv is
    # never baked with a build-machine absolute path.
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
