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

function Assert-ValidationPackageListIsSupported {
    param([string] $Path)
    if (-not (Test-Path $Path)) {
        throw "Validation package list was not found at $Path"
    }
    $packages = Read-CondaListFile -Path $Path
    $python = $packages["python"]
    $jpype = $packages["jpype1"]
    $platformOnly = @("ld_impl_linux-64", "libgcc-ng", "libstdcxx-ng", "dbus", "gst-plugins-base", "gstreamer")
    $presentPlatformOnly = @()
    foreach ($name in $platformOnly) {
        if ($packages.ContainsKey($name)) {
            $presentPlatformOnly += $name
        }
    }
    $pythonVersion = if ($python) { $python.Version } else { "unknown" }
    $jpypeVersion = if ($jpype) { $jpype.Version } else { "not listed" }
    $reasons = @()
    if ($pythonVersion.StartsWith("3.7.")) {
        $reasons += "pkg.txt pins Python $pythonVersion, but the current MARVIS package and injected validation cells require Python >=3.11."
    }
    if ($presentPlatformOnly.Count -gt 0) {
        $reasons += "pkg.txt contains Linux-only conda packages that cannot be installed into a native Windows runtime: $($presentPlatformOnly -join ', ')."
    }
    if ($jpypeVersion -eq "1.5.0") {
        $reasons += "pkg.txt pins jpype1==1.5.0; Windows cp37 wheel resolution fails because that version requires a newer Python than 3.7."
    }
    if ($reasons.Count -gt 0) {
        throw @"
The requested validation execution environment cannot be bundled as a working Windows kernel yet.

$($reasons -join "`n")

Keep the MARVIS platform runtime separate. To support this legacy validation package list, the validation pipeline needs a compatibility bridge that runs the user notebook in the legacy kernel but runs MARVIS injected deterministic validation cells in the platform runtime, or the package list needs to be rebuilt on Python >=3.11 with compatible package versions.
"@
    }
}

function Write-ValidationPackageInstallSpecs {
    param(
        [string] $PackageListPath,
        [string] $CondaSpecPath,
        [string] $PipRequirementsPath
    )

    $packages = Read-CondaListFile -Path $PackageListPath
    $skipConda = @(
        "_libgcc_mutex",
        "anaconda",
        "anaconda-client",
        "anaconda-navigator",
        "anaconda-project",
        "conda",
        "conda-build",
        "conda-env",
        "conda-package-handling",
        "conda-verify",
        "ipykernel",
        "ld_impl_linux-64",
        "navigator-updater",
        "pip",
        "python",
        "setuptools",
        "wheel"
    )
    $skip = @{}
    foreach ($name in $skipConda) {
        $skip[$name] = $true
    }

    $condaSpecs = New-Object System.Collections.Generic.List[string]
    $pipRequirements = New-Object System.Collections.Generic.List[string]
    foreach ($row in ($packages.Values | Sort-Object { $_.Name.ToLowerInvariant() })) {
        $name = [string] $row.Name
        $version = [string] $row.Version
        $build = [string] $row.Build
        $channel = [string] $row.Channel
        if ([string]::IsNullOrWhiteSpace($name) -or [string]::IsNullOrWhiteSpace($version)) {
            continue
        }
        if ($channel -eq "pypi" -or $build -eq "pypi_0") {
            $pipRequirements.Add("$name==$version")
            continue
        }
        if ($skip.ContainsKey($name.ToLowerInvariant())) {
            continue
        }
        $condaSpecs.Add("$name=$version")
    }

    Set-Content -Encoding ascii -Path $CondaSpecPath -Value $condaSpecs.ToArray()
    Set-Content -Encoding ascii -Path $PipRequirementsPath -Value $pipRequirements.ToArray()
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
    Assert-ValidationPackageListIsSupported -Path $ValidationPackageList
    Write-Host "Creating optional validation execution runtime at $ValidationRuntimeRoot"
    & $MicromambaExe create -y -p $ValidationRuntimeRoot -f (Join-Path $ScriptRoot "validation\environment.yml")
    if ($LASTEXITCODE -ne 0) {
        throw "Validation execution runtime creation failed"
    }
    $ValidationCondaSpecs = Join-Path $BuildRoot "validation-conda-specs.txt"
    $ValidationPipRequirements = Join-Path $BuildRoot "validation-requirements.txt"
    Write-ValidationPackageInstallSpecs `
        -PackageListPath $ValidationPackageList `
        -CondaSpecPath $ValidationCondaSpecs `
        -PipRequirementsPath $ValidationPipRequirements
    if ((Get-Item $ValidationCondaSpecs).Length -gt 0) {
        & $MicromambaExe install -y -p $ValidationRuntimeRoot --file $ValidationCondaSpecs
        if ($LASTEXITCODE -ne 0) {
            throw "Installing validation conda package specs failed"
        }
    }
    $ValidationPython = Join-Path $ValidationRuntimeRoot "python.exe"
    if ((Get-Item $ValidationPipRequirements).Length -gt 0) {
        & $ValidationPython -m pip install --no-cache-dir -r $ValidationPipRequirements
        if ($LASTEXITCODE -ne 0) {
            throw "Installing validation pip requirements failed"
        }
    }
    & $ValidationPython -m ipykernel --version
    if ($LASTEXITCODE -ne 0) {
        throw "Validation runtime ipykernel smoke check failed"
    }
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
