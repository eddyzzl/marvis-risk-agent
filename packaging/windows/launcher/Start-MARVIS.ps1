[CmdletBinding()]
param(
    [string] $HostName = "127.0.0.1",
    [int] $Port = 8000,
    [string] $Workspace = "",
    [switch] $NoBrowser
)

$ErrorActionPreference = "Stop"

$InstallRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $InstallRoot "runtime"
$PythonExe = Join-Path $RuntimeRoot "python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "MARVIS runtime python was not found at $PythonExe"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = Join-Path $env:LOCALAPPDATA "MARVIS-Agent\workspace"
}

$LogRoot = Join-Path $env:LOCALAPPDATA "MARVIS-Agent\logs"
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$JavaHome = Join-Path $RuntimeRoot "Library"
$env:MARVIS_HOME = $InstallRoot
$env:JAVA_HOME = $JavaHome
$env:PATH = "$RuntimeRoot;$RuntimeRoot\Scripts;$RuntimeRoot\Library\bin;$env:PATH"

function Initialize-ValidationKernelSpec {
    param([string] $InstallRoot)

    $ValidationPython = Join-Path $InstallRoot "validation-runtime\python.exe"
    if (-not (Test-Path $ValidationPython)) {
        return
    }

    $ValidationRoot = Split-Path -Parent $ValidationPython
    $ValidationPath = "$ValidationRoot;$ValidationRoot\Scripts;$ValidationRoot\Library\bin;$env:PATH"
    $KernelSpec = [ordered]@{
        argv = @(
            $ValidationPython,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}"
        )
        display_name = "MARVIS Validation (pkg.txt)"
        language = "python"
        env = @{
            CONDA_DEFAULT_ENV = $ValidationRoot
            CONDA_PREFIX = $ValidationRoot
            PATH = $ValidationPath
            PYTHONNOUSERSITE = "1"
            PYTHONPATH = ""
        }
        metadata = @{
            marvis = @{
                role = "validation"
                source = "packaging/windows/validation/pkg.txt"
                runtime = "validation-runtime"
            }
        }
    }
    $KernelDirs = New-Object System.Collections.Generic.List[string]
    $KernelDirs.Add((Join-Path $InstallRoot "kernels\marvis-validation-pkg"))
    if (-not [string]::IsNullOrWhiteSpace($env:APPDATA)) {
        $KernelDirs.Add((Join-Path $env:APPDATA "jupyter\kernels\marvis-validation-pkg"))
    }
    foreach ($KernelDir in $KernelDirs) {
        New-Item -ItemType Directory -Force -Path $KernelDir | Out-Null
        $KernelSpec |
            ConvertTo-Json -Depth 8 |
            Set-Content -Encoding utf8 -Path (Join-Path $KernelDir "kernel.json")
    }
}

Initialize-ValidationKernelSpec -InstallRoot $InstallRoot
if ([string]::IsNullOrWhiteSpace($env:JUPYTER_PATH)) {
    $env:JUPYTER_PATH = $InstallRoot
}
else {
    $env:JUPYTER_PATH = "$InstallRoot;$env:JUPYTER_PATH"
}

$BaseUrl = "http://${HostName}:${Port}"
$HealthUrl = "$BaseUrl/api/health"

function Test-MarvisHealth {
    param([string] $Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}

function Get-MarvisListenerProcessInfo {
    param([int] $Port)
    try {
        $Connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $Connection) {
            return $null
        }
        $Process = Get-Process -Id $Connection.OwningProcess -ErrorAction SilentlyContinue
        $CimProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($Connection.OwningProcess)" -ErrorAction SilentlyContinue
        return [pscustomobject]@{
            Id = $Connection.OwningProcess
            Process = $Process
            Path = if ($null -ne $Process) { $Process.Path } else { "" }
            CommandLine = if ($null -ne $CimProcess) { $CimProcess.CommandLine } else { "" }
            StartTime = if ($null -ne $Process) { $Process.StartTime } else { $null }
        }
    }
    catch {
        return $null
    }
}

function Test-SamePath {
    param(
        [string] $Left,
        [string] $Right
    )
    if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) {
        return $false
    }
    try {
        return ([System.IO.Path]::GetFullPath($Left) -ieq [System.IO.Path]::GetFullPath($Right))
    }
    catch {
        return ($Left -ieq $Right)
    }
}

function Test-MarvisServeProcess {
    param($ProcessInfo)
    if ($null -eq $ProcessInfo) {
        return $false
    }
    $CommandLine = "$($ProcessInfo.CommandLine)"
    return ($CommandLine -match "(?i)\bmarvis\b" -and $CommandLine -match "(?i)\bserve\b")
}

function Test-CurrentInstallProcess {
    param(
        $ProcessInfo,
        [string] $PythonExe,
        [string] $InstallRoot
    )
    if ($null -eq $ProcessInfo) {
        return $false
    }
    if (-not (Test-SamePath -Left $ProcessInfo.Path -Right $PythonExe)) {
        return $false
    }
    $VersionFile = Join-Path $InstallRoot "VERSION.txt"
    if (Test-Path $VersionFile) {
        try {
            $VersionWriteTime = (Get-Item $VersionFile).LastWriteTime
            if ($ProcessInfo.StartTime -and $ProcessInfo.StartTime -lt $VersionWriteTime) {
                return $false
            }
        }
        catch {
            return $false
        }
    }
    return $true
}

function Wait-PortRelease {
    param(
        [int] $Port,
        [int] $TimeoutSeconds = 10
    )
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        $Connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $Connection) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

$ShouldStart = $true
if (Test-MarvisHealth -Url $HealthUrl) {
    $ExistingProcess = Get-MarvisListenerProcessInfo -Port $Port
    if (Test-CurrentInstallProcess -ProcessInfo $ExistingProcess -PythonExe $PythonExe -InstallRoot $InstallRoot) {
        $ShouldStart = $false
    }
    elseif (Test-MarvisServeProcess -ProcessInfo $ExistingProcess) {
        Write-Host "Stopping stale MARVIS process on port $Port (PID $($ExistingProcess.Id)) before launching this installation."
        Stop-Process -Id $ExistingProcess.Id -Force
        if (-not (Wait-PortRelease -Port $Port)) {
            Write-Error "Port $Port did not become available after stopping stale MARVIS process $($ExistingProcess.Id)."
            exit 1
        }
    }
    else {
        Write-Error "Port $Port is already serving /api/health but is not a MARVIS process from this installation. Stop that process or launch MARVIS-Agent with a different -Port."
        exit 1
    }
}

if ($ShouldStart) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $StdoutLog = Join-Path $LogRoot "marvis-$Timestamp.out.log"
    $StderrLog = Join-Path $LogRoot "marvis-$Timestamp.err.log"
    $Arguments = @(
        "-m",
        "marvis",
        "serve",
        "--host",
        $HostName,
        "--port",
        "$Port",
        "--workspace",
        $Workspace
    )

    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $InstallRoot `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog

    $deadline = (Get-Date).AddSeconds(75)
    while ((Get-Date) -lt $deadline) {
        if ($process.HasExited) {
            Write-Host "MARVIS exited during startup. Logs:"
            Write-Host "stdout: $StdoutLog"
            Write-Host "stderr: $StderrLog"
            exit $process.ExitCode
        }
        if (Test-MarvisHealth -Url $HealthUrl) {
            break
        }
        Start-Sleep -Seconds 1
    }

    if (-not (Test-MarvisHealth -Url $HealthUrl)) {
        Write-Host "MARVIS did not become ready at $HealthUrl before the startup timeout."
        Write-Host "stdout: $StdoutLog"
        Write-Host "stderr: $StderrLog"
        exit 1
    }
}

if (-not $NoBrowser) {
    Start-Process $BaseUrl
}

Write-Host "MARVIS-Agent is running at $BaseUrl"
Write-Host "Workspace: $Workspace"
Write-Host "Logs: $LogRoot"
