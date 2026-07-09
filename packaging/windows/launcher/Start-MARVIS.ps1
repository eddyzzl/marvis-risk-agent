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

    $KernelDir = Join-Path $InstallRoot "kernels\marvis-validation-pkg"
    New-Item -ItemType Directory -Force -Path $KernelDir | Out-Null

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
        metadata = @{
            marvis = @{
                role = "validation"
                source = "packaging/windows/validation/pkg.txt"
                runtime = "validation-runtime"
            }
        }
    }
    $KernelSpec |
        ConvertTo-Json -Depth 8 |
        Set-Content -Encoding utf8 -Path (Join-Path $KernelDir "kernel.json")
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

if (-not (Test-MarvisHealth -Url $HealthUrl)) {
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
