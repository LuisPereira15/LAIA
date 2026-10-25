# Run from a normal PowerShell window with Docker Desktop's Linux engine active.
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
$taskLogDir = Join-Path $taskRoot "artifacts"
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
$taskLog = Join-Path $taskLogDir "docker-check.log"
Start-Transcript -Path $taskLog -Force | Out-Null
$env:COMPOSE_ANSI = "never"
$env:BUILDKIT_PROGRESS = "plain"
$taskCompose = @("compose", "-f", "docker-compose.yml", "-f", "compose.demo.yml")
function Invoke-TaskDocker {
    param([string[]]$DockerArguments)
    Write-Host ("docker " + ($DockerArguments -join " "))
    $taskPreviousPreference = $ErrorActionPreference
    try {
        # Docker emits normal build progress on stderr. Capture it as text.
        $ErrorActionPreference = "Continue"
        & docker @DockerArguments 2>&1 | ForEach-Object { Write-Host $_ }
        $taskExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $taskPreviousPreference
    }
    if ($taskExitCode -ne 0) {
        throw "Docker command failed with exit code $taskExitCode. See artifacts/docker-check.log."
    }
}
try {
    Invoke-TaskDocker -DockerArguments @("version")
    $env:GIT_COMMIT = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not determine Git commit." }
    if (& git status --porcelain) { $env:GIT_COMMIT += "-dirty" }
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("config", "--quiet"))
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("build", "mlflow", "train", "api", "test", "verify"))
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("run", "--rm", "--no-deps", "test"))
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("up", "-d", "--wait", "mlflow"))
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("run", "--rm", "train"))
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("up", "-d", "--wait", "api"))
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("run", "--rm", "verify"))
    # Capture the exact run before recreation. Named volumes must preserve it.
    $taskConfigJson = & docker @taskCompose config --format json
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect Compose configuration." }
    $taskConfig = $taskConfigJson | ConvertFrom-Json
    $taskApiPort = $taskConfig.services.api.ports[0].published
    $taskApiUrl = "http://127.0.0.1:$taskApiPort"
    $taskBefore = (Invoke-RestMethod "$taskApiUrl/health/ready").run_id
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("up", "-d", "--force-recreate", "--wait", "mlflow", "api"))
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("run", "--rm", "verify"))
    $taskAfter = (Invoke-RestMethod "$taskApiUrl/health/ready").run_id
    if ($taskBefore -ne $taskAfter) { throw "Loaded model changed after recreation." }
    Invoke-TaskDocker -DockerArguments ($taskCompose + @("ps"))
    Write-Host "PASS: container tests, training, API, MLflow, registry, artifacts and persistence."
    Write-Host "Synthetic demo only. API: $taskApiUrl/docs"
    Write-Host "Services remain running. Logs: $taskLog"
} catch {
    & docker @taskCompose logs --no-color --tail 100 mlflow api
    throw
} finally {
    Stop-Transcript | Out-Null
    Pop-Location
}
