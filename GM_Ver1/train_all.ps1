param(
    [Parameter(Mandatory = $true)][string]$ArtifactRoot,
    [string]$Python = "python",
    [string]$Dataset = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $RepoRoot
try {
    $Arguments = @("-m", "GM_Ver1.run", "--out", $ArtifactRoot)
    if ($Dataset) { $Arguments += @("--dataset", $Dataset) }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Forecast training failed: $LASTEXITCODE" }
    $ExportArguments = @("-m", "GM_Ver1.export", "--out", $ArtifactRoot)
    if ($Dataset) { $ExportArguments += @("--dataset", $Dataset) }
    & $Python @ExportArguments
    if ($LASTEXITCODE -ne 0) { throw "Forecast verification/export failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
