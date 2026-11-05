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
    & $Python -m GM_Ver1.repair_origins --out $ArtifactRoot
    if ($LASTEXITCODE -ne 0) { throw "Forecast origin validation failed: $LASTEXITCODE" }
    $ExportArguments = @("-m", "GM_Ver1.export", "--out", $ArtifactRoot)
    if ($Dataset) { $ExportArguments += @("--dataset", $Dataset) }
    & $Python @ExportArguments
    if ($LASTEXITCODE -ne 0) { throw "Forecast verification/export failed: $LASTEXITCODE" }
    & $Python -m GM_Ver1.sync_outputs --out $ArtifactRoot
    if ($LASTEXITCODE -ne 0) { throw "Forecast artifact sync failed: $LASTEXITCODE" }
    $StackArguments = @("-m", "GM_Ver1.stacking", "--out", $ArtifactRoot)
    if ($Dataset) { $StackArguments += @("--dataset", $Dataset) }
    & $Python @StackArguments
    if ($LASTEXITCODE -ne 0) { throw "Stacking failed: $LASTEXITCODE" }
    $GradientArguments = @("-m", "GM_Ver1.gradient_descent", "--out", $ArtifactRoot)
    if ($Dataset) { $GradientArguments += @("--dataset", $Dataset) }
    & $Python @GradientArguments
    if ($LASTEXITCODE -ne 0) { throw "Gradient optimization failed: $LASTEXITCODE" }
    $PortfolioArguments = @("-m", "GM_Ver1.genetic_algorithm", "--out", $ArtifactRoot)
    if ($Dataset) { $PortfolioArguments += @("--dataset", $Dataset) }
    & $Python @PortfolioArguments
    if ($LASTEXITCODE -ne 0) { throw "GA portfolio optimization failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
