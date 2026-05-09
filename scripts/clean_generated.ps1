param(
    [switch]$IncludeModels
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Assert-InWorkspace {
    param([Parameter(Mandatory = $true)][string]$PathToCheck)

    $full = if (Test-Path -LiteralPath $PathToCheck) {
        (Resolve-Path -LiteralPath $PathToCheck).Path
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $Root $PathToCheck))
    }

    if (-not ($full.Equals($Root, [System.StringComparison]::OrdinalIgnoreCase) -or
        $full.StartsWith($Root + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing to touch path outside workspace: $full"
    }
    return $full
}

function Remove-WorkspaceItem {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $full = Assert-InWorkspace $RelativePath
    if (Test-Path -LiteralPath $full) {
        Write-Host "[clean] remove $RelativePath"
        Remove-Item -LiteralPath $full -Recurse -Force
    }
}

Write-Host "[clean] root $Root"

Get-ChildItem -LiteralPath $Root -Recurse -Force -Directory -Filter "__pycache__" |
    ForEach-Object {
        $full = Assert-InWorkspace $_.FullName
        Write-Host "[clean] remove $full"
        Remove-Item -LiteralPath $full -Recurse -Force
    }

foreach ($pattern in @("*.pyc", "*.pyo")) {
    Get-ChildItem -LiteralPath $Root -Recurse -Force -File -Filter $pattern |
        ForEach-Object {
            $full = Assert-InWorkspace $_.FullName
            Write-Host "[clean] remove $full"
            Remove-Item -LiteralPath $full -Force
        }
}

foreach ($path in @(
    "agent\runs",
    "agent\logs",
    "agent\ppo_abr_tensorboard",
    "processor\uploads",
    "processor\dash_output"
)) {
    Remove-WorkspaceItem $path
}

if ($IncludeModels) {
    foreach ($path in @(
        "agent\models\netllm_sft",
        "agent\models\netllm_offline_rl",
        "agent\models\netllm_rl",
        "agent\models\pensieve_torch"
    )) {
        Remove-WorkspaceItem $path
    }

    $modelDir = Assert-InWorkspace "agent\models"
    foreach ($pattern in @("evaluation_*.json", "*validation_history.json", "sft_val_traces_*.txt")) {
        Get-ChildItem -LiteralPath $modelDir -Force -File -Filter $pattern |
            ForEach-Object {
                $full = Assert-InWorkspace $_.FullName
                Write-Host "[clean] remove $full"
                Remove-Item -LiteralPath $full -Force
            }
    }
}

foreach ($path in @(
    "agent\models",
    "processor\uploads",
    "processor\dash_output"
)) {
    $full = Assert-InWorkspace $path
    New-Item -ItemType Directory -Force -Path $full | Out-Null
}

Write-Host "[clean] done"
