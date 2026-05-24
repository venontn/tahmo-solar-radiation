# Copy project to Google Drive for Desktop (when installed).
# Default Drive folder paths are tried automatically.

param(
    [string]$DriveRoot = "",
    [string]$ProjectName = "tahmo-solar-radiation"
)

$source = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $source "src\train_predict.py"))) {
    $source = "C:\Users\PC\Projects\tahmo-solar-radiation"
}

$candidates = @(
    $DriveRoot,
    "$env:USERPROFILE\Google Drive",
    "$env:USERPROFILE\My Drive",
    "G:\My Drive",
    "G:\Other computers\My Laptop\Google Drive"
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $candidates) {
    Write-Error @"
Google Drive folder not found. Either:
  1. Install Google Drive for Desktop: https://www.google.com/drive/download/
  2. Pass -DriveRoot 'C:\path\to\My Drive'
  3. Use notebooks/tahmo_colab_drive.ipynb in Google Colab (upload data to Drive first)
"@
    exit 1
}

$destRoot = Join-Path $candidates[0] $ProjectName
New-Item -ItemType Directory -Force -Path $destRoot | Out-Null

$exclude = @('.venv', '__pycache__', '.git', 'output\submission.csv')
robocopy $source $destRoot /E /XD .venv .git __pycache__ /XF *.pyc /NFL /NDL /NJH /NJS /nc /ns /np
if ($LASTEXITCODE -ge 8) { exit $LASTEXITCODE }

Write-Host "Synced to: $destRoot"
Write-Host "Open notebooks/tahmo_colab_drive.ipynb in Colab and set PROJECT_DIR to:"
Write-Host "  /content/drive/MyDrive/$ProjectName"
