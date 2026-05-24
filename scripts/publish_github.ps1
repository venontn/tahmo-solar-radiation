# Create GitHub repo and push (run after: gh auth login)
# Author: Venon Takunda Nyadombo — no third-party co-authors on commits.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "Install GitHub CLI: https://cli.github.com/"
}

gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run: gh auth login"
    exit 1
}

git branch -M main
$exists = gh repo view venontn/tahmo-solar-radiation 2>$null
if ($LASTEXITCODE -ne 0) {
    gh repo create tahmo-solar-radiation `
        --public `
        --source . `
        --remote origin `
        --description "TAHMO solar radiation prediction (Zindi) by Venon Takunda Nyadombo" `
        --push
} else {
    git push -u origin main
}

Write-Host "Done: https://github.com/venontn/tahmo-solar-radiation"
