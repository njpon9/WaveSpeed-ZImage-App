<#
.SYNOPSIS
  Copy WaveSpeed Desktop sd-bin + stable-diffusion models into this repo's bundled layout.

.DESCRIPTION
  Source (default):
    $env:APPDATA\wavespeed-desktop\sd-bin          -> <repo>\bin
    $env:APPDATA\wavespeed-desktop\models\stable-diffusion -> <repo>\models\stable-diffusion

  LoRAs are NOT copied (user-specific, large). Place .safetensors in <repo>\LoRas manually.

.PARAMETER ProjectRoot
  Path to repo root (folder containing zimage_ui.py). Default: parent of scripts\

.EXAMPLE
  .\scripts\copy_wavespeed_bundle.ps1
#>
[CmdletBinding()]
param(
    [string] $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$srcBin = Join-Path $env:APPDATA "wavespeed-desktop\sd-bin"
$srcModels = Join-Path $env:APPDATA "wavespeed-desktop\models\stable-diffusion"
$dstBin = Join-Path $ProjectRoot "bin"
$dstModels = Join-Path $ProjectRoot "models\stable-diffusion"

if (-not (Test-Path (Join-Path $ProjectRoot "zimage_ui.py"))) {
    Write-Error "ProjectRoot does not look like the app repo (missing zimage_ui.py): $ProjectRoot"
}

if (-not (Test-Path $srcBin)) {
    Write-Error "WaveSpeed sd-bin not found: $srcBin`nInstall WaveSpeed Desktop or copy binaries into bin\ manually."
}
if (-not (Test-Path $srcModels)) {
    Write-Error "WaveSpeed models not found: $srcModels`nInstall WaveSpeed Desktop or copy models into models\stable-diffusion\ manually."
}

New-Item -ItemType Directory -Force -Path $dstBin | Out-Null
New-Item -ItemType Directory -Force -Path $dstModels | Out-Null

Write-Host "Robocopy sd-bin -> bin\ ..."
robocopy $srcBin $dstBin /E /NFL /NDL /NJH /NJS /nc /ns /np
$rc1 = $LASTEXITCODE
if ($rc1 -ge 8) { Write-Error "robocopy sd-bin failed with exit code $rc1" }

Write-Host "Robocopy models\stable-diffusion -> models\stable-diffusion\ ..."
robocopy $srcModels $dstModels /E /NFL /NDL /NJH /NJS /nc /ns /np
$rc2 = $LASTEXITCODE
if ($rc2 -ge 8) { Write-Error "robocopy models failed with exit code $rc2" }

Write-Host ""
Write-Host "Done. Next: copy your LoRA .safetensors into: $(Join-Path $ProjectRoot 'LoRas')"
Write-Host "Then: start-ui.bat  or  py -3 zimage_ui.py"
