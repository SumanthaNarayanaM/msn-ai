<#
.SYNOPSIS
    Download the MSN AI local chat model (a small GGUF file) into
    D:\MSN AI\model_storage\base.

.DESCRIPTION
    Fetches the default Qwen2.5-1.5B-Instruct Q4_K_M GGUF (~1.1 GB) from
    Hugging Face. Override the repo/file/path via environment variables or
    -Repo / -File / -Destination parameters. Uses curl.exe so a partial
    download can be resumed by re-running the script.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\fetch_model.ps1
#>
param(
    [string]$Repo  = $env:MSN_AI_MODEL_HF_REPO,
    [string]$File  = $env:MSN_AI_MODEL_HF_FILE,
    [string]$Destination = $env:MSN_AI_MODEL_PATH
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not $Repo) { $Repo = "Qwen/Qwen2.5-1.5B-Instruct-GGUF" }
if (-not $File) { $File = "qwen2.5-1.5b-instruct-q4_k_m.gguf" }
if (-not $Destination) { $Destination = Join-Path $root "model_storage\base\$File" }

$targetDir = Split-Path -Parent $Destination
New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

$url = "https://huggingface.co/$Repo/resolve/main/$File"

Write-Host "MSN AI model download"
Write-Host "  URL : $url"
Write-Host "  TO  : $Destination"
Write-Host "This is an openly available local model (Qwen2.5-1.5B-Instruct, Apache-2.0)."

if (Test-Path $Destination) {
    $sizeGB = [Math]::Round((Get-Item $Destination).Length / 1GB, 2)
    Write-Host "Target already exists ($sizeGB GB). Re-running continues/resumes the download."
}

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw "curl.exe is required (built into Windows 10/11). Aborting."
}

& $curl.Source -L --fail --retry 3 -C - -o $Destination $url
if ($LASTEXITCODE -ne 0) {
    throw "Download failed with exit code $LASTEXITCODE."
}

$final = Get-Item $Destination
Write-Host ""
Write-Host "SUCCESS: $($final.FullName) ($([Math]::Round($final.Length/1GB,2)) GB)"
Write-Host "Set MSN_AI_PROVIDER=llamacpp (and MSN_AI_MODEL_NAME=qwen2.5-1.5b-instruct) in .env, then start the server."