# Usage: pwsh -File scripts/setup_env.ps1
# Creates a .venv and installs dependencies. For GPU, install a CUDA-matched torch manually if needed.

param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Set --Python to an executable path."; exit 1
}

if (-not (Test-Path $VenvPath)) {
    & $Python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv."; exit 1 }
}

$activate = Join-Path $VenvPath "Scripts" "Activate.ps1"
if (-not (Test-Path $activate)) { Write-Error "Activate script not found at $activate"; exit 1 }
. $activate

Write-Host "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Done. To use this venv later, run: . $activate"
