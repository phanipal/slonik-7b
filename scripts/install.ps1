$ErrorActionPreference = "Stop"

$python = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py -3.11" }

Write-Host "Creating venv .venv..." -ForegroundColor Cyan
& $python -m venv .venv

$activate = Join-Path $PWD ".venv\Scripts\Activate.ps1"
. $activate

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip wheel setuptools

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Using uv for fast install..." -ForegroundColor Cyan
    uv pip install -r requirements.txt
} else {
    Write-Host "Installing with pip..." -ForegroundColor Cyan
    pip install -r requirements.txt
}

Write-Host "Installing project (editable)..." -ForegroundColor Cyan
pip install -e .

Write-Host "Verifying CUDA + bitsandbytes..." -ForegroundColor Cyan
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import bitsandbytes; print('bnb:', bitsandbytes.__version__)"
python -c "from unsloth import FastLanguageModel; print('unsloth: ok')"

Write-Host "Install complete." -ForegroundColor Green
Write-Host "Copy .env.example to .env and fill in HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY before training." -ForegroundColor Yellow
