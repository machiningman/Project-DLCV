# Simple sequential training script for overnight runs
# Activates venv, runs Pretrain → Training_conditional with logging

# Configuration
$venvPath = "E:\Python\DLCV\.venv\Scripts\Activate.ps1"
$logDir = "logs"

# Create logs directory
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Overnight Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate virtual environment
Write-Host "`n[1/3] Activating virtual environment..." -ForegroundColor Yellow
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
if (-not (Test-Path $venvPath)) {
    Write-Error "Activation script not found: $venvPath"
    exit 1
}
try {
    . $venvPath
} catch {
    Write-Error ("Failed to activate virtual environment: " + $_.Exception.Message)
    exit 1
}
Write-Host "Virtual environment activated" -ForegroundColor Green
try {
    $pyPath = (Get-Command python -ErrorAction Stop).Source
    Write-Host "Python executable: $pyPath" -ForegroundColor Gray
} catch {
    # If python not found, try to derive python.exe from the venv
    $venvRoot = Split-Path -Parent $venvPath
    $possiblePython = Join-Path -Path $venvRoot -ChildPath "python.exe"
    if (Test-Path $possiblePython) {
        $pyPath = $possiblePython
        Write-Host "Python executable found inside venv: $pyPath" -ForegroundColor Gray
    } else {
        Write-Error "Error: python not found in PATH and not found in venv. Make sure venv activation succeeded or provide correct path." -ForegroundColor Red
        exit 1
    }
}
    
# Use explicit python executable if available (avoids dependency on dot-sourcing)
try {
    if (-not $pyPath) {
        $pyPath = (Join-Path -Path (Split-Path $venvPath -Parent) -ChildPath "python.exe")
    }
} catch {
    # fallback: leave default python in PATH
}

# Step 1: Pretrain rain detector (~30 minutes)
Write-Host "`n[2/3] Starting Phase 0: Rain Detector Pretraining..." -ForegroundColor Yellow
Write-Host "Estimated time: ~30 minutes" -ForegroundColor Gray
Write-Host "Logs: $logDir\pretrain-out.txt" -ForegroundColor Gray

$startTime = Get-Date
& "$pyPath" Pretrain_rain_detector.py 2>&1 | Tee-Object -FilePath "$logDir\pretrain-out.txt"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Pretrain failed! Check $logDir\pretrain-out.txt for errors"
    exit 1
}

$pretrainTime = (Get-Date) - $startTime
Write-Host "Pretrain completed in $($pretrainTime.ToString('hh\:mm\:ss'))" -ForegroundColor Green

# Step 2: Train conditional model (~6-8 hours)
Write-Host "`n[3/3] Starting Phases 1-3: Conditional Model Training..." -ForegroundColor Yellow
Write-Host "Estimated time: ~6-8 hours" -ForegroundColor Gray
Write-Host "Logs: $logDir\training-out.txt" -ForegroundColor Gray

$startTime = Get-Date
& "$pyPath" Training_conditional.py 2>&1 | Tee-Object -FilePath "$logDir\training-out.txt"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed! Check $logDir\training-out.txt for errors"
    exit 1
}

$trainingTime = (Get-Date) - $startTime
Write-Host "Training completed in $($trainingTime.ToString('hh\:mm\:ss'))" -ForegroundColor Green

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All Training Jobs Completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pretrain time: $($pretrainTime.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "Training time: $($trainingTime.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "Total time: $(((Get-Date) - $startTime + $pretrainTime).ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "`nOutputs:" -ForegroundColor White
Write-Host "  - Rain detector: .\rain_detector_pretrained\rain_detector_best.pt" -ForegroundColor Gray
Write-Host "  - Conditional model: .\outputs_conditional\best_conditional" -ForegroundColor Gray
Write-Host "  - Logs: .\logs" -ForegroundColor Gray
Write-Host ""
Write-Host 'Next step: python Eval_conditional.py' -ForegroundColor Yellow
