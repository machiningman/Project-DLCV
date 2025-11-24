<#
PowerShell helper to run multiple training scripts concurrently.

Usage examples:
    # One-GPU machine: run pretrain on CPU and conditional training on GPU 0
    .\run_training_jobs.ps1 -Jobs @(
        @{ Script = 'Pretrain_rain_detector.py'; GPU = 'cpu'; Log = 'logs/pretrain' },
        @{ Script = 'Training_conditional.py'; GPU = '0'; Log = 'logs/training_conditional' }
    )

    # Multi-GPU: run two jobs on GPUs 0 and 1
    .\run_training_jobs.ps1 -Jobs @(
        @{ Script = 'Training_conditional.py'; GPU = '0'; Log = 'logs/training_conditional_0' },
        @{ Script = 'Training_integrated.py'; GPU = '1'; Log = 'logs/training_integrated_1' }
    )

Notes:
- Uses Start-Job to spawn background jobs. You'll get job IDs you can query with Get-Job.
- Set GPU to 'cpu' to force the job to run on CPU (avoid GPU contention).
- For conda or virtualenv, ensure the Python interpreter is accessible from this PowerShell session.
- If you need to specify a full python executable path, include it in the Script or modify the $pyExe variable.
#>
param(
    [Parameter(Mandatory=$true)]
    [array]$Jobs
)

# Create logs directory
$logsDir = Join-Path -Path $PSScriptRoot -ChildPath 'logs'
if (-not (Test-Path -Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

# Python binary (modify if you need to run in virtualenv/conda)
$pyExe = 'python'  # e.g., "C:\Users\You\anaconda3\envs\dlcv\python.exe"

# Helper to detect GPU count (if nvidia-smi is installed)
function Get-GPUCount {
    try {
        $out = & nvidia-smi --query-gpu=index --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return 0 }
        $lines = ($out -split "`n") | Where-Object { $_ -ne '' }
        return $lines.Count
    } catch {
        return 0
    }
}

$gpuCount = Get-GPUCount
Write-Host "Detected GPUs: $gpuCount"

$jobObjects = @()
$index = 0
foreach ($job in $Jobs) {
    $script = $job.Script
    $gpu = $job.GPU
    $logPrefix = $job.Log

    if (-not $script) { Write-Warning "Skipping job at index $index: no Script specified"; continue }
    if (-not $logPrefix) { $logPrefix = "job_$index" }

    # Prepare log paths
    $outLog = Join-Path -Path $logsDir -ChildPath ("$logPrefix-out.txt")
    $errLog = Join-Path -Path $logsDir -ChildPath ("$logPrefix-err.txt")

    # Build command script block
    if ($gpu -eq 'cpu') {
        $cmd = "`$env:CUDA_VISIBLE_DEVICES=''; $pyExe `"$script`" 1> `"$outLog`" 2> `"$errLog`""
    } else {
        # gpu is expected to be string index '0','1',... or comma separated list '0,1'
        $cmd = "`$env:CUDA_VISIBLE_DEVICES='$gpu'; $pyExe `"$script`" 1> `"$outLog`" 2> `"$errLog`""
    }

    # Use Start-Job to run the PowerShell command in background. This isolates each job
    $scriptBlock = [ScriptBlock]::Create($cmd)
    $psJob = Start-Job -ScriptBlock $scriptBlock -Name "train-job-$index"
    $jobObjects += $psJob

    Write-Host "Started job $index for script: $script (GPU=$gpu) => Job Id: $($psJob.Id)"
    Write-Host "  stdout -> $outLog" -ForegroundColor Gray
    Write-Host "  stderr -> $errLog" -ForegroundColor Gray

    $index++
}

Write-Host "All jobs started. Use Get-Job | Format-Table to monitor (Get-Job)."
Write-Host "Use Receive-Job -Id <id> to get output, or Get-Content -Path logs/<logfile> -Wait to tail logs." 
