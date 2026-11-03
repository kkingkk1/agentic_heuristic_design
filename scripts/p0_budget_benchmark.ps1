param(
    [Parameter(Mandatory = $true)]
    [string]$RunName
)

$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') {
    throw 'RunName 只允许字母、数字、下划线和连字符。'
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outputRoot = Join-Path $projectRoot 'outputs\p0_budget'
$runRoot = Join-Path $outputRoot $RunName
if (Test-Path -LiteralPath $runRoot) {
    throw "运行目录已存在：$runRoot"
}

$python = 'C:\Users\a_wei\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe'
$reference = Join-Path $projectRoot 'outputs\p0_tsp_aco\20261103_174821_76657347\reference_final_a\problems\tsp_aco\gpt.py'
$supervisor = Join-Path $projectRoot 'scripts\p0_tsp_aco_evaluate.py'
foreach ($required in @($python, $reference, $supervisor)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "缺少测量输入：$required"
    }
}
if ((Get-Process -Id $PID).PriorityClass -ne 'Normal') {
    throw '测量进程不是 Normal 优先级。'
}

New-Item -ItemType Directory -Path $runRoot | Out-Null
$environmentNames = @('PYTHONPATH', 'PYTHONUTF8', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
$originalEnvironment = @{}
foreach ($name in $environmentNames) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:PYTHONPATH = 'D:\Tools\paddleocr-document-rebuild\.venv\Lib\site-packages;D:\Tools\mineru\.venv\Lib\site-packages'
    $env:PYTHONUTF8 = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'

    $runtimePath = Join-Path $runRoot 'runtime.txt'
    & $python -B -c 'import numpy as np, torch; print("torch_intraop:", torch.get_num_threads(), "torch_interop:", torch.get_num_interop_threads()); np.show_runtime()' |
        Out-File -LiteralPath $runtimePath -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "读取数值库运行状态失败，退出码：$LASTEXITCODE"
    }

    $workers = @()
    $start = [System.Diagnostics.Stopwatch]::StartNew()
    foreach ($index in 1..4) {
        $workerRoot = Join-Path $runRoot "reference_$index"
        $candidateDir = Join-Path $workerRoot 'problems\tsp_aco'
        New-Item -ItemType Directory -Path $candidateDir -Force | Out-Null
        Copy-Item -LiteralPath $reference -Destination (Join-Path $candidateDir 'gpt.py')
        $process = Start-Process -FilePath $python -ArgumentList @('-B', $supervisor, $workerRoot) `
            -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $workerRoot 'driver_stdout.txt') `
            -RedirectStandardError (Join-Path $workerRoot 'driver_stderr.txt')
        $workers += [pscustomobject]@{ index = $index; root = $workerRoot; process = $process }
    }

    $samples = [System.Collections.Generic.List[object]]::new()
    while (@($workers | Where-Object { -not $_.process.HasExited }).Count -gt 0) {
        $workerIds = @($workers | ForEach-Object { $_.process.Id })
        $children = @(Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -in $workerIds -and $_.Name -eq 'python.exe'
        })
        $processInfo = @($children | ForEach-Object {
            $child = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
            if ($child) {
                [pscustomobject]@{
                    pid = $child.Id
                    parent_pid = $_.ParentProcessId
                    working_set_bytes = $child.WorkingSet64
                    priority = $child.PriorityClass.ToString()
                }
            }
        })
        $os = Get-CimInstance Win32_OperatingSystem
        $samples.Add([pscustomobject]@{
            elapsed_seconds = $start.Elapsed.TotalSeconds
            available_memory_bytes = [int64]$os.FreePhysicalMemory * 1024
            evaluators = $processInfo
        })
        Start-Sleep -Seconds 1
    }
    $start.Stop()

    $results = @($workers | ForEach-Object {
        $_.process.WaitForExit()
        $resultPath = Join-Path $_.root 'result.json'
        [pscustomobject]@{
            index = $_.index
            supervisor_exit_code = $_.process.ExitCode
            result_path = $resultPath
            result = if (Test-Path -LiteralPath $resultPath) {
                Get-Content -Raw -LiteralPath $resultPath -Encoding utf8 | ConvertFrom-Json
            } else { $null }
        }
    })
    $summary = [ordered]@{
        started_utc = [DateTimeOffset]::UtcNow.AddSeconds(-$start.Elapsed.TotalSeconds).ToString('o')
        concurrency = 4
        elapsed_seconds = $start.Elapsed.TotalSeconds
        source_reference = $reference
        thread_environment = @{ OMP_NUM_THREADS = 1; MKL_NUM_THREADS = 1; OPENBLAS_NUM_THREADS = 1; NUMEXPR_NUM_THREADS = 1 }
        samples = $samples
        workers = $results
    }
    $summaryPath = Join-Path $runRoot 'benchmark.json'
    $summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    Write-Output "测量记录：$summaryPath"
    if (@($results | Where-Object { $_.supervisor_exit_code -ne 0 -or $_.result.status -ne 'passed' }).Count -gt 0) {
        throw '至少一个参考评价未通过；保留全部测量输出。'
    }
} finally {
    foreach ($name in $environmentNames) {
        if ($null -eq $originalEnvironment[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], 'Process')
        }
    }
}
