<#
Resumable stage runner (Windows PowerShell). One call runs one stage or a comma-separated chain.

Each stage writes, under 04_results\ce1\_runs\:
  <stage>.log           stage output (UTF-8), written line by line
  <stage>.json          run record: status, times, host, free disk, python exit code (UTF-8 without BOM)
  <stage>.detail.json   checks and input and output md5, written by the stage itself
  console_<time>.txt    transcript of the whole call
A stage whose record says ok is skipped unless -Force is given, so an interrupted chain resumes
where it stopped.

Usage:
  powershell -ExecutionPolicy Bypass -File "<repository folder>\03_code\ce1\ce1_run.ps1" -Stage S0
  powershell -ExecutionPolicy Bypass -File "<repository folder>\03_code\ce1\ce1_run.ps1" -Stage S1,S1M,S1L
#>
param(
    [Parameter(Mandatory = $true)][string[]]$Stage,
    [switch]$Force,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
# "-Stage S1,S1M" arrives as ONE string when the script is started with -File (Windows
# PowerShell does not parse arrays there), so every value is split on commas here.
$Stage = @($Stage | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Code = Join-Path $Root "03_code"
$RunDir = Join-Path $Root "04_results\ce1\_runs"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$transcript = Join-Path $RunDir ("console_" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".txt")
Start-Transcript -Path $transcript -Force | Out-Null

$Stages = [ordered]@{
    "S0" = "ce1\ce1_envcheck.py"
    "S1" = "ce1\ce1_preprocess.py"   # WP-14: causal preprocessing, fold-specific scalars
    "S1M" = "ce1\ce1_pvmap.py"   # WP-43: available-PV plant model (D-074, D-079)
    "S1L" = "ce1\ce1_load.py"   # WP-44: representative load retrieval and scales (D-080, D-084)
    "S2" = "ce1\ce1_folds.py"
    "S4" = "ce1\ce1_point.py"   # WP-16: point benchmark (classical every fold; GRU / GRU-TCN fold 2024, both budgets) (D-089)
    "S5" = "ce1\ce1_conformal.py"   # WP-16: interval layer, static CP, baselines, ACI in event time, every fold (D-089)
    "S6" = "ce1\ce1_s6.py"   # WP-15: load-based dispatch, every fold, horizon and scenario (D-080, D-084)
    "S6P" = "ce1\ce1_s6.py"
    "S6B" = "ce1\ce1_s6.py"
    "S48" = "ce1\ce1_wp48.py"   # WP-48: fold-2024 target sensitivities SF and B, records then S2/S5/S6/S6B (D-090)
    "S48D" = "ce1\ce1_wp48.py"   # WP-88: S48's dispatch substeps S6/S6B only, v2.1 simulator, on the kept records (D-161)
    "S2X" = "ce1\ce1_folds.py"   # WP-50: extra GBM quantiles tau 0.255-0.500 for theta 0.500-0.745, Yulara folds (D-092)
    "S6X" = "ce1\ce1_s6.py"   # WP-50: load model on the new low theta levels, every scenario, plus probe rows (D-092)
    "S6P2" = "ce1\ce1_s6.py"   # WP-50: proxy rerun on the extended theta set (D-092)
    "S56" = "ce1\ce1_wp56.py"   # WP-56: dispatch realism: contingency reserve, no-load x2, minimum loading 30/50%, A-D per variant (D-112)
    "S57" = "ce1\ce1_wp57.py"   # WP-57: H1 scored with H2-fitted scores, for the cross-fitted A-R selection (D-112)
    "S1E" = "ce1\ce1_wp67_pc.py"   # WP-67 step 3: S1 on 2016-2025, must reproduce every S1 output (D-129, D-132)
    "S2E" = "ce1\ce1_wp67_pc.py"
    "S5E" = "ce1\ce1_wp67_pc.py"   # WP-67 step 3: interval layer on the S2E caches, plus an S5 reproduction (D-129, D-132)
}

function Write-Stamp($path, $obj) {
    [System.IO.File]::WriteAllText($path, ($obj | ConvertTo-Json), $Utf8NoBom)
}

function Get-FreeGB {
    $d = Get-PSDrive -Name ($Root.Substring(0, 1))
    return [math]::Round($d.Free / 1e9, 2)   # decimal GB, same unit as ce1_envcheck.py
}

$exitCode = 0
try {
    $Py = $null
    foreach ($c in @("python", "py")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { $Py = $cmd.Source; break }
    }
    if (-not $Py) { throw "Python was not found on PATH." }

    foreach ($s in $Stage) {
        if (-not $Stages.Contains($s)) {
            Write-Host "Stage '$s' is not registered in this runner (registered: $($Stages.Keys -join ', ')). Nothing was run." -ForegroundColor Yellow
            $exitCode = 2; break
        }
        $stamp = Join-Path $RunDir $(if ($Smoke) { "${s}_smoke.json" } else { "$s.json" })
        if ((Test-Path $stamp) -and -not $Force) {
            $j = [System.IO.File]::ReadAllText($stamp) | ConvertFrom-Json
            if ($j.ok) { Write-Host "Stage $s already completed ($($j.finished)); skipping. Use -Force to repeat."; continue }
        }
        $log = Join-Path $RunDir "$s.log"
        [System.IO.File]::WriteAllText($log, "", $Utf8NoBom)
        $start = Get-Date
        $freeBefore = Get-FreeGB
        Write-Host "=== CE1 stage $s started $($start.ToString('s')) (free disk $freeBefore GB) ===" -ForegroundColor Cyan
        $pyArgs = @($Stages[$s], "--stage", $s)
        if ($Smoke) { $pyArgs += "--smoke" }
        $ok = $true; $err = ""; $pyExit = $null
        Push-Location $Code
        # Windows PowerShell 5.1 turns native stderr lines into errors under "Stop"; python
        # writes warnings to stderr, so the call runs under "Continue" and success is the exit code.
        $ErrorActionPreference = "Continue"
        try {
            & $Py @pyArgs 2>&1 | ForEach-Object {
                $line = "$_"
                Write-Host $line
                [System.IO.File]::AppendAllText($log, $line + "`r`n", $Utf8NoBom)
            }
            $pyExit = $LASTEXITCODE
            if ($pyExit -ne 0) { $ok = $false; $err = "python exit code $pyExit" }
        } catch { $ok = $false; $err = $_.Exception.Message }
        $ErrorActionPreference = "Stop"
        Pop-Location
        $end = Get-Date
        Write-Stamp $stamp ([ordered]@{
            stage = $s; ok = $ok; error = $err; python_exit = $pyExit
            started = $start.ToString("s"); finished = $end.ToString("s")
            minutes = [math]::Round(($end - $start).TotalMinutes, 1)
            host = $env:COMPUTERNAME; free_gb_before = $freeBefore; free_gb_after = (Get-FreeGB)
            smoke = [bool]$Smoke; transcript = (Split-Path $transcript -Leaf) })
        if (-not $ok) {
            Write-Host "Stage $s FAILED ($err). The log is in 04_results\ce1\_runs." -ForegroundColor Red
            $exitCode = 1; break
        }
        Write-Host "=== CE1 stage $s finished in $([math]::Round(($end - $start).TotalMinutes, 1)) min ===" -ForegroundColor Green
    }
} catch {
    Write-Host "RUNNER ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Stamp (Join-Path $RunDir "runner_error.json") ([ordered]@{
        ok = $false; error = $_.Exception.Message; when = (Get-Date).ToString("s"); stages = ($Stage -join ",") })
    $exitCode = 1
}
if ($exitCode -eq 0) {
    Write-Host "Done. You can close this window. Logs and stamps are in 04_results\ce1\_runs." -ForegroundColor Green
}
Stop-Transcript | Out-Null
exit $exitCode
