# LoreKeeper Stop Script
#
#   .\stop.ps1                Stop backend and UI cleanly; kill whatever still runs after -Timeout seconds
#   .\stop.ps1 -Force         Also while an ingest is running (the run is aborted; the next ingest redoes it)
#
# Finds the processes by the ports start.ps1 uses and sends Ctrl+C to their console windows — the
# same as pressing it there, so uvicorn runs its shutdown and the backend stops the Ollama it started.
# Closing a window instead kills the process before that shutdown runs (an Ollama started by
# LoreKeeper then keeps running). Also run by the "LoreKeeper beenden" button in the UI sidebar
# (ui/shutdown.py); its output then goes to logs/stop.log.

param([switch]$Force, [int]$Timeout = 20)

$ROOT         = $PSScriptRoot
$PYTHON       = Join-Path $ROOT ".venv\Scripts\python.exe"
$BACKEND_PORT = 8000
$UI_PORT      = 8501

# Attach to the target's console and raise Ctrl+C there. Runs in its own python process: detaching
# this PowerShell from its console would cut off its own output.
$SEND_CTRL_C = @'
import ctypes, sys
k = ctypes.windll.kernel32
k.FreeConsole()
if not k.AttachConsole(int(sys.argv[1])):
    sys.exit(1)
k.SetConsoleCtrlHandler(None, True)  # ignore the event in this helper itself
sys.exit(0 if k.GenerateConsoleCtrlEvent(0, 0) else 1)
'@

function Get-ListenerPid($port) {
    (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
}

# --- Refuse during an ingest -------------------------------------------
# A forced kill mid-ingest can leave a half-written index; waiting is cheap, -Force is explicit.
if (-not $Force -and (Get-ListenerPid $BACKEND_PORT)) {
    $job = $null
    try { $job = Invoke-RestMethod -Uri "http://127.0.0.1:$BACKEND_PORT/ingest/active" -TimeoutSec 5 } catch { }
    # .job_id, not just $job: Windows PowerShell 5.1 (what the UI button runs) turns the JSON null of
    # "no job" into a non-empty value
    if ($job.job_id) {
        Write-Warning "An ingest is running ($($job.documents_processed)/$($job.documents_total) documents). Wait for it, or use -Force to abort it."
        exit 1
    }
}

# --- Find the process trees --------------------------------------------
$all = @(Get-CimInstance Win32_Process)
$byId = @{}
foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }

function Get-Root($id) {
    # The port can belong to a child: the venv's python.exe is a launcher for the real interpreter,
    # and with --reload the worker is a multiprocessing child of the reloader. Ctrl+C goes to the top one.
    $p = $byId[[int]$id]
    while ($p) {
        $parent = $byId[[int]$p.ParentProcessId]
        if (-not $parent -or $parent.Name -ne "python.exe" -or $parent.CommandLine -notmatch "uvicorn|streamlit|multiprocessing") { break }
        $p = $parent
    }
    $p
}

function Get-Descendants($id) {
    foreach ($child in $all | Where-Object { $_.ParentProcessId -eq $id -and $_.ProcessId -ne $id }) {
        $child
        Get-Descendants $child.ProcessId
    }
}

$roots = @()
foreach ($port in $BACKEND_PORT, $UI_PORT) {
    $listener = Get-ListenerPid $port
    if ($listener) {
        $root = Get-Root $listener
        if ($root -and $roots.ProcessId -notcontains $root.ProcessId) { $roots += $root }
    }
}
if (-not $roots) {
    Write-Host "LoreKeeper is not running (nothing listens on ports $BACKEND_PORT/$UI_PORT)." -ForegroundColor Green
    exit 0
}
$tree = @($roots) + @($roots | ForEach-Object { Get-Descendants $_.ProcessId })
# Recorded now: once the backend is gone, an Ollama it started can no longer be traced back to it
$ollama = @($tree | Where-Object { $_.Name -like "ollama*" })

# --- Ctrl+C, wait, kill ----------------------------------------------
$sw = [Diagnostics.Stopwatch]::StartNew()
foreach ($root in $roots) {
    $label = $root.CommandLine -replace '^.*?python\.exe"?\s*', ''
    if ($label.Length -gt 60) { $label = $label.Substring(0, 60) }
    Write-Host "Sending Ctrl+C to PID $($root.ProcessId) ($label)"
    & $PYTHON -c $SEND_CTRL_C $root.ProcessId
    if ($LASTEXITCODE -ne 0) { Write-Warning "PID $($root.ProcessId) has no console to send Ctrl+C to - it is killed after the timeout." }
}

$ids = @($tree | Where-Object { $_.Name -notlike "ollama*" } | ForEach-Object { [int]$_.ProcessId })
Wait-Process -Id $ids -Timeout $Timeout -ErrorAction SilentlyContinue
$left = @(Get-Process -Id ($ids + @($ollama | ForEach-Object { [int]$_.ProcessId })) -ErrorAction SilentlyContinue)
if ($left) {
    Write-Warning "Still running after $Timeout s - killing: $(($left | ForEach-Object { "$($_.Id) $($_.ProcessName)" }) -join ', ')"
    $left | Stop-Process -Force -ErrorAction SilentlyContinue
}

if ((Get-ListenerPid $BACKEND_PORT) -or (Get-ListenerPid $UI_PORT)) {
    Write-Warning "A port is still in use - check with: Get-NetTCPConnection -State Listen -LocalPort $BACKEND_PORT,$UI_PORT"
    exit 1
}
Write-Host ("LoreKeeper stopped after {0:N1} s{1}." -f $sw.Elapsed.TotalSeconds, $(if ($left) { " (some processes killed)" } else { " (clean)" })) -ForegroundColor Green
