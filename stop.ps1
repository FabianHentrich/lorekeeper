# LoreKeeper Stop Script
#
#   .\stop.ps1                Stop backend and UI cleanly; kill whatever still runs after -Timeout seconds
#   .\stop.ps1 -Force         Also while an ingest is running (the run is aborted; the next ingest redoes it)
#   .\stop.ps1 -List          Only print the processes that would be stopped, stop nothing
#   .\stop.ps1 -Restart       Stop, then start.ps1 again in the mode that ran (server or development);
#                             no start when the stop failed (ingest running, port still busy)
#
# Finds the processes by the ports start.ps1 uses and sends Ctrl+C to their console windows — the
# same as pressing it there, so uvicorn runs its shutdown and the backend stops the Ollama it started.
# Closing a window instead kills the process before that shutdown runs (an Ollama started by
# LoreKeeper then keeps running). Also run by the "LoreKeeper beenden" / "neu starten" buttons in the
# UI sidebar (ui/shutdown.py); their output then goes to logs/stop.log.

param([switch]$Force, [switch]$List, [switch]$Restart, [int]$Timeout = 20)

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
# -List stops nothing, so it also works during an ingest.
if (-not $Force -and -not $List -and (Get-ListenerPid $BACKEND_PORT)) {
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
    # Windows reuses PIDs: a process whose parent has exited still names that PID, and if LoreKeeper's
    # python later gets it, the orphan looks like its child. A real child starts after its parent — the
    # check keeps e.g. the browser start.ps1 opened out of the kill list (it was killed on 14.10.).
    $parent = $byId[[int]$id]
    foreach ($child in $all | Where-Object { $_.ParentProcessId -eq $id -and $_.ProcessId -ne $id }) {
        if ($parent -and $child.CreationDate -lt $parent.CreationDate) { continue }
        $child
        Get-Descendants $child.ProcessId
    }
}

$roots = @()
$ownPorts = @()
foreach ($port in $BACKEND_PORT, $UI_PORT) {
    $listener = Get-ListenerPid $port
    if ($listener) {
        $root = Get-Root $listener
        # Only LoreKeeper's own processes — another dev server on the same port is left alone
        if (-not $root -or $root.CommandLine -notmatch 'src\.main:app|ui[\\/]LoreKeeper\.py') {
            Write-Warning "Port $port belongs to PID $listener ($($byId[[int]$listener].Name)), not LoreKeeper - skipped."
            continue
        }
        $ownPorts += $port
        if ($roots.ProcessId -notcontains $root.ProcessId) { $roots += $root }
    }
}
if (-not $roots) {
    Write-Host "LoreKeeper is not running (no LoreKeeper process on ports $BACKEND_PORT/$UI_PORT)." -ForegroundColor Green
    if ($Restart) { Write-Host "Nothing to restart - start it with .\start.ps1 (or -Server)." }
    exit 0
}
# The mode to restart in, read before the processes are gone: development runs the backend with --reload
# and the UI on 127.0.0.1, the server neither
$devMode = @($roots | Where-Object { $_.CommandLine -match '--reload|--server\.address"?\s+"?127\.0\.0\.1' }).Count -gt 0
$tree = @($roots) + @($roots | ForEach-Object { Get-Descendants $_.ProcessId })
# Started by the UI button, this script is a descendant of the UI process: without this it would wait
# the full timeout for itself and then kill itself before reaching a hung python process
$self = @($PID) + @(Get-Descendants $PID | ForEach-Object { [int]$_.ProcessId })
$tree = @($tree | Where-Object { $self -notcontains [int]$_.ProcessId })
# Recorded now: once the backend is gone, an Ollama it started can no longer be traced back to it
$ollama = @($tree | Where-Object { $_.Name -like "ollama*" })

if ($List) {
    foreach ($p in $tree) {
        # Shortened after collapsing the spaces — measured on the raw line, Substring overran the shorter one
        $cmd = "$($p.CommandLine)" -replace '\s+', ' '
        "{0,6} {1,-16} {2}" -f $p.ProcessId, $p.Name, $cmd.Substring(0, [math]::Min(70, $cmd.Length))
    }
    exit 0
}

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
# Scan again before killing: the dev reloader can start a new worker (and its Ollama) during the wait,
# e.g. for a code change pending when the Ctrl+C came. That worker never got the Ctrl+C, the first scan
# missed it, and it stayed behind as an orphan holding port 8000 (14.10., roadmap 12.3). Only from
# processes that are still the same ones (start time) — a reused PID must not bring in strangers.
$all = @(Get-CimInstance Win32_Process)
$byId = @{}
foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }
$late = @($tree | Where-Object { ($now = $byId[[int]$_.ProcessId]) -and $now.CreationDate -eq $_.CreationDate } |
    ForEach-Object { Get-Descendants $_.ProcessId } | ForEach-Object { [int]$_.ProcessId } |
    Where-Object { $self -notcontains $_ })
$left = @(Get-Process -Id ($ids + @($ollama | ForEach-Object { [int]$_.ProcessId }) + $late | Select-Object -Unique) -ErrorAction SilentlyContinue)
if ($left) {
    Write-Warning "Still running after $Timeout s - killing: $(($left | ForEach-Object { "$($_.Id) $($_.ProcessName)" }) -join ', ')"
    $left | Stop-Process -Force -ErrorAction SilentlyContinue
}

$busy = @($ownPorts | Where-Object { Get-ListenerPid $_ })
if ($busy) {
    Write-Warning "Port $($busy -join ', ') still in use - check with: Get-NetTCPConnection -State Listen -LocalPort $($busy -join ',')"
    exit 1
}
Write-Host ("LoreKeeper stopped after {0:N1} s{1}." -f $sw.Elapsed.TotalSeconds, $(if ($left) { " (some processes killed)" } else { " (clean)" })) -ForegroundColor Green

if ($Restart) {
    # The UI button starts this script in its own process group, which switches Ctrl+C off — and children
    # inherit that: the new backend would ignore the next stop's Ctrl+C, be killed after the timeout and
    # leave its Ollama behind. Switch it back on before starting them.
    Add-Type -Namespace LoreKeeper -Name ConsoleCtrl -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern bool SetConsoleCtrlHandler(System.IntPtr handler, bool add);'
    [LoreKeeper.ConsoleCtrl]::SetConsoleCtrlHandler([IntPtr]::Zero, $false) | Out-Null
    Write-Host "Restarting in $(if ($devMode) { 'development' } else { 'server' }) mode ..."
    # $PSScriptRoot, not $ROOT: PowerShell names ignore case, and the loop variable $root above overwrote it
    & (Join-Path $PSScriptRoot "start.ps1") -Server:(-not $devMode)
}
