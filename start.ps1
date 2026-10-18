# LoreKeeper Startup Script
#
#   .\start.ps1           Development: auto-reload on code changes, UI only on this PC (127.0.0.1)
#   .\start.ps1 -Server   Home-network server: no auto-reload, UI reachable in the LAN, no browser
#                         (one-time firewall rule: scripts\setup-lan.ps1, see roadmap 4.1; no autostart)
#   .\start.ps1 -Server -Browser   Same, but opens the UI in the browser — also when LoreKeeper already
#                         runs (what the desktop shortcut does, scripts\create-shortcuts.ps1, roadmap 15)
#   .\stop.ps1            Stop both cleanly (see there)
#
# The backend always stays on 127.0.0.1 — the UI talks to it server-side, so only port 8501 is exposed.

param([switch]$Server, [switch]$Browser)

$ErrorActionPreference = "Stop"

$ROOT         = $PSScriptRoot
$VENV         = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
$PYTHON       = Join-Path $ROOT ".venv\Scripts\python.exe"
$STREAMLIT    = Join-Path $ROOT ".venv\Scripts\streamlit.exe"  # fallback check below
$HEALTH_URL   = "http://127.0.0.1:8000/health"
$UI_URL       = "http://127.0.0.1:8501"
$BACKEND_PORT = 8000
$UI_PORT      = 8501
$TIMEOUT      = 90

# --- Preflight checks -------------------------------------------------
if (-not (Test-Path $VENV)) {
    Write-Error "Virtual environment not found. Run: python -m venv .venv && pip install -r requirements.txt"
    exit 1
}

if ($Server) {
    $envFile = Join-Path $ROOT ".env"
    $hasPassword = (Test-Path $envFile) -and (Select-String -Path $envFile -Pattern '^\s*LOREKEEPER_ADMIN_PASSWORD\s*=\s*\S' -Quiet)
    if (-not $hasPassword) {
        Write-Warning "LOREKEEPER_ADMIN_PASSWORD is not set in .env - in the LAN everyone can open the admin pages (Sources, Settings, wipe)."
    }
}

function Test-PortInUse($port) {
    # Only a listening socket means a running server. Without -State Listen, TimeWait remnants of closed
    # connections (owner process 0, ~2 min after many requests) made the start report "already running".
    $conn = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    return $null -ne $conn
}

if (Test-PortInUse $BACKEND_PORT) {
    Write-Warning "Port $BACKEND_PORT already in use - backend may already be running."
    $skipBackend = $true
} else {
    $skipBackend = $false
}

if (Test-PortInUse $UI_PORT) {
    Write-Warning "Port $UI_PORT already in use - UI may already be running."
    $skipUI = $true
} else {
    $skipUI = $false
}

# --- Start backend ----------------------------------------------------
if (-not $skipBackend) {
    Write-Host "Starting backend..." -ForegroundColor Cyan
    $backendArgs = @("-m", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "$BACKEND_PORT")
    if (-not $Server) { $backendArgs += @("--reload", "--reload-dir", "src") }
    Start-Process -FilePath $PYTHON `
        -ArgumentList $backendArgs `
        -WorkingDirectory $ROOT `
        -WindowStyle $(if ($Server) { "Minimized" } else { "Normal" })

    Write-Host -NoNewline "Waiting for backend " -ForegroundColor Yellow
    $elapsed = 0
    $ready = $false
    while ($elapsed -lt $TIMEOUT) {
        Start-Sleep -Seconds 1
        $elapsed++
        Write-Host -NoNewline "." -ForegroundColor Yellow
        try {
            $resp = Invoke-RestMethod -Uri $HEALTH_URL -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "healthy" -or $resp.status -eq "degraded") {
                $ready = $true
                break
            }
        } catch {
            # not up yet
        }
    }
    Write-Host ""

    if ($ready) {
        Write-Host "Backend ready after ${elapsed}s (status: $($resp.status))" -ForegroundColor Green
    } else {
        Write-Warning "Backend did not respond within $TIMEOUT s - starting UI anyway."
    }
} else {
    Write-Host "Backend already running on port $BACKEND_PORT." -ForegroundColor Green
}

# --- Start UI ---------------------------------------------------------
if (-not $skipUI) {
    Write-Host "Starting Streamlit UI..." -ForegroundColor Cyan
    $uiAddress = if ($Server) { "0.0.0.0" } else { "127.0.0.1" }
    Start-Process -FilePath $PYTHON `
        -ArgumentList "-m", "streamlit", "run", "ui/LoreKeeper.py", "--server.headless", "true", "--server.address", $uiAddress `
        -WorkingDirectory $ROOT `
        -WindowStyle $(if ($Server) { "Minimized" } else { "Normal" })
} else {
    Write-Host "UI already running on port $UI_PORT." -ForegroundColor Green
}

# Development opens the browser on a fresh UI; -Browser always — a second double-click on the shortcut
# then just brings the page up again
if ((-not $Server -and -not $skipUI) -or $Browser) {
    if (-not $skipUI) { Start-Sleep -Seconds 3 }  # let Streamlit bind first
    Start-Process $UI_URL
}

# --- Summary ----------------------------------------------------------
Write-Host ""
Write-Host "LoreKeeper is running$(if ($Server) { ' (server mode)' })." -ForegroundColor Green
Write-Host "  Backend : http://127.0.0.1:$BACKEND_PORT (this PC only)"
Write-Host "  UI      : $UI_URL"
if ($Server) {
    Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
        ForEach-Object { Write-Host "  LAN     : http://$($_.IPAddress):$UI_PORT" }
}
Write-Host "  Docs    : http://127.0.0.1:$BACKEND_PORT/docs"
Write-Host ""
Write-Host "Stop with .\stop.ps1 or the UI sidebar button (closing the windows skips the clean shutdown)." -ForegroundColor DarkGray
