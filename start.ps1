# LoreKeeper Startup Script

$ErrorActionPreference = "Stop"

$ROOT         = $PSScriptRoot
$VENV         = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
$PYTHON       = Join-Path $ROOT ".venv\Scripts\python.exe"
$STREAMLIT    = Join-Path $ROOT ".venv\Scripts\streamlit.exe"  # fallback check below
$HEALTH_URL   = "http://127.0.0.1:8000/health"
$UI_URL       = "http://localhost:8501"
$BACKEND_PORT = 8000
$UI_PORT      = 8501
$TIMEOUT      = 90

# --- Preflight checks -------------------------------------------------
if (-not (Test-Path $VENV)) {
    Write-Error "Virtual environment not found. Run: python -m venv .venv && pip install -r requirements.txt"
    exit 1
}

function Test-PortInUse($port) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
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
    Start-Process -FilePath $PYTHON `
        -ArgumentList "-m", "uvicorn", "src.main:app", "--reload", "--port", "$BACKEND_PORT" `
        -WorkingDirectory $ROOT `
        -WindowStyle Normal

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
    Start-Process -FilePath $PYTHON `
        -ArgumentList "-m", "streamlit", "run", "ui/LoreKeeper.py", "--server.headless", "true" `
        -WorkingDirectory $ROOT `
        -WindowStyle Normal

    # Wait briefly for Streamlit to bind, then open browser
    Start-Sleep -Seconds 3
    Start-Process $UI_URL
} else {
    Write-Host "UI already running on port $UI_PORT." -ForegroundColor Green
}

# --- Summary ----------------------------------------------------------
Write-Host ""
Write-Host "LoreKeeper is running." -ForegroundColor Green
Write-Host "  Backend : http://localhost:$BACKEND_PORT"
Write-Host "  UI      : $UI_URL"
Write-Host "  Docs    : http://localhost:$BACKEND_PORT/docs"
Write-Host ""
Write-Host "Close the process windows to stop." -ForegroundColor DarkGray
