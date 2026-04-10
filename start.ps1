# LoreKeeper Startup Script

$ErrorActionPreference = "Stop"

$ROOT    = $PSScriptRoot
$VENV    = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
$API_URL = "http://localhost:8000/health"
$TIMEOUT = 30

if (-not (Test-Path $VENV)) {
    Write-Error "Virtual environment not found. Run: python -m venv .venv && pip install -r requirements.txt"
    exit 1
}
. $VENV

Write-Host "Starting backend..." -ForegroundColor Cyan
$backendCmd = "cd '$ROOT'; .venv\Scripts\Activate.ps1; uvicorn src.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Write-Host "Waiting for backend (max $TIMEOUT s)..." -ForegroundColor Yellow
$elapsed = 0
$ready = $false
while ($elapsed -lt $TIMEOUT) {
    Start-Sleep -Seconds 1
    $elapsed++
    try {
        $resp = Invoke-RestMethod -Uri $API_URL -TimeoutSec 2 -ErrorAction Stop
        if ($resp.status -eq "healthy" -or $resp.status -eq "degraded") {
            $ready = $true
            break
        }
    } catch {
        # not up yet
    }
}

if (-not $ready) {
    Write-Warning "Backend did not respond within $TIMEOUT s - starting UI anyway."
}

Write-Host "Starting Streamlit UI..." -ForegroundColor Cyan
$uiCmd = "cd '$ROOT'; .venv\Scripts\Activate.ps1; streamlit run ui/LoreKeeper.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $uiCmd

Write-Host ""
Write-Host "LoreKeeper is running." -ForegroundColor Green
Write-Host "  Backend : http://localhost:8000"
Write-Host "  UI      : http://localhost:8501"
Write-Host ""
Write-Host "Close the two terminal windows to stop."
