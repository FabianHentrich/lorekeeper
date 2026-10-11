#Requires -RunAsAdministrator
# LoreKeeper home-network setup (roadmap 4.1) — run once as administrator on the gaming PC:
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup-lan.ps1              # install
#   powershell -ExecutionPolicy Bypass -File scripts\setup-lan.ps1 -Uninstall   # remove the rule again
#
# Firewall: allows inbound TCP 8501 (Streamlit UI) on the "Private" network profile only.
# The backend (8000) stays on 127.0.0.1 and needs no rule. Safe to run again — the rule is replaced.
#
# No autostart: start the server yourself with `.\start.ps1 -Server` when you need it.
# (An autostart task from an earlier version of this script is removed.)

param([switch]$Uninstall)

$ErrorActionPreference = "Stop"

$RuleName = "LoreKeeper UI (TCP 8501)"
$TaskName = "LoreKeeper Server"   # autostart task of an earlier script version — only removed
$Root     = Split-Path -Parent $PSScriptRoot
$Python   = Join-Path $Root ".venv\Scripts\python.exe"

if ($Uninstall) {
    Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed firewall rule '$RuleName'." -ForegroundColor Green
    exit 0
}

# --- Firewall -----------------------------------------------------------
Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort 8501 `
    -Action Allow -Profile Private | Out-Null
Write-Host "Firewall: inbound TCP 8501 allowed on the Private profile." -ForegroundColor Green

# Block rules win over allow rules — e.g. from clicking "Cancel" on Windows' first-run prompt for python.exe
$blocking = Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue |
    Where-Object { $_.Program -and ($_.Program -ieq $Python -or $_.Program -like "*\python*.exe") } |
    Get-NetFirewallRule | Where-Object { $_.Action -eq "Block" -and $_.Enabled -eq "True" -and $_.Direction -eq "Inbound" }
foreach ($rule in $blocking) {
    Write-Warning "Inbound BLOCK rule '$($rule.DisplayName)' ($($rule.Profile)) for Python overrides the allow rule. Remove it in 'Windows Defender Firewall with Advanced Security' if the UI is not reachable."
}

# The rule only applies when the home network is classified as Private
foreach ($profile in Get-NetConnectionProfile) {
    if ($profile.NetworkCategory -ne "Private") {
        Write-Warning "Network '$($profile.Name)' is '$($profile.NetworkCategory)' - set it to Private (Settings > Network > Properties), otherwise port 8501 stays closed."
    }
}

# --- No autostart: remove the task an earlier version of this script created --
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed old autostart task '$TaskName' - start the server with .\start.ps1 -Server." -ForegroundColor Green
}

# --- Summary --------------------------------------------------------------
Write-Host ""
Write-Host "Start the server with:  .\start.ps1 -Server" -ForegroundColor Cyan
Write-Host "Then reach LoreKeeper from the laptop at:" -ForegroundColor Cyan
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    ForEach-Object { Write-Host "  http://$($_.IPAddress):8501" }
Write-Host "Tip: reserve this IP for the PC in your router (DHCP reservation), so the address stays the same."
