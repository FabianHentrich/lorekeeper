# LoreKeeper desktop shortcuts (roadmap 15) — run once, no administrator needed:
#
#   powershell -ExecutionPolicy Bypass -File scripts\create-shortcuts.ps1              # create / renew
#   powershell -ExecutionPolicy Bypass -File scripts\create-shortcuts.ps1 -Uninstall   # remove them
#
# "LoreKeeper starten" runs `start.ps1 -Server -Browser` (home-network server, opens the UI),
# "LoreKeeper beenden" runs `stop.ps1`. Their console stays open a few seconds so the LAN URLs and
# warnings (no admin password, ingest running) can be read. Starting stays a double-click — no autostart
# (roadmap 4.1). The shortcuts hold absolute paths: run this again after moving the repository.

param([switch]$Uninstall)

$ErrorActionPreference = "Stop"

$Root       = Split-Path -Parent $PSScriptRoot
$Desktop    = [Environment]::GetFolderPath("Desktop")   # follows a desktop redirected to OneDrive
$PowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

$Shortcuts = @(
    @{ Name = "LoreKeeper starten"; Script = "start.ps1"; Args = "-Server -Browser"; Seconds = 15; Icon = $null },
    @{ Name = "LoreKeeper beenden"; Script = "stop.ps1";  Args = "";                 Seconds = 8;  Icon = "shell32.dll,27" }
)

if ($Uninstall) {
    foreach ($s in $Shortcuts) {
        Remove-Item (Join-Path $Desktop "$($s.Name).lnk") -ErrorAction SilentlyContinue
    }
    Write-Host "Removed the LoreKeeper shortcuts from $Desktop." -ForegroundColor Green
    exit 0
}

$shell = New-Object -ComObject WScript.Shell
foreach ($s in $Shortcuts) {
    $script = (Join-Path $Root $s.Script) -replace "'", "''"   # single-quoted inside -Command
    # catch, not finally: an error printed after the pause would vanish with the window
    $command = "try { & '$script' $($s.Args) } catch { Write-Host `$_ -ForegroundColor Red }; Start-Sleep -Seconds $($s.Seconds)"
    $link = $shell.CreateShortcut((Join-Path $Desktop "$($s.Name).lnk"))
    $link.TargetPath       = $PowerShell
    $link.Arguments        = "-NoProfile -ExecutionPolicy Bypass -Command `"$command`""
    $link.WorkingDirectory = $Root
    if ($s.Icon) { $link.IconLocation = $s.Icon }
    $link.Save()
    Write-Host "Created '$($s.Name)' on the desktop." -ForegroundColor Green
}
