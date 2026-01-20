# diagnose-uv-windows.ps1
# Diagnostic script to troubleshoot uv installation issues on Windows
# Run in PowerShell: .\diagnose-uv-windows.ps1

$ErrorActionPreference = "SilentlyContinue"

$separator = "=" * 60
Write-Host "`n$separator" -ForegroundColor Cyan
Write-Host "  UV DIAGNOSTIC TOOL FOR WINDOWS" -ForegroundColor Cyan
Write-Host "  JHU Modeling Macroeconomics" -ForegroundColor Cyan
Write-Host "$separator`n" -ForegroundColor Cyan

# --- System Information ---
$dashLine = "-" * 40
Write-Host "SYSTEM INFORMATION" -ForegroundColor Magenta
Write-Host $dashLine

$info = @(
    @("Username", $env:USERNAME),
    @("Computer Name", $env:COMPUTERNAME),
    @("User Profile", $env:USERPROFILE),
    @("Local AppData", $env:LOCALAPPDATA),
    @("Home Drive", $env:HOMEDRIVE),
    @("Home Path", $env:HOMEPATH),
    @("Current Directory", $PWD.Path),
    @("PowerShell Version", $PSVersionTable.PSVersion.ToString()),
    @("OS Version", [System.Environment]::OSVersion.VersionString),
    @("64-bit OS", [Environment]::Is64BitOperatingSystem),
    @("64-bit Process", [Environment]::Is64BitProcess),
    @("Terminal", $(if ($env:TERM_PROGRAM) { $env:TERM_PROGRAM } else { "Native" })),
    @("VS Code Terminal", $($env:TERM_PROGRAM -eq "vscode"))
)

foreach ($item in $info) {
    Write-Host ("  {0,-20}: " -f $item[0]) -NoNewline
    Write-Host $item[1] -ForegroundColor White
}

# Check for non-ASCII characters
$hasNonAscii = $env:USERPROFILE -match '[^\x00-\x7F]'
Write-Host ("  {0,-20}: " -f "Non-ASCII in path") -NoNewline
if ($hasNonAscii) {
    Write-Host "YES - This may cause issues!" -ForegroundColor Red
} else {
    Write-Host "No" -ForegroundColor Green
}

# --- PATH Analysis ---
Write-Host "`nPATH ANALYSIS" -ForegroundColor Magenta
Write-Host $dashLine

Write-Host "`n  User PATH entries:" -ForegroundColor Yellow
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$userPathEntries = $userPath -split ';' | Where-Object { $_ -ne '' }
$i = 1
foreach ($entry in $userPathEntries) {
    $exists = Test-Path $entry
    $color = if ($exists) { "Green" } else { "Red" }
    $status = if ($exists) { "[OK]" } else { "[MISSING]" }
    Write-Host ("    {0,3}. {1} " -f $i, $status) -ForegroundColor $color -NoNewline
    Write-Host $entry
    $i++
}

Write-Host "`n  System PATH entries:" -ForegroundColor Yellow
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$machinePathEntries = $machinePath -split ';' | Where-Object { $_ -ne '' }
$i = 1
foreach ($entry in $machinePathEntries) {
    $exists = Test-Path $entry
    $color = if ($exists) { "Green" } else { "Red" }
    $status = if ($exists) { "[OK]" } else { "[MISSING]" }
    Write-Host ("    {0,3}. {1} " -f $i, $status) -ForegroundColor $color -NoNewline
    Write-Host $entry
    $i++
}

Write-Host "`n  PATH entries containing 'uv':" -ForegroundColor Yellow
$uvPaths = ($userPathEntries + $machinePathEntries) | Where-Object { $_ -like '*uv*' }
if ($uvPaths) {
    foreach ($entry in $uvPaths) {
        $exists = Test-Path $entry
        $color = if ($exists) { "Green" } else { "Red" }
        Write-Host "    " -NoNewline
        Write-Host $entry -ForegroundColor $color
    }
} else {
    Write-Host "    None found - uv may not be in PATH" -ForegroundColor Red
}

# --- Search for uv installations ---
Write-Host "`nSEARCHING FOR UV INSTALLATIONS" -ForegroundColor Magenta
Write-Host $dashLine

$searchLocations = @(
    (Join-Path $env:LOCALAPPDATA "uv"),
    (Join-Path $env:LOCALAPPDATA "uv\bin"),
    (Join-Path $env:USERPROFILE ".local\bin"),
    (Join-Path $env:USERPROFILE ".cargo\bin"),
    (Join-Path $env:APPDATA "uv"),
    (Join-Path $env:APPDATA "uv\bin"),
    "C:\uv",
    "C:\uv\bin",
    "C:\Program Files\uv",
    "C:\Program Files (x86)\uv",
    (Join-Path $env:USERPROFILE "uv"),
    (Join-Path $env:USERPROFILE "uv\bin")
)

# Also check CARGO_HOME if set
if ($env:CARGO_HOME -and (Test-Path $env:CARGO_HOME)) {
    $searchLocations += (Join-Path $env:CARGO_HOME "bin")
}

$foundLocations = @()

foreach ($loc in $searchLocations) {
    $uvExe = Join-Path $loc "uv.exe"
    if (Test-Path $uvExe) {
        $foundLocations += $uvExe
        Write-Host "  FOUND: " -ForegroundColor Green -NoNewline
        Write-Host $uvExe
        
        # Try to get version
        try {
            $version = & $uvExe --version 2>&1
            Write-Host "         Version: $version" -ForegroundColor White
        } catch {
            Write-Host "         Could not get version" -ForegroundColor Yellow
        }
    }
}

if ($foundLocations.Count -eq 0) {
    Write-Host "  No uv.exe found in common locations" -ForegroundColor Red
}

# Also search via where command
Write-Host "`n  Searching via 'where' command:" -ForegroundColor Yellow
$whereResult = where.exe uv 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  FOUND: " -ForegroundColor Green -NoNewline
    Write-Host $whereResult
} else {
    Write-Host "  'where uv' returned no results" -ForegroundColor Red
}

# --- Test uv command ---
Write-Host "`nTESTING UV COMMAND" -ForegroundColor Magenta
Write-Host $dashLine

Write-Host "`n  Test 1: Get-Command uv" -ForegroundColor Yellow
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    Write-Host "    SUCCESS - Found at: $($uvCommand.Source)" -ForegroundColor Green
} else {
    Write-Host "    FAILED - 'uv' not found as a command" -ForegroundColor Red
}

Write-Host "`n  Test 2: uv --version" -ForegroundColor Yellow
try {
    $version = & uv --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    SUCCESS - $version" -ForegroundColor Green
    } else {
        Write-Host "    FAILED - Exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host "    Output: $version" -ForegroundColor Red
    }
} catch {
    Write-Host "    FAILED - Error: $_" -ForegroundColor Red
}

# --- Environment Variables ---
Write-Host "`nRELEVANT ENVIRONMENT VARIABLES" -ForegroundColor Magenta
Write-Host $dashLine

$envVars = @(
    "UV_INSTALL_DIR",
    "UV_CACHE_DIR", 
    "UV_PYTHON_INSTALL_DIR",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX"
)

foreach ($var in $envVars) {
    $value = [System.Environment]::GetEnvironmentVariable($var)
    Write-Host ("  {0,-25}: " -f $var) -NoNewline
    if ($value) {
        Write-Host $value -ForegroundColor White
    } else {
        Write-Host "(not set)" -ForegroundColor Gray
    }
}

# --- Recommendations ---
Write-Host "`nRECOMMENDATIONS" -ForegroundColor Magenta
Write-Host $dashLine

$recommendations = @()

if ($hasNonAscii) {
    $recommendations += "Your user profile path contains non-ASCII characters. Install uv to C:\uv instead."
}

if ($foundLocations.Count -eq 0) {
    $recommendations += "uv is not installed. Run: powershell -ExecutionPolicy Bypass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
}

if ($foundLocations.Count -gt 0 -and -not $uvCommand) {
    $recommendations += "uv is installed but not in PATH. Add this to your PATH: $($foundLocations[0] | Split-Path -Parent)"
}

if ($env:TERM_PROGRAM -eq "vscode" -and -not $uvCommand -and $foundLocations.Count -gt 0) {
    $recommendations += "VS Code may have cached an old PATH. Restart VS Code completely."
}

if ($recommendations.Count -eq 0) {
    Write-Host "  Everything looks good! uv should be working." -ForegroundColor Green
} else {
    $i = 1
    foreach ($rec in $recommendations) {
        Write-Host "  $i. $rec" -ForegroundColor Yellow
        $i++
    }
}

# --- Quick Fix Commands ---
Write-Host "`nQUICK FIX COMMANDS" -ForegroundColor Magenta
Write-Host $dashLine

if ($foundLocations.Count -gt 0) {
    $uvDir = $foundLocations[0] | Split-Path -Parent
    Write-Host @"

  To add uv to your current session's PATH, run:
  
    `$env:Path = "$uvDir;`$env:Path"

  To permanently add uv to your User PATH, run:
  
    [Environment]::SetEnvironmentVariable("Path", "$uvDir;" + [Environment]::GetEnvironmentVariable("Path", "User"), "User")

  Then restart your terminal (or VS Code).

"@ -ForegroundColor White
}

Write-Host "`n$separator" -ForegroundColor Cyan
Write-Host "  DIAGNOSTIC COMPLETE" -ForegroundColor Cyan  
Write-Host "$separator`n" -ForegroundColor Cyan
