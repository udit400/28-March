$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$extensionPath = (Resolve-Path (Join-Path $projectRoot "extension")).Path
$profilePath = Join-Path $projectRoot ".chromium-profile"

$browserCandidates = @(
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "$env:ProgramFiles(x86)\Google\Chrome\Application\chrome.exe",
  "$env:LocalAppData\Google\Chrome\Application\chrome.exe",
  "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
  "$env:ProgramFiles(x86)\Microsoft\Edge\Application\msedge.exe",
  "$env:LocalAppData\Microsoft\Edge\Application\msedge.exe",
  "$env:ProgramFiles\BraveSoftware\Brave-Browser\Application\brave.exe",
  "$env:ProgramFiles(x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
  "$env:LocalAppData\BraveSoftware\Brave-Browser\Application\brave.exe"
)

$registryCandidates = @(
  'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe',
  'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe',
  'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe',
  'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe',
  'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\brave.exe',
  'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\brave.exe'
)

$browserPath = $browserCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $browserPath) {
  $browserPath = foreach ($registryKey in $registryCandidates) {
    if (Test-Path $registryKey) {
      $candidate = (Get-ItemProperty $registryKey).'(default)'
      if ($candidate -and (Test-Path $candidate)) {
        $candidate
        break
      }
    }
  }
}

if (-not $browserPath) {
  throw "No supported Chromium-based browser was found. Install Chrome, Edge, or Brave, then rerun this script."
}

New-Item -ItemType Directory -Path $profilePath -Force | Out-Null

$profilePath = (Resolve-Path $profilePath).Path

$arguments = @(
  "--user-data-dir=`"$profilePath`"",
  "--load-extension=`"$extensionPath`"",
  'http://127.0.0.1:8000/demo'
)

Start-Process -FilePath $browserPath -ArgumentList $arguments